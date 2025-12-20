# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.trtllm import (
    get_trtllm_kv_cache_quant_mode,
    get_trtllm_thop,
    has_trtllm_thop,
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    KVCacheLayoutType,
    PAD_SLOT_ID,
    get_kv_cache_layout,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    is_pin_memory_available,
    slice_query_start_locs,
    split_decodes_and_prefills,
)

logger = init_logger(__name__)

_REQ_TYPE_CONTEXT = 0
_REQ_TYPE_GENERATION = 1
_ATTN_INPUT_CONTEXT = 1
_ATTN_INPUT_GENERATION = 2
_MASK_TYPE_CAUSAL = 1


@dataclass
class TRTLLMCallMetadata:
    num_reqs: int
    num_tokens: int
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    context_lens: torch.Tensor
    context_lens_cpu: torch.Tensor
    host_past_kv_lens: torch.Tensor
    host_total_kv_lens: torch.Tensor
    host_request_types: torch.Tensor
    kv_cache_block_offsets: torch.Tensor
    host_kv_cache_block_offsets: torch.Tensor
    host_kv_cache_pool_mapping: torch.Tensor
    max_seq_len: int
    max_num_requests: int
    attention_input_type: int


@dataclass
class TRTLLMAttentionMetadata:
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    decode: TRTLLMCallMetadata | None
    prefill: TRTLLMCallMetadata | None


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def _build_kv_cache_block_offsets(block_table: torch.Tensor) -> torch.Tensor:
    block_table = block_table.to(torch.int32)
    valid = block_table != PAD_SLOT_ID
    k_offsets = torch.where(valid, block_table * 2, torch.zeros_like(block_table))
    v_offsets = torch.where(valid, block_table * 2 + 1, torch.zeros_like(block_table))
    kv_offsets = torch.stack((k_offsets, v_offsets), dim=1)
    return kv_offsets.unsqueeze(0)


class TRTLLMAttentionImpl(AttentionImpl[TRTLLMAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        del alibi_slopes, kv_sharing_target_layer_name
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("TRTLLM attention only supports decoder.")
        if sinks is not None:
            raise NotImplementedError("TRTLLM attention does not support sinks.")
        if logits_soft_cap not in (None, 0.0):
            raise NotImplementedError(
                "TRTLLM attention does not support logits soft cap."
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "TRTLLM attention sliding window is not wired yet."
            )
        if num_kv_heads is None:
            raise ValueError("num_kv_heads must be provided for TRTLLM attention.")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.quant_mode = get_trtllm_kv_cache_quant_mode(kv_cache_dtype)

        vllm_config = get_current_vllm_config()
        self.max_num_requests = vllm_config.scheduler_config.max_num_seqs
        self.max_context_length = vllm_config.model_config.max_model_len

        self._workspace: torch.Tensor | None = None
        self._host_kv_cache_pool_pointers: torch.Tensor | None = None

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.empty(0, dtype=torch.uint8, device=device)
        return self._workspace

    def _get_pool_pointers(self, kv_cache: torch.Tensor) -> torch.Tensor:
        ptr = kv_cache.data_ptr()
        if (
            self._host_kv_cache_pool_pointers is None
            or self._host_kv_cache_pool_pointers.numel() == 0
            or self._host_kv_cache_pool_pointers[0, 0].item() != ptr
        ):
            self._host_kv_cache_pool_pointers = torch.tensor(
                [[ptr, ptr]], dtype=torch.int64, device="cpu"
            )
        return self._host_kv_cache_pool_pointers

    def _fuse_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        qkv = torch.cat((query, key, value), dim=1)
        return qkv.reshape(qkv.shape[0], -1).contiguous()

    def _run_trtllm(
        self,
        layer_idx: int,
        qkv: torch.Tensor,
        output: torch.Tensor,
        call_meta: TRTLLMCallMetadata,
        kv_cache: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor | None,
        kv_scale_quant_orig: torch.Tensor | None,
    ) -> None:
        thop = get_trtllm_thop()
        if thop is None:
            raise RuntimeError("TRTLLM thop bindings are not available.")

        kv_cache_layout = get_kv_cache_layout()
        if kv_cache_layout != "HND":
            raise ValueError("TRTLLM attention requires HND KV cache layout.")

        block_size = kv_cache.shape[3]
        if not _is_power_of_two(block_size):
            raise ValueError("TRTLLM attention requires power-of-two block size.")

        thop.attention(
            q=qkv,
            k=None,
            v=None,
            output=output,
            output_sf=None,
            out_dtype=None,
            workspace_=self._get_workspace(qkv.device),
            sequence_length=call_meta.seq_lens,
            host_past_key_value_lengths=call_meta.host_past_kv_lens,
            host_total_kv_lens=call_meta.host_total_kv_lens,
            context_lengths=call_meta.context_lens,
            host_context_lengths=call_meta.context_lens_cpu,
            host_request_types=call_meta.host_request_types,
            kv_cache_block_offsets=call_meta.kv_cache_block_offsets,
            host_kv_cache_block_offsets=call_meta.host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers=self._get_pool_pointers(kv_cache),
            host_kv_cache_pool_mapping=call_meta.host_kv_cache_pool_mapping,
            cache_indirection=None,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            out_scale=None,
            rotary_inv_freq=None,
            rotary_cos_sin=None,
            latent_cache=None,
            q_pe=None,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=True,
            update_kv_cache=True,
            predicted_tokens_per_seq=1,
            layer_idx=layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            tokens_per_block=block_size,
            max_num_requests=call_meta.max_num_requests,
            max_context_length=self.max_context_length,
            attention_window_size=call_meta.max_seq_len,
            sink_token_length=0,
            beam_width=1,
            mask_type=_MASK_TYPE_CAUSAL,
            quant_mode=self.quant_mode,
            q_scaling=self.scale,
            position_embedding_type=0,
            rotary_embedding_dim=0,
            rotary_embedding_base=1.0,
            rotary_embedding_scale_type=0,
            rotary_embedding_scales=[1.0, 1.0, 1.0],
            rotary_embedding_max_position_info=[0, 0],
            use_paged_context_fmha=True,
            attention_input_type=call_meta.attention_input_type,
            is_mla_enable=False,
            chunked_prefill_buffer_batch_size=None,
            q_lora_rank=None,
            kv_lora_rank=None,
            qk_nope_head_dim=None,
            qk_rope_head_dim=None,
            v_head_dim=None,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            mla_tensor_params=[],
            attention_chunk_size=None,
            softmax_stats_tensor=None,
            spec_decoding_bool_params=[False, False, False],
            spec_decoding_tensor_params=[None, None, None],
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=1,
            sparse_mla_topk=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
            mla_bmm1_scale=None,
            mla_bmm2_scale=None,
            quant_q_buffer=None,
        )

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TRTLLMAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(
                (query.shape[0], self.num_heads * self.head_size),
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "TRTLLM attention does not support fused output quantization."
            )
        if self.kv_cache_dtype != "auto":
            raise NotImplementedError(
                "TRTLLM attention currently requires kv_cache_dtype=auto."
            )

        try:
            from vllm.model_executor.models.utils import extract_layer_index

            layer_idx = extract_layer_index(layer.layer_name)
        except Exception:  # pylint: disable=broad-except
            layer_idx = 0

        kv_scale_orig_quant = None
        kv_scale_quant_orig = None

        num_decode_tokens = attn_metadata.num_decode_tokens
        if attn_metadata.decode is not None:
            qkv = self._fuse_qkv(
                query[:num_decode_tokens],
                key[:num_decode_tokens],
                value[:num_decode_tokens],
            )
            self._run_trtllm(
                layer_idx,
                qkv,
                output[:num_decode_tokens],
                attn_metadata.decode,
                kv_cache,
                kv_scale_orig_quant,
                kv_scale_quant_orig,
            )

        if attn_metadata.prefill is not None:
            qkv = self._fuse_qkv(
                query[num_decode_tokens:],
                key[num_decode_tokens:],
                value[num_decode_tokens:],
            )
            self._run_trtllm(
                layer_idx,
                qkv,
                output[num_decode_tokens:],
                attn_metadata.prefill,
                kv_cache,
                kv_scale_orig_quant,
                kv_scale_quant_orig,
            )

        return output


class TRTLLMAttentionMetadataBuilder(AttentionMetadataBuilder[TRTLLMAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_dim = kv_cache_spec.head_size
        self.block_size = kv_cache_spec.block_size
        self.max_num_requests = vllm_config.scheduler_config.max_num_seqs

        if self.num_kv_heads == 1 or self.num_qo_heads % self.num_kv_heads != 0:
            raise NotImplementedError(
                "TRTLLM attention requires num_kv_heads != 1 and "
                "num_q_heads % num_kv_heads == 0."
            )
        if self.parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "TRTLLM attention does not support decode context parallelism."
            )
        if not _is_power_of_two(self.block_size) or self.block_size < 8:
            raise ValueError(
                "TRTLLM attention requires block_size to be power-of-two >= 8."
            )

        cache_layout = get_kv_cache_layout()
        if cache_layout != "HND":
            raise ValueError("TRTLLM attention requires HND KV cache layout.")

        if kv_cache_spec.cache_dtype_str not in (None, "auto"):
            raise NotImplementedError(
                "TRTLLM attention currently requires kv_cache_dtype=auto."
            )

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        per_layer_params = get_per_layer_parameters(
            vllm_config, layer_names, TRTLLMAttentionImpl
        )
        global_params = infer_global_hyperparameters(per_layer_params)
        if global_params.has_sinks:
            raise NotImplementedError("TRTLLM attention does not support sinks.")
        if global_params.logits_soft_cap not in (None, 0.0):
            raise NotImplementedError(
                "TRTLLM attention does not support logits soft cap."
            )
        if global_params.window_left >= 0:
            raise NotImplementedError(
                "TRTLLM attention sliding window is not wired yet."
            )

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self._host_kv_cache_pool_mapping = torch.zeros(
            (num_layers, 2), dtype=torch.int32, device="cpu"
        )
        self._pin_memory = is_pin_memory_available()

    def _build_call_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        request_slice: slice,
        attention_input_type: int,
    ) -> TRTLLMCallMetadata:
        seq_lens = common_attn_metadata.seq_lens[request_slice]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[request_slice]

        query_start_loc_cpu = slice_query_start_locs(
            common_attn_metadata.query_start_loc_cpu, request_slice
        )
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        if attention_input_type == _ATTN_INPUT_GENERATION:
            context_lens_cpu = seq_lens_cpu - query_lens_cpu
        else:
            context_lens_cpu = query_lens_cpu

        context_lens = context_lens_cpu.to(self.device)

        host_past_kv_lens = (
            context_lens_cpu.clone()
            if attention_input_type == _ATTN_INPUT_GENERATION
            else torch.zeros_like(context_lens_cpu)
        )

        host_total_kv_lens = torch.zeros(2, dtype=torch.int32, device="cpu")
        total_kv_len = int(seq_lens_cpu.sum().item())
        if attention_input_type == _ATTN_INPUT_GENERATION:
            host_total_kv_lens[1] = total_kv_len
        else:
            host_total_kv_lens[0] = total_kv_len

        num_reqs = seq_lens_cpu.numel()
        host_request_types = torch.full(
            (num_reqs,),
            _REQ_TYPE_GENERATION
            if attention_input_type == _ATTN_INPUT_GENERATION
            else _REQ_TYPE_CONTEXT,
            dtype=torch.int32,
            device="cpu",
        )

        block_table = common_attn_metadata.block_table_tensor[request_slice]
        kv_cache_block_offsets = _build_kv_cache_block_offsets(block_table)
        host_kv_cache_block_offsets = kv_cache_block_offsets.cpu()
        if self._pin_memory:
            host_kv_cache_block_offsets = host_kv_cache_block_offsets.pin_memory()

        return TRTLLMCallMetadata(
            num_reqs=num_reqs,
            num_tokens=int(seq_lens_cpu.sum().item()),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            context_lens=context_lens,
            context_lens_cpu=context_lens_cpu,
            host_past_kv_lens=host_past_kv_lens,
            host_total_kv_lens=host_total_kv_lens,
            host_request_types=host_request_types,
            kv_cache_block_offsets=kv_cache_block_offsets,
            host_kv_cache_block_offsets=host_kv_cache_block_offsets,
            host_kv_cache_pool_mapping=self._host_kv_cache_pool_mapping,
            max_seq_len=common_attn_metadata.max_seq_len,
            max_num_requests=self.max_num_requests,
            attention_input_type=attention_input_type,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TRTLLMAttentionMetadata:
        del common_prefix_len, fast_build

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold or 1,
                require_uniform=True,
            )
        )

        query_lens = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        if num_decodes > 0:
            decode_lens = query_lens[:num_decodes]
            if not torch.all(
                (decode_lens == decode_lens[0]) | (decode_lens == 0)
            ):
                raise ValueError(
                    "TRTLLM decode requires uniform query lengths per request."
                )

        decode_meta = None
        prefill_meta = None
        if num_decodes > 0:
            decode_meta = self._build_call_metadata(
                common_attn_metadata,
                slice(0, num_decodes),
                _ATTN_INPUT_GENERATION,
            )
        if num_prefills > 0:
            prefill_meta = self._build_call_metadata(
                common_attn_metadata,
                slice(num_decodes, num_decodes + num_prefills),
                _ATTN_INPUT_CONTEXT,
            )

        return TRTLLMAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            decode=decode_meta,
            prefill=prefill_meta,
        )


class TRTLLMAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TRTLLMAttentionImpl"]:
        return TRTLLMAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TRTLLMAttentionMetadataBuilder"]:
        return TRTLLMAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        if cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 4, 0, 3, 5)
        if cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        return (0, 1, 2, 3, 4)

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major in (10, 12)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size >= 8 and _is_power_of_two(block_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        if current_platform.is_device_capability_family(100) or (
            current_platform.is_device_capability_family(120)
        ):
            return "HND"
        return None

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability,
        attn_type: str,
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            device_capability=device_capability,
            attn_type=attn_type,
        )
        if not has_trtllm_thop():
            invalid_reasons.append("TRTLLM bindings not available")
        if head_size % 8 != 0 or head_size > 256:
            invalid_reasons.append("unsupported head size for TRTLLM attention")
        return invalid_reasons


__all__ = ["TRTLLMAttentionBackend"]
