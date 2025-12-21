# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.trtllm import (
    get_trtllm_kv_cache_quant_mode,
    has_trtllm_thop,
    run_trtllm_attention,
    trtllm_attention_supports_nvfp4,
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
from vllm.v1.spec_decode.spec_tree_manager import SpecTreeManager

logger = init_logger(__name__)

_REQ_TYPE_CONTEXT = 0
_REQ_TYPE_GENERATION = 1
_ATTN_INPUT_CONTEXT = 1
_ATTN_INPUT_GENERATION = 2
_MASK_TYPE_CAUSAL = 1
_MASK_TYPE_SLIDING_OR_CHUNKED_CAUSAL = 2
_TRTLLM_DECODE_BACKEND_AUTO = "auto"
_TRTLLM_DECODE_BACKEND_XQA = "xqa"
_TRTLLM_DECODE_BACKEND_MMHA = "mmha"
_TRTLLM_DECODE_BACKEND_TRTLLM_GEN = "trtllm-gen"
_XQA_TOKENS_PER_BLOCK = {8, 16, 32, 64, 128}


def _normalize_trtllm_decode_backend(value: str | None) -> str:
    if value is None:
        return _TRTLLM_DECODE_BACKEND_AUTO
    return value.lower()


def _force_trtllm_xqa(enable: bool) -> None:
    if enable:
        os.environ["TRTLLM_FORCE_XQA"] = "1"
    elif "TRTLLM_FORCE_XQA" in os.environ:
        os.environ["TRTLLM_FORCE_XQA"] = "0"


@dataclass
class TRTLLMCallMetadata:
    num_reqs: int
    num_tokens: int
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    predicted_tokens_per_seq: int
    is_spec_decoding_enabled: bool
    use_spec_decoding: bool
    is_spec_dec_tree: bool
    spec_decoding_generation_lengths: torch.Tensor | None
    spec_decoding_position_offsets: torch.Tensor | None
    spec_decoding_packed_mask: torch.Tensor | None
    spec_decoding_bl_tree_mask_offset: torch.Tensor | None
    spec_decoding_bl_tree_mask: torch.Tensor | None
    spec_bl_tree_first_sparse_mask_offset_kv: torch.Tensor | None
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
    beam_width: int
    cache_indirection: torch.Tensor | None


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


def _is_linear_spec_tree(tree_choices: list[tuple[int, ...]]) -> bool:
    if not tree_choices:
        return True
    sorted_choices = sorted(tree_choices, key=lambda t: (len(t), t))
    max_depth = len(sorted_choices[-1])
    if len(sorted_choices) != max_depth:
        return False
    prev: tuple[int, ...] = ()
    for depth, path in enumerate(sorted_choices, start=1):
        if len(path) != depth:
            return False
        if depth > 1 and path[:-1] != prev:
            return False
        prev = path
    return True


def _build_kv_cache_block_offsets(block_table: torch.Tensor) -> torch.Tensor:
    block_table = block_table.to(torch.int32)
    valid = block_table != PAD_SLOT_ID
    k_offsets = torch.where(valid, block_table * 2, torch.zeros_like(block_table))
    v_offsets = torch.where(valid, block_table * 2 + 1, torch.zeros_like(block_table))
    kv_offsets = torch.stack((k_offsets, v_offsets), dim=1)
    return kv_offsets.unsqueeze(0)


class _TRTLLMCustomOp:
    def __init__(self, impl: "TRTLLMAttentionImpl") -> None:
        self._impl = impl

    def _get_spec_decoding_params(
        self,
        call_meta: "TRTLLMCallMetadata",
    ) -> tuple[list[bool], list[torch.Tensor | None]]:
        if not call_meta.use_spec_decoding:
            bool_params = [False, False, False]
            tensor_params: list[torch.Tensor | None] = [None, None, None]
            if current_platform.is_device_capability_family(100) and not (
                current_platform.is_device_capability_family(120)
            ):
                tensor_params.extend([None, None, None])
            return bool_params, tensor_params

        bool_params = [
            call_meta.is_spec_decoding_enabled,
            call_meta.use_spec_decoding,
            call_meta.is_spec_dec_tree,
        ]
        tensor_params = [
            call_meta.spec_decoding_generation_lengths,
            call_meta.spec_decoding_position_offsets,
            call_meta.spec_decoding_packed_mask,
        ]
        if current_platform.is_device_capability_family(100) and not (
            current_platform.is_device_capability_family(120)
        ):
            tensor_params.extend(
                [
                    call_meta.spec_decoding_bl_tree_mask_offset,
                    call_meta.spec_decoding_bl_tree_mask,
                    call_meta.spec_bl_tree_first_sparse_mask_offset_kv,
                ]
            )
        if any(param is None for param in tensor_params):
            raise ValueError(
                "TRTLLM spec decoding is enabled but required tensors are missing."
            )
        return bool_params, tensor_params

    def run(
        self,
        layer_idx: int,
        qkv: torch.Tensor,
        output: torch.Tensor,
        call_meta: "TRTLLMCallMetadata",
        kv_cache: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor | None,
        kv_scale_quant_orig: torch.Tensor | None,
        kv_scale_cache: torch.Tensor | None,
    ) -> None:
        spec_decoding_bool_params, spec_decoding_tensor_params = (
            self._get_spec_decoding_params(call_meta)
        )
        if (
            call_meta.predicted_tokens_per_seq > 1
            and not any(spec_decoding_bool_params)
        ):
            logger.warning_once(
                "TRTLLM multi-token decode is active without spec-decoding "
                "masks; using standard causal attention."
            )

        run_trtllm_attention(
            **self._build_attention_kwargs(
                layer_idx=layer_idx,
                qkv=qkv,
                output=output,
                call_meta=call_meta,
                kv_cache=kv_cache,
                kv_scale_orig_quant=kv_scale_orig_quant,
                kv_scale_quant_orig=kv_scale_quant_orig,
                kv_scale_cache=kv_scale_cache,
                spec_decoding_bool_params=spec_decoding_bool_params,
                spec_decoding_tensor_params=spec_decoding_tensor_params,
            )
        )

    def _build_attention_kwargs(
        self,
        *,
        layer_idx: int,
        qkv: torch.Tensor,
        output: torch.Tensor,
        call_meta: "TRTLLMCallMetadata",
        kv_cache: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor | None,
        kv_scale_quant_orig: torch.Tensor | None,
        kv_scale_cache: torch.Tensor | None,
        spec_decoding_bool_params: list[bool],
        spec_decoding_tensor_params: list[torch.Tensor | None],
    ) -> dict:
        return dict(
            q=qkv,
            k=None,
            v=None,
            output=output,
            output_sf=None,
            out_dtype=None,
            workspace_=self._impl._get_workspace(qkv.device),
            sequence_length=call_meta.seq_lens,
            host_past_key_value_lengths=call_meta.host_past_kv_lens,
            host_total_kv_lens=call_meta.host_total_kv_lens,
            context_lengths=call_meta.context_lens,
            host_context_lengths=call_meta.context_lens_cpu,
            host_request_types=call_meta.host_request_types,
            kv_cache_block_offsets=call_meta.kv_cache_block_offsets,
            host_kv_cache_block_offsets=call_meta.host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers=self._impl._get_pool_pointers(
                kv_cache, kv_scale_cache
            ),
            host_kv_cache_pool_mapping=call_meta.host_kv_cache_pool_mapping,
            cache_indirection=call_meta.cache_indirection,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            out_scale=None,
            rotary_inv_freq=None,
            rotary_cos_sin=None,
            latent_cache=None,
            q_pe=None,
            block_ids_per_seq=None,
            attention_sinks=self._impl._get_attention_sinks(qkv.device),
            is_fused_qkv=True,
            update_kv_cache=True,
            predicted_tokens_per_seq=call_meta.predicted_tokens_per_seq,
            layer_idx=layer_idx,
            num_heads=self._impl.num_heads,
            num_kv_heads=self._impl.num_kv_heads,
            head_size=self._impl.head_size,
            tokens_per_block=kv_cache.shape[2],
            max_num_requests=call_meta.max_num_requests,
            max_context_length=self._impl.max_context_length,
            attention_window_size=self._impl._get_attention_window_size(
                call_meta.max_seq_len
            ),
            sink_token_length=self._impl.sink_token_length,
            beam_width=call_meta.beam_width,
            mask_type=self._impl._get_attention_mask_type(),
            quant_mode=self._impl.quant_mode,
            q_scaling=self._impl.scale,
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
            attention_chunk_size=self._impl.attention_chunk_size,
            softmax_stats_tensor=None,
            spec_decoding_bool_params=spec_decoding_bool_params,
            spec_decoding_tensor_params=spec_decoding_tensor_params,
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
        if logits_soft_cap not in (None, 0.0):
            raise NotImplementedError(
                "TRTLLM attention does not support logits soft cap."
            )
        if num_kv_heads is None:
            raise ValueError("num_kv_heads must be provided for TRTLLM attention.")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.quant_mode = get_trtllm_kv_cache_quant_mode(kv_cache_dtype)
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = self.sliding_window[0]
        self._nvfp4_support_checked = False
        self._nvfp4_support_mask_type: int | None = None
        self._nvfp4_support_block_size: int | None = None

        vllm_config = get_current_vllm_config()
        self.max_num_requests = vllm_config.scheduler_config.max_num_seqs
        self.max_context_length = vllm_config.model_config.max_model_len
        self.attention_chunk_size = vllm_config.model_config.attention_chunk_size
        self.sink_token_length = envs.VLLM_TRTLLM_SINK_TOKEN_LENGTH
        if self.sink_token_length < 0:
            raise ValueError("TRTLLM sink token length must be >= 0.")

        self._workspace: torch.Tensor | None = None
        self._host_kv_cache_pool_pointers: torch.Tensor | None = None
        self._nvfp4_kv_scales: torch.Tensor | None = None
        self._nvfp4_kv_scales_inv: torch.Tensor | None = None
        self._custom_op = _TRTLLMCustomOp(self)
        self.sinks: torch.Tensor | None = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks
            if self.sink_token_length == 0:
                logger.warning_once(
                    "TRTLLM attention sinks are enabled but "
                    "VLLM_TRTLLM_SINK_TOKEN_LENGTH=0; sink tokens are disabled."
                )

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.empty(0, dtype=torch.uint8, device=device)
        return self._workspace

    def _get_pool_pointers(
        self, kv_cache: torch.Tensor, kv_scale_cache: torch.Tensor | None
    ) -> torch.Tensor:
        ptr = kv_cache.data_ptr()
        if kv_scale_cache is None:
            if (
                self._host_kv_cache_pool_pointers is None
                or self._host_kv_cache_pool_pointers.numel() != 2
                or self._host_kv_cache_pool_pointers[0, 0].item() != ptr
            ):
                self._host_kv_cache_pool_pointers = torch.tensor(
                    [[ptr, ptr]], dtype=torch.int64, device="cpu"
                )
            return self._host_kv_cache_pool_pointers

        scale_ptr = kv_scale_cache.data_ptr()
        if (
            self._host_kv_cache_pool_pointers is None
            or self._host_kv_cache_pool_pointers.numel() != 4
            or self._host_kv_cache_pool_pointers[0, 0, 0].item() != ptr
            or self._host_kv_cache_pool_pointers[0, 0, 1].item() != scale_ptr
        ):
            self._host_kv_cache_pool_pointers = torch.tensor(
                [[[ptr, scale_ptr], [ptr, scale_ptr]]],
                dtype=torch.int64,
                device="cpu",
            )
        return self._host_kv_cache_pool_pointers

    def _fuse_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        qkv = torch.cat((query, key, value), dim=1)
        return qkv.reshape(qkv.shape[0], -1).contiguous()

    def _get_kv_scale_tensors(
        self, layer: torch.nn.Module, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.kv_cache_dtype.startswith("fp8"):
            k_scale = getattr(layer, "_k_scale", None)
            v_scale = getattr(layer, "_v_scale", None)
            if k_scale is None or v_scale is None:
                raise ValueError(
                    "TRTLLM attention requires k/v scale tensors for fp8 KV cache. "
                    "Select a different attention backend if scales are absent."
                )
            if k_scale.numel() != 1 or v_scale.numel() != 1:
                raise ValueError("TRTLLM only supports per-tensor KV scales.")
            k_scale_f32 = k_scale.to(device=device, dtype=torch.float32)
            v_scale_f32 = v_scale.to(device=device, dtype=torch.float32)
            if not torch.allclose(k_scale_f32, v_scale_f32):
                raise ValueError(
                    "TRTLLM requires identical k_scale and v_scale for fp8 KV cache. "
                    "Select a different attention backend for per-K/V scaling."
                )
            kv_scale_quant_orig = k_scale_f32
            kv_scale_orig_quant = 1.0 / k_scale_f32
            return kv_scale_orig_quant, kv_scale_quant_orig

        if self.kv_cache_dtype == "nvfp4":
            k_global_scale = float(getattr(layer, "_k_global_scale_float", 1.0))
            v_global_scale = float(getattr(layer, "_v_global_scale_float", 1.0))
            if (
                self._nvfp4_kv_scales is None
                or self._nvfp4_kv_scales.device != device
                or self._nvfp4_kv_scales.numel() != 3
            ):
                self._nvfp4_kv_scales = torch.empty(
                    3, dtype=torch.float32, device=device
                )
                self._nvfp4_kv_scales_inv = torch.empty(
                    3, dtype=torch.float32, device=device
                )
            # TRTLLM expects 3 scales for NVFP4: [dummy, k, v].
            self._nvfp4_kv_scales[0] = 1.0
            self._nvfp4_kv_scales[1] = k_global_scale
            self._nvfp4_kv_scales[2] = v_global_scale
            k_inv = 0.0 if k_global_scale == 0.0 else 1.0 / k_global_scale
            v_inv = 0.0 if v_global_scale == 0.0 else 1.0 / v_global_scale
            self._nvfp4_kv_scales_inv[0] = 1.0
            self._nvfp4_kv_scales_inv[1] = k_inv
            self._nvfp4_kv_scales_inv[2] = v_inv
            return self._nvfp4_kv_scales_inv, self._nvfp4_kv_scales

        return None, None

    def _get_nvfp4_scale_cache(self, layer: torch.nn.Module) -> torch.Tensor:
        kv_scale_cache = getattr(layer, "kv_cache_k_scale", None)
        kv_scale_cache_v = getattr(layer, "kv_cache_v_scale", None)
        if kv_scale_cache is None or kv_scale_cache_v is None:
            raise ValueError(
                "TRTLLM NVFP4 KV cache requires kv_cache_k_scale/v_scale tensors. "
                "Select a different attention backend if scales are absent."
            )
        if (
            kv_scale_cache.untyped_storage().data_ptr()
            != kv_scale_cache_v.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "TRTLLM NVFP4 KV cache requires shared scale storage for K/V."
            )
        if kv_scale_cache.storage_offset() != 0:
            raise ValueError(
                "TRTLLM NVFP4 KV cache requires scale storage to start at offset 0."
            )
        return kv_scale_cache

    def _get_attention_sinks(self, device: torch.device) -> torch.Tensor | None:
        if self.sinks is None:
            return None
        if self.sinks.device != device or self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(device=device, dtype=torch.float32)
        return self.sinks

    def _get_attention_window_size(self, max_seq_len: int) -> int:
        if self.window_left < 0:
            window_size = max_seq_len
        else:
            window_size = min(self.window_left + 1, max_seq_len)
        if self.sink_token_length > window_size:
            raise ValueError(
                "TRTLLM sink token length must be <= attention window size."
            )
        if self.sink_token_length > 0 and self.window_left < 0:
            logger.warning_once(
                "TRTLLM sink token length is set but sliding window is disabled; "
                "sink tokens will be applied with full attention."
            )
        return window_size

    def _get_attention_mask_type(self) -> int:
        if self.window_left >= 0:
            return _MASK_TYPE_SLIDING_OR_CHUNKED_CAUSAL
        if self.attention_chunk_size is not None and self.attention_chunk_size > 0:
            return _MASK_TYPE_SLIDING_OR_CHUNKED_CAUSAL
        return _MASK_TYPE_CAUSAL

    def _run_trtllm(
        self,
        layer_idx: int,
        qkv: torch.Tensor,
        output: torch.Tensor,
        call_meta: TRTLLMCallMetadata,
        kv_cache: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor | None,
        kv_scale_quant_orig: torch.Tensor | None,
        kv_scale_cache: torch.Tensor | None,
    ) -> None:
        kv_cache_layout = get_kv_cache_layout()
        if kv_cache_layout != "HND":
            raise ValueError("TRTLLM attention requires HND KV cache layout.")

        block_size = kv_cache.shape[2]
        if not _is_power_of_two(block_size):
            raise ValueError("TRTLLM attention requires power-of-two block size.")
        self._custom_op.run(
            layer_idx,
            qkv,
            output,
            call_meta,
            kv_cache,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            kv_scale_cache,
        )

    def _ensure_nvfp4_support(self, tokens_per_block: int) -> None:
        if self.kv_cache_dtype != "nvfp4":
            return
        mask_type = self._get_attention_mask_type()
        if (
            self._nvfp4_support_checked
            and self._nvfp4_support_mask_type == mask_type
            and self._nvfp4_support_block_size == tokens_per_block
        ):
            return
        support = trtllm_attention_supports_nvfp4(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            tokens_per_block=tokens_per_block,
            mask_type=mask_type,
            kv_cache_dtype=self.kv_cache_dtype,
            use_paged_context_fmha=True,
            is_mla_enable=False,
        )
        if support is not True:
            raise NotImplementedError(
                "TRTLLM NVFP4 attention kernels are not available for this "
                "configuration (mask_type="
                f"{mask_type}, tokens_per_block={tokens_per_block})."
            )
        self._nvfp4_support_checked = True
        self._nvfp4_support_mask_type = mask_type
        self._nvfp4_support_block_size = tokens_per_block

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(torch.float32)

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
        if self.kv_cache_dtype == "nvfp4":
            self._ensure_nvfp4_support(kv_cache.shape[2])

        try:
            from vllm.model_executor.models.utils import extract_layer_index

            layer_idx = extract_layer_index(layer.layer_name)
        except Exception:  # pylint: disable=broad-except
            layer_idx = 0

        kv_scale_orig_quant, kv_scale_quant_orig = self._get_kv_scale_tensors(
            layer, query.device
        )
        kv_scale_cache = None
        if self.kv_cache_dtype == "nvfp4":
            kv_scale_cache = self._get_nvfp4_scale_cache(layer)

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
                kv_scale_cache,
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
                kv_scale_cache,
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
        self.kv_cache_dtype = kv_cache_spec.cache_dtype_str or "auto"

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

        if kv_cache_spec.cache_dtype_str not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "nvfp4",
        ):
            raise NotImplementedError(
                "TRTLLM attention only supports auto/fp8/nvfp4 KV cache dtypes."
            )
        if kv_cache_spec.cache_dtype_str == "nvfp4":
            if self.head_dim % 16 != 0 or self.head_dim > 256:
                raise ValueError(
                    "TRTLLM NVFP4 KV cache requires head_size to be a "
                    "multiple of 16 and <= 256."
                )

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        per_layer_params = get_per_layer_parameters(
            vllm_config, layer_names, TRTLLMAttentionImpl
        )
        global_params = infer_global_hyperparameters(per_layer_params)
        if any(
            params.logits_soft_cap not in (None, 0.0)
            for params in per_layer_params.values()
        ):
            raise NotImplementedError(
                "TRTLLM attention does not support logits soft cap."
            )

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self._host_kv_cache_pool_mapping = torch.zeros(
            (num_layers, 2), dtype=torch.int32, device="cpu"
        )
        self._pin_memory = is_pin_memory_available()
        self._has_sliding_window = False
        self._has_mixed_window = False
        self._configure_decode_backend(per_layer_params, global_params)
        self.speculative_config = vllm_config.speculative_config
        self._spec_decode_enabled = bool(
            self.speculative_config
            and self.speculative_config.num_speculative_tokens
        )
        self._spec_decoding_position_offsets: torch.Tensor | None = None
        self._spec_decoding_packed_mask: torch.Tensor | None = None
        self._spec_decoding_generation_lengths: torch.Tensor | None = None
        self._spec_decoding_bl_tree_mask_offset: torch.Tensor | None = None
        self._spec_decoding_bl_tree_mask: torch.Tensor | None = None
        self._spec_bl_tree_first_sparse_mask_offset_kv: torch.Tensor | None = None
        self._spec_decoding_bl_tree_mask_size = 0
        self._spec_decoding_max_len = 0
        self._spec_decoding_mask_kind: str | None = None
        self._spec_decoding_tree_manager: SpecTreeManager | None = None
        self._spec_decoding_tree_is_linear = True
        self._spec_decoding_tree_len = 0
        self._spec_decoding_tree_num_blocks = 0
        self._spec_decoding_tree_packed_mask: torch.Tensor | None = None
        self._spec_decoding_tree_position_offsets: torch.Tensor | None = None
        if self._spec_decode_enabled:
            self._init_spec_decoding_tree_manager()

    def _configure_decode_backend(self, per_layer_params, global_params) -> None:
        decode_backend = _normalize_trtllm_decode_backend(
            envs.VLLM_TRTLLM_DECODE_BACKEND
        )
        has_sliding_window = any(
            params.window_left >= 0 for params in per_layer_params.values()
        )
        has_mixed_window = global_params.has_same_window_lefts is False
        self._has_sliding_window = has_sliding_window
        self._has_mixed_window = has_mixed_window
        num_q_per_kv = self.num_qo_heads // self.num_kv_heads
        if self.kv_cache_dtype == "nvfp4":
            if decode_backend not in (
                _TRTLLM_DECODE_BACKEND_AUTO,
                _TRTLLM_DECODE_BACKEND_TRTLLM_GEN,
            ):
                raise NotImplementedError(
                    "TRTLLM NVFP4 requires VLLM_TRTLLM_DECODE_BACKEND=auto or "
                    "trtllm-gen."
                )
            if current_platform.is_device_capability_family(120):
                raise NotImplementedError(
                    "TRTLLM-GEN decode backend is not supported on SM120 GPUs."
                )
            if not current_platform.is_device_capability_family(100):
                raise NotImplementedError(
                    "TRTLLM-GEN decode backend requires SM100 GPUs."
                )
            if num_q_per_kv > 16:
                raise NotImplementedError(
                    "TRTLLM-GEN requires num_q_heads/num_kv_heads <= 16."
                )
            _force_trtllm_xqa(False)
            if decode_backend == _TRTLLM_DECODE_BACKEND_AUTO:
                logger.info_once(
                    "TRTLLM decode backend forced to trtllm-gen for NVFP4."
                )
            return
        xqa_supported = (
            self.head_dim % 16 == 0
            and self.head_dim <= 256
            and self.block_size in _XQA_TOKENS_PER_BLOCK
            and num_q_per_kv <= 32
        )

        if decode_backend == _TRTLLM_DECODE_BACKEND_AUTO:
            if (
                current_platform.is_device_capability_family(100)
                and not current_platform.is_device_capability_family(120)
                and num_q_per_kv <= 16
            ):
                _force_trtllm_xqa(False)
                logger.info_once(
                    "TRTLLM decode backend auto-selected: trtllm-gen (SM100)."
                )
                return
            if xqa_supported:
                _force_trtllm_xqa(True)
                if has_mixed_window:
                    logger.info_once(
                        "TRTLLM decode backend auto-selected: xqa "
                        "(mixed sliding-window sizes detected)."
                    )
                else:
                    logger.info_once("TRTLLM decode backend auto-selected: xqa.")
                if has_sliding_window:
                    logger.warning_once(
                        "TRTLLM XQA selected with sliding window; kernels may "
                        "fall back to MMHA. Set VLLM_TRTLLM_DECODE_BACKEND=mmha "
                        "to avoid silent fallback."
                    )
                return
            _force_trtllm_xqa(False)
            logger.info_once("TRTLLM decode backend auto-selected: mmha.")
            return

        if decode_backend == _TRTLLM_DECODE_BACKEND_XQA:
            _force_trtllm_xqa(True)
            if not xqa_supported:
                reason = "unsupported head size or KV ratio"
                if self.block_size not in _XQA_TOKENS_PER_BLOCK:
                    reason = f"block_size not in {sorted(_XQA_TOKENS_PER_BLOCK)}"
                raise NotImplementedError(
                    f"TRTLLM XQA is not supported for this configuration ({reason})."
                )
            if has_sliding_window and has_mixed_window:
                logger.warning_once(
                    "TRTLLM XQA selected with mixed sliding-window sizes; "
                    "kernels may fall back to MMHA."
                )
            elif has_sliding_window:
                logger.warning_once(
                    "TRTLLM XQA selected with sliding window; kernels may "
                    "fall back to MMHA."
                )
        elif decode_backend == _TRTLLM_DECODE_BACKEND_TRTLLM_GEN:
            _force_trtllm_xqa(False)
            if current_platform.is_device_capability_family(120):
                raise NotImplementedError(
                    "TRTLLM-GEN decode backend is not supported on SM120 GPUs."
                )
            if not current_platform.is_device_capability_family(100):
                raise NotImplementedError(
                    "TRTLLM-GEN decode backend requires SM100 GPUs."
                )
            if num_q_per_kv > 16:
                raise NotImplementedError(
                    "TRTLLM-GEN requires num_q_heads/num_kv_heads <= 16."
                )
        elif decode_backend == _TRTLLM_DECODE_BACKEND_MMHA:
            _force_trtllm_xqa(False)
            logger.warning_once(
                "TRTLLM MMHA selection is advisory; XQA kernels may still be "
                "selected when supported."
            )
        else:
            return

    def _init_spec_decoding_tree_manager(self) -> None:
        spec_token_tree = (
            self.speculative_config.speculative_token_tree
            if self.speculative_config is not None
            else None
        )
        if not spec_token_tree:
            return
        try:
            tree_choices = ast.literal_eval(spec_token_tree)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning_once(
                "Failed to parse speculative_token_tree for TRTLLM "
                "spec decoding: %s",
                exc,
            )
            return
        if not tree_choices:
            return
        self._spec_decoding_tree_is_linear = _is_linear_spec_tree(tree_choices)
        if self._spec_decoding_tree_is_linear:
            return
        self._spec_decoding_tree_manager = SpecTreeManager(
            tree_choices=tree_choices,
            max_batch_size=self.max_num_requests,
            device=self.device,
        )
        self._spec_decoding_tree_len = (
            self._spec_decoding_tree_manager.total_drafts + 1
        )
        self._spec_decoding_tree_packed_mask = (
            self._spec_decoding_tree_manager.spec_dec_packed_mask
        )
        self._spec_decoding_tree_position_offsets = (
            self._spec_decoding_tree_manager.spec_dec_position_offsets
        )
        self._spec_decoding_tree_num_blocks = int(
            self._spec_decoding_tree_packed_mask.shape[1]
        )

    def _ensure_spec_decoding_buffers(self, max_query_len: int) -> None:
        if max_query_len <= 1:
            return
        capture_graph = torch.cuda.is_current_stream_capturing()
        max_len = int(max_query_len)
        num_blocks = (max_len + 31) // 32

        if capture_graph and (
            self._spec_decoding_position_offsets is None
            or self._spec_decoding_packed_mask is None
            or self._spec_decoding_generation_lengths is None
            or self._spec_decoding_position_offsets.shape[1] < max_len
            or self._spec_decoding_packed_mask.shape[1] < max_len
            or self._spec_decoding_packed_mask.shape[2] < num_blocks
        ):
            raise RuntimeError(
                "TRTLLM spec-decoding buffers are not initialized for CUDA "
                "graph capture. Run a warmup step to allocate them first."
            )

        if (
            self._spec_decoding_position_offsets is None
            or self._spec_decoding_position_offsets.shape[1] < max_len
        ):
            self._spec_decoding_position_offsets = torch.empty(
                (self.max_num_requests, max_len),
                dtype=torch.int32,
                device=self.device,
            )
        if (
            self._spec_decoding_packed_mask is None
            or self._spec_decoding_packed_mask.shape[1] < max_len
            or self._spec_decoding_packed_mask.shape[2] < num_blocks
        ):
            self._spec_decoding_packed_mask = torch.empty(
                (self.max_num_requests, max_len, num_blocks),
                dtype=torch.int32,
                device=self.device,
            )
        if (
            self._spec_decoding_generation_lengths is None
            or self._spec_decoding_generation_lengths.shape[0]
            < self.max_num_requests
        ):
            self._spec_decoding_generation_lengths = torch.empty(
                (self.max_num_requests,),
                dtype=torch.int32,
                device=self.device,
            )

        if (
            max_len <= self._spec_decoding_max_len
            and self._spec_decoding_mask_kind == "linear"
        ):
            return
        if capture_graph:
            raise RuntimeError(
                "TRTLLM spec-decoding buffers need regeneration during CUDA "
                "graph capture. Run a warmup step with the max spec length "
                "before capturing."
            )

        position_offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int32
        )
        self._spec_decoding_position_offsets[:, :max_len].copy_(position_offsets)

        positions = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(1)
        block_indices = torch.arange(
            num_blocks, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        num_valid = positions - block_indices * 32 + 1
        num_valid = num_valid.clamp(min=0, max=32)
        ones = torch.ones_like(num_valid, dtype=torch.int64)
        packed_mask = (ones << num_valid) - 1
        packed_mask = packed_mask.to(torch.int32)
        self._spec_decoding_packed_mask[:, :max_len, :num_blocks].copy_(
            packed_mask
        )
        self._spec_decoding_max_len = max_len
        self._spec_decoding_mask_kind = "linear"

    def _ensure_spec_decoding_tree_buffers(self, tree_len: int) -> None:
        if tree_len <= 1:
            return
        capture_graph = torch.cuda.is_current_stream_capturing()
        num_blocks = self._spec_decoding_tree_num_blocks
        if (
            self._spec_decoding_position_offsets is None
            or self._spec_decoding_position_offsets.shape[1] < tree_len
        ):
            self._spec_decoding_position_offsets = torch.empty(
                (self.max_num_requests, tree_len),
                dtype=torch.int32,
                device=self.device,
            )
        if (
            self._spec_decoding_packed_mask is None
            or self._spec_decoding_packed_mask.shape[1] < tree_len
            or self._spec_decoding_packed_mask.shape[2] < num_blocks
        ):
            self._spec_decoding_packed_mask = torch.empty(
                (self.max_num_requests, tree_len, num_blocks),
                dtype=torch.int32,
                device=self.device,
            )
        if (
            self._spec_decoding_generation_lengths is None
            or self._spec_decoding_generation_lengths.shape[0]
            < self.max_num_requests
        ):
            self._spec_decoding_generation_lengths = torch.empty(
                (self.max_num_requests,),
                dtype=torch.int32,
                device=self.device,
            )

        if (
            self._spec_decoding_mask_kind == "tree"
            and tree_len <= self._spec_decoding_max_len
        ):
            return
        if capture_graph:
            raise RuntimeError(
                "TRTLLM spec-decoding tree buffers need regeneration during "
                "CUDA graph capture. Run a warmup step with the tree length "
                "before capturing."
            )

        if (
            self._spec_decoding_tree_position_offsets is None
            or self._spec_decoding_tree_packed_mask is None
        ):
            raise RuntimeError(
                "TRTLLM spec-decoding tree buffers requested without "
                "initialized tree metadata."
            )

        position_offsets = self._spec_decoding_tree_position_offsets[:tree_len]
        packed_mask = self._spec_decoding_tree_packed_mask[:tree_len, :num_blocks]
        self._spec_decoding_position_offsets[: self.max_num_requests, :tree_len].copy_(
            position_offsets.unsqueeze(0).expand(self.max_num_requests, -1)
        )
        self._spec_decoding_packed_mask[
            : self.max_num_requests, :tree_len, :num_blocks
        ].copy_(
            packed_mask.unsqueeze(0).expand(self.max_num_requests, -1, -1)
        )
        self._spec_decoding_max_len = tree_len
        self._spec_decoding_mask_kind = "tree"

    @staticmethod
    def _compute_max_num_custom_mask_tiles_kv_upper_bound(
        max_seq_len_kv: int, min_first_sparse_mask_offset_kv: int, tile_size_kv_per_cta: int
    ) -> int:
        first_sparse_tile_offset = min_first_sparse_mask_offset_kv // tile_size_kv_per_cta
        num_tiles_kv_total = math.ceil(max_seq_len_kv / tile_size_kv_per_cta)
        return num_tiles_kv_total - first_sparse_tile_offset

    def _ensure_trtllm_gen_spec_decoding_buffers(
        self,
        max_seq_len: int,
        num_heads_per_kv: int,
        min_first_sparse_mask_offset_kv: int | None = None,
    ) -> None:
        if (
            self._spec_decoding_bl_tree_mask_offset is None
            or self._spec_decoding_bl_tree_mask_offset.shape[0]
            < self.max_num_requests
        ):
            self._spec_decoding_bl_tree_mask_offset = torch.zeros(
                (self.max_num_requests,),
                dtype=torch.int64,
                device=self.device,
            )
        if (
            self._spec_bl_tree_first_sparse_mask_offset_kv is None
            or self._spec_bl_tree_first_sparse_mask_offset_kv.shape[0]
            < self.max_num_requests
        ):
            self._spec_bl_tree_first_sparse_mask_offset_kv = torch.zeros(
                (self.max_num_requests,),
                dtype=torch.int32,
                device=self.device,
            )

        tile_size_kv = 128
        tile_size_q = 128
        num_instances_q = 1
        num_instances_kv = 2
        tile_size_kv_per_cta = tile_size_kv * num_instances_kv
        tile_size_q_per_cta = tile_size_q * num_instances_q
        min_first_sparse_mask_offset_kv = (
            int(min_first_sparse_mask_offset_kv)
            if min_first_sparse_mask_offset_kv is not None
            else 0
        )
        if min_first_sparse_mask_offset_kv < 0:
            min_first_sparse_mask_offset_kv = 0
        max_num_custom_mask_tiles_kv = (
            self._compute_max_num_custom_mask_tiles_kv_upper_bound(
                max_seq_len, min_first_sparse_mask_offset_kv, tile_size_kv_per_cta
            )
        )
        max_num_tiles_q = math.ceil(
            (max_seq_len * num_heads_per_kv) / tile_size_q_per_cta
        )
        mask_size = int(
            self.max_num_requests
            * max_num_tiles_q
            * max_num_custom_mask_tiles_kv
            * num_instances_q
            * num_instances_kv
            * tile_size_q
            * tile_size_kv
            / 32
        )
        if (
            self._spec_decoding_bl_tree_mask is None
            or self._spec_decoding_bl_tree_mask.numel() < mask_size
        ):
            self._spec_decoding_bl_tree_mask = torch.zeros(
                (mask_size,),
                dtype=torch.uint32,
                device=self.device,
            )
            self._spec_decoding_bl_tree_mask_size = mask_size

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

        num_reqs = seq_lens_cpu.numel()

        if attention_input_type == _ATTN_INPUT_GENERATION:
            predicted_tokens_per_seq = int(query_lens_cpu.max().item())
            if predicted_tokens_per_seq <= 0:
                predicted_tokens_per_seq = 1
        else:
            predicted_tokens_per_seq = 1

        capture_graph = torch.cuda.is_current_stream_capturing()
        is_spec_decoding_enabled = False
        use_spec_decoding = False
        is_spec_dec_tree = False
        spec_decoding_generation_lengths = None
        spec_decoding_position_offsets = None
        spec_decoding_packed_mask = None
        spec_decoding_bl_tree_mask_offset = None
        spec_decoding_bl_tree_mask = None
        spec_bl_tree_first_sparse_mask_offset_kv = None
        if (
            attention_input_type == _ATTN_INPUT_GENERATION
            and self._spec_decode_enabled
            and predicted_tokens_per_seq > 1
        ):
            custom_position_offsets = (
                common_attn_metadata.spec_decoding_position_offsets
            )
            custom_packed_mask = common_attn_metadata.spec_decoding_packed_mask
            custom_generation_lengths = (
                common_attn_metadata.spec_decoding_generation_lengths
            )
            custom_is_tree = common_attn_metadata.spec_decoding_is_tree
            custom_bl_tree_mask_offset = (
                common_attn_metadata.spec_decoding_bl_tree_mask_offset
            )
            custom_bl_tree_mask = common_attn_metadata.spec_decoding_bl_tree_mask
            custom_first_sparse_offset = (
                common_attn_metadata.spec_bl_tree_first_sparse_mask_offset_kv
            )
            custom_masks = (
                custom_position_offsets is not None or custom_packed_mask is not None
            )
            if custom_masks and (
                custom_position_offsets is None or custom_packed_mask is None
            ):
                raise ValueError(
                    "TRTLLM spec decoding requires both position offsets and "
                    "packed mask tensors when custom masks are provided."
                )

            use_tree_mask = (
                self._spec_decoding_tree_manager is not None
                and not self._spec_decoding_tree_is_linear
                and not custom_masks
            )
            tree_enabled = bool(custom_is_tree) if custom_masks else use_tree_mask
            if custom_masks and custom_is_tree is None:
                logger.warning_once(
                    "TRTLLM custom spec-decoding masks provided without "
                    "spec_decoding_is_tree; assuming linear masks."
                )

            if tree_enabled and self._has_sliding_window:
                raise NotImplementedError(
                    "TRTLLM spec-decoding tree does not support sliding-window "
                    "attention. Disable sliding window or use a different backend."
                )
            if (
                tree_enabled
                and current_platform.is_device_capability_family(100)
                and not current_platform.is_device_capability_family(120)
            ):
                raise NotImplementedError(
                    "TRTLLM-GEN does not support spec-decoding tree on SM100. "
                    "Disable tree spec decoding or use SM120."
                )

            if custom_masks:
                if custom_position_offsets.dim() != 2:
                    raise ValueError(
                        "TRTLLM spec_decoding_position_offsets must be 2D."
                    )
                if custom_packed_mask.dim() != 3:
                    raise ValueError(
                        "TRTLLM spec_decoding_packed_mask must be 3D."
                    )
                if custom_position_offsets.shape[0] < num_reqs:
                    raise ValueError(
                        "TRTLLM spec_decoding_position_offsets batch dimension "
                        "is smaller than num_reqs."
                    )
                if custom_packed_mask.shape[0] < num_reqs:
                    raise ValueError(
                        "TRTLLM spec_decoding_packed_mask batch dimension "
                        "is smaller than num_reqs."
                    )
                if custom_position_offsets.shape[1] != custom_packed_mask.shape[1]:
                    raise ValueError(
                        "TRTLLM spec_decoding_position_offsets and "
                        "spec_decoding_packed_mask length mismatch."
                    )

                spec_len = int(custom_position_offsets.shape[1])
                num_blocks = int(custom_packed_mask.shape[2])
                if predicted_tokens_per_seq != spec_len:
                    raise ValueError(
                        "TRTLLM spec-decoding mask length mismatch: "
                        f"expected {predicted_tokens_per_seq}, got {spec_len}."
                    )
                if custom_position_offsets.device != self.device:
                    if capture_graph:
                        raise RuntimeError(
                            "TRTLLM spec-decoding position offsets must be on "
                            "device before CUDA graph capture."
                        )
                    custom_position_offsets = custom_position_offsets.to(
                        device=self.device, non_blocking=self._pin_memory
                    )
                if custom_packed_mask.device != self.device:
                    if capture_graph:
                        raise RuntimeError(
                            "TRTLLM spec-decoding packed mask must be on "
                            "device before CUDA graph capture."
                        )
                    custom_packed_mask = custom_packed_mask.to(
                        device=self.device, non_blocking=self._pin_memory
                    )
                if custom_position_offsets.dtype != torch.int32:
                    if capture_graph:
                        raise RuntimeError(
                            "TRTLLM spec-decoding position offsets must be "
                            "int32 before CUDA graph capture."
                        )
                    custom_position_offsets = custom_position_offsets.to(torch.int32)
                if custom_packed_mask.dtype != torch.int32:
                    if capture_graph:
                        raise RuntimeError(
                            "TRTLLM spec-decoding packed mask must be int32 "
                            "before CUDA graph capture."
                        )
                    custom_packed_mask = custom_packed_mask.to(torch.int32)

                spec_decoding_position_offsets = custom_position_offsets[
                    request_slice, :spec_len
                ]
                spec_decoding_packed_mask = custom_packed_mask[
                    request_slice, :spec_len, :num_blocks
                ]
                is_spec_dec_tree = bool(custom_is_tree)
            else:
                num_blocks = (predicted_tokens_per_seq + 31) // 32
                spec_len = predicted_tokens_per_seq
                if use_tree_mask:
                    tree_len = self._spec_decoding_tree_len
                    if predicted_tokens_per_seq != tree_len:
                        raise ValueError(
                            "TRTLLM spec-decoding tree length mismatch: "
                            f"expected {tree_len}, got {predicted_tokens_per_seq}."
                        )
                    self._ensure_spec_decoding_tree_buffers(tree_len)
                    num_blocks = self._spec_decoding_tree_num_blocks
                    spec_len = tree_len
                    is_spec_dec_tree = True
                else:
                    self._ensure_spec_decoding_buffers(predicted_tokens_per_seq)

                assert self._spec_decoding_position_offsets is not None
                assert self._spec_decoding_packed_mask is not None
                spec_decoding_position_offsets = (
                    self._spec_decoding_position_offsets[:num_reqs, :spec_len]
                )
                spec_decoding_packed_mask = self._spec_decoding_packed_mask[
                    :num_reqs, :spec_len, :num_blocks
                ]

            if custom_generation_lengths is not None:
                if custom_generation_lengths.dim() != 1:
                    raise ValueError(
                        "TRTLLM spec_decoding_generation_lengths must be 1D."
                    )
                if custom_generation_lengths.shape[0] < num_reqs:
                    raise ValueError(
                        "TRTLLM spec_decoding_generation_lengths batch dimension "
                        "is smaller than num_reqs."
                    )
                if custom_generation_lengths.device != self.device:
                    if capture_graph:
                        raise RuntimeError(
                            "TRTLLM spec-decoding generation lengths must be "
                            "on device before CUDA graph capture."
                        )
                    custom_generation_lengths = custom_generation_lengths.to(
                        device=self.device, non_blocking=self._pin_memory
                    )
                if custom_generation_lengths.dtype != torch.int32:
                    if capture_graph:
                        raise RuntimeError(
                            "TRTLLM spec-decoding generation lengths must be "
                            "int32 before CUDA graph capture."
                        )
                    custom_generation_lengths = custom_generation_lengths.to(
                        torch.int32
                    )
                if (
                    self._spec_decoding_generation_lengths is None
                    or self._spec_decoding_generation_lengths.shape[0]
                    < self.max_num_requests
                ):
                    self._spec_decoding_generation_lengths = torch.empty(
                        (self.max_num_requests,),
                        dtype=torch.int32,
                        device=self.device,
                    )
                self._spec_decoding_generation_lengths[:num_reqs].copy_(
                    custom_generation_lengths[:num_reqs], non_blocking=True
                )
            else:
                generation_lengths = query_lens_cpu.to(
                    device=self.device, non_blocking=self._pin_memory
                )
                if (
                    self._spec_decoding_generation_lengths is None
                    or self._spec_decoding_generation_lengths.shape[0]
                    < self.max_num_requests
                ):
                    self._spec_decoding_generation_lengths = torch.empty(
                        (self.max_num_requests,),
                        dtype=torch.int32,
                        device=self.device,
                    )
                self._spec_decoding_generation_lengths[:num_reqs].copy_(
                    generation_lengths, non_blocking=True
                )
            if num_reqs < self.max_num_requests:
                assert self._spec_decoding_generation_lengths is not None
                self._spec_decoding_generation_lengths[num_reqs:].fill_(0)
            spec_decoding_generation_lengths = (
                self._spec_decoding_generation_lengths[:num_reqs]
            )

            if current_platform.is_device_capability_family(120):
                is_spec_decoding_enabled = True
                use_spec_decoding = True
            elif current_platform.is_device_capability_family(100):
                num_heads_per_kv = self.num_qo_heads // self.num_kv_heads
                min_first_sparse = None
                if custom_first_sparse_offset is not None:
                    if custom_first_sparse_offset.dim() != 1:
                        raise ValueError(
                            "TRTLLM spec_bl_tree_first_sparse_mask_offset_kv "
                            "must be 1D."
                        )
                    if custom_first_sparse_offset.shape[0] < num_reqs:
                        raise ValueError(
                            "TRTLLM spec_bl_tree_first_sparse_mask_offset_kv "
                            "batch dimension is smaller than num_reqs."
                        )
                    if custom_first_sparse_offset.device != self.device:
                        if capture_graph:
                            raise RuntimeError(
                                "TRTLLM spec_bl_tree_first_sparse_mask_offset_kv "
                                "must be on device before CUDA graph capture."
                            )
                        custom_first_sparse_offset = custom_first_sparse_offset.to(
                            device=self.device, non_blocking=self._pin_memory
                        )
                    if custom_first_sparse_offset.dtype != torch.int32:
                        if capture_graph:
                            raise RuntimeError(
                                "TRTLLM spec_bl_tree_first_sparse_mask_offset_kv "
                                "must be int32 before CUDA graph capture."
                            )
                        custom_first_sparse_offset = custom_first_sparse_offset.to(
                            torch.int32
                        )
                    min_first_sparse = int(
                        custom_first_sparse_offset[:num_reqs].min().item()
                    )
                else:
                    min_first_sparse = int(context_lens_cpu.min().item())
                self._ensure_trtllm_gen_spec_decoding_buffers(
                    common_attn_metadata.max_seq_len,
                    num_heads_per_kv,
                    min_first_sparse,
                )
                assert self._spec_decoding_bl_tree_mask_offset is not None
                assert self._spec_decoding_bl_tree_mask is not None
                assert self._spec_bl_tree_first_sparse_mask_offset_kv is not None
                if custom_bl_tree_mask_offset is not None:
                    if custom_bl_tree_mask_offset.dim() != 1:
                        raise ValueError(
                            "TRTLLM spec_decoding_bl_tree_mask_offset must be 1D."
                        )
                    if custom_bl_tree_mask_offset.shape[0] < num_reqs:
                        raise ValueError(
                            "TRTLLM spec_decoding_bl_tree_mask_offset batch "
                            "dimension is smaller than num_reqs."
                        )
                    if custom_bl_tree_mask_offset.device != self.device:
                        if capture_graph:
                            raise RuntimeError(
                                "TRTLLM spec_decoding_bl_tree_mask_offset must "
                                "be on device before CUDA graph capture."
                            )
                        custom_bl_tree_mask_offset = custom_bl_tree_mask_offset.to(
                            device=self.device, non_blocking=self._pin_memory
                        )
                    if custom_bl_tree_mask_offset.dtype != torch.int64:
                        if capture_graph:
                            raise RuntimeError(
                                "TRTLLM spec_decoding_bl_tree_mask_offset must "
                                "be int64 before CUDA graph capture."
                            )
                        custom_bl_tree_mask_offset = custom_bl_tree_mask_offset.to(
                            torch.int64
                        )
                    spec_decoding_bl_tree_mask_offset = custom_bl_tree_mask_offset[
                        request_slice
                    ]
                else:
                    self._spec_decoding_bl_tree_mask_offset[:num_reqs].fill_(0)
                    spec_decoding_bl_tree_mask_offset = (
                        self._spec_decoding_bl_tree_mask_offset[:num_reqs]
                    )

                if custom_first_sparse_offset is not None:
                    self._spec_bl_tree_first_sparse_mask_offset_kv[
                        :num_reqs
                    ].copy_(custom_first_sparse_offset[:num_reqs], non_blocking=True)
                else:
                    self._spec_bl_tree_first_sparse_mask_offset_kv[
                        :num_reqs
                    ].copy_(
                        context_lens_cpu.to(device=self.device, dtype=torch.int32),
                        non_blocking=True,
                    )
                spec_bl_tree_first_sparse_mask_offset_kv = (
                    self._spec_bl_tree_first_sparse_mask_offset_kv[:num_reqs]
                )

                if custom_bl_tree_mask is not None:
                    if custom_bl_tree_mask.device != self.device:
                        if capture_graph:
                            raise RuntimeError(
                                "TRTLLM spec_decoding_bl_tree_mask must be on "
                                "device before CUDA graph capture."
                            )
                        custom_bl_tree_mask = custom_bl_tree_mask.to(
                            device=self.device, non_blocking=self._pin_memory
                        )
                    if custom_bl_tree_mask.dtype != torch.uint32:
                        if capture_graph:
                            raise RuntimeError(
                                "TRTLLM spec_decoding_bl_tree_mask must be "
                                "uint32 before CUDA graph capture."
                            )
                        custom_bl_tree_mask = custom_bl_tree_mask.to(torch.uint32)
                    spec_decoding_bl_tree_mask = custom_bl_tree_mask
                else:
                    spec_decoding_bl_tree_mask = self._spec_decoding_bl_tree_mask
                is_spec_decoding_enabled = True
                use_spec_decoding = True
            else:
                logger.warning_once(
                    "TRTLLM spec decoding is only supported on SM100/SM120; "
                    "running TRTLLM attention without spec-decoding masks."
                )

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

        beam_width = int(getattr(common_attn_metadata, "beam_width", 1))
        if beam_width < 1:
            raise ValueError("TRTLLM beam_width must be >= 1.")
        cache_indirection = getattr(common_attn_metadata, "cache_indirection", None)
        if beam_width > 1:
            if cache_indirection is None:
                raise NotImplementedError(
                    "TRTLLM beam search requires cache_indirection tensor."
                )
            if cache_indirection.dim() < 2:
                raise ValueError(
                    "TRTLLM cache_indirection must have at least 2 dimensions."
                )
            if cache_indirection.shape[0] < num_reqs:
                raise ValueError(
                    "TRTLLM cache_indirection batch dimension is smaller than "
                    "num_reqs."
                )
            if cache_indirection.device != self.device:
                if capture_graph:
                    raise RuntimeError(
                        "TRTLLM cache_indirection must be on device before "
                        "CUDA graph capture."
                    )
                cache_indirection = cache_indirection.to(
                    device=self.device, non_blocking=self._pin_memory
                )
            if cache_indirection.dtype != torch.int32:
                if capture_graph:
                    raise RuntimeError(
                        "TRTLLM cache_indirection must be int32 before CUDA "
                        "graph capture."
                    )
                cache_indirection = cache_indirection.to(torch.int32)
            cache_indirection = cache_indirection[request_slice]
        else:
            cache_indirection = None

        return TRTLLMCallMetadata(
            num_reqs=num_reqs,
            num_tokens=int(seq_lens_cpu.sum().item()),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            is_spec_decoding_enabled=is_spec_decoding_enabled,
            use_spec_decoding=use_spec_decoding,
            is_spec_dec_tree=is_spec_dec_tree,
            spec_decoding_generation_lengths=spec_decoding_generation_lengths,
            spec_decoding_position_offsets=spec_decoding_position_offsets,
            spec_decoding_packed_mask=spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=(
                spec_bl_tree_first_sparse_mask_offset_kv
            ),
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
            beam_width=beam_width,
            cache_indirection=cache_indirection,
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
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(8)]

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
        if cache_dtype_str == "nvfp4":
            if head_size % 16 != 0 or head_size > 256:
                raise ValueError(
                    "TRTLLM NVFP4 KV cache requires head_size to be a "
                    "multiple of 16 and <= 256."
                )
            packed_head_size = head_size // 8
            return (num_blocks, 2, block_size, num_kv_heads, packed_head_size)
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
    def supports_sink(cls) -> bool:
        return True

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
        if envs.VLLM_TRTLLM_DISABLE:
            invalid_reasons.append("TRTLLM attention disabled by VLLM_TRTLLM_DISABLE")
        if not has_trtllm_thop():
            invalid_reasons.append("TRTLLM bindings not available")
        if head_size % 8 != 0 or head_size > 256:
            invalid_reasons.append("unsupported head size for TRTLLM attention")
        if kv_cache_dtype == "nvfp4":
            if head_size % 16 != 0:
                invalid_reasons.append("nvfp4 requires head_size multiple of 16")
            if head_size > 256:
                invalid_reasons.append("nvfp4 requires head_size <= 256")
        return invalid_reasons


__all__ = ["TRTLLMAttentionBackend"]
