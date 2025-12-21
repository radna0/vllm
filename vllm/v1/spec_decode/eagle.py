# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import time
from contextlib import contextmanager, nullcontext
from typing import Any
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.trtllm_attn import TRTLLMAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.spec_tree_manager import SpecTreeManager
from vllm.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.cudagraph_utils import get_cudagraph_sizes
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class _EagleDraftLoopWrapper(nn.Module):
    def __init__(self, proposer: "EagleProposer"):
        super().__init__()
        self.proposer = proposer

    def forward(
        self,
        batch_size: int,
        input_batch_size: int,
        per_layer_attn_metadata: dict[str, Any],
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        sampling_metadata: SamplingMetadata,
        force_greedy: bool,
    ) -> None:
        self.proposer._run_draft_loop(
            batch_size=batch_size,
            input_batch_size=input_batch_size,
            per_layer_attn_metadata=per_layer_attn_metadata,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            sampling_metadata=sampling_metadata,
            force_greedy=force_greedy,
        )


class EagleProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        self._trace_enabled = envs.VLLM_SPEC_DECODE_TRACE
        self._trace_interval_s = envs.VLLM_SPEC_DECODE_TRACE_INTERVAL
        self._trace_sync = envs.VLLM_SPEC_DECODE_TRACE_SYNC
        self._trace_last_log = time.monotonic()
        self._trace_draft_s = 0.0
        self._trace_draft_calls = 0
        self._fast_logits = envs.VLLM_EAGLE_FAST_LOGITS
        self._allow_random_cudagraph = envs.VLLM_EAGLE_CUDAGRAPH_ALLOW_RANDOM
        self._opt_mode = envs.VLLM_EAGLE_OPT_MODE
        self._use_packed_kv_compact = self._opt_mode != "baseline"
        self._dynamic_tree = envs.VLLM_EAGLE_DYNAMIC_TREE
        self._dynamic_topk = max(0, envs.VLLM_EAGLE_DYNAMIC_TOPK)
        self._dynamic_tree_kernels = envs.VLLM_EAGLE_DYNAMIC_TREE_KERNELS
        self._use_tree_cuda_draft = envs.VLLM_EAGLE_CUDA_TREE_DRAFT
        self._dynamic_buffers_initialized = False
        self._dyn_paths_a: torch.Tensor | None = None
        self._dyn_paths_b: torch.Tensor | None = None
        self._dyn_prev_scores: torch.Tensor | None = None
        self._dyn_cur_scores: torch.Tensor | None = None
        self._dyn_second_topk_ids: torch.Tensor | None = None
        self._dyn_next_expand_indices: torch.Tensor | None = None
        self._dyn_current_expand_indices: torch.Tensor | None = None
        self._dyn_draft_ids_in: torch.Tensor | None = None
        self._dyn_draft_ids_out: torch.Tensor | None = None
        self._dyn_draft_lens_in: torch.Tensor | None = None
        self._dyn_draft_lens_out: torch.Tensor | None = None
        self._dyn_output_scores: torch.Tensor | None = None
        self._dyn_all_layers_scores: torch.Tensor | None = None
        self._dyn_all_layers_draft_ids: torch.Tensor | None = None
        self._dyn_all_layers_predecessor: torch.Tensor | None = None
        self._dyn_first_topk_logprobs: torch.Tensor | None = None
        self._dyn_first_topk_ids: torch.Tensor | None = None
        self._dyn_third_topk_input_ptrs: torch.Tensor | None = None
        self._dyn_third_topk_output_ptrs: torch.Tensor | None = None
        self._dyn_third_topk_ids: torch.Tensor | None = None
        self._dyn_third_topks: torch.Tensor | None = None
        self._dyn_candidate_scores: torch.Tensor | None = None
        self._dyn_candidate_ids: torch.Tensor | None = None
        self._dyn_second_topk_tokens: torch.Tensor | None = None
        self._dyn_second_topk_input_ptrs: torch.Tensor | None = None
        self._dyn_second_topk_output_ptrs: torch.Tensor | None = None
        self._dyn_second_topk_logprobs: torch.Tensor | None = None
        self._dyn_use_paths_a = True
        self._dyn_paths_final: torch.Tensor | None = None
        self._dynamic_max_draft_tokens = 0
        self._dynamic_max_tokens = 0
        self._dynamic_max_path_len = 0
        self._dynamic_tree_warned = False
        self._dyn_node_token_ids: torch.Tensor | None = None
        self._dyn_position_offsets: torch.Tensor | None = None
        self._dyn_cached_req_ids: list[str] | None = None
        self._dyn_cached_req_id_to_index: dict[str, int] | None = None
        self._dyn_cached_paths: torch.Tensor | None = None
        self._dyn_cached_node_token_ids: torch.Tensor | None = None
        self._dyn_cached_position_offsets: torch.Tensor | None = None
        self._dyn_cached_packed_mask: torch.Tensor | None = None
        self._dyn_cached_generation_lengths: torch.Tensor | None = None
        self._dyn_cached_tree_len = 0
        self._dynamic_active_topk = 0
        self._dyn_tree_draft_pos_offsets: torch.Tensor | None = None
        if self._opt_mode:
            logger.info("EAGLE opt mode: %s", self._opt_mode)
        if self._dynamic_tree:
            logger.warning(
                "EAGLE dynamic tree enabled: drafts are selected dynamically. "
                "TRTLLM backends consume dynamic masks for reduced draft compute; "
                "other backends still draft from static tree logits."
            )
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()
        self.vocab_size = self.draft_model_config.get_vocab_size()
        pin_memory = is_pin_memory_available()
        self._spec_query_start_loc_cpu = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self._spec_seq_lens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self._spec_token_indices_cpu = torch.empty(
            self.max_num_tokens,
            dtype=torch.int32,
            pin_memory=pin_memory,
        )

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.draft_indexer_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []
        self.indexer_layer_names: list[str] = []
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.use_cuda_graph = False

        self.compilation_config = self.vllm_config.compilation_config
        if self.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            cudagraph_mode = self.compilation_config.cudagraph_mode
            if cudagraph_mode != CUDAGraphMode.NONE and not cudagraph_mode.has_mode(
                CUDAGraphMode.PIECEWISE
            ):
                logger.warning(
                    "Currently the eagle proposer only supports cudagraph_mode "
                    "PIECEWISE, if you want the drafter to use cuda graphs, "
                    "please set compilation_config.cudagraph_mode to PIECEWISE "
                    "or FULL_AND_PIECEWISE"
                )
            self.use_cuda_graph = (
                cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE)
                and not self.speculative_config.enforce_eager
            )

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        self.uses_mrope = self.vllm_config.model_config.uses_mrope
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_num_tokens + 1), dtype=torch.int64, device=device
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )
        self.arange_int64 = self.arange.to(dtype=torch.int64)
        self._draft_tokens = torch.zeros(
            max_batch_size,
            self.num_speculative_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._draft_temperature = torch.empty(
            max_batch_size, dtype=torch.float32, device=device
        )
        self._draft_top_k = torch.empty(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._draft_top_p = torch.empty(
            max_batch_size, dtype=torch.float32, device=device
        )
        self._draft_seq_lens_backup = torch.empty(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._draft_slot_mapping_backup = torch.empty(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._draft_sampling_ready = False
        self._draft_seq_lens = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._draft_slot_mapping = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._draft_query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: max_batch_size + 1]
        ).clone()
        self._draft_block_table = torch.zeros(
            max_batch_size,
            (self.max_model_len + self.block_size - 1) // self.block_size,
            dtype=torch.int32,
            device=device,
        )
        self._draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._draft_graph_pool = torch.cuda.graph_pool_handle()
        self._draft_cudagraph_sizes: dict[int, int] = {}
        self._draft_common_attn_metadata_cache: dict[int, CommonAttentionMetadata] = {}
        self._draft_loop = _EagleDraftLoopWrapper(self)
        if self.compilation_config.cudagraph_capture_sizes:
            self._draft_cudagraph_sizes = get_cudagraph_sizes(
                self.compilation_config.cudagraph_capture_sizes,
                max_batch_size,
                max_batch_size,
                CUDAGraphMode.FULL,
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            rocm_types = [TritonAttentionMetadata, FlashAttentionMetadata]
            # ROCM_AITER_FA is an optional backend
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.v1.attention.backends.mla.common import MLACommonMetadata

            rocm_types.append(MLACommonMetadata)

            self.allowed_attn_types = tuple(rocm_types)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        self.tree_choices: list[tuple[int, ...]] = ast.literal_eval(spec_token_tree)
        self.tree_manager = SpecTreeManager(
            self.tree_choices, max_batch_size=max_batch_size, device=device
        )
        self.num_drafts_per_level = self.tree_manager.num_drafts_per_level
        self.cu_drafts_per_level = self.tree_manager.cu_drafts_per_level
        self.tree_draft_pos_offsets = self.tree_manager.tree_draft_pos_offsets
        self._tree_total_drafts = self.tree_manager.total_drafts
        self._tree_level_offsets = self.tree_manager.level_offsets
        self._tree_query_start_loc = self.tree_manager.query_start_loc
        self._tree_parent_indices = self.tree_manager.parent_indices_per_level
        self._tree_child_indices = self.tree_manager.child_indices_per_level
        self._tree_children_per_parent = self.tree_manager.children_per_parent_per_level
        self._tree_max_topk_per_level = self.tree_manager.max_topk_per_level
        self._tree_root_num_children = self.tree_manager.root_children_count
        self._tree_tokens_gather_idx = (
            self.tree_manager.tokens_gather_idx_for_drafter_model
        )
        self._tree_top_k_list = self.tree_manager.top_k_list_cuda
        self._tree_draft_tokens_indices_cumsum = (
            self.tree_manager.draft_tokens_indices_cumsum
        )
        self._tree_max_top_k = self.tree_manager.max_top_k
        self._tree_nodes_per_level = [
            torch.tensor(nodes, dtype=torch.int64, device=device)
            for nodes in self.tree_manager.nodes_per_level
        ]
        self._tree_draft_tokens_buffer: torch.Tensor | None = None
        self._tree_new_draft_tokens: torch.Tensor | None = None
        if self._use_tree_cuda_draft and self._tree_total_drafts > 0:
            self._tree_draft_tokens_buffer = torch.full(
                (self.max_num_reqs, self._tree_total_drafts + 1),
                PADDING_SLOT_ID,
                dtype=torch.int32,
                device=device,
            )
            if self._tree_max_top_k > 0:
                self._tree_new_draft_tokens = torch.full(
                    (
                        self.max_num_reqs,
                        self._tree_total_drafts + 1,
                        self._tree_max_top_k,
                    ),
                    PADDING_SLOT_ID,
                    dtype=torch.int32,
                    device=device,
                )
        self._tree_slot_mapping = torch.empty(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._tree_depth = len(self.num_drafts_per_level)
        self._tree_is_linear = all(
            count <= 1 for count in self.num_drafts_per_level
        )
        self._tree_level_offsets_tensor = torch.tensor(
            self._tree_level_offsets, dtype=torch.int32, device=device
        )
        self._tree_level_sizes_tensor = torch.tensor(
            self.num_drafts_per_level, dtype=torch.int32, device=device
        )
        children_offsets_list: list[torch.Tensor] = []
        children_offsets_start: list[int] = []
        offset = 0
        for counts in self._tree_children_per_parent:
            children_offsets_start.append(offset)
            counts_i32 = counts.to(dtype=torch.int32)
            offsets = torch.zeros(
                counts_i32.numel() + 1, dtype=torch.int32, device=device
            )
            if counts_i32.numel() > 0:
                offsets[1:] = torch.cumsum(counts_i32, dim=0)
            children_offsets_list.append(offsets)
            offset += offsets.numel()
        if children_offsets_list:
            self._tree_children_offsets_flat = torch.cat(children_offsets_list, dim=0)
        else:
            self._tree_children_offsets_flat = torch.empty(
                (0,), dtype=torch.int32, device=device
            )
        self._tree_children_offsets_start = torch.tensor(
            children_offsets_start, dtype=torch.int32, device=device
        )
        self._tree_accept_indices = torch.full(
            (self.max_num_reqs, self._tree_depth),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._tree_accept_counts = torch.zeros(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self._tree_accept_offsets = torch.zeros(
            (self.max_num_reqs + 1,), dtype=torch.int32, device=device
        )
        if self._dynamic_tree:
            if self._dynamic_topk <= 0:
                self._dynamic_topk = self._tree_root_num_children
            if self._dynamic_topk <= 0:
                raise ValueError("VLLM_EAGLE_DYNAMIC_TOPK must be > 0")

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        else:
            self.positions[:num_tokens] = positions

    def _get_draft_cudagraph_size(self, batch_size: int) -> int | None:
        return self._draft_cudagraph_sizes.get(batch_size)

    def _ensure_dynamic_tree_buffers(self) -> None:
        if self._dynamic_buffers_initialized or not self._dynamic_tree:
            return
        dyn_k = self._dynamic_topk
        max_draft_tokens = dyn_k * max(self._tree_depth, 1)
        self._dynamic_max_draft_tokens = max_draft_tokens
        self._dynamic_max_tokens = max_draft_tokens + 1
        self._dynamic_max_path_len = max(self._tree_depth + 1, 2)
        device = self.device
        self._dyn_paths_a = torch.full(
            (self.max_num_reqs, self._dynamic_max_tokens, self._dynamic_max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_paths_b = torch.full(
            (self.max_num_reqs, self._dynamic_max_tokens, self._dynamic_max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_prev_scores = torch.full(
            (self.max_num_reqs, dyn_k), -float("inf"), dtype=torch.float32, device=device
        )
        self._dyn_cur_scores = torch.empty(
            (self.max_num_reqs, dyn_k), dtype=torch.float32, device=device
        )
        self._dyn_second_topk_ids = torch.empty(
            (self.max_num_reqs, dyn_k), dtype=torch.int32, device=device
        )
        self._dyn_next_expand_indices = torch.empty(
            (self.max_num_reqs, dyn_k), dtype=torch.int32, device=device
        )
        self._dyn_current_expand_indices = torch.zeros(
            (self.max_num_reqs, dyn_k), dtype=torch.int32, device=device
        )
        self._dyn_draft_ids_in = torch.zeros(
            (self.max_num_reqs, max_draft_tokens), dtype=torch.int32, device=device
        )
        self._dyn_draft_ids_out = torch.zeros(
            (self.max_num_reqs, max_draft_tokens), dtype=torch.int32, device=device
        )
        self._dyn_draft_lens_in = torch.zeros(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self._dyn_draft_lens_out = torch.zeros(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self._dyn_output_scores = torch.empty(
            (self.max_num_reqs, max_draft_tokens),
            dtype=torch.float32,
            device=device,
        )
        all_layers_size = self._tree_depth * max_draft_tokens * max_draft_tokens
        self._dyn_all_layers_scores = torch.full(
            (self.max_num_reqs, all_layers_size),
            -float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self._dyn_all_layers_draft_ids = torch.full(
            (self.max_num_reqs, all_layers_size),
            PADDING_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_all_layers_predecessor = torch.full(
            (self.max_num_reqs, all_layers_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_first_topk_logprobs = torch.full(
            (self.max_num_reqs * dyn_k, max_draft_tokens),
            -float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self._dyn_first_topk_ids = torch.full(
            (self.max_num_reqs * dyn_k, max_draft_tokens),
            PADDING_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_third_topk_input_ptrs = torch.empty(
            (self.max_num_reqs,), dtype=torch.int64, device=device
        )
        self._dyn_third_topk_output_ptrs = torch.empty(
            (self.max_num_reqs,), dtype=torch.int64, device=device
        )
        self._dyn_third_topk_ids = torch.full(
            (self.max_num_reqs, max_draft_tokens),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_third_topks = torch.empty(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self._dyn_candidate_scores = torch.empty(
            (self.max_num_reqs * dyn_k, max_draft_tokens),
            dtype=torch.float32,
            device=device,
        )
        self._dyn_candidate_ids = torch.full(
            (self.max_num_reqs, dyn_k, max_draft_tokens),
            PADDING_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_second_topk_tokens = torch.empty(
            (self.max_num_reqs, dyn_k), dtype=torch.int32, device=device
        )
        self._dyn_second_topk_input_ptrs = torch.empty(
            (self.max_num_reqs,), dtype=torch.int64, device=device
        )
        self._dyn_second_topk_output_ptrs = torch.empty(
            (self.max_num_reqs,), dtype=torch.int64, device=device
        )
        self._dyn_second_topk_logprobs = torch.empty(
            (self.max_num_reqs, max_draft_tokens),
            dtype=torch.float32,
            device=device,
        )
        self._dyn_node_token_ids = torch.full(
            (self.max_num_reqs, self._dynamic_max_tokens),
            PADDING_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_paths_final = torch.full(
            (self.max_num_reqs, self._dynamic_max_tokens, self._dynamic_max_path_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._dyn_position_offsets = torch.zeros(
            (self._dynamic_max_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self._dyn_tree_draft_pos_offsets = torch.arange(
            1, max_draft_tokens + 1, device=device, dtype=torch.int32
        ).repeat(self.max_num_reqs, 1)
        if dyn_k > 0:
            for level in range(self._tree_depth):
                start = 1 + level * dyn_k
                end = start + dyn_k
                if end <= self._dynamic_max_tokens:
                    self._dyn_position_offsets[start:end] = level + 1
        self._dyn_use_paths_a = True
        self._dynamic_buffers_initialized = True

    def _reset_dynamic_tree_state(self, batch_size: int) -> None:
        if not self._dynamic_tree or not self._dynamic_buffers_initialized:
            return
        assert self._dyn_prev_scores is not None
        assert self._dyn_draft_lens_in is not None
        assert self._dyn_draft_lens_out is not None
        assert self._dyn_paths_a is not None
        assert self._dyn_paths_b is not None
        assert self._dyn_node_token_ids is not None
        assert self._dyn_current_expand_indices is not None
        assert self._dyn_all_layers_scores is not None
        assert self._dyn_all_layers_draft_ids is not None
        assert self._dyn_all_layers_predecessor is not None
        assert self._dyn_first_topk_logprobs is not None
        assert self._dyn_first_topk_ids is not None
        assert self._dyn_third_topk_ids is not None
        assert self._dyn_third_topks is not None
        assert self._dyn_paths_final is not None
        self._dyn_prev_scores[:batch_size].fill_(-float("inf"))
        self._dyn_draft_lens_in[:batch_size].zero_()
        self._dyn_draft_lens_out[:batch_size].zero_()
        self._dyn_paths_a[:batch_size].fill_(-1)
        self._dyn_paths_b[:batch_size].fill_(-1)
        self._dyn_paths_final[:batch_size].fill_(-1)
        self._dyn_node_token_ids[:batch_size].fill_(PADDING_SLOT_ID)
        self._dyn_current_expand_indices[:batch_size].zero_()
        self._dyn_all_layers_scores[:batch_size].fill_(-float("inf"))
        self._dyn_all_layers_draft_ids[:batch_size].fill_(PADDING_SLOT_ID)
        self._dyn_all_layers_predecessor[:batch_size].fill_(-1)
        dyn_k = self._dynamic_topk
        if dyn_k > 0:
            self._dyn_first_topk_logprobs[: batch_size * dyn_k].fill_(-float("inf"))
            self._dyn_first_topk_ids[: batch_size * dyn_k].fill_(PADDING_SLOT_ID)
        self._dyn_third_topk_ids[:batch_size].fill_(-1)
        self._dyn_third_topks[:batch_size].zero_()
        self._dyn_use_paths_a = True
        self._dynamic_active_topk = 0

    def _dynamic_tree_update(
        self,
        level: int,
        topk_ids: torch.Tensor,
        topk_logprobs: torch.Tensor,
        batch_size: int,
    ) -> None:
        if not self._dynamic_tree:
            return
        if not (
            hasattr(torch.ops.vllm, "eagle_update_scores")
            and hasattr(torch.ops.vllm, "eagle_update_path")
            and hasattr(torch.ops.vllm, "eagle_update_draft_tokens_and_scores")
        ):
            return
        self._ensure_dynamic_tree_buffers()
        dyn_k = self._dynamic_topk
        if dyn_k <= 0:
            return
        assert self._dyn_prev_scores is not None
        assert self._dyn_cur_scores is not None
        assert self._dyn_second_topk_ids is not None
        assert self._dyn_candidate_scores is not None
        assert self._dyn_candidate_ids is not None
        assert self._dyn_second_topk_tokens is not None
        assert self._dyn_second_topk_input_ptrs is not None
        assert self._dyn_second_topk_output_ptrs is not None
        assert self._dyn_second_topk_logprobs is not None
        assert self._dyn_next_expand_indices is not None
        assert self._dyn_draft_ids_in is not None
        assert self._dyn_draft_ids_out is not None
        assert self._dyn_draft_lens_in is not None
        assert self._dyn_draft_lens_out is not None
        assert self._dyn_output_scores is not None
        assert self._dyn_paths_a is not None
        assert self._dyn_paths_b is not None
        assert self._dyn_current_expand_indices is not None
        assert self._dyn_all_layers_scores is not None
        assert self._dyn_all_layers_draft_ids is not None
        assert self._dyn_all_layers_predecessor is not None
        assert self._dyn_first_topk_logprobs is not None
        assert self._dyn_first_topk_ids is not None
        assert self._dyn_third_topk_input_ptrs is not None
        assert self._dyn_third_topk_output_ptrs is not None
        assert self._dyn_third_topk_ids is not None
        assert self._dyn_third_topks is not None
        assert self._dyn_paths_final is not None

        use_full_dynamic = (
            self._dynamic_tree_kernels
            and hasattr(torch.ops.vllm, "eagle_copy_scores_and_draft_token_ids")
            and hasattr(torch.ops.vllm, "eagle_assemble_third_topk_inputs")
            and hasattr(torch.ops.vllm, "eagle_reconstruct_final_path")
            and hasattr(torch.ops.vllm, "eagle_copy_final_draft_tokens")
        )

        if topk_ids.dim() == 2:
            topk_ids = topk_ids[:, :dyn_k]
            topk_logprobs = topk_logprobs[:, :dyn_k]
        else:
            topk_ids = topk_ids[:, :dyn_k, :dyn_k]
            topk_logprobs = topk_logprobs[:, :dyn_k, :dyn_k]

        max_draft_tokens = self._dynamic_max_draft_tokens
        num_eagle_layers = self._tree_depth

        if level == 0:
            self._dyn_prev_scores[:batch_size, :dyn_k] = topk_logprobs.to(
                torch.float32
            )
            if use_full_dynamic:
                first_logprobs = self._dyn_first_topk_logprobs[:batch_size]
                first_ids = self._dyn_first_topk_ids[:batch_size]
                first_logprobs.fill_(-float("inf"))
                first_ids.fill_(PADDING_SLOT_ID)
                first_logprobs[:, :dyn_k] = topk_logprobs.to(torch.float32)
                first_ids[:, :dyn_k] = topk_ids.to(dtype=torch.int32)
                torch.ops.vllm.eagle_copy_scores_and_draft_token_ids(
                    level,
                    num_eagle_layers,
                    max_draft_tokens,
                    dyn_k,
                    self._dyn_current_expand_indices[:batch_size],
                    self._dyn_all_layers_scores[:batch_size],
                    self._dyn_all_layers_draft_ids[:batch_size],
                    self._dyn_all_layers_predecessor[:batch_size],
                    self._dyn_all_layers_scores[:batch_size],
                    self._dyn_all_layers_draft_ids[:batch_size],
                    self._dyn_all_layers_predecessor[:batch_size],
                    first_logprobs,
                    first_ids,
                )
            prev_paths = self._dyn_paths_a if self._dyn_use_paths_a else self._dyn_paths_b
            new_paths = self._dyn_paths_b if self._dyn_use_paths_a else self._dyn_paths_a
            torch.ops.vllm.eagle_update_path(
                level,
                dyn_k,
                prev_paths[:batch_size],
                self._dyn_second_topk_ids[:batch_size],
                new_paths[:batch_size],
                self._dyn_next_expand_indices[:batch_size],
            )
            self._dyn_current_expand_indices[:batch_size] = self._dyn_next_expand_indices[
                :batch_size
            ]
            torch.ops.vllm.eagle_update_draft_tokens_and_scores(
                level,
                dyn_k,
                topk_ids.to(dtype=torch.int32),
                self._dyn_draft_ids_in[:batch_size],
                self._dyn_draft_lens_in[:batch_size],
                self._dyn_draft_ids_out[:batch_size],
                self._dyn_draft_lens_out[:batch_size],
                self._dyn_prev_scores[:batch_size],
                self._dyn_output_scores[:batch_size],
            )
            self._dyn_use_paths_a = not self._dyn_use_paths_a
            self._dyn_draft_ids_in[:batch_size] = self._dyn_draft_ids_out[:batch_size]
            self._dyn_draft_lens_in[:batch_size] = self._dyn_draft_lens_out[:batch_size]
            if use_full_dynamic and level == num_eagle_layers - 1:
                self._finalize_dynamic_tree_paths(
                    batch_size,
                    dyn_k,
                    max_draft_tokens,
                    num_eagle_layers,
                )
            return

        if topk_ids.dim() != 3:
            return

        if topk_ids.size(1) != dyn_k or topk_logprobs.size(1) != dyn_k:
            if not self._dynamic_tree_warned:
                logger.warning(
                    "Dynamic tree expects %d parents per level, got %d; "
                    "skipping dynamic update for this level.",
                    dyn_k,
                    topk_ids.size(1),
                )
                self._dynamic_tree_warned = True
            return
        if use_full_dynamic:
            first_logprobs = self._dyn_first_topk_logprobs[: batch_size * dyn_k]
            first_ids = self._dyn_first_topk_ids[: batch_size * dyn_k]
            first_logprobs.fill_(-float("inf"))
            first_ids.fill_(PADDING_SLOT_ID)
            first_logprobs[:, :dyn_k] = topk_logprobs.to(torch.float32).reshape(
                batch_size * dyn_k, dyn_k
            )
            first_ids[:, :dyn_k] = topk_ids.to(dtype=torch.int32).reshape(
                batch_size * dyn_k, dyn_k
            )
            torch.ops.vllm.eagle_copy_scores_and_draft_token_ids(
                level,
                num_eagle_layers,
                max_draft_tokens,
                dyn_k,
                self._dyn_current_expand_indices[:batch_size],
                self._dyn_all_layers_scores[:batch_size],
                self._dyn_all_layers_draft_ids[:batch_size],
                self._dyn_all_layers_predecessor[:batch_size],
                self._dyn_all_layers_scores[:batch_size],
                self._dyn_all_layers_draft_ids[:batch_size],
                self._dyn_all_layers_predecessor[:batch_size],
                first_logprobs,
                first_ids,
            )
        candidate_scores = self._dyn_candidate_scores[
            : batch_size * dyn_k, :max_draft_tokens
        ]
        candidate_scores.fill_(-float("inf"))
        candidate_scores[:, :dyn_k] = topk_logprobs.to(torch.float32).reshape(
            batch_size * dyn_k, dyn_k
        )
        candidate_ids = self._dyn_candidate_ids[
            :batch_size, :dyn_k, :max_draft_tokens
        ]
        candidate_ids.fill_(PADDING_SLOT_ID)
        candidate_ids[:, :, :dyn_k] = topk_ids.to(torch.int32)
        cur_scores = candidate_scores
        torch.ops.vllm.eagle_update_scores(cur_scores, self._dyn_prev_scores[:batch_size], dyn_k)
        flat_scores = cur_scores.view(batch_size, dyn_k * max_draft_tokens)
        if (
            hasattr(torch.ops.vllm, "eagle_topk_small")
            and self._opt_mode != "baseline"
            and dyn_k <= 64
        ):
            second_scores, second_ids = torch.ops.vllm.eagle_topk_small(
                flat_scores, dyn_k
            )
        else:
            second_scores, second_ids = torch.topk(flat_scores, dyn_k, dim=-1)
        self._dyn_second_topk_ids[:batch_size, :dyn_k] = second_ids.to(torch.int32)

        prev_paths = self._dyn_paths_a if self._dyn_use_paths_a else self._dyn_paths_b
        new_paths = self._dyn_paths_b if self._dyn_use_paths_a else self._dyn_paths_a
        torch.ops.vllm.eagle_update_path(
            level,
            dyn_k,
            prev_paths[:batch_size],
            self._dyn_second_topk_ids[:batch_size],
            new_paths[:batch_size],
            self._dyn_next_expand_indices[:batch_size],
        )
        self._dyn_current_expand_indices[:batch_size] = self._dyn_next_expand_indices[
            :batch_size
        ]

        batch_arange = self.arange_int64[:batch_size].unsqueeze(1)
        use_cuda_second = (
            self._dynamic_tree_kernels
            and
            hasattr(torch.ops.vllm, "eagle_assemble_second_topk_inputs")
            and hasattr(torch.ops.vllm, "eagle_extract_scores_and_real_draft_tokens")
        )
        if use_cuda_second:
            self._dyn_second_topk_tokens[:batch_size, :dyn_k] = (
                self._dyn_second_topk_ids[:batch_size, :dyn_k]
            )
            torch.ops.vllm.eagle_assemble_second_topk_inputs(
                flat_scores,
                self._dyn_second_topk_input_ptrs[:batch_size],
                self._dyn_second_topk_tokens[:batch_size],
                self._dyn_second_topk_output_ptrs[:batch_size],
                dyn_k,
            )
            torch.ops.vllm.eagle_extract_scores_and_real_draft_tokens(
                self._dyn_second_topk_input_ptrs[:batch_size],
                self._dyn_second_topk_output_ptrs[:batch_size],
                candidate_ids.view(batch_size, dyn_k * max_draft_tokens),
                self._dyn_second_topk_logprobs[:batch_size],
                dyn_k,
            )
            selected_ids = self._dyn_second_topk_tokens[:batch_size, :dyn_k]
            selected_scores = self._dyn_second_topk_logprobs[:batch_size, :dyn_k]
        else:
            parent_idx = second_ids // max_draft_tokens
            child_idx = second_ids % max_draft_tokens
            selected_ids = candidate_ids[batch_arange, parent_idx, child_idx]
            selected_scores = second_scores
        self._dyn_cur_scores[:batch_size, :dyn_k] = selected_scores.to(torch.float32)
        torch.ops.vllm.eagle_update_draft_tokens_and_scores(
            level,
            dyn_k,
            selected_ids.to(dtype=torch.int32),
            self._dyn_draft_ids_in[:batch_size],
            self._dyn_draft_lens_in[:batch_size],
            self._dyn_draft_ids_out[:batch_size],
            self._dyn_draft_lens_out[:batch_size],
            self._dyn_cur_scores[:batch_size],
            self._dyn_output_scores[:batch_size],
        )
        self._dyn_use_paths_a = not self._dyn_use_paths_a
        self._dyn_draft_ids_in[:batch_size] = self._dyn_draft_ids_out[:batch_size]
        self._dyn_draft_lens_in[:batch_size] = self._dyn_draft_lens_out[:batch_size]
        self._dyn_prev_scores[:batch_size, :dyn_k] = self._dyn_cur_scores[:batch_size]
        if use_full_dynamic and level == num_eagle_layers - 1:
            self._finalize_dynamic_tree_paths(
                batch_size,
                dyn_k,
                max_draft_tokens,
                num_eagle_layers,
            )

    def _finalize_dynamic_tree_paths(
        self,
        batch_size: int,
        dyn_k: int,
        max_draft_tokens: int,
        num_eagle_layers: int,
    ) -> None:
        if batch_size <= 0 or dyn_k <= 0 or max_draft_tokens <= 0:
            return
        assert self._dyn_all_layers_scores is not None
        assert self._dyn_all_layers_predecessor is not None
        assert self._dyn_all_layers_draft_ids is not None
        assert self._dyn_third_topk_input_ptrs is not None
        assert self._dyn_third_topk_output_ptrs is not None
        assert self._dyn_third_topk_ids is not None
        assert self._dyn_third_topks is not None
        assert self._dyn_paths_final is not None
        assert self._dyn_paths_a is not None
        assert self._dyn_draft_ids_in is not None
        assert self._dyn_draft_lens_in is not None

        total_num_tokens = dyn_k
        if num_eagle_layers > 1:
            total_num_tokens += (num_eagle_layers - 1) * dyn_k * dyn_k
        max_nodes_on_final_tree = min(max_draft_tokens, total_num_tokens)
        if max_nodes_on_final_tree <= 0:
            return

        all_scores = self._dyn_all_layers_scores[:batch_size]
        torch.ops.vllm.eagle_assemble_third_topk_inputs(
            all_scores,
            self._dyn_third_topk_input_ptrs[:batch_size],
            self._dyn_third_topk_ids[:batch_size],
            self._dyn_third_topk_output_ptrs[:batch_size],
            self._dyn_third_topks[:batch_size],
            num_eagle_layers,
            max_nodes_on_final_tree,
        )
        flat_scores = all_scores.view(batch_size, -1)
        max_nodes_on_final_tree = min(
            max_nodes_on_final_tree, flat_scores.size(1)
        )
        if max_nodes_on_final_tree <= 0:
            return
        _, topk_ids = torch.topk(
            flat_scores, max_nodes_on_final_tree, dim=-1
        )
        self._dyn_third_topk_ids[:batch_size, :max_nodes_on_final_tree] = (
            topk_ids.to(torch.int32)
        )

        torch.ops.vllm.eagle_reconstruct_final_path(
            self._dyn_third_topk_output_ptrs[:batch_size],
            self._dyn_all_layers_predecessor[:batch_size],
            self._dyn_paths_final[:batch_size],
            dyn_k,
            max_draft_tokens,
            max_draft_tokens + 1,
            self._dynamic_max_path_len,
            num_eagle_layers,
            max_nodes_on_final_tree,
        )
        torch.ops.vllm.eagle_copy_final_draft_tokens(
            self._dyn_third_topk_output_ptrs[:batch_size],
            self._dyn_all_layers_draft_ids[:batch_size],
            self._dyn_draft_ids_in[:batch_size],
            self._dyn_draft_lens_in[:batch_size],
            num_eagle_layers,
            max_draft_tokens,
            max_nodes_on_final_tree,
        )
        self._dyn_paths_a[:batch_size].copy_(self._dyn_paths_final[:batch_size])
        self._dyn_use_paths_a = True

    def _build_dynamic_tree_packed_mask(
        self, paths: torch.Tensor, tree_len: int, exclude_root: bool = False
    ) -> torch.Tensor:
        batch_size = paths.size(0)
        device = paths.device
        if (
            hasattr(torch.ops.vllm, "eagle_build_packed_tree_mask")
            and paths.is_cuda
        ):
            return torch.ops.vllm.eagle_build_packed_tree_mask(
                paths, tree_len, exclude_root
            )
        if exclude_root:
            if tree_len <= 1:
                return torch.empty(
                    (batch_size, 0, 0), dtype=torch.int32, device=device
                )
            paths = paths[:, 1:tree_len, 1:] - 1
            tree_len = tree_len - 1
        max_path_len = paths.size(2)
        mask = torch.zeros(
            (batch_size, tree_len, tree_len), dtype=torch.int32, device=device
        )
        mask[:, :, 0] = 1
        mask[:, 0, 0] = 1
        for depth in range(1, max_path_len):
            nodes = paths[:, :tree_len, depth]
            valid_nodes = (nodes >= 0) & (nodes < tree_len)
            if not valid_nodes.any():
                continue
            ancestors = paths[:, :tree_len, : depth + 1]
            valid_anc = (ancestors >= 0) & (ancestors < tree_len)
            batch_idx = self.arange_int64[:batch_size].view(
                batch_size, 1, 1
            ).expand_as(ancestors)
            node_idx = nodes.unsqueeze(-1).expand_as(ancestors).to(torch.int64)
            valid = valid_nodes.unsqueeze(-1) & valid_anc
            mask[
                batch_idx[valid],
                node_idx[valid],
                ancestors[valid].to(torch.int64),
            ] = 1

        num_blocks = (tree_len + 31) // 32
        packed = torch.zeros(
            (batch_size, tree_len, num_blocks), dtype=torch.int32, device=device
        )
        weights = (1 << torch.arange(32, device=device, dtype=torch.int32)).view(
            1, 1, -1
        )
        for block_idx in range(num_blocks):
            start = block_idx * 32
            end = min(start + 32, tree_len)
            bits = mask[:, :, start:end]
            packed[:, :, block_idx] = (bits * weights[:, :, : end - start]).sum(
                dim=-1
            )
        return packed

    def _cache_dynamic_tree_metadata(
        self,
        req_ids: list[str],
        batch_size: int,
        dyn_tree_len: int,
    ) -> None:
        if not req_ids:
            self._dyn_cached_req_ids = []
            self._dyn_cached_req_id_to_index = {}
            self._dyn_cached_paths = None
            self._dyn_cached_node_token_ids = None
            self._dyn_cached_position_offsets = None
            self._dyn_cached_packed_mask = None
            self._dyn_cached_generation_lengths = None
            self._dyn_cached_tree_len = 0
            return
        assert self._dyn_paths_a is not None
        assert self._dyn_node_token_ids is not None
        assert self._dyn_position_offsets is not None
        paths = self._dyn_paths_a[:batch_size, :dyn_tree_len]
        node_tokens = self._dyn_node_token_ids[:batch_size, :dyn_tree_len]
        draft_len = max(dyn_tree_len - 1, 0)
        if draft_len == 0:
            position_offsets = torch.empty(
                (batch_size, 0), dtype=torch.int32, device=paths.device
            )
            packed_mask = torch.empty(
                (batch_size, 0, 0), dtype=torch.int32, device=paths.device
            )
        else:
            position_offsets = self._dyn_position_offsets[1:dyn_tree_len].expand(
                batch_size, draft_len
            )
            packed_mask = self._build_dynamic_tree_packed_mask(
                paths, dyn_tree_len, exclude_root=True
            )
        generation_lengths = torch.full(
            (batch_size,),
            draft_len,
            dtype=torch.int32,
            device=paths.device,
        )
        self._dyn_cached_req_ids = list(req_ids)
        self._dyn_cached_req_id_to_index = {
            req_id: idx for idx, req_id in enumerate(req_ids)
        }
        self._dyn_cached_paths = paths.clone()
        self._dyn_cached_node_token_ids = node_tokens.clone()
        self._dyn_cached_position_offsets = position_offsets.clone()
        self._dyn_cached_packed_mask = packed_mask
        self._dyn_cached_generation_lengths = generation_lengths
        self._dyn_cached_tree_len = draft_len

    def get_dynamic_tree_metadata(
        self, req_ids: list[str], spec_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            not self._dynamic_tree
            or self._dyn_cached_req_id_to_index is None
            or self._dyn_cached_position_offsets is None
            or self._dyn_cached_packed_mask is None
            or self._dyn_cached_generation_lengths is None
        ):
            return None
        if spec_len <= 0 or spec_len > self._dyn_cached_tree_len:
            return None
        if not req_ids:
            return None
        indices = []
        for req_id in req_ids:
            idx = self._dyn_cached_req_id_to_index.get(req_id, -1)
            indices.append(idx)
        idx_tensor = torch.tensor(
            indices, dtype=torch.int64, device=self._dyn_cached_position_offsets.device
        )
        valid_mask = idx_tensor >= 0
        batch_size = len(req_ids)
        num_blocks = (spec_len + 31) // 32
        position_offsets = torch.zeros(
            (batch_size, spec_len),
            dtype=torch.int32,
            device=self._dyn_cached_position_offsets.device,
        )
        packed_mask = torch.zeros(
            (batch_size, spec_len, num_blocks),
            dtype=torch.int32,
            device=self._dyn_cached_packed_mask.device,
        )
        generation_lengths = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=self._dyn_cached_generation_lengths.device,
        )
        if valid_mask.any():
            position_offsets[valid_mask] = self._dyn_cached_position_offsets[
                idx_tensor[valid_mask], :spec_len
            ]
            packed_mask[valid_mask] = self._dyn_cached_packed_mask[
                idx_tensor[valid_mask], :spec_len, :num_blocks
            ]
            generation_lengths[valid_mask] = self._dyn_cached_generation_lengths[
                idx_tensor[valid_mask]
            ]
        return position_offsets, packed_mask, generation_lengths

    def _get_dynamic_tree_cache(
        self, req_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not self._dynamic_tree
            or self._dyn_cached_req_id_to_index is None
            or self._dyn_cached_paths is None
            or self._dyn_cached_node_token_ids is None
        ):
            return None
        indices = []
        for req_id in req_ids:
            idx = self._dyn_cached_req_id_to_index.get(req_id, -1)
            if idx < 0:
                return None
            indices.append(idx)
        idx_tensor = torch.tensor(
            indices, dtype=torch.int64, device=self._dyn_cached_paths.device
        )
        return (
            self._dyn_cached_paths.index_select(0, idx_tensor),
            self._dyn_cached_node_token_ids.index_select(0, idx_tensor),
        )

    def _build_dynamic_tree_accepted_indices(
        self,
        sampled_token_ids: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.runner is None:
            return None, None
        req_ids = self.runner.input_batch.req_ids
        cache = self._get_dynamic_tree_cache(req_ids)
        if cache is None:
            return None, None
        paths, node_token_ids = cache
        batch_size = paths.size(0)
        if batch_size == 0:
            return None, None
        tree_len = paths.size(1)
        max_path_len = paths.size(2)
        max_depth = min(
            self._tree_depth, sampled_token_ids.size(1), max_path_len - 1
        )
        if max_depth <= 0:
            return None, None

        paths_clamped = paths[:, :tree_len, :max_path_len].clamp(min=0).to(
            torch.int64
        )
        flat_indices = paths_clamped.view(batch_size, -1)
        path_tokens = node_token_ids.gather(1, flat_indices).view(
            batch_size, tree_len, max_path_len
        )
        path_tokens = path_tokens[:, :, 1 : 1 + max_depth]
        valid_nodes = paths[:, :tree_len, 1 : 1 + max_depth] >= 0
        sampled = sampled_token_ids[:, :max_depth].to(path_tokens.dtype)
        matches = (path_tokens == sampled[:, None, :]) & valid_nodes
        prefix = matches.to(torch.int32).cumprod(dim=-1)
        prefix_len = prefix.sum(dim=-1)
        max_accept = (valid_sampled_tokens_count - 1).clamp(
            min=0, max=max_depth
        )
        prefix_len = torch.minimum(prefix_len, max_accept[:, None])
        best_len, best_idx = torch.max(prefix_len, dim=-1)

        accepted_indices = self._tree_accept_indices[:batch_size, :max_depth]
        accepted_indices.fill_(-1)
        selected_paths = paths[torch.arange(batch_size, device=paths.device), best_idx]
        selected_nodes = selected_paths[:, 1 : 1 + max_depth]
        draft_indices = selected_nodes - 1
        mask = self.arange_int64[:max_depth].view(1, -1) < best_len.view(-1, 1)
        accepted_indices[mask] = draft_indices[mask].to(torch.int32)
        counts = self._tree_accept_counts[:batch_size]
        counts.copy_(best_len.to(torch.int32))
        offsets = self._tree_accept_offsets[: batch_size + 1]
        offsets.zero_()
        offsets[1 : batch_size + 1] = torch.cumsum(
            counts, dim=0, dtype=torch.int32
        )
        total = int(offsets[batch_size].item())
        if not hasattr(torch.ops.vllm, "pack_accepted_tokens"):
            return None, None
        packed = torch.ops.vllm.pack_accepted_tokens(
            accepted_indices, offsets, counts, total
        )
        return packed, offsets

    def _init_draft_state(
        self,
        batch_size: int,
        input_batch_size: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        self._draft_tokens[:batch_size, 0] = draft_token_ids
        self.input_ids[:batch_size] = draft_token_ids.to(torch.int32)
        self._set_positions(batch_size, positions)
        self.hidden_states[:batch_size] = hidden_states
        self._draft_seq_lens[:batch_size].copy_(common_attn_metadata.seq_lens)
        block_table = common_attn_metadata.block_table_tensor
        self._draft_block_table[:batch_size, : block_table.shape[1]].copy_(block_table)
        if input_batch_size > batch_size:
            pad_token_id = self.vllm_config.model_config.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0
            self.input_ids[batch_size:input_batch_size] = pad_token_id
            if self.uses_mrope:
                self.mrope_positions[:, batch_size:input_batch_size] = 0
            else:
                self.positions[batch_size:input_batch_size] = 0
            self.hidden_states[batch_size:input_batch_size].zero_()
            self._draft_seq_lens[batch_size:input_batch_size] = 0
            self._draft_slot_mapping[batch_size:input_batch_size] = PADDING_SLOT_ID
            self._draft_block_table[batch_size:input_batch_size].zero_()
        if (
            envs.VLLM_EAGLE_CUDA_DRAFT
            and hasattr(torch.ops.vllm, "eagle_compute_slot_mapping")
        ):
            torch.ops.vllm.eagle_compute_slot_mapping(
                self._get_positions(batch_size),
                self._draft_block_table,
                self._draft_block_table.stride(0),
                self.block_size,
                self.max_model_len,
                self._draft_slot_mapping,
                PADDING_SLOT_ID,
            )
        else:
            _eagle_compute_slot_mapping_kernel[(batch_size,)](
                self._get_positions(batch_size),
                self._draft_block_table,
                self._draft_block_table.stride(0),
                self.block_size,
                self.max_model_len,
                self._draft_slot_mapping,
                PAD_ID=PADDING_SLOT_ID,
                BLOCK_SIZE=1024,
            )

    def _get_draft_common_attn_metadata(
        self,
        input_batch_size: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> CommonAttentionMetadata:
        cached = self._draft_common_attn_metadata_cache.get(input_batch_size)
        query_start_loc = self.arange[: input_batch_size + 1]
        query_start_loc_cpu = self._draft_query_start_loc_cpu[: input_batch_size + 1]
        max_seq_len = min(
            self.max_model_len,
            common_attn_metadata.max_seq_len + self.num_speculative_tokens,
        )
        if cached is None:
            cached = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=self._draft_seq_lens[:input_batch_size],
                num_reqs=input_batch_size,
                num_actual_tokens=input_batch_size,
                max_query_len=1,
                max_seq_len=max_seq_len,
                block_table_tensor=self._draft_block_table[:input_batch_size],
                slot_mapping=self._draft_slot_mapping[:input_batch_size],
                causal=True,
                dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
            )
            self._draft_common_attn_metadata_cache[input_batch_size] = cached
            return cached

        cached.query_start_loc = query_start_loc
        cached.query_start_loc_cpu = query_start_loc_cpu
        cached.seq_lens = self._draft_seq_lens[:input_batch_size]
        cached.num_reqs = input_batch_size
        cached.num_actual_tokens = input_batch_size
        cached.max_query_len = 1
        cached.max_seq_len = max_seq_len
        cached.block_table_tensor = self._draft_block_table[:input_batch_size]
        cached.slot_mapping = self._draft_slot_mapping[:input_batch_size]
        cached.causal = True
        cached.dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        cached._seq_lens_cpu = None
        cached._num_computed_tokens_cpu = None
        return cached

    def _prepare_draft_sampling_buffers(
        self,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
        force_greedy: bool,
    ) -> None:
        if sampling_metadata.temperature is None:
            self._draft_temperature[:batch_size].fill_(1.0)
        else:
            self._draft_temperature[:batch_size].copy_(
                sampling_metadata.temperature[:batch_size]
            )
        if force_greedy or sampling_metadata.all_greedy:
            self._draft_temperature[:batch_size].fill_(0.0)
        if sampling_metadata.top_k is None:
            self._draft_top_k[:batch_size].fill_(self.vocab_size)
        else:
            self._draft_top_k[:batch_size].copy_(
                sampling_metadata.top_k[:batch_size].to(dtype=torch.int32)
            )
        if sampling_metadata.top_p is None:
            self._draft_top_p[:batch_size].fill_(1.0)
        else:
            self._draft_top_p[:batch_size].copy_(
                sampling_metadata.top_p[:batch_size]
            )
        self._draft_sampling_ready = True

    def _sample_draft_tokens_buffered(
        self, logits: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        return torch.ops.vllm.eagle_sample_topk_topp_gumbel(
            logits,
            self._draft_top_k[:batch_size],
            self._draft_top_p[:batch_size],
            self._draft_temperature[:batch_size],
            _SAMPLING_EPS,
        )

    @contextmanager
    def _save_draft_metadata_state(self, input_batch_size: int):
        self._draft_seq_lens_backup[:input_batch_size].copy_(
            self._draft_seq_lens[:input_batch_size]
        )
        self._draft_slot_mapping_backup[:input_batch_size].copy_(
            self._draft_slot_mapping[:input_batch_size]
        )
        try:
            yield
        finally:
            self._draft_seq_lens[:input_batch_size].copy_(
                self._draft_seq_lens_backup[:input_batch_size]
            )
            self._draft_slot_mapping[:input_batch_size].copy_(
                self._draft_slot_mapping_backup[:input_batch_size]
            )

    def _run_draft_loop(
        self,
        batch_size: int,
        input_batch_size: int,
        per_layer_attn_metadata: dict[str, Any],
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
        sampling_metadata: SamplingMetadata,
        force_greedy: bool,
    ) -> None:
        use_buffered_sampling = (
            self._draft_sampling_ready
            and envs.VLLM_EAGLE_CUDA_SAMPLE
            and hasattr(torch.ops.vllm, "eagle_sample_topk_topp_gumbel")
        )
        state_ctx = (
            self._save_draft_metadata_state(input_batch_size)
            if self.use_cuda_graph
            else nullcontext()
        )
        with state_ctx:
            for step in range(1, self.num_speculative_tokens):
                if self.supports_mm_inputs:
                    self.inputs_embeds[:input_batch_size] = self.model.embed_input_ids(
                        self.input_ids[:input_batch_size]
                    )
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                else:
                    input_ids = self.input_ids[:input_batch_size]
                    inputs_embeds = None
                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                ):
                    ret_hidden_states = self.model(
                        input_ids=input_ids,
                        positions=self._get_positions(input_batch_size),
                        hidden_states=self.hidden_states[:input_batch_size],
                        inputs_embeds=inputs_embeds,
                    )
                    if self.method == "mtp":
                        last_hidden_states = ret_hidden_states
                        hidden_states = ret_hidden_states
                    else:
                        last_hidden_states, hidden_states = ret_hidden_states
                logits = self.model.compute_logits(last_hidden_states[:batch_size])
                if use_buffered_sampling:
                    draft_token_ids = self._sample_draft_tokens_buffered(
                        logits, batch_size
                    )
                else:
                    draft_token_ids = self._sample_draft_tokens(
                        logits, sampling_metadata, force_greedy
                    )
                if draft_token_ids.dtype != torch.int32:
                    draft_token_ids_i32 = draft_token_ids.to(torch.int32)
                else:
                    draft_token_ids_i32 = draft_token_ids
                stored_tokens = False
                if step < self.num_speculative_tokens - 1:
                    if (
                        envs.VLLM_EAGLE_CUDA_DRAFT
                        and hasattr(torch.ops.vllm, "eagle_update_draft_state")
                    ):
                        if hasattr(
                            torch.ops.vllm, "eagle_update_draft_state_and_tokens"
                        ):
                            torch.ops.vllm.eagle_update_draft_state_and_tokens(
                                draft_token_ids_i32,
                                hidden_states,
                                hidden_states.stride(0),
                                self.input_ids,
                                self._get_positions(batch_size),
                                self._positions_stride0(),
                                self.hidden_states,
                                self.hidden_states.stride(0),
                                self._draft_seq_lens,
                                self._draft_slot_mapping,
                                self._draft_block_table,
                                self._draft_block_table.stride(0),
                                self._draft_tokens,
                                self._draft_tokens.stride(0),
                                step,
                                self.hidden_size,
                                self.block_size,
                                self.max_model_len,
                                PADDING_SLOT_ID,
                                self.uses_mrope,
                            )
                            stored_tokens = True
                        else:
                            torch.ops.vllm.eagle_update_draft_state(
                                draft_token_ids_i32,
                                hidden_states,
                                hidden_states.stride(0),
                                self.input_ids,
                                self._get_positions(batch_size),
                                self._positions_stride0(),
                                self.hidden_states,
                                self.hidden_states.stride(0),
                                self._draft_seq_lens,
                                self._draft_slot_mapping,
                                self._draft_block_table,
                                self._draft_block_table.stride(0),
                                self.hidden_size,
                                self.block_size,
                                self.max_model_len,
                                PADDING_SLOT_ID,
                                self.uses_mrope,
                            )
                    else:
                        _eagle_update_draft_state_kernel[(batch_size,)](
                            draft_token_ids_i32,
                            hidden_states,
                            hidden_states.stride(0),
                            self.input_ids,
                            self._get_positions(batch_size),
                            self._positions_stride0(),
                            self.hidden_states,
                            self.hidden_states.stride(0),
                            self._draft_seq_lens,
                            self._draft_slot_mapping,
                            self._draft_block_table,
                            self._draft_block_table.stride(0),
                            self.hidden_size,
                            self.block_size,
                            self.max_model_len,
                            PAD_ID=PADDING_SLOT_ID,
                            USE_MROPE=self.uses_mrope,
                            BLOCK_SIZE=1024,
                        )
                if not stored_tokens:
                    self._draft_tokens[:batch_size, step] = draft_token_ids

    def _positions_stride0(self) -> int:
        if self.uses_mrope:
            return self.mrope_positions.stride(0)
        return 0

    def _sample_draft_tokens(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        force_greedy: bool,
        return_probs: bool = False,
    ) -> torch.Tensor:
        if force_greedy or sampling_metadata.all_greedy:
            if (
                envs.VLLM_EAGLE_CUDA_SAMPLE
                and hasattr(torch.ops.vllm, "eagle_sample_argmax")
                and logits.is_cuda
            ):
                return torch.ops.vllm.eagle_sample_argmax(logits)
            return logits.argmax(dim=-1)
        if (
            envs.VLLM_EAGLE_CUDA_SAMPLE
            and hasattr(torch.ops.vllm, "eagle_sample_topk_topp_gumbel")
            and logits.is_cuda
        ):
            return torch.ops.vllm.eagle_sample_topk_topp_gumbel(
                logits,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
                sampling_metadata.temperature,
                _SAMPLING_EPS,
            )
        if self._fast_logits or not return_probs:
            if (
                sampling_metadata.top_k is not None
                or sampling_metadata.top_p is not None
            ):
                logits = apply_top_k_top_p(
                    logits, sampling_metadata.top_k, sampling_metadata.top_p
                )
            temperature = sampling_metadata.temperature
            if temperature is not None:
                logits = logits / temperature.view(-1, 1)
            q = torch.empty_like(logits)
            q.exponential_()
            gumbel = -q.log_()
            return (logits + gumbel).argmax(dim=-1)
        next_token_ids, _ = compute_probs_and_sample_next_token(
            logits, sampling_metadata
        )
        return next_token_ids

    def _record_draft_latency(self, start_time: float | None) -> None:
        if not self._trace_enabled or start_time is None:
            return
        if self._trace_sync:
            torch.cuda.synchronize()
        duration = time.perf_counter() - start_time
        self._trace_draft_s += duration
        self._trace_draft_calls += 1
        now = time.monotonic()
        if self._trace_interval_s > 0 and (
            now - self._trace_last_log
        ) < self._trace_interval_s:
            return
        avg_ms = (
            (self._trace_draft_s / self._trace_draft_calls) * 1e3
            if self._trace_draft_calls
            else 0.0
        )
        logger.info(
            "EAGLE draft latency: avg=%.3f ms over %d calls",
            avg_ms,
            self._trace_draft_calls,
        )
        self._trace_last_log = now
        self._trace_draft_s = 0.0
        self._trace_draft_calls = 0

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_ids: list[str] | None = None,
    ) -> torch.Tensor:
        trace_start = None
        if self._trace_enabled:
            if self._trace_sync:
                torch.cuda.synchronize()
            trace_start = time.perf_counter()

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )
        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens,
            num_tokens_padded=num_tokens,
        )

        cudagraph_runtime_mode = CUDAGraphMode.NONE
        if (
            self.use_cuda_graph
            and num_tokens_dp_padded
            <= self.compilation_config.max_cudagraph_capture_size
        ):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens_dp_padded)
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self._get_positions(num_input_tokens),
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            self._record_draft_latency(trace_start)
            return draft_token_ids.view(-1, 1)

        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]
        if self.method in (
            "deepseek_mtp",
            "ernie_mtp",
            "longcat_flash_mtp",
            "pangu_ultra_moe_mtp",
        ):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # Draft using tree attention.
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            draft_token_ids = torch.cat(draft_token_ids_list, dim=1)
            if self._dynamic_tree and req_ids is not None:
                dyn_k = self._dynamic_active_topk or self._dynamic_topk
                dyn_tree_len = 1 + dyn_k * self._tree_depth
                self._cache_dynamic_tree_metadata(
                    req_ids=req_ids,
                    batch_size=batch_size,
                    dyn_tree_len=dyn_tree_len,
                )
            # [batch_size, num_tree_tokens]
            self._record_draft_latency(trace_start)
            return draft_token_ids

        force_greedy = False
        self._prepare_draft_sampling_buffers(
            sampling_metadata, batch_size, force_greedy
        )
        if (
            envs.VLLM_EAGLE_CUDA_SAMPLE
            and hasattr(torch.ops.vllm, "eagle_sample_topk_topp_gumbel")
            and logits.is_cuda
        ):
            draft_token_ids = self._sample_draft_tokens_buffered(
                logits, batch_size
            )
        else:
            draft_token_ids = self._sample_draft_tokens(
                logits, sampling_metadata, force_greedy=force_greedy
            )

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=batch_size,
            num_tokens_padded=batch_size,
        )

        disable_padded = self.speculative_config.disable_padded_drafter_batch
        use_cudagraph = (
            self.use_cuda_graph
            and not disable_padded
            and (sampling_metadata.all_greedy or self._allow_random_cudagraph)
        )

        draft_cudagraph_size = None
        if use_cudagraph:
            draft_cudagraph_size = self._get_draft_cudagraph_size(batch_size_dp_padded)
        if draft_cudagraph_size is not None:
            input_batch_size = draft_cudagraph_size
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            input_batch_size = batch_size_dp_padded
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        if batch_size_across_dp is not None:
            batch_size_across_dp[self.dp_rank] = input_batch_size

        self._init_draft_state(
            batch_size=batch_size,
            input_batch_size=input_batch_size,
            positions=positions,
            hidden_states=hidden_states,
            draft_token_ids=draft_token_ids,
            common_attn_metadata=common_attn_metadata,
        )

        draft_common_attn_metadata = self._get_draft_common_attn_metadata(
            input_batch_size, common_attn_metadata
        )

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=draft_common_attn_metadata, draft_index=1
        )
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        use_captured_graph = use_cudagraph and draft_cudagraph_size == input_batch_size
        if use_captured_graph:
            if input_batch_size not in self._draft_graphs:
                # Warm up.
                self._draft_loop(
                    batch_size,
                    input_batch_size,
                    per_layer_attn_metadata,
                    batch_size_across_dp,
                    CUDAGraphMode.NONE,
                    sampling_metadata,
                    force_greedy=False,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self._draft_graph_pool):
                    self._draft_loop(
                        batch_size,
                        input_batch_size,
                        per_layer_attn_metadata,
                        batch_size_across_dp,
                        CUDAGraphMode.NONE,
                        sampling_metadata,
                        force_greedy=False,
                    )
                self._draft_graphs[input_batch_size] = graph
            self._draft_graphs[input_batch_size].replay()
        else:
            self._draft_loop(
                batch_size,
                input_batch_size,
                per_layer_attn_metadata,
                batch_size_across_dp,
                cudagraph_runtime_mode,
                sampling_metadata,
                force_greedy=False,
            )

        self._record_draft_latency(trace_start)
        return self._draft_tokens[:batch_size]

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute backup tokens for requests that are not sampled
        num_reqs = gpu_input_batch.num_reqs
        backup_tokens_gpu: torch.Tensor
        if gpu_input_batch.prompt_token_ids_gpu is not None:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            seq_lens = seq_lens.to(dtype=torch.int64)
            max_index = self.max_model_len - 1
            if max_index >= 0:
                seq_lens = torch.clamp(seq_lens, max=max_index)
            backup_tokens_gpu = gpu_input_batch.prompt_token_ids_gpu[
                self.arange_int64[:num_reqs], seq_lens
            ]
        else:
            self.backup_next_token_ids.np[:num_reqs] = np.array(
                [
                    requests[gpu_input_batch.req_ids[i]].get_token_id(
                        common_attn_metadata.seq_lens_cpu[i].item()
                    )
                    for i in range(num_reqs)
                ],
                dtype=np.int32,
            )
            self.backup_next_token_ids.copy_to_gpu(num_reqs)
            backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty((batch_size,), dtype=torch.int32, device=device)
        valid_sampled_tokens_count = torch.empty(
            (batch_size,), dtype=torch.int32, device=device
        )

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def build_tree_accepted_indices(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
        sampled_token_ids: torch.Tensor | None,
        valid_sampled_tokens_count: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._dynamic_tree:
            if (
                not envs.VLLM_EAGLE_CUDA_KV_COMPACT
                or spec_decode_metadata is None
                or sampled_token_ids is None
                or valid_sampled_tokens_count is None
            ):
                return None, None
            return self._build_dynamic_tree_accepted_indices(
                sampled_token_ids, valid_sampled_tokens_count
            )
        if (
            not envs.VLLM_EAGLE_CUDA_KV_COMPACT
            or self._tree_is_linear
            or self._tree_depth == 0
        ):
            return None, None
        if (
            spec_decode_metadata is None
            or sampled_token_ids is None
            or valid_sampled_tokens_count is None
        ):
            return None, None
        if not hasattr(torch.ops.vllm, "eagle_build_tree_accepted_indices"):
            return None, None
        if spec_decode_metadata.max_spec_len > self._tree_total_drafts:
            return None, None
        batch_size = sampled_token_ids.shape[0]
        if batch_size == 0:
            return None, None
        expected = batch_size * self._tree_total_drafts
        if spec_decode_metadata.draft_token_ids.numel() < expected:
            if hasattr(torch.ops.vllm, "eagle_expand_draft_tokens"):
                draft_tokens = torch.ops.vllm.eagle_expand_draft_tokens(
                    spec_decode_metadata.draft_token_ids,
                    spec_decode_metadata.cu_num_draft_tokens,
                    self._tree_total_drafts,
                    PADDING_SLOT_ID,
                )
            else:
                draft_tokens = torch.full(
                    (batch_size, self._tree_total_drafts),
                    PADDING_SLOT_ID,
                    dtype=torch.int32,
                    device=sampled_token_ids.device,
                )
                offset = 0
                for req_idx, count in enumerate(
                    spec_decode_metadata.num_draft_tokens[:batch_size]
                ):
                    if count <= 0:
                        continue
                    end = offset + count
                    draft_tokens[req_idx, :count] = spec_decode_metadata.draft_token_ids[
                        offset:end
                    ]
                    offset = end
        else:
            draft_tokens = spec_decode_metadata.draft_token_ids[:expected].reshape(
                batch_size, self._tree_total_drafts
            )
        accepted_indices = self._tree_accept_indices[
            :batch_size, : self._tree_depth
        ]
        accepted_indices.fill_(-1)
        counts = self._tree_accept_counts[:batch_size]
        torch.ops.vllm.eagle_build_tree_accepted_indices(
            draft_tokens,
            sampled_token_ids,
            valid_sampled_tokens_count,
            self._tree_level_offsets_tensor,
            self._tree_level_sizes_tensor,
            self._tree_children_offsets_flat,
            self._tree_children_offsets_start,
            accepted_indices,
            counts,
        )
        offsets = self._tree_accept_offsets[: batch_size + 1]
        offsets.zero_()
        offsets[1 : batch_size + 1] = torch.cumsum(
            counts, dim=0, dtype=torch.int32
        )
        total = int(offsets[batch_size].item())
        if not hasattr(torch.ops.vllm, "pack_accepted_tokens"):
            return None, None
        packed = torch.ops.vllm.pack_accepted_tokens(
            accepted_indices, offsets, counts, total
        )
        return packed, offsets

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
        skip_rewind: bool = False,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        # Kernel grid: one program per request (row)
        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_reqs,
        )
        if not skip_rewind:
            if (
                envs.VLLM_EAGLE_CUDA_REWIND
                and hasattr(torch.ops.vllm, "eagle_rewind_slot_mapping")
            ):
                torch.ops.vllm.eagle_rewind_slot_mapping(
                    spec_decode_metadata.cu_num_draft_tokens,
                    valid_sampled_tokens_count,
                    common_attn_metadata.query_start_loc,
                    common_attn_metadata.slot_mapping,
                    PADDING_SLOT_ID,
                )
            else:
                rewind_block_size = min(
                    1024, triton.next_power_of_2(self.num_speculative_tokens + 1)
                )
                _eagle_rewind_slot_mapping_kernel[grid](
                    spec_decode_metadata.cu_num_draft_tokens,
                    valid_sampled_tokens_count,
                    common_attn_metadata.query_start_loc,
                    common_attn_metadata.slot_mapping,
                    num_reqs,
                    PAD_ID=PADDING_SLOT_ID,
                    BLOCK_SIZE=rewind_block_size,
                )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices_to_sample

    def propose_tree(
        self,
        batch_size: int,
        # [num_tokens, vocab_size]
        logits: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        # [num_tokens, hidden_size]
        hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[torch.Tensor]:
        attn_metadata_builder = self._get_attention_metadata_builder()
        use_dynamic_masks = self._dynamic_tree and isinstance(
            attn_metadata_builder, TRTLLMAttentionMetadataBuilder
        )
        tree_attn_metadata_builder = None
        if not use_dynamic_masks:
            tree_attn_metadata_builder = self.runner.attn_groups[0][
                0
            ].get_metadata_builder()
            assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)
        if self._dynamic_tree:
            self._ensure_dynamic_tree_buffers()
            self._reset_dynamic_tree_state(batch_size)

        use_tree_cuda_draft = (
            self._use_tree_cuda_draft
            and not use_dynamic_drafts
            and not self._dynamic_tree
            and self._tree_draft_tokens_buffer is not None
            and self._tree_new_draft_tokens is not None
            and hasattr(torch.ops.vllm, "eagle_extract_real_draft_tokens")
            and self._tree_max_top_k > 0
        )

        # Sample a draft token for each child at the tree root level.
        root_num_children = self._tree_root_num_children
        root_topk_logprobs = None
        if root_num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            root_topk_logprobs = torch.log_softmax(logits, dim=-1).gather(
                1, draft_token_ids
            )
        else:
            if (
                self._opt_mode != "baseline"
                and hasattr(torch.ops.vllm, "eagle_topk_logits_custom")
                and root_num_children <= 64
            ):
                draft_token_ids, root_topk_logprobs = (
                    torch.ops.vllm.eagle_topk_logits_custom(
                        logits, root_num_children
                    )
                )
                draft_token_ids = draft_token_ids.view(batch_size, -1)
                root_topk_logprobs = root_topk_logprobs.view(batch_size, -1)
            elif (
                self._opt_mode != "baseline"
                and hasattr(torch.ops.vllm, "eagle_topk_logits")
            ):
                draft_token_ids, root_topk_logprobs = (
                    torch.ops.vllm.eagle_topk_logits(logits, root_num_children)
                )
                draft_token_ids = draft_token_ids.view(batch_size, -1)
                root_topk_logprobs = root_topk_logprobs.view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(
                    logits, root_num_children, dim=-1
                ).indices.view(batch_size, -1)
        if use_tree_cuda_draft:
            assert self._tree_draft_tokens_buffer is not None
            assert self._tree_new_draft_tokens is not None
            self._tree_draft_tokens_buffer[:batch_size].fill_(PADDING_SLOT_ID)
            self._tree_new_draft_tokens[:batch_size].fill_(PADDING_SLOT_ID)
            root_topk_i32 = draft_token_ids.to(dtype=torch.int32)
            self._tree_new_draft_tokens[:batch_size, 0, :root_num_children] = (
                root_topk_i32
            )
            torch.ops.vllm.eagle_extract_real_draft_tokens(
                0,
                self._tree_depth,
                self._tree_total_drafts,
                self._tree_max_top_k,
                int(self._tree_tokens_gather_idx[0].numel()),
                self._tree_tokens_gather_idx[0],
                self._tree_top_k_list[0],
                self._tree_draft_tokens_indices_cumsum,
                self._tree_new_draft_tokens[:batch_size],
                self._tree_draft_tokens_buffer[:batch_size],
            )
            start = int(self._tree_draft_tokens_indices_cumsum[0].item())
            end = int(self._tree_draft_tokens_indices_cumsum[1].item())
            draft_token_ids = self._tree_draft_tokens_buffer[
                :batch_size, start:end
            ]
        use_dynamic_drafts = use_dynamic_masks
        use_opt_dynamic = use_dynamic_drafts and self._opt_mode != "baseline"
        dyn_k = 0
        dyn_paths = None
        dyn_node_token_ids = None
        parent_static_to_dynamic = None
        static_scores_prev = None
        dyn_parent_scores = None
        dyn_parent_indices = None
        if self._dynamic_tree:
            dyn_k = self._dynamic_topk
            if dyn_k <= 0:
                raise ValueError(
                    "VLLM_EAGLE_DYNAMIC_TOPK must be > 0 when dynamic tree is enabled"
                )
            if root_num_children < dyn_k:
                raise ValueError(
                    "Dynamic tree top-k exceeds root children: "
                    f"{dyn_k} > {root_num_children}"
                )
            self._dynamic_active_topk = dyn_k
            assert self._dyn_paths_a is not None
            assert self._dyn_node_token_ids is not None
            dyn_paths = self._dyn_paths_a[:batch_size, : self._dynamic_max_tokens]
            dyn_node_token_ids = self._dyn_node_token_ids[
                :batch_size, : self._dynamic_max_tokens
            ]
            dyn_paths.fill_(-1)
            dyn_node_token_ids.fill_(PADDING_SLOT_ID)
            dyn_paths[:, 0, 0] = 0

            if root_topk_logprobs is None:
                root_logprobs = torch.log_softmax(logits, dim=-1)
                root_topk_logprobs = root_logprobs.gather(1, draft_token_ids)
            if use_opt_dynamic:
                self._dynamic_tree_update(
                    0, draft_token_ids, root_topk_logprobs, batch_size
                )
                if use_dynamic_drafts:
                    draft_token_ids = self._dyn_draft_ids_in[
                        :batch_size, :dyn_k
                    ]
                    dyn_parent_indices = torch.zeros(
                        (batch_size, dyn_k),
                        dtype=torch.int64,
                        device=draft_token_ids.device,
                    )
                dyn_paths = (
                    self._dyn_paths_a if self._dyn_use_paths_a else self._dyn_paths_b
                )
            else:
                if dyn_k == root_num_children:
                    selected_static_idx = self.arange_int64[:root_num_children].view(
                        1, -1
                    ).expand(batch_size, -1)
                    selected_scores = root_topk_logprobs
                    selected_tokens = draft_token_ids
                else:
                    if (
                        hasattr(torch.ops.vllm, "eagle_topk_small")
                        and self._opt_mode != "baseline"
                        and dyn_k <= 64
                    ):
                        scores32 = root_topk_logprobs.to(torch.float32)
                        selected_scores, selected_static_idx = (
                            torch.ops.vllm.eagle_topk_small(scores32, dyn_k)
                        )
                        selected_static_idx = selected_static_idx.to(torch.int64)
                    else:
                        selected_scores, selected_static_idx = torch.topk(
                            root_topk_logprobs, dyn_k, dim=-1
                        )
                    selected_tokens = draft_token_ids.gather(1, selected_static_idx)

                dyn_idx = (
                    self.arange_int64[:dyn_k].to(dtype=torch.int32) + 1
                ).view(1, -1).expand(batch_size, -1)
                dyn_idx_i64 = dyn_idx.to(torch.int64)
                batch_idx = self.arange_int64[:batch_size].view(batch_size, 1).expand(
                    batch_size, dyn_k
                )
                dyn_node_token_ids[batch_idx, dyn_idx_i64] = selected_tokens.to(
                    torch.int32
                )
                dyn_paths[batch_idx, dyn_idx_i64, 0] = 0
                dyn_paths[batch_idx, dyn_idx_i64, 1] = dyn_idx

                dyn_parent_scores = selected_scores
                if use_dynamic_drafts:
                    draft_token_ids = selected_tokens
                    dyn_parent_indices = torch.zeros(
                        (batch_size, dyn_k),
                        dtype=torch.int64,
                        device=selected_tokens.device,
                    )
                else:
                    parent_static_to_dynamic = torch.full(
                        (batch_size, root_num_children),
                        -1,
                        dtype=torch.int32,
                        device=selected_static_idx.device,
                    )
                    parent_static_to_dynamic[batch_idx, selected_static_idx] = dyn_idx

                    static_scores_prev = torch.full(
                        (batch_size, root_num_children),
                        -float("inf"),
                        dtype=torch.float32,
                        device=selected_scores.device,
                    )
                    static_scores_prev[batch_idx, selected_static_idx] = selected_scores
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

        tree_total_drafts = self._tree_total_drafts
        if use_dynamic_drafts:
            tree_total_drafts = dyn_k * self._tree_depth
        use_tree_buffers = (
            not self.uses_mrope
            and batch_size * tree_total_drafts <= self.max_num_tokens
        )
        if use_tree_buffers:
            tree_input_ids = self.input_ids[
                : batch_size * tree_total_drafts
            ].view(batch_size, tree_total_drafts)
            tree_positions = self.positions[
                : batch_size * tree_total_drafts
            ].view(batch_size, tree_total_drafts)
            tree_hidden_states = self.hidden_states[
                : batch_size * tree_total_drafts
            ].view(batch_size, tree_total_drafts, -1)
        else:
            # Initialize empty tensors for concatenation with the level outputs.
            tree_input_ids = torch.empty(
                0, device=self.input_ids.device, dtype=self.input_ids.dtype
            )
            tree_positions = torch.empty(
                0, device=self.positions.device, dtype=self.positions.dtype
            )
            tree_hidden_states = torch.empty(
                0, device=self.hidden_states.device, dtype=self.hidden_states.dtype
            )
        positions_view = positions.view(batch_size, -1)
        if use_dynamic_drafts:
            assert self._dyn_tree_draft_pos_offsets is not None
            flattened_draft_positions = (
                positions_view
                + self._dyn_tree_draft_pos_offsets[:batch_size, :tree_total_drafts]
            )
        else:
            flattened_draft_positions = (
                positions_view + self.tree_draft_pos_offsets[:batch_size, :]
            )
        tree_depth = len(self.num_drafts_per_level)
        for level in range(tree_depth - 1):
            if use_dynamic_drafts:
                level_num_drafts = dyn_k
                total_num_drafts = dyn_k * (level + 1)
            else:
                level_num_drafts = self.num_drafts_per_level[level]
                total_num_drafts = self.cu_drafts_per_level[level]
            if self._tree_max_topk_per_level[level] == 0:
                break
            # Get draft positions for RoPE.
            draft_positions = positions_view + (level + 1)
            exceeds_max_model_len = (
                positions_view[:, 0] + total_num_drafts
            ) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len.view(batch_size, 1),
                0,
                draft_positions,
            )

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.expand(batch_size, level_num_drafts)

            if use_dynamic_drafts:
                if level_num_drafts > 1:
                    assert dyn_parent_indices is not None
                    parent_idx = dyn_parent_indices
                    parent_idx_expanded = parent_idx.unsqueeze(-1).expand(
                        batch_size, level_num_drafts, self.hidden_size
                    )
                    draft_hidden_states = draft_hidden_states.gather(
                        1, parent_idx_expanded
                    )
            else:
                # Repeat draft hidden states for each child.
                parent_idx = self._tree_parent_indices[level]
                if parent_idx is not None:
                    if (
                        envs.VLLM_EAGLE_CUDA_TREE_COPY
                        and hasattr(torch.ops.vllm, "eagle_tree_gather_hidden_states")
                    ):
                        gathered_hidden_states = torch.empty(
                            (batch_size, parent_idx.numel(), self.hidden_size),
                            device=draft_hidden_states.device,
                            dtype=draft_hidden_states.dtype,
                        )
                        torch.ops.vllm.eagle_tree_gather_hidden_states(
                            draft_hidden_states,
                            parent_idx,
                            gathered_hidden_states,
                        )
                        draft_hidden_states = gathered_hidden_states
                    else:
                        draft_hidden_states = draft_hidden_states.index_select(
                            1, parent_idx
                        )

            if use_tree_buffers:
                if use_dynamic_drafts:
                    start = level * dyn_k
                else:
                    start = self._tree_level_offsets[level]
                end = start + level_num_drafts
                if (
                    envs.VLLM_EAGLE_CUDA_TREE_COPY
                    and hasattr(torch.ops.vllm, "eagle_tree_copy_level")
                ):
                    torch.ops.vllm.eagle_tree_copy_level(
                        draft_token_ids,
                        draft_positions,
                        draft_hidden_states,
                        tree_input_ids,
                        tree_positions,
                        tree_hidden_states,
                        start,
                        self.hidden_size,
                    )
                else:
                    tree_input_ids[:, start:end] = draft_token_ids
                    tree_positions[:, start:end] = draft_positions
                    tree_hidden_states[:, start:end] = draft_hidden_states
            else:
                # Concatenate the draft tokens, positions, and hidden states.
                tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
                tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
                tree_hidden_states = torch.cat(
                    [tree_hidden_states, draft_hidden_states], dim=1
                )

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            if use_dynamic_drafts:
                assert dyn_paths is not None
                assert self._dyn_position_offsets is not None
                query_start_loc = (
                    self.arange[: batch_size + 1] * query_len
                ).to(torch.int32)
                query_start_loc_cpu = self._spec_query_start_loc_cpu[
                    : batch_size + 1
                ]
                query_start_loc_cpu.copy_(
                    torch.arange(batch_size + 1, dtype=torch.int32) * query_len
                )
                dyn_tree_len = 1 + query_len
                position_offsets = self._dyn_position_offsets[
                    1:dyn_tree_len
                ].expand(batch_size, query_len)
                packed_mask = self._build_dynamic_tree_packed_mask(
                    dyn_paths[:batch_size, :dyn_tree_len],
                    dyn_tree_len,
                    exclude_root=True,
                )
                common_attn_metadata = replace(
                    common_attn_metadata,
                    query_start_loc=query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                    num_actual_tokens=batch_size * query_len,
                    max_query_len=query_len,
                    spec_decoding_position_offsets=position_offsets,
                    spec_decoding_packed_mask=packed_mask,
                    spec_decoding_is_tree=True,
                )
                attn_metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    fast_build=True,
                )
            else:
                common_attn_metadata = replace(
                    common_attn_metadata,
                    query_start_loc=self._tree_query_start_loc[level][
                        : batch_size + 1
                    ],
                    seq_lens=common_attn_metadata.seq_lens + level_num_drafts,
                    num_actual_tokens=batch_size * query_len,
                    max_query_len=query_len,
                )
                attn_metadata = tree_attn_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=level + 1,
                )

            # Apply new attention metadata to all layers.
            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # Consider max model length.
            attn_metadata.max_seq_len = min(
                attn_metadata.max_seq_len, self.max_model_len
            )
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            if use_tree_buffers:
                input_ids = tree_input_ids[:, :query_len].reshape(-1)
                positions_flat = tree_positions[:, :query_len].reshape(-1)
                hidden_in = tree_hidden_states[:, :query_len].reshape(num_tokens, -1)
            else:
                input_ids = tree_input_ids.view(-1)
                positions_flat = tree_positions.view(-1)
                hidden_in = tree_hidden_states.view(num_tokens, -1)

            # Compute the slot mapping.
            query_positions = flattened_draft_positions[:, level : level + query_len]
            query_positions_flat = query_positions.reshape(-1)
            slot_mapping = self._tree_slot_mapping[:num_tokens]
            BLOCK_SIZE_TOKENS = 256
            grid = (triton.cdiv(num_tokens, BLOCK_SIZE_TOKENS),)
            _eagle_tree_slot_mapping_kernel[grid](
                query_positions_flat,
                attn_metadata.block_table,
                attn_metadata.block_table.stride(0),
                self.block_size,
                self.max_model_len,
                query_len,
                num_tokens,
                slot_mapping,
                PAD_ID=PADDING_SLOT_ID,
                BLOCK_SIZE=BLOCK_SIZE_TOKENS,
            )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping = slot_mapping.view(batch_size, query_len)
            slot_mapping.masked_fill_(exceeds_max_model_len.view(-1, 1), PADDING_SLOT_ID)
            attn_metadata.slot_mapping = slot_mapping.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = positions_flat
            self.hidden_states[:num_tokens] = hidden_in

            if (
                self.use_cuda_graph
                and num_tokens <= self.compilation_config.max_cudagraph_capture_size
            ):
                num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
                cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            else:
                num_input_tokens = num_tokens
                cudagraph_runtime_mode = CUDAGraphMode.NONE
            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            ):
                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=None,
                )

            # Get the output hidden states for the draft tokens.
            draft_hidden_states = hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]
            draft_last_hidden_states = last_hidden_states[:num_tokens].view(
                batch_size, query_len, -1
            )[:, -level_num_drafts:]

            # Get the output logits for the draft tokens.
            logits = self.model.compute_logits(
                draft_last_hidden_states.reshape(batch_size * level_num_drafts, -1)
            )

            # Sample a draft token for each child at the next tree level.
            next_parent_idx = self._tree_parent_indices[level + 1]
            next_child_idx = self._tree_child_indices[level + 1]
            if next_parent_idx is None or next_child_idx is None:
                break
            max_topk = self._tree_max_topk_per_level[level + 1]
            topk_logprobs = None
            if self._opt_mode != "baseline" and (use_dynamic_drafts or self._dynamic_tree):
                if (
                    hasattr(torch.ops.vllm, "eagle_topk_logits_custom")
                    and max_topk <= 64
                ):
                    topk_ids, topk_logprobs = (
                        torch.ops.vllm.eagle_topk_logits_custom(logits, max_topk)
                    )
                    topk_logprobs = topk_logprobs.view(
                        batch_size, level_num_drafts, max_topk
                    )
                elif hasattr(torch.ops.vllm, "eagle_topk_logits"):
                    topk_ids, topk_logprobs = torch.ops.vllm.eagle_topk_logits(
                        logits, max_topk
                    )
                    topk_logprobs = topk_logprobs.view(
                        batch_size, level_num_drafts, max_topk
                    )
                else:
                    topk_ids = torch.topk(logits, max_topk, dim=-1).indices
                    topk_logprobs = torch.log_softmax(logits, dim=-1)
                    topk_logprobs = topk_logprobs.gather(1, topk_ids)
                    topk_logprobs = topk_logprobs.view(
                        batch_size, level_num_drafts, max_topk
                    )
            else:
                topk_ids = torch.topk(logits, max_topk, dim=-1).indices
                if use_dynamic_drafts or self._dynamic_tree:
                    topk_logprobs = torch.log_softmax(logits, dim=-1)
                    topk_logprobs = topk_logprobs.gather(1, topk_ids)
                    topk_logprobs = topk_logprobs.view(
                        batch_size, level_num_drafts, max_topk
                    )
            topk_ids = topk_ids.view(batch_size, level_num_drafts, max_topk)
            if use_dynamic_drafts:
                if use_opt_dynamic:
                    assert topk_logprobs is not None
                    self._dynamic_tree_update(
                        level + 1, topk_ids, topk_logprobs, batch_size
                    )
                    dyn_paths = (
                        self._dyn_paths_a
                        if self._dyn_use_paths_a
                        else self._dyn_paths_b
                    )
                    second_ids = self._dyn_second_topk_ids[:batch_size, :dyn_k]
                    dyn_parent_indices = (second_ids // dyn_k).to(torch.int64)
                    start = level * dyn_k
                    end = start + dyn_k
                    draft_token_ids = self._dyn_draft_ids_in[:batch_size, start:end]
                    draft_token_ids_list.append(draft_token_ids)
                else:
                    assert dyn_parent_scores is not None
                    assert topk_logprobs is not None
                    node_scores = topk_logprobs + dyn_parent_scores.view(
                        batch_size, level_num_drafts, 1
                    )
                    if exceeds_max_model_len.any():
                        node_scores = torch.where(
                            exceeds_max_model_len.view(batch_size, 1, 1),
                            -float("inf"),
                            node_scores,
                        )
                    flat_scores = node_scores.view(
                        batch_size, level_num_drafts * max_topk
                    )
                    if (
                        hasattr(torch.ops.vllm, "eagle_topk_small")
                        and self._opt_mode != "baseline"
                        and dyn_k <= 64
                    ):
                        scores32 = flat_scores.to(torch.float32)
                        selected_scores, selected_flat_idx = (
                            torch.ops.vllm.eagle_topk_small(scores32, dyn_k)
                        )
                        selected_flat_idx = selected_flat_idx.to(torch.int64)
                    else:
                        selected_scores, selected_flat_idx = torch.topk(
                            flat_scores, dyn_k, dim=-1
                        )
                    parent_idx = selected_flat_idx // max_topk
                    child_idx = selected_flat_idx % max_topk
                    batch_idx = self.arange_int64[:batch_size].view(
                        batch_size, 1
                    ).expand(batch_size, dyn_k)
                    selected_tokens = topk_ids[batch_idx, parent_idx, child_idx]
                    draft_token_ids = selected_tokens
                    draft_token_ids_list.append(draft_token_ids)
                    dyn_parent_scores = selected_scores
                    dyn_parent_indices = parent_idx.to(torch.int64)

                    assert dyn_paths is not None
                    assert dyn_node_token_ids is not None
                    dyn_offset = 1 + (level + 1) * dyn_k
                    dyn_idx = (
                        self.arange_int64[:dyn_k].to(dtype=torch.int32) + dyn_offset
                    ).view(1, -1).expand(batch_size, -1)
                    dyn_idx_i64 = dyn_idx.to(torch.int64)
                    dyn_node_token_ids[batch_idx, dyn_idx_i64] = selected_tokens.to(
                        torch.int32
                    )
                    level_offset = 1 + level * dyn_k
                    parent_dyn_idx = (level_offset + parent_idx).to(torch.int64)
                    parent_paths = dyn_paths[batch_idx, parent_dyn_idx]
                    dyn_paths[batch_idx, dyn_idx_i64, : level + 1] = parent_paths[
                        :, :, : level + 1
                    ]
                    dyn_paths[batch_idx, dyn_idx_i64, level + 1] = dyn_idx
            else:
                if use_tree_cuda_draft:
                    assert self._tree_new_draft_tokens is not None
                    assert self._tree_draft_tokens_buffer is not None
                    self._tree_new_draft_tokens[:batch_size].fill_(PADDING_SLOT_ID)
                    level_nodes = self._tree_nodes_per_level[level + 1]
                    if level_nodes.numel() > 0:
                        buffer_idx = level_nodes - 1
                        self._tree_new_draft_tokens[
                            :batch_size, buffer_idx, :max_topk
                        ] = topk_ids.to(dtype=torch.int32)
                    torch.ops.vllm.eagle_extract_real_draft_tokens(
                        level + 1,
                        self._tree_depth,
                        self._tree_total_drafts,
                        self._tree_max_top_k,
                        int(self._tree_tokens_gather_idx[level + 1].numel()),
                        self._tree_tokens_gather_idx[level + 1],
                        self._tree_top_k_list[level + 1],
                        self._tree_draft_tokens_indices_cumsum,
                        self._tree_new_draft_tokens[:batch_size],
                        self._tree_draft_tokens_buffer[:batch_size],
                    )
                    start = int(
                        self._tree_draft_tokens_indices_cumsum[level + 1].item()
                    )
                    end = int(
                        self._tree_draft_tokens_indices_cumsum[level + 2].item()
                    )
                    draft_token_ids = self._tree_draft_tokens_buffer[
                        :batch_size, start:end
                    ]
                elif (
                    envs.VLLM_EAGLE_CUDA_TREE_COPY
                    and hasattr(torch.ops.vllm, "eagle_tree_select_next_tokens")
                ):
                    draft_token_ids = torch.empty(
                        (batch_size, next_parent_idx.numel()),
                        device=topk_ids.device,
                        dtype=topk_ids.dtype,
                    )
                    torch.ops.vllm.eagle_tree_select_next_tokens(
                        topk_ids,
                        next_parent_idx,
                        next_child_idx,
                        draft_token_ids,
                    )
                else:
                    draft_token_ids = topk_ids[:, next_parent_idx, next_child_idx]
                draft_token_ids_list.append(draft_token_ids)
                if self._dynamic_tree:
                    assert dyn_paths is not None
                    assert dyn_node_token_ids is not None
                    assert parent_static_to_dynamic is not None
                    assert static_scores_prev is not None
                    assert topk_logprobs is not None
                    node_logprobs = topk_logprobs[:, next_parent_idx, next_child_idx]
                    parent_scores = static_scores_prev[:, next_parent_idx]
                    node_scores = node_logprobs + parent_scores
                    if exceeds_max_model_len.any():
                        node_scores = torch.where(
                            exceeds_max_model_len.view(batch_size, 1),
                            -float("inf"),
                            node_scores,
                        )
                    if (
                        hasattr(torch.ops.vllm, "eagle_topk_small")
                        and self._opt_mode != "baseline"
                        and dyn_k <= 64
                    ):
                        scores32 = node_scores.to(torch.float32)
                        selected_scores, selected_static_idx = (
                            torch.ops.vllm.eagle_topk_small(scores32, dyn_k)
                        )
                        selected_static_idx = selected_static_idx.to(torch.int64)
                    else:
                        selected_scores, selected_static_idx = torch.topk(
                            node_scores, dyn_k, dim=-1
                        )
                    selected_tokens = draft_token_ids.gather(1, selected_static_idx)
                    dyn_offset = 1 + (level + 1) * dyn_k
                    dyn_idx = (
                        self.arange_int64[:dyn_k].to(dtype=torch.int32) + dyn_offset
                    ).view(1, -1).expand(batch_size, -1)
                    dyn_idx_i64 = dyn_idx.to(torch.int64)
                    batch_idx = self.arange_int64[:batch_size].view(
                        batch_size, 1
                    ).expand(batch_size, dyn_k)
                    dyn_node_token_ids[batch_idx, dyn_idx_i64] = selected_tokens.to(
                        torch.int32
                    )
                    parent_static_idx = next_parent_idx[selected_static_idx]
                    parent_dyn_idx = parent_static_to_dynamic[
                        batch_idx, parent_static_idx
                    ].to(torch.int64)
                    parent_paths = dyn_paths[batch_idx, parent_dyn_idx]
                    dyn_paths[batch_idx, dyn_idx_i64, : level + 1] = parent_paths[
                        :, :, : level + 1
                    ]
                    dyn_paths[batch_idx, dyn_idx_i64, level + 1] = dyn_idx

                    parent_static_to_dynamic = torch.full(
                        (batch_size, next_parent_idx.numel()),
                        -1,
                        dtype=torch.int32,
                        device=node_scores.device,
                    )
                    parent_static_to_dynamic[batch_idx, selected_static_idx] = dyn_idx
                    static_scores_prev = torch.full(
                        (batch_size, next_parent_idx.numel()),
                        -float("inf"),
                        dtype=torch.float32,
                        device=node_scores.device,
                    )
                    static_scores_prev[batch_idx, selected_static_idx] = selected_scores

            # Update the draft token ids for the next tree level.
        if self._dynamic_tree:
            assert dyn_node_token_ids is not None
            if use_opt_dynamic:
                assert self._dyn_draft_ids_in is not None
                assert self._dyn_draft_lens_in is not None
                dyn_node_token_ids.fill_(PADDING_SLOT_ID)
                max_len = self._dyn_draft_ids_in.size(1)
                if max_len > 0:
                    draft_lens = self._dyn_draft_lens_in[:batch_size]
                    mask = (
                        self.arange_int64[:max_len]
                        .view(1, -1)
                        .expand(batch_size, -1)
                        < draft_lens.view(-1, 1)
                    )
                    dyn_node_token_ids[:, 1 : 1 + max_len][mask] = (
                        self._dyn_draft_ids_in[:batch_size, :max_len][mask]
                    )
            dyn_tree_len = 1 + dyn_k * self._tree_depth
            dyn_tokens = dyn_node_token_ids[:batch_size, 1:dyn_tree_len].to(
                torch.int64
            )
            return [dyn_tokens]
        return draft_token_ids_list

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        num_reqs = common_attn_metadata.num_reqs
        new_query_start_loc_cpu = self._spec_query_start_loc_cpu[: num_reqs + 1]
        new_query_start_loc_cpu.zero_()
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_cpu = self._spec_token_indices_cpu[:total_num_tokens]
        token_indices_np = token_indices_cpu.numpy()
        np.subtract(
            self.token_arange_np[:total_num_tokens],
            new_query_start_locs_expanded,
            out=token_indices_np,
        )
        token_indices_np += old_query_start_locs_expanded
        token_indices = token_indices_cpu.to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def prepare_inputs_unpadded_gpu(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        valid_sampled_tokens_count: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor | None = None,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        GPU path for unpadded speculative decoding. This avoids Python/Numpy
        bookkeeping by computing token indices and updated metadata on GPU.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = common_attn_metadata.query_start_loc.device
        valid_counts = valid_sampled_tokens_count.to(dtype=torch.int32)

        query_start_loc = common_attn_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_rejected = query_lens - valid_counts
        new_seq_lens = common_attn_metadata.seq_lens - num_rejected

        if cu_num_draft_tokens is not None and num_reqs > 0:
            if (
                envs.VLLM_EAGLE_CUDA_REWIND
                and hasattr(torch.ops.vllm, "eagle_rewind_slot_mapping")
            ):
                torch.ops.vllm.eagle_rewind_slot_mapping(
                    cu_num_draft_tokens,
                    valid_counts,
                    query_start_loc,
                    common_attn_metadata.slot_mapping,
                    PADDING_SLOT_ID,
                )
            else:
                rewind_block_size = min(
                    1024, triton.next_power_of_2(self.num_speculative_tokens + 1)
                )
                _eagle_rewind_slot_mapping_kernel[(num_reqs,)](
                    cu_num_draft_tokens,
                    valid_counts,
                    query_start_loc,
                    common_attn_metadata.slot_mapping,
                    num_reqs,
                    PAD_ID=PADDING_SLOT_ID,
                    BLOCK_SIZE=rewind_block_size,
                )

        new_query_start_loc = torch.empty_like(query_start_loc)
        new_query_start_loc[0] = 0
        new_query_start_loc[1:] = torch.cumsum(valid_counts, dim=0)
        total_num_tokens = int(new_query_start_loc[-1].item())

        if total_num_tokens > 0:
            repeat_counts = valid_counts.to(torch.int64)
            req_ids = torch.repeat_interleave(
                torch.arange(num_reqs, device=device, dtype=torch.int64),
                repeat_counts,
            )
            token_positions = (
                torch.arange(total_num_tokens, device=device, dtype=torch.int32)
                - new_query_start_loc[req_ids]
            )
            token_indices = query_start_loc[req_ids] + token_positions
        else:
            token_indices = torch.empty((0,), device=device, dtype=torch.int32)

        query_start_loc_cpu = self._spec_query_start_loc_cpu[: num_reqs + 1]
        query_start_loc_cpu.copy_(new_query_start_loc, non_blocking=True)
        seq_lens_cpu = self._spec_seq_lens_cpu[:num_reqs]
        seq_lens_cpu.copy_(new_seq_lens, non_blocking=True)

        max_query_len = int(valid_counts.max().item()) if num_reqs else 0
        max_seq_len = int(new_seq_lens.max().item()) if num_reqs else 0

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=new_seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        # FIXME: support hybrid kv for draft model
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config, DeepseekV32IndexerCache
            ).keys()
        )

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
            - target_attn_layer_names
        )
        indexer_layers = get_layers_from_vllm_config(
            self.vllm_config, DeepseekV32IndexerCache
        )
        draft_indexer_layer_names = indexer_layers.keys() - target_indexer_layer_names
        self.attn_layer_names = list(draft_attn_layer_names - draft_indexer_layer_names)
        self.indexer_layer_names = list(draft_indexer_layer_names)

        if self.indexer_layer_names:
            first_layer = self.indexer_layer_names[0]
            self.draft_indexer_metadata_builder = (
                indexer_layers[first_layer]
                .get_attn_backend()
                .get_builder_cls()(
                    indexer_layers[first_layer].get_kv_cache_spec(self.vllm_config),
                    self.indexer_layer_names,
                    self.vllm_config,
                    self.device,
                )
            )
        else:
            self.draft_indexer_metadata_builder = None

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            if hasattr(target_language_model.model, "embed_tokens"):
                target_embed_tokens = target_language_model.model.embed_tokens
            elif hasattr(target_language_model.model, "embedding"):
                target_embed_tokens = target_language_model.model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

        # share lm_head with the target model if needed
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs=True,
        is_graph_capturing=False,
    ) -> None:
        # Determine if CUDA graphs should be used for this run.
        cudagraphs_enabled = use_cudagraphs and self.use_cuda_graph

        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        for fwd_idx in range(
            self.num_speculative_tokens if not is_graph_capturing else 1
        ):
            if fwd_idx <= 1:
                num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    num_tokens_padded=num_tokens,
                )
                if (
                    cudagraphs_enabled
                    and num_tokens_dp_padded
                    <= self.compilation_config.max_cudagraph_capture_size
                ):
                    num_input_tokens = self.vllm_config.pad_for_cudagraph(
                        num_tokens_dp_padded
                    )
                else:
                    num_input_tokens = num_tokens_dp_padded
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[self.dp_rank] = num_input_tokens

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
                if cudagraphs_enabled
                else CUDAGraphMode.NONE,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=inputs_embeds,
                )

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find and return the attention metadata builders for EAGLE layers.

        Returns:
            The metadata builders for EAGLE layers.

        Raises:
            AssertionError: If no metadata builders are found for EAGLE layers.
        """
        builder = None
        chosen_layer = self.attn_layer_names[0]

        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            "Failed to find attention metadata builder for EAGLE layers."
        )
        return builder

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self.attn_layer_names
                    ]
                )
            )
            == 1
        ), "All eagle layers should belong to the same kv cache group"

    def _pad_batch_across_dp(
        self,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> tuple[int, torch.Tensor]:
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_toks_across_dp, _ = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens_unpadded,
            parallel_config=self.vllm_config.parallel_config,
            allow_microbatching=False,
            allow_dp_padding=self.use_cuda_graph,
            num_tokens_padded=num_tokens_padded,
            uniform_decode=None,
            num_scheduled_tokens_per_request=None,
        )
        assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

        num_tokens_dp_padded = num_tokens_padded
        if num_toks_across_dp is not None:
            num_tokens_dp_padded = int(num_toks_across_dp[self.dp_rank].item())
        return num_tokens_dp_padded, num_toks_across_dp


@triton.jit
def _eagle_compute_slot_mapping_kernel(
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    max_model_len,
    slot_mapping_ptr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + req_idx)
    exceeds = pos >= max_model_len
    pos = tl.where(exceeds, 0, pos)
    block_idx = pos // block_size
    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_idx)
    slot_id = block_id * block_size + pos % block_size
    slot_id = tl.where(exceeds, PAD_ID, slot_id)
    tl.store(slot_mapping_ptr + req_idx, slot_id)


@triton.jit
def _eagle_tree_slot_mapping_kernel(
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    max_model_len,
    query_len,
    num_tokens,
    slot_mapping_ptr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_tokens
    pos = tl.load(positions_ptr + offs, mask=mask, other=0)
    exceeds = pos >= max_model_len
    pos = tl.where(exceeds, 0, pos)
    req_idx = offs // query_len
    block_idx = pos // block_size
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_idx,
        mask=mask,
        other=0,
    )
    slot_id = block_id * block_size + pos % block_size
    slot_id = tl.where(exceeds, PAD_ID, slot_id)
    tl.store(slot_mapping_ptr + offs, slot_id, mask=mask)


@triton.jit
def _eagle_update_draft_state_kernel(
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    positions_stride0,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    seq_lens_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    hidden_size,
    block_size,
    max_model_len,
    PAD_ID: tl.constexpr,
    USE_MROPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    draft_token = tl.load(draft_tokens_ptr + req_idx).to(tl.int32)
    tl.store(input_ids_ptr + req_idx, draft_token)

    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + req_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    pos = tl.load(positions_ptr + req_idx)
    new_pos = pos + 1
    exceeds = new_pos >= max_model_len
    new_pos = tl.minimum(new_pos, max_model_len - 1)
    if USE_MROPE:
        base_ptr = positions_ptr
        for dim in range(3):
            tl.store(base_ptr + dim * positions_stride0 + req_idx, new_pos)
    else:
        tl.store(positions_ptr + req_idx, new_pos)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    seq_len = tl.where(exceeds, 1, seq_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)

    block_idx = new_pos // block_size
    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_idx)
    slot_id = block_id * block_size + new_pos % block_size
    slot_id = tl.where(exceeds, PAD_ID, slot_id)
    tl.store(slot_mapping_ptr + req_idx, slot_id)


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    logits = apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(
            is_greedy,
            greedy_token_ids,
            next_token_ids,
        )
    return next_token_ids, probs


@triton.jit
def _eagle_rewind_slot_mapping_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_ptr,  # [num_reqs + 1]
    slot_mapping_ptr,  # [num_tokens]
    num_reqs,  # tl.int32
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx >= num_reqs:
        return

    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    if num_rejected_tokens <= 0:
        return

    end_idx = tl.load(query_start_loc_ptr + req_idx + 1)
    start_idx = end_idx - num_rejected_tokens

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_rejected_tokens
    tl.store(slot_mapping_ptr + start_idx + offs, PAD_ID, mask=mask)
