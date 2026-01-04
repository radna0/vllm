# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import os
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import torch
import torch.nn as nn

# Environment flag to enable advanced sampling filters (min_p, top_p, top_k)
# for draft tokens. When enabled, draft tokens better match target sampling
# constraints, potentially improving acceptance rates.
ENABLE_DRAFT_SAMPLING = os.environ.get("VLLM_EAGLE_DRAFT_SAMPLING", "0") == "1"

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
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.tree_attn import (
    TreeAttentionMetadata,
    TreeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
)
from vllm.v1.worker.gpu.spec_decode.spec_tree_manager import (
    SpecTreeManager,
    create_spec_tree_manager_from_choices,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

try:
    from vllm_eagle import ops as eagle_ops
except ImportError:
    eagle_ops = None

logger = init_logger(__name__)

# Phase 2: Fused draft-verify kernel (eliminates 2-3 kernel launches)
PHASE2_FUSED_VERIFY = os.environ.get("VLLM_EAGLE_PHASE2_FUSED", "0") == "1"
try:
    if PHASE2_FUSED_VERIFY:
        from vllm_eagle._C import (
            fused_gumbel_sample_warp_optimized,
            fused_draft_verify_sample,
        )

        logger.info("Phase 2 fused kernels enabled")
    else:
        fused_gumbel_sample_warp_optimized = None
        fused_draft_verify_sample = None
except ImportError:
    fused_gumbel_sample_warp_optimized = None
    fused_draft_verify_sample = None
    if PHASE2_FUSED_VERIFY:
        logger.warning("Phase 2 fused kernels requested but not available")

PADDING_SLOT_ID = -1


def fused_sample_and_verify(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
    min_p: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Phase 2 Optimization: Fused draft sampling + verification kernel.

    Combines three operations into one GPU kernel:
    1. Sample draft tokens using Gumbel-Max
    2. Verify each token against target distribution
    3. Return accepted tokens with early exit on rejection

    Performance:
    - Eliminates 2-3 kernel launches
    - Reduces memory bandwidth by ~40%
    - Estimated +5-8% throughput improvement

    Args:
        draft_logits: [batch_size, max_draft_len, vocab_size] - Draft model logits
        target_logits: [batch_size, max_draft_len+1, vocab_size] - Target model logits
        temperature: Sampling temperature (applied to both draft and target)
        min_p: Min-p filtering threshold

    Returns:
        Tuple of (accepted_tokens, num_accepted):
        - accepted_tokens: [batch_size, max_draft_len] - Accepted token IDs
        - num_accepted: [batch_size] - Number of tokens accepted per batch
    """
    if fused_draft_verify_sample is None:
        raise RuntimeError(
            "Phase 2 fused kernels not available. Set VLLM_EAGLE_PHASE2_FUSED=1 and rebuild vllm_eagle."
        )

    batch_size = draft_logits.size(0)
    max_draft_len = draft_logits.size(1)
    device = draft_logits.device

    # GOLD STANDARD: We no longer scale here. The fused kernel handles T.
    # This preserves raw logit precision for the 120B model.

    # Pre-generate uniform samples for Gumbel and verification
    uniform_samples = torch.empty(
        batch_size,
        max_draft_len,
        draft_logits.size(2),
        dtype=torch.float32,
        device=device,
    ).uniform_(1e-10, 1.0)

    verify_samples = torch.empty(
        batch_size, max_draft_len, dtype=torch.float32, device=device
    ).uniform_(0.0, 1.0)

    min_p_tensor = torch.full((batch_size,), min_p, dtype=torch.float32, device=device)

    # Output buffers
    accepted_tokens = torch.empty(
        batch_size, max_draft_len, dtype=torch.int32, device=device
    )
    num_accepted = torch.empty(batch_size, dtype=torch.int32, device=device)

    # Single fused kernel call (replaces 3+ kernel launches)
    # GOLD STANDARD: Fused kernel now handles temperature scaling internally
    # and correctly implements rejection sampling probability.
    fused_draft_verify_sample(
        accepted_tokens,
        num_accepted,
        draft_logits,  # RAW
        target_logits,  # RAW
        uniform_samples,
        verify_samples,
        min_p_tensor,
        torch.full((batch_size,), temperature, dtype=torch.float32, device=device),
    )

    return accepted_tokens, num_accepted


def _apply_sampling_filters_logits(
    logits: torch.Tensor,
    sampling_metadata: "SamplingMetadata",
) -> torch.Tensor:
    """Apply min_p, top_p, top_k filtering in the logit domain.

    This uses optimized CUDA kernels to avoid Python loop overhead.
    """
    batch_size = logits.shape[0]
    device = logits.device

    # Extract filter parameters as tensors
    top_k = sampling_metadata.top_k
    if top_k is None:
        top_k = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    else:
        top_k = top_k.to(torch.int32)

    top_p = sampling_metadata.top_p
    if top_p is None:
        top_p = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)

    min_p = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    if hasattr(sampling_metadata, "logitsprocs"):
        lprocs = sampling_metadata.logitsprocs
        if hasattr(lprocs, "min_p") and lprocs.min_p.numel() > 0:
            min_p = lprocs.min_p.to(torch.float32)

    # Call optimized GPU kernel (handles batched top_k, min_p, top_p)
    if eagle_ops is not None:
        eagle_ops.apply_logit_filters(logits, top_k, top_p, min_p)
    else:
        # Complete PyTorch fallback: Apply all filters (top_k, min_p, top_p)
        # This is slower than the CUDA kernel but ensures correctness
        batch_size, vocab_size = logits.shape

        # Apply top-k filtering
        if (top_k > 0).any():
            for i in range(batch_size):
                k = top_k[i].item()
                if 0 < k < vocab_size:
                    topk_vals, _ = torch.topk(logits[i], k)
                    threshold = topk_vals[-1]
                    logits[i].masked_fill_(logits[i] < threshold, -10000.0)

        # Apply min-p filtering
        if (min_p > 0).any():
            max_logit = logits.max(dim=-1, keepdim=True).values
            threshold = max_logit + torch.log(min_p.unsqueeze(-1) + 1e-10)
            logits.masked_fill_(logits < threshold, -10000.0)

        # Apply top-p filtering (nucleus sampling)
        if (top_p < 1.0).any():
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(probs, dim=-1)

            for i in range(batch_size):
                p = top_p[i].item()
                if p < 1.0:
                    # Find cutoff index where cumulative prob exceeds top_p
                    cutoff_mask = cumsum_probs[i] > p
                    cutoff_idx = cutoff_mask.nonzero(as_tuple=True)[0]
                    if len(cutoff_idx) > 0:
                        cutoff = cutoff_idx[0].item() + 1
                        # Mask out tokens beyond cutoff
                        mask_indices = sorted_indices[i, cutoff:]
                        logits[i, mask_indices] = -10000.0

    return logits


def _apply_sampling_filters(
    probs: torch.Tensor,
    sampling_metadata: "SamplingMetadata",
) -> torch.Tensor:
    """Apply min_p, top_p, top_k filtering to probability distribution.

    This function filters the probability distribution to better match
    the target model's sampling constraints, improving acceptance rates.
    Used for multi-sample (tree decoding) paths.
    """
    batch_size, vocab_size = probs.shape

    # Apply top_k filtering first (most restrictive)
    if sampling_metadata.top_k is not None:
        top_k = sampling_metadata.top_k
        top_k = torch.clamp(top_k, max=vocab_size)
        for i in range(batch_size):
            k = int(top_k[i].item())
            if k > 0 and k < vocab_size:
                threshold = probs[i].topk(k).values[-1]
                probs[i] = torch.where(
                    probs[i] >= threshold, probs[i], torch.zeros_like(probs[i])
                )

    # Apply top_p (nucleus) filtering
    if sampling_metadata.top_p is not None:
        top_p = sampling_metadata.top_p
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumsum_probs - sorted_probs > top_p.unsqueeze(-1)
        sorted_probs = torch.where(
            nucleus_mask, torch.zeros_like(sorted_probs), sorted_probs
        )
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

    # Apply min_p filtering
    if hasattr(sampling_metadata, "logitsprocs"):
        logitsprocs = sampling_metadata.logitsprocs
        if hasattr(logitsprocs, "min_p") and logitsprocs.min_p.numel() > 0:
            min_p = logitsprocs.min_p
            if min_p.dim() == 1:
                min_p = min_p.unsqueeze(-1)
            max_probs = probs.max(dim=-1, keepdim=True).values
            adjusted_min_p = min_p * max_probs
            probs = torch.where(probs >= adjusted_min_p, probs, torch.zeros_like(probs))

    # Renormalize
    probs_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    probs = probs / probs_sum

    return probs


def sample_draft_tokens(
    logits: torch.Tensor,
    sampling_metadata: "SamplingMetadata | None" = None,
    num_samples: int = 1,
    return_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sample draft tokens with temperature support.

    When sampling_metadata is provided and temperature > 0, applies temperature
    scaling before sampling. Otherwise falls back to argmax (greedy).

    When VLLM_EAGLE_DRAFT_SAMPLING=1 is set, also applies min_p, top_p, top_k
    filtering to better match target sampling constraints.

    Args:
        logits: [batch_size, vocab_size] logits from draft model
        sampling_metadata: Optional sampling params with temperature
        num_samples: Number of tokens to sample per position (for tree decoding)
        return_probs: If True, also return probability distribution for rejection sampling

    Returns:
        If return_probs=False: Token IDs [batch_size] or [batch_size, num_samples]
        If return_probs=True: (token_ids, probs) where probs is [batch_size, vocab_size]
    """
    # Fast path: no metadata or all greedy
    if sampling_metadata is None or sampling_metadata.all_greedy:
        if num_samples == 1:
            tokens = logits.argmax(dim=-1)
        else:
            # For tree decoding, we still use top-k on raw logits
            tokens = torch.topk(logits, num_samples, dim=-1).indices

        if return_probs:
            # For greedy, we set draft_prob=1 (handled by rejection sampler)
            return tokens, None
        return tokens

    assert sampling_metadata.temperature is not None
    temperature = sampling_metadata.temperature

    # Handle mixed greedy/random requests
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        # Avoid division by zero for greedy requests
        temperature = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    # Deduplicate scaling if needed for probs or multi-sample
    # We only compute this IF we are in a path that requires the full distribution
    logits_scaled = None
    if return_probs or num_samples > 1:
        logits_scaled = logits_raw / temperature.unsqueeze(-1)

    if num_samples == 1:
        # Optimized path (num_samples=1): Use FlashSampling/Logit-domain Gumbel-Max
        if ENABLE_DRAFT_SAMPLING:
            # Apply filters in logit domain (FastSampling) using optimized kernel
            # This operates on RAW logits per the Gold Standard
            logits_raw = _apply_sampling_filters_logits(logits_raw, sampling_metadata)

        # Fused Gumbel-Max sampling
        # We pre-generate uniform samples to keep the kernel pure (no RNG dependency)
        u = torch.empty_like(logits_raw, dtype=torch.float32).uniform_(1e-10, 1.0)
        batch_size = logits_raw.shape[0]
        device = logits_raw.device
        draft_tokens = torch.empty(batch_size, dtype=torch.int32, device=device)

        if fused_gumbel_sample_warp_optimized is not None:
            # Phase 2: Warp-optimized Gumbel sampling using Noise-Scaling
            # argmax(logit + T*g)
            min_p_tensor = torch.zeros(
                (batch_size,), dtype=torch.float32, device=device
            )
            fused_gumbel_sample_warp_optimized(
                draft_tokens,
                logits_raw,
                u,
                min_p_tensor,
                temperature,  # Pass temperature for noise scaling
            )
            draft_tokens = draft_tokens.to(torch.int64)
        elif eagle_ops is not None:
            # Original: Block-level Gumbel-Max sampling using Noise-Scaling
            eagle_ops.fused_gumbel_sample(
                draft_tokens,
                logits_raw,
                torch.full((batch_size,), -1, dtype=torch.int32, device=device),
                torch.full((batch_size,), 1.0, dtype=torch.float32, device=device),
                torch.zeros((batch_size,), dtype=torch.float32, device=device),
                temperature,  # Pass temperature
                u,
            )
            draft_tokens = draft_tokens.to(torch.int64)
        else:
            # Fallback (Manual argmax(logits + T*gumbel))
            gumbel = -torch.log(-torch.log(u))
            draft_tokens = (logits_raw + temperature.unsqueeze(-1) * gumbel).argmax(
                dim=-1
            )

        # Override with argmax for greedy requests
        if is_greedy is not None:
            greedy_tokens = logits.argmax(dim=-1)
            draft_tokens = torch.where(is_greedy, greedy_tokens, draft_tokens)

        if return_probs:
            # We rarely need probs here but if requested, we must compute them
            # This follows the original logic but optimized.
            # Note: For Return Probs, we use the standard softmax(y/T)
            # to match the rejection sampler's expectations.
            probs = torch.softmax(logits_scaled, dim=-1)
            return draft_tokens, probs
    else:
        # Multi-sample path (tree decoding): Compute probs for multinomial
        # We MUST use logit-domain filtering here to avoid OOM from prob-domain allocations.
        if ENABLE_DRAFT_SAMPLING:
            # Apply filters in logit domain (In-place)
            logits_scaled = _apply_sampling_filters_logits(
                logits_scaled, sampling_metadata
            )

        probs = torch.softmax(logits_scaled, dim=-1)

        # For multi-sample (tree decoding), sample without replacement
        draft_tokens = torch.multinomial(probs, num_samples, replacement=False)

        # Override with topk for greedy requests
        if is_greedy is not None:
            greedy_tokens = torch.topk(logits, num_samples, dim=-1).indices
            # Expand is_greedy for broadcasting
            is_greedy_expanded = is_greedy.unsqueeze(-1).expand_as(draft_tokens)
            draft_tokens = torch.where(is_greedy_expanded, greedy_tokens, draft_tokens)

        if return_probs:
            return draft_tokens, probs
    return draft_tokens


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
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

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
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + num_drafts_per_level[level]
            )
            self.child_drafts_per_level.append(
                num_drafts_per_level[level] // num_drafts_per_level[level - 1]
            )
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1,
            len(self.tree_choices) + 1,
            device=device,
            dtype=torch.int32,
        ).repeat(max_batch_size, 1)

        # Initialize SpecTreeManager for tree-aware attention masks.
        # This provides consistent tree attention bias for TreeAttentionBackend
        # and enables future integration with TreeDraftingLoopWrapper.
        self.spec_tree_manager = create_spec_tree_manager_from_choices(
            eagle_choices=self.tree_choices,
            max_num_requests=max_batch_size,
            device=str(device),
        )
        logger.info(
            "Initialized SpecTreeManager: max_draft_len=%d, max_total_draft_tokens=%d",
            self.spec_tree_manager.max_draft_len,
            self.spec_tree_manager.max_total_draft_tokens,
        )

        # Dynamic Eagle buffers for SGLang-style tree construction
        self.use_dynamic_eagle = self.method == "dynamic_eagle"
        if self.use_dynamic_eagle:
            topk = getattr(self.speculative_config, "speculative_eagle_topk", 8)
            num_draft = getattr(
                self.speculative_config, "speculative_num_draft_tokens", 64
            )
            self.dyn_topk = topk
            self.dyn_num_draft_tokens = num_draft

            # Buffers for build_tree_kernel_efficient
            self.dyn_tree_mask = torch.zeros(
                (max_batch_size, num_draft, num_draft), dtype=torch.bool, device=device
            )
            self.dyn_positions = torch.zeros(
                (max_batch_size, num_draft), dtype=torch.int64, device=device
            )
            self.dyn_retrive_index = torch.zeros(
                (max_batch_size, num_draft), dtype=torch.int64, device=device
            )
            self.dyn_retrive_next_token = torch.full(
                (max_batch_size, num_draft), -1, dtype=torch.int64, device=device
            )
            self.dyn_retrive_next_sibling = torch.full(
                (max_batch_size, num_draft), -1, dtype=torch.int64, device=device
            )
            self.dyn_parent_list = torch.zeros(
                (max_batch_size, topk * (self.num_speculative_tokens - 1) + 1),
                dtype=torch.int64,
                device=device,
            )
            self.dyn_selected_index = torch.zeros(
                (max_batch_size, num_draft - 1), dtype=torch.int64, device=device
            )
            self.dyn_verified_seq_len = torch.zeros(
                max_batch_size, dtype=torch.int64, device=device
            )
            self.dyn_draft_tokens = torch.zeros(
                (max_batch_size, num_draft), dtype=torch.int64, device=device
            )

        # Storage for draft probabilities - populated during propose() for rejection sampling
        # If None, rejection sampler uses draft_prob=1 (argmax behavior)
        self.last_draft_probs: torch.Tensor | None = None

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        else:
            self.positions[:num_tokens] = positions

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
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        # Clear previous draft probs - will be populated during sampling
        self.last_draft_probs = None
        draft_probs_list: list[torch.Tensor] = []

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids, probs = sample_draft_tokens(
                logits, sampling_metadata, return_probs=True
            )
            self.last_draft_probs = probs
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

        # Dynamic Eagle: use SGLang-style dynamic tree construction
        if self.use_dynamic_eagle:
            seq_lens = common_attn_metadata.seq_lens[:batch_size]
            draft_tokens, retrive_index, retrive_next_token, retrive_next_sibling = (
                self.propose_dynamic_tree(
                    batch_size=batch_size,
                    logits=logits,
                    positions=positions,
                    hidden_states=hidden_states,
                    seq_lens=seq_lens,
                    common_attn_metadata=common_attn_metadata,
                )
            )
            # Store tree structure for verification step
            self._last_retrive_index = retrive_index
            self._last_retrive_next_token = retrive_next_token
            self._last_retrive_next_sibling = retrive_next_sibling
            return draft_tokens

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # Draft using tree attention.
            # TODO: Implement return_probs for tree attention path
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        # First draft token with temperature awareness
        draft_token_ids, probs = sample_draft_tokens(
            logits, sampling_metadata, return_probs=True
        )
        if probs is not None:
            draft_probs_list.append(probs)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=batch_size,
            num_tokens_padded=batch_size,
        )

        if (
            self.use_cuda_graph
            and batch_size_dp_padded
            <= self.compilation_config.max_cudagraph_capture_size
        ):
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size_dp_padded)
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            input_batch_size = batch_size_dp_padded
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        if batch_size_across_dp is not None:
            batch_size_across_dp[self.dp_rank] = input_batch_size

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata.seq_lens += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # Compute the slot mapping.
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // self.block_size
            else:
                block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                common_attn_metadata.slot_mapping = (
                    block_ids * self.block_size + clamped_positions[0] % self.block_size
                )
            else:
                common_attn_metadata.slot_mapping = (
                    block_ids * self.block_size + clamped_positions % self.block_size
                )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )

            # Rebuild attention metadata
            attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
                common_attn_metadata=common_attn_metadata, draft_index=token_index + 1
            )
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
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
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            # Use temperature-aware sampling
            draft_token_ids, probs = sample_draft_tokens(
                logits, sampling_metadata, return_probs=True
            )
            if probs is not None:
                draft_probs_list.append(probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)

        # Concatenate and store probs for rejection sampling
        if draft_probs_list:
            # Shape: [batch_size * num_speculative_tokens, vocab_size]
            self.last_draft_probs = torch.cat(draft_probs_list, dim=0)

        return draft_token_ids

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
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
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

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc.to(torch.int32),
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
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

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

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
        tree_attn_metadata_builder = self.runner.attn_groups[0][
            0
        ].get_metadata_builder()
        assert isinstance(tree_attn_metadata_builder, TreeAttentionMetadataBuilder)

        total_num_drafts = self.cu_drafts_per_level[0]
        level_num_drafts = total_num_drafts
        # Sample a draft token for each child at the tree root level.
        num_children = self.child_drafts_per_level[0]
        if num_children == 1:
            draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
        else:
            draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                batch_size, -1
            )
        draft_token_ids_list = [draft_token_ids]
        draft_hidden_states = hidden_states.view(batch_size, 1, -1)

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
        # Precompute the draft token positions.
        flattened_draft_positions = (
            positions.view(batch_size, -1) + self.tree_draft_pos_offsets[:batch_size, :]
        )
        tree_depth = len(self.cu_drafts_per_level)
        for level in range(tree_depth - 1):
            # Get draft positions for RoPE.
            draft_positions = positions + (level + 1)
            exceeds_max_model_len = (positions + total_num_drafts) >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            draft_positions = torch.where(
                exceeds_max_model_len,
                0,
                draft_positions,
            ).view(batch_size, -1)

            if level_num_drafts > 1:
                # Repeat the positions for each draft at this level.
                draft_positions = draft_positions.repeat_interleave(
                    level_num_drafts, dim=1
                )

            if num_children > 1:
                # Repeat draft hidden states for each child.
                draft_hidden_states = draft_hidden_states.repeat_interleave(
                    num_children, dim=1
                )

            # Concatenate the draft tokens, positions, and hidden states.
            tree_input_ids = torch.cat([tree_input_ids, draft_token_ids], dim=1)
            tree_positions = torch.cat([tree_positions, draft_positions], dim=1)
            tree_hidden_states = torch.cat(
                [tree_hidden_states, draft_hidden_states], dim=1
            )

            # Build new attention metadata for the next level of drafts.
            # This is necessary to support tree attention.
            query_len = total_num_drafts
            common_attn_metadata = replace(
                common_attn_metadata,
                query_start_loc=query_len * self.arange[: batch_size + 1],
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

            # Compute the slot mapping.
            query_positions = flattened_draft_positions[:, level : level + query_len]
            block_numbers = query_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers)
            slot_mapping = (
                block_ids * self.block_size + query_positions % self.block_size
            )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping[exceeds_max_model_len] = PADDING_SLOT_ID
            attn_metadata.slot_mapping = slot_mapping.view(-1)

            # Copy inputs to buffer for cudagraph.
            num_tokens = attn_metadata.num_actual_tokens
            input_ids = tree_input_ids.view(-1)
            self.input_ids[:num_tokens] = input_ids
            self.positions[:num_tokens] = tree_positions.view(-1)
            self.hidden_states[:num_tokens] = tree_hidden_states.view(num_tokens, -1)

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
            num_children = self.child_drafts_per_level[level + 1]
            if num_children == 1:
                draft_token_ids = logits.argmax(dim=-1).view(batch_size, -1)
            else:
                draft_token_ids = torch.topk(logits, num_children, dim=-1).indices.view(
                    batch_size, -1
                )
            draft_token_ids_list.append(draft_token_ids)

            # Update the # drafts counters for the next tree level.
            level_num_drafts = self.cu_drafts_per_level[level + 1] - total_num_drafts
            total_num_drafts = self.cu_drafts_per_level[level + 1]
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
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
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
        token_offests = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

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
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

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
                cudagraph_runtime_mode=(
                    CUDAGraphMode.PIECEWISE
                    if cudagraphs_enabled
                    else CUDAGraphMode.NONE
                ),
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

        assert (
            builder is not None
        ), "Failed to find attention metadata builder for EAGLE layers."
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

    def propose_dynamic_tree(
        self,
        batch_size: int,
        logits: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform dynamic tree-based draft token generation.

        This implements the SGLang-style dynamic tree construction using
        the ported CUDA kernels. It generates a tree of draft tokens where
        the tree structure is determined dynamically based on token probabilities.

        Args:
            batch_size: Number of requests in the batch
            logits: [batch_size, vocab_size] logits from target model
            positions: [batch_size] current positions for each request
            hidden_states: [batch_size, hidden_size] hidden states
            seq_lens: [batch_size] sequence lengths
            common_attn_metadata: Attention metadata for building tree attention

        Returns:
            Tuple of (draft_token_ids, retrive_index, retrive_next_token, retrive_next_sibling)
        """
        from vllm import _custom_ops as ops

        topk = self.dyn_topk
        num_draft_tokens = self.dyn_num_draft_tokens
        num_steps = self.num_speculative_tokens

        # Reset buffers
        self.dyn_tree_mask[:batch_size].zero_()
        self.dyn_retrive_next_token[:batch_size].fill_(-1)
        self.dyn_retrive_next_sibling[:batch_size].fill_(-1)

        # Step 0: Get top-k from initial logits
        probs = logits.softmax(dim=-1)
        topk_p, topk_index = torch.topk(probs, topk, dim=-1)  # [b, topk]

        # Set verified sequence lengths
        self.dyn_verified_seq_len[:batch_size] = seq_lens[:batch_size].to(torch.int64)

        # Root token (position 0 in the tree) - use argmax of initial logits
        root_tokens = logits.argmax(dim=-1)
        self.dyn_draft_tokens[:batch_size, 0] = root_tokens

        # First level children (positions 1 to topk)
        self.dyn_draft_tokens[:batch_size, 1 : topk + 1] = topk_index

        # Initialize scores for beam search
        scores = topk_p  # [b, topk]

        # Track parent indices for tree construction
        # parent_list[0] = -1 (root has no parent)
        # parent_list[1:topk+1] = 0 (all first-level children have root as parent)
        self.dyn_parent_list[:batch_size, 0] = -1
        self.dyn_parent_list[:batch_size, 1 : topk + 1] = 0

        # Initialize selected_index for first level
        self.dyn_selected_index[:batch_size, 0:topk] = (
            torch.arange(topk, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )

        current_token_idx = topk + 1

        # Expand hidden states for frontier
        frontier_hidden = hidden_states.repeat_interleave(
            topk, dim=0
        )  # [b*topk, hidden]

        # For subsequent levels, we would need to run the draft model on the frontier
        # and update the tree structure. For now, we complete the tree construction
        # with the first level and call the kernel.

        # Call build_tree_kernel_efficient to finalize tree structure
        ops.build_tree_kernel_efficient(
            self.dyn_parent_list[:batch_size].contiguous(),
            self.dyn_selected_index[:batch_size].contiguous(),
            self.dyn_verified_seq_len[:batch_size].contiguous(),
            self.dyn_tree_mask[:batch_size].contiguous().view(batch_size, -1),
            self.dyn_positions[:batch_size].contiguous(),
            self.dyn_retrive_index[:batch_size].contiguous(),
            self.dyn_retrive_next_token[:batch_size].contiguous(),
            self.dyn_retrive_next_sibling[:batch_size].contiguous(),
            topk,
            num_steps,
            min(current_token_idx, num_draft_tokens),
            1,  # LOCAL_MASK mode (avoid out-of-bounds at large context)
        )

        # Return draft tokens and tree structure for verification
        draft_tokens = self.dyn_draft_tokens[:batch_size, :current_token_idx]
        retrive_index = self.dyn_retrive_index[:batch_size, :current_token_idx]
        retrive_next_token = self.dyn_retrive_next_token[
            :batch_size, :current_token_idx
        ]
        retrive_next_sibling = self.dyn_retrive_next_sibling[
            :batch_size, :current_token_idx
        ]

        return draft_tokens, retrive_index, retrive_next_token, retrive_next_sibling

    def verify_dynamic_tree(
        self,
        batch_size: int,
        candidates: torch.Tensor,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
        deterministic: bool = True,
        bonus_token_sampling: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Verify draft tokens using tree speculative sampling.

        Uses the SGLang-style tree verification kernel to accept/reject
        draft tokens based on target model probabilities.

        Args:
            batch_size: Number of requests in batch
            candidates: [batch_size, num_draft_tokens] draft token IDs
            target_probs: [batch_size, num_draft_tokens, vocab_size] target probabilities
            draft_probs: [batch_size, num_draft_tokens, vocab_size] draft probabilities (optional)
            deterministic: Use deterministic acceptance (greedy)
            bonus_token_sampling: Sample bonus token when all accepted

        Returns:
            Tuple of (predicts, accept_index, accept_token_num)
        """
        from vllm import _custom_ops as ops

        num_draft_tokens = candidates.shape[1]
        vocab_size = target_probs.shape[-1]

        # Output buffers
        predicts = torch.zeros(
            (batch_size, num_draft_tokens), dtype=torch.int64, device=self.device
        )
        accept_index = torch.zeros(
            (batch_size, num_draft_tokens), dtype=torch.int64, device=self.device
        )
        accept_token_num = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )

        # Use stored tree structure from propose_dynamic_tree
        retrive_index = getattr(self, "_last_retrive_index", None)
        retrive_next_token = getattr(self, "_last_retrive_next_token", None)
        retrive_next_sibling = getattr(self, "_last_retrive_next_sibling", None)

        if retrive_index is None:
            raise RuntimeError(
                "verify_dynamic_tree called without prior propose_dynamic_tree"
            )

        # If no draft probs provided, use uniform distribution
        if draft_probs is None:
            draft_probs = torch.ones_like(target_probs) / vocab_size

        ops.tree_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates.to(torch.int64),
            retrive_index.contiguous(),
            retrive_next_token.contiguous(),
            retrive_next_sibling.contiguous(),
            target_probs.contiguous(),
            draft_probs.contiguous(),
            deterministic,
            bonus_token_sampling,
        )

        return predicts, accept_index, accept_token_num


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
