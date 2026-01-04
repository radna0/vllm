import torch
from typing import Optional

try:
    from . import _C
except ImportError:
    import warnings

    warnings.warn(
        "vllm_eagle C++ extension not installed. Kernels will be unavailable."
    )
    _C = None


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")
    _C.build_tree_kernel_efficient(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")
    _C.reconstruct_indices_from_tree_mask(
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        draft_token_num,
    )


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    deterministic: bool,
    bonus_token_sampling: bool = True,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    threshold_single: float = 0.0,
    threshold_acc: float = 0.0,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")

    # Generate random samples if not provided and not deterministic
    # Check if we need to generate them. The C++ kernel takes them regardless of deterministic flag?
    # Yes, looking at the kernel, it takes them as arguments.

    if uniform_samples is None:
        uniform_samples = torch.empty_like(
            candidates, dtype=target_probs.dtype
        ).uniform_(0.0, 1.0)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.empty_like(
            accept_index, dtype=target_probs.dtype
        ).uniform_(0.0, 1.0)

    _C.tree_speculative_sampling_target_only(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    )


def apply_logit_filters(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")
    _C.apply_logit_filters(logits, top_k, top_p, min_p)


def fused_gumbel_sample(
    out_tokens: torch.Tensor,
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    temperatures: torch.Tensor,
    seed: int,
    offset: int,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")
    _C.fused_gumbel_sample(
        out_tokens, logits, top_k, top_p, min_p, temperatures, seed, offset
    )


def fused_gumbel_sample_warp_optimized(
    out_tokens: torch.Tensor,
    logits: torch.Tensor,
    seed: int,
    offset: int,
    min_p: torch.Tensor,
    temperatures: torch.Tensor,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")
    _C.fused_gumbel_sample_warp_optimized(
        out_tokens, logits, seed, offset, min_p, temperatures
    )


def fused_eagle_metadata_update(
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    is_terminated: torch.Tensor,
    max_model_len: int,
    block_size: int,
) -> None:
    if _C is None:
        raise ImportError("vllm_eagle C++ extension is not available")
    _C.fused_eagle_metadata_update(
        positions,
        seq_lens,
        slot_mapping,
        block_table,
        is_terminated,
        max_model_len,
        block_size,
    )
