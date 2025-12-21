# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform


def _cpu_build_indices(num_draft_tokens, cu_num_scheduled_tokens):
    num_sampled_tokens = num_draft_tokens + 1
    cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
    cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
    bonus_logits_indices = cu_num_sampled_tokens - 1

    logits_indices = []
    target_logits_indices = []
    for idx, num_draft in enumerate(num_draft_tokens):
        num_sampled = num_draft + 1
        logits_start = cu_num_scheduled_tokens[idx] - num_sampled
        logits_indices.extend(range(logits_start, logits_start + num_sampled))
        target_start = cu_num_sampled_tokens[idx] - num_sampled
        target_logits_indices.extend(range(target_start, target_start + num_draft))

    return (
        cu_num_draft_tokens,
        cu_num_sampled_tokens,
        np.array(logits_indices, dtype=np.int32),
        np.array(target_logits_indices, dtype=np.int32),
        bonus_logits_indices,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
def test_build_spec_decode_indices_matches_cpu():
    if not hasattr(torch.ops.vllm, "build_spec_decode_indices"):
        pytest.skip("build_spec_decode_indices op not available")

    num_draft_tokens = np.array([3, 0, 2, 1], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([4, 104, 107, 109], dtype=np.int32)

    (
        cu_num_draft_ref,
        cu_num_sampled_ref,
        logits_indices_ref,
        target_logits_indices_ref,
        bonus_logits_indices_ref,
    ) = _cpu_build_indices(num_draft_tokens, cu_num_scheduled_tokens)

    device = torch.device("cuda")
    num_draft_gpu = torch.from_numpy(num_draft_tokens).to(device)
    cu_num_sched_gpu = torch.from_numpy(cu_num_scheduled_tokens).to(device)

    total_draft = int(num_draft_tokens.sum())
    total_sampled = total_draft + num_draft_tokens.shape[0]

    cu_num_draft_out = torch.empty_like(num_draft_gpu)
    cu_num_sampled_out = torch.empty_like(num_draft_gpu)
    logits_indices_out = torch.empty((total_sampled,), device=device, dtype=torch.int32)
    target_logits_indices_out = torch.empty(
        (total_draft,), device=device, dtype=torch.int32
    )
    bonus_logits_indices_out = torch.empty_like(num_draft_gpu)

    torch.ops.vllm.build_spec_decode_indices(
        num_draft_gpu,
        cu_num_sched_gpu,
        cu_num_draft_out,
        cu_num_sampled_out,
        logits_indices_out,
        target_logits_indices_out,
        bonus_logits_indices_out,
    )

    assert torch.equal(cu_num_draft_out.cpu(), torch.from_numpy(cu_num_draft_ref))
    assert torch.equal(
        cu_num_sampled_out.cpu(), torch.from_numpy(cu_num_sampled_ref)
    )
    assert torch.equal(
        logits_indices_out.cpu(), torch.from_numpy(logits_indices_ref)
    )
    assert torch.equal(
        target_logits_indices_out.cpu(),
        torch.from_numpy(target_logits_indices_ref),
    )
    assert torch.equal(
        bonus_logits_indices_out.cpu(),
        torch.from_numpy(bonus_logits_indices_ref),
    )
