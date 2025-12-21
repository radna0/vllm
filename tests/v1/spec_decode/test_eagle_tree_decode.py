# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random

import numpy as np
import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_DRAFT_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"
DEFAULT_TREE = "[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]"


def _generate_tokens(
    model: str,
    draft_model: str | None,
    prompts: list[str],
    sampling_params: SamplingParams,
    num_spec_tokens: int,
    spec_tree: str,
    attention_backend: str | None,
) -> list[list[int]]:
    speculative_config = None
    if draft_model is not None:
        speculative_config = {
            "method": "eagle3",
            "model": draft_model,
            "num_speculative_tokens": num_spec_tokens,
            "speculative_token_tree": spec_tree,
            "max_model_len": 2048,
        }

    llm = LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=len(prompts),
        attention_backend=attention_backend,
        speculative_config=speculative_config,
    )

    outputs = llm.generate(prompts, sampling_params)
    token_ids = [output.outputs[0].token_ids for output in outputs]

    del llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()
    return token_ids


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.skipif(
    os.getenv("VLLM_TEST_EAGLE_TREE", "0") != "1",
    reason="Set VLLM_TEST_EAGLE_TREE=1 to run EAGLE tree decode tests.",
)
def test_eagle_tree_greedy_equivalence():
    model = os.getenv("VLLM_EAGLE_TREE_TARGET_MODEL", DEFAULT_MODEL)
    draft_model = os.getenv("VLLM_EAGLE_TREE_DRAFT_MODEL", DEFAULT_DRAFT_MODEL)
    spec_tree = os.getenv("VLLM_EAGLE_TREE", DEFAULT_TREE)
    num_spec_tokens = int(os.getenv("VLLM_EAGLE_TREE_NUM_SPEC", "2"))

    prompts = [
        "Write a short sentence about GPUs.",
        "Summarize speculative decoding in one sentence.",
    ]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=0,
        max_tokens=32,
    )

    baseline = _generate_tokens(
        model=model,
        draft_model=None,
        prompts=prompts,
        sampling_params=sampling_params,
        num_spec_tokens=num_spec_tokens,
        spec_tree=spec_tree,
        attention_backend=None,
    )

    spec = _generate_tokens(
        model=model,
        draft_model=draft_model,
        prompts=prompts,
        sampling_params=sampling_params,
        num_spec_tokens=num_spec_tokens,
        spec_tree=spec_tree,
        attention_backend="TREE_ATTN",
    )

    assert spec == baseline
