# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import time

import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

DEFAULT_PROMPT = "Write a short paragraph about GPUs."
DEFAULT_TREE = "[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]"


def _build_llm(args: argparse.Namespace, speculative_config: dict | None) -> LLM:
    return LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        attention_backend=args.attention_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
    )


def _run_once(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> int:
    outputs = llm.generate(prompts, sampling_params)
    token_count = 0
    for output in outputs:
        token_count += len(output.outputs[0].token_ids)
    return token_count


def _benchmark(args: argparse.Namespace, speculative_config: dict | None) -> tuple[float, int]:
    prompts = [args.prompt] * args.batch_size
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    llm = _build_llm(args, speculative_config)

    for _ in range(args.warmup_iters):
        _run_once(llm, prompts, sampling_params)

    total_tokens = 0
    total_time_s = 0.0
    for _ in range(args.num_iters):
        start = time.perf_counter()
        total_tokens += _run_once(llm, prompts, sampling_params)
        total_time_s += time.perf_counter() - start

    tokens_per_s = total_tokens / total_time_s if total_time_s > 0 else 0.0
    return tokens_per_s, total_tokens


def main() -> None:
    if not current_platform.is_cuda():
        raise SystemExit("CUDA is required for this benchmark.")

    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE3 tree-based speculative decoding."
    )
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument(
        "--draft-model", required=True, help="EAGLE3 draft model path."
    )
    parser.add_argument(
        "--method",
        default="eagle3",
        choices=("eagle", "eagle3"),
        help="Speculative method to use.",
    )
    parser.add_argument(
        "--num-spec-tokens",
        type=int,
        default=2,
        help="Number of speculative tokens (tree depth).",
    )
    parser.add_argument(
        "--spec-token-tree",
        default=DEFAULT_TREE,
        help="Speculative token tree (string literal).",
    )
    parser.add_argument(
        "--attention-backend",
        default="TREE_ATTN",
        help="Attention backend (e.g. TREE_ATTN or TRTLLM_ATTN).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline (no spec decode) for comparison.",
    )
    args = parser.parse_args()

    spec_config = {
        "method": args.method,
        "model": args.draft_model,
        "num_speculative_tokens": args.num_spec_tokens,
        "speculative_token_tree": args.spec_token_tree,
        "max_model_len": args.max_model_len,
    }

    tokens_per_s, total_tokens = _benchmark(args, spec_config)
    print("=== EAGLE3 tree decode ===")
    print(f"tokens/sec: {tokens_per_s:.2f}")
    print(f"total tokens: {total_tokens}")

    if args.compare_baseline:
        baseline_tps, baseline_tokens = _benchmark(args, None)
        print("=== Baseline (no spec decode) ===")
        print(f"tokens/sec: {baseline_tps:.2f}")
        print(f"total tokens: {baseline_tokens}")
        if baseline_tps > 0:
            print(f"speedup: {tokens_per_s / baseline_tps:.2f}x")


if __name__ == "__main__":
    main()
