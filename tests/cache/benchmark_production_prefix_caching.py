#!/usr/bin/env python3
"""
Production Prefix Caching Benchmark for GPT-OSS-120B.

Compares throughput with:
1. Prefix caching DISABLED (baseline)
2. Prefix caching ENABLED (vLLM default)

Uses real model inference, not mocks.
"""

import argparse
import time
import torch
import gc
import os
from typing import List, Dict, Any
from pathlib import Path

# VLLM imports
from vllm import LLM, SamplingParams


def measure_memory() -> Dict[str, float]:
    """Measure current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization_pct": (allocated / total) * 100,
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "utilization_pct": 0}


def generate_shared_prefix_prompts(
    batch_size: int, shared_prefix_tokens: int, unique_suffix_tokens: int
) -> List[str]:
    """
    Generate prompts with shared prefix (simulating system prompt) and unique suffix.

    This simulates production workloads where many requests share a system prompt
    but have unique user queries.
    """
    # Base system prompt (shared across all requests)
    system_prompt = """You are a highly advanced AI assistant specialized in providing 
    comprehensive, accurate, and detailed responses. Your knowledge spans across multiple 
    domains including science, technology, mathematics, history, literature, and current 
    events. When responding, you should:
    
    1. Provide thorough explanations with relevant examples
    2. Break down complex topics into understandable components
    3. Cite sources or reasoning when making factual claims
    4. Consider multiple perspectives on subjective topics
    5. Maintain a professional yet approachable tone
    
    You have been trained on data up to a recent cutoff and can engage in nuanced 
    discussions about a wide range of topics. Your responses should be helpful, 
    harmless, and honest.
    
    """ * (
        shared_prefix_tokens // 100
    )  # Repeat to hit target token count

    prompts = []
    for i in range(batch_size):
        # Unique suffix per request
        unique_query = (
            f"\n\nUser Query #{i}: Please explain the concept of "
            + f"topic_{i} " * (unique_suffix_tokens // 10)
            + "in detail.\n\nAssistant:"
        )

        prompts.append(system_prompt + unique_query)

    return prompts


def run_benchmark(
    model_path: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    shared_prefix_tokens: int = 1000,
    unique_suffix_tokens: int = 100,
    enable_prefix_caching: bool = True,
    warmup_runs: int = 1,
    measurement_runs: int = 3,
    tensor_parallel_size: int = 1,
) -> Dict[str, Any]:
    """Run the benchmark with specified configuration."""

    print(f"\n{'='*70}")
    print(f"PRODUCTION PREFIX CACHING BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Shared Prefix Tokens: {shared_prefix_tokens}")
    print(f"Unique Suffix Tokens: {unique_suffix_tokens}")
    print(f"Prefix Caching: {'ENABLED' if enable_prefix_caching else 'DISABLED'}")
    print(f"Tensor Parallel: {tensor_parallel_size}")
    print(f"{'='*70}\n")

    # Generate prompts
    prompts = generate_shared_prefix_prompts(
        batch_size=batch_size,
        shared_prefix_tokens=shared_prefix_tokens,
        unique_suffix_tokens=unique_suffix_tokens,
    )

    print(f"Generated {len(prompts)} prompts")
    print(f"Approximate prompt length: {len(prompts[0])} chars")

    # Create LLM instance
    print("\nLoading model...")
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.90,
        max_model_len=8192,  # Reasonable context for benchmark
        enforce_eager=False,
        disable_log_stats=True,
    )

    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # Greedy for consistency
        top_p=1.0,
    )

    # Memory after load
    mem_loaded = measure_memory()
    print(
        f"Memory after load: {mem_loaded['allocated_gb']:.2f} GB ({mem_loaded['utilization_pct']:.1f}%)"
    )

    # Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        _ = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        print(f"  Warmup {i+1} complete")

    # Measurement
    print(f"\nMeasurement ({measurement_runs} runs)...")
    latencies = []
    throughputs = []
    token_counts = []

    for run_idx in range(measurement_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        total_output_tokens = sum(
            len(output.outputs[0].token_ids) for output in outputs
        )
        total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)

        throughput = total_output_tokens / total_time
        latency_per_token = (total_time / total_output_tokens) * 1000

        latencies.append(latency_per_token)
        throughputs.append(throughput)
        token_counts.append(total_output_tokens)

        print(
            f"  Run {run_idx + 1}: {throughput:.1f} tok/s, {latency_per_token:.2f} ms/tok, "
            f"{total_output_tokens} tokens generated"
        )

    # Memory after runs
    mem_final = measure_memory()

    # Aggregate results
    results = {
        "config": {
            "model": model_path,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "shared_prefix_tokens": shared_prefix_tokens,
            "unique_suffix_tokens": unique_suffix_tokens,
            "prefix_caching_enabled": enable_prefix_caching,
            "tensor_parallel_size": tensor_parallel_size,
        },
        "metrics": {
            "throughput_tok_per_sec": {
                "mean": sum(throughputs) / len(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
            },
            "latency_ms_per_tok": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
            },
            "total_tokens_generated": sum(token_counts),
        },
        "memory": {
            "after_load_gb": mem_loaded["allocated_gb"],
            "after_runs_gb": mem_final["allocated_gb"],
            "peak_utilization_pct": mem_final["utilization_pct"],
        },
    }

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Production Prefix Caching Benchmark for GPT-OSS-120B"
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/aimo/models/gpt-oss-120b",
        help="Path to model",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max tokens to generate"
    )
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=1000,
        help="Approximate shared prefix token count",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs")
    parser.add_argument(
        "--measurement-runs", type=int, default=3, help="Measurement runs"
    )

    args = parser.parse_args()

    # Run with prefix caching ENABLED
    print("\n" + "=" * 70)
    print("TEST 1: PREFIX CACHING ENABLED")
    print("=" * 70)
    results_enabled = run_benchmark(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
        enable_prefix_caching=True,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        tensor_parallel_size=args.tensor_parallel,
    )

    # Run with prefix caching DISABLED
    print("\n" + "=" * 70)
    print("TEST 2: PREFIX CACHING DISABLED")
    print("=" * 70)
    results_disabled = run_benchmark(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
        enable_prefix_caching=False,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        tensor_parallel_size=args.tensor_parallel,
    )

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    enabled_throughput = results_enabled["metrics"]["throughput_tok_per_sec"]["mean"]
    disabled_throughput = results_disabled["metrics"]["throughput_tok_per_sec"]["mean"]
    speedup = enabled_throughput / disabled_throughput if disabled_throughput > 0 else 0

    print(
        f"\n{'Metric':<30} {'Prefix Caching OFF':<20} {'Prefix Caching ON':<20} {'Speedup'}"
    )
    print("-" * 80)
    print(
        f"{'Throughput (tok/s)':<30} {disabled_throughput:<20.2f} {enabled_throughput:<20.2f} {speedup:.2f}x"
    )
    print(
        f"{'Latency (ms/tok)':<30} {results_disabled['metrics']['latency_ms_per_tok']['mean']:<20.2f} {results_enabled['metrics']['latency_ms_per_tok']['mean']:<20.2f}"
    )
    print(
        f"{'Memory (GB)':<30} {results_disabled['memory']['after_runs_gb']:<20.2f} {results_enabled['memory']['after_runs_gb']:<20.2f}"
    )

    print(f"\n🚀 Prefix Caching provides {speedup:.2f}x speedup!")


if __name__ == "__main__":
    main()
