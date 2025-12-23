#!/usr/bin/env python3
"""
Comprehensive benchmark script for VLLM speculative decoding validation.

Tests multiple configurations and collects detailed metrics, adapted from
lmdeploy version to maintain compatibility for direct comparison.
"""

import argparse
import json
import time
import torch
import psutil
import os
import gc
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# VLLM imports
try:
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.engine.arg_utils import EngineArgs

    VLLM_AVAILABLE = True
except ImportError:
    print("VLLM not available. Please install VLLM first.")
    VLLM_AVAILABLE = False
    vllm_version = "unknown"


class VLLMBenchmarkRunner:
    """Run benchmarks with detailed metric collection using VLLM."""

    def __init__(
        self, model_path: str, spec_model_path: str = None, output_dir: str = "results"
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is required but not installed")

        self.model_path = model_path
        self.spec_model_path = spec_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # GPU info
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
        else:
            self.gpu_name = "No GPU"
            self.gpu_memory_total = 0

        # VLLM build info
        try:
            from vllm import __version__ as vllm_version

            self.vllm_version = vllm_version
        except ImportError:
            self.vllm_version = "unknown"

    def create_speculative_config(
        self,
        method: str = "eagle",
        num_speculative_tokens: int = 3,
        draft_model_path: str = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Create VLLM speculative config for different methods."""

        if method == "baseline" or not method:
            return None

        spec_config = {
            "num_speculative_tokens": num_speculative_tokens,
        }

        if method in ["eagle", "eagle3"]:
            spec_config.update(
                {
                    "method": method,
                    "model": draft_model_path or self.spec_model_path,
                    "draft_tensor_parallel_size": 1,  # EAGLE requires TP=1
                }
            )
        elif method == "ngram":
            spec_config.update(
                {
                    "method": "ngram",
                    "prompt_lookup_max": kwargs.get("prompt_lookup_max", 4),
                }
            )
        elif method == "suffix":
            spec_config.update(
                {
                    "method": "suffix",
                }
            )
        elif method == "mlp":
            spec_config.update(
                {
                    "model": draft_model_path or self.spec_model_path,
                    "draft_tensor_parallel_size": 1,
                }
            )
        else:
            raise ValueError(f"Unsupported speculation method: {method}")

        return spec_config

    def create_llm_instance(
        self,
        speculative_config: Optional[Dict[str, Any]] = None,
        max_batch_size: int = 8,
        session_len: int = 32768,
        **kwargs,
    ) -> LLM:
        """Create VLLM LLM instance with specified config."""

        # Clean up any existing instances
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VLLM engine args
        llm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": 1,
            "max_model_len": session_len,
            "gpu_memory_utilization": 0.8,  # Conservative memory usage
            "enforce_eager": False,  # Use CUDA graphs when possible
            "disable_log_stats": True,  # Reduce log noise for benchmarking
        }

        # Add speculative config if provided
        if speculative_config:
            llm_kwargs["speculative_config"] = speculative_config

        # Additional engine configurations
        if "swap_space" in kwargs:
            llm_kwargs["swap_space"] = kwargs["swap_space"]
        if "block_size" in kwargs:
            llm_kwargs["block_size"] = kwargs["block_size"]

        try:
            llm = LLM(**llm_kwargs)
            return llm
        except Exception as e:
            print(f"Error creating VLLM instance: {e}")
            raise

    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization_pct": (allocated / self.gpu_memory_total) * 100,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "utilization_pct": 0}

    def generate_prompts(self, batch_size: int, context_length: int) -> List[str]:
        """Generate prompts of specified context length."""

        # Create a base prompt that can be repeated
        base_prompt = "Explain the concept of artificial intelligence and machine learning in detail, covering topics such as neural networks, deep learning, natural language processing, computer vision, and the latest advances in AI research and applications."

        # Calculate how many times we need to repeat the base prompt
        # Roughly 50 tokens per repetition
        repetitions = max(1, context_length // 50)

        full_prompt = base_prompt * repetitions

        # Create batch of identical prompts
        prompts = [full_prompt] * batch_size
        return prompts

    def run_benchmark(
        self,
        llm: LLM,
        prompts: List[str],
        sampling_params: List[SamplingParams],
        warmup_runs: int = 2,
        measurement_runs: int = 2,
        method: str = "baseline",
    ) -> Dict[str, Any]:
        """Run benchmark and collect metrics."""

        # Allow callers to override run counts via environment variables
        try:
            warmup_runs = int(os.getenv("SPEC_SUITE_WARMUP_RUNS", warmup_runs))
            measurement_runs = int(
                os.getenv("SPEC_SUITE_MEASUREMENT_RUNS", measurement_runs)
            )
        except ValueError:
            pass

        # Warmup
        print(f"  Warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Measurement
        print(f"  Measurement ({measurement_runs} runs)...")
        latencies = []
        throughputs = []
        memory_usages = []
        speculative_metrics = []

        for run_idx in range(measurement_runs):
            # Measure memory before
            mem_before = self.measure_memory()

            # Run inference
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            # Measure memory after
            mem_after = self.measure_memory()

            # Calculate core latency/throughput metrics
            total_time = end_time - start_time
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

            latency_per_token = (total_time / total_tokens) * 1000  # ms
            throughput = total_tokens / total_time  # tokens/sec

            latencies.append(latency_per_token)
            throughputs.append(throughput)
            memory_usages.append(mem_after)

            # Collect speculative metrics if available
            run_spec_metrics = {}
            for output in outputs:
                # VLLM may have speculative metrics in output.metadata or similar
                # This is a placeholder - actual VLLM speculative metrics extraction
                # will depend on VLLM's specific API
                if hasattr(output, "metadata") and output.metadata:
                    spec_info = output.metadata.get("speculative_info", {})
                    run_spec_metrics.update(spec_info)

            speculative_metrics.append(run_spec_metrics)

            print(
                f"    Run {run_idx + 1}: {throughput:.1f} tok/s, "
                f"{latency_per_token:.2f} ms/tok, "
                f"{mem_after['allocated_gb']:.2f} GB"
            )

        # Aggregate results
        results: Dict[str, Any] = {
            "latency_ms_per_token": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "values": latencies,
            },
            "throughput_tokens_per_sec": {
                "mean": sum(throughputs) / len(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
                "values": throughputs,
            },
            "memory_gb": {
                "mean": sum(m["allocated_gb"] for m in memory_usages)
                / len(memory_usages),
                "max": max(m["allocated_gb"] for m in memory_usages),
                "utilization_pct": sum(m["utilization_pct"] for m in memory_usages)
                / len(memory_usages),
            },
            "total_runs": measurement_runs,
        }

        # Add speculative metrics if available
        if method != "baseline" and any(speculative_metrics):
            # Placeholder for speculative metrics aggregation
            # This will need to be adapted based on VLLM's actual metric format
            results["speculative_metrics"] = {
                "enabled": True,
                "method": method,
                "runs": speculative_metrics,
            }

        return results

    def run_test_scenario(
        self,
        scenario_name: str,
        batch_size: int,
        context_length: int,
        max_new_tokens: int,
        method: str = "baseline",
        num_spec_tokens: int = 3,
        warmup_runs: int = 2,
        measurement_runs: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a complete test scenario."""

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"  Method: {method.upper() if method != 'baseline' else 'BASELINE'}")
        if method != "baseline":
            print(f"  Speculative tokens: {num_spec_tokens}")
        print(f"{'='*60}")

        # Check for baseline disable flag (same as lmdeploy)
        disable_baseline_env = (
            os.getenv("LMDEPLOY_EAGLE_DISABLE_BASELINE", "").strip().lower()
        )
        disable_baseline = disable_baseline_env in ("1", "true", "yes", "on")

        if disable_baseline and method == "baseline":
            print(
                "LMDEPLOY_EAGLE_DISABLE_BASELINE=1 and method=baseline; "
                "skipping baseline scenario without running the engine."
            )
            # Return stub results to maintain JSON structure consistency
            results = self._create_stub_results(
                scenario_name,
                batch_size,
                context_length,
                max_new_tokens,
                method,
                num_spec_tokens,
            )

            filename = f"{scenario_name.replace(' ', '_').lower()}.json"
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nBaseline scenario skipped; stub results saved to: {filepath}")
            return results

        # Create speculative config
        speculative_config = None
        if method != "baseline":
            try:
                speculative_config = self.create_speculative_config(
                    method=method, num_speculative_tokens=num_spec_tokens, **kwargs
                )
                print(f"  Speculative config: {speculative_config}")
            except Exception as e:
                print(f"  Error creating speculative config: {e}")
                return {"error": str(e), "scenario": scenario_name}

        # Create VLLM instance
        print("Creating VLLM instance...")
        try:
            llm = self.create_llm_instance(
                speculative_config=speculative_config,
                max_batch_size=batch_size,
                session_len=context_length + max_new_tokens,
                **kwargs,
            )
        except Exception as e:
            print(f"  Error creating VLLM instance: {e}")
            return {"error": str(e), "scenario": scenario_name}

        # Generate prompts
        prompts = self.generate_prompts(batch_size, context_length)

        # Create sampling parameters
        sampling_params = self._create_sampling_params(max_new_tokens, batch_size)

        # Run benchmark
        try:
            results = self.run_benchmark(
                llm,
                prompts,
                sampling_params,
                warmup_runs=warmup_runs,
                measurement_runs=measurement_runs,
                method=method,
            )
        except Exception as e:
            print(f"  Error during benchmark: {e}")
            return {"error": str(e), "scenario": scenario_name}
        finally:
            # Clean up
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Add scenario info
        results["scenario"] = {
            "name": scenario_name,
            "batch_size": batch_size,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "method": method,
            "num_spec_tokens": num_spec_tokens if method != "baseline" else 0,
        }

        results["system"] = {
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total,
            "timestamp": datetime.now().isoformat(),
            "vllm_version": self.vllm_version,
        }

        # Save results
        filename = f"{scenario_name.replace(' ', '_').lower()}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        print(f"  Throughput: {results['throughput_tokens_per_sec']['mean']:.1f} tok/s")
        print(f"  Latency: {results['latency_ms_per_token']['mean']:.2f} ms/tok")
        print(f"  Memory: {results['memory_gb']['mean']:.2f} GB")

        # Print speculative metrics if available
        if "speculative_metrics" in results:
            spec = results["speculative_metrics"]
            print(
                f"  Speculative: method={spec.get('method', 'unknown')}, enabled={spec.get('enabled', False)}"
            )

        return results

    def _create_sampling_params(
        self, max_new_tokens: int, batch_size: int
    ) -> List[SamplingParams]:
        """Create VLLM sampling parameters."""

        # Allow override via environment (same as lmdeploy)
        temperature = float(os.getenv("SPEC_SUITE_TEMPERATURE", "0.0"))
        top_k = int(os.getenv("SPEC_SUITE_TOP_K", "20"))
        top_p = float(os.getenv("SPEC_SUITE_TOP_P", "0.8"))
        min_p = float(os.getenv("SPEC_SUITE_MIN_P", "0.0"))

        # Check for performance mode
        perf_mode_env = os.getenv("LMDEPLOY_EAGLE_PERF_MODE", "").strip().lower()
        perf_mode = perf_mode_env in ("1", "true", "yes", "on")

        if perf_mode:
            temperature = 0.0
            top_k = 0
            top_p = 1.0
            min_p = 0.0

        # Enable sampling when any stochastic knob is active
        do_sample = temperature > 0.0 or top_p < 1.0 or top_k > 0 or min_p > 0.0

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_k=top_k if do_sample else -1,
            top_p=top_p if do_sample else 1.0,
            skip_special_tokens=False,  # Keep all tokens for consistency
        )

        return [sampling_params] * batch_size

    def _create_stub_results(
        self,
        scenario_name: str,
        batch_size: int,
        context_length: int,
        max_new_tokens: int,
        method: str,
        num_spec_tokens: int,
    ) -> Dict[str, Any]:
        """Create stub results for skipped baseline scenarios."""

        micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
        micro_steps = None
        if micro_steps_env:
            try:
                v = int(micro_steps_env)
                if v > 0:
                    micro_steps = v
            except ValueError:
                micro_steps = None

        perf_mode_env = os.getenv("LMDEPLOY_EAGLE_PERF_MODE", "").strip().lower()
        perf_mode = perf_mode_env in ("1", "true", "yes", "on")

        results: Dict[str, Any] = {
            "latency_ms_per_token": {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "values": [],
            },
            "throughput_tokens_per_sec": {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "values": [],
            },
            "memory_gb": {
                "mean": 0.0,
                "max": 0.0,
                "utilization_pct": 0.0,
            },
            "total_runs": 0,
            "scenario": {
                "name": scenario_name,
                "batch_size": batch_size,
                "context_length": context_length,
                "max_new_tokens": max_new_tokens,
                "method": method,
                "num_spec_tokens": num_spec_tokens if method != "baseline" else 0,
                "perf_mode": perf_mode,
                "micro_run": micro_steps is not None,
            },
            "system": {
                "gpu_name": self.gpu_name,
                "gpu_memory_total_gb": self.gpu_memory_total,
                "timestamp": datetime.now().isoformat(),
                "vllm_version": self.vllm_version,
            },
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLLM speculative decoding performance"
    )
    parser.add_argument("--model-path", required=True, help="Path to main model")
    parser.add_argument("--spec-model-path", help="Path to speculation model (EAGLE3)")
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--method",
        choices=["all", "baseline", "eagle", "eagle3", "ngram", "suffix", "mlp"],
        default="all",
        help="Which speculation method(s) to test",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "baseline", "single", "batch", "large-context", "stress"],
        default="all",
        help="Which scenario(s) to run",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per scenario",
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=2,
        help="Number of measurement runs per scenario",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences per iteration",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (non-speculative) benchmark runs. Only run speculative methods.",
    )

    args = parser.parse_args()

    print(f"VLLM Benchmark Suite")
    print(
        f"VLLM Version: {VLLMBenchmarkRunner(args.model_path, args.spec_model_path, args.output_dir).vllm_version}"
    )

    runner = VLLMBenchmarkRunner(args.model_path, args.spec_model_path, args.output_dir)

    scenarios = []
    methods = []

    # Determine which methods to test
    if args.method == "all":
        methods = ["baseline", "eagle", "eagle3", "ngram", "suffix"]
    else:
        methods = [args.method]

    # Skip baseline if requested
    if args.skip_baseline:
        methods = [m for m in methods if m != "baseline"]
        print("Skipping baseline benchmark (--skip-baseline flag set)")

    # Baseline scenarios (no speculation)
    if args.scenario in ["all", "baseline"]:
        # Only add baseline scenarios if not skipping baseline
        if not args.skip_baseline:
            scenarios.extend(
                [
                    {
                        "scenario_name": f"Baseline_Single_Context8K",
                        "batch_size": 1,
                        "context_length": 8192,
                        "max_new_tokens": 8192,
                        "method": "baseline",
                    }
                ]
            )

        # Add speculative baseline scenarios for each method
        for method in methods:
            if method == "baseline":
                continue

            token_counts = [2, 3, 4, 5] if method in ["eagle", "eagle3"] else [5]
            for num_tokens in token_counts:
                scenarios.extend(
                    [
                        {
                            "scenario_name": f"Speculative_{method.capitalize()}_Single_Context8K_{num_tokens}tokens",
                            "batch_size": 1,
                            "context_length": 8192,
                            "max_new_tokens": 8192,
                            "method": method,
                            "num_spec_tokens": num_tokens,
                        }
                    ]
                )

    # Single batch scenarios
    if args.scenario in ["all", "single"]:
        micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
        micro_steps = None
        if micro_steps_env:
            try:
                v = int(micro_steps_env)
                if v > 0:
                    micro_steps = v
            except ValueError:
                micro_steps = None

        # Baseline single 32K - only if not skipping baseline
        if not args.skip_baseline:
            scenarios.append(
                {
                    "scenario_name": "Baseline_Single_Context32K",
                    "batch_size": 1,
                    "context_length": 32768,
                    "max_new_tokens": micro_steps if micro_steps is not None else 32768,
                    "method": "baseline",
                }
            )

        # Speculative single 32K scenarios
        for method in methods:
            if method == "baseline":
                continue

            token_counts = [2, 3, 4, 5] if method in ["eagle", "eagle3"] else [5]
            for num_tokens in token_counts:
                scenarios.append(
                    {
                        "scenario_name": f"Speculative_{method.capitalize()}_Single_Context32K_{num_tokens}tokens",
                        "batch_size": 1,
                        "context_length": 32768,
                        "max_new_tokens": (
                            micro_steps if micro_steps is not None else 32768
                        ),
                        "method": method,
                        "num_spec_tokens": num_tokens,
                    }
                )

    # Batch scenarios
    if args.scenario in ["all", "batch"]:
        # Baseline batch - only if not skipping baseline
        if not args.skip_baseline:
            scenarios.append(
                {
                    "scenario_name": "Baseline_Batch8_Context8K",
                    "batch_size": 8,
                    "context_length": 8192,
                    "max_new_tokens": 8192,
                    "method": "baseline",
                }
            )

        # Speculative batch scenarios
        for method in methods:
            if method == "b                                     ":
                continue

            # Use typical token counts for batch scenarios
            token_counts = [3, 5] if method in ["eagle", "eagle3"] else [5]
            for num_tokens in token_counts:
                scenarios.append(
                    {
                        "scenario_name": f"Speculative_{method.capitalize()}_Batch8_Context8K_{num_tokens}tokens",
                        "batch_size": 8,
                        "context_length": 8192,
                        "max_new_tokens": 8192,
                        "method": method,
                        "num_spec_tokens": num_tokens,
                    }
                )

    # Large context scenarios
    if args.scenario in ["all", "large-context"]:
        micro_steps_env = os.getenv("LMDEPLOY_EAGLE_MICRO_STEPS", "").strip()
        micro_steps = None
        if micro_steps_env:
            try:
                v = int(micro_steps_env)
                if v > 0:
                    micro_steps = v
            except ValueError:
                micro_steps = None

        # Baseline large context - only if not skipping baseline
        if not args.skip_baseline:
            scenarios.append(
                {
                    "scenario_name": "Baseline_Batch4_Context16K",
                    "batch_size": 4,
                    "context_length": 16384,
                    "max_new_tokens": micro_steps if micro_steps is not None else 16384,
                    "method": "baseline",
                }
            )

        # Speculative large context scenarios
        for method in methods:
            if method == "baseline":
                continue

            token_counts = [2, 3, 4, 5] if method in ["eagle", "eagle3"] else [5]
            for num_tokens in token_counts:
                scenarios.append(
                    {
                        "scenario_name": f"Speculative_{method.capitalize()}_Batch4_Context16K_{num_tokens}tokens",
                        "batch_size": 4,
                        "context_length": 16384,
                        "max_new_tokens": (
                            micro_steps if micro_steps is not None else 16384
                        ),
                        "method": method,
                        "num_spec_tokens": num_tokens,
                    }
                )

    # Stress test scenarios
    if args.scenario in ["all", "stress"]:
        # Stress baseline - only if not skipping baseline
        if not args.skip_baseline:
            scenarios.extend(
                [
                    {
                        "scenario_name": "Baseline_Batch8_LongGen",
                        "batch_size": 8,
                        "context_length": 16384,
                        "max_new_tokens": 16384,
                        "method": "baseline",
                    }
                ]
            )

        for method in methods:
            if method == "baseline":
                continue
            scenarios.append(
                {
                    "scenario_name": f"Speculative_{method.capitalize()}_Batch8_LongGen_3tokens",
                    "batch_size": 8,
                    "context_length": 16384,
                    "max_new_tokens": 16384,
                    "method": method,
                    "num_spec_tokens": 3,
                }
            )

    # Run all scenarios
    all_results = []
    for scenario in scenarios:
        result = runner.run_test_scenario(
            **scenario,
            warmup_runs=args.warmup_runs,
            measurement_runs=args.measurement_runs,
        )
        if "error" not in result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        name = result["scenario"]["name"]
        method = result["scenario"]["method"]
        throughput = result["throughput_tokens_per_sec"]["mean"]
        latency = result["latency_ms_per_token"]["mean"]
        print(f"{name:50s} {throughput:8.1f} tok/s  {latency:6.2f} ms/tok  [{method}]")


if __name__ == "__main__":
    main()
