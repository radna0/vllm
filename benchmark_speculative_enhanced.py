#!/usr/bin/env python3
"""
Enhanced benchmark script for VLLM speculative decoding validation.

Supports 3 modes:
1. baseline - Standard VLLM without speculation
2. vllm_native - VLLM's built-in speculative methods (EAGLE, EAGLE3, ngram, suffix)
3. arctic_inference - Arctic Inference with Arctic LSTM Speculator

Uses OpenAI API server for VLLM and Arctic Inference for consistent benchmarking.
"""

import argparse
import json
import time
import os
import subprocess
import signal
import aiohttp
import asyncio
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import required dependencies with error handling
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available, using Python alternatives")
    np = None

try:
    import torch
except ImportError:
    print("Warning: torch not available, GPU info will be limited")
    torch = None

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp required for server communication")
    aiohttp = None

try:
    import openai
except ImportError:
    print("Error: openai package required for API client")
    openai = None

try:
    import arctic_inference
    ARCTIC_AVAILABLE = True
except ImportError:
    print("Warning: arctic_inference not available")
    ARCTIC_AVAILABLE = False


class EnhancedVLLMBenchmark:
    """Run benchmarks with multiple speculation modes."""

    def __init__(
        self, model_path: str, spec_model_path: str = None, output_dir: str = "results"
    ):
        self.model_path = model_path
        self.spec_model_path = spec_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.server_process = None
        self.server_port = 8000
        
        # GPU info (if torch available)
        if torch and torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
        else:
            self.gpu_name = "No GPU"
            self.gpu_memory_total = 0

        # VLLM version
        self.vllm_version = self._get_vllm_version()

    def _get_vllm_version(self) -> str:
        """Get VLLM version from installed package."""
        try:
            import vllm
            return getattr(vllm, '__version__', 'unknown')
        except (ImportError, AttributeError):
            return "unknown"

    def create_vllm_server_command(
        self,
        method: str = "eagle3",
        num_spec_tokens: int = 3,
        **kwargs
    ) -> List[str]:
        """Create VLLM serve command with specific speculative config."""
        
        # Base command components
        cmd_base = [
            "vllm", "serve", self.model_path,
            "--host", "0.0.0.0", 
            "--port", str(self.server_port),
            "--gpu-memory-utilization", "0.85",
        ]

        # Method-specific configurations
        if method == "baseline":
            return cmd_base
            
        elif method in ["eagle", "eagle3"]:
            if not self.spec_model_path:
                raise ValueError(f"Spec model path required for {method} method")
                
            spec_config = {
                "method": method,
                "model": self.spec_model_path,
                "num_speculative_tokens": num_spec_tokens,
                "draft_tensor_parallel_size": 1,
            }
            
        elif method == "ngram":
            spec_config = {
                "method": "ngram", 
                "num_speculative_tokens": num_spec_tokens,
                "prompt_lookup_max": kwargs.get("prompt_lookup_max", 4),
            }
            
        elif method == "suffix":
            spec_config = {
                "method": "suffix",
                "num_speculative_tokens": num_spec_tokens,
            }
            
        else:
            raise ValueError(f"Unsupported VLLM speculation method: {method}")

        # Add speculative config to command
        cmd_base.extend([
            "--speculative-config", json.dumps(spec_config)
        ])
        
        return cmd_base

    def create_arctic_server_command(
        self,
        num_spec_tokens: int = 3,
        **kwargs
    ) -> List[str]:
        """Create Arctic Inference server command."""
        
        cmd = [
            "vllm", "serve", "openai/gpt-oss-120b",
            "--host", "0.0.0.0",
            "--port", str(self.server_port),
            "--tensor-parallel-size", "4",
        ]

        # Arctic speculative config
        arctic_config = {
            "method": "arctic",
            "model": "/workspace/aimo/models/Arctic-LSTM-Speculator-gpt-oss-120b",
            "num_speculative_tokens": num_spec_tokens,
            "disable_by_batch_size": 64,
            "arctic_inference": {
                "use_plugin": True,
                "model_path": "openai/gpt-oss-120b"
            }
        }

        cmd.extend([
            "--speculative-config", json.dumps(arctic_config)
        ])
        
        return cmd

    def start_server(self, mode: str = "baseline", method: str = None, num_spec_tokens: int = 3, **kwargs) -> bool:
        """Start server with specific configuration based on mode."""
        
        if self.server_process and self.server_process.poll() is None:
            print("Server is already running")
            return True
            
        try:
            if mode == "arctic_inference":
                cmd = self.create_arctic_server_command(num_spec_tokens, **kwargs)
            elif mode == "vllm_native":
                cmd = self.create_vllm_server_command(method, num_spec_tokens, **kwargs)
            else:  # baseline
                cmd = self.create_vllm_server_command("baseline", 0, **kwargs)
                
            print(f"Starting server: {' '.join(cmd)}")
            
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )
            
            # Wait for server to be ready
            if self.wait_for_server_ready_sync():
                print("Server started successfully")
                return True
            else:
                print("Failed to start server")
                self.stop_server()
                return False
                
        except Exception as e:
            print(f"Error starting server: {e}")
            return False

    def wait_for_server_ready_sync(self, timeout: int = 300) -> bool:
        """Synchronous server readiness check."""
        start_time = time.time()
        health_url = f"http://localhost:{self.server_port}/health"
        
        while time.time() - start_time < timeout:
            try:
                if aiohttp:
                    async def check_health():
                        async with aiohttp.ClientSession() as session:
                            async with session.get(health_url, timeout=5) as response:
                                return response.status == 200
                    return asyncio.run(check_health())
                else:
                    # Fallback to simple HTTP request if aiohttp not available
                    import urllib.request
                    try:
                        with urllib.request.urlopen(health_url, timeout=5) as response:
                            return response.getcode() == 200
                    except Exception:
                        pass
                
            except Exception:
                pass
                
            time.sleep(2)
            
        return False

    def fetch_spec_decode_metrics(self) -> Dict[str, Any]:
        """Fetch spec decode metrics from the /metrics endpoint."""
        metrics_url = f"http://localhost:{self.server_port}/metrics"
        try:
            with urllib.request.urlopen(metrics_url, timeout=5) as response:
                payload = response.read().decode("utf-8")
        except Exception:
            return {}

        wanted = {
            "vllm:spec_decode_num_drafts_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
        }
        values: Dict[str, float] = {}
        for line in payload.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            name, value = parts
            if name in wanted:
                try:
                    values[name] = float(value)
                except ValueError:
                    continue

        if not values:
            return {}

        draft_tokens = values.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
        accepted_tokens = values.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
        num_drafts = values.get("vllm:spec_decode_num_drafts_total", 0.0)
        acceptance_rate = (
            (accepted_tokens / draft_tokens) * 100.0 if draft_tokens > 0 else None
        )
        mean_accept_len = (
            1.0 + (accepted_tokens / num_drafts) if num_drafts > 0 else None
        )

        return {
            "raw": values,
            "acceptance_rate_pct": acceptance_rate,
            "mean_accept_len": mean_accept_len,
        }

    def stop_server(self):
        """Stop server gracefully."""
        if self.server_process and self.server_process.poll() is None:
            print("Stopping server...")
            
            # Try graceful shutdown first
            try:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("Server did not terminate gracefully, forcing...")
                    self.server_process.kill()
                    self.server_process.wait()
            except Exception as e:
                print(f"Error stopping server: {e}")
                
        self.server_process = None
        print("Server stopped")

    def generate_prompts(
        self, 
        batch_size: int, 
        context_length: int
    ) -> List[str]:
        """Generate prompts of specified context length."""
        
        # Create a base prompt that can be repeated
        base_prompt = "Explain the concept of quantum computing in detail, covering topics such as neural networks, deep learning, natural language processing, computer vision, and latest advances in AI research and applications."
        
        # Calculate how many times we need to repeat base prompt
        # Roughly 50 tokens per repetition
        repetitions = max(1, context_length // 50)
        
        full_prompt = base_prompt * repetitions
        
        # Create batch of identical prompts
        prompts = [full_prompt] * batch_size
        return prompts

    def run_api_benchmark_sync(
        self,
        prompts: List[str],
        max_new_tokens: int,
        warmup_runs: int = 2,
        measurement_runs: int = 2,
    ) -> Dict[str, Any]:
        """Run benchmark using OpenAI API client synchronously."""
        
        if not openai:
            return {"error": "OpenAI client not available"}
            
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{self.server_port}/v1"
        )

        # Override environment variables like in lmdeploy
        try:
            warmup_runs = int(os.getenv("SPEC_SUITE_WARMUP_RUNS", str(warmup_runs)))
            measurement_runs = int(
                os.getenv("SPEC_SUITE_MEASUREMENT_RUNS", str(measurement_runs))
            )
        except ValueError:
                pass

        # Warmup runs
        print(f"  API Warmup ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            try:
                client.completions.create(
                    model="default",
                    prompt=prompts[0],
                    max_tokens=max_new_tokens,
                    temperature=0.0
                )
            except Exception as e:
                print(f"    Warmup error: {e}")

        # Measurement runs
        print(f"  API Measurement ({measurement_runs} runs)...")
        latencies = []
        throughputs = []
        all_responses = []
        
        start_time = time.time()
        
        for run_idx in range(measurement_runs):
            for prompt_idx, prompt in enumerate(prompts):
                try:
                    request_start = time.time()
                    
                    response = client.completions.create(
                        model="default",
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=0.0,
                        stream=False
                    )
                    
                    request_end = time.time()
                    total_time = request_end - request_start
                    
                    # Extract metrics from response
                    generated_text = response.choices[0].text
                    # Count tokens roughly (approximate)
                    output_tokens = len(generated_text.split())
                    
                    if output_tokens > 0:
                        latency_per_token = (total_time * 1000) / output_tokens  # ms
                        throughput = output_tokens / total_time  # tokens/sec
                        
                        latencies.append(latency_per_token)
                        throughputs.append(throughput)
                        
                        all_responses.append({
                            "text": generated_text,
                            "output_tokens": output_tokens,
                            "latency": total_time,
                            "latency_per_token": latency_per_token,
                            "throughput": throughput
                        })
                    
                    print(
                        f"    Run {run_idx + 1}, Prompt {prompt_idx + 1}: "
                        f"{throughput:.1f} tok/s, {latency_per_token:.2f} ms/tok"
                    )
                    
                except Exception as e:
                    print(f"    API error: {e}")
                    
        end_time = time.time()
        total_benchmark_time = end_time - start_time
        
        # Aggregate results
        if np:
            latency_stats = {
                "mean": float(np.mean(latencies)) if latencies else 0.0,
                "min": float(np.min(latencies)) if latencies else 0.0,
                "max": float(np.max(latencies)) if latencies else 0.0,
                "values": latencies,
            }
            throughput_stats = {
                "mean": float(np.mean(throughputs)) if throughputs else 0.0,
                "min": float(np.min(throughputs)) if throughputs else 0.0,
                "max": float(np.max(throughputs)) if throughputs else 0.0,
                "values": throughputs,
            }
        else:
            # Fallback to Python built-ins
            def mean(lst):
                return sum(lst) / len(lst) if lst else 0.0
                
            latency_stats = {
                "mean": mean(latencies),
                "min": min(latencies) if latencies else 0.0,
                "max": max(latencies) if latencies else 0.0,
                "values": latencies,
            }
            throughput_stats = {
                "mean": mean(throughputs),
                "min": min(throughputs) if throughputs else 0.0,
                "max": max(throughputs) if throughputs else 0.0,
                "values": throughputs,
            }

        results = {
            "latency_ms_per_token": latency_stats,
            "throughput_tokens_per_sec": throughput_stats,
            "memory_gb": {
                "mean": 0.0,  # Server memory tracking not available via API
                "max": 0.0,
                "utilization_pct": 0.0,
            },
            "total_runs": measurement_runs,
            "total_time": total_benchmark_time,
            "api_responses": all_responses,
        }

        return results

    def run_test_scenario(
        self,
        scenario_name: str,
        batch_size: int,
        context_length: int,
        max_new_tokens: int,
        mode: str = "baseline",
        method: str = None,
        num_spec_tokens: int = 3,
        warmup_runs: int = 2,
        measurement_runs: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a complete test scenario."""

        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"  Mode: {mode}")
        print(f"  Method: {method.upper() if method else 'BASELINE'}")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Max new tokens: {max_new_tokens}")
        if method and method != "baseline":
            print(f"  Speculative tokens: {num_spec_tokens}")
        print(f"{'='*60}")

        # Check for baseline disable flag (same as lmdeploy)
        disable_baseline_env = os.getenv(
            "LMDEPLOY_EAGLE_DISABLE_BASELINE", ""
        ).strip().lower()
        disable_baseline = disable_baseline_env in ("1", "true", "yes", "on")
        
        if disable_baseline and mode == "baseline":
            print(
                "LMDEPLOY_EAGLE_DISABLE_BASELINE=1 and mode=baseline; "
                "skipping baseline scenario without running the engine."
            )
            results = self._create_stub_results(
                scenario_name, batch_size, context_length, max_new_tokens, 
                mode, method, num_spec_tokens
            )
            
            filename = f"{scenario_name.replace(' ', '_').lower()}.json"
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nBaseline scenario skipped; stub results saved to: {filepath}")
            return results

        # Start server
        print("Starting server...")
        if not self.start_server(mode, method, num_spec_tokens, **kwargs):
            return {"error": "Failed to start server", "scenario": scenario_name}

        spec_decode_metrics: Dict[str, Any] = {}
        try:
            # Generate prompts
            prompts = self.generate_prompts(batch_size, context_length)

            # Run benchmark
            results = self.run_api_benchmark_sync(
                    prompts, max_new_tokens, warmup_runs, measurement_runs
                )
            spec_decode_metrics = self.fetch_spec_decode_metrics()

        except Exception as e:
            print(f"Error during benchmark: {e}")
            results = {"error": str(e), "scenario": scenario_name}
            
        finally:
            # Stop server
            self.stop_server()

        # Add scenario info
        results["scenario"] = {
            "name": scenario_name,
            "batch_size": batch_size,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "mode": mode,
            "method": method or "baseline",
            "num_spec_tokens": num_spec_tokens if method and method != "baseline" else 0,
        }

        results["system"] = {
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total,
            "timestamp": datetime.now().isoformat(),
            "vllm_version": self.vllm_version,
            "server_port": self.server_port,
        }
        if spec_decode_metrics:
            results["spec_decode_metrics"] = spec_decode_metrics

        # Save results
        filename = f"{scenario_name.replace(' ', '_').lower()}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        if "latency_ms_per_token" in results:
            print(f"  Throughput: {results['latency_ms_per_token']['mean']:.1f} tok/s")
            print(f"  Latency: {results['latency_ms_per_token']['mean']:.2f} ms/tok")
        if "memory_gb" in results:
            print(f"  Memory: {results['memory_gb']['mean']:.2f} GB")

        return results

    def _create_stub_results(
        self, 
        scenario_name: str, 
        batch_size: int, 
        context_length: int, 
        max_new_tokens: int, 
        mode: str,
        method: str,
        num_spec_tokens: int
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

        results = {
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
                "mode": mode,
                "method": method,
                "num_spec_tokens": num_spec_tokens if method and method != "baseline" else 0,
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
        description="Enhanced VLLM speculative decoding benchmark with Arctic Inference support"
    )
    parser.add_argument("--model-path", required=True, help="Path to main model")
    parser.add_argument("--spec-model-path", help="Path to speculation model (EAGLE3)")
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "baseline", "vllm_native", "arctic_inference"],
        default="all",
        help="Which benchmark mode(s) to test",
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "eagle", "eagle3", "ngram", "suffix"],
        help="VLLM speculative method (for vllm_native mode)",
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
        help="Measurement runs per scenario",
    )

    args = parser.parse_args()

    print(f"Enhanced VLLM Benchmark Suite")
    print(f"VLLM Version: {EnhancedVLLMBenchmark(args.model_path, args.spec_model_path, args.output_dir).vllm_version}")

    runner = EnhancedVLLMBenchmark(args.model_path, args.spec_model_path, args.output_dir)

    scenarios = []

    # Determine which modes to test
    if args.mode == "all":
        modes = ["baseline", "vllm_native", "arctic_inference"]
    else:
        modes = [args.mode]

    # Baseline scenarios (no speculation)
    if args.scenario in ["all", "baseline"]:
        scenarios.extend([
            {
                "scenario_name": "Baseline_Single_Context8K",
                "batch_size": 1,
                "context_length": 8192,
                "max_new_tokens": 8192,
                "mode": "baseline",
            }
        ])

    # VLLM Native scenarios
    if args.scenario in ["all", "single"]:
        modes_to_test = [m for m in modes if m != "baseline"]
        
        for mode in modes_to_test:
            if not args.method:
                methods = ["eagle", "eagle3", "ngram", "suffix"]
            else:
                methods = [args.method]
                
            for method in methods:
                scenarios.extend([
                    {
                        "scenario_name": f"VLLM_Native_{method.upper()}_Single_Context8K_3tokens",
                        "batch_size": 1,
                        "context_length": 8192,
                        "max_new_tokens": 8192,
                        "mode": "vllm_native",
                        "method": method,
                        "num_spec_tokens": 3,
                    }
                ])
                    
    # Batch scenarios
    if args.scenario in ["all", "batch"]:
        modes_to_test = [m for m in modes if m != "baseline"]
        
        for mode in modes_to_test:
            if not args.method:
                methods = ["eagle", "eagle3", "ngram", "suffix"]
            else:
                methods = [args.method]
                
            for method in methods:
                scenarios.extend([
                    {
                        "scenario_name": f"VLLM_Native_{method.upper()}_Batch8_Context8K_3tokens",
                        "batch_size": 8,
                        "context_length": 8192,
                        "max_new_tokens": 8192,
                        "mode": "vllm_native",
                        "method": method,
                        "num_spec_tokens": 3,
                    }
                ])

    # Arctic Inference scenarios
    if args.scenario in ["all", "single"] and ARCTIC_AVAILABLE:
        scenarios.extend([
            {
                "scenario_name": "Arctic_Inference_Single_Context8K_3tokens",
                "batch_size": 1,
                "context_length": 8192,
                "max_new_tokens": 8192,
                "mode": "arctic_inference",
                "method": "arctic_lstm",
                "num_spec_tokens": 3,
            }
        ])

    # Stress test scenarios
    if args.scenario in ["all", "stress"]:
        modes_to_test = [m for m in modes if m != "baseline"]
        
        for mode in modes_to_test:
            if not args.method:
                methods = ["eagle", "e2025", "ngram", "suffix"]
            else:
                methods = [args.method]
                
            scenarios.append({
                "scenario_name": f"VLLM_Native_{method.upper()}_Batch8_LongGen_3tokens",
                "batch_size": 8,
                "context_length": 16384,
                "max_new_tokens": 16384,
                "mode": "vllm_native",
                "method": method,
                "num_spec_tokens": 3,
            })

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
        mode = result["scenario"]["mode"]
        method = result["scenario"]["method"]
        if "throughput_tokens_per_sec" in result:
            throughput = result["throughput_tokens_per_sec"]["mean"]
            latency = result["latency_ms_per_token"]["mean"]
            print(f"{name:60s} {throughput:8.1f} tok/s  {latency:6.2f} ms/tok  [{mode}:{method}]")


if __name__ == "__main__":
    main()
