#!/usr/bin/env python3
"""
Simple benchmark script for VLLM speculative decoding validation.
"""

import argparse
import json
import time
import os
import subprocess
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import requests
except ImportError:
    print("Error: requests package required")
    exit(1)


class VLLMServer:
    def __init__(self, model_path: str, port: int = 8000):
        self.model_path = model_path
        self.port = port
        self.process = None
        
    def start(self, speculative_config: Optional[str] = None) -> bool:
        # Use system-installed vllm instead of local source
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--host", "0.0.0.0",
            "--gpu-memory-utilization", "0.8",
            "--max-model-len", "32768"
        ]
        
        if speculative_config:
            cmd.extend(["--speculative-model", speculative_config])
            
        print(f"Starting server: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(cmd)
            time.sleep(10)  # Wait for server to start
            return self.check_health()
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
            
    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            time.sleep(2)
            
    def check_health(self) -> bool:
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class BenchmarkRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.server = VLLMServer(model_path)
        
    def run_test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n=== Running {scenario['scenario_name']} ===")
        
        # Start server with appropriate config
        if scenario["mode"] == "baseline":
            success = self.server.start()
        else:
            success = self.server.start(scenario.get("speculative_model"))
            
        if not success:
            return {"error": "Failed to start server"}
            
        try:
            # Run simple benchmark
            start_time = time.time()
            
            # Make API calls
            results = []
            for i in range(5):  # Simple test with 5 requests
                try:
                    response = requests.post(
                        f"http://localhost:{self.server.port}/v1/completions",
                        json={
                            "model": self.model_path,
                            "prompt": "The future of AI is",
                            "max_tokens": scenario.get("max_new_tokens", 50)
                        },
                        timeout=30
                    )
                    if response.status_code == 200:
                        results.append(response.json())
                except Exception as e:
                    print(f"Request {i} failed: {e}")
                    
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            successful_requests = len(results)
            throughput = successful_requests / total_time if total_time > 0 else 0
            
            return {
                "scenario_name": scenario["scenario_name"],
                "mode": scenario["mode"],
                "batch_size": scenario.get("batch_size", 1),
                "context_length": scenario.get("context_length", 2048),
                "max_new_tokens": scenario.get("max_new_tokens", 50),
                "throughput": throughput,
                "successful_requests": successful_requests,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description="Simple VLLM Benchmark")
    parser.add_argument("--model", type=str, default="/workspace/aimo/models/gpt-oss-120b", 
                       help="Model path")
    parser.add_argument("--speculative-model", type=str, 
                       default="/workspace/aimo/models/Arctic-LSTM-Speculator-gpt-oss-120b",
                       help="Speculative model path")
    parser.add_argument("--output", type=str, default="vllm_simple_benchmark_results.json",
                       help="Output file")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.model)
    
    # Define test scenarios
    scenarios = [
        {
            "scenario_name": "Baseline_Batch1_Tokens50",
            "mode": "baseline",
            "batch_size": 1,
            "max_new_tokens": 50
        },
        {
            "scenario_name": "Speculative_Batch1_Tokens50",
            "mode": "speculative",
            "speculative_model": args.speculative_model,
            "batch_size": 1,
            "max_new_tokens": 50
        }
    ]
    
    all_results = []
    for scenario in scenarios:
        result = runner.run_test_scenario(scenario)
        all_results.append(result)
        
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nResults saved to {args.output}")
    for result in all_results:
        if "error" not in result and "scenario_name" in result:
            print(f"{result['scenario_name']}: {result['throughput']:.2f} req/s")
        elif "scenario_name" in result:
            print(f"{result['scenario_name']}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            print(f"Unknown result format: {result}")


if __name__ == "__main__":
    main()