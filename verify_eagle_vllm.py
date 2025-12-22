import os
import sys
import subprocess
from pathlib import Path

# Config
PYTHON_BIN = "/workspace/aimo/miniconda/envs/vllm/bin/python"  # Tentative, will verify with env list
BENCHMARK_SCRIPT = "/workspace/aimo/LM/vllm/benchmark_speculative.py"
MODEL_PATH = "/workspace/aimo/models/gpt-oss-120b"
EAGLE_MODEL = "/workspace/aimo/models/gpt-oss-120b-eagle3"
OUTPUT_DIR = "/workspace/aimo/logs/vllm_verification"


def run_verification():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        BENCHMARK_SCRIPT,
        "--model-path",
        MODEL_PATH,
        "--spec-model-path",
        EAGLE_MODEL,
        "--method",
        "eagle3",
        "--scenario",
        "single",
        "--warmup-runs",
        "1",
        "--measurement-runs",
        "1",
        "--output-dir",
        OUTPUT_DIR,
    ]

    print(f"Running verification: {' '.join(cmd)}")
    env = os.environ.copy()
    env["LMDEPLOY_EAGLE_MICRO_STEPS"] = "20"  # Verify functionality only
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # Standard backend fallback
    # Force EAGLE3 features if needed via env vars

    try:
        subprocess.run(cmd, env=env, check=True)
        print("\nVerification Run Completed Successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nVerification Failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    run_verification()
