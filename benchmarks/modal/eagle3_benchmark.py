# EAGLE3 Temperature Benchmark on Modal
#
# Tests EAGLE3 speculative decoding with temperature-aware draft sampling.
# Compares temperature=0.0 (greedy) vs temperature=1.0 (random sampling).
#
# Uses vllm-drift fork with temperature-aware sampling injected into vLLM 0.13.0.

import modal
import os
import time

app = modal.App("eagle3-temperature-benchmark")
models_volume = modal.Volume.from_name("eagle3-models", create_if_missing=True)

# Image with vLLM 0.13.0 + injected vllm-drift changes
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio torchcodec triton --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    # Install vLLM 0.13.0 deps first, then override with local vllm-drift
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .run_commands("pip install vllm==0.13.0")
    # Inject local vllm-drift with temperature-aware EAGLE3 sampling
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm",
        remote_path="/root/vllm-drift-src",
        copy=True,
    )
    # Replace vLLM site-packages with local vllm-drift
    .run_commands(
        "cp -rfv /root/vllm-drift-src/* /usr/local/lib/python3.11/site-packages/vllm/"
    )
)


# ============================================================================
# BENCHMARK CONFIGURATION (from user settings)
# ============================================================================
BATCH_SIZE = 8
MAX_MODEL_LEN = 65536
MAX_TOKENS = 1024
NUM_ITERS = 2
WARMUP_ITERS = 1
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.9


@app.function(
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=7200,
    image=vllm_image,
    env={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
)
def run_eagle3_temperature_test():
    """Run EAGLE3 benchmark at temperature=0.0 and temperature=1.0."""
    import os
    import torch
    import gc
    import time
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # Patch vLLM files at runtime
    print("Patching vLLM site-packages with vllm-drift EAGLE3 code...")
    subprocess.run(
        "cp -rf /root/vllm-drift-src/* /usr/local/lib/python3.11/site-packages/vllm/",
        shell=True,
        check=True,
    )
    subprocess.run(
        "find /usr/local/lib/python3.11/site-packages/vllm -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true",
        shell=True,
    )
    print("Patch complete.")

    import vllm

    print(f"vLLM version: {vllm.__version__}")

    # Verify EAGLE3 temperature-aware sampling
    from vllm.v1.spec_decode.eagle import sample_draft_tokens
    import inspect

    source = inspect.getsource(sample_draft_tokens)
    has_temp = "temperature" in source.lower()
    print(f"sample_draft_tokens has temperature support: {has_temp}")

    os.environ["HF_HOME"] = "/models/hf_cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Models - 120B
    TARGET_MODEL = "openai/gpt-oss-120b"
    DRAFT_MODEL = "nvidia/gpt-oss-120b-Eagle3-throughput"
    TARGET_PATH = "/models/openai--gpt-oss-120b"
    DRAFT_PATH = "/models/nvidia--gpt-oss-120b-eagle3-throughput"

    # Download models
    for repo, path in [(TARGET_MODEL, TARGET_PATH), (DRAFT_MODEL, DRAFT_PATH)]:
        if not Path(path).exists():
            print(f"Downloading {repo}...")
            snapshot_download(
                repo_id=repo, local_dir=path, local_dir_use_symlinks=False
            )
            models_volume.commit()
            print(f"Downloaded {repo}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)

    # Generate enough prompts for batch
    base_prompts = [
        "Explain quantum computing in simple terms.",
        "What is the theory of relativity?",
        "How do neural networks learn?",
        "Describe the water cycle.",
        "What causes earthquakes?",
        "How does photosynthesis work?",
        "Explain the concept of inflation in economics.",
        "What is machine learning?",
    ]

    prompts = []
    for p in base_prompts[:BATCH_SIZE]:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    results = {}

    # EAGLE3 speculative config - only 1 spec token allowed
    spec_config = {
        "method": "eagle3",
        "model": DRAFT_PATH,
        "num_speculative_tokens": 1,
    }

    print(f"\n{'='*60}")
    print(f"BENCHMARK CONFIG:")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  max_model_len={MAX_MODEL_LEN}")
    print(f"  max_tokens={MAX_TOKENS}")
    print(f"  num_iters={NUM_ITERS}")
    print(f"  warmup_iters={WARMUP_ITERS}")
    print(f"  gpu_memory_utilization={GPU_MEMORY_UTILIZATION}")
    print("=" * 60)

    for temperature in [0.0, 1.0]:
        print(f"\n{'='*60}")
        print(f"EAGLE3 Test: temperature={temperature}")
        print("=" * 60)

        sample_params = SamplingParams(
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )

        llm = LLM(
            model=TARGET_PATH,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_num_seqs=BATCH_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            speculative_config=spec_config,
        )

        # Warmup
        print(f"[Warmup] Running {WARMUP_ITERS} iteration(s)...")
        for _ in range(WARMUP_ITERS):
            _ = llm.generate(prompts, sample_params)
        print("[Warmup] Complete.")

        # Benchmark
        total_tokens = 0
        total_time = 0.0

        for i in range(NUM_ITERS):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sample_params)
            elapsed = time.perf_counter() - start
            tokens = sum(len(tokenizer.encode(o.outputs[0].text)) for o in outputs)
            total_tokens += tokens
            total_time += elapsed
            print(
                f"  Iter {i+1}: {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.2f} tok/s)"
            )

        tps = total_tokens / total_time

        results[temperature] = {
            "tps": tps,
            "total_tokens": total_tokens,
            "elapsed": total_time,
            "sample_output": outputs[0].outputs[0].text[:300],
        }

        print(f"Avg TPS: {tps:.2f}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # ===== BASELINE (no speculative decoding) =====
    print(f"\n{'='*60}")
    print("BASELINE: No speculative decoding, temperature=0.0")
    print("=" * 60)

    sample_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    llm_baseline = LLM(
        model=TARGET_PATH,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=BATCH_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = llm_baseline.generate(prompts, sample_params)

    total_tokens = 0
    total_time = 0.0
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        outputs = llm_baseline.generate(prompts, sample_params)
        elapsed = time.perf_counter() - start
        tokens = sum(len(tokenizer.encode(o.outputs[0].text)) for o in outputs)
        total_tokens += tokens
        total_time += elapsed
        print(
            f"  Iter {i+1}: {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.2f} tok/s)"
        )

    baseline_tps = total_tokens / total_time

    results["baseline"] = {
        "tps": baseline_tps,
        "total_tokens": total_tokens,
    }

    print(f"Baseline Avg TPS: {baseline_tps:.2f}")

    del llm_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: EAGLE3 Temperature Performance")
    print("=" * 60)
    for key, r in results.items():
        print(f"  {key}: {r['tps']:.2f} tok/s")

    if "baseline" in results and 0.0 in results:
        speedup = results[0.0]["tps"] / results["baseline"]["tps"]
        print(f"\nEAGLE3 Speedup (temp=0.0 vs baseline): {speedup:.2f}x")

    if 0.0 in results and 1.0 in results:
        ratio = results[1.0]["tps"] / results[0.0]["tps"]
        print(f"Temperature 1.0 vs 0.0: {ratio:.2%}")
        if ratio > 0.8:
            print("✓ Temperature sampling maintains good performance!")
        else:
            print("⚠ Significant slowdown with temperature sampling")

    # Save logs
    import json
    from datetime import datetime

    log_dir = Path("/models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"eagle3_temp_benchmark_{timestamp}.log"

    log_data = {
        "timestamp": timestamp,
        "config": {
            "batch_size": BATCH_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "num_iters": NUM_ITERS,
        },
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "results": {str(k): v for k, v in results.items()},
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\nLog saved to: {log_path}")
    models_volume.commit()

    return results


@app.local_entrypoint()
def main():
    run_eagle3_temperature_test.remote()
