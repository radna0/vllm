# EAGLE3 min_p Draft Sampling Benchmark
#
# Compares acceptance rates with/without VLLM_EAGLE_DRAFT_SAMPLING enabled.
# Tests min_p=0.05 with temperature=1.0

import modal
import os
import time

app = modal.App("eagle3-minp-benchmark")
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
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .run_commands("pip install vllm==0.13.0")
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm-drift/vllm",
        remote_path="/root/vllm-drift-src",
        copy=True,
    )
    .run_commands(
        "cp -rfv /root/vllm-drift-src/* /usr/local/lib/python3.11/site-packages/vllm/"
    )
)


BATCH_SIZE = 8
MAX_MODEL_LEN = 65536
MAX_TOKENS = 512
NUM_ITERS = 4


@app.function(
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=7200,
    image=vllm_image,
    env={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
)
def run_minp_benchmark():
    """Compare EAGLE3 with/without draft sampling filters."""
    import os
    import torch
    import gc
    import time
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # Patch vLLM
    print("Patching vLLM site-packages with vllm-drift code...")
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

    os.environ["HF_HOME"] = "/models/hf_cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)

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

    spec_config = {
        "method": "eagle3",
        "model": DRAFT_PATH,
        "num_speculative_tokens": 1,
    }

    # Test with min_p=0.05, temperature=1.0
    sample_params = SamplingParams(
        temperature=1.0,
        min_p=0.05,
        max_tokens=MAX_TOKENS,
    )

    results = {}

    # ===== Test 1: Draft sampling DISABLED (current behavior) =====
    print("\n" + "=" * 60)
    print("TEST 1: Draft sampling DISABLED (VLLM_EAGLE_DRAFT_SAMPLING=0)")
    print("=" * 60)
    os.environ["VLLM_LOG_STATS_INTERVAL"] = "1"
    os.environ["VLLM_EAGLE_DRAFT_SAMPLING"] = "0"

    llm1 = LLM(
        model=TARGET_PATH,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        max_num_seqs=BATCH_SIZE,
        speculative_config=spec_config,
        disable_log_stats=False,
    )

    # Warmup
    _ = llm1.generate(prompts, sample_params)

    # Benchmark
    total_tokens = 0
    total_time = 0.0
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        outputs = llm1.generate(prompts, sample_params)
        elapsed = time.perf_counter() - start
        tokens = sum(len(tokenizer.encode(o.outputs[0].text)) for o in outputs)
        total_tokens += tokens
        total_time += elapsed
        print(
            f"  Iter {i+1}: {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.2f} tok/s)"
        )

    print("\nFinal Stats (Draft sampling DISABLED):")
    llm1.llm_engine.do_log_stats()

    results["disabled"] = {
        "tps": total_tokens / total_time,
        "total_tokens": total_tokens,
    }
    print(f"TPS (disabled): {results['disabled']['tps']:.2f}")

    del llm1
    gc.collect()
    torch.cuda.empty_cache()

    # ===== Test 2: Draft sampling ENABLED =====
    print("\n" + "=" * 60)
    print("TEST 2: Draft sampling ENABLED (VLLM_EAGLE_DRAFT_SAMPLING=1)")
    print("=" * 60)
    os.environ["VLLM_EAGLE_DRAFT_SAMPLING"] = "1"

    # Need to reimport to pick up env var change
    # This requires a fresh LLM instance
    llm2 = LLM(
        model=TARGET_PATH,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        max_num_seqs=BATCH_SIZE,
        speculative_config=spec_config,
        disable_log_stats=False,
    )

    # Warmup
    _ = llm2.generate(prompts, sample_params)

    # Benchmark
    total_tokens = 0
    total_time = 0.0
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        outputs = llm2.generate(prompts, sample_params)
        elapsed = time.perf_counter() - start
        tokens = sum(len(tokenizer.encode(o.outputs[0].text)) for o in outputs)
        total_tokens += tokens
        total_time += elapsed
        print(
            f"  Iter {i+1}: {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.2f} tok/s)"
        )

    print("\nFinal Stats (Draft sampling ENABLED):")
    llm2.llm_engine.do_log_stats()

    results["enabled"] = {
        "tps": total_tokens / total_time,
        "total_tokens": total_tokens,
    }
    print(f"TPS (enabled): {results['enabled']['tps']:.2f}")

    del llm2
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: min_p Draft Sampling Impact")
    print("=" * 60)
    print(f"  Draft sampling DISABLED: {results['disabled']['tps']:.2f} tok/s")
    print(f"  Draft sampling ENABLED:  {results['enabled']['tps']:.2f} tok/s")

    ratio = results["enabled"]["tps"] / results["disabled"]["tps"]
    print(f"\nRatio (enabled/disabled): {ratio:.2%}")

    if ratio > 1.0:
        print("✓ Draft sampling IMPROVES throughput (better acceptance rate)!")
    elif ratio > 0.95:
        print("≈ Draft sampling has minimal impact on throughput")
    else:
        print("⚠ Draft sampling REDUCES throughput")

    # Save logs
    import json
    from datetime import datetime

    log_dir = Path("/models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"eagle3_minp_benchmark_{timestamp}.log"

    with open(log_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": {"min_p": 0.05, "temperature": 1.0},
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nLog saved to: {log_path}")
    models_volume.commit()

    return results


@app.local_entrypoint()
def main():
    run_minp_benchmark.remote()
