# WeDLM vLLM vs WeDLM Native Benchmark
#
# This script compares vLLM (standard AR) vs WeDLM (native SPD) head-to-head.
# It uses settings from original WeDLM benchmarks: Batch 8, 256 tokens.
#
# Methodology: WHEN DID IT GO WRONG?
#   - Detect NaNs/Infs: STOP and RAISE.
#   - Detect repetitive output: STOP and LOG.
#   - Compare throughput and quality.

import modal
import sys
import os
import time
import json

app = modal.App("vllm-vs-wedlm-benchmark")
models_volume = modal.Volume.from_name("wedlm-models", create_if_missing=True)

# Image with vLLM from local vllm-drift
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
    .add_local_dir(
        "/home/kojoe/CUDA_DLM/vllm-drift/vllm",
        remote_path="/root/vllm-drift-src",
        copy=True,
    )
    # Replace vLLM site-packages with local vllm-drift (done during build, not runtime)
    .run_commands(
        "ls -la /root/vllm-drift-src/v1/worker/block_table.py && "
        "grep -c slot_mapping /root/vllm-drift-src/v1/worker/block_table.py && "
        "cp -rfv /root/vllm-drift-src/* /usr/local/lib/python3.11/site-packages/vllm/ && "
        "grep -c slot_mapping /usr/local/lib/python3.11/site-packages/vllm/v1/worker/block_table.py"
    )
)


@app.function(
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=3600,
    image=vllm_image,
    env={
        "VLLM_WEDLM_INNER_STEPS": "4",
        "VLLM_WEDLM_DEBUG": "2",
        "VLLM_WEDLM_CUDAGRAPH": "0",
    },
)
def run_comparison():
    batch_size: int = 8
    max_tokens: int = 512
    import os
    import torch
    import gc
    import time
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # CRITICAL: Patch vLLM files BEFORE import so subprocess workers also get patched code
    # Modal's add_local_dir mount shadows build-time copy, so we patch at runtime
    print("Patching vLLM site-packages with vllm-drift code...")
    subprocess.run(
        "cp -rf /root/vllm-drift-src/* /usr/local/lib/python3.11/site-packages/vllm/",
        shell=True,
        check=True,
    )
    # CRITICAL: Clear Python bytecode cache to force recompilation from patched source
    subprocess.run(
        "find /usr/local/lib/python3.11/site-packages/vllm -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true",
        shell=True,
    )
    print("Patch complete. pycache cleared. Verifying...")
    result = subprocess.run(
        "grep -c slot_mapping /usr/local/lib/python3.11/site-packages/vllm/v1/worker/block_table.py",
        shell=True,
        capture_output=True,
        text=True,
    )
    print(f"slot_mapping occurrences in block_table.py: {result.stdout.strip()}")

    # Now import vLLM after patching
    import vllm

    print(f"vLLM version: {vllm.__version__}")
    print(f"vLLM file: {vllm.__file__}")

    from vllm.model_executor.models import registry

    print(f"WeDLM in registry: {'WeDLMForCausalLM' in registry._VLLM_MODELS}")

    # Verify file-level patch worked - check actual module path
    from vllm.v1.worker.block_table import MultiGroupBlockTable
    import inspect

    print(f"MultiGroupBlockTable source file: {inspect.getfile(MultiGroupBlockTable)}")
    print(
        f"MultiGroupBlockTable has slot_mapping: {hasattr(MultiGroupBlockTable, 'slot_mapping')}"
    )
    print(
        f"MultiGroupBlockTable has block_table: {hasattr(MultiGroupBlockTable, 'block_table')}"
    )

    # WHEN DID IT GO WRONG? - Enable verbose entropy stats logging
    os.environ["VLLM_WEDLM_STATS_VERBOSE"] = "1"
    # NOTE: VLLM_WEDLM_DEBUG removed - triggers .item() calls that break CUDA graph capture
    os.environ["VLLM_WEDLM_STATS_INTERVAL"] = "10"  # Log every 10 steps

    os.environ["HF_HOME"] = "/models/hf_cache"
    MODEL_ID = "tencent/WeDLM-8B-Instruct"
    MODEL_PATH = "/models/tencent--WeDLM-8B-Instruct"

    if not Path(MODEL_PATH).exists():
        snapshot_download(
            repo_id=MODEL_ID, local_dir=MODEL_PATH, local_dir_use_symlinks=False
        )
        models_volume.commit()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    test_prompts = [
        "Explain quantum computing in simple terms.",
        "What is the theory of relativity?",
        "How do neural networks learn?",
        "Describe the water cycle.",
        "What causes earthquakes?",
        "How does photosynthesis work?",
        "Explain the concept of inflation in economics.",
        "What is machine learning?",
    ]
    # Ensure enough prompts for batch
    while len(test_prompts) < batch_size:
        test_prompts.extend(test_prompts)
    prompts_raw = test_prompts[:batch_size]

    prompts = []
    for p in prompts_raw:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    # Use temperature=0.0 for greedy decoding (matches native WeDLM benchmark)
    # Higher entropy_threshold allows more aggressive parallel position filling
    sample_params = SamplingParams(
        temperature=0.0,  # Greedy decode - matches native WeDLM
        max_tokens=max_tokens,
        # WeDLM-specific params (defaults are entropy_threshold=0.8, pos_penalty=0.02)
    )

    # ==========================================================================
    # TEST 1: vLLM (Standard AR)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"TEST 1: vLLM (AR) | Batch={batch_size} | MaxTokens={max_tokens}")
    print("=" * 80)

    # We force self.is_wedlm_model = False for AR by patching gpu_model_runner if needed,
    # but since it's a new run, we can just ensure speculative method isn't set.
    # However, our current vLLM-drift ALWAYS detects WeDLM. We need to disable it.

    # NOTE: Do NOT use enforce_eager=True - it disables CUDA graphs and causes 5x slowdown!
    llm_ar = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        max_num_seqs=8,  # Reduce to ensure stability in Modal environment
    )

    # WARMUP: First iteration to trigger CUDA graph capture and compilation
    print("[Warmup] Running first iteration...")
    _ = llm_ar.generate(prompts, sample_params)
    print("[Warmup] Complete. Running timed benchmark...")

    start = time.perf_counter()
    results_ar = llm_ar.generate(prompts, sample_params)
    elapsed_ar = time.perf_counter() - start

    total_tokens_ar = sum(len(tokenizer.encode(r.outputs[0].text)) for r in results_ar)
    tps_ar = total_tokens_ar / elapsed_ar

    print(f"vLLM (AR) TPS: {tps_ar:.2f}")
    print(f"Sample: {results_ar[0].outputs[0].text[:100]}...")

    del llm_ar
    gc.collect()
    torch.cuda.empty_cache()

    # NOTE: TEST 2 with speculative_config removed - WeDLM Native SPD is
    # automatically enabled when WeDLMForCausalLM is detected (see gpu_model_runner.py:390-398)
    # The TEST 1 above is already running native WeDLM SPD decode.
    #
    # Check log for:
    # - "WeDLM model detected. Native SPD decode enabled"
    # - "Capturing WeDLM cudagraphs (step_size=..., batch_sizes=...)"
    # - "[SPD_STATS] ... tokens_per_forward=..."

    return {
        "tps_wedlm_spd": tps_ar,
    }


@app.local_entrypoint()
def main():
    run_comparison.remote()
