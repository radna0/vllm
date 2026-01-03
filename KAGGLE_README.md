# EAGLE Phase 2 - Kaggle Integration Guide

This guide provides complete instructions for benchmarking EAGLE Phase 2 optimizations on Kaggle with an H100 GPU.

## Repository Setup

The EAGLE Phase 2 code is split across two repositories:

1. **vllm (drift branch)**: Modified vLLM with EAGLE Phase 2 integration
   - Repository: `https://github.com/radna0/vllm` (branch: `drift`)
   - Contains: Phase 2 kernel integration in `vllm/v1/spec_decode/eagle.py`

2. **vllm_eagle**: Custom CUDA kernels for H100
   - Repository: `https://github.com/radna0/vllm_eagle`
   - Contains: Warp-optimized sampling kernels and fused operations

## Kaggle Notebook Setup

### Cell 1: Environment Cleanup and Dependencies

```python
# Clean up default Kaggle environment
! pip uninstall --yes "tensorflow" "matplotlib" "keras" "scikit-learn" "protobuf" "numpy" "torch"
! pip cache purge

# Install optimized dependencies
! pip install --target=/kaggle/working \
    torch==2.9.0 torchvision torchaudio torchcodec triton \
    'transformers>=4.57.1,!=4.57.2' sentence-transformers \
    numpy==2.2.0 vllm==0.13.0 pandas polars 'openai-harmony>=0.0.8' \
    hf_transfer jupyter_client ipykernel mcp msgspec tiktoken \
    flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://flashinfer.ai/whl/cu128
```

### Cell 2: Clone Repository

```python
# Clone EAGLE Phase 2 repository (everything is now in vllm-drift)
! git clone -b drift https://github.com/radna0/vllm.git /kaggle/working/vllm-drift

# Verify cloning
import os
print("✓ vllm-drift cloned" if os.path.exists("/kaggle/working/vllm-drift") else "✗ vllm-drift missing")
```

### Cell 3: Patch vLLM and Build Custom Kernels

```python
import os
import sys

# 1. Add working directory to Python path FIRST
sys.path.insert(0, "/kaggle/working")

# 2. Hot-patch vLLM with Phase 2 logic
%cd /kaggle/working/vllm-drift
! bash PLUGINS/inject.sh /kaggle/working/vllm

# 3. Build and Install Phase 2 Custom Kernels (Kaggle Optimized)
! bash PLUGINS/install_kaggle.sh

# 4. CRITICAL: Force-load PyTorch libraries (The "Nuclear Option")
# This ensures the newly built .so files can find their dependencies
import torch
import ctypes
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
try:
    ctypes.CDLL(os.path.join(torch_lib_path, "libc10.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(torch_lib_path, "libtorch_cpu.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(torch_lib_path, "libtorch_cuda.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(torch_lib_path, "libtorch_python.so"), mode=ctypes.RTLD_GLOBAL)
    print("✓ PyTorch libraries force-loaded")
except Exception as e:
    print(f"⚠️  PyTorch library pre-load warning: {e}")

%cd /kaggle/working

# 5. FINAL VERIFICATION
print("\n" + "="*80)
print("VERIFYING PHASE 2 KERNEL AVAILABILITY")
print("="*80)

try:
    import vllm_eagle._C as C
    print("✓ vllm_eagle._C imported successfully!")
    funcs = [f for f in dir(C) if not f.startswith('_')]
    print(f"✓ Available ops: {funcs}")
    
    if 'fused_gumbel_sample_warp_optimized' in funcs:
        print("🚀 SUCCESS: Phase 2 Kernels are LIVE!")
    else:
        print("⚠️  Warning: Package imported but Phase 2 symbols missing.")
except Exception as e:
    print(f"✗ ERROR: Import failed: {e}")
```

### Cell 4: Tokenizer Setup

```python
# Setup Harmony Encodings and Tiktoken Cache
! TIKTOKEN_RS_CACHE_DIR=/kaggle/working python -c 'from openai_harmony import load_harmony_encoding; load_harmony_encoding("HarmonyGptOss")'

! mkdir -p tiktoken_encodings
! wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
! wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Verify
! ls -lh tiktoken_encodings/
```

### Cell 5: Environment Variables

```bash
%%bash
export TIKTOKEN_ENCODINGS_BASE=${PWD}/tiktoken_encodings
export PYTHONPATH=$PYTHONPATH:/kaggle/working
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
```

### Cell 6: Download Benchmark Script and Data

```python
# Download the benchmark script and reference data
! wget -O /kaggle/working/kaggle_benchmark_86e8e5.py \
    https://raw.githubusercontent.com/radna0/vllm/drift/kaggle_benchmark_86e8e5.py

! wget -O /kaggle/working/local_python_tool.py \
    https://raw.githubusercontent.com/radna0/vllm/drift/local_python_tool.py

! wget -O /kaggle/working/reference.csv \
    https://raw.githubusercontent.com/radna0/vllm/drift/reference.csv

print("✓ Benchmark files downloaded")
```

### Cell 7: Run Baseline Benchmark (Phase 2 OFF)

```python
import os
import sys
sys.path.append("/kaggle/working")

# Configure environment
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/working/tiktoken_encodings"

# BASELINE: Phase 2 OFF
os.environ["VLLM_EAGLE_PHASE2_FUSED"] = "0"
os.environ["VLLM_EAGLE_DRAFT_SAMPLING"] = "1"

print("="*80)
print("BASELINE BENCHMARK (Phase 2 OFF)")
print("="*80)

! python /kaggle/working/kaggle_benchmark_86e8e5.py
```

### Cell 8: Run Optimized Benchmark (Phase 2 ON)

```python
import os
import sys
sys.path.append("/kaggle/working")

# Configure environment
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/working/tiktoken_encodings"

# OPTIMIZED: Phase 2 ON
os.environ["VLLM_EAGLE_PHASE2_FUSED"] = "1"
os.environ["VLLM_EAGLE_DRAFT_SAMPLING"] = "1"

print("="*80)
print("OPTIMIZED BENCHMARK (Phase 2 ON)")
print("="*80)

! python /kaggle/working/kaggle_benchmark_86e8e5.py
```

## Benchmark Parameters

The benchmark script (`kaggle_benchmark_86e8e5.py`) uses the following parameters:

- **Problem**: `86e8e5` from `reference.csv`
- **Batch Size**: 8
- **Max Model Length**: 65,536 tokens
- **Temperature**: 1.0
- **Min-P**: 0.02
- **Top-P**: 1.0
- **Seed**: 42
- **Stream Interval**: 200
- **Num Speculative Tokens**: 3

## Expected Results

Based on our H100 benchmarks:

| Configuration | Decode Throughput | Speedup |
|:---|:---:|:---:|
| Baseline (Phase 2 OFF) | ~180-250 tok/s | 1.0x |
| Optimized (Phase 2 ON) | **~350-400 tok/s** | **~1.6-2.0x** |

## Troubleshooting

### Repository Not Found
If you see "Repository not found" errors when cloning, the repositories may be private. Contact the repository owner for access.

### Build Failures
Ensure you're using an H100 GPU in Kaggle settings. The custom kernels are optimized for compute capability 9.0.

### Import Errors
Make sure `/kaggle/working` is in your Python path:
```python
import sys
sys.path.append("/kaggle/working")
```

### Model Path Issues
Adjust `MODEL_PATH` and `DRAFT_MODEL_PATH` in the benchmark script to match your Kaggle input dataset paths.

## Notes

- The benchmark automatically handles server startup, problem execution, and cleanup.
- Logs will show detailed metrics including acceptance rates and per-turn latency.
- For quality inspection, the script saves conversation logs for manual review.
