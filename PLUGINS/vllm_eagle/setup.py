from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

# Get the current directory
cwd = os.path.dirname(os.path.abspath(__file__))

# Try to find flashinfer headers
include_dirs = []
flash_found = False

# 1. Check environment variable (Passed from Modal build script)
env_inc = os.environ.get("FLASHINFER_INCLUDE")
if env_inc and os.path.exists(env_inc):
    include_dirs.append(env_inc)
    print(f"vllm_eagle build: Found flashinfer headers via ENV: {env_inc}")
    flash_found = True

# 2. Try importing it
if not flash_found:
    try:
        import flashinfer

        flashinfer_dir = os.path.dirname(flashinfer.__file__)
        candidates = [
            os.path.join(flashinfer_dir, "include"),
            os.path.join(os.path.dirname(flashinfer_dir), "flashinfer", "include"),
            os.path.join(flashinfer_dir, "data", "include"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                include_dirs.append(cand)
                print(f"vllm_eagle build: Found flashinfer headers via import: {cand}")
                flash_found = True
                break
    except Exception:
        pass

# 3. Fallback to standard location
if not flash_found:
    standard_cand = "/usr/local/lib/python3.11/site-packages/flashinfer/include"
    if os.path.exists(standard_cand):
        include_dirs.append(standard_cand)
        print(
            f"vllm_eagle build: Found flashinfer headers at standard fallback: {standard_cand}"
        )
        flash_found = True

if not flash_found:
    print("WARNING: flashinfer headers NOT FOUND! Compilation will likely fail.")

setup(
    name="vllm_eagle",
    version="0.1.0",
    packages=["vllm_eagle"],
    ext_modules=[
        CUDAExtension(
            name="vllm_eagle._C",
            sources=[
                "csrc/pybind.cpp",
                "csrc/speculative_sampling.cu",
                "csrc/tree_utils.cu",
                "csrc/phase2_optimizations.cu",  # Phase 2: Warp-level + Fused kernels
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
# break cache Fri Jan  2 14:32:00 UTC 2026
# break cache # break cache Fri Jan  2 14:35:27 UTC 2026
# last fix Fri Jan  2 14:45:09 UTC 2026
# retry 32 Fri Jan  2 14:48:23 UTC 2026
