# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch
from packaging.version import Version
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)


def _check_nvfp4_support():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NVFP4 benchmark.")
    if not current_platform.has_device_capability(100):
        raise RuntimeError("NVFP4 requires a Blackwell-class GPU (cc >= 10.0).")
    cuda_version = torch.version.cuda
    if cuda_version is None or Version(cuda_version) < Version("12.8"):
        raise RuntimeError("NVFP4 requires CUDA >= 12.8.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("float8_e4m3fn dtype is required for NVFP4.")


@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    num_iters: int,
    fused: bool,
    benchmark_mode: str,
    device: str = "cuda",
) -> float:
    if head_size % 16 != 0 or head_size > 256:
        raise ValueError("NVFP4 requires head_size to be a multiple of 16 <= 256.")
    if block_size % 4 != 0:
        raise ValueError("NVFP4 requires block_size to be a multiple of 4.")
    if num_tokens > num_blocks * block_size:
        raise ValueError("num_tokens cannot exceed the total cache slots.")

    current_platform.seed_everything(42)
    torch.set_default_device(device)

    positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    q = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    v = torch.randn_like(k)

    packed_head_size = head_size // 8
    key_cache = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        packed_head_size,
        dtype=torch.int32,
        device=device,
    )
    value_cache = torch.empty_like(key_cache)
    scale_shape = (num_blocks, block_size, num_kv_heads, head_size // 16)
    k_scale = torch.empty(scale_shape, dtype=torch.float8_e4m3fn, device=device)
    v_scale = torch.empty(scale_shape, dtype=torch.float8_e4m3fn, device=device)

    rotary_emb = RotaryEmbedding(
        head_size,
        head_size,
        num_tokens,
        10000.0,
        True,
        dtype,
    )
    rotary_emb._match_cos_sin_cache_dtype(q)

    def baseline():
        q_2d = q.view(num_tokens, num_heads * head_size)
        k_2d = k.view(num_tokens, num_kv_heads * head_size)
        _, k_rot = rotary_emb(positions, q_2d, k_2d)
        k_rot = k_rot.view(num_tokens, num_kv_heads, head_size)
        ops.reshape_and_cache_flash(
            k_rot,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            "nvfp4",
            k_scale,
            v_scale,
        )

    def fused_kernel():
        ops.fused_rope_and_cache_flash_nvfp4(
            q,
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            positions,
            rotary_emb.cos_sin_cache,
            rotary_emb.is_neox_style,
            k_scale,
            v_scale,
        )

    function_under_test = fused_kernel if fused else baseline
    if benchmark_mode == "cudagraph":
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            function_under_test()
        torch.cuda.synchronize()
        function_under_test = lambda: g.replay()

    def run_cuda_benchmark(n_iters: int) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            function_under_test()
            torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) / n_iters

    run_cuda_benchmark(3)
    lat = run_cuda_benchmark(num_iters)

    del q, k, v, key_cache, value_cache, k_scale, v_scale, slot_mapping, positions
    torch.cuda.empty_cache()
    return lat


def main(args):
    _check_nvfp4_support()

    rows = []
    for exp in range(args.min_exp, args.max_exp + 1):
        num_tokens = 2**exp
        base_lat = run_benchmark(
            num_tokens=num_tokens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads or args.num_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            num_iters=args.iters,
            fused=False,
            benchmark_mode=args.mode,
            device="cuda",
        )
        fused_lat = run_benchmark(
            num_tokens=num_tokens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads or args.num_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
            num_iters=args.iters,
            fused=True,
            benchmark_mode=args.mode,
            device="cuda",
        )
        rows.append(
            [
                num_tokens,
                f"{base_lat * 1e6:.3f}",
                f"{fused_lat * 1e6:.3f}",
                f"{base_lat / fused_lat:.2f}",
            ]
        )

    print(f"NVFP4 fused RoPE+cache vs baseline ({args.mode}):")
    print(tabulate(rows, headers=["num_tokens", "baseline (us)", "fused (us)", "speedup"]))


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=0)
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        default=128,
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--num-blocks", type=int, default=128 * 512)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eager", "cudagraph"],
        default="eager",
    )
    parser.add_argument("--min-exp", type=int, default=1)
    parser.add_argument("--max-exp", type=int, default=16)

    args = parser.parse_args()
    if args.num_kv_heads == 0:
        args.num_kv_heads = args.num_heads
    main(args)
