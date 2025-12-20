# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import break_fp4_bytes, kE2M1ToFloat
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 requires compute capability of 10 or above.",
        allow_module_level=True,
    )


def _dequantize_nvfp4_kv_cache(
    packed: torch.Tensor, scales: torch.Tensor, head_size: int
) -> torch.Tensor:
    packed_head_size = head_size // 8
    scale_elems = head_size // 16
    packed_bytes = packed.view(torch.uint8).reshape(-1, packed_head_size * 4)
    values = break_fp4_bytes(packed_bytes, dtype=torch.float32)
    scales_f32 = scales.view(torch.float8_e4m3fn).to(torch.float32)
    scales_f32 = scales_f32.reshape(-1, scale_elems)
    values = values.reshape(-1, scale_elems, 16) * scales_f32[:, :, None]
    return values.reshape(-1, head_size)


def _deinterleave_v_scales(scales: torch.Tensor, head_size: int) -> torch.Tensor:
    scale_elems = head_size // 16
    block_size = scales.shape[1]
    token_idx = torch.arange(block_size, device=scales.device)
    scale_idx = torch.arange(scale_elems, device=scales.device)
    token_group = token_idx // 4
    token_mod = token_idx % 4
    interleaved_idx = (
        token_group[:, None] * (4 * scale_elems)
        + scale_idx[None, :] * 4
        + token_mod[:, None]
    )
    interleaved_token = interleaved_idx // scale_elems
    interleaved_scale = interleaved_idx % scale_elems
    out = torch.empty_like(scales)
    for block in range(scales.shape[0]):
        for head in range(scales.shape[2]):
            out[block, :, head, :] = scales[block, interleaved_token, head, interleaved_scale]
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_nvfp4_kv_cache_pack_unpack_order(dtype: torch.dtype) -> None:
    device = "cuda"
    head_size = 16
    block_size = 4
    num_tokens = 4
    num_heads = 1

    base_vals = kE2M1ToFloat.to(device=device)
    values = torch.cat((base_vals, -base_vals)).to(dtype=dtype)
    factors = torch.tensor([1.0, 0.5, 0.25, 0.125], device=device, dtype=dtype)
    key = (factors[:, None] * values[None, :]).view(
        num_tokens, num_heads, head_size
    )
    value = key.clone()

    num_blocks = 1
    packed_head_size = head_size // 8
    key_cache = torch.empty(
        (num_blocks, block_size, num_heads, packed_head_size),
        dtype=torch.int32,
        device=device,
    )
    value_cache = torch.empty_like(key_cache)
    k_scale = torch.empty(
        (num_blocks, block_size, num_heads, head_size // 16),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_scale = torch.empty_like(k_scale)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "nvfp4",
        k_scale,
        v_scale,
    )

    dequant = _dequantize_nvfp4_kv_cache(key_cache, k_scale, head_size)
    torch.testing.assert_close(
        dequant.view_as(key),
        key,
        rtol=1e-2,
        atol=1e-2,
    )
    v_scale_linear = _deinterleave_v_scales(v_scale, head_size)
    dequant_v = _dequantize_nvfp4_kv_cache(value_cache, v_scale_linear, head_size)
    torch.testing.assert_close(
        dequant_v.view_as(value),
        value,
        rtol=1e-2,
        atol=1e-2,
    )
