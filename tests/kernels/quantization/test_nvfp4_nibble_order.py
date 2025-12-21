# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import break_fp4_bytes
from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 requires compute capability of 10 or above.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@torch.inference_mode()
def test_nvfp4_unpack_fp4x2_order(dtype: torch.dtype) -> None:
    device = "cuda"
    packed = torch.arange(256, device=device, dtype=torch.uint8).reshape(-1, 1)
    out = torch.empty(packed.numel() * 2, device=device, dtype=dtype)

    torch.ops._C.nvfp4_unpack_fp4x2(out, packed.flatten())

    decoded = out.view(packed.shape[0], 2)
    expected = break_fp4_bytes(packed, dtype=dtype)
    torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)
