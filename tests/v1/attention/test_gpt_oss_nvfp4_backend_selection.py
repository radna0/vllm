# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import CacheConfig, VllmConfig
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability


class _DummyHfConfig:
    model_type = "gpt_oss"


class _DummyModelConfig:
    dtype = torch.bfloat16
    hf_config = _DummyHfConfig()
    is_mm_prefix_lm = False

    def get_head_size(self) -> int:
        return 128


def test_gpt_oss_nvfp4_prefers_trtllm_backend(monkeypatch):
    class _DummyBackend:
        @classmethod
        def validate_configuration(cls, **kwargs):
            return []

    monkeypatch.setattr(
        AttentionBackendEnum.TRTLLM_ATTN, "get_class", lambda: _DummyBackend
    )
    monkeypatch.setattr(
        CudaPlatform,
        "is_device_capability_ge",
        classmethod(lambda cls, capability: True),
        raising=False,
    )
    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls: DeviceCapability(10, 0)),
        raising=False,
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_LAYOUT", None, raising=False)
    monkeypatch.setattr(
        "vllm.platforms.cuda.trtllm_attention_supports_nvfp4",
        lambda *args, **kwargs: True,
    )

    vllm_config = VllmConfig(
        model_config=_DummyModelConfig(),
        cache_config=CacheConfig(cache_dtype="nvfp4", block_size=16),
    )
    CudaPlatform.check_and_update_config(vllm_config)

    assert vllm_config.attention_config.nvfp4_backend == AttentionBackendEnum.TRTLLM_ATTN
