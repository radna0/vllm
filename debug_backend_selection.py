import os
import torch
from vllm.config import VllmConfig
from vllm.attention.selector import get_attn_backend
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

# Mock VLLM Config if needed, or rely on defaults/env
# We need to mock get_current_vllm_config if it's used directly
# But get_attn_backend calls it.

print("--- Backend Selection Debug ---")
head_size = 128
dtype = torch.bfloat16
kv_cache_dtype = "auto"
block_size = 16

# 1. Default Selection
print("\n[Case 1: Default]")
try:
    backend_cls = get_attn_backend(head_size, dtype, kv_cache_dtype, block_size)
    print(f"Selected Backend: {backend_cls.__name__}")
    print(f"Builder: {backend_cls.get_builder_cls().__name__}")
    print(f"Impl: {backend_cls.get_impl_cls().__name__}")
except Exception as e:
    print(f"Error: {e}")

# 2. Force FlashInfer
print("\n[Case 2: Force FLASHINFER]")
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
# Clear cache if possible, but get_attn_backend is cached.
# We might need to reload or patch.
# For this script we just restart process or assume it checks env.
# Actually get_attn_backend reads config. config reads env.
# But config is loaded once?
# Let's try to reload config or just re-import selector if needed,
# but vllm config is usually a singleton.

# Re-simulate by manually calling with new config if we could,
# but easier to just print what current config says.

from vllm.config import get_current_vllm_config

try:
    cfg = get_current_vllm_config()
    print(f"Current VLLM Config Backend: {cfg.attention_config.backend}")
except:
    print("Could not get current config (maybe not initialized)")
