# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from numbers import Integral

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import _calc_nvfp4_global_scale, get_attention_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

_LOG_FUSED_NVFP4 = os.environ.get("VLLM_LOG_FUSED_NVFP4") == "1"
_logger = init_logger(__name__)

def _is_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return torch._dynamo.is_compiling()  # type: ignore[attr-defined]
    except Exception:
        return False


def try_fused_rope_and_cache_nvfp4(
    *,
    attn,
    rotary_emb,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    layer_name = getattr(attn, "layer_name", "<unknown>")
    log_fused = _LOG_FUSED_NVFP4 and not _is_compiling()
    if log_fused:
        def _skip(reason: str) -> bool:
            _logger.info_once(
                "Skipping fused RoPE+NVFP4 for %s: %s", layer_name, reason
            )
            return False
        def _note(message: str, *args) -> None:
            _logger.info_once(message, *args)

    else:

        def _skip(reason: str) -> bool:
            return False
        def _note(message: str, *args) -> None:
            return None

    if attn is None or rotary_emb is None:
        return _skip("attn/rotary missing")
    if not isinstance(rotary_emb, RotaryEmbedding):
        return _skip("rotary_emb type mismatch")
    if getattr(attn, "kv_cache_dtype", None) != "nvfp4":
        return _skip("kv_cache_dtype is not nvfp4")
    if getattr(attn, "attn_type", None) != AttentionType.DECODER:
        return _skip("attn_type is not decoder")
    if not getattr(attn, "use_direct_call", False):
        return _skip("use_direct_call is False")
    if getattr(attn, "kv_sharing_target_layer_name", None) is not None:
        return _skip("kv_sharing_target_layer_name set")
    if head_dim % 16 != 0 or head_dim > 256:
        return _skip("head_dim not supported")
    positions_1d = positions
    if positions.dim() == 2:
        if 1 not in positions.shape:
            return _skip("positions dim 2 unsupported")
        positions_1d = positions.reshape(-1)
    elif positions.dim() != 1:
        return _skip("positions dim not 1/2")
    if positions_1d.dtype != torch.long:
        return _skip("positions dtype not int64")
    if query.dtype not in (torch.float16, torch.bfloat16):
        return _skip("query dtype not fp16/bf16")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return _skip("qkv not on cuda")
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        return _skip("qkv not contiguous")

    if query.dim() == 2:
        if query.shape[1] != num_heads * head_dim:
            return _skip("query shape mismatch")
        q_3d = query.view(-1, num_heads, head_dim)
    elif query.dim() == 3:
        if query.shape[1] != num_heads or query.shape[2] != head_dim:
            return _skip("query shape mismatch")
        q_3d = query
    else:
        return _skip("query dim not 2/3")

    if key.dim() == 2:
        if key.shape[1] != num_kv_heads * head_dim:
            return _skip("key shape mismatch")
        k_3d = key.view(-1, num_kv_heads, head_dim)
    elif key.dim() == 3:
        if key.shape[1] != num_kv_heads or key.shape[2] != head_dim:
            return _skip("key shape mismatch")
        k_3d = key
    else:
        return _skip("key dim not 2/3")

    if value.dim() == 2:
        if value.shape[1] != num_kv_heads * head_dim:
            return _skip("value shape mismatch")
        v_3d = value.view(-1, num_kv_heads, head_dim)
    elif value.dim() == 3:
        if value.shape[1] != num_kv_heads or value.shape[2] != head_dim:
            return _skip("value shape mismatch")
        v_3d = value
    else:
        return _skip("value dim not 2/3")

    attn_metadata, attn_layer, kv_cache = get_attention_context(attn.layer_name)
    if isinstance(attn_metadata, list):
        try:
            from vllm.v1.worker.ubatching import dbo_current_ubatch_id

            ubatch_id = dbo_current_ubatch_id()
            attn_metadata = attn_metadata[ubatch_id]
        except Exception:
            return _skip("ubatch metadata unavailable")
    if isinstance(attn_metadata, dict):
        layer_key = getattr(attn, "layer_name", None)
        if layer_key is None or layer_key not in attn_metadata:
            return _skip("layer metadata missing")
        attn_metadata = attn_metadata[layer_key]
    if attn_metadata is None or not hasattr(attn_metadata, "slot_mapping"):
        return _skip("missing slot_mapping")
    if getattr(attn_metadata, "use_cascade", False):
        return _skip("use_cascade enabled")
    if getattr(attn.impl, "dcp_world_size", 1) != 1:
        return _skip("dcp_world_size != 1")

    slot_mapping = attn_metadata.slot_mapping
    try:
        backend = attn.get_attn_backend()
        if backend.get_name() != "TRITON_ATTN":
            return _skip("backend not TRITON_ATTN")
    except Exception:
        return _skip("backend lookup failed")

    if slot_mapping is None:
        return _skip("slot_mapping missing")
    if slot_mapping.dim() > 1:
        if 1 not in slot_mapping.shape:
            return _skip("slot_mapping dim not 1")
        slot_mapping = slot_mapping.reshape(-1)
    if (
        slot_mapping.dtype != torch.long
        or slot_mapping.device != query.device
        or positions_1d.device != query.device
    ):
        return _skip("slot_mapping/positions device or dtype mismatch")
    slot_tokens = slot_mapping.numel()
    pos_tokens = positions_1d.numel()
    q_tokens = q_3d.shape[0]
    k_tokens = k_3d.shape[0]
    v_tokens = v_3d.shape[0]
    num_tokens = min(slot_tokens, pos_tokens, q_tokens, k_tokens, v_tokens)
    if num_tokens <= 0:
        return _skip("empty token set")
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
    if isinstance(num_actual_tokens, torch.Tensor) and num_actual_tokens.numel() == 1:
        num_actual_tokens = int(num_actual_tokens.item())
        if isinstance(num_actual_tokens, Integral):
            num_actual_tokens = int(num_actual_tokens)
            if 0 < num_actual_tokens < num_tokens:
                num_tokens = num_actual_tokens
    if log_fused and len({slot_tokens, pos_tokens, q_tokens, k_tokens, v_tokens}) > 1:
        _note(
            "Fused RoPE+NVFP4 token trim for %s: slot=%d pos=%d q=%d k=%d v=%d -> %d",
            layer_name,
            slot_tokens,
            pos_tokens,
            q_tokens,
            k_tokens,
            v_tokens,
            num_tokens,
        )
    if slot_mapping.numel() != num_tokens:
        slot_mapping = slot_mapping[:num_tokens]
    if positions_1d.numel() != num_tokens:
        positions_1d = positions_1d[:num_tokens]
    if q_3d.shape[0] != num_tokens:
        q_3d = q_3d[:num_tokens].contiguous()
    if k_3d.shape[0] != num_tokens:
        k_3d = k_3d[:num_tokens].contiguous()
    if v_3d.shape[0] != num_tokens:
        v_3d = v_3d[:num_tokens].contiguous()
    if not slot_mapping.is_contiguous():
        slot_mapping = slot_mapping.contiguous()
    if not positions_1d.is_contiguous():
        positions_1d = positions_1d.contiguous()

    if kv_cache.dim() < 2 or kv_cache.shape[1] != 2:
        return _skip("kv_cache shape mismatch")
    key_cache, value_cache = kv_cache.unbind(1)

    k_scale = getattr(attn_layer, "kv_cache_k_scale", None)
    v_scale = getattr(attn_layer, "kv_cache_v_scale", None)
    if (
        k_scale is None
        or v_scale is None
        or k_scale.dtype != torch.float8_e4m3fn
        or v_scale.dtype != torch.float8_e4m3fn
        or k_scale.device != query.device
        or v_scale.device != query.device
        or k_scale.dim() != 4
        or v_scale.dim() != 4
        or k_scale.stride(3) != 1
        or v_scale.stride(3) != 1
        or k_scale.stride() != v_scale.stride()
    ):
        return _skip("kv scale tensors incompatible")

    if getattr(attn, "calculate_kv_scales", False):
        k_tokens = k_3d[:num_tokens]
        v_tokens = v_3d[:num_tokens]
        valid_mask = slot_mapping >= 0
        if not bool(valid_mask.any().item()):
            return _skip("no valid tokens for nvfp4 calibration")
        if not bool(valid_mask.all().item()):
            k_tokens = k_tokens[valid_mask]
            v_tokens = v_tokens[valid_mask]
        k_calib = _calc_nvfp4_global_scale(k_tokens)
        v_calib = _calc_nvfp4_global_scale(v_tokens)
        attn_layer._k_global_scale_float = k_calib
        attn_layer._v_global_scale_float = v_calib
        attn.calculate_kv_scales = False

    k_global_scale = getattr(attn_layer, "_k_global_scale_float", 1.0)
    v_global_scale = getattr(attn_layer, "_v_global_scale_float", 1.0)

    rotary_emb._match_cos_sin_cache_dtype(query)
    if (
        rotary_emb.cos_sin_cache.dim() != 2
        or not rotary_emb.cos_sin_cache.is_contiguous()
    ):
        return _skip("cos_sin_cache shape/contiguity mismatch")

    from vllm import _custom_ops as ops

    ops.fused_rope_and_cache_flash_nvfp4(
        q_3d,
        k_3d,
        v_3d,
        key_cache,
        value_cache,
        slot_mapping,
        positions_1d,
        rotary_emb.cos_sin_cache,
        rotary_emb.is_neox_style,
        k_scale,
        v_scale,
        k_global_scale,
        v_global_scale,
    )
    if _LOG_FUSED_NVFP4 and not _is_compiling():
        _logger.info_once(
            "Using fused RoPE+NVFP4 cache path for %s.", attn.layer_name
        )
    return True
