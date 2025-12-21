# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from packaging.version import Version

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.rope_utils import try_fused_rope_and_cache_nvfp4
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import set_kv_cache_layout


def _nvfp4_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if not current_platform.has_device_capability(100):
        return False
    cuda_version = torch.version.cuda
    if cuda_version is None or Version(cuda_version) < Version("12.8"):
        return False
    return hasattr(torch, "float8_e4m3fn")


def _build_block_table(batch_size: int, max_blocks: int, device: torch.device):
    return torch.arange(
        batch_size * max_blocks, dtype=torch.int32, device=device
    ).view(batch_size, max_blocks)


def _populate_slot_mapping(
    batch_spec: BatchSpec,
    block_table: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    slot_mapping = torch.empty(
        batch_spec.compute_num_tokens(), dtype=torch.int64, device=device
    )
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1, dtype=torch.int64, device=device
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int64, device=device
    ).cumsum(0)
    for idx, (seq_len, query_len) in enumerate(
        zip(batch_spec.seq_lens, batch_spec.query_lens)
    ):
        context_len = seq_len - query_len
        token_offsets = torch.arange(
            query_len, dtype=torch.int64, device=device
        ) + context_len
        block_indices = token_offsets // block_size
        token_in_block = token_offsets % block_size
        start = int(query_start_loc[idx].item())
        end = int(query_start_loc[idx + 1].item())
        slot_mapping[start:end] = (
            block_table[idx, block_indices] * block_size + token_in_block
        )
    return slot_mapping


def _make_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    block_table: torch.Tensor,
):
    common = create_common_attn_metadata(
        batch_spec, block_size, device, arange_block_indices=False
    )
    common.block_table_tensor = block_table
    common.slot_mapping = _populate_slot_mapping(
        batch_spec, block_table, block_size, device
    )
    return common


def _make_positions(batch_spec: BatchSpec, device: torch.device) -> torch.Tensor:
    positions = []
    for seq_len, query_len in zip(batch_spec.seq_lens, batch_spec.query_lens):
        context_len = seq_len - query_len
        positions.append(
            torch.arange(
                context_len, seq_len, dtype=torch.long, device=device
            )
        )
    return torch.cat(positions)


def _build_attn_metadata(
    attn: Attention,
    common_attn_metadata,
    vllm_config,
    device: torch.device,
):
    backend_cls = attn.get_attn_backend()
    builder_cls = backend_cls.get_builder_cls()
    kv_cache_spec = attn.get_kv_cache_spec(vllm_config)
    builder = builder_cls(kv_cache_spec, [attn.layer_name], vllm_config, device)
    return builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)


def _run_attention(
    attn: Attention,
    rotary_emb: RotaryEmbedding,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    use_fused = try_fused_rope_and_cache_nvfp4(
        attn=attn,
        rotary_emb=rotary_emb,
        positions=positions,
        query=q,
        key=k,
        value=v,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if not use_fused:
        q, k = rotary_emb(positions, q, k)
    output = attn(q, None, None) if use_fused else attn(q, k, v)
    return output, use_fused


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 not supported")
@torch.inference_mode()
def test_nvfp4_fused_rope_cache_prefill_decode():
    device = torch.device("cuda")
    set_kv_cache_layout("NHD")
    current_platform.seed_everything(0)

    block_size = 16
    num_heads = 4
    num_kv_heads = 4
    head_dim = 64
    hidden_size = num_heads * head_dim
    dtype = torch.bfloat16

    batch_spec_prefill = BatchSpec(seq_lens=[8, 8], query_lens=[8, 8])
    batch_spec_decode = BatchSpec(seq_lens=[9, 9], query_lens=[1, 1])
    max_blocks = (max(batch_spec_decode.seq_lens) + block_size - 1) // block_size
    block_table = _build_block_table(batch_spec_prefill.batch_size, max_blocks, device)
    num_blocks = block_table.numel()

    vllm_config = create_vllm_config(
        model_name="meta-llama/Meta-Llama-3-8B",
        max_model_len=128,
        block_size=block_size,
        dtype=dtype,
        enable_chunked_prefill=False,
        hf_config_override={
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "max_position_embeddings": 128,
        },
    )
    vllm_config.attention_config.nvfp4_backend = AttentionBackendEnum.TRITON_ATTN

    cache_config_nvfp4 = CacheConfig(
        block_size=block_size, cache_dtype="nvfp4", swap_space=0
    )
    cache_config_bf16 = CacheConfig(
        block_size=block_size, cache_dtype="auto", swap_space=0
    )

    with set_current_vllm_config(vllm_config):
        attn_nvfp4 = Attention(
            num_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config_nvfp4,
            prefix="test_nvfp4_attn",
        )
        attn_bf16 = Attention(
            num_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config_bf16,
            prefix="test_bf16_attn",
        )
        rotary_emb = RotaryEmbedding(
            head_dim,
            head_dim,
            128,
            10000.0,
            True,
            dtype,
        )

        packed_head_size = head_dim // 8
        kv_cache_nvfp4 = torch.empty(
            num_blocks,
            2,
            block_size,
            num_kv_heads,
            packed_head_size,
            dtype=torch.int32,
            device=device,
        )
        kv_cache_bf16 = torch.empty(
            num_blocks,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        attn_nvfp4.kv_cache[0] = kv_cache_nvfp4
        attn_bf16.kv_cache[0] = kv_cache_bf16

        scale_shape = (num_blocks, block_size, num_kv_heads, head_dim // 16)
        attn_nvfp4.kv_cache_k_scale = torch.empty(
            scale_shape, dtype=torch.float8_e4m3fn, device=device
        )
        attn_nvfp4.kv_cache_v_scale = torch.empty(
            scale_shape, dtype=torch.float8_e4m3fn, device=device
        )

        common_prefill = _make_common_attn_metadata(
            batch_spec_prefill, block_size, device, block_table
        )
        attn_metadata_prefill = {
            attn_nvfp4.layer_name: _build_attn_metadata(
                attn_nvfp4, common_prefill, vllm_config, device
            ),
            attn_bf16.layer_name: _build_attn_metadata(
                attn_bf16, common_prefill, vllm_config, device
            ),
        }
        q = torch.randn(
            batch_spec_prefill.compute_num_tokens(),
            num_heads * head_dim,
            dtype=dtype,
            device=device,
        )
        k = torch.randn(
            batch_spec_prefill.compute_num_tokens(),
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
        )
        v = torch.randn_like(k)
        positions = _make_positions(batch_spec_prefill, device)

        with set_forward_context(attn_metadata_prefill, vllm_config):
            q_nvfp4 = q.clone()
            k_nvfp4 = k.clone()
            v_nvfp4 = v.clone()
            q_bf16 = q.clone()
            k_bf16 = k.clone()
            v_bf16 = v.clone()
            out_nvfp4, used_fused_prefill = _run_attention(
                attn_nvfp4,
                rotary_emb,
                positions,
                q_nvfp4,
                k_nvfp4,
                v_nvfp4,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            out_bf16, _ = _run_attention(
                attn_bf16,
                rotary_emb,
                positions,
                q_bf16,
                k_bf16,
                v_bf16,
                num_heads,
                num_kv_heads,
                head_dim,
            )

        assert used_fused_prefill
        atol = 0.2
        rtol = 0.2
        if current_platform.has_device_capability(120):
            # SM120 NVFP4 quantization can introduce slightly larger error.
            atol = 0.7
        torch.testing.assert_close(out_nvfp4, out_bf16, atol=atol, rtol=rtol)

        common_decode = _make_common_attn_metadata(
            batch_spec_decode, block_size, device, block_table
        )
        attn_metadata_decode = {
            attn_nvfp4.layer_name: _build_attn_metadata(
                attn_nvfp4, common_decode, vllm_config, device
            ),
            attn_bf16.layer_name: _build_attn_metadata(
                attn_bf16, common_decode, vllm_config, device
            ),
        }
        q = torch.randn(
            batch_spec_decode.compute_num_tokens(),
            num_heads * head_dim,
            dtype=dtype,
            device=device,
        )
        k = torch.randn(
            batch_spec_decode.compute_num_tokens(),
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
        )
        v = torch.randn_like(k)
        positions = _make_positions(batch_spec_decode, device)

        with set_forward_context(attn_metadata_decode, vllm_config):
            q_nvfp4 = q.clone()
            k_nvfp4 = k.clone()
            v_nvfp4 = v.clone()
            q_bf16 = q.clone()
            k_bf16 = k.clone()
            v_bf16 = v.clone()
            out_nvfp4, used_fused_decode = _run_attention(
                attn_nvfp4,
                rotary_emb,
                positions,
                q_nvfp4,
                k_nvfp4,
                v_nvfp4,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            out_bf16, _ = _run_attention(
                attn_bf16,
                rotary_emb,
                positions,
                q_bf16,
                k_bf16,
                v_bf16,
                num_heads,
                num_kv_heads,
                head_dim,
            )

        assert used_fused_decode
        atol = 0.2
        rtol = 0.2
        if current_platform.has_device_capability(120):
            atol = 0.7
        torch.testing.assert_close(out_nvfp4, out_bf16, atol=atol, rtol=rtol)
