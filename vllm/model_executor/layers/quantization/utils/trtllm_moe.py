# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRTLLM MoE custom-op helpers."""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.utils.trtllm import (
    has_trtllm_fp4_block_scale_moe,
    has_trtllm_fp8_block_scale_moe,
)

logger = init_logger(__name__)


def trtllm_moe_enabled() -> bool:
    return envs.VLLM_USE_TRTLLM_MOE


def can_use_trtllm_fp8_moe() -> bool:
    return trtllm_moe_enabled() and has_trtllm_fp8_block_scale_moe()


def can_use_trtllm_fp4_moe() -> bool:
    return trtllm_moe_enabled() and has_trtllm_fp4_block_scale_moe()


def warn_trtllm_moe_unavailable(reason: str) -> None:
    logger.warning_once(
        "TRTLLM MoE custom ops requested, but unavailable (%s); "
        "falling back to FlashInfer.",
        reason,
    )


def trtllm_fp8_block_scale_moe(
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    num_expert_group: int | None,
    topk_group: int | None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if not has_trtllm_fp8_block_scale_moe():
        raise RuntimeError("TRTLLM FP8 block-scale MoE kernel unavailable.")

    if (topk_weights is None) ^ (topk_ids is None):
        raise ValueError(
            "TRTLLM FP8 MoE requires both topk_weights and topk_ids or neither."
        )

    x_fp8, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)

    routing_bias = layer.e_score_correction_bias
    if routing_bias is not None:
        routing_bias = routing_bias.to(torch.bfloat16)

    routing_method_type = layer.routing_method_type
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    routing_logits = router_logits
    if topk_weights is not None:
        routing_logits = None
        routing_bias = None
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

    return torch.ops.trtllm.fp8_block_scale_moe_runner(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=x_fp8,
        hidden_states_scale=x_sf,
        gemm1_weights=layer.w13_weight,
        gemm1_weights_scale=layer.w13_weight_scale_inv,
        gemm2_weights=layer.w2_weight,
        gemm2_weights_scale=layer.w2_weight_scale_inv,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group if num_expert_group is not None else 0,
        topk_group=topk_group if topk_group is not None else 0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=layer.routed_scaling_factor,
        routing_method_type=routing_method_type,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def trtllm_fp4_block_scale_moe(
    layer: torch.nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    router_logits: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    num_expert_group: int | None,
    topk_group: int | None,
    custom_routing_function: object | None,
    e_score_correction_bias: torch.Tensor | None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if not has_trtllm_fp4_block_scale_moe():
        raise RuntimeError("TRTLLM FP4 block-scale MoE kernel unavailable.")

    if (topk_weights is None) ^ (topk_ids is None):
        raise ValueError(
            "TRTLLM FP4 MoE requires both topk_weights and topk_ids or neither."
        )

    if isinstance(x, tuple):
        hidden_states_fp4, hidden_states_scale_linear_fp4 = x
    else:
        a1_gscale = layer.w13_input_scale_quant
        hidden_states_fp4, hidden_states_scale_linear_fp4 = (
            torch.ops.trtllm.fp4_quantize(
                x,
                a1_gscale,
                layer.quant_config.group_size,
                False,
                False,
            )
        )

    use_llama4_routing = False
    if custom_routing_function is not None:
        from vllm.model_executor.models.llama4 import Llama4MoE

        use_llama4_routing = (
            custom_routing_function is Llama4MoE.custom_routing_function
        )

    routing_method_type = layer.routing_method_type
    if use_llama4_routing:
        routing_method_type = RoutingMethodType.Llama4

    routing_bias = e_score_correction_bias
    if routing_bias is not None:
        routing_bias = routing_bias.to(torch.bfloat16)

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    routing_logits = router_logits
    if topk_weights is not None:
        routing_logits = None
        routing_bias = None
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

    outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale_linear_fp4.view(
            torch.float8_e4m3fn
        ).flatten(),
        gemm1_weights=layer.w13_weight.data,
        gemm1_weights_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer.w2_weight.data,
        gemm2_weights_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=layer.g1_scale_c.data,
        output1_scale_gate_scalar=layer.g1_alphas.data,
        output2_scale_scalar=layer.g2_alphas.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group if num_expert_group is not None else 0,
        topk_group=topk_group if topk_group is not None else 0,
        intermediate_size=layer.intermediate_size_per_partition,
        valid_hidden_size=None,
        valid_intermediate_size=None,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type,
        do_finalize=True,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    return outputs[0]
