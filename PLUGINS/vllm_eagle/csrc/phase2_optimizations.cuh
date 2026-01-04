/*
 * Phase 2 Kernel Optimizations - Header File
 * Warp-Level Primitives + Fused Draft-Verify Kernel
 */

#pragma once

#include <torch/extension.h>

// Optimized Gumbel sampling with warp-level primitives
void fused_gumbel_sample_warp_optimized(
    torch::Tensor& out_tokens,
    torch::Tensor& logits,
    torch::Tensor& uniform_samples,
    torch::Tensor& min_p,
    torch::Tensor& temperatures
);

// Fused draft-verify-sample kernel
void fused_draft_verify_sample(
    torch::Tensor& accepted_tokens,
    torch::Tensor& num_accepted,
    torch::Tensor& draft_logits,
    torch::Tensor& target_logits,
    torch::Tensor& uniform_samples,
    torch::Tensor& verify_samples,
    torch::Tensor& min_p,
    torch::Tensor& temperatures
);
