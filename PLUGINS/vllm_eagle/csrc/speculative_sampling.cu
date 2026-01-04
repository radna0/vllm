/*
 * Copyright (c) 2025 by SGLang team.
 * Ported to vLLM by Antigravity.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Undefine Torch-inserted macros that break flashinfer/CUDA conversion operators
// MUST BE AT THE ABSOLUTE TOP before any other includes
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "speculative_sampling.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#ifndef BLOCK_SCAN_WARP_SCAN
#define BLOCK_SCAN_WARP_SCAN cub::BLOCK_SCAN_WARP_SCANS
#endif

#ifndef BLOCK_REDUCE_WARP_REDUCE
#define BLOCK_REDUCE_WARP_REDUCE cub::BLOCK_REDUCE_WARP_REDUCTIONS
#endif

void tree_speculative_sampling_target_only(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,  // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor uniform_samples,
    at::Tensor uniform_samples_for_final_sampling,
    at::Tensor target_probs,
    at::Tensor draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(uniform_samples_for_final_sampling);
  CHECK_INPUT(target_probs);
  auto device = target_probs.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_EQ(uniform_samples_for_final_sampling.device(), device);
  CHECK_EQ(target_probs.device(), device);
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, uniform_samples);
  CHECK_DIM(3, target_probs);
  CHECK_DIM(3, draft_probs);

  unsigned int batch_size = uniform_samples.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_probs.size(2);

  CHECK_EQ(batch_size, candidates.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(num_draft_tokens, uniform_samples.size(1));
  CHECK_EQ(num_draft_tokens, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));

  TORCH_CHECK(predicts.scalar_type() == at::kInt, "Expected 'predicts' to be int32");
  TORCH_CHECK(accept_index.scalar_type() == at::kInt, "Expected 'accept_index' to be int32");
  TORCH_CHECK(accept_token_num.scalar_type() == at::kInt, "Expected 'accept_token_num' to be int32");
  TORCH_CHECK(candidates.scalar_type() == at::kLong, "Expected 'candidates' to be int64");
  TORCH_CHECK(retrive_index.scalar_type() == at::kLong, "Expected 'retrive_index' to be int64");
  TORCH_CHECK(retrive_next_token.scalar_type() == at::kLong, "Expected 'retrive_next_token' to be int64");
  TORCH_CHECK(retrive_next_sibling.scalar_type() == at::kLong, "Expected 'retrive_next_sibling' to be int64");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (target_probs.scalar_type() == at::ScalarType::Float) {
    using scalar_t = float;
    if (deterministic) {
      flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, true,
                                                          scalar_t, int, int64_t>(
          (int*)predicts.data_ptr(), (int*)accept_index.data_ptr(), (int*)accept_token_num.data_ptr(),
          (int64_t*)candidates.data_ptr(), (int64_t*)retrive_index.data_ptr(), (int64_t*)retrive_next_token.data_ptr(),
          (int64_t*)retrive_next_sibling.data_ptr(), (scalar_t*)uniform_samples.data_ptr(),
          (scalar_t*)uniform_samples_for_final_sampling.data_ptr(), (scalar_t*)target_probs.data_ptr(),
          (scalar_t*)draft_probs.data_ptr(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          threshold_single, threshold_acc, stream);
    } else {
      flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, false,
                                                          scalar_t, int, int64_t>(
          (int*)predicts.data_ptr(), (int*)accept_index.data_ptr(), (int*)accept_token_num.data_ptr(),
          (int64_t*)candidates.data_ptr(), (int64_t*)retrive_index.data_ptr(), (int64_t*)retrive_next_token.data_ptr(),
          (int64_t*)retrive_next_sibling.data_ptr(), (scalar_t*)uniform_samples.data_ptr(),
          (scalar_t*)uniform_samples_for_final_sampling.data_ptr(), (scalar_t*)target_probs.data_ptr(),
          (scalar_t*)draft_probs.data_ptr(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          threshold_single, threshold_acc, stream);
    }
  } else if (target_probs.scalar_type() == at::ScalarType::Half) {
    using scalar_t = half;
    if (deterministic) {
      flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, true,
                                                          scalar_t, int, int64_t>(
          (int*)predicts.data_ptr(), (int*)accept_index.data_ptr(), (int*)accept_token_num.data_ptr(),
          (int64_t*)candidates.data_ptr(), (int64_t*)retrive_index.data_ptr(), (int64_t*)retrive_next_token.data_ptr(),
          (int64_t*)retrive_next_sibling.data_ptr(), (scalar_t*)uniform_samples.data_ptr(),
          (scalar_t*)uniform_samples_for_final_sampling.data_ptr(), (scalar_t*)target_probs.data_ptr(),
          (scalar_t*)draft_probs.data_ptr(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          threshold_single, threshold_acc, stream);
    } else {
      flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, false,
                                                          scalar_t, int, int64_t>(
          (int*)predicts.data_ptr(), (int*)accept_index.data_ptr(), (int*)accept_token_num.data_ptr(),
          (int64_t*)candidates.data_ptr(), (int64_t*)retrive_index.data_ptr(), (int64_t*)retrive_next_token.data_ptr(),
          (int64_t*)retrive_next_sibling.data_ptr(), (scalar_t*)uniform_samples.data_ptr(),
          (scalar_t*)uniform_samples_for_final_sampling.data_ptr(), (scalar_t*)target_probs.data_ptr(),
          (scalar_t*)draft_probs.data_ptr(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          threshold_single, threshold_acc, stream);
    }
  } else if (target_probs.scalar_type() == at::ScalarType::BFloat16) {
    using scalar_t = __nv_bfloat16;
    if (deterministic) {
      flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, true,
                                                          scalar_t, int, int64_t>(
          (int*)predicts.data_ptr(), (int*)accept_index.data_ptr(), (int*)accept_token_num.data_ptr(),
          (int64_t*)candidates.data_ptr(), (int64_t*)retrive_index.data_ptr(), (int64_t*)retrive_next_token.data_ptr(),
          (int64_t*)retrive_next_sibling.data_ptr(), (scalar_t*)uniform_samples.data_ptr(),
          (scalar_t*)uniform_samples_for_final_sampling.data_ptr(), (scalar_t*)target_probs.data_ptr(),
          (scalar_t*)draft_probs.data_ptr(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          threshold_single, threshold_acc, stream);
    } else {
      flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, false,
                                                          scalar_t, int, int64_t>(
          (int*)predicts.data_ptr(), (int*)accept_index.data_ptr(), (int*)accept_token_num.data_ptr(),
          (int64_t*)candidates.data_ptr(), (int64_t*)retrive_index.data_ptr(), (int64_t*)retrive_next_token.data_ptr(),
          (int64_t*)retrive_next_sibling.data_ptr(), (scalar_t*)uniform_samples.data_ptr(),
          (scalar_t*)uniform_samples_for_final_sampling.data_ptr(), (scalar_t*)target_probs.data_ptr(),
          (scalar_t*)draft_probs.data_ptr(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          threshold_single, threshold_acc, stream);
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype for tree_speculative_sampling_target_only");
  }
}

// Optimized Kernels for FlashSampling
#include <cub/cub.cuh>
#include <curand_kernel.h>

#define NEG_INF_VAL -1e34f

namespace vllm_eagle {

// Noise-scaling Gumbel sampling helper
__device__ __forceinline__ float sample_scaled_gumbel(float u, float temperature) {
    // Identity: argmax(y/T + g) == argmax(y + T*g)
    // We use y + T*g to preserve numerical precision of raw logits
    return temperature * -logf(-logf(fmaxf(u, 1e-10f)));
}

template <typename scalar_t>
__global__ void apply_logit_filters_kernel(
    scalar_t* logits,          // [batch_size, vocab_size] (RAW LOGITS)
    const int* top_k,          // [batch_size]
    const float* top_p,        // [batch_size]
    const float* min_p,        // [batch_size]
    const int batch_size,
    const int vocab_size
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    scalar_t* row_logits = logits + row * vocab_size;
    int row_top_k = top_k[row];
    float row_min_p = min_p[row];
    
    // 1. Find Max Logit from RAW distribution
    float local_max = -1e34f;
    float local_min = 1e34f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = (float)row_logits[i];
        local_max = fmaxf(local_max, val);
        local_min = fminf(local_min, val);
    }

    typedef cub::BlockReduce<float, 1024> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_storage_f;
    float aggregate_max = BlockReduceFloat(temp_storage_f).Reduce(local_max, cub::Max());
    __syncthreads();
    float aggregate_min = BlockReduceFloat(temp_storage_f).Reduce(local_min, cub::Min());

    __shared__ float s_max_logit;
    __shared__ float s_min_logit;
    if (threadIdx.x == 0) {
        s_max_logit = aggregate_max;
        s_min_logit = aggregate_min;
    }
    __syncthreads();

    float threshold = -1e34f;

    // 2. Min-P Threshold (using RAW logits per paper)
    float min_p_threshold = s_max_logit + logf(row_min_p + 1e-10f);
    threshold = fmaxf(threshold, min_p_threshold);

    // 3. Top-K Threshold (Binary Search if K is valid)
    if (row_top_k > 0 && row_top_k < vocab_size) {
        float lo = s_min_logit;
        float hi = s_max_logit;
        for (int iter = 0; iter < 16; ++iter) {
            float mid = (lo + hi) * 0.5f;
            int count = 0;
            for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
                if ((float)row_logits[i] >= mid) count++;
            }
            typedef cub::BlockReduce<int, 1024> BlockReduceInt;
            __shared__ typename BlockReduceInt::TempStorage temp_storage_i;
            int aggregate_count = BlockReduceInt(temp_storage_i).Sum(count);
            __syncthreads();
            
            __shared__ int s_count;
            if (threadIdx.x == 0) s_count = aggregate_count;
            __syncthreads();

            if (s_count >= row_top_k) lo = mid;
            else hi = mid;
        }
        threshold = fmaxf(threshold, lo);
    }
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        if ((float)row_logits[i] < threshold) {
            row_logits[i] = (scalar_t)NEG_INF_VAL;
        }
    }
}

template <typename scalar_t>
__global__ void fused_gumbel_sample_kernel(
    int* out_tokens,
    const scalar_t* logits,          // [batch_size, vocab_size] (RAW LOGITS)
    const uint64_t seed,
    const uint64_t offset,
    const float* min_p,
    const float* temperatures,       // [batch_size]
    const int batch_size,
    const int vocab_size
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const scalar_t* row_logits = logits + row * vocab_size;
    float temp = temperatures[row];

    // 1. Find Max Logit for Min-P filtering
    float local_max = -1e34f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, (float)row_logits[i]);
    }

    typedef cub::BlockReduce<float, 1024> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_storage_f;
    float aggregate_max = BlockReduceFloat(temp_storage_f).Reduce(local_max, cub::Max());
    __syncthreads();

    __shared__ float s_max_logit;
    if (threadIdx.x == 0) s_max_logit = aggregate_max;
    __syncthreads();
    
    // 2. Threshold from RAW distribution
    float threshold = s_max_logit + logf(min_p[row] + 1e-10f);

    // 3. Noise-Scaled Gumbel + ArgMax
    // argmax(logit + temp * gumbel)
    float best_val = -1e34f;
    int best_idx = -1;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit = (float)row_logits[i];
        if (logit >= threshold) {
            // Generate Gumbel noise on-the-fly using Philox RNG
            curandStatePhilox4_32_10_t state;
            curand_init(seed, row * vocab_size + i, offset, &state);
            float u = curand_uniform(&state);
            
            float g_logit = logit + sample_scaled_gumbel(u, temp);
            if (g_logit > best_val) {
                best_val = g_logit;
                best_idx = i;
            }
        }
    }

    // Block-level reduction for ArgMax
    struct ValIdx { float val; int idx; };
    struct ArgMaxOp {
        __device__ __forceinline__ ValIdx operator()(const ValIdx& a, const ValIdx& b) const {
            if (a.idx == -1) return b;
            if (b.idx == -1) return a;
            return (a.val > b.val) ? a : b;
        }
    };

    typedef cub::BlockReduce<ValIdx, 1024> BlockReduceArgMax;
    __shared__ typename BlockReduceArgMax::TempStorage temp_storage_argmax;
    
    ValIdx local_best = {best_val, best_idx};
    ValIdx global_best = BlockReduceArgMax(temp_storage_argmax).Reduce(local_best, ArgMaxOp());

    if (threadIdx.x == 0) {
        out_tokens[row] = global_best.idx;
    }
}

} // namespace vllm_eagle

void apply_logit_filters(torch::Tensor& logits, torch::Tensor& top_k,
                         torch::Tensor& top_p, torch::Tensor& min_p) {
    CHECK_INPUT(logits);
    int batch_size = logits.size(0);
    int vocab_size = logits.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAGuard device_guard(logits.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "apply_logit_filters", [&] {
        vllm_eagle::apply_logit_filters_kernel<scalar_t><<<batch_size, 1024, 0, stream>>>(
            logits.data_ptr<scalar_t>(),
            top_k.data_ptr<int>(),
            top_p.data_ptr<float>(),
            min_p.data_ptr<float>(),
            batch_size,
            vocab_size
        );
    });
}

void fused_gumbel_sample(torch::Tensor& out_tokens, torch::Tensor& logits,
                         torch::Tensor& top_k, torch::Tensor& top_p,
                         torch::Tensor& min_p, torch::Tensor& temperatures,
                         uint64_t seed, uint64_t offset) {
    CHECK_INPUT(logits);
    int batch_size = logits.size(0);
    int vocab_size = logits.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAGuard device_guard(logits.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "fused_gumbel_sample", [&] {
        vllm_eagle::fused_gumbel_sample_kernel<scalar_t><<<batch_size, 1024, 0, stream>>>(
            out_tokens.data_ptr<int>(),
            logits.data_ptr<scalar_t>(),
            seed,
            offset,
            min_p.data_ptr<float>(),
            temperatures.data_ptr<float>(),
            batch_size,
            vocab_size
        );
    });
}
