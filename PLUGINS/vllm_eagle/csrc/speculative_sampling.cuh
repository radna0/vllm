/*
 * Copyright (c) 2025 by SGLang team.
 * Ported to vLLM by Antigravity.
 */
#ifndef SPECULATIVE_SAMPLING_CUH_
#define SPECULATIVE_SAMPLING_CUH_

// Undefine Torch-inserted macros that break flashinfer/CUDA conversion operators
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

#include <assert.h>
#include <flashinfer/sampling.cuh>

namespace flashinfer {
namespace sampling {

using namespace cub;

template <typename T>
__device__ __forceinline__ float to_f32(T x) {
  return (float)x;
}
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ float to_f32(half x) { return __half2float(x); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
#endif

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    typename DType,
    typename IdType,
    typename IdType2>
__global__ void TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,          // mutable
    IdType* accept_index,      // mutable
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    float threshold_single,
    float threshold_acc) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

  float prob_acc = 0.0f;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  float coin = to_f32(uniform_samples[bx * num_draft_tokens]);
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      float target_prob_single = to_f32(target_probs[cur_prob_offset + draft_token_id]);
      prob_acc += target_prob_single;

      if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
        // accept token
        prob_acc = 0.0f;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = to_f32(uniform_samples[bx * num_draft_tokens + cur_index]);
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] = target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;

  coin = to_f32(uniform_samples_for_final_sampling[bx]);

  float sum_relu_q_minus_p(0.0f);
  vec_t<DType, VEC_SIZE> q_vec, p_vec;
  float relu_q_minus_p[VEC_SIZE];
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(to_f32(q_vec[j]) - to_f32(p_vec[j]), 0.0f);
    }
    sum_relu_q_minus_p += BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                              .Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }
  temp_storage.sampled_id = d - 1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  float u = coin * sum_relu_q_minus_p;

  float aggregate_relu_q_minus_p(0.0f);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    vec_t<float, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(to_f32(q_vec[j]) - to_f32(p_vec[j]), 0.0f);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        i, d, [&](float x) { return x > 0.0f; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  predicts[last_accepted_retrive_idx] = temp_storage.sampled_id;
}

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    typename DType,
    typename IdType,
    typename IdType2>
void TreeSpeculativeSamplingTargetOnlyLauncher(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    DType* draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    float threshold_single,
    float threshold_acc,
    cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(BLOCK_THREADS);
  uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>);
  TreeSpeculativeSamplingTargetOnly<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, VEC_SIZE, DETERMINISTIC, DType, IdType,
                                    IdType2><<<grid, block, smem_size, stream>>>(
      predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling,
      uniform_samples, uniform_samples_for_final_sampling, target_probs, draft_probs, batch_size,
      num_speculative_tokens, num_draft_tokens, d, threshold_single, threshold_acc);
}

} // namespace sampling
} // namespace flashinfer

#endif
