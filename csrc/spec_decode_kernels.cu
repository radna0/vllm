#include <torch/extension.h>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <torch/cuda.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#if defined(__has_include)
#if __has_include(<c10/util/Float8_e4m3fn.h>)
#define VLLM_HAS_FP8_TYPES 1
#else
#define VLLM_HAS_FP8_TYPES 0
#endif
#else
#define VLLM_HAS_FP8_TYPES 0
#endif

namespace vllm {

constexpr int kEagleTopKMax = 64;
constexpr int kEagleTopKBlockSize = 256;

template <typename T>
__host__ __device__ __forceinline__ T eagle_div_up(T m, T n) {
  return (m + n - 1) / n;
}

__device__ __forceinline__ int32_t eagle_flat_index3(int32_t i,
                                                     int32_t j,
                                                     int32_t k,
                                                     int32_t dim1,
                                                     int32_t dim2) {
  return (i * dim1 + j) * dim2 + k;
}

template <typename T>
__device__ __forceinline__ float eagle_to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float eagle_to_float<at::Half>(at::Half v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float eagle_to_float<at::BFloat16>(
    at::BFloat16 v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__global__ void eagle_topk_logits_custom_kernel(
    const scalar_t* __restrict__ logits,
    int64_t logits_stride0,
    int64_t logits_stride1,
    int32_t vocab_size,
    int32_t top_k,
    int64_t* __restrict__ out_ids,
    int64_t out_ids_stride0,
    int64_t out_ids_stride1,
    float* __restrict__ out_logprobs,
    int64_t out_lp_stride0,
    int64_t out_lp_stride1) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  if (top_k <= 0) {
    return;
  }
  __shared__ float s_thread_vals[kEagleTopKBlockSize];
  __shared__ int32_t s_thread_idx[kEagleTopKBlockSize];
  __shared__ float s_best_vals[kEagleTopKMax];
  __shared__ int32_t s_best_idx[kEagleTopKMax];
  __shared__ float s_row_max;
  __shared__ float s_logsumexp;

  const scalar_t* row_ptr =
      logits + static_cast<int64_t>(row) * logits_stride0;

  float local_max = -INFINITY;
  for (int32_t i = tid; i < vocab_size; i += kEagleTopKBlockSize) {
    float v = eagle_to_float(row_ptr[static_cast<int64_t>(i) * logits_stride1]);
    if (v > local_max) {
      local_max = v;
    }
  }
  s_thread_vals[tid] = local_max;
  __syncthreads();
  if (tid == 0) {
    float max_val = s_thread_vals[0];
    for (int32_t i = 1; i < kEagleTopKBlockSize; ++i) {
      if (s_thread_vals[i] > max_val) {
        max_val = s_thread_vals[i];
      }
    }
    s_row_max = max_val;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int32_t i = tid; i < vocab_size; i += kEagleTopKBlockSize) {
    float v = eagle_to_float(row_ptr[static_cast<int64_t>(i) * logits_stride1]);
    local_sum += expf(v - s_row_max);
  }
  s_thread_vals[tid] = local_sum;
  __syncthreads();
  if (tid == 0) {
    float sum_val = 0.0f;
    for (int32_t i = 0; i < kEagleTopKBlockSize; ++i) {
      sum_val += s_thread_vals[i];
    }
    s_logsumexp = logf(sum_val) + s_row_max;
  }
  __syncthreads();

  if (tid == 0) {
    for (int32_t k = 0; k < top_k; ++k) {
      s_best_vals[k] = -INFINITY;
      s_best_idx[k] = -1;
    }
  }
  __syncthreads();

  for (int32_t k = 0; k < top_k; ++k) {
    float tmax = -INFINITY;
    int32_t tidx = -1;
    for (int32_t i = tid; i < vocab_size; i += kEagleTopKBlockSize) {
      bool used = false;
      for (int32_t j = 0; j < k; ++j) {
        if (s_best_idx[j] == i) {
          used = true;
          break;
        }
      }
      if (used) {
        continue;
      }
      float v = eagle_to_float(row_ptr[static_cast<int64_t>(i) * logits_stride1]);
      if (v > tmax) {
        tmax = v;
        tidx = i;
      }
    }
    s_thread_vals[tid] = tmax;
    s_thread_idx[tid] = tidx;
    __syncthreads();
    if (tid == 0) {
      float best_val = s_thread_vals[0];
      int32_t best_idx = s_thread_idx[0];
      for (int32_t i = 1; i < kEagleTopKBlockSize; ++i) {
        float v = s_thread_vals[i];
        int32_t idx = s_thread_idx[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
          best_val = v;
          best_idx = idx;
        }
      }
      s_best_vals[k] = best_val;
      s_best_idx[k] = best_idx;
    }
    __syncthreads();
  }

  if (tid == 0) {
    int64_t* out_id_row =
        out_ids + static_cast<int64_t>(row) * out_ids_stride0;
    float* out_lp_row =
        out_logprobs + static_cast<int64_t>(row) * out_lp_stride0;
    for (int32_t k = 0; k < top_k; ++k) {
      out_id_row[static_cast<int64_t>(k) * out_ids_stride1] =
          static_cast<int64_t>(s_best_idx[k]);
      out_lp_row[static_cast<int64_t>(k) * out_lp_stride1] =
          s_best_vals[k] - s_logsumexp;
    }
  }
}

__global__ void eagle_topk_small_kernel(
    const float* __restrict__ scores,
    int64_t scores_stride0,
    int64_t scores_stride1,
    float* __restrict__ out_scores,
    int64_t out_scores_stride0,
    int64_t out_scores_stride1,
    int32_t* __restrict__ out_indices,
    int64_t out_indices_stride0,
    int64_t out_indices_stride1,
    int32_t batch_size,
    int32_t num_scores,
    int32_t top_k) {
  const int32_t bix = static_cast<int32_t>(blockIdx.x);
  if (bix >= batch_size || top_k <= 0) {
    return;
  }
  float best_vals[kEagleTopKMax];
  int32_t best_idx[kEagleTopKMax];
  for (int32_t i = 0; i < top_k; ++i) {
    best_vals[i] = -INFINITY;
    best_idx[i] = -1;
  }
  const float* row = scores + static_cast<int64_t>(bix) * scores_stride0;
  for (int32_t j = 0; j < num_scores; ++j) {
    const float v = row[static_cast<int64_t>(j) * scores_stride1];
    if (v <= best_vals[top_k - 1]) {
      continue;
    }
    int32_t pos = top_k - 1;
    while (pos > 0 && v > best_vals[pos - 1]) {
      best_vals[pos] = best_vals[pos - 1];
      best_idx[pos] = best_idx[pos - 1];
      --pos;
    }
    best_vals[pos] = v;
    best_idx[pos] = j;
  }
  float* out_row = out_scores + static_cast<int64_t>(bix) * out_scores_stride0;
  int32_t* out_idx_row =
      out_indices + static_cast<int64_t>(bix) * out_indices_stride0;
  for (int32_t i = 0; i < top_k; ++i) {
    out_row[static_cast<int64_t>(i) * out_scores_stride1] = best_vals[i];
    out_idx_row[static_cast<int64_t>(i) * out_indices_stride1] = best_idx[i];
  }
}

std::vector<torch::Tensor> eagle_topk_small(torch::Tensor scores,
                                            int64_t top_k) {
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32,
              "scores must be float32");
  TORCH_CHECK(scores.dim() == 2, "scores must be 2D");
  const int32_t batch_size = static_cast<int32_t>(scores.size(0));
  const int32_t num_scores = static_cast<int32_t>(scores.size(1));
  TORCH_CHECK(top_k > 0, "top_k must be > 0");
  TORCH_CHECK(top_k <= kEagleTopKMax,
              "top_k exceeds kernel limit");
  TORCH_CHECK(num_scores >= top_k,
              "num_scores must be >= top_k");

  auto out_scores =
      torch::empty({batch_size, top_k}, scores.options());
  auto out_indices =
      torch::empty({batch_size, top_k},
                   scores.options().dtype(torch::kInt32));
  if (batch_size == 0) {
    return {out_scores, out_indices};
  }
  c10::cuda::CUDAGuard device_guard(scores.device());
  const dim3 grid(batch_size);
  const dim3 block(1);
  eagle_topk_small_kernel<<<grid, block, 0,
                            at::cuda::getDefaultCUDAStream()>>>(
      scores.data_ptr<float>(),
      scores.stride(0),
      scores.stride(1),
      out_scores.data_ptr<float>(),
      out_scores.stride(0),
      out_scores.stride(1),
      out_indices.data_ptr<int32_t>(),
      out_indices.stride(0),
      out_indices.stride(1),
      batch_size,
      num_scores,
      static_cast<int32_t>(top_k));
  return {out_scores, out_indices};
}

__global__ void pack_accepted_tokens_kernel(
    const int32_t* __restrict__ output_token_ids,
    int64_t output_stride,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ packed,
    int64_t max_len) {
  const int req_idx = blockIdx.x;
  const int32_t count = counts[req_idx];
  if (count <= 0) {
    return;
  }
  const int32_t offset = offsets[req_idx];
  const int64_t row_start = static_cast<int64_t>(req_idx) * output_stride;
  for (int i = threadIdx.x; i < count && i < max_len; i += blockDim.x) {
    packed[offset + i] = output_token_ids[row_start + i];
  }
}

torch::Tensor pack_accepted_tokens(torch::Tensor output_token_ids,
                                   torch::Tensor offsets,
                                   torch::Tensor counts,
                                   int64_t total_tokens) {
  TORCH_CHECK(output_token_ids.is_cuda(),
              "output_token_ids must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
  TORCH_CHECK(counts.is_cuda(), "counts must be a CUDA tensor");
  TORCH_CHECK(output_token_ids.scalar_type() == torch::kInt32,
              "output_token_ids must be int32");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32,
              "offsets must be int32");
  TORCH_CHECK(counts.scalar_type() == torch::kInt32,
              "counts must be int32");
  TORCH_CHECK(output_token_ids.dim() == 2,
              "output_token_ids must be 2D");
  TORCH_CHECK(offsets.numel() == counts.numel(),
              "offsets and counts must have same length");

  auto device = output_token_ids.device();
  c10::cuda::CUDAGuard device_guard(device);
  auto packed = torch::empty({total_tokens}, output_token_ids.options());
  if (total_tokens == 0) {
    return packed;
  }
  const int64_t batch_size = output_token_ids.size(0);
  const int64_t max_len = output_token_ids.size(1);
  const int threads = 256;
  const dim3 grid(batch_size);
  const dim3 block(threads);
  pack_accepted_tokens_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
      output_token_ids.data_ptr<int32_t>(),
      output_token_ids.stride(0),
      offsets.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      packed.data_ptr<int32_t>(),
      max_len);
  return packed;
}

__global__ void build_spec_decode_indices_kernel(
    const int32_t* __restrict__ num_draft_tokens,
    const int32_t* __restrict__ cu_num_scheduled_tokens,
    int32_t* __restrict__ cu_num_draft_tokens,
    int32_t* __restrict__ cu_num_sampled_tokens,
    int32_t* __restrict__ logits_indices,
    int32_t* __restrict__ target_logits_indices,
    int32_t* __restrict__ bonus_logits_indices,
    int32_t batch_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int32_t draft_offset = 0;
  int32_t sampled_offset = 0;
  int32_t logits_offset = 0;
  int32_t target_offset = 0;
  for (int32_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    const int32_t num_draft = num_draft_tokens[req_idx];
    const int32_t num_sampled = num_draft + 1;
    const int32_t logits_start =
        cu_num_scheduled_tokens[req_idx] - num_sampled;
    for (int32_t i = 0; i < num_sampled; ++i) {
      logits_indices[logits_offset++] = logits_start + i;
    }
    for (int32_t i = 0; i < num_draft; ++i) {
      target_logits_indices[target_offset++] = sampled_offset + i;
    }
    draft_offset += num_draft;
    sampled_offset += num_sampled;
    cu_num_draft_tokens[req_idx] = draft_offset;
    cu_num_sampled_tokens[req_idx] = sampled_offset;
    bonus_logits_indices[req_idx] = sampled_offset - 1;
  }
}

void build_spec_decode_indices(
    torch::Tensor num_draft_tokens,
    torch::Tensor cu_num_scheduled_tokens,
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor cu_num_sampled_tokens,
    torch::Tensor logits_indices,
    torch::Tensor target_logits_indices,
    torch::Tensor bonus_logits_indices) {
  TORCH_CHECK(num_draft_tokens.is_cuda(),
              "num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(cu_num_scheduled_tokens.is_cuda(),
              "cu_num_scheduled_tokens must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(cu_num_sampled_tokens.is_cuda(),
              "cu_num_sampled_tokens must be a CUDA tensor");
  TORCH_CHECK(logits_indices.is_cuda(),
              "logits_indices must be a CUDA tensor");
  TORCH_CHECK(target_logits_indices.is_cuda(),
              "target_logits_indices must be a CUDA tensor");
  TORCH_CHECK(bonus_logits_indices.is_cuda(),
              "bonus_logits_indices must be a CUDA tensor");
  TORCH_CHECK(num_draft_tokens.scalar_type() == torch::kInt32,
              "num_draft_tokens must be int32");
  TORCH_CHECK(cu_num_scheduled_tokens.scalar_type() == torch::kInt32,
              "cu_num_scheduled_tokens must be int32");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(cu_num_sampled_tokens.scalar_type() == torch::kInt32,
              "cu_num_sampled_tokens must be int32");
  TORCH_CHECK(logits_indices.scalar_type() == torch::kInt32,
              "logits_indices must be int32");
  TORCH_CHECK(target_logits_indices.scalar_type() == torch::kInt32,
              "target_logits_indices must be int32");
  TORCH_CHECK(bonus_logits_indices.scalar_type() == torch::kInt32,
              "bonus_logits_indices must be int32");
  TORCH_CHECK(num_draft_tokens.dim() == 1,
              "num_draft_tokens must be 1D");
  TORCH_CHECK(cu_num_scheduled_tokens.dim() == 1,
              "cu_num_scheduled_tokens must be 1D");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");
  TORCH_CHECK(cu_num_sampled_tokens.dim() == 1,
              "cu_num_sampled_tokens must be 1D");
  TORCH_CHECK(logits_indices.dim() == 1,
              "logits_indices must be 1D");
  TORCH_CHECK(target_logits_indices.dim() == 1,
              "target_logits_indices must be 1D");
  TORCH_CHECK(bonus_logits_indices.dim() == 1,
              "bonus_logits_indices must be 1D");

  const int32_t batch_size = num_draft_tokens.size(0);
  TORCH_CHECK(cu_num_scheduled_tokens.numel() == batch_size,
              "cu_num_scheduled_tokens length mismatch");
  TORCH_CHECK(cu_num_draft_tokens.numel() == batch_size,
              "cu_num_draft_tokens length mismatch");
  TORCH_CHECK(cu_num_sampled_tokens.numel() == batch_size,
              "cu_num_sampled_tokens length mismatch");
  TORCH_CHECK(bonus_logits_indices.numel() == batch_size,
              "bonus_logits_indices length mismatch");
  TORCH_CHECK(
      logits_indices.numel() ==
          target_logits_indices.numel() + static_cast<int64_t>(batch_size),
      "logits_indices length mismatch");
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(num_draft_tokens.device());
  const dim3 grid(1);
  const dim3 block(1);
  build_spec_decode_indices_kernel<<<grid, block, 0,
                                     at::cuda::getDefaultCUDAStream()>>>(
      num_draft_tokens.data_ptr<int32_t>(),
      cu_num_scheduled_tokens.data_ptr<int32_t>(),
      cu_num_draft_tokens.data_ptr<int32_t>(),
      cu_num_sampled_tokens.data_ptr<int32_t>(),
      logits_indices.data_ptr<int32_t>(),
      target_logits_indices.data_ptr<int32_t>(),
      bonus_logits_indices.data_ptr<int32_t>(),
      batch_size);
}

template <typename scalar_t>
__global__ void eagle_sample_argmax_kernel(
    const scalar_t* __restrict__ logits,
    int64_t stride0,
    int64_t stride1,
    int32_t* __restrict__ output,
    int32_t vocab_size) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  float max_val = -FLT_MAX;
  int32_t max_idx = 0;
  for (int32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    const float v =
        static_cast<float>(logits[row * stride0 + static_cast<int64_t>(i) * stride1]);
    if (v > max_val) {
      max_val = v;
      max_idx = i;
    }
  }
  extern __shared__ unsigned char smem[];
  float* svals = reinterpret_cast<float*>(smem);
  int32_t* sidx = reinterpret_cast<int32_t*>(svals + blockDim.x);
  svals[threadIdx.x] = max_val;
  sidx[threadIdx.x] = max_idx;
  __syncthreads();
  for (int32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const float other = svals[threadIdx.x + offset];
      const int32_t other_idx = sidx[threadIdx.x + offset];
      if (other > svals[threadIdx.x]) {
        svals[threadIdx.x] = other;
        sidx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = sidx[0];
  }
}

torch::Tensor eagle_sample_argmax(torch::Tensor logits) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat ||
                  logits.scalar_type() == torch::kHalf ||
                  logits.scalar_type() == torch::kBFloat16,
              "logits must be float/half/bfloat16");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  const int64_t batch_size = logits.size(0);
  const int64_t vocab_size = logits.size(1);
  auto output = torch::empty(
      {batch_size},
      logits.options().dtype(torch::kInt32));
  if (batch_size == 0 || vocab_size == 0) {
    return output;
  }
  c10::cuda::CUDAGuard device_guard(logits.device());
  const int threads = 256;
  const dim3 grid(static_cast<uint32_t>(batch_size));
  const size_t shared_bytes =
      threads * (sizeof(float) + sizeof(int32_t));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "eagle_sample_argmax", [&] {
        eagle_sample_argmax_kernel<scalar_t>
            <<<grid, threads, shared_bytes,
               at::cuda::getDefaultCUDAStream()>>>(
                logits.data_ptr<scalar_t>(),
                logits.stride(0),
                logits.stride(1),
                output.data_ptr<int32_t>(),
                static_cast<int32_t>(vocab_size));
      });
  return output;
}

static torch::Tensor apply_top_k_only_cuda(torch::Tensor logits,
                                           torch::Tensor top_k) {
  const auto vocab_size = logits.size(1);
  auto no_top_k_mask = top_k == vocab_size;
  auto k = top_k.masked_fill(no_top_k_mask, 1);
  const int64_t max_top_k = k.max().item<int64_t>();
  auto topk_values = std::get<0>(logits.topk(max_top_k, 1, true, true));
  auto k_index = (k - 1).to(torch::kLong).unsqueeze(1);
  auto top_k_mask = topk_values.gather(1, k_index);
  top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -INFINITY);
  logits.masked_fill_(logits < top_k_mask, -INFINITY);
  return logits;
}

static torch::Tensor apply_top_k_top_p_cuda(
    torch::Tensor logits,
    const std::optional<torch::Tensor>& top_k,
    const std::optional<torch::Tensor>& top_p) {
  if (!top_p.has_value()) {
    if (!top_k.has_value()) {
      return logits;
    }
    return apply_top_k_only_cuda(logits, *top_k);
  }

  auto sort_res = logits.sort(-1, /*descending=*/false);
  auto logits_sort = std::get<0>(sort_res);
  auto logits_idx = std::get<1>(sort_res);

  if (top_k.has_value()) {
    auto k = *top_k;
    auto top_k_mask = logits_sort.size(1) - k.to(torch::kLong);
    top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(1));
    logits_sort.masked_fill_(logits_sort < top_k_mask, -INFINITY);
  }

  auto probs_sort = logits_sort.softmax(-1, torch::kFloat32);
  probs_sort = probs_sort.cumsum(-1);
  auto top_p_mask = probs_sort <= (1 - top_p->unsqueeze(1));
  top_p_mask.select(1, top_p_mask.size(1) - 1).fill_(false);
  logits_sort.masked_fill_(top_p_mask, -INFINITY);

  logits = logits_sort.scatter(-1, logits_idx, logits_sort);
  return logits;
}

torch::Tensor eagle_sample_topk_topp_gumbel(
    torch::Tensor logits,
    std::optional<torch::Tensor> top_k,
    std::optional<torch::Tensor> top_p,
    std::optional<torch::Tensor> temperature,
    double sampling_eps) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  const int64_t batch_size = logits.size(0);
  const int64_t vocab_size = logits.size(1);
  auto device = logits.device();
  c10::cuda::CUDAGuard device_guard(device);

  if (top_k.has_value()) {
    TORCH_CHECK(top_k->is_cuda(), "top_k must be a CUDA tensor");
    TORCH_CHECK(top_k->dim() == 1, "top_k must be 1D");
    TORCH_CHECK(top_k->numel() == batch_size,
                "top_k batch mismatch");
  }
  if (top_p.has_value()) {
    TORCH_CHECK(top_p->is_cuda(), "top_p must be a CUDA tensor");
    TORCH_CHECK(top_p->dim() == 1, "top_p must be 1D");
    TORCH_CHECK(top_p->numel() == batch_size,
                "top_p batch mismatch");
  }
  if (temperature.has_value()) {
    TORCH_CHECK(temperature->is_cuda(),
                "temperature must be a CUDA tensor");
    TORCH_CHECK(temperature->dim() == 1,
                "temperature must be 1D");
    TORCH_CHECK(temperature->numel() == batch_size,
                "temperature batch mismatch");
  }

  auto working = logits.to(torch::kFloat32);
  torch::Tensor is_greedy;
  if (temperature.has_value()) {
    auto temp = temperature->to(torch::kFloat32);
    is_greedy = temp < sampling_eps;
    auto safe_temp = torch::where(is_greedy,
                                  torch::ones_like(temp),
                                  temp);
    working = working / safe_temp.unsqueeze(1);
  }

  working = apply_top_k_top_p_cuda(working, top_k, top_p);

  auto q = torch::empty_like(working);
  q.exponential_();
  auto gumbel = -q.log();
  auto sampled = (working + gumbel).argmax(-1);

  if (temperature.has_value()) {
    auto greedy_tokens = working.argmax(-1);
    sampled = torch::where(is_greedy, greedy_tokens, sampled);
  }

  return sampled.to(torch::kInt32);
}

__global__ void eagle_expand_draft_tokens_kernel(
    const int32_t* __restrict__ draft_token_ids,
    const int32_t* __restrict__ cu_num_draft_tokens,
    int32_t* __restrict__ output,
    int64_t output_stride,
    int32_t max_len,
    int32_t batch_size,
    int32_t pad_id) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  if (req_idx >= batch_size) {
    return;
  }
  const int32_t start = req_idx == 0 ? 0 : cu_num_draft_tokens[req_idx - 1];
  const int32_t end = cu_num_draft_tokens[req_idx];
  int32_t count = end - start;
  if (count < 0) {
    count = 0;
  }
  const int64_t base = static_cast<int64_t>(req_idx) * output_stride;
  for (int32_t i = threadIdx.x; i < max_len; i += blockDim.x) {
    int32_t value = pad_id;
    if (i < count) {
      value = draft_token_ids[start + i];
    }
    output[base + i] = value;
  }
}

torch::Tensor eagle_expand_draft_tokens(torch::Tensor draft_token_ids,
                                        torch::Tensor cu_num_draft_tokens,
                                        int64_t max_draft_len,
                                        int64_t pad_id) {
  TORCH_CHECK(draft_token_ids.is_cuda(),
              "draft_token_ids must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(draft_token_ids.scalar_type() == torch::kInt32,
              "draft_token_ids must be int32");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(draft_token_ids.dim() == 1,
              "draft_token_ids must be 1D");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");

  const int32_t batch_size = cu_num_draft_tokens.size(0);
  auto device = draft_token_ids.device();
  c10::cuda::CUDAGuard device_guard(device);
  auto output = torch::empty(
      {batch_size, max_draft_len}, draft_token_ids.options());
  if (batch_size == 0 || max_draft_len <= 0) {
    return output;
  }
  const int threads = 256;
  const dim3 grid(batch_size);
  const dim3 block(threads);
  eagle_expand_draft_tokens_kernel<<<grid, block, 0,
                                     at::cuda::getDefaultCUDAStream()>>>(
      draft_token_ids.data_ptr<int32_t>(),
      cu_num_draft_tokens.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      output.stride(0),
      static_cast<int32_t>(max_draft_len),
      batch_size,
      static_cast<int32_t>(pad_id));
  return output;
}

__global__ void eagle_rewind_slot_mapping_kernel(
    const int32_t* __restrict__ cu_num_draft_tokens,
    const int32_t* __restrict__ valid_sampled_tokens_count,
    const int32_t* __restrict__ query_start_loc,
    int64_t* __restrict__ slot_mapping,
    int32_t num_reqs,
    int64_t pad_id) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  if (req_idx >= num_reqs) {
    return;
  }
  int32_t num_draft = cu_num_draft_tokens[req_idx];
  if (req_idx > 0) {
    num_draft -= cu_num_draft_tokens[req_idx - 1];
  }
  if (num_draft <= 0) {
    return;
  }
  const int32_t valid_count = valid_sampled_tokens_count[req_idx];
  int32_t num_rejected = num_draft + 1 - valid_count;
  if (num_rejected <= 0) {
    return;
  }
  const int32_t end_idx = query_start_loc[req_idx + 1];
  const int32_t start_idx = end_idx - num_rejected;
  for (int32_t i = threadIdx.x; i < num_rejected; i += blockDim.x) {
    slot_mapping[start_idx + i] = pad_id;
  }
}

void eagle_rewind_slot_mapping(
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor valid_sampled_tokens_count,
    torch::Tensor query_start_loc,
    torch::Tensor slot_mapping,
    int64_t pad_id) {
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(valid_sampled_tokens_count.is_cuda(),
              "valid_sampled_tokens_count must be a CUDA tensor");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(),
              "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(valid_sampled_tokens_count.scalar_type() == torch::kInt32,
              "valid_sampled_tokens_count must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");
  TORCH_CHECK(valid_sampled_tokens_count.dim() == 1,
              "valid_sampled_tokens_count must be 1D");
  TORCH_CHECK(query_start_loc.dim() == 1,
              "query_start_loc must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1,
              "slot_mapping must be 1D");

  const int32_t num_reqs = cu_num_draft_tokens.size(0);
  if (num_reqs == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(cu_num_draft_tokens.device());
  const int threads = 256;
  const dim3 grid(num_reqs);
  const dim3 block(threads);
  eagle_rewind_slot_mapping_kernel<<<grid, block, 0,
                                     at::cuda::getDefaultCUDAStream()>>>(
      cu_num_draft_tokens.data_ptr<int32_t>(),
      valid_sampled_tokens_count.data_ptr<int32_t>(),
      query_start_loc.data_ptr<int32_t>(),
      slot_mapping.data_ptr<int64_t>(),
      num_reqs,
      pad_id);
}

__global__ void eagle_compute_slot_mapping_kernel(
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ block_table,
    int64_t block_table_stride,
    int32_t block_size,
    int32_t max_model_len,
    int64_t* __restrict__ slot_mapping,
    int32_t batch_size,
    int64_t pad_id) {
  const int32_t req_idx =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (req_idx >= batch_size) {
    return;
  }
  int64_t pos = positions[req_idx];
  if (pos < 0 || pos >= max_model_len) {
    slot_mapping[req_idx] = pad_id;
    return;
  }
  const int32_t block_idx = static_cast<int32_t>(pos / block_size);
  const int32_t block_id =
      block_table[req_idx * block_table_stride + block_idx];
  const int64_t slot_id =
      static_cast<int64_t>(block_id) * block_size + (pos % block_size);
  slot_mapping[req_idx] = slot_id;
}

void eagle_compute_slot_mapping(
    torch::Tensor positions,
    torch::Tensor block_table,
    int64_t block_table_stride,
    int64_t block_size,
    int64_t max_model_len,
    torch::Tensor slot_mapping,
    int64_t pad_id) {
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64,
              "positions must be int64");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "positions must be 1D or 2D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");

  const int32_t batch_size =
      positions.dim() == 1 ? positions.size(0) : positions.size(1);
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(positions.device());
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  eagle_compute_slot_mapping_kernel<<<blocks, threads, 0,
                                      at::cuda::getDefaultCUDAStream()>>>(
      positions.data_ptr<int64_t>(),
      block_table.data_ptr<int32_t>(),
      block_table_stride,
      static_cast<int32_t>(block_size),
      static_cast<int32_t>(max_model_len),
      slot_mapping.data_ptr<int64_t>(),
      batch_size,
      pad_id);
}

template <typename scalar_t>
__global__ void eagle_update_draft_state_kernel(
    const int32_t* __restrict__ draft_tokens,
    const scalar_t* __restrict__ output_hidden_states,
    int64_t output_hidden_states_stride,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t positions_stride0,
    scalar_t* __restrict__ input_hidden_states,
    int64_t input_hidden_states_stride,
    int32_t* __restrict__ seq_lens,
    int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_table,
    int64_t block_table_stride,
    int32_t hidden_size,
    int32_t block_size,
    int32_t max_model_len,
    int64_t pad_id,
    bool use_mrope,
    int32_t batch_size) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  if (req_idx >= batch_size) {
    return;
  }
  if (threadIdx.x == 0) {
    const int32_t draft_token = draft_tokens[req_idx];
    input_ids[req_idx] = draft_token;

    int64_t pos = positions[req_idx];
    int64_t new_pos = pos + 1;
    const bool exceeds = new_pos >= max_model_len;
    if (new_pos >= max_model_len) {
      new_pos = max_model_len - 1;
    }
    if (use_mrope) {
      int64_t* base_ptr = positions;
      for (int dim = 0; dim < 3; ++dim) {
        base_ptr[dim * positions_stride0 + req_idx] = new_pos;
      }
    } else {
      positions[req_idx] = new_pos;
    }

    int32_t seq_len = seq_lens[req_idx] + 1;
    if (seq_len > max_model_len) {
      seq_len = max_model_len;
    }
    if (exceeds) {
      seq_len = 1;
    }
    seq_lens[req_idx] = seq_len;

    const int32_t block_idx = static_cast<int32_t>(new_pos / block_size);
    const int32_t block_id =
        block_table[req_idx * block_table_stride + block_idx];
    int64_t slot_id =
        static_cast<int64_t>(block_id) * block_size + (new_pos % block_size);
    if (exceeds) {
      slot_id = pad_id;
    }
    slot_mapping[req_idx] = slot_id;
  }

  const int64_t out_base = static_cast<int64_t>(req_idx) *
      output_hidden_states_stride;
  const int64_t in_base = static_cast<int64_t>(req_idx) *
      input_hidden_states_stride;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    input_hidden_states[in_base + i] = output_hidden_states[out_base + i];
  }
}

void eagle_update_draft_state(
    torch::Tensor draft_tokens,
    torch::Tensor output_hidden_states,
    int64_t output_hidden_states_stride,
    torch::Tensor input_ids,
    torch::Tensor positions,
    int64_t positions_stride0,
    torch::Tensor input_hidden_states,
    int64_t input_hidden_states_stride,
    torch::Tensor seq_lens,
    torch::Tensor slot_mapping,
    torch::Tensor block_table,
    int64_t block_table_stride,
    int64_t hidden_size,
    int64_t block_size,
    int64_t max_model_len,
    int64_t pad_id,
    bool use_mrope) {
  TORCH_CHECK(draft_tokens.is_cuda(), "draft_tokens must be a CUDA tensor");
  TORCH_CHECK(output_hidden_states.is_cuda(),
              "output_hidden_states must be a CUDA tensor");
  TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(input_hidden_states.is_cuda(),
              "input_hidden_states must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(draft_tokens.scalar_type() == torch::kInt32,
              "draft_tokens must be int32");
  TORCH_CHECK(input_ids.scalar_type() == torch::kInt32,
              "input_ids must be int32");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64,
              "positions must be int64");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32,
              "seq_lens must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(draft_tokens.dim() == 1, "draft_tokens must be 1D");
  TORCH_CHECK(input_ids.dim() == 1, "input_ids must be 1D");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "positions must be 1D or 2D");
  TORCH_CHECK(output_hidden_states.dim() == 2,
              "output_hidden_states must be 2D");
  TORCH_CHECK(input_hidden_states.dim() == 2,
              "input_hidden_states must be 2D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");

  const int32_t batch_size = draft_tokens.size(0);
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(draft_tokens.device());
  const int threads = 256;
  const dim3 grid(batch_size);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      output_hidden_states.scalar_type(), "eagle_update_draft_state", [&] {
        eagle_update_draft_state_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                draft_tokens.data_ptr<int32_t>(),
                output_hidden_states.data_ptr<scalar_t>(),
                output_hidden_states_stride,
                input_ids.data_ptr<int32_t>(),
                positions.data_ptr<int64_t>(),
                positions_stride0,
                input_hidden_states.data_ptr<scalar_t>(),
                input_hidden_states_stride,
                seq_lens.data_ptr<int32_t>(),
                slot_mapping.data_ptr<int64_t>(),
                block_table.data_ptr<int32_t>(),
                block_table_stride,
                static_cast<int32_t>(hidden_size),
                static_cast<int32_t>(block_size),
                static_cast<int32_t>(max_model_len),
                pad_id,
                use_mrope,
                batch_size);
      });
}

template <typename scalar_t>
__global__ void eagle_update_draft_state_and_tokens_kernel(
    const int32_t* __restrict__ draft_tokens,
    const scalar_t* __restrict__ output_hidden_states,
    int64_t output_hidden_states_stride,
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int64_t positions_stride0,
    scalar_t* __restrict__ input_hidden_states,
    int64_t input_hidden_states_stride,
    int32_t* __restrict__ seq_lens,
    int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_table,
    int64_t block_table_stride,
    int64_t* __restrict__ output_draft_tokens,
    int64_t output_draft_stride,
    int32_t step,
    int32_t hidden_size,
    int32_t block_size,
    int32_t max_model_len,
    int64_t pad_id,
    bool use_mrope,
    int32_t batch_size) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  if (req_idx >= batch_size) {
    return;
  }
  const int32_t draft_token = draft_tokens[req_idx];
  if (threadIdx.x == 0) {
    output_draft_tokens[static_cast<int64_t>(req_idx) * output_draft_stride +
                        step] = static_cast<int64_t>(draft_token);
    input_ids[req_idx] = draft_token;

    int64_t pos = positions[req_idx];
    int64_t new_pos = pos + 1;
    const bool exceeds = new_pos >= max_model_len;
    if (new_pos >= max_model_len) {
      new_pos = max_model_len - 1;
    }
    if (use_mrope) {
      int64_t* base_ptr = positions;
      for (int dim = 0; dim < 3; ++dim) {
        base_ptr[dim * positions_stride0 + req_idx] = new_pos;
      }
    } else {
      positions[req_idx] = new_pos;
    }

    int32_t seq_len = seq_lens[req_idx] + 1;
    if (seq_len > max_model_len) {
      seq_len = max_model_len;
    }
    if (exceeds) {
      seq_len = 1;
    }
    seq_lens[req_idx] = seq_len;

    const int32_t block_idx = static_cast<int32_t>(new_pos / block_size);
    const int32_t block_id =
        block_table[req_idx * block_table_stride + block_idx];
    int64_t slot_id =
        static_cast<int64_t>(block_id) * block_size + (new_pos % block_size);
    if (exceeds) {
      slot_id = pad_id;
    }
    slot_mapping[req_idx] = slot_id;
  }

  const int64_t out_base =
      static_cast<int64_t>(req_idx) * output_hidden_states_stride;
  const int64_t in_base =
      static_cast<int64_t>(req_idx) * input_hidden_states_stride;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    input_hidden_states[in_base + i] = output_hidden_states[out_base + i];
  }
}

void eagle_update_draft_state_and_tokens(
    torch::Tensor draft_tokens,
    torch::Tensor output_hidden_states,
    int64_t output_hidden_states_stride,
    torch::Tensor input_ids,
    torch::Tensor positions,
    int64_t positions_stride0,
    torch::Tensor input_hidden_states,
    int64_t input_hidden_states_stride,
    torch::Tensor seq_lens,
    torch::Tensor slot_mapping,
    torch::Tensor block_table,
    int64_t block_table_stride,
    torch::Tensor output_draft_tokens,
    int64_t output_draft_stride,
    int64_t step,
    int64_t hidden_size,
    int64_t block_size,
    int64_t max_model_len,
    int64_t pad_id,
    bool use_mrope) {
  TORCH_CHECK(draft_tokens.is_cuda(), "draft_tokens must be a CUDA tensor");
  TORCH_CHECK(output_hidden_states.is_cuda(),
              "output_hidden_states must be a CUDA tensor");
  TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(input_hidden_states.is_cuda(),
              "input_hidden_states must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  TORCH_CHECK(output_draft_tokens.is_cuda(),
              "output_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(draft_tokens.scalar_type() == torch::kInt32,
              "draft_tokens must be int32");
  TORCH_CHECK(input_ids.scalar_type() == torch::kInt32,
              "input_ids must be int32");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64,
              "positions must be int64");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32,
              "seq_lens must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(output_draft_tokens.scalar_type() == torch::kInt64,
              "output_draft_tokens must be int64");
  TORCH_CHECK(draft_tokens.dim() == 1, "draft_tokens must be 1D");
  TORCH_CHECK(input_ids.dim() == 1, "input_ids must be 1D");
  TORCH_CHECK(positions.dim() == 1 || positions.dim() == 2,
              "positions must be 1D or 2D");
  TORCH_CHECK(output_hidden_states.dim() == 2,
              "output_hidden_states must be 2D");
  TORCH_CHECK(input_hidden_states.dim() == 2,
              "input_hidden_states must be 2D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(output_draft_tokens.dim() == 2,
              "output_draft_tokens must be 2D");

  const int32_t batch_size = draft_tokens.size(0);
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(draft_tokens.device());
  const int threads = 256;
  const dim3 grid(batch_size);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      output_hidden_states.scalar_type(), "eagle_update_draft_state_and_tokens",
      [&] {
        eagle_update_draft_state_and_tokens_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                draft_tokens.data_ptr<int32_t>(),
                output_hidden_states.data_ptr<scalar_t>(),
                output_hidden_states_stride,
                input_ids.data_ptr<int32_t>(),
                positions.data_ptr<int64_t>(),
                positions_stride0,
                input_hidden_states.data_ptr<scalar_t>(),
                input_hidden_states_stride,
                seq_lens.data_ptr<int32_t>(),
                slot_mapping.data_ptr<int64_t>(),
                block_table.data_ptr<int32_t>(),
                block_table_stride,
                output_draft_tokens.data_ptr<int64_t>(),
                output_draft_stride,
                static_cast<int32_t>(step),
                static_cast<int32_t>(hidden_size),
                static_cast<int32_t>(block_size),
                static_cast<int32_t>(max_model_len),
                pad_id,
                use_mrope,
                batch_size);
      });
}

template <typename scalar_t>
__global__ void eagle_tree_copy_level_kernel(
    const int64_t* __restrict__ draft_token_ids,
    int64_t draft_token_stride0,
    const int64_t* __restrict__ draft_positions,
    int64_t draft_pos_stride0,
    const scalar_t* __restrict__ draft_hidden_states,
    int64_t draft_hidden_stride0,
    int64_t draft_hidden_stride1,
    int64_t* __restrict__ tree_input_ids,
    int64_t tree_input_stride0,
    int64_t* __restrict__ tree_positions,
    int64_t tree_positions_stride0,
    scalar_t* __restrict__ tree_hidden_states,
    int64_t tree_hidden_stride0,
    int64_t tree_hidden_stride1,
    int32_t level_offset,
    int32_t level_num_drafts,
    int32_t hidden_size,
    int32_t num_tokens) {
  const int32_t token_idx = static_cast<int32_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }
  const int32_t req_idx = token_idx / level_num_drafts;
  const int32_t local_idx = token_idx - req_idx * level_num_drafts;
  const int64_t src_base = static_cast<int64_t>(req_idx) * draft_token_stride0 + local_idx;
  const int64_t dst_base = static_cast<int64_t>(req_idx) * tree_input_stride0 +
      static_cast<int64_t>(level_offset + local_idx);
  if (threadIdx.x == 0) {
    tree_input_ids[dst_base] = draft_token_ids[src_base];
    tree_positions[static_cast<int64_t>(req_idx) * tree_positions_stride0 +
                   static_cast<int64_t>(level_offset + local_idx)] =
        draft_positions[static_cast<int64_t>(req_idx) * draft_pos_stride0 +
                        local_idx];
  }
  const int64_t hidden_src_base =
      static_cast<int64_t>(req_idx) * draft_hidden_stride0 +
      static_cast<int64_t>(local_idx) * draft_hidden_stride1;
  const int64_t hidden_dst_base =
      static_cast<int64_t>(req_idx) * tree_hidden_stride0 +
      static_cast<int64_t>(level_offset + local_idx) * tree_hidden_stride1;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    tree_hidden_states[hidden_dst_base + i] = draft_hidden_states[hidden_src_base + i];
  }
}

void eagle_tree_copy_level(torch::Tensor draft_token_ids,
                           torch::Tensor draft_positions,
                           torch::Tensor draft_hidden_states,
                           torch::Tensor tree_input_ids,
                           torch::Tensor tree_positions,
                           torch::Tensor tree_hidden_states,
                           int64_t level_offset,
                           int64_t hidden_size) {
  TORCH_CHECK(draft_token_ids.is_cuda(), "draft_token_ids must be CUDA");
  TORCH_CHECK(draft_positions.is_cuda(), "draft_positions must be CUDA");
  TORCH_CHECK(draft_hidden_states.is_cuda(), "draft_hidden_states must be CUDA");
  TORCH_CHECK(tree_input_ids.is_cuda(), "tree_input_ids must be CUDA");
  TORCH_CHECK(tree_positions.is_cuda(), "tree_positions must be CUDA");
  TORCH_CHECK(tree_hidden_states.is_cuda(), "tree_hidden_states must be CUDA");
  TORCH_CHECK(draft_token_ids.scalar_type() == torch::kInt64,
              "draft_token_ids must be int64");
  TORCH_CHECK(draft_positions.scalar_type() == torch::kInt64,
              "draft_positions must be int64");
  TORCH_CHECK(tree_input_ids.scalar_type() == torch::kInt64,
              "tree_input_ids must be int64");
  TORCH_CHECK(tree_positions.scalar_type() == torch::kInt64,
              "tree_positions must be int64");
  TORCH_CHECK(draft_hidden_states.dim() == 3,
              "draft_hidden_states must be 3D");
  TORCH_CHECK(tree_hidden_states.dim() == 3,
              "tree_hidden_states must be 3D");
  TORCH_CHECK(draft_token_ids.dim() == 2, "draft_token_ids must be 2D");
  TORCH_CHECK(draft_positions.dim() == 2, "draft_positions must be 2D");
  TORCH_CHECK(tree_input_ids.dim() == 2, "tree_input_ids must be 2D");
  TORCH_CHECK(tree_positions.dim() == 2, "tree_positions must be 2D");
  TORCH_CHECK(draft_token_ids.size(0) == draft_positions.size(0),
              "draft_token_ids batch mismatch");
  TORCH_CHECK(draft_token_ids.size(1) == draft_positions.size(1),
              "draft_token_ids length mismatch");
  const int32_t batch_size = static_cast<int32_t>(draft_token_ids.size(0));
  const int32_t level_num_drafts =
      static_cast<int32_t>(draft_token_ids.size(1));
  if (batch_size == 0 || level_num_drafts == 0 || hidden_size <= 0) {
    return;
  }
  const int32_t num_tokens = batch_size * level_num_drafts;
  c10::cuda::CUDAGuard device_guard(draft_token_ids.device());
  const int threads = 256;
  const dim3 grid(num_tokens);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      draft_hidden_states.scalar_type(), "eagle_tree_copy_level", [&] {
        eagle_tree_copy_level_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                draft_token_ids.data_ptr<int64_t>(),
                draft_token_ids.stride(0),
                draft_positions.data_ptr<int64_t>(),
                draft_positions.stride(0),
                draft_hidden_states.data_ptr<scalar_t>(),
                draft_hidden_states.stride(0),
                draft_hidden_states.stride(1),
                tree_input_ids.data_ptr<int64_t>(),
                tree_input_ids.stride(0),
                tree_positions.data_ptr<int64_t>(),
                tree_positions.stride(0),
                tree_hidden_states.data_ptr<scalar_t>(),
                tree_hidden_states.stride(0),
                tree_hidden_states.stride(1),
                static_cast<int32_t>(level_offset),
                level_num_drafts,
                static_cast<int32_t>(hidden_size),
                num_tokens);
      });
}

__global__ void eagle_tree_select_next_tokens_kernel(
    const int64_t* __restrict__ topk_ids,
    int64_t topk_stride0,
    int64_t topk_stride1,
    int64_t topk_stride2,
    const int64_t* __restrict__ parent_indices,
    const int64_t* __restrict__ child_indices,
    int64_t* __restrict__ output_tokens,
    int64_t output_stride0,
    int32_t num_children,
    int32_t total) {
  const int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) {
    return;
  }
  const int32_t req_idx = idx / num_children;
  const int32_t child_idx = idx - req_idx * num_children;
  const int64_t parent = parent_indices[child_idx];
  const int64_t child = child_indices[child_idx];
  output_tokens[static_cast<int64_t>(req_idx) * output_stride0 + child_idx] =
      topk_ids[static_cast<int64_t>(req_idx) * topk_stride0 +
               parent * topk_stride1 + child * topk_stride2];
}

void eagle_tree_select_next_tokens(torch::Tensor topk_ids,
                                   torch::Tensor parent_indices,
                                   torch::Tensor child_indices,
                                   torch::Tensor output_tokens) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA");
  TORCH_CHECK(parent_indices.is_cuda(), "parent_indices must be CUDA");
  TORCH_CHECK(child_indices.is_cuda(), "child_indices must be CUDA");
  TORCH_CHECK(output_tokens.is_cuda(), "output_tokens must be CUDA");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt64,
              "topk_ids must be int64");
  TORCH_CHECK(parent_indices.scalar_type() == torch::kInt64,
              "parent_indices must be int64");
  TORCH_CHECK(child_indices.scalar_type() == torch::kInt64,
              "child_indices must be int64");
  TORCH_CHECK(output_tokens.scalar_type() == torch::kInt64,
              "output_tokens must be int64");
  TORCH_CHECK(topk_ids.dim() == 3, "topk_ids must be 3D");
  TORCH_CHECK(parent_indices.dim() == 1, "parent_indices must be 1D");
  TORCH_CHECK(child_indices.dim() == 1, "child_indices must be 1D");
  TORCH_CHECK(output_tokens.dim() == 2, "output_tokens must be 2D");

  const int32_t batch_size = static_cast<int32_t>(topk_ids.size(0));
  const int32_t num_children = static_cast<int32_t>(parent_indices.numel());
  TORCH_CHECK(child_indices.numel() == num_children,
              "child_indices length mismatch");
  TORCH_CHECK(output_tokens.size(0) == batch_size,
              "output_tokens batch mismatch");
  TORCH_CHECK(output_tokens.size(1) == num_children,
              "output_tokens length mismatch");

  if (batch_size == 0 || num_children == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(topk_ids.device());
  const int32_t total = batch_size * num_children;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  eagle_tree_select_next_tokens_kernel<<<blocks, threads, 0,
                                         at::cuda::getDefaultCUDAStream()>>>(
      topk_ids.data_ptr<int64_t>(),
      topk_ids.stride(0),
      topk_ids.stride(1),
      topk_ids.stride(2),
      parent_indices.data_ptr<int64_t>(),
      child_indices.data_ptr<int64_t>(),
      output_tokens.data_ptr<int64_t>(),
      output_tokens.stride(0),
      num_children,
      total);
}

template <typename scalar_t>
__global__ void eagle_tree_gather_hidden_states_kernel(
    const scalar_t* __restrict__ hidden_states,
    int64_t hidden_stride0,
    int64_t hidden_stride1,
    const int64_t* __restrict__ parent_indices,
    scalar_t* __restrict__ output_hidden_states,
    int64_t output_stride0,
    int64_t output_stride1,
    int32_t num_parents,
    int32_t hidden_size,
    int32_t total) {
  const int32_t idx = static_cast<int32_t>(blockIdx.x);
  if (idx >= total) {
    return;
  }
  const int32_t req_idx = idx / num_parents;
  const int32_t parent_idx = idx - req_idx * num_parents;
  const int64_t parent = parent_indices[parent_idx];
  const int64_t src_base =
      static_cast<int64_t>(req_idx) * hidden_stride0 +
      parent * hidden_stride1;
  const int64_t dst_base =
      static_cast<int64_t>(req_idx) * output_stride0 +
      static_cast<int64_t>(parent_idx) * output_stride1;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    output_hidden_states[dst_base + i] = hidden_states[src_base + i];
  }
}

void eagle_tree_gather_hidden_states(torch::Tensor hidden_states,
                                     torch::Tensor parent_indices,
                                     torch::Tensor output_hidden_states) {
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
  TORCH_CHECK(parent_indices.is_cuda(), "parent_indices must be CUDA");
  TORCH_CHECK(output_hidden_states.is_cuda(),
              "output_hidden_states must be CUDA");
  TORCH_CHECK(parent_indices.scalar_type() == torch::kInt64,
              "parent_indices must be int64");
  TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D");
  TORCH_CHECK(parent_indices.dim() == 1, "parent_indices must be 1D");
  TORCH_CHECK(output_hidden_states.dim() == 3,
              "output_hidden_states must be 3D");

  const int32_t batch_size = static_cast<int32_t>(hidden_states.size(0));
  const int32_t num_parents = static_cast<int32_t>(parent_indices.numel());
  TORCH_CHECK(output_hidden_states.size(0) == batch_size,
              "output_hidden_states batch mismatch");
  TORCH_CHECK(output_hidden_states.size(1) == num_parents,
              "output_hidden_states length mismatch");
  const int32_t hidden_size = static_cast<int32_t>(hidden_states.size(2));
  if (batch_size == 0 || num_parents == 0 || hidden_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  const int32_t total = batch_size * num_parents;
  const int threads = 256;
  const dim3 grid(total);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      hidden_states.scalar_type(), "eagle_tree_gather_hidden_states", [&] {
        eagle_tree_gather_hidden_states_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                hidden_states.data_ptr<scalar_t>(),
                hidden_states.stride(0),
                hidden_states.stride(1),
                parent_indices.data_ptr<int64_t>(),
                output_hidden_states.data_ptr<scalar_t>(),
                output_hidden_states.stride(0),
                output_hidden_states.stride(1),
                num_parents,
                hidden_size,
                total);
      });
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void eagle_assemble_target_logits_offsets_kernel(
    int64_t* __restrict__ logits_ptrs,
    int32_t* __restrict__ decoding_tokens,
    const scalar_t* __restrict__ logits,
    int64_t logits_stride0,
    const int32_t* __restrict__ draft_decoding_tokens,
    int32_t batch_size,
    int32_t max_decoding_tokens) {
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int32_t bid = static_cast<int32_t>(threadIdx.x);
  int32_t num_decoding_tokens = 0;
  if (bid < batch_size) {
    num_decoding_tokens = draft_decoding_tokens[bid] + 1;
    decoding_tokens[bid] = num_decoding_tokens;
  }

  int32_t logits_offset = 0;
  BlockScan(temp_storage).ExclusiveSum(num_decoding_tokens, logits_offset);
  __syncthreads();

  if (bid < batch_size) {
    for (int32_t ti = 0; ti < num_decoding_tokens; ++ti) {
      const scalar_t* row_ptr = logits +
          static_cast<int64_t>(logits_offset + ti) * logits_stride0;
      logits_ptrs[static_cast<int64_t>(bid) * max_decoding_tokens + ti] =
          static_cast<int64_t>(reinterpret_cast<intptr_t>(row_ptr));
    }
  }
}

void eagle_assemble_target_logits_offsets(
    torch::Tensor logits_ptrs,
    torch::Tensor decoding_tokens,
    torch::Tensor logits,
    torch::Tensor draft_decoding_tokens,
    int64_t max_decoding_tokens) {
  TORCH_CHECK(logits_ptrs.is_cuda(), "logits_ptrs must be a CUDA tensor");
  TORCH_CHECK(decoding_tokens.is_cuda(),
              "decoding_tokens must be a CUDA tensor");
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(draft_decoding_tokens.is_cuda(),
              "draft_decoding_tokens must be a CUDA tensor");
  TORCH_CHECK(logits_ptrs.scalar_type() == torch::kInt64,
              "logits_ptrs must be int64");
  TORCH_CHECK(decoding_tokens.scalar_type() == torch::kInt32,
              "decoding_tokens must be int32");
  TORCH_CHECK(draft_decoding_tokens.scalar_type() == torch::kInt32,
              "draft_decoding_tokens must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  const int32_t batch_size =
      static_cast<int32_t>(draft_decoding_tokens.numel());
  TORCH_CHECK(logits_ptrs.numel() >=
                  static_cast<int64_t>(batch_size) * max_decoding_tokens,
              "logits_ptrs length mismatch");
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(logits.device());
  constexpr int kBlockSize = 512;
  TORCH_CHECK(batch_size <= kBlockSize,
              "batch_size exceeds assemble_target_logits_offsets limit");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "eagle_assemble_target_logits_offsets", [&] {
        eagle_assemble_target_logits_offsets_kernel<scalar_t, kBlockSize>
            <<<1, kBlockSize, 0, at::cuda::getDefaultCUDAStream()>>>(
                logits_ptrs.data_ptr<int64_t>(),
                decoding_tokens.data_ptr<int32_t>(),
                logits.data_ptr<scalar_t>(),
                logits.stride(0),
                draft_decoding_tokens.data_ptr<int32_t>(),
                batch_size,
                static_cast<int32_t>(max_decoding_tokens));
      });
}

template <typename scalar_t>
__global__ void eagle_assemble_draft_logits_offsets_kernel(
    int64_t* __restrict__ logits_ptrs,
    const scalar_t* __restrict__ logits,
    int64_t logits_stride0,
    int64_t* __restrict__ output_ids_ptrs,
    int32_t* __restrict__ output_ids,
    bool* __restrict__ skip_decode,
    const int32_t* __restrict__ num_valid_logits,
    int32_t num_input_logits,
    int32_t max_decoding_draft_tokens) {
  const int32_t tix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const bool is_valid = tix < num_valid_logits[0];
  if (tix < num_input_logits) {
    skip_decode[tix] = !is_valid;
  }
  if (!is_valid) {
    return;
  }
  const scalar_t* row_ptr =
      logits + static_cast<int64_t>(tix) * logits_stride0;
  logits_ptrs[tix] = static_cast<int64_t>(reinterpret_cast<intptr_t>(row_ptr));
  output_ids_ptrs[tix] = static_cast<int64_t>(reinterpret_cast<intptr_t>(
      output_ids + static_cast<int64_t>(tix) * max_decoding_draft_tokens));
}

void eagle_assemble_draft_logits_offsets(
    torch::Tensor logits_ptrs,
    torch::Tensor logits,
    torch::Tensor output_ids_ptrs,
    torch::Tensor output_ids,
    torch::Tensor skip_decode,
    torch::Tensor num_valid_logits,
    int64_t num_input_logits,
    int64_t max_decoding_draft_tokens) {
  TORCH_CHECK(logits_ptrs.is_cuda(), "logits_ptrs must be a CUDA tensor");
  TORCH_CHECK(output_ids_ptrs.is_cuda(),
              "output_ids_ptrs must be a CUDA tensor");
  TORCH_CHECK(output_ids.is_cuda(), "output_ids must be a CUDA tensor");
  TORCH_CHECK(skip_decode.is_cuda(), "skip_decode must be a CUDA tensor");
  TORCH_CHECK(num_valid_logits.is_cuda(),
              "num_valid_logits must be a CUDA tensor");
  TORCH_CHECK(logits_ptrs.scalar_type() == torch::kInt64,
              "logits_ptrs must be int64");
  TORCH_CHECK(output_ids_ptrs.scalar_type() == torch::kInt64,
              "output_ids_ptrs must be int64");
  TORCH_CHECK(output_ids.scalar_type() == torch::kInt32,
              "output_ids must be int32");
  TORCH_CHECK(skip_decode.scalar_type() == torch::kBool,
              "skip_decode must be bool");
  TORCH_CHECK(num_valid_logits.scalar_type() == torch::kInt32,
              "num_valid_logits must be int32");
  TORCH_CHECK(num_valid_logits.numel() == 1,
              "num_valid_logits must have 1 element");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(output_ids.dim() == 2, "output_ids must be 2D");
  if (num_input_logits <= 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(logits.device());
  constexpr int kBlockSize = 256;
  const int blocks = static_cast<int>(eagle_div_up<int64_t>(
      num_input_logits, kBlockSize));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "eagle_assemble_draft_logits_offsets", [&] {
        eagle_assemble_draft_logits_offsets_kernel<scalar_t>
            <<<blocks, kBlockSize, 0, at::cuda::getDefaultCUDAStream()>>>(
                logits_ptrs.data_ptr<int64_t>(),
                logits.data_ptr<scalar_t>(),
                logits.stride(0),
                output_ids_ptrs.data_ptr<int64_t>(),
                output_ids.data_ptr<int32_t>(),
                skip_decode.data_ptr<bool>(),
                num_valid_logits.data_ptr<int32_t>(),
                static_cast<int32_t>(num_input_logits),
                static_cast<int32_t>(max_decoding_draft_tokens));
      });
}

__global__ void eagle_copy_output_tokens_ids_kernel(
    const int64_t* __restrict__ tmp_output_ids_ptrs,
    const int32_t* __restrict__ top_ks,
    const int32_t* __restrict__ top_k_offsets,
    const int32_t* __restrict__ input_draft_ids,
    const int32_t* __restrict__ input_draft_lens,
    const int32_t* __restrict__ num_valid_logits,
    int32_t* __restrict__ output_draft_ids,
    int32_t* __restrict__ output_draft_lens,
    int32_t layer_id,
    int32_t batch_size,
    int32_t max_decoding_draft_tokens,
    const int32_t* __restrict__ input_paths,
    int32_t* __restrict__ output_paths,
    int32_t max_path_len) {
  const int32_t tix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tix >= batch_size) {
    return;
  }
  int32_t* cur_output_draft_ids =
      output_draft_ids + tix * max_decoding_draft_tokens;
  const int32_t* cur_input_draft_ids =
      input_draft_ids + tix * max_decoding_draft_tokens;
  const int32_t prev_len = (layer_id == 0) ? 0 : input_draft_lens[tix];
  for (int32_t ii = 0; ii < prev_len; ++ii) {
    cur_output_draft_ids[ii] = cur_input_draft_ids[ii];
  }
  int32_t cur_len = prev_len;
  const int32_t start_topk = top_k_offsets[tix];
  const int32_t end_topk =
      (tix + 1 < batch_size) ? top_k_offsets[tix + 1] : num_valid_logits[0];
  for (int32_t ii = start_topk; ii < end_topk; ++ii) {
    const int32_t topk = top_ks[ii];
    const int64_t ptr_val = tmp_output_ids_ptrs[ii];
    const int32_t* tmp_ptr =
        reinterpret_cast<const int32_t*>(static_cast<intptr_t>(ptr_val));
    for (int32_t jj = 0; jj < topk; ++jj) {
      cur_output_draft_ids[cur_len] = tmp_ptr[jj];
      ++cur_len;
    }
  }
  output_draft_lens[tix] = cur_len;

  const int32_t max_decoding_tokens = max_decoding_draft_tokens + 1;
  const int32_t* in_paths =
      input_paths + tix * max_decoding_tokens * max_path_len;
  int32_t* out_paths =
      output_paths + tix * max_decoding_tokens * max_path_len;
  const int32_t total = max_decoding_tokens * max_path_len;
  for (int32_t ii = 0; ii < total; ++ii) {
    out_paths[ii] = in_paths[ii];
  }
}

void eagle_copy_output_tokens_ids(
    torch::Tensor tmp_output_ids_ptrs,
    torch::Tensor top_ks,
    torch::Tensor top_k_offsets,
    torch::Tensor input_draft_ids,
    torch::Tensor input_draft_lens,
    torch::Tensor num_valid_logits,
    torch::Tensor output_draft_ids,
    torch::Tensor output_draft_lens,
    int64_t layer_id,
    torch::Tensor input_paths,
    torch::Tensor output_paths,
    int64_t max_path_len) {
  TORCH_CHECK(tmp_output_ids_ptrs.is_cuda(),
              "tmp_output_ids_ptrs must be a CUDA tensor");
  TORCH_CHECK(top_ks.is_cuda(), "top_ks must be a CUDA tensor");
  TORCH_CHECK(top_k_offsets.is_cuda(), "top_k_offsets must be a CUDA tensor");
  TORCH_CHECK(input_draft_ids.is_cuda(),
              "input_draft_ids must be a CUDA tensor");
  TORCH_CHECK(input_draft_lens.is_cuda(),
              "input_draft_lens must be a CUDA tensor");
  TORCH_CHECK(num_valid_logits.is_cuda(),
              "num_valid_logits must be a CUDA tensor");
  TORCH_CHECK(output_draft_ids.is_cuda(),
              "output_draft_ids must be a CUDA tensor");
  TORCH_CHECK(output_draft_lens.is_cuda(),
              "output_draft_lens must be a CUDA tensor");
  TORCH_CHECK(input_paths.is_cuda(), "input_paths must be a CUDA tensor");
  TORCH_CHECK(output_paths.is_cuda(), "output_paths must be a CUDA tensor");
  TORCH_CHECK(tmp_output_ids_ptrs.scalar_type() == torch::kInt64,
              "tmp_output_ids_ptrs must be int64");
  TORCH_CHECK(top_ks.scalar_type() == torch::kInt32,
              "top_ks must be int32");
  TORCH_CHECK(top_k_offsets.scalar_type() == torch::kInt32,
              "top_k_offsets must be int32");
  TORCH_CHECK(input_draft_ids.scalar_type() == torch::kInt32,
              "input_draft_ids must be int32");
  TORCH_CHECK(input_draft_lens.scalar_type() == torch::kInt32,
              "input_draft_lens must be int32");
  TORCH_CHECK(num_valid_logits.scalar_type() == torch::kInt32,
              "num_valid_logits must be int32");
  TORCH_CHECK(output_draft_ids.scalar_type() == torch::kInt32,
              "output_draft_ids must be int32");
  TORCH_CHECK(output_draft_lens.scalar_type() == torch::kInt32,
              "output_draft_lens must be int32");
  TORCH_CHECK(input_paths.scalar_type() == torch::kInt32,
              "input_paths must be int32");
  TORCH_CHECK(output_paths.scalar_type() == torch::kInt32,
              "output_paths must be int32");
  TORCH_CHECK(num_valid_logits.numel() == 1,
              "num_valid_logits must have 1 element");
  TORCH_CHECK(input_draft_ids.dim() == 2,
              "input_draft_ids must be 2D");
  TORCH_CHECK(output_draft_ids.dim() == 2,
              "output_draft_ids must be 2D");
  const int32_t batch_size =
      static_cast<int32_t>(input_draft_ids.size(0));
  if (batch_size == 0) {
    return;
  }
  const int32_t max_decoding_draft_tokens =
      static_cast<int32_t>(input_draft_ids.size(1));
  c10::cuda::CUDAGuard device_guard(input_draft_ids.device());
  constexpr int kBlockSize = 256;
  const int blocks =
      static_cast<int>(eagle_div_up(batch_size, kBlockSize));
  eagle_copy_output_tokens_ids_kernel<<<blocks, kBlockSize, 0,
                                        at::cuda::getDefaultCUDAStream()>>>(
      tmp_output_ids_ptrs.data_ptr<int64_t>(),
      top_ks.data_ptr<int32_t>(),
      top_k_offsets.data_ptr<int32_t>(),
      input_draft_ids.data_ptr<int32_t>(),
      input_draft_lens.data_ptr<int32_t>(),
      num_valid_logits.data_ptr<int32_t>(),
      output_draft_ids.data_ptr<int32_t>(),
      output_draft_lens.data_ptr<int32_t>(),
      static_cast<int32_t>(layer_id),
      batch_size,
      max_decoding_draft_tokens,
      input_paths.data_ptr<int32_t>(),
      output_paths.data_ptr<int32_t>(),
      static_cast<int32_t>(max_path_len));
}

__global__ void eagle_extract_real_draft_tokens_kernel(
    int32_t cur_draft_idx,
    int32_t batch_size,
    int32_t max_draft_len,
    int32_t max_total_draft_tokens,
    int32_t max_top_k,
    int32_t num_tokens_expand_this_layer,
    const int32_t* __restrict__ tokens_gather_idx,
    const int32_t* __restrict__ top_k_list,
    const int32_t* __restrict__ draft_tokens_indices_cumsum,
    const int32_t* __restrict__ new_draft_tokens,
    int32_t* __restrict__ draft_tokens_buffer) {
  const int32_t tix = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tix >= batch_size) {
    return;
  }
  const int32_t new_tokens_offset =
      cur_draft_idx == 0 ? 1 : max_total_draft_tokens + 1;
  const int32_t base_offset =
      draft_tokens_indices_cumsum[cur_draft_idx];
  const int32_t* new_tokens_ptr =
      new_draft_tokens + tix * new_tokens_offset * max_top_k;
  int32_t* out_ptr =
      draft_tokens_buffer + tix * (max_total_draft_tokens + 1) + base_offset;
  int32_t count = 0;
  for (int32_t i = 0; i < num_tokens_expand_this_layer; ++i) {
    const int32_t token_gather_idx = tokens_gather_idx[i];
    const int32_t topk = top_k_list[i];
    const int32_t* token_ptr = new_tokens_ptr + token_gather_idx * max_top_k;
    for (int32_t j = 0; j < topk; ++j) {
      out_ptr[count] = token_ptr[j];
      ++count;
    }
  }
}

void eagle_extract_real_draft_tokens(
    int64_t cur_draft_idx,
    int64_t max_draft_len,
    int64_t max_total_draft_tokens,
    int64_t max_top_k,
    int64_t num_tokens_expand_this_layer,
    torch::Tensor tokens_gather_idx,
    torch::Tensor top_k_list,
    torch::Tensor draft_tokens_indices_cumsum,
    torch::Tensor new_draft_tokens,
    torch::Tensor draft_tokens_buffer) {
  TORCH_CHECK(tokens_gather_idx.is_cuda(),
              "tokens_gather_idx must be a CUDA tensor");
  TORCH_CHECK(top_k_list.is_cuda(), "top_k_list must be a CUDA tensor");
  TORCH_CHECK(draft_tokens_indices_cumsum.is_cuda(),
              "draft_tokens_indices_cumsum must be a CUDA tensor");
  TORCH_CHECK(new_draft_tokens.is_cuda(),
              "new_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(draft_tokens_buffer.is_cuda(),
              "draft_tokens_buffer must be a CUDA tensor");
  TORCH_CHECK(tokens_gather_idx.scalar_type() == torch::kInt32,
              "tokens_gather_idx must be int32");
  TORCH_CHECK(top_k_list.scalar_type() == torch::kInt32,
              "top_k_list must be int32");
  TORCH_CHECK(draft_tokens_indices_cumsum.scalar_type() == torch::kInt32,
              "draft_tokens_indices_cumsum must be int32");
  TORCH_CHECK(new_draft_tokens.scalar_type() == torch::kInt32,
              "new_draft_tokens must be int32");
  TORCH_CHECK(draft_tokens_buffer.scalar_type() == torch::kInt32,
              "draft_tokens_buffer must be int32");
  TORCH_CHECK(new_draft_tokens.dim() >= 2,
              "new_draft_tokens must be at least 2D");
  TORCH_CHECK(draft_tokens_buffer.dim() == 2,
              "draft_tokens_buffer must be 2D");
  const int32_t batch_size =
      static_cast<int32_t>(draft_tokens_buffer.size(0));
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(draft_tokens_buffer.device());
  constexpr int kBlockSize = 256;
  const int blocks =
      static_cast<int>(eagle_div_up(batch_size, kBlockSize));
  eagle_extract_real_draft_tokens_kernel<<<blocks, kBlockSize, 0,
                                           at::cuda::getDefaultCUDAStream()>>>(
      static_cast<int32_t>(cur_draft_idx),
      batch_size,
      static_cast<int32_t>(max_draft_len),
      static_cast<int32_t>(max_total_draft_tokens),
      static_cast<int32_t>(max_top_k),
      static_cast<int32_t>(num_tokens_expand_this_layer),
      tokens_gather_idx.data_ptr<int32_t>(),
      top_k_list.data_ptr<int32_t>(),
      draft_tokens_indices_cumsum.data_ptr<int32_t>(),
      new_draft_tokens.data_ptr<int32_t>(),
      draft_tokens_buffer.data_ptr<int32_t>());
}

template <int BLOCK_SIZE>
__global__ void eagle_prepare_ctx_inputs_kernel(
    int32_t* __restrict__ eagle_seq_lens,
    int32_t* __restrict__ eagle_ctx_lens,
    int32_t* __restrict__ output_ids,
    int32_t* __restrict__ position_ids,
    int32_t* __restrict__ hidden_states_indices,
    int32_t* __restrict__ last_token_indices,
    int32_t* __restrict__ num_last_token_indices,
    int32_t* __restrict__ hidden_size_batch_level_starts,
    const int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ chunked_context_next_tokens,
    const int32_t* __restrict__ base_seq_lens,
    const int32_t* __restrict__ base_ctx_lens,
    const int32_t* __restrict__ accepted_tokens,
    const int32_t* __restrict__ accepted_lens,
    const int32_t* __restrict__ prev_draft_lens,
    const int32_t* __restrict__ prev_paths,
    const int32_t* __restrict__ best_path_ids,
    int32_t batch_size,
    int32_t max_path_len,
    int32_t max_decoding_tokens,
    int32_t max_non_leaves_per_layer) {
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int32_t bid = static_cast<int32_t>(threadIdx.x);
  bool is_valid = bid < batch_size;
  bool is_context = false;
  int32_t num_decoding_tokens = 0;
  int32_t num_input_tokens = 0;
  int32_t prev_draft_len = 0;
  if (is_valid) {
    prev_draft_len = prev_draft_lens[bid];
    is_context = (prev_draft_len == 0);
    if (is_context) {
      num_input_tokens = base_ctx_lens[bid];
      num_decoding_tokens = num_input_tokens;
    } else {
      num_input_tokens = prev_draft_len + 1;
      num_decoding_tokens = accepted_lens[bid];
    }
  }
  for (int32_t ii = bid; ii < max_non_leaves_per_layer * batch_size;
       ii += BLOCK_SIZE) {
    last_token_indices[ii] = 1;
  }
  int32_t output_start = 0;
  int32_t input_index_base = 0;
  int32_t last_token_index = 0;
  BlockScan(temp_storage).ExclusiveSum(num_input_tokens, input_index_base);
  __syncthreads();
  BlockScan(temp_storage).ExclusiveSum(num_decoding_tokens, output_start);
  __syncthreads();
  BlockScan(temp_storage).InclusiveSum(num_decoding_tokens, last_token_index);

  if (is_valid) {
    const int32_t chunked_next = chunked_context_next_tokens[bid];
    const int32_t old_seq_len =
        base_seq_lens[bid] - num_input_tokens;
    for (int32_t ti = 0; ti < num_decoding_tokens; ++ti) {
      int32_t token = 0;
      if (is_context) {
        if (ti == num_decoding_tokens - 1) {
          token = (chunked_next >= 0) ? chunked_next
                                      : accepted_tokens[bid * max_path_len];
        } else {
          token = input_ids[input_index_base + ti + 1];
        }
      } else {
        token = accepted_tokens[bid * max_path_len + ti];
      }
      output_ids[output_start + ti] = token;
      position_ids[output_start + ti] = old_seq_len + ti;
    }
    eagle_ctx_lens[bid] = num_decoding_tokens;
    eagle_seq_lens[bid] = old_seq_len + num_decoding_tokens;

    const int32_t best_path_id = best_path_ids[bid];
    for (int32_t ii = 0; ii < num_decoding_tokens; ++ii) {
      int32_t index = 0;
      if (is_context) {
        index = input_index_base + ii;
      } else {
        const int32_t path_idx =
            eagle_flat_index3(bid, best_path_id, ii,
                              max_decoding_tokens, max_path_len);
        const int32_t last_token_id = prev_paths[path_idx];
        index = input_index_base + last_token_id;
      }
      hidden_states_indices[output_start + ii] = index;
    }
    last_token_indices[bid] = last_token_index;
    hidden_size_batch_level_starts[bid] = bid;
  }

  if (bid == BLOCK_SIZE - 1) {
    num_last_token_indices[0] = batch_size;
    hidden_size_batch_level_starts[batch_size] = batch_size;
  }
}

void eagle_prepare_ctx_eagle_inputs(
    torch::Tensor eagle_seq_lens,
    torch::Tensor eagle_ctx_lens,
    torch::Tensor output_ids,
    torch::Tensor position_ids,
    torch::Tensor hidden_states_indices,
    torch::Tensor last_token_indices,
    torch::Tensor num_last_token_indices,
    torch::Tensor hidden_size_batch_level_starts,
    torch::Tensor input_ids,
    torch::Tensor chunked_context_next_tokens,
    torch::Tensor base_seq_lens,
    torch::Tensor base_ctx_lens,
    torch::Tensor accepted_tokens,
    torch::Tensor accepted_lens,
    torch::Tensor prev_draft_lens,
    torch::Tensor prev_paths,
    torch::Tensor best_path_ids,
    int64_t max_path_len,
    int64_t max_decoding_tokens,
    int64_t max_non_leaves_per_layer) {
  TORCH_CHECK(eagle_seq_lens.is_cuda(),
              "eagle_seq_lens must be a CUDA tensor");
  TORCH_CHECK(eagle_ctx_lens.is_cuda(),
              "eagle_ctx_lens must be a CUDA tensor");
  TORCH_CHECK(output_ids.is_cuda(), "output_ids must be a CUDA tensor");
  TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
  TORCH_CHECK(hidden_states_indices.is_cuda(),
              "hidden_states_indices must be a CUDA tensor");
  TORCH_CHECK(last_token_indices.is_cuda(),
              "last_token_indices must be a CUDA tensor");
  TORCH_CHECK(num_last_token_indices.is_cuda(),
              "num_last_token_indices must be a CUDA tensor");
  TORCH_CHECK(hidden_size_batch_level_starts.is_cuda(),
              "hidden_size_batch_level_starts must be a CUDA tensor");
  TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
  TORCH_CHECK(chunked_context_next_tokens.is_cuda(),
              "chunked_context_next_tokens must be a CUDA tensor");
  TORCH_CHECK(base_seq_lens.is_cuda(),
              "base_seq_lens must be a CUDA tensor");
  TORCH_CHECK(base_ctx_lens.is_cuda(),
              "base_ctx_lens must be a CUDA tensor");
  TORCH_CHECK(accepted_tokens.is_cuda(),
              "accepted_tokens must be a CUDA tensor");
  TORCH_CHECK(accepted_lens.is_cuda(),
              "accepted_lens must be a CUDA tensor");
  TORCH_CHECK(prev_draft_lens.is_cuda(),
              "prev_draft_lens must be a CUDA tensor");
  TORCH_CHECK(prev_paths.is_cuda(), "prev_paths must be a CUDA tensor");
  TORCH_CHECK(best_path_ids.is_cuda(),
              "best_path_ids must be a CUDA tensor");
  TORCH_CHECK(eagle_seq_lens.scalar_type() == torch::kInt32,
              "eagle_seq_lens must be int32");
  TORCH_CHECK(eagle_ctx_lens.scalar_type() == torch::kInt32,
              "eagle_ctx_lens must be int32");
  TORCH_CHECK(output_ids.scalar_type() == torch::kInt32,
              "output_ids must be int32");
  TORCH_CHECK(position_ids.scalar_type() == torch::kInt32,
              "position_ids must be int32");
  TORCH_CHECK(hidden_states_indices.scalar_type() == torch::kInt32,
              "hidden_states_indices must be int32");
  TORCH_CHECK(last_token_indices.scalar_type() == torch::kInt32,
              "last_token_indices must be int32");
  TORCH_CHECK(num_last_token_indices.scalar_type() == torch::kInt32,
              "num_last_token_indices must be int32");
  TORCH_CHECK(hidden_size_batch_level_starts.scalar_type() == torch::kInt32,
              "hidden_size_batch_level_starts must be int32");
  TORCH_CHECK(input_ids.scalar_type() == torch::kInt32,
              "input_ids must be int32");
  TORCH_CHECK(chunked_context_next_tokens.scalar_type() == torch::kInt32,
              "chunked_context_next_tokens must be int32");
  TORCH_CHECK(base_seq_lens.scalar_type() == torch::kInt32,
              "base_seq_lens must be int32");
  TORCH_CHECK(base_ctx_lens.scalar_type() == torch::kInt32,
              "base_ctx_lens must be int32");
  TORCH_CHECK(accepted_tokens.scalar_type() == torch::kInt32,
              "accepted_tokens must be int32");
  TORCH_CHECK(accepted_lens.scalar_type() == torch::kInt32,
              "accepted_lens must be int32");
  TORCH_CHECK(prev_draft_lens.scalar_type() == torch::kInt32,
              "prev_draft_lens must be int32");
  TORCH_CHECK(prev_paths.scalar_type() == torch::kInt32,
              "prev_paths must be int32");
  TORCH_CHECK(best_path_ids.scalar_type() == torch::kInt32,
              "best_path_ids must be int32");
  const int32_t batch_size = static_cast<int32_t>(base_seq_lens.numel());
  if (batch_size == 0) {
    return;
  }
  constexpr int kBlockSize = 512;
  TORCH_CHECK(batch_size <= kBlockSize,
              "batch_size exceeds prepare_ctx_eagle_inputs limit");
  c10::cuda::CUDAGuard device_guard(input_ids.device());
  eagle_prepare_ctx_inputs_kernel<kBlockSize>
      <<<1, kBlockSize, 0, at::cuda::getDefaultCUDAStream()>>>(
          eagle_seq_lens.data_ptr<int32_t>(),
          eagle_ctx_lens.data_ptr<int32_t>(),
          output_ids.data_ptr<int32_t>(),
          position_ids.data_ptr<int32_t>(),
          hidden_states_indices.data_ptr<int32_t>(),
          last_token_indices.data_ptr<int32_t>(),
          num_last_token_indices.data_ptr<int32_t>(),
          hidden_size_batch_level_starts.data_ptr<int32_t>(),
          input_ids.data_ptr<int32_t>(),
          chunked_context_next_tokens.data_ptr<int32_t>(),
          base_seq_lens.data_ptr<int32_t>(),
          base_ctx_lens.data_ptr<int32_t>(),
          accepted_tokens.data_ptr<int32_t>(),
          accepted_lens.data_ptr<int32_t>(),
          prev_draft_lens.data_ptr<int32_t>(),
          prev_paths.data_ptr<int32_t>(),
          best_path_ids.data_ptr<int32_t>(),
          batch_size,
          static_cast<int32_t>(max_path_len),
          static_cast<int32_t>(max_decoding_tokens),
          static_cast<int32_t>(max_non_leaves_per_layer));
}

__global__ void eagle_build_leaf_mask_kernel(
    int8_t* __restrict__ is_leaf_mask,
    const int32_t* __restrict__ paths,
    int32_t max_decoding_tokens,
    int32_t max_path_len) {
  const int32_t bid = static_cast<int32_t>(blockIdx.x);
  const int32_t level = static_cast<int32_t>(blockIdx.y);
  for (int32_t path_idx = static_cast<int32_t>(threadIdx.x);
       path_idx < max_decoding_tokens;
       path_idx += static_cast<int32_t>(blockDim.x)) {
    const int32_t next_offset =
        eagle_flat_index3(bid, path_idx, level + 1,
                          max_decoding_tokens, max_path_len);
    const int32_t cur_offset =
        eagle_flat_index3(bid, path_idx, level,
                          max_decoding_tokens, max_path_len);
    const int32_t cur_token_idx = paths[cur_offset];
    if (cur_token_idx != -1 && paths[next_offset] != -1) {
      is_leaf_mask[bid * max_decoding_tokens + cur_token_idx] = 0;
    }
  }
}

__global__ void eagle_get_non_leaf_subtree_kernel(
    int32_t* __restrict__ selected_draft_indices,
    int32_t* __restrict__ selected_pos_offsets,
    bool* __restrict__ mask,
    int32_t* __restrict__ num_selected_draft_indices,
    int32_t* __restrict__ parent_non_leaf_in_level_offset,
    int32_t* __restrict__ non_leaves_in_level_offsets,
    const int8_t* __restrict__ is_leaf_mask,
    const int32_t* __restrict__ paths,
    int32_t level_idx,
    int32_t max_decoding_tokens,
    int32_t max_path_len) {
  const int32_t bid = static_cast<int32_t>(blockIdx.x);
  const int32_t max_decoding_draft_tokens = max_decoding_tokens - 1;

  extern __shared__ char smem_buf[];
  int32_t* histogram = reinterpret_cast<int32_t*>(smem_buf);
  int32_t* pos_offset = reinterpret_cast<int32_t*>(
      smem_buf + max_decoding_tokens * sizeof(int32_t));
  int32_t* selected_paths = reinterpret_cast<int32_t*>(
      smem_buf + 2 * max_decoding_tokens * sizeof(int32_t));
  int32_t* token_pos = reinterpret_cast<int32_t*>(
      smem_buf + 3 * max_decoding_tokens * sizeof(int32_t));

  for (int32_t ii = static_cast<int32_t>(threadIdx.x);
       ii < max_decoding_tokens;
       ii += static_cast<int32_t>(blockDim.x)) {
    histogram[ii] = 0;
    pos_offset[ii] = -1;
    selected_paths[ii] = -1;
  }
  __syncthreads();

  for (int32_t pi = static_cast<int32_t>(threadIdx.x);
       pi < max_decoding_tokens;
       pi += static_cast<int32_t>(blockDim.x)) {
    const int32_t token_offset =
        eagle_flat_index3(bid, pi, level_idx,
                          max_decoding_tokens, max_path_len);
    const int32_t token_idx = paths[token_offset];
    if (token_idx >= 0 &&
        !is_leaf_mask[bid * max_decoding_tokens + token_idx]) {
      atomicCAS(&selected_paths[token_idx], -1, pi);
      for (int32_t li = 1; li <= level_idx; ++li) {
        const int32_t token_offset_li =
            eagle_flat_index3(bid, pi, li,
                              max_decoding_tokens, max_path_len);
        const int32_t token_idx_li = paths[token_offset_li];
        atomicAdd(&histogram[token_idx_li], 1);
        atomicCAS(&pos_offset[token_idx_li], -1, li - 1);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t selected_count = 0;
    int32_t prev_pos_offset = -1;
    int32_t non_leaves_counter = 0;
    non_leaves_in_level_offsets[bid * max_decoding_tokens + 0] = 0;
    for (int32_t ti = 1; ti < max_decoding_tokens; ++ti) {
      if (histogram[ti] > 0) {
        selected_draft_indices[bid * max_decoding_draft_tokens + selected_count] =
            ti;
        const int32_t pos = pos_offset[ti];
        selected_pos_offsets[bid * max_decoding_draft_tokens + selected_count] =
            pos;
        if (pos != prev_pos_offset) {
          non_leaves_counter = 0;
          prev_pos_offset = pos;
        }
        if (!is_leaf_mask[bid * max_decoding_tokens + ti]) {
          non_leaves_in_level_offsets[bid * max_decoding_tokens + ti] =
              non_leaves_counter;
          non_leaves_counter++;
        }
        token_pos[ti] = selected_count;
        selected_count++;
      }
    }
    for (int32_t pi = 0; pi < max_decoding_tokens; ++pi) {
      for (int32_t li = 1; li <= level_idx; ++li) {
        const int32_t cur_offset =
            eagle_flat_index3(bid, pi, li,
                              max_decoding_tokens, max_path_len);
        const int32_t prev_offset =
            eagle_flat_index3(bid, pi, li - 1,
                              max_decoding_tokens, max_path_len);
        const int32_t token_cur = paths[cur_offset];
        const int32_t token_prev = paths[prev_offset];
        if (token_cur >= 0 && token_prev >= 0) {
          parent_non_leaf_in_level_offset[bid * max_decoding_tokens + token_cur] =
              non_leaves_in_level_offsets[bid * max_decoding_tokens + token_prev];
        }
      }
    }
    num_selected_draft_indices[bid] = selected_count;
  }
  __syncthreads();

  for (int32_t ti = static_cast<int32_t>(threadIdx.x);
       ti < max_decoding_tokens;
       ti += static_cast<int32_t>(blockDim.x)) {
    const int32_t path_id = selected_paths[ti];
    if (path_id >= 0) {
      for (int32_t li = 1; li <= level_idx; ++li) {
        const int32_t token_offset_i =
            eagle_flat_index3(bid, path_id, li,
                              max_decoding_tokens, max_path_len);
        const int32_t token_idx_i = paths[token_offset_i];
        const int32_t token_pos_i = token_pos[token_idx_i];
        for (int32_t lj = 1; lj <= li; ++lj) {
          const int32_t token_offset_j =
              eagle_flat_index3(bid, path_id, lj,
                                max_decoding_tokens, max_path_len);
          const int32_t token_idx_j = paths[token_offset_j];
          const int32_t token_pos_j = token_pos[token_idx_j];
          const int32_t mask_offset =
              eagle_flat_index3(bid, token_pos_i, token_pos_j,
                                max_decoding_tokens, max_decoding_tokens);
          mask[mask_offset] = true;
        }
      }
    }
  }
}

template <int BLOCK_SIZE>
__global__ void eagle_prepare_gen_inputs_kernel(
    int32_t* __restrict__ next_sequence_lengths,
    int32_t* __restrict__ next_context_lengths,
    int32_t* __restrict__ output_ids,
    int32_t* __restrict__ position_ids,
    int32_t* __restrict__ spec_decoding_gen_lengths,
    int32_t* __restrict__ spec_decoding_position_offsets,
    int32_t* __restrict__ spec_decoding_packed_masks,
    int32_t* __restrict__ hidden_states_indices,
    int32_t* __restrict__ last_token_indices,
    int32_t* __restrict__ num_last_token_indices,
    int32_t* __restrict__ output_hidden_size_batch_starts_per_level,
    int32_t* __restrict__ cum_sum_generation_lengths,
    int32_t* __restrict__ max_generation_length,
    const int32_t* __restrict__ next_draft_ids,
    const int32_t* __restrict__ selected_draft_indices,
    const int32_t* __restrict__ selected_draft_pos_offsets,
    const int32_t* __restrict__ num_selected_draft_indices,
    const int32_t* __restrict__ eagle_net0_sequence_lengths,
    const int32_t* __restrict__ prev_context_lengths,
    const int32_t* __restrict__ input_hidden_size_batch_starts_per_level,
    const int32_t* __restrict__ parent_non_leaf_in_level_offset,
    int32_t level_idx,
    int32_t batch_size,
    int32_t max_path_len,
    int32_t max_decoding_tokens,
    int32_t max_non_leaves_per_layer) {
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  using BlockReduce = cub::BlockReduce<int32_t, BLOCK_SIZE>;
  __shared__ union {
    typename BlockScan::TempStorage scan;
    typename BlockReduce::TempStorage reduce;
  } temp_storage;

  const int32_t bid = static_cast<int32_t>(threadIdx.x);
  const int32_t max_decoding_draft_tokens = max_decoding_tokens - 1;
  bool is_valid = bid < batch_size;
  int32_t next_draft_len = 0;
  int32_t num_next_logits = 0;
  if (is_valid) {
    next_draft_len = num_selected_draft_indices[bid];
    for (int32_t ti = 0; ti < next_draft_len; ++ti) {
      const int32_t pos_offset =
          selected_draft_pos_offsets[bid * max_decoding_draft_tokens + ti];
      if (pos_offset == level_idx - 1) {
        num_next_logits++;
      }
    }
  }

  int32_t output_index_base = 0;
  int32_t gen_length_cumsum = 0;
  int32_t last_indices = 0;
  int32_t output_last_indices_base = 0;
  BlockScan(temp_storage.scan).ExclusiveSum(next_draft_len, output_index_base);
  __syncthreads();
  BlockScan(temp_storage.scan).InclusiveSum(next_draft_len, gen_length_cumsum);
  __syncthreads();
  BlockScan(temp_storage.scan).InclusiveSum(num_next_logits, last_indices);
  __syncthreads();
  BlockScan(temp_storage.scan)
      .ExclusiveSum(num_next_logits, output_last_indices_base);
  __syncthreads();
  const int32_t max_gen_length =
      BlockReduce(temp_storage.reduce).Reduce(next_draft_len, cub::Max());
  if (bid == 0) {
    max_generation_length[0] = max_gen_length;
  }
  if (is_valid) {
    spec_decoding_gen_lengths[bid] = next_draft_len;
    next_context_lengths[bid] = prev_context_lengths[bid];
    const int32_t sequence_len = eagle_net0_sequence_lengths[bid];
    next_sequence_lengths[bid] = sequence_len + next_draft_len;
    cum_sum_generation_lengths[bid] = gen_length_cumsum;
    position_ids[bid] = sequence_len;

    int32_t last_token_idx = 0;
    for (int32_t ti = 0; ti < next_draft_len; ++ti) {
      const int32_t draft_idx =
          selected_draft_indices[bid * max_decoding_draft_tokens + ti] - 1;
      output_ids[output_index_base + ti] =
          next_draft_ids[bid * max_decoding_draft_tokens + draft_idx];
      const int32_t pos_offset =
          selected_draft_pos_offsets[bid * max_decoding_draft_tokens + ti];
      spec_decoding_position_offsets[bid * max_decoding_tokens + ti] =
          pos_offset;
      const int32_t in_level_token_offset =
          parent_non_leaf_in_level_offset[bid * max_decoding_tokens +
                                          draft_idx + 1];
      hidden_states_indices[output_index_base + ti] =
          input_hidden_size_batch_starts_per_level[
              pos_offset * batch_size + bid] +
          in_level_token_offset;
      if (pos_offset == level_idx - 1) {
        last_token_indices[output_last_indices_base + last_token_idx] =
            output_index_base + ti + 1;
        last_token_idx++;
      }
    }
    for (int32_t li = 0; li < level_idx; ++li) {
      output_hidden_size_batch_starts_per_level[li * batch_size + bid] =
          input_hidden_size_batch_starts_per_level[li * batch_size + bid];
    }
    const int32_t last_start =
        input_hidden_size_batch_starts_per_level[(level_idx - 1) * batch_size +
                                                 batch_size];
    output_hidden_size_batch_starts_per_level[level_idx * batch_size + bid] =
        last_start + bid * max_non_leaves_per_layer;
  }
  __syncthreads();
  if (bid == batch_size - 1) {
    num_last_token_indices[0] = last_indices;
    output_hidden_size_batch_starts_per_level[level_idx * batch_size +
                                              batch_size] =
        output_hidden_size_batch_starts_per_level[level_idx * batch_size +
                                                  batch_size - 1] +
        max_non_leaves_per_layer;
  }
}

__device__ __forceinline__ void eagle_mask_to_packed(
    int32_t* __restrict__ output_ptr,
    const char* __restrict__ sh_mask,
    int32_t max_generation_length,
    int32_t num_packed_masks) {
  for (int32_t mask_id = 0; mask_id < num_packed_masks; ++mask_id) {
    if (mask_id * 32 >= max_generation_length) {
      output_ptr[mask_id] = 0;
      return;
    }
    const int32_t sh_start =
        ((max_generation_length - (mask_id + 1) * 32) < 0)
            ? 0
            : (max_generation_length - (mask_id + 1) * 32);
    const int32_t sh_end = max_generation_length - mask_id * 32;
    const int32_t valid_bits = sh_end - sh_start;
    const bool first_bit = (sh_mask[sh_start] == '1');
    int32_t mask31 = 0;
    if (valid_bits != 1) {
      for (int32_t i = sh_start + 1; i < sh_end; ++i) {
        const int32_t index = (valid_bits - 1) - (i - sh_start);
        mask31 += (sh_mask[i] == '1') ? (1 << index) : 0;
      }
    }
    int32_t mask32 = 0;
    if (valid_bits == 32) {
      mask32 = first_bit ? mask31 - (1 << (valid_bits - 1)) : mask31;
    } else {
      mask32 = first_bit ? mask31 + (1 << (valid_bits - 1)) : mask31;
    }
    output_ptr[mask_id] = mask32;
  }
}

__global__ void eagle_get_packed_mask_kernel(
    const int32_t* __restrict__ cum_generation_lengths,
    const int32_t* __restrict__ max_generation_lengths,
    const bool* __restrict__ mask,
    int32_t max_decoding_draft_tokens,
    int32_t* __restrict__ packed_mask) {
  const int32_t batch_idx = static_cast<int32_t>(blockIdx.y);
  const int32_t token_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t num_tokens = (batch_idx == 0)
      ? cum_generation_lengths[0]
      : (cum_generation_lengths[batch_idx] -
         cum_generation_lengths[batch_idx - 1]);
  if (token_idx >= num_tokens) {
    return;
  }
  const int32_t max_generation_length = max_generation_lengths[0];
  const int32_t max_decoding_tokens = max_decoding_draft_tokens + 1;
  const int32_t num_packed_masks = eagle_div_up(max_decoding_tokens, 32);
  const int32_t output_start =
      (batch_idx == 0) ? 0 : cum_generation_lengths[batch_idx - 1];
  int32_t* output_ptr =
      packed_mask + (output_start + token_idx) * num_packed_masks;
  if (token_idx == 0) {
    for (int32_t mask_id = static_cast<int32_t>(threadIdx.x);
         mask_id < num_packed_masks;
         mask_id += static_cast<int32_t>(blockDim.x)) {
      output_ptr[mask_id] = (mask_id == 0) ? 1 : 0;
    }
    return;
  }
  const bool* mask_ptr =
      mask + batch_idx * max_decoding_tokens * max_decoding_tokens +
      token_idx * max_decoding_tokens;
  extern __shared__ char sh_mask[];
  for (int32_t ti = static_cast<int32_t>(threadIdx.x);
       ti < max_generation_length;
       ti += static_cast<int32_t>(blockDim.x)) {
    const int32_t sh_index = max_generation_length - 1 - ti;
    sh_mask[sh_index] = mask_ptr[ti] ? '1' : '0';
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    eagle_mask_to_packed(output_ptr, sh_mask, max_generation_length,
                         num_packed_masks);
  }
}

void eagle_prepare_gen_eagle_inputs(
    torch::Tensor next_sequence_lengths,
    torch::Tensor next_context_lengths,
    torch::Tensor output_ids,
    torch::Tensor position_ids,
    torch::Tensor spec_decoding_gen_lengths,
    torch::Tensor spec_decoding_position_offsets,
    torch::Tensor spec_decoding_packed_masks,
    torch::Tensor hidden_states_indices,
    torch::Tensor last_token_indices,
    torch::Tensor num_last_token_indices,
    torch::Tensor output_hidden_size_batch_starts_per_level,
    torch::Tensor is_leaf_mask,
    torch::Tensor selected_draft_indices,
    torch::Tensor selected_draft_pos_offsets,
    torch::Tensor num_selected_draft_indices,
    torch::Tensor selected_masks,
    torch::Tensor cum_sum_generation_lengths,
    torch::Tensor max_generation_length,
    torch::Tensor non_leaves_in_level_offsets,
    torch::Tensor parent_non_leaf_in_level_offset,
    torch::Tensor next_draft_ids,
    torch::Tensor eagle_net0_sequence_lengths,
    torch::Tensor prev_context_lengths,
    torch::Tensor input_hidden_size_batch_starts_per_level,
    torch::Tensor next_paths,
    int64_t level_idx,
    int64_t max_path_len,
    int64_t max_decoding_tokens,
    int64_t max_non_leaves_per_layer) {
  TORCH_CHECK(next_sequence_lengths.is_cuda(),
              "next_sequence_lengths must be a CUDA tensor");
  TORCH_CHECK(next_context_lengths.is_cuda(),
              "next_context_lengths must be a CUDA tensor");
  TORCH_CHECK(output_ids.is_cuda(), "output_ids must be a CUDA tensor");
  TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
  TORCH_CHECK(spec_decoding_gen_lengths.is_cuda(),
              "spec_decoding_gen_lengths must be a CUDA tensor");
  TORCH_CHECK(spec_decoding_position_offsets.is_cuda(),
              "spec_decoding_position_offsets must be a CUDA tensor");
  TORCH_CHECK(spec_decoding_packed_masks.is_cuda(),
              "spec_decoding_packed_masks must be a CUDA tensor");
  TORCH_CHECK(hidden_states_indices.is_cuda(),
              "hidden_states_indices must be a CUDA tensor");
  TORCH_CHECK(last_token_indices.is_cuda(),
              "last_token_indices must be a CUDA tensor");
  TORCH_CHECK(num_last_token_indices.is_cuda(),
              "num_last_token_indices must be a CUDA tensor");
  TORCH_CHECK(output_hidden_size_batch_starts_per_level.is_cuda(),
              "output_hidden_size_batch_starts_per_level must be a CUDA tensor");
  TORCH_CHECK(is_leaf_mask.is_cuda(), "is_leaf_mask must be a CUDA tensor");
  TORCH_CHECK(selected_draft_indices.is_cuda(),
              "selected_draft_indices must be a CUDA tensor");
  TORCH_CHECK(selected_draft_pos_offsets.is_cuda(),
              "selected_draft_pos_offsets must be a CUDA tensor");
  TORCH_CHECK(num_selected_draft_indices.is_cuda(),
              "num_selected_draft_indices must be a CUDA tensor");
  TORCH_CHECK(selected_masks.is_cuda(),
              "selected_masks must be a CUDA tensor");
  TORCH_CHECK(cum_sum_generation_lengths.is_cuda(),
              "cum_sum_generation_lengths must be a CUDA tensor");
  TORCH_CHECK(max_generation_length.is_cuda(),
              "max_generation_length must be a CUDA tensor");
  TORCH_CHECK(non_leaves_in_level_offsets.is_cuda(),
              "non_leaves_in_level_offsets must be a CUDA tensor");
  TORCH_CHECK(parent_non_leaf_in_level_offset.is_cuda(),
              "parent_non_leaf_in_level_offset must be a CUDA tensor");
  TORCH_CHECK(next_draft_ids.is_cuda(), "next_draft_ids must be a CUDA tensor");
  TORCH_CHECK(eagle_net0_sequence_lengths.is_cuda(),
              "eagle_net0_sequence_lengths must be a CUDA tensor");
  TORCH_CHECK(prev_context_lengths.is_cuda(),
              "prev_context_lengths must be a CUDA tensor");
  TORCH_CHECK(input_hidden_size_batch_starts_per_level.is_cuda(),
              "input_hidden_size_batch_starts_per_level must be a CUDA tensor");
  TORCH_CHECK(next_paths.is_cuda(), "next_paths must be a CUDA tensor");
  TORCH_CHECK(next_sequence_lengths.scalar_type() == torch::kInt32,
              "next_sequence_lengths must be int32");
  TORCH_CHECK(next_context_lengths.scalar_type() == torch::kInt32,
              "next_context_lengths must be int32");
  TORCH_CHECK(output_ids.scalar_type() == torch::kInt32,
              "output_ids must be int32");
  TORCH_CHECK(position_ids.scalar_type() == torch::kInt32,
              "position_ids must be int32");
  TORCH_CHECK(spec_decoding_gen_lengths.scalar_type() == torch::kInt32,
              "spec_decoding_gen_lengths must be int32");
  TORCH_CHECK(spec_decoding_position_offsets.scalar_type() == torch::kInt32,
              "spec_decoding_position_offsets must be int32");
  TORCH_CHECK(spec_decoding_packed_masks.scalar_type() == torch::kInt32,
              "spec_decoding_packed_masks must be int32");
  TORCH_CHECK(hidden_states_indices.scalar_type() == torch::kInt32,
              "hidden_states_indices must be int32");
  TORCH_CHECK(last_token_indices.scalar_type() == torch::kInt32,
              "last_token_indices must be int32");
  TORCH_CHECK(num_last_token_indices.scalar_type() == torch::kInt32,
              "num_last_token_indices must be int32");
  TORCH_CHECK(output_hidden_size_batch_starts_per_level.scalar_type() ==
                  torch::kInt32,
              "output_hidden_size_batch_starts_per_level must be int32");
  TORCH_CHECK(is_leaf_mask.scalar_type() == torch::kInt8,
              "is_leaf_mask must be int8");
  TORCH_CHECK(selected_draft_indices.scalar_type() == torch::kInt32,
              "selected_draft_indices must be int32");
  TORCH_CHECK(selected_draft_pos_offsets.scalar_type() == torch::kInt32,
              "selected_draft_pos_offsets must be int32");
  TORCH_CHECK(num_selected_draft_indices.scalar_type() == torch::kInt32,
              "num_selected_draft_indices must be int32");
  TORCH_CHECK(selected_masks.scalar_type() == torch::kBool,
              "selected_masks must be bool");
  TORCH_CHECK(cum_sum_generation_lengths.scalar_type() == torch::kInt32,
              "cum_sum_generation_lengths must be int32");
  TORCH_CHECK(max_generation_length.scalar_type() == torch::kInt32,
              "max_generation_length must be int32");
  TORCH_CHECK(non_leaves_in_level_offsets.scalar_type() == torch::kInt32,
              "non_leaves_in_level_offsets must be int32");
  TORCH_CHECK(parent_non_leaf_in_level_offset.scalar_type() == torch::kInt32,
              "parent_non_leaf_in_level_offset must be int32");
  TORCH_CHECK(next_draft_ids.scalar_type() == torch::kInt32,
              "next_draft_ids must be int32");
  TORCH_CHECK(eagle_net0_sequence_lengths.scalar_type() == torch::kInt32,
              "eagle_net0_sequence_lengths must be int32");
  TORCH_CHECK(prev_context_lengths.scalar_type() == torch::kInt32,
              "prev_context_lengths must be int32");
  TORCH_CHECK(input_hidden_size_batch_starts_per_level.scalar_type() ==
                  torch::kInt32,
              "input_hidden_size_batch_starts_per_level must be int32");
  TORCH_CHECK(next_paths.scalar_type() == torch::kInt32,
              "next_paths must be int32");
  const int32_t batch_size = static_cast<int32_t>(next_draft_ids.size(0));
  if (batch_size == 0) {
    return;
  }
  constexpr int kBlockSize = 512;
  TORCH_CHECK(batch_size <= kBlockSize,
              "batch_size exceeds prepare_gen_eagle_inputs limit");
  c10::cuda::CUDAGuard device_guard(next_draft_ids.device());

  const int32_t max_decoding_tokens_i32 =
      static_cast<int32_t>(max_decoding_tokens);
  const int32_t max_path_len_i32 = static_cast<int32_t>(max_path_len);
  dim3 leaf_grid(batch_size, max_path_len_i32 - 1);
  eagle_build_leaf_mask_kernel<<<leaf_grid, kBlockSize, 0,
                                 at::cuda::getDefaultCUDAStream()>>>(
      is_leaf_mask.data_ptr<int8_t>(),
      next_paths.data_ptr<int32_t>(),
      max_decoding_tokens_i32,
      max_path_len_i32);

  const size_t smem_size =
      4 * max_decoding_tokens_i32 * sizeof(int32_t);
  eagle_get_non_leaf_subtree_kernel<<<batch_size, kBlockSize, smem_size,
                                      at::cuda::getDefaultCUDAStream()>>>(
      selected_draft_indices.data_ptr<int32_t>(),
      selected_draft_pos_offsets.data_ptr<int32_t>(),
      selected_masks.data_ptr<bool>(),
      num_selected_draft_indices.data_ptr<int32_t>(),
      parent_non_leaf_in_level_offset.data_ptr<int32_t>(),
      non_leaves_in_level_offsets.data_ptr<int32_t>(),
      is_leaf_mask.data_ptr<int8_t>(),
      next_paths.data_ptr<int32_t>(),
      static_cast<int32_t>(level_idx),
      max_decoding_tokens_i32,
      max_path_len_i32);

  eagle_prepare_gen_inputs_kernel<kBlockSize>
      <<<1, kBlockSize, 0, at::cuda::getDefaultCUDAStream()>>>(
          next_sequence_lengths.data_ptr<int32_t>(),
          next_context_lengths.data_ptr<int32_t>(),
          output_ids.data_ptr<int32_t>(),
          position_ids.data_ptr<int32_t>(),
          spec_decoding_gen_lengths.data_ptr<int32_t>(),
          spec_decoding_position_offsets.data_ptr<int32_t>(),
          spec_decoding_packed_masks.data_ptr<int32_t>(),
          hidden_states_indices.data_ptr<int32_t>(),
          last_token_indices.data_ptr<int32_t>(),
          num_last_token_indices.data_ptr<int32_t>(),
          output_hidden_size_batch_starts_per_level.data_ptr<int32_t>(),
          cum_sum_generation_lengths.data_ptr<int32_t>(),
          max_generation_length.data_ptr<int32_t>(),
          next_draft_ids.data_ptr<int32_t>(),
          selected_draft_indices.data_ptr<int32_t>(),
          selected_draft_pos_offsets.data_ptr<int32_t>(),
          num_selected_draft_indices.data_ptr<int32_t>(),
          eagle_net0_sequence_lengths.data_ptr<int32_t>(),
          prev_context_lengths.data_ptr<int32_t>(),
          input_hidden_size_batch_starts_per_level.data_ptr<int32_t>(),
          parent_non_leaf_in_level_offset.data_ptr<int32_t>(),
          static_cast<int32_t>(level_idx),
          batch_size,
          max_path_len_i32,
          max_decoding_tokens_i32,
          static_cast<int32_t>(max_non_leaves_per_layer));

  dim3 pack_block(32);
  dim3 pack_grid(max_decoding_tokens_i32, batch_size);
  size_t pack_smem = max_decoding_tokens_i32 * sizeof(char);
  eagle_get_packed_mask_kernel<<<pack_grid, pack_block, pack_smem,
                                 at::cuda::getDefaultCUDAStream()>>>(
      cum_sum_generation_lengths.data_ptr<int32_t>(),
      max_generation_length.data_ptr<int32_t>(),
      selected_masks.data_ptr<bool>(),
      max_decoding_tokens_i32 - 1,
      spec_decoding_packed_masks.data_ptr<int32_t>());
}

__device__ inline void eagle_insertion_sort_int32(int32_t* data,
                                                  int32_t n) {
  for (int32_t i = 1; i < n; ++i) {
    int32_t key = data[i];
    int32_t j = i - 1;
    while (j >= 0 && data[j] > key) {
      data[j + 1] = data[j];
      --j;
    }
    data[j + 1] = key;
  }
}

__device__ inline int32_t eagle_find_ancestor_path_index(
    const int32_t* prev_paths,
    int32_t ancestor_id,
    int32_t ancestor_layer_idx,
    int32_t max_decoding_tokens,
    int32_t max_path_len) {
  if (ancestor_layer_idx == -1) {
    return 0;
  }
  for (int32_t i = 0; i < max_decoding_tokens; ++i) {
    if (prev_paths[i * max_path_len + ancestor_layer_idx + 1] ==
        ancestor_id) {
      return i;
    }
  }
  return -1;
}

template <typename scalar_t>
__global__ void eagle_update_scores_kernel(
    scalar_t* __restrict__ cur_log_probs,
    int64_t cur_stride0,
    int64_t cur_stride1,
    const scalar_t* __restrict__ prev_layer_scores,
    int64_t prev_stride0,
    int32_t batch_size,
    int32_t dynamic_topk,
    int32_t max_decoding_draft_tokens) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  const scalar_t* prev_ptr =
      prev_layer_scores + static_cast<int64_t>(bix) * prev_stride0;
  for (int32_t ii = 0; ii < dynamic_topk; ++ii) {
    const scalar_t score = prev_ptr[ii];
    scalar_t* cur_ptr =
        cur_log_probs +
        (static_cast<int64_t>(bix) * dynamic_topk + ii) * cur_stride0;
    for (int32_t jj = 0; jj < max_decoding_draft_tokens; ++jj) {
      if (jj < dynamic_topk) {
        cur_ptr[jj * cur_stride1] += score;
      } else {
        cur_ptr[jj * cur_stride1] = static_cast<scalar_t>(-INFINITY);
      }
    }
  }
}

void eagle_update_scores(torch::Tensor cur_log_probs,
                         torch::Tensor prev_layer_scores,
                         int64_t dynamic_tree_max_topk) {
  TORCH_CHECK(cur_log_probs.is_cuda(), "cur_log_probs must be CUDA");
  TORCH_CHECK(prev_layer_scores.is_cuda(),
              "prev_layer_scores must be CUDA");
  TORCH_CHECK(cur_log_probs.dim() == 2, "cur_log_probs must be 2D");
  TORCH_CHECK(prev_layer_scores.dim() == 2,
              "prev_layer_scores must be 2D");
  TORCH_CHECK(cur_log_probs.scalar_type() ==
                  prev_layer_scores.scalar_type(),
              "cur_log_probs and prev_layer_scores must have same dtype");
  const int32_t batch_size =
      static_cast<int32_t>(prev_layer_scores.size(0));
  const int32_t max_draft_tokens =
      static_cast<int32_t>(cur_log_probs.size(1));
  TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
  TORCH_CHECK(dynamic_tree_max_topk >= 0,
              "dynamic_tree_max_topk must be non-negative");
  TORCH_CHECK(
      batch_size == 0 ||
          cur_log_probs.size(0) ==
              static_cast<int64_t>(batch_size) * dynamic_tree_max_topk,
      "cur_log_probs shape mismatch");
  if (batch_size == 0 || max_draft_tokens == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(cur_log_probs.device());
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      cur_log_probs.scalar_type(), "eagle_update_scores", [&] {
        eagle_update_scores_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                cur_log_probs.data_ptr<scalar_t>(),
                cur_log_probs.stride(0),
                cur_log_probs.stride(1),
                prev_layer_scores.data_ptr<scalar_t>(),
                prev_layer_scores.stride(0),
                batch_size,
                static_cast<int32_t>(dynamic_tree_max_topk),
                max_draft_tokens);
      });
}

__global__ void eagle_update_path_kernel(
    int32_t layer_idx,
    int32_t batch_size,
    int32_t dynamic_topk,
    int32_t max_decoding_tokens,
    int32_t max_path_len,
    const int32_t* __restrict__ prev_paths,
    int64_t prev_stride0,
    int64_t prev_stride1,
    int32_t* __restrict__ second_topk_output_ids,
    int64_t second_stride0,
    int32_t* __restrict__ new_paths,
    int64_t new_stride0,
    int64_t new_stride1,
    int32_t* __restrict__ next_expand_indices,
    int64_t next_stride0) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  const int32_t max_decoding_draft_tokens = max_decoding_tokens - 1;
  const int32_t non_leaf_signal = max_decoding_tokens + 1;

  const int32_t* prev_ptr =
      prev_paths + static_cast<int64_t>(bix) * prev_stride0;
  int32_t* new_ptr =
      new_paths + static_cast<int64_t>(bix) * new_stride0;
  int32_t* next_ptr =
      next_expand_indices + static_cast<int64_t>(bix) * next_stride0;
  int32_t* second_ptr =
      second_topk_output_ids + static_cast<int64_t>(bix) * second_stride0;

  const int32_t total = max_decoding_tokens * max_path_len;
  for (int32_t i = 0; i < total; ++i) {
    new_ptr[i] = -1;
  }

  if (layer_idx == 0) {
    for (int32_t ii = 0; ii < dynamic_topk; ++ii) {
      new_ptr[ii * max_path_len + 0] = 0;
      new_ptr[ii * max_path_len + 1] = ii + 1;
      new_ptr[ii * max_path_len + 2] = non_leaf_signal;
      next_ptr[ii] = ii + 1;
    }
    return;
  }

  int32_t prev_layer_num_paths = 0;
  for (int32_t ii = 0; ii < max_decoding_tokens; ++ii) {
    if (prev_ptr[ii * max_path_len + 0] != -1) {
      prev_layer_num_paths++;
    } else {
      break;
    }
  }

  eagle_insertion_sort_int32(second_ptr, dynamic_topk);

  const int32_t offset_to_final_tree =
      layer_idx == 1 ? dynamic_topk + 1
                     : (layer_idx - 1) * dynamic_topk * dynamic_topk +
                           dynamic_topk + 1;
  for (int32_t ii = 0; ii < dynamic_topk; ++ii) {
    const int32_t row_idx = second_ptr[ii] / max_decoding_draft_tokens;
    const int32_t col_idx = second_ptr[ii] % max_decoding_draft_tokens;
    next_ptr[ii] = row_idx * dynamic_topk + col_idx + offset_to_final_tree;
  }

  const int32_t start_index_current = layer_idx * dynamic_topk + 1;
  const int32_t start_index_prev = start_index_current - dynamic_topk;
  int32_t used_prev_paths_index = -1;
  int32_t num_new_path = 0;
  for (int32_t ii = 0; ii < dynamic_topk; ++ii) {
    const int32_t new_index = ii + start_index_current;
    const int32_t ancestor_index =
        second_ptr[ii] / max_decoding_draft_tokens + start_index_prev;
    const int32_t ancestor_path_idx = eagle_find_ancestor_path_index(
        prev_ptr, ancestor_index, layer_idx - 1, max_decoding_tokens,
        max_path_len);

    if (ancestor_path_idx == used_prev_paths_index + 1) {
      used_prev_paths_index++;
    } else if (ancestor_path_idx > used_prev_paths_index + 1) {
      while (ancestor_path_idx > used_prev_paths_index + 1) {
        used_prev_paths_index++;
        for (int32_t jj = 0; jj <= layer_idx; ++jj) {
          new_ptr[num_new_path * max_path_len + jj] =
              prev_ptr[used_prev_paths_index * max_path_len + jj];
        }
        num_new_path++;
      }
      used_prev_paths_index++;
    }

    for (int32_t jj = 0; jj <= layer_idx; ++jj) {
      new_ptr[num_new_path * max_path_len + jj] =
          prev_ptr[ancestor_path_idx * max_path_len + jj];
    }
    new_ptr[num_new_path * max_path_len + layer_idx + 1] = new_index;
    new_ptr[num_new_path * max_path_len + layer_idx + 2] = non_leaf_signal;
    num_new_path++;
  }

  while (used_prev_paths_index < prev_layer_num_paths) {
    used_prev_paths_index++;
    for (int32_t jj = 0; jj <= layer_idx; ++jj) {
      new_ptr[num_new_path * max_path_len + jj] =
          prev_ptr[used_prev_paths_index * max_path_len + jj];
    }
    num_new_path++;
  }
}

void eagle_update_path(int64_t layer_idx,
                       int64_t dynamic_tree_max_topk,
                       torch::Tensor prev_paths,
                       torch::Tensor second_topk_output_ids,
                       torch::Tensor new_paths,
                       torch::Tensor next_expand_indices) {
  TORCH_CHECK(prev_paths.is_cuda(), "prev_paths must be CUDA");
  TORCH_CHECK(second_topk_output_ids.is_cuda(),
              "second_topk_output_ids must be CUDA");
  TORCH_CHECK(new_paths.is_cuda(), "new_paths must be CUDA");
  TORCH_CHECK(next_expand_indices.is_cuda(),
              "next_expand_indices must be CUDA");
  TORCH_CHECK(prev_paths.scalar_type() == torch::kInt32,
              "prev_paths must be int32");
  TORCH_CHECK(new_paths.scalar_type() == torch::kInt32,
              "new_paths must be int32");
  TORCH_CHECK(second_topk_output_ids.scalar_type() == torch::kInt32,
              "second_topk_output_ids must be int32");
  TORCH_CHECK(next_expand_indices.scalar_type() == torch::kInt32,
              "next_expand_indices must be int32");
  TORCH_CHECK(prev_paths.dim() == 3, "prev_paths must be 3D");
  TORCH_CHECK(new_paths.dim() == 3, "new_paths must be 3D");
  TORCH_CHECK(second_topk_output_ids.dim() == 2,
              "second_topk_output_ids must be 2D");
  TORCH_CHECK(next_expand_indices.dim() == 2,
              "next_expand_indices must be 2D");

  const int32_t batch_size = static_cast<int32_t>(prev_paths.size(0));
  const int32_t max_decoding_tokens =
      static_cast<int32_t>(prev_paths.size(1));
  const int32_t max_path_len =
      static_cast<int32_t>(prev_paths.size(2));
  TORCH_CHECK(new_paths.size(0) == batch_size &&
                  new_paths.size(1) == max_decoding_tokens &&
                  new_paths.size(2) == max_path_len,
              "new_paths shape mismatch");
  TORCH_CHECK(second_topk_output_ids.size(0) == batch_size,
              "second_topk_output_ids batch mismatch");
  TORCH_CHECK(second_topk_output_ids.size(1) >= dynamic_tree_max_topk,
              "second_topk_output_ids length mismatch");
  TORCH_CHECK(next_expand_indices.size(0) == batch_size,
              "next_expand_indices batch mismatch");

  if (batch_size == 0 || max_decoding_tokens == 0 || max_path_len == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(prev_paths.device());
  const int threads = 128;
  const int blocks = (batch_size + threads - 1) / threads;
  eagle_update_path_kernel<<<blocks, threads, 0,
                             at::cuda::getDefaultCUDAStream()>>>(
      static_cast<int32_t>(layer_idx),
      batch_size,
      static_cast<int32_t>(dynamic_tree_max_topk),
      max_decoding_tokens,
      max_path_len,
      prev_paths.data_ptr<int32_t>(),
      prev_paths.stride(0),
      prev_paths.stride(1),
      second_topk_output_ids.data_ptr<int32_t>(),
      second_topk_output_ids.stride(0),
      new_paths.data_ptr<int32_t>(),
      new_paths.stride(0),
      new_paths.stride(1),
      next_expand_indices.data_ptr<int32_t>(),
      next_expand_indices.stride(0));
}

__global__ void eagle_update_draft_tokens_and_scores_kernel(
    int32_t layer_idx,
    int32_t batch_size,
    int32_t dynamic_topk,
    int32_t max_decoding_draft_tokens,
    const int32_t* __restrict__ cur_draft_ids,
    int64_t cur_stride0,
    int64_t cur_stride1,
    const int32_t* __restrict__ input_draft_ids,
    int64_t input_stride0,
    const int32_t* __restrict__ input_draft_lens,
    int32_t* __restrict__ output_draft_ids,
    int64_t output_stride0,
    int32_t* __restrict__ output_draft_lens,
    const float* __restrict__ cur_layer_scores,
    int64_t scores_stride0,
    float* __restrict__ output_current_scores,
    int64_t output_scores_stride0) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  int32_t prev_len = layer_idx == 0 ? 0 : input_draft_lens[bix];
  int32_t cur_len = prev_len;

  int32_t* out_ids =
      output_draft_ids + static_cast<int64_t>(bix) * output_stride0;
  const int32_t* in_ids =
      input_draft_ids + static_cast<int64_t>(bix) * input_stride0;
  for (int32_t ii = 0; ii < prev_len; ++ii) {
    out_ids[ii] = in_ids[ii];
  }

  const int32_t* cur_ids =
      cur_draft_ids + static_cast<int64_t>(bix) * cur_stride0;
  for (int32_t jj = 0; jj < dynamic_topk; ++jj) {
    out_ids[cur_len] = cur_ids[jj * cur_stride1];
    cur_len++;
  }
  output_draft_lens[bix] = cur_len;

  const float* cur_scores =
      cur_layer_scores + static_cast<int64_t>(bix) * scores_stride0;
  float* out_scores =
      output_current_scores + static_cast<int64_t>(bix) * output_scores_stride0;
  for (int32_t ii = 0; ii < max_decoding_draft_tokens; ++ii) {
    out_scores[ii] = cur_scores[ii];
  }
}

void eagle_update_draft_tokens_and_scores(
    int64_t layer_idx,
    int64_t dynamic_tree_max_topk,
    torch::Tensor cur_draft_ids,
    torch::Tensor input_draft_ids,
    torch::Tensor input_draft_lens,
    torch::Tensor output_draft_ids,
    torch::Tensor output_draft_lens,
    torch::Tensor cur_layer_scores,
    torch::Tensor output_current_scores) {
  TORCH_CHECK(cur_draft_ids.is_cuda(), "cur_draft_ids must be CUDA");
  TORCH_CHECK(input_draft_ids.is_cuda(), "input_draft_ids must be CUDA");
  TORCH_CHECK(input_draft_lens.is_cuda(), "input_draft_lens must be CUDA");
  TORCH_CHECK(output_draft_ids.is_cuda(), "output_draft_ids must be CUDA");
  TORCH_CHECK(output_draft_lens.is_cuda(), "output_draft_lens must be CUDA");
  TORCH_CHECK(cur_layer_scores.is_cuda(), "cur_layer_scores must be CUDA");
  TORCH_CHECK(output_current_scores.is_cuda(),
              "output_current_scores must be CUDA");
  TORCH_CHECK(cur_draft_ids.scalar_type() == torch::kInt32,
              "cur_draft_ids must be int32");
  TORCH_CHECK(input_draft_ids.scalar_type() == torch::kInt32,
              "input_draft_ids must be int32");
  TORCH_CHECK(input_draft_lens.scalar_type() == torch::kInt32,
              "input_draft_lens must be int32");
  TORCH_CHECK(output_draft_ids.scalar_type() == torch::kInt32,
              "output_draft_ids must be int32");
  TORCH_CHECK(output_draft_lens.scalar_type() == torch::kInt32,
              "output_draft_lens must be int32");
  TORCH_CHECK(cur_layer_scores.scalar_type() == torch::kFloat32,
              "cur_layer_scores must be float32");
  TORCH_CHECK(output_current_scores.scalar_type() == torch::kFloat32,
              "output_current_scores must be float32");
  TORCH_CHECK(cur_draft_ids.dim() == 2, "cur_draft_ids must be 2D");
  TORCH_CHECK(input_draft_ids.dim() == 2, "input_draft_ids must be 2D");
  TORCH_CHECK(input_draft_lens.dim() == 1, "input_draft_lens must be 1D");
  TORCH_CHECK(output_draft_ids.dim() == 2, "output_draft_ids must be 2D");
  TORCH_CHECK(output_draft_lens.dim() == 1,
              "output_draft_lens must be 1D");
  TORCH_CHECK(cur_layer_scores.dim() == 2, "cur_layer_scores must be 2D");
  TORCH_CHECK(output_current_scores.dim() == 2,
              "output_current_scores must be 2D");

  const int32_t batch_size =
      static_cast<int32_t>(input_draft_ids.size(0));
  const int32_t max_draft_tokens =
      static_cast<int32_t>(input_draft_ids.size(1));
  TORCH_CHECK(cur_draft_ids.size(0) == batch_size,
              "cur_draft_ids batch mismatch");
  TORCH_CHECK(cur_draft_ids.size(1) >= dynamic_tree_max_topk,
              "cur_draft_ids length mismatch");
  TORCH_CHECK(output_draft_ids.size(0) == batch_size &&
                  output_draft_ids.size(1) == max_draft_tokens,
              "output_draft_ids shape mismatch");
  TORCH_CHECK(input_draft_lens.numel() == batch_size,
              "input_draft_lens batch mismatch");
  TORCH_CHECK(output_draft_lens.numel() == batch_size,
              "output_draft_lens batch mismatch");
  TORCH_CHECK(cur_layer_scores.size(0) == batch_size &&
                  cur_layer_scores.size(1) == max_draft_tokens,
              "cur_layer_scores shape mismatch");
  TORCH_CHECK(output_current_scores.size(0) == batch_size &&
                  output_current_scores.size(1) == max_draft_tokens,
              "output_current_scores shape mismatch");

  if (batch_size == 0 || max_draft_tokens == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(cur_draft_ids.device());
  const int threads = 128;
  const int blocks = (batch_size + threads - 1) / threads;
  eagle_update_draft_tokens_and_scores_kernel<<<blocks, threads, 0,
                                                at::cuda::getDefaultCUDAStream()>>>(
      static_cast<int32_t>(layer_idx),
      batch_size,
      static_cast<int32_t>(dynamic_tree_max_topk),
      max_draft_tokens,
      cur_draft_ids.data_ptr<int32_t>(),
      cur_draft_ids.stride(0),
      cur_draft_ids.stride(1),
      input_draft_ids.data_ptr<int32_t>(),
      input_draft_ids.stride(0),
      input_draft_lens.data_ptr<int32_t>(),
      output_draft_ids.data_ptr<int32_t>(),
      output_draft_ids.stride(0),
      output_draft_lens.data_ptr<int32_t>(),
      cur_layer_scores.data_ptr<float>(),
      cur_layer_scores.stride(0),
      output_current_scores.data_ptr<float>(),
      output_current_scores.stride(0));
}

template <typename scalar_t>
__global__ void eagle_kv_cache_rewind_kernel(
    scalar_t* __restrict__ kv_cache,
    int64_t block_stride,
    int64_t kv_stride,
    int64_t pos_stride,
    int64_t head_stride,
    int64_t head_dim_stride,
    int32_t num_blocks,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t block_size,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ cu_num_draft_tokens,
    const int32_t* __restrict__ valid_sampled_tokens_count,
    const int32_t* __restrict__ query_start_loc,
    int32_t num_reqs,
    int64_t pad_id) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t head_idx = static_cast<int32_t>(blockIdx.y);
  const int32_t kv_idx = static_cast<int32_t>(blockIdx.z);
  if (req_idx >= num_reqs || head_idx >= num_kv_heads || kv_idx >= 2) {
    return;
  }
  int32_t num_draft = cu_num_draft_tokens[req_idx];
  if (req_idx > 0) {
    num_draft -= cu_num_draft_tokens[req_idx - 1];
  }
  if (num_draft <= 0) {
    return;
  }
  const int32_t valid_count = valid_sampled_tokens_count[req_idx];
  const int32_t num_rejected = num_draft + 1 - valid_count;
  if (num_rejected <= 0) {
    return;
  }
  const int32_t end_idx = query_start_loc[req_idx + 1];
  const int32_t start_idx = end_idx - num_rejected;
  for (int32_t i = 0; i < num_rejected; ++i) {
    const int64_t slot_id = slot_mapping[start_idx + i];
    if (slot_id == pad_id || slot_id < 0) {
      continue;
    }
    const int32_t block_id = static_cast<int32_t>(slot_id / block_size);
    if (block_id < 0 || block_id >= num_blocks) {
      continue;
    }
    const int32_t pos = static_cast<int32_t>(slot_id - block_id * block_size);
    const int64_t base_offset =
        static_cast<int64_t>(block_id) * block_stride +
        static_cast<int64_t>(kv_idx) * kv_stride +
        static_cast<int64_t>(pos) * pos_stride +
        static_cast<int64_t>(head_idx) * head_stride;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
      kv_cache[base_offset + static_cast<int64_t>(d) * head_dim_stride] =
          static_cast<scalar_t>(0);
    }
  }
}

void eagle_kv_cache_rewind(
    torch::Tensor kv_cache,
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor valid_sampled_tokens_count,
    torch::Tensor query_start_loc,
    torch::Tensor slot_mapping,
    int64_t block_size,
    int64_t pad_id) {
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(valid_sampled_tokens_count.is_cuda(),
              "valid_sampled_tokens_count must be a CUDA tensor");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(valid_sampled_tokens_count.scalar_type() == torch::kInt32,
              "valid_sampled_tokens_count must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(kv_cache.dim() == 5, "kv_cache must be 5D");
  TORCH_CHECK(kv_cache.size(2) == block_size,
              "kv_cache block_size dimension mismatch");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");
  TORCH_CHECK(valid_sampled_tokens_count.dim() == 1,
              "valid_sampled_tokens_count must be 1D");
  TORCH_CHECK(query_start_loc.dim() == 1,
              "query_start_loc must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");

  const auto sizes = kv_cache.sizes();
  const auto strides = kv_cache.strides();
  int kv_dim = -1;
  if (sizes[0] == 2) {
    kv_dim = 0;
  } else if (sizes[1] == 2) {
    kv_dim = 1;
  } else {
    TORCH_CHECK(false, "kv_cache layout missing KV dimension");
  }
  const int block_dim = kv_dim == 0 ? 1 : 0;
  const int32_t num_blocks = static_cast<int32_t>(sizes[block_dim]);
  const int32_t num_kv_heads = static_cast<int32_t>(sizes[3]);
  const int32_t head_size = static_cast<int32_t>(sizes[4]);
  const int32_t num_reqs = static_cast<int32_t>(cu_num_draft_tokens.size(0));
  if (num_reqs == 0 || num_kv_heads == 0 || head_size == 0) {
    return;
  }

  const int64_t block_stride = strides[block_dim];
  const int64_t kv_stride = strides[kv_dim];
  const int64_t pos_stride = strides[2];
  const int64_t head_stride = strides[3];
  const int64_t head_dim_stride = strides[4];
  const dim3 grid(num_reqs, num_kv_heads, 2);
  const int threads = 256;
  c10::cuda::CUDAGuard device_guard(kv_cache.device());

  const auto dtype = kv_cache.scalar_type();
#if VLLM_HAS_FP8_TYPES
  if (c10::isFloat8Type(dtype)) {
    return;
  }
#endif

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dtype,
      "eagle_kv_cache_rewind", [&] {
        eagle_kv_cache_rewind_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                kv_cache.data_ptr<scalar_t>(),
                block_stride,
                kv_stride,
                pos_stride,
                head_stride,
                head_dim_stride,
                num_blocks,
                num_kv_heads,
                head_size,
                static_cast<int32_t>(block_size),
                slot_mapping.data_ptr<int64_t>(),
                cu_num_draft_tokens.data_ptr<int32_t>(),
                valid_sampled_tokens_count.data_ptr<int32_t>(),
                query_start_loc.data_ptr<int32_t>(),
                num_reqs,
                pad_id);
      });
}

__global__ void eagle_compact_slot_mapping_kernel(
    const int32_t* __restrict__ cu_num_draft_tokens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ accepted_offsets,
    const int32_t* __restrict__ packed_indices,
    int64_t* __restrict__ slot_mapping,
    int32_t num_reqs,
    int64_t pad_id) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  if (req_idx >= num_reqs) {
    return;
  }
  int32_t num_draft = cu_num_draft_tokens[req_idx];
  if (req_idx > 0) {
    num_draft -= cu_num_draft_tokens[req_idx - 1];
  }
  if (num_draft <= 0) {
    return;
  }
  const int32_t accepted_start = accepted_offsets[req_idx];
  const int32_t accepted_end = accepted_offsets[req_idx + 1];
  int32_t accepted_count = accepted_end - accepted_start;
  if (accepted_count < 0) {
    accepted_count = 0;
  }
  const int32_t end_idx = query_start_loc[req_idx + 1];
  const int32_t draft_start = end_idx - (num_draft + 1);
  // Compact accepted slot mappings to the front.
  for (int32_t i = 0; i < accepted_count; ++i) {
    const int32_t src_pos = packed_indices[accepted_start + i];
    slot_mapping[draft_start + i] = slot_mapping[draft_start + src_pos];
  }
  // Pad the remaining draft + bonus positions.
  const int32_t total_tokens = num_draft + 1;
  for (int32_t i = accepted_count; i < total_tokens; ++i) {
    slot_mapping[draft_start + i] = pad_id;
  }
}

void eagle_compact_slot_mapping(
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor query_start_loc,
    torch::Tensor accepted_offsets,
    torch::Tensor packed_indices,
    torch::Tensor slot_mapping,
    int64_t pad_id) {
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be a CUDA tensor");
  TORCH_CHECK(accepted_offsets.is_cuda(),
              "accepted_offsets must be a CUDA tensor");
  TORCH_CHECK(packed_indices.is_cuda(),
              "packed_indices must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(accepted_offsets.scalar_type() == torch::kInt32,
              "accepted_offsets must be int32");
  TORCH_CHECK(packed_indices.scalar_type() == torch::kInt32,
              "packed_indices must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");
  TORCH_CHECK(query_start_loc.dim() == 1,
              "query_start_loc must be 1D");
  TORCH_CHECK(accepted_offsets.dim() == 1,
              "accepted_offsets must be 1D");
  TORCH_CHECK(packed_indices.dim() == 1, "packed_indices must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");

  const int32_t num_reqs = cu_num_draft_tokens.size(0);
  TORCH_CHECK(accepted_offsets.numel() == static_cast<int64_t>(num_reqs + 1),
              "accepted_offsets length mismatch");
  if (num_reqs == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(cu_num_draft_tokens.device());
  const dim3 grid(num_reqs);
  const dim3 block(1);
  eagle_compact_slot_mapping_kernel<<<grid, block, 0,
                                      at::cuda::getDefaultCUDAStream()>>>(
      cu_num_draft_tokens.data_ptr<int32_t>(),
      query_start_loc.data_ptr<int32_t>(),
      accepted_offsets.data_ptr<int32_t>(),
      packed_indices.data_ptr<int32_t>(),
      slot_mapping.data_ptr<int64_t>(),
      num_reqs,
      pad_id);
}

__global__ void eagle_build_tree_accepted_indices_kernel(
    const int32_t* __restrict__ draft_token_ids,
    int64_t draft_stride,
    const int32_t* __restrict__ sampled_token_ids,
    int64_t sampled_stride,
    const int32_t* __restrict__ valid_sampled_tokens_count,
    const int32_t* __restrict__ level_offsets,
    const int32_t* __restrict__ level_sizes,
    int32_t tree_depth,
    const int32_t* __restrict__ children_offsets,
    const int32_t* __restrict__ children_offsets_start,
    int32_t* __restrict__ output_indices,
    int64_t output_stride,
    int32_t* __restrict__ output_counts,
    int32_t max_depth) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  int32_t accepted_len = valid_sampled_tokens_count[req_idx] - 1;
  if (accepted_len < 0) {
    accepted_len = 0;
  }
  if (accepted_len > max_depth) {
    accepted_len = max_depth;
  }
  output_counts[req_idx] = accepted_len;
  if (accepted_len == 0) {
    return;
  }
  int32_t parent_pos = 0;
  for (int32_t level = 0; level < accepted_len; ++level) {
    if (level >= tree_depth) {
      output_counts[req_idx] = level;
      break;
    }
    const int32_t parent_count = level_sizes[level];
    if (parent_pos < 0 || parent_pos >= parent_count) {
      output_counts[req_idx] = level;
      break;
    }
    const int32_t level_offset = level_offsets[level];
    const int32_t offsets_start = children_offsets_start[level];
    const int32_t child_start = children_offsets[offsets_start + parent_pos];
    const int32_t child_end = children_offsets[offsets_start + parent_pos + 1];
    const int32_t token =
        sampled_token_ids[static_cast<int64_t>(req_idx) * sampled_stride + level];
    int32_t chosen = -1;
    for (int32_t idx = child_start; idx < child_end; ++idx) {
      const int32_t draft_token = draft_token_ids[
          static_cast<int64_t>(req_idx) * draft_stride +
          static_cast<int64_t>(level_offset + idx)];
      if (draft_token == token) {
        chosen = idx;
        break;
      }
    }
    if (chosen < 0) {
      output_counts[req_idx] = level;
      break;
    }
    output_indices[static_cast<int64_t>(req_idx) * output_stride + level] =
        level_offset + chosen;
    parent_pos = chosen;
  }
}

void eagle_build_tree_accepted_indices(
    torch::Tensor draft_token_ids,
    torch::Tensor sampled_token_ids,
    torch::Tensor valid_sampled_tokens_count,
    torch::Tensor level_offsets,
    torch::Tensor level_sizes,
    torch::Tensor children_offsets,
    torch::Tensor children_offsets_start,
    torch::Tensor output_indices,
    torch::Tensor output_counts) {
  TORCH_CHECK(draft_token_ids.is_cuda(),
              "draft_token_ids must be a CUDA tensor");
  TORCH_CHECK(sampled_token_ids.is_cuda(),
              "sampled_token_ids must be a CUDA tensor");
  TORCH_CHECK(valid_sampled_tokens_count.is_cuda(),
              "valid_sampled_tokens_count must be a CUDA tensor");
  TORCH_CHECK(level_offsets.is_cuda(), "level_offsets must be a CUDA tensor");
  TORCH_CHECK(level_sizes.is_cuda(), "level_sizes must be a CUDA tensor");
  TORCH_CHECK(children_offsets.is_cuda(),
              "children_offsets must be a CUDA tensor");
  TORCH_CHECK(children_offsets_start.is_cuda(),
              "children_offsets_start must be a CUDA tensor");
  TORCH_CHECK(output_indices.is_cuda(),
              "output_indices must be a CUDA tensor");
  TORCH_CHECK(output_counts.is_cuda(),
              "output_counts must be a CUDA tensor");
  TORCH_CHECK(draft_token_ids.scalar_type() == torch::kInt32,
              "draft_token_ids must be int32");
  TORCH_CHECK(sampled_token_ids.scalar_type() == torch::kInt32,
              "sampled_token_ids must be int32");
  TORCH_CHECK(valid_sampled_tokens_count.scalar_type() == torch::kInt32,
              "valid_sampled_tokens_count must be int32");
  TORCH_CHECK(level_offsets.scalar_type() == torch::kInt32,
              "level_offsets must be int32");
  TORCH_CHECK(level_sizes.scalar_type() == torch::kInt32,
              "level_sizes must be int32");
  TORCH_CHECK(children_offsets.scalar_type() == torch::kInt32,
              "children_offsets must be int32");
  TORCH_CHECK(children_offsets_start.scalar_type() == torch::kInt32,
              "children_offsets_start must be int32");
  TORCH_CHECK(output_indices.scalar_type() == torch::kInt32,
              "output_indices must be int32");
  TORCH_CHECK(output_counts.scalar_type() == torch::kInt32,
              "output_counts must be int32");
  TORCH_CHECK(draft_token_ids.dim() == 2,
              "draft_token_ids must be 2D");
  TORCH_CHECK(sampled_token_ids.dim() == 2,
              "sampled_token_ids must be 2D");
  TORCH_CHECK(valid_sampled_tokens_count.dim() == 1,
              "valid_sampled_tokens_count must be 1D");
  TORCH_CHECK(level_offsets.dim() == 1, "level_offsets must be 1D");
  TORCH_CHECK(level_sizes.dim() == 1, "level_sizes must be 1D");
  TORCH_CHECK(children_offsets.dim() == 1, "children_offsets must be 1D");
  TORCH_CHECK(children_offsets_start.dim() == 1,
              "children_offsets_start must be 1D");
  TORCH_CHECK(output_indices.dim() == 2, "output_indices must be 2D");
  TORCH_CHECK(output_counts.dim() == 1, "output_counts must be 1D");

  const int32_t batch_size = draft_token_ids.size(0);
  TORCH_CHECK(sampled_token_ids.size(0) == batch_size,
              "sampled_token_ids batch mismatch");
  TORCH_CHECK(valid_sampled_tokens_count.numel() == batch_size,
              "valid_sampled_tokens_count length mismatch");
  TORCH_CHECK(output_indices.size(0) == batch_size,
              "output_indices batch mismatch");
  TORCH_CHECK(output_counts.numel() == batch_size,
              "output_counts length mismatch");
  const int32_t tree_depth = level_offsets.size(0);
  TORCH_CHECK(level_sizes.numel() == tree_depth,
              "level_sizes length mismatch");
  TORCH_CHECK(children_offsets_start.numel() == tree_depth,
              "children_offsets_start length mismatch");
  if (batch_size == 0 || tree_depth == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(draft_token_ids.device());
  const dim3 grid(batch_size);
  const dim3 block(1);
  int32_t max_depth = static_cast<int32_t>(output_indices.size(1));
  const int32_t sampled_max = static_cast<int32_t>(sampled_token_ids.size(1));
  if (sampled_max < max_depth) {
    max_depth = sampled_max;
  }
  eagle_build_tree_accepted_indices_kernel<<<grid, block, 0,
                                             at::cuda::getDefaultCUDAStream()>>>(
      draft_token_ids.data_ptr<int32_t>(),
      draft_token_ids.stride(0),
      sampled_token_ids.data_ptr<int32_t>(),
      sampled_token_ids.stride(0),
      valid_sampled_tokens_count.data_ptr<int32_t>(),
      level_offsets.data_ptr<int32_t>(),
      level_sizes.data_ptr<int32_t>(),
      tree_depth,
      children_offsets.data_ptr<int32_t>(),
      children_offsets_start.data_ptr<int32_t>(),
      output_indices.data_ptr<int32_t>(),
      output_indices.stride(0),
      output_counts.data_ptr<int32_t>(),
      max_depth);
}

template <typename scalar_t>
__global__ void eagle_kv_cache_compact_kernel(
    scalar_t* __restrict__ kv_cache,
    int64_t block_stride,
    int64_t kv_stride,
    int64_t pos_stride,
    int64_t head_stride,
    int64_t head_dim_stride,
    int32_t num_blocks,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t block_size,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ cu_num_draft_tokens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ accepted_offsets,
    const int32_t* __restrict__ packed_indices,
    int32_t num_reqs,
    int64_t pad_id) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t head_idx = static_cast<int32_t>(blockIdx.y);
  const int32_t kv_idx = static_cast<int32_t>(blockIdx.z);
  if (req_idx >= num_reqs || head_idx >= num_kv_heads || kv_idx >= 2) {
    return;
  }
  int32_t num_draft = cu_num_draft_tokens[req_idx];
  if (req_idx > 0) {
    num_draft -= cu_num_draft_tokens[req_idx - 1];
  }
  if (num_draft <= 0) {
    return;
  }
  const int32_t accepted_start = accepted_offsets[req_idx];
  const int32_t accepted_end = accepted_offsets[req_idx + 1];
  const int32_t accepted_count = accepted_end - accepted_start;
  if (accepted_count <= 0) {
    return;
  }
  const int32_t end_idx = query_start_loc[req_idx + 1];
  const int32_t draft_start = end_idx - (num_draft + 1);
  for (int32_t i = 0; i < accepted_count; ++i) {
    const int32_t src_pos = packed_indices[accepted_start + i];
    const int32_t dst_pos = i;
    const int64_t src_slot = slot_mapping[draft_start + src_pos];
    const int64_t dst_slot = slot_mapping[draft_start + dst_pos];
    if (src_slot == pad_id || dst_slot == pad_id || src_slot < 0 ||
        dst_slot < 0) {
      continue;
    }
    const int32_t src_block = static_cast<int32_t>(src_slot / block_size);
    const int32_t dst_block = static_cast<int32_t>(dst_slot / block_size);
    if (src_block < 0 || src_block >= num_blocks || dst_block < 0 ||
        dst_block >= num_blocks) {
      continue;
    }
    const int32_t src_pos_in_block =
        static_cast<int32_t>(src_slot - src_block * block_size);
    const int32_t dst_pos_in_block =
        static_cast<int32_t>(dst_slot - dst_block * block_size);
    const int64_t src_base =
        static_cast<int64_t>(src_block) * block_stride +
        static_cast<int64_t>(kv_idx) * kv_stride +
        static_cast<int64_t>(src_pos_in_block) * pos_stride +
        static_cast<int64_t>(head_idx) * head_stride;
    const int64_t dst_base =
        static_cast<int64_t>(dst_block) * block_stride +
        static_cast<int64_t>(kv_idx) * kv_stride +
        static_cast<int64_t>(dst_pos_in_block) * pos_stride +
        static_cast<int64_t>(head_idx) * head_stride;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
      kv_cache[dst_base + static_cast<int64_t>(d) * head_dim_stride] =
          kv_cache[src_base + static_cast<int64_t>(d) * head_dim_stride];
    }
  }
}

constexpr int kEagleKvCompactSharedBytes = 16384;

template <typename scalar_t>
__global__ void eagle_kv_cache_compact_packed_kernel(
    scalar_t* __restrict__ kv_cache,
    int64_t block_stride,
    int64_t kv_stride,
    int64_t pos_stride,
    int64_t head_stride,
    int64_t head_dim_stride,
    int32_t num_blocks,
    int32_t num_kv_heads,
    int32_t head_size,
    int32_t block_size,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ cu_num_draft_tokens,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ accepted_offsets,
    const int32_t* __restrict__ packed_indices,
    int32_t num_reqs,
    int64_t pad_id) {
  const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t head_idx = static_cast<int32_t>(blockIdx.y);
  const int32_t kv_idx = static_cast<int32_t>(blockIdx.z);
  if (req_idx >= num_reqs || head_idx >= num_kv_heads || kv_idx >= 2) {
    return;
  }
  int32_t num_draft = cu_num_draft_tokens[req_idx];
  if (req_idx > 0) {
    num_draft -= cu_num_draft_tokens[req_idx - 1];
  }
  if (num_draft <= 0) {
    return;
  }
  const int32_t accepted_start = accepted_offsets[req_idx];
  const int32_t accepted_end = accepted_offsets[req_idx + 1];
  const int32_t accepted_count = accepted_end - accepted_start;
  if (accepted_count <= 0) {
    return;
  }
  const int32_t end_idx = query_start_loc[req_idx + 1];
  const int32_t draft_start = end_idx - (num_draft + 1);

  const int32_t max_elems =
      static_cast<int32_t>(kEagleKvCompactSharedBytes / sizeof(scalar_t));
  int32_t tile = max_elems / accepted_count;
  if (tile <= 0) {
    for (int32_t i = 0; i < accepted_count; ++i) {
      const int32_t src_pos = packed_indices[accepted_start + i];
      const int32_t dst_pos = i;
      const int64_t src_slot = slot_mapping[draft_start + src_pos];
      const int64_t dst_slot = slot_mapping[draft_start + dst_pos];
      if (src_slot == pad_id || dst_slot == pad_id || src_slot < 0 ||
          dst_slot < 0) {
        continue;
      }
      const int32_t src_block = static_cast<int32_t>(src_slot / block_size);
      const int32_t dst_block = static_cast<int32_t>(dst_slot / block_size);
      if (src_block < 0 || src_block >= num_blocks || dst_block < 0 ||
          dst_block >= num_blocks) {
        continue;
      }
      const int32_t src_pos_in_block =
          static_cast<int32_t>(src_slot - src_block * block_size);
      const int32_t dst_pos_in_block =
          static_cast<int32_t>(dst_slot - dst_block * block_size);
      const int64_t src_base =
          static_cast<int64_t>(src_block) * block_stride +
          static_cast<int64_t>(kv_idx) * kv_stride +
          static_cast<int64_t>(src_pos_in_block) * pos_stride +
          static_cast<int64_t>(head_idx) * head_stride;
      const int64_t dst_base =
          static_cast<int64_t>(dst_block) * block_stride +
          static_cast<int64_t>(kv_idx) * kv_stride +
          static_cast<int64_t>(dst_pos_in_block) * pos_stride +
          static_cast<int64_t>(head_idx) * head_stride;
      for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        kv_cache[dst_base + static_cast<int64_t>(d) * head_dim_stride] =
            kv_cache[src_base + static_cast<int64_t>(d) * head_dim_stride];
      }
    }
    return;
  }
  if (tile > head_size) {
    tile = head_size;
  }

  __shared__ char load_smem[kEagleKvCompactSharedBytes];
  auto* smem_ptr = reinterpret_cast<scalar_t*>(load_smem);
  const int32_t warp_idx = static_cast<int32_t>(threadIdx.x / 32);
  const int32_t warp_count = static_cast<int32_t>(blockDim.x / 32);
  const int32_t lane_idx = static_cast<int32_t>(threadIdx.x & 0x1f);

  for (int32_t start = 0; start < head_size; start += tile) {
    int32_t chunk = head_size - start;
    if (chunk > tile) {
      chunk = tile;
    }
    // Load into shared buffer.
    for (int32_t token = warp_idx; token < accepted_count;
         token += warp_count) {
      const int32_t src_pos = packed_indices[accepted_start + token];
      const int64_t src_slot = slot_mapping[draft_start + src_pos];
      bool valid = src_slot != pad_id && src_slot >= 0;
      int32_t src_block = valid ? static_cast<int32_t>(src_slot / block_size) : -1;
      if (src_block < 0 || src_block >= num_blocks) {
        valid = false;
      }
      const int32_t src_pos_in_block =
          valid ? static_cast<int32_t>(src_slot - src_block * block_size) : 0;
      const int64_t src_base =
          static_cast<int64_t>(src_block) * block_stride +
          static_cast<int64_t>(kv_idx) * kv_stride +
          static_cast<int64_t>(src_pos_in_block) * pos_stride +
          static_cast<int64_t>(head_idx) * head_stride;
      scalar_t* token_buf = smem_ptr + token * chunk;
      for (int32_t d = lane_idx; d < chunk; d += 32) {
        if (valid) {
          token_buf[d] =
              kv_cache[src_base +
                       static_cast<int64_t>(start + d) * head_dim_stride];
        } else {
          token_buf[d] = static_cast<scalar_t>(0);
        }
      }
    }
    __syncthreads();
    // Store from shared buffer.
    for (int32_t token = warp_idx; token < accepted_count;
         token += warp_count) {
      const int64_t dst_slot = slot_mapping[draft_start + token];
      if (dst_slot == pad_id || dst_slot < 0) {
        continue;
      }
      const int32_t dst_block = static_cast<int32_t>(dst_slot / block_size);
      if (dst_block < 0 || dst_block >= num_blocks) {
        continue;
      }
      const int32_t dst_pos_in_block =
          static_cast<int32_t>(dst_slot - dst_block * block_size);
      const int64_t dst_base =
          static_cast<int64_t>(dst_block) * block_stride +
          static_cast<int64_t>(kv_idx) * kv_stride +
          static_cast<int64_t>(dst_pos_in_block) * pos_stride +
          static_cast<int64_t>(head_idx) * head_stride;
      scalar_t* token_buf = smem_ptr + token * chunk;
      for (int32_t d = lane_idx; d < chunk; d += 32) {
        kv_cache[dst_base +
                 static_cast<int64_t>(start + d) * head_dim_stride] =
            token_buf[d];
      }
    }
    __syncthreads();
  }
}

void eagle_kv_cache_compact(
    torch::Tensor kv_cache,
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor query_start_loc,
    torch::Tensor slot_mapping,
    torch::Tensor accepted_offsets,
    torch::Tensor packed_indices,
    int64_t block_size,
    int64_t pad_id) {
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(accepted_offsets.is_cuda(),
              "accepted_offsets must be a CUDA tensor");
  TORCH_CHECK(packed_indices.is_cuda(), "packed_indices must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(accepted_offsets.scalar_type() == torch::kInt32,
              "accepted_offsets must be int32");
  TORCH_CHECK(packed_indices.scalar_type() == torch::kInt32,
              "packed_indices must be int32");
  TORCH_CHECK(kv_cache.dim() == 5, "kv_cache must be 5D");
  TORCH_CHECK(kv_cache.size(2) == block_size,
              "kv_cache block_size dimension mismatch");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");
  TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(accepted_offsets.dim() == 1, "accepted_offsets must be 1D");
  TORCH_CHECK(packed_indices.dim() == 1, "packed_indices must be 1D");

  const auto sizes = kv_cache.sizes();
  const auto strides = kv_cache.strides();
  int kv_dim = -1;
  if (sizes[0] == 2) {
    kv_dim = 0;
  } else if (sizes[1] == 2) {
    kv_dim = 1;
  } else {
    TORCH_CHECK(false, "kv_cache layout missing KV dimension");
  }
  const int block_dim = kv_dim == 0 ? 1 : 0;
  const int32_t num_blocks = static_cast<int32_t>(sizes[block_dim]);
  const int32_t num_kv_heads = static_cast<int32_t>(sizes[3]);
  const int32_t head_size = static_cast<int32_t>(sizes[4]);
  const int32_t num_reqs = static_cast<int32_t>(cu_num_draft_tokens.size(0));
  TORCH_CHECK(accepted_offsets.numel() == static_cast<int64_t>(num_reqs + 1),
              "accepted_offsets length mismatch");
  if (num_reqs == 0 || num_kv_heads == 0 || head_size == 0) {
    return;
  }

  const int64_t block_stride = strides[block_dim];
  const int64_t kv_stride = strides[kv_dim];
  const int64_t pos_stride = strides[2];
  const int64_t head_stride = strides[3];
  const int64_t head_dim_stride = strides[4];
  const dim3 grid(num_reqs, num_kv_heads, 2);
  const int threads = 256;
  c10::cuda::CUDAGuard device_guard(kv_cache.device());

  const auto dtype = kv_cache.scalar_type();
#if VLLM_HAS_FP8_TYPES
  if (c10::isFloat8Type(dtype)) {
    return;
  }
#endif

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dtype,
      "eagle_kv_cache_compact", [&] {
        eagle_kv_cache_compact_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                kv_cache.data_ptr<scalar_t>(),
                block_stride,
                kv_stride,
                pos_stride,
                head_stride,
                head_dim_stride,
                num_blocks,
                num_kv_heads,
                head_size,
                static_cast<int32_t>(block_size),
                slot_mapping.data_ptr<int64_t>(),
                cu_num_draft_tokens.data_ptr<int32_t>(),
                query_start_loc.data_ptr<int32_t>(),
                accepted_offsets.data_ptr<int32_t>(),
                packed_indices.data_ptr<int32_t>(),
                num_reqs,
                pad_id);
      });
}

void eagle_kv_cache_compact_packed(
    torch::Tensor kv_cache,
    torch::Tensor cu_num_draft_tokens,
    torch::Tensor query_start_loc,
    torch::Tensor slot_mapping,
    torch::Tensor accepted_offsets,
    torch::Tensor packed_indices,
    int64_t block_size,
    int64_t pad_id) {
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "cu_num_draft_tokens must be a CUDA tensor");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(accepted_offsets.is_cuda(),
              "accepted_offsets must be a CUDA tensor");
  TORCH_CHECK(packed_indices.is_cuda(), "packed_indices must be a CUDA tensor");
  TORCH_CHECK(cu_num_draft_tokens.scalar_type() == torch::kInt32,
              "cu_num_draft_tokens must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(accepted_offsets.scalar_type() == torch::kInt32,
              "accepted_offsets must be int32");
  TORCH_CHECK(packed_indices.scalar_type() == torch::kInt32,
              "packed_indices must be int32");
  TORCH_CHECK(kv_cache.dim() == 5, "kv_cache must be 5D");
  TORCH_CHECK(kv_cache.size(2) == block_size,
              "kv_cache block_size dimension mismatch");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "cu_num_draft_tokens must be 1D");
  TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1D");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(accepted_offsets.dim() == 1, "accepted_offsets must be 1D");
  TORCH_CHECK(packed_indices.dim() == 1, "packed_indices must be 1D");

  const auto sizes = kv_cache.sizes();
  const auto strides = kv_cache.strides();
  int kv_dim = -1;
  if (sizes[0] == 2) {
    kv_dim = 0;
  } else if (sizes[1] == 2) {
    kv_dim = 1;
  } else {
    TORCH_CHECK(false, "kv_cache layout missing KV dimension");
  }
  const int block_dim = kv_dim == 0 ? 1 : 0;
  const int32_t num_blocks = static_cast<int32_t>(sizes[block_dim]);
  const int32_t num_kv_heads = static_cast<int32_t>(sizes[3]);
  const int32_t head_size = static_cast<int32_t>(sizes[4]);
  const int32_t num_reqs = static_cast<int32_t>(cu_num_draft_tokens.size(0));
  if (num_reqs == 0 || num_kv_heads == 0 || head_size == 0) {
    return;
  }

  const int64_t block_stride = strides[block_dim];
  const int64_t kv_stride = strides[kv_dim];
  const int64_t pos_stride = strides[2];
  const int64_t head_stride = strides[3];
  const int64_t head_dim_stride = strides[4];
  const dim3 grid(num_reqs, num_kv_heads, 2);
  const int threads = 128;
  c10::cuda::CUDAGuard device_guard(kv_cache.device());

  const auto dtype = kv_cache.scalar_type();
#if VLLM_HAS_FP8_TYPES
  if (c10::isFloat8Type(dtype)) {
    return;
  }
#endif

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dtype,
      "eagle_kv_cache_compact_packed", [&] {
        eagle_kv_cache_compact_packed_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                kv_cache.data_ptr<scalar_t>(),
                block_stride,
                kv_stride,
                pos_stride,
                head_stride,
                head_dim_stride,
                num_blocks,
                num_kv_heads,
                head_size,
                static_cast<int32_t>(block_size),
                slot_mapping.data_ptr<int64_t>(),
                cu_num_draft_tokens.data_ptr<int32_t>(),
                query_start_loc.data_ptr<int32_t>(),
                accepted_offsets.data_ptr<int32_t>(),
                packed_indices.data_ptr<int32_t>(),
                num_reqs,
                pad_id);
      });
}

__global__ void eagle_build_packed_tree_mask_kernel(
    const int32_t* __restrict__ paths,
    int64_t paths_stride0,
    int64_t paths_stride1,
    int64_t paths_stride2,
    int32_t* __restrict__ packed_mask,
    int64_t mask_stride0,
    int64_t mask_stride1,
    int64_t mask_stride2,
    int32_t batch_size,
    int32_t tree_len_out,
    int32_t tree_len_in,
    int32_t max_path_len,
    int32_t num_blocks,
    int32_t node_offset) {
  const int32_t idx =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t total = batch_size * tree_len_out;
  if (idx >= total || tree_len_out <= 0) {
    return;
  }
  const int32_t b = idx / tree_len_out;
  const int32_t row = idx - b * tree_len_out;
  int32_t* row_out = packed_mask +
      static_cast<int64_t>(b) * mask_stride0 +
      static_cast<int64_t>(row) * mask_stride1;
  for (int32_t i = 0; i < num_blocks; ++i) {
    row_out[static_cast<int64_t>(i) * mask_stride2] = 0;
  }
  row_out[0] |= 1;
  const int32_t path_row = row + node_offset;
  if (path_row < 0 || path_row >= tree_len_in) {
    return;
  }
  const int32_t* path_ptr = paths +
      static_cast<int64_t>(b) * paths_stride0 +
      static_cast<int64_t>(path_row) * paths_stride1;
  for (int32_t d = 0; d < max_path_len; ++d) {
    int32_t anc = path_ptr[static_cast<int64_t>(d) * paths_stride2];
    if (anc < 0 || anc >= tree_len_in) {
      continue;
    }
    anc -= node_offset;
    if (anc < 0 || anc >= tree_len_out) {
      continue;
    }
    const int32_t block = anc >> 5;
    const int32_t bit = anc & 31;
    row_out[static_cast<int64_t>(block) * mask_stride2] |= (1 << bit);
  }
}

torch::Tensor eagle_build_packed_tree_mask(torch::Tensor paths,
                                           int64_t tree_len,
                                           bool exclude_root) {
  TORCH_CHECK(paths.is_cuda(), "paths must be a CUDA tensor");
  TORCH_CHECK(paths.scalar_type() == torch::kInt32,
              "paths must be int32");
  TORCH_CHECK(paths.dim() == 3, "paths must be 3D");
  const int32_t batch_size = static_cast<int32_t>(paths.size(0));
  const int32_t tree_len_in =
      static_cast<int32_t>(paths.size(1));
  const int32_t max_path_len =
      static_cast<int32_t>(paths.size(2));
  int32_t tree_len_effective = tree_len_in;
  if (tree_len > 0 && tree_len <= tree_len_in) {
    tree_len_effective = static_cast<int32_t>(tree_len);
  }
  const int32_t node_offset = exclude_root ? 1 : 0;
  const int32_t tree_len_out = tree_len_effective - node_offset;
  if (tree_len_out <= 0 || batch_size == 0) {
    return torch::empty({batch_size, 0, 0},
                        paths.options());
  }
  const int32_t num_blocks = (tree_len_out + 31) / 32;
  auto packed_mask = torch::zeros(
      {batch_size, tree_len_out, num_blocks}, paths.options());
  const int32_t total = batch_size * tree_len_out;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  c10::cuda::CUDAGuard device_guard(paths.device());
  eagle_build_packed_tree_mask_kernel<<<blocks, threads, 0,
                                        at::cuda::getDefaultCUDAStream()>>>(
      paths.data_ptr<int32_t>(),
      paths.stride(0),
      paths.stride(1),
      paths.stride(2),
      packed_mask.data_ptr<int32_t>(),
      packed_mask.stride(0),
      packed_mask.stride(1),
      packed_mask.stride(2),
      batch_size,
      tree_len_out,
      tree_len_effective,
      max_path_len,
      num_blocks,
      node_offset);
  return packed_mask;
}

__global__ void eagle_set_topks_from_dynamic_tree_kernel(
    int32_t layer_idx,
    int32_t batch_size,
    int32_t num_input_logits,
    int32_t* __restrict__ top_ks,
    int32_t* __restrict__ top_k_offsets,
    int32_t dynamic_topk,
    const int32_t* __restrict__ num_valid_logits) {
  const int32_t tix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tix < num_input_logits) {
    if (num_valid_logits[0] <= 0 || tix < num_valid_logits[0]) {
      top_ks[tix] = dynamic_topk;
    }
  }
  if (tix < batch_size) {
    top_k_offsets[tix] = (layer_idx == 0) ? tix : tix * dynamic_topk;
  }
}

void eagle_set_topks_from_dynamic_tree(int64_t layer_idx,
                                       torch::Tensor top_ks,
                                       torch::Tensor top_k_offsets,
                                       int64_t dynamic_tree_max_topk,
                                       torch::Tensor num_valid_logits) {
  TORCH_CHECK(top_ks.is_cuda(), "top_ks must be CUDA");
  TORCH_CHECK(top_k_offsets.is_cuda(), "top_k_offsets must be CUDA");
  TORCH_CHECK(num_valid_logits.is_cuda(),
              "num_valid_logits must be CUDA");
  TORCH_CHECK(top_ks.scalar_type() == torch::kInt32,
              "top_ks must be int32");
  TORCH_CHECK(top_k_offsets.scalar_type() == torch::kInt32,
              "top_k_offsets must be int32");
  TORCH_CHECK(num_valid_logits.scalar_type() == torch::kInt32,
              "num_valid_logits must be int32");
  TORCH_CHECK(num_valid_logits.numel() == 1,
              "num_valid_logits must have 1 element");
  const int32_t num_input_logits = static_cast<int32_t>(top_ks.numel());
  const int32_t batch_size = static_cast<int32_t>(top_k_offsets.numel());
  if (num_input_logits <= 0 || batch_size <= 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(top_ks.device());
  const int threads = 256;
  const int blocks = (num_input_logits + threads - 1) / threads;
  eagle_set_topks_from_dynamic_tree_kernel<<<
      blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      static_cast<int32_t>(layer_idx),
      batch_size,
      num_input_logits,
      top_ks.data_ptr<int32_t>(),
      top_k_offsets.data_ptr<int32_t>(),
      static_cast<int32_t>(dynamic_tree_max_topk),
      num_valid_logits.data_ptr<int32_t>());
}

__global__ void eagle_assemble_second_topk_inputs_kernel(
    int32_t batch_size,
    int32_t dynamic_topk,
    int32_t max_decoding_draft_tokens,
    float* __restrict__ first_topk_logprobs,
    int64_t* __restrict__ second_topk_input_ptrs,
    int32_t* __restrict__ second_topk_output_ids_flat,
    int64_t* __restrict__ second_topk_output_ptrs) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  float* scores_ptr =
      first_topk_logprobs +
      static_cast<int64_t>(bix) * dynamic_topk * max_decoding_draft_tokens;
  int32_t* output_ptr =
      second_topk_output_ids_flat +
      static_cast<int64_t>(bix) * max_decoding_draft_tokens;
  second_topk_input_ptrs[bix] =
      static_cast<int64_t>(reinterpret_cast<intptr_t>(scores_ptr));
  second_topk_output_ptrs[bix] =
      static_cast<int64_t>(reinterpret_cast<intptr_t>(output_ptr));
}

void eagle_assemble_second_topk_inputs(torch::Tensor first_topk_logprobs,
                                       torch::Tensor second_topk_input_ptrs,
                                       torch::Tensor second_topk_output_ids,
                                       torch::Tensor second_topk_output_ptrs,
                                       int64_t dynamic_tree_max_topk) {
  TORCH_CHECK(first_topk_logprobs.is_cuda(),
              "first_topk_logprobs must be CUDA");
  TORCH_CHECK(second_topk_input_ptrs.is_cuda(),
              "second_topk_input_ptrs must be CUDA");
  TORCH_CHECK(second_topk_output_ids.is_cuda(),
              "second_topk_output_ids must be CUDA");
  TORCH_CHECK(second_topk_output_ptrs.is_cuda(),
              "second_topk_output_ptrs must be CUDA");
  TORCH_CHECK(second_topk_input_ptrs.scalar_type() == torch::kInt64,
              "second_topk_input_ptrs must be int64");
  TORCH_CHECK(second_topk_output_ptrs.scalar_type() == torch::kInt64,
              "second_topk_output_ptrs must be int64");
  TORCH_CHECK(second_topk_output_ids.scalar_type() == torch::kInt32,
              "second_topk_output_ids must be int32");
  TORCH_CHECK(first_topk_logprobs.dim() == 2,
              "first_topk_logprobs must be 2D");
  TORCH_CHECK(second_topk_input_ptrs.dim() == 1,
              "second_topk_input_ptrs must be 1D");
  TORCH_CHECK(second_topk_output_ptrs.dim() == 1,
              "second_topk_output_ptrs must be 1D");
  const int32_t batch_size =
      static_cast<int32_t>(second_topk_input_ptrs.numel());
  const int32_t max_decoding_draft_tokens =
      static_cast<int32_t>(second_topk_output_ids.size(1));
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(first_topk_logprobs.device());
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  eagle_assemble_second_topk_inputs_kernel<<<
      blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      batch_size,
      static_cast<int32_t>(dynamic_tree_max_topk),
      max_decoding_draft_tokens,
      first_topk_logprobs.data_ptr<float>(),
      second_topk_input_ptrs.data_ptr<int64_t>(),
      second_topk_output_ids.data_ptr<int32_t>(),
      second_topk_output_ptrs.data_ptr<int64_t>());
}

__global__ void eagle_extract_scores_and_real_draft_tokens_kernel(
    int32_t batch_size,
    int32_t dynamic_topk,
    int32_t max_decoding_draft_tokens,
    const int64_t* __restrict__ second_topk_input_ptrs,
    int64_t* __restrict__ second_topk_output_ptrs,
    const int32_t* __restrict__ first_topk_output_ids,
    float* __restrict__ second_topk_output_logprobs) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  const float* scores_ptr = reinterpret_cast<const float*>(
      static_cast<intptr_t>(second_topk_input_ptrs[bix]));
  int32_t* output_ids_ptr = reinterpret_cast<int32_t*>(
      static_cast<intptr_t>(second_topk_output_ptrs[bix]));
  float* out_logprobs =
      second_topk_output_logprobs +
      static_cast<int64_t>(bix) * max_decoding_draft_tokens;
  const int32_t* first_ids_ptr =
      first_topk_output_ids +
      static_cast<int64_t>(bix) * dynamic_topk * max_decoding_draft_tokens;
  for (int32_t ii = 0; ii < dynamic_topk; ++ii) {
    const int32_t idx = output_ids_ptr[ii];
    const int32_t row = idx / max_decoding_draft_tokens;
    const int32_t col = idx % max_decoding_draft_tokens;
    out_logprobs[ii] = scores_ptr[row * max_decoding_draft_tokens + col];
    output_ids_ptr[ii] =
        first_ids_ptr[row * max_decoding_draft_tokens + col];
  }
}

void eagle_extract_scores_and_real_draft_tokens(
    torch::Tensor second_topk_input_ptrs,
    torch::Tensor second_topk_output_ptrs,
    torch::Tensor first_topk_output_ids,
    torch::Tensor second_topk_output_logprobs,
    int64_t dynamic_tree_max_topk) {
  TORCH_CHECK(second_topk_input_ptrs.is_cuda(),
              "second_topk_input_ptrs must be CUDA");
  TORCH_CHECK(second_topk_output_ptrs.is_cuda(),
              "second_topk_output_ptrs must be CUDA");
  TORCH_CHECK(first_topk_output_ids.is_cuda(),
              "first_topk_output_ids must be CUDA");
  TORCH_CHECK(second_topk_output_logprobs.is_cuda(),
              "second_topk_output_logprobs must be CUDA");
  TORCH_CHECK(second_topk_input_ptrs.scalar_type() == torch::kInt64,
              "second_topk_input_ptrs must be int64");
  TORCH_CHECK(second_topk_output_ptrs.scalar_type() == torch::kInt64,
              "second_topk_output_ptrs must be int64");
  TORCH_CHECK(first_topk_output_ids.scalar_type() == torch::kInt32,
              "first_topk_output_ids must be int32");
  TORCH_CHECK(second_topk_output_logprobs.scalar_type() == torch::kFloat32,
              "second_topk_output_logprobs must be float32");
  const int32_t batch_size =
      static_cast<int32_t>(second_topk_input_ptrs.numel());
  const int32_t max_decoding_draft_tokens =
      static_cast<int32_t>(second_topk_output_logprobs.size(1));
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(second_topk_output_logprobs.device());
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  eagle_extract_scores_and_real_draft_tokens_kernel<<<
      blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      batch_size,
      static_cast<int32_t>(dynamic_tree_max_topk),
      max_decoding_draft_tokens,
      second_topk_input_ptrs.data_ptr<int64_t>(),
      second_topk_output_ptrs.data_ptr<int64_t>(),
      first_topk_output_ids.data_ptr<int32_t>(),
      second_topk_output_logprobs.data_ptr<float>());
}

__global__ void eagle_assemble_third_topk_inputs_kernel(
    int32_t batch_size,
    int32_t max_decoding_draft_tokens,
    int32_t num_eagle_layers,
    int32_t max_nodes_on_final_tree,
    int32_t* __restrict__ third_topks,
    float* __restrict__ all_layers_scores,
    int64_t* __restrict__ third_topk_input_ptrs,
    int32_t* __restrict__ third_topk_output_ids,
    int64_t* __restrict__ third_topk_output_ptrs) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  float* scores_ptr =
      all_layers_scores +
      static_cast<int64_t>(bix) * num_eagle_layers *
          max_decoding_draft_tokens * max_decoding_draft_tokens;
  int32_t* output_ptr =
      third_topk_output_ids +
      static_cast<int64_t>(bix) * max_decoding_draft_tokens;
  third_topk_input_ptrs[bix] =
      static_cast<int64_t>(reinterpret_cast<intptr_t>(scores_ptr));
  third_topk_output_ptrs[bix] =
      static_cast<int64_t>(reinterpret_cast<intptr_t>(output_ptr));
  third_topks[bix] = max_nodes_on_final_tree;
}

void eagle_assemble_third_topk_inputs(torch::Tensor all_layers_scores,
                                      torch::Tensor third_topk_input_ptrs,
                                      torch::Tensor third_topk_output_ids,
                                      torch::Tensor third_topk_output_ptrs,
                                      torch::Tensor third_topks,
                                      int64_t num_eagle_layers,
                                      int64_t max_nodes_on_final_tree) {
  TORCH_CHECK(all_layers_scores.is_cuda(), "all_layers_scores must be CUDA");
  TORCH_CHECK(third_topk_input_ptrs.is_cuda(),
              "third_topk_input_ptrs must be CUDA");
  TORCH_CHECK(third_topk_output_ids.is_cuda(),
              "third_topk_output_ids must be CUDA");
  TORCH_CHECK(third_topk_output_ptrs.is_cuda(),
              "third_topk_output_ptrs must be CUDA");
  TORCH_CHECK(third_topks.is_cuda(), "third_topks must be CUDA");
  TORCH_CHECK(third_topk_input_ptrs.scalar_type() == torch::kInt64,
              "third_topk_input_ptrs must be int64");
  TORCH_CHECK(third_topk_output_ptrs.scalar_type() == torch::kInt64,
              "third_topk_output_ptrs must be int64");
  TORCH_CHECK(third_topk_output_ids.scalar_type() == torch::kInt32,
              "third_topk_output_ids must be int32");
  TORCH_CHECK(third_topks.scalar_type() == torch::kInt32,
              "third_topks must be int32");
  const int32_t batch_size =
      static_cast<int32_t>(third_topk_input_ptrs.numel());
  if (batch_size == 0) {
    return;
  }
  const int32_t max_decoding_draft_tokens =
      static_cast<int32_t>(third_topk_output_ids.size(1));
  c10::cuda::CUDAGuard device_guard(all_layers_scores.device());
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  eagle_assemble_third_topk_inputs_kernel<<<
      blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      batch_size,
      max_decoding_draft_tokens,
      static_cast<int32_t>(num_eagle_layers),
      static_cast<int32_t>(max_nodes_on_final_tree),
      third_topks.data_ptr<int32_t>(),
      all_layers_scores.data_ptr<float>(),
      third_topk_input_ptrs.data_ptr<int64_t>(),
      third_topk_output_ids.data_ptr<int32_t>(),
      third_topk_output_ptrs.data_ptr<int64_t>());
}

__device__ inline int32_t eagle_find_index_in_paths(
    const int32_t* index_map,
    int32_t max_decoding_tokens,
    int32_t index_among_all_draft_tokens) {
  for (int32_t i = 0; i < max_decoding_tokens; ++i) {
    if (index_map[i] == index_among_all_draft_tokens) {
      return i;
    }
  }
  return -1;
}

__global__ void eagle_reconstruct_final_path_kernel(
    int32_t batch_size,
    int32_t dynamic_topk,
    int32_t max_decoding_draft_tokens,
    int32_t max_decoding_tokens,
    int32_t max_path_len,
    int32_t num_eagle_layers,
    int32_t max_nodes_on_final_tree,
    const int64_t* __restrict__ third_topk_output_ptrs,
    const int32_t* __restrict__ all_layers_predecessor,
    int32_t* __restrict__ output_paths) {
  const int32_t bix =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (bix >= batch_size) {
    return;
  }
  extern __shared__ int32_t shared_mem[];
  int32_t* ping_pong_paths = shared_mem;
  int32_t* index_map =
      shared_mem + 2 * batch_size * max_decoding_tokens * max_path_len;
  int32_t* temp_paths[2];

  int32_t* cur_index_map = index_map + bix * max_decoding_tokens;
  for (int32_t ii = 0; ii < max_decoding_tokens; ++ii) {
    cur_index_map[ii] = -1;
  }

  temp_paths[0] = ping_pong_paths + bix * max_decoding_tokens * max_path_len;
  temp_paths[1] = ping_pong_paths +
      (batch_size + bix) * max_decoding_tokens * max_path_len;

  for (int32_t ii = 0; ii < max_decoding_tokens * max_path_len; ++ii) {
    temp_paths[0][ii] = -1;
    temp_paths[1][ii] = -1;
  }

  const int32_t* third_topk_output_ids = reinterpret_cast<const int32_t*>(
      static_cast<intptr_t>(third_topk_output_ptrs[bix]));
  const int32_t* all_layers_predecessor_ptr =
      all_layers_predecessor +
      static_cast<int64_t>(bix) * num_eagle_layers *
          max_decoding_draft_tokens * max_decoding_draft_tokens;
  int32_t* output_paths_ptr =
      output_paths + static_cast<int64_t>(bix) * max_decoding_tokens * max_path_len;

  int32_t* third_topk_sorted =
      const_cast<int32_t*>(third_topk_output_ids);
  eagle_insertion_sort_int32(third_topk_sorted, max_nodes_on_final_tree);

  int32_t* root_path = temp_paths[0];
  int32_t cur_layer_idx = 0;
  root_path[0] = 0;
  int32_t cur_layer_num_paths = 1;
  cur_index_map[0] = 0;

  int32_t cur_topk_index = 0;
  for (int32_t layer = 0; layer < num_eagle_layers; ++layer) {
    int32_t* prev_paths = temp_paths[cur_layer_idx];
    int32_t prev_layer_num_paths = cur_layer_num_paths;
    cur_layer_idx = (cur_layer_idx + 1) & 1;
    int32_t* cur_paths = temp_paths[cur_layer_idx];
    cur_layer_num_paths = 0;

    int32_t used_prev_path_idx = -1;
    const int32_t layer_start = (layer == 0)
        ? 0
        : (layer - 1) * dynamic_topk * dynamic_topk + dynamic_topk;
    const int32_t layer_end = (layer == 0)
        ? layer_start + dynamic_topk
        : layer_start + dynamic_topk * dynamic_topk;

    while (cur_topk_index < max_nodes_on_final_tree) {
      const int32_t idx = third_topk_sorted[cur_topk_index];
      if (idx < layer_start || idx >= layer_end) {
        if (idx >= layer_end) {
          break;
        }
        ++cur_topk_index;
        continue;
      }

      int32_t pred = all_layers_predecessor_ptr[idx];
      int32_t pred_path_idx = -1;
      if (layer == 0) {
        pred_path_idx = 0;
      } else {
        pred_path_idx = eagle_find_index_in_paths(
            cur_index_map, max_decoding_tokens, pred);
      }
      if (pred_path_idx == -1) {
        ++cur_topk_index;
        continue;
      }

      if (pred_path_idx == used_prev_path_idx + 1) {
        used_prev_path_idx++;
      } else if (pred_path_idx > used_prev_path_idx + 1) {
        while (pred_path_idx > used_prev_path_idx + 1) {
          used_prev_path_idx++;
          for (int32_t jj = 0; jj <= layer; ++jj) {
            cur_paths[cur_layer_num_paths * max_path_len + jj] =
                prev_paths[used_prev_path_idx * max_path_len + jj];
          }
          cur_layer_num_paths++;
        }
        used_prev_path_idx++;
      }

      for (int32_t jj = 0; jj <= layer; ++jj) {
        cur_paths[cur_layer_num_paths * max_path_len + jj] =
            prev_paths[pred_path_idx * max_path_len + jj];
      }
      cur_paths[cur_layer_num_paths * max_path_len + layer + 1] = idx + 1;
      cur_index_map[cur_layer_num_paths] = idx + 1;
      cur_layer_num_paths++;
      ++cur_topk_index;
    }

    while (used_prev_path_idx < prev_layer_num_paths - 1) {
      used_prev_path_idx++;
      for (int32_t jj = 0; jj <= layer; ++jj) {
        cur_paths[cur_layer_num_paths * max_path_len + jj] =
            prev_paths[used_prev_path_idx * max_path_len + jj];
      }
      cur_layer_num_paths++;
    }
  }

  int32_t* final_paths = temp_paths[cur_layer_idx];
  for (int32_t ii = 0; ii < max_decoding_tokens * max_path_len; ++ii) {
    output_paths_ptr[ii] = final_paths[ii];
  }
}

void eagle_reconstruct_final_path(torch::Tensor third_topk_output_ptrs,
                                  torch::Tensor all_layers_predecessor,
                                  torch::Tensor output_paths,
                                  int64_t dynamic_tree_max_topk,
                                  int64_t max_decoding_draft_tokens,
                                  int64_t max_decoding_tokens,
                                  int64_t max_path_len,
                                  int64_t num_eagle_layers,
                                  int64_t max_nodes_on_final_tree) {
  TORCH_CHECK(third_topk_output_ptrs.is_cuda(),
              "third_topk_output_ptrs must be CUDA");
  TORCH_CHECK(all_layers_predecessor.is_cuda(),
              "all_layers_predecessor must be CUDA");
  TORCH_CHECK(output_paths.is_cuda(), "output_paths must be CUDA");
  TORCH_CHECK(third_topk_output_ptrs.scalar_type() == torch::kInt64,
              "third_topk_output_ptrs must be int64");
  TORCH_CHECK(all_layers_predecessor.scalar_type() == torch::kInt32,
              "all_layers_predecessor must be int32");
  TORCH_CHECK(output_paths.scalar_type() == torch::kInt32,
              "output_paths must be int32");
  const int32_t batch_size =
      static_cast<int32_t>(third_topk_output_ptrs.numel());
  if (batch_size == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(output_paths.device());
  const int threads = 32;
  const int blocks = (batch_size + threads - 1) / threads;
  const int64_t ping_pong_size =
      2LL * batch_size * max_decoding_tokens * max_path_len * sizeof(int32_t);
  const int64_t index_map_size =
      static_cast<int64_t>(batch_size) * max_decoding_tokens * sizeof(int32_t);
  const int64_t smem_size = ping_pong_size + index_map_size;
  eagle_reconstruct_final_path_kernel<<<
      blocks, threads, static_cast<size_t>(smem_size),
      at::cuda::getDefaultCUDAStream()>>>(
      batch_size,
      static_cast<int32_t>(dynamic_tree_max_topk),
      static_cast<int32_t>(max_decoding_draft_tokens),
      static_cast<int32_t>(max_decoding_tokens),
      static_cast<int32_t>(max_path_len),
      static_cast<int32_t>(num_eagle_layers),
      static_cast<int32_t>(max_nodes_on_final_tree),
      third_topk_output_ptrs.data_ptr<int64_t>(),
      all_layers_predecessor.data_ptr<int32_t>(),
      output_paths.data_ptr<int32_t>());
}

std::vector<torch::Tensor> eagle_topk_logits(torch::Tensor logits,
                                             int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(top_k > 0, "top_k must be > 0");
  auto topk = logits.topk(top_k, 1, true, true);
  auto topk_ids = std::get<1>(topk);
  auto logprobs = logits.log_softmax(1);
  auto topk_logprobs = logprobs.gather(1, topk_ids);
  return {topk_ids, topk_logprobs};
}

std::vector<torch::Tensor> eagle_topk_logits_custom(torch::Tensor logits,
                                                    int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(top_k > 0, "top_k must be > 0");
  const int32_t batch_size = static_cast<int32_t>(logits.size(0));
  const int32_t vocab_size = static_cast<int32_t>(logits.size(1));
  TORCH_CHECK(vocab_size >= top_k, "top_k exceeds vocab size");
  TORCH_CHECK(top_k <= kEagleTopKMax,
              "top_k exceeds custom kernel limit");

  auto out_ids = torch::empty(
      {batch_size, top_k},
      logits.options().dtype(torch::kInt64));
  auto out_logprobs = torch::empty(
      {batch_size, top_k},
      logits.options().dtype(torch::kFloat32));
  if (batch_size == 0) {
    return {out_ids, out_logprobs};
  }
  c10::cuda::CUDAGuard device_guard(logits.device());
  const dim3 grid(batch_size);
  const dim3 block(kEagleTopKBlockSize);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "eagle_topk_logits_custom", [&] {
        eagle_topk_logits_custom_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                logits.data_ptr<scalar_t>(),
                logits.stride(0),
                logits.stride(1),
                vocab_size,
                static_cast<int32_t>(top_k),
                out_ids.data_ptr<int64_t>(),
                out_ids.stride(0),
                out_ids.stride(1),
                out_logprobs.data_ptr<float>(),
                out_logprobs.stride(0),
                out_logprobs.stride(1));
      });
  return {out_ids, out_logprobs};
}

}  // namespace vllm

// Global wrappers for torch_bindings.cpp symbol resolution.
torch::Tensor pack_accepted_tokens(torch::Tensor output_token_ids,
                                   torch::Tensor offsets,
                                   torch::Tensor counts,
                                   int64_t total_tokens) {
  return vllm::pack_accepted_tokens(output_token_ids, offsets, counts,
                                    total_tokens);
}

void build_spec_decode_indices(torch::Tensor num_draft_tokens,
                               torch::Tensor cu_num_scheduled_tokens,
                               torch::Tensor cu_num_draft_tokens,
                               torch::Tensor cu_num_sampled_tokens,
                               torch::Tensor logits_indices,
                               torch::Tensor target_logits_indices,
                               torch::Tensor bonus_logits_indices) {
  vllm::build_spec_decode_indices(num_draft_tokens, cu_num_scheduled_tokens,
                                  cu_num_draft_tokens, cu_num_sampled_tokens,
                                  logits_indices, target_logits_indices,
                                  bonus_logits_indices);
}

void eagle_rewind_slot_mapping(torch::Tensor cu_num_draft_tokens,
                               torch::Tensor valid_sampled_tokens_count,
                               torch::Tensor query_start_loc,
                               torch::Tensor slot_mapping,
                               int64_t pad_id) {
  vllm::eagle_rewind_slot_mapping(cu_num_draft_tokens,
                                 valid_sampled_tokens_count, query_start_loc,
                                 slot_mapping, pad_id);
}

void eagle_compute_slot_mapping(torch::Tensor positions,
                                torch::Tensor block_table,
                                int64_t block_table_stride,
                                int64_t block_size,
                                int64_t max_model_len,
                                torch::Tensor slot_mapping,
                                int64_t pad_id) {
  vllm::eagle_compute_slot_mapping(positions, block_table, block_table_stride,
                                   block_size, max_model_len, slot_mapping,
                                   pad_id);
}

void eagle_update_draft_state(torch::Tensor draft_tokens,
                              torch::Tensor output_hidden_states,
                              int64_t output_hidden_states_stride,
                              torch::Tensor input_ids,
                              torch::Tensor positions,
                              int64_t positions_stride0,
                              torch::Tensor input_hidden_states,
                              int64_t input_hidden_states_stride,
                              torch::Tensor seq_lens,
                              torch::Tensor slot_mapping,
                              torch::Tensor block_table,
                              int64_t block_table_stride,
                              int64_t hidden_size,
                              int64_t block_size,
                              int64_t max_model_len,
                              int64_t pad_id,
                              bool use_mrope) {
  vllm::eagle_update_draft_state(
      draft_tokens, output_hidden_states, output_hidden_states_stride,
      input_ids, positions, positions_stride0, input_hidden_states,
      input_hidden_states_stride, seq_lens, slot_mapping, block_table,
      block_table_stride, hidden_size, block_size, max_model_len, pad_id,
      use_mrope);
}

torch::Tensor eagle_sample_argmax(torch::Tensor logits) {
  return vllm::eagle_sample_argmax(logits);
}

torch::Tensor eagle_sample_topk_topp_gumbel(
    torch::Tensor logits,
    std::optional<torch::Tensor> top_k,
    std::optional<torch::Tensor> top_p,
    std::optional<torch::Tensor> temperature,
    double sampling_eps) {
  return vllm::eagle_sample_topk_topp_gumbel(logits, top_k, top_p, temperature,
                                             sampling_eps);
}

torch::Tensor eagle_expand_draft_tokens(torch::Tensor draft_token_ids,
                                        torch::Tensor cu_num_draft_tokens,
                                        int64_t max_draft_len,
                                        int64_t pad_id) {
  return vllm::eagle_expand_draft_tokens(draft_token_ids, cu_num_draft_tokens,
                                         max_draft_len, pad_id);
}

void eagle_update_draft_state_and_tokens(torch::Tensor draft_tokens,
                                         torch::Tensor output_hidden_states,
                                         int64_t output_hidden_states_stride,
                                         torch::Tensor input_ids,
                                         torch::Tensor positions,
                                         int64_t positions_stride0,
                                         torch::Tensor input_hidden_states,
                                         int64_t input_hidden_states_stride,
                                         torch::Tensor seq_lens,
                                         torch::Tensor slot_mapping,
                                         torch::Tensor block_table,
                                         int64_t block_table_stride,
                                         torch::Tensor output_draft_tokens,
                                         int64_t output_draft_stride,
                                         int64_t step,
                                         int64_t hidden_size,
                                         int64_t block_size,
                                         int64_t max_model_len,
                                         int64_t pad_id,
                                         bool use_mrope) {
  vllm::eagle_update_draft_state_and_tokens(
      draft_tokens, output_hidden_states, output_hidden_states_stride,
      input_ids, positions, positions_stride0, input_hidden_states,
      input_hidden_states_stride, seq_lens, slot_mapping, block_table,
      block_table_stride, output_draft_tokens, output_draft_stride, step,
      hidden_size, block_size, max_model_len, pad_id, use_mrope);
}

void eagle_tree_copy_level(torch::Tensor draft_token_ids,
                           torch::Tensor draft_positions,
                           torch::Tensor draft_hidden_states,
                           torch::Tensor tree_input_ids,
                           torch::Tensor tree_positions,
                           torch::Tensor tree_hidden_states,
                           int64_t level_offset,
                           int64_t hidden_size) {
  vllm::eagle_tree_copy_level(draft_token_ids, draft_positions,
                              draft_hidden_states, tree_input_ids,
                              tree_positions, tree_hidden_states, level_offset,
                              hidden_size);
}

void eagle_tree_select_next_tokens(torch::Tensor topk_ids,
                                   torch::Tensor parent_indices,
                                   torch::Tensor child_indices,
                                   torch::Tensor output_tokens) {
  vllm::eagle_tree_select_next_tokens(topk_ids, parent_indices, child_indices,
                                      output_tokens);
}

void eagle_tree_gather_hidden_states(torch::Tensor hidden_states,
                                     torch::Tensor parent_indices,
                                     torch::Tensor output_hidden_states) {
  vllm::eagle_tree_gather_hidden_states(hidden_states, parent_indices,
                                        output_hidden_states);
}

void eagle_assemble_target_logits_offsets(torch::Tensor logits_ptrs,
                                          torch::Tensor decoding_tokens,
                                          torch::Tensor logits,
                                          torch::Tensor draft_decoding_tokens,
                                          int64_t max_decoding_tokens) {
  vllm::eagle_assemble_target_logits_offsets(logits_ptrs, decoding_tokens,
                                             logits, draft_decoding_tokens,
                                             max_decoding_tokens);
}

void eagle_assemble_draft_logits_offsets(torch::Tensor logits_ptrs,
                                         torch::Tensor logits,
                                         torch::Tensor output_ids_ptrs,
                                         torch::Tensor output_ids,
                                         torch::Tensor skip_decode,
                                         torch::Tensor num_valid_logits,
                                         int64_t num_input_logits,
                                         int64_t max_decoding_draft_tokens) {
  vllm::eagle_assemble_draft_logits_offsets(
      logits_ptrs, logits, output_ids_ptrs, output_ids, skip_decode,
      num_valid_logits, num_input_logits, max_decoding_draft_tokens);
}

void eagle_copy_output_tokens_ids(torch::Tensor tmp_output_ids_ptrs,
                                  torch::Tensor top_ks,
                                  torch::Tensor top_k_offsets,
                                  torch::Tensor input_draft_ids,
                                  torch::Tensor input_draft_lens,
                                  torch::Tensor num_valid_logits,
                                  torch::Tensor output_draft_ids,
                                  torch::Tensor output_draft_lens,
                                  int64_t layer_id,
                                  torch::Tensor input_paths,
                                  torch::Tensor output_paths,
                                  int64_t max_path_len) {
  vllm::eagle_copy_output_tokens_ids(
      tmp_output_ids_ptrs, top_ks, top_k_offsets, input_draft_ids,
      input_draft_lens, num_valid_logits, output_draft_ids,
      output_draft_lens, layer_id, input_paths, output_paths, max_path_len);
}

void eagle_extract_real_draft_tokens(int64_t cur_draft_idx,
                                     int64_t max_draft_len,
                                     int64_t max_total_draft_tokens,
                                     int64_t max_top_k,
                                     int64_t num_tokens_expand_this_layer,
                                     torch::Tensor tokens_gather_idx,
                                     torch::Tensor top_k_list,
                                     torch::Tensor draft_tokens_indices_cumsum,
                                     torch::Tensor new_draft_tokens,
                                     torch::Tensor draft_tokens_buffer) {
  vllm::eagle_extract_real_draft_tokens(
      cur_draft_idx, max_draft_len, max_total_draft_tokens, max_top_k,
      num_tokens_expand_this_layer, tokens_gather_idx, top_k_list,
      draft_tokens_indices_cumsum, new_draft_tokens, draft_tokens_buffer);
}

void eagle_prepare_ctx_eagle_inputs(
    torch::Tensor eagle_seq_lens,
    torch::Tensor eagle_ctx_lens,
    torch::Tensor output_ids,
    torch::Tensor position_ids,
    torch::Tensor hidden_states_indices,
    torch::Tensor last_token_indices,
    torch::Tensor num_last_token_indices,
    torch::Tensor hidden_size_batch_level_starts,
    torch::Tensor input_ids,
    torch::Tensor chunked_context_next_tokens,
    torch::Tensor base_seq_lens,
    torch::Tensor base_ctx_lens,
    torch::Tensor accepted_tokens,
    torch::Tensor accepted_lens,
    torch::Tensor prev_draft_lens,
    torch::Tensor prev_paths,
    torch::Tensor best_path_ids,
    int64_t max_path_len,
    int64_t max_decoding_tokens,
    int64_t max_non_leaves_per_layer) {
  vllm::eagle_prepare_ctx_eagle_inputs(
      eagle_seq_lens, eagle_ctx_lens, output_ids, position_ids,
      hidden_states_indices, last_token_indices, num_last_token_indices,
      hidden_size_batch_level_starts, input_ids, chunked_context_next_tokens,
      base_seq_lens, base_ctx_lens, accepted_tokens, accepted_lens,
      prev_draft_lens, prev_paths, best_path_ids, max_path_len,
      max_decoding_tokens, max_non_leaves_per_layer);
}

void eagle_prepare_gen_eagle_inputs(
    torch::Tensor next_sequence_lengths,
    torch::Tensor next_context_lengths,
    torch::Tensor output_ids,
    torch::Tensor position_ids,
    torch::Tensor spec_decoding_gen_lengths,
    torch::Tensor spec_decoding_position_offsets,
    torch::Tensor spec_decoding_packed_masks,
    torch::Tensor hidden_states_indices,
    torch::Tensor last_token_indices,
    torch::Tensor num_last_token_indices,
    torch::Tensor output_hidden_size_batch_starts_per_level,
    torch::Tensor is_leaf_mask,
    torch::Tensor selected_draft_indices,
    torch::Tensor selected_draft_pos_offsets,
    torch::Tensor num_selected_draft_indices,
    torch::Tensor selected_masks,
    torch::Tensor cum_sum_generation_lengths,
    torch::Tensor max_generation_length,
    torch::Tensor non_leaves_in_level_offsets,
    torch::Tensor parent_non_leaf_in_level_offset,
    torch::Tensor next_draft_ids,
    torch::Tensor eagle_net0_sequence_lengths,
    torch::Tensor prev_context_lengths,
    torch::Tensor input_hidden_size_batch_starts_per_level,
    torch::Tensor next_paths,
    int64_t level_idx,
    int64_t max_path_len,
    int64_t max_decoding_tokens,
    int64_t max_non_leaves_per_layer) {
  vllm::eagle_prepare_gen_eagle_inputs(
      next_sequence_lengths, next_context_lengths, output_ids, position_ids,
      spec_decoding_gen_lengths, spec_decoding_position_offsets,
      spec_decoding_packed_masks, hidden_states_indices, last_token_indices,
      num_last_token_indices, output_hidden_size_batch_starts_per_level,
      is_leaf_mask, selected_draft_indices, selected_draft_pos_offsets,
      num_selected_draft_indices, selected_masks, cum_sum_generation_lengths,
      max_generation_length, non_leaves_in_level_offsets,
      parent_non_leaf_in_level_offset, next_draft_ids,
      eagle_net0_sequence_lengths, prev_context_lengths,
      input_hidden_size_batch_starts_per_level, next_paths, level_idx,
      max_path_len, max_decoding_tokens, max_non_leaves_per_layer);
}

void eagle_update_scores(torch::Tensor cur_log_probs,
                         torch::Tensor prev_layer_scores,
                         int64_t dynamic_tree_max_topk) {
  vllm::eagle_update_scores(cur_log_probs, prev_layer_scores,
                            dynamic_tree_max_topk);
}

void eagle_update_path(int64_t layer_idx,
                       int64_t dynamic_tree_max_topk,
                       torch::Tensor prev_paths,
                       torch::Tensor second_topk_output_ids,
                       torch::Tensor new_paths,
                       torch::Tensor next_expand_indices) {
  vllm::eagle_update_path(layer_idx, dynamic_tree_max_topk, prev_paths,
                          second_topk_output_ids, new_paths,
                          next_expand_indices);
}

void eagle_update_draft_tokens_and_scores(int64_t layer_idx,
                                          int64_t dynamic_tree_max_topk,
                                          torch::Tensor cur_draft_ids,
                                          torch::Tensor input_draft_ids,
                                          torch::Tensor input_draft_lens,
                                          torch::Tensor output_draft_ids,
                                          torch::Tensor output_draft_lens,
                                          torch::Tensor cur_layer_scores,
                                          torch::Tensor output_current_scores) {
  vllm::eagle_update_draft_tokens_and_scores(
      layer_idx, dynamic_tree_max_topk, cur_draft_ids, input_draft_ids,
      input_draft_lens, output_draft_ids, output_draft_lens, cur_layer_scores,
      output_current_scores);
}

void eagle_set_topks_from_dynamic_tree(int64_t layer_idx,
                                       torch::Tensor top_ks,
                                       torch::Tensor top_k_offsets,
                                       int64_t dynamic_tree_max_topk,
                                       torch::Tensor num_valid_logits) {
  vllm::eagle_set_topks_from_dynamic_tree(layer_idx, top_ks, top_k_offsets,
                                          dynamic_tree_max_topk,
                                          num_valid_logits);
}

void eagle_assemble_second_topk_inputs(torch::Tensor first_topk_logprobs,
                                       torch::Tensor second_topk_input_ptrs,
                                       torch::Tensor second_topk_output_ids,
                                       torch::Tensor second_topk_output_ptrs,
                                       int64_t dynamic_tree_max_topk) {
  vllm::eagle_assemble_second_topk_inputs(
      first_topk_logprobs, second_topk_input_ptrs, second_topk_output_ids,
      second_topk_output_ptrs, dynamic_tree_max_topk);
}

void eagle_extract_scores_and_real_draft_tokens(
    torch::Tensor second_topk_input_ptrs,
    torch::Tensor second_topk_output_ptrs,
    torch::Tensor first_topk_output_ids,
    torch::Tensor second_topk_output_logprobs,
    int64_t dynamic_tree_max_topk) {
  vllm::eagle_extract_scores_and_real_draft_tokens(
      second_topk_input_ptrs, second_topk_output_ptrs, first_topk_output_ids,
      second_topk_output_logprobs, dynamic_tree_max_topk);
}

void eagle_assemble_third_topk_inputs(torch::Tensor all_layers_scores,
                                      torch::Tensor third_topk_input_ptrs,
                                      torch::Tensor third_topk_output_ids,
                                      torch::Tensor third_topk_output_ptrs,
                                      torch::Tensor third_topks,
                                      int64_t num_eagle_layers,
                                      int64_t max_nodes_on_final_tree) {
  vllm::eagle_assemble_third_topk_inputs(
      all_layers_scores, third_topk_input_ptrs, third_topk_output_ids,
      third_topk_output_ptrs, third_topks, num_eagle_layers,
      max_nodes_on_final_tree);
}

void eagle_reconstruct_final_path(torch::Tensor third_topk_output_ptrs,
                                  torch::Tensor all_layers_predecessor,
                                  torch::Tensor output_paths,
                                  int64_t dynamic_tree_max_topk,
                                  int64_t max_decoding_draft_tokens,
                                  int64_t max_decoding_tokens,
                                  int64_t max_path_len,
                                  int64_t num_eagle_layers,
                                  int64_t max_nodes_on_final_tree) {
  vllm::eagle_reconstruct_final_path(
      third_topk_output_ptrs, all_layers_predecessor, output_paths,
      dynamic_tree_max_topk, max_decoding_draft_tokens, max_decoding_tokens,
      max_path_len, num_eagle_layers, max_nodes_on_final_tree);
}

void eagle_kv_cache_rewind(torch::Tensor kv_cache,
                           torch::Tensor cu_num_draft_tokens,
                           torch::Tensor valid_sampled_tokens_count,
                           torch::Tensor query_start_loc,
                           torch::Tensor slot_mapping,
                           int64_t block_size,
                           int64_t pad_id) {
  vllm::eagle_kv_cache_rewind(kv_cache, cu_num_draft_tokens,
                             valid_sampled_tokens_count, query_start_loc,
                             slot_mapping, block_size, pad_id);
}

void eagle_compact_slot_mapping(torch::Tensor cu_num_draft_tokens,
                                torch::Tensor query_start_loc,
                                torch::Tensor accepted_offsets,
                                torch::Tensor packed_indices,
                                torch::Tensor slot_mapping,
                                int64_t pad_id) {
  vllm::eagle_compact_slot_mapping(cu_num_draft_tokens, query_start_loc,
                                   accepted_offsets, packed_indices,
                                   slot_mapping, pad_id);
}

void eagle_build_tree_accepted_indices(torch::Tensor draft_token_ids,
                                       torch::Tensor sampled_token_ids,
                                       torch::Tensor valid_sampled_tokens_count,
                                       torch::Tensor level_offsets,
                                       torch::Tensor level_sizes,
                                       torch::Tensor children_offsets,
                                       torch::Tensor children_offsets_start,
                                       torch::Tensor output_indices,
                                       torch::Tensor output_counts) {
  vllm::eagle_build_tree_accepted_indices(
      draft_token_ids, sampled_token_ids, valid_sampled_tokens_count,
      level_offsets, level_sizes, children_offsets, children_offsets_start,
      output_indices, output_counts);
}

void eagle_kv_cache_compact(torch::Tensor kv_cache,
                            torch::Tensor cu_num_draft_tokens,
                            torch::Tensor query_start_loc,
                            torch::Tensor slot_mapping,
                            torch::Tensor accepted_offsets,
                            torch::Tensor packed_indices,
                            int64_t block_size,
                            int64_t pad_id) {
  vllm::eagle_kv_cache_compact(kv_cache, cu_num_draft_tokens, query_start_loc,
                               slot_mapping, accepted_offsets, packed_indices,
                               block_size, pad_id);
}

void eagle_kv_cache_compact_packed(torch::Tensor kv_cache,
                                   torch::Tensor cu_num_draft_tokens,
                                   torch::Tensor query_start_loc,
                                   torch::Tensor slot_mapping,
                                   torch::Tensor accepted_offsets,
                                   torch::Tensor packed_indices,
                                   int64_t block_size,
                                   int64_t pad_id) {
  vllm::eagle_kv_cache_compact_packed(
      kv_cache, cu_num_draft_tokens, query_start_loc, slot_mapping,
      accepted_offsets, packed_indices, block_size, pad_id);
}

std::vector<torch::Tensor> eagle_topk_small(torch::Tensor scores,
                                            int64_t top_k) {
  return vllm::eagle_topk_small(scores, top_k);
}

torch::Tensor eagle_build_packed_tree_mask(torch::Tensor paths,
                                           int64_t tree_len,
                                           bool exclude_root) {
  return vllm::eagle_build_packed_tree_mask(paths, tree_len, exclude_root);
}

std::vector<torch::Tensor> eagle_topk_logits(torch::Tensor logits,
                                             int64_t top_k) {
  return vllm::eagle_topk_logits(logits, top_k);
}

std::vector<torch::Tensor> eagle_topk_logits_custom(torch::Tensor logits,
                                                    int64_t top_k) {
  return vllm::eagle_topk_logits_custom(logits, top_k);
}
