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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "speculative_sampling.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

using namespace flashinfer;

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

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(target_probs.scalar_type(), scalar_t, [&] {
    if (deterministic) {
      sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, true,
                                                          scalar_t, int, int64_t>(
          predicts.data_ptr<int>(), accept_index.data_ptr<int>(), accept_token_num.data_ptr<int>(),
          candidates.data_ptr<int64_t>(), retrive_index.data_ptr<int64_t>(), retrive_next_token.data_ptr<int64_t>(),
          retrive_next_sibling.data_ptr<int64_t>(), uniform_samples.data_ptr<scalar_t>(),
          uniform_samples_for_final_sampling.data_ptr<scalar_t>(), target_probs.data_ptr<scalar_t>(),
          draft_probs.data_ptr<scalar_t>(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          scalar_t(threshold_single), scalar_t(threshold_acc), stream);
    } else {
      sampling::TreeSpeculativeSamplingTargetOnlyLauncher<128, BLOCK_SCAN_WARP_SCAN, BLOCK_REDUCE_WARP_REDUCE, 4, false,
                                                          scalar_t, int, int64_t>(
          predicts.data_ptr<int>(), accept_index.data_ptr<int>(), accept_token_num.data_ptr<int>(),
          candidates.data_ptr<int64_t>(), retrive_index.data_ptr<int64_t>(), retrive_next_token.data_ptr<int64_t>(),
          retrive_next_sibling.data_ptr<int64_t>(), uniform_samples.data_ptr<scalar_t>(),
          uniform_samples_for_final_sampling.data_ptr<scalar_t>(), target_probs.data_ptr<scalar_t>(),
          draft_probs.data_ptr<scalar_t>(), batch_size, num_spec_step, num_draft_tokens, vocab_size,
          scalar_t(threshold_single), scalar_t(threshold_acc), stream);
    }
    return true;
  });
}
