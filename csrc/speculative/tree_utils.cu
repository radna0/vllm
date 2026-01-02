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

#include "../cuda_compat.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token] 
// positions [bs * draft_token] 
// retrive_index [b, draft_token] 
// retrive_next_token [b, draft_token] 
// retrive_next_sibling [b, draft_token]
__global__ void build_tree_efficient_kernel(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
  }
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        // Potential error handling here if needed
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    return;
  }

  int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + tid - 1] / topk;
  if (parent_tb_idx == 0) {
    position = 1;

    if (tree_mask_mode == FULL_MASK) {
      for (int i = 0; i < seq_len; i++) {
        tree_mask[seq_tree_idx + (seq_len + draft_token_num) * tid + i] = true;
      }
    }
  } else {
    int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
    int parent_tid = -1;
    for (int i = 0; i < draft_token_num - 1; i++) {
      if (selected_index[bid * (draft_token_num - 1) + i] == parent_token_idx) {
        parent_tid = i + 1;
        break;
      }
    }
    if (parent_tid != -1) {
      position = positions[bid * draft_token_num + parent_tid] - seq_len + 1;
      int parent_token_tree_idx;
      if (tree_mask_mode == FULL_MASK) {
        parent_token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * parent_tid;
        for (int i = 0; i < seq_len + parent_tid + 1; i++) {
          if (tree_mask[parent_token_tree_idx + i]) {
            tree_mask[token_tree_idx + i - 1] = true;
          }
        }
      } else {
        parent_token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * parent_tid;
        for (int i = 0; i < parent_tid + 1; i++) {
          if (tree_mask[parent_token_tree_idx + i]) {
            tree_mask[token_tree_idx + i - 1] = true;
          }
        }
      }
    }
  }
  positions[bid * draft_token_num + tid] = position + seq_len;
}

void build_tree_kernel_efficient(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  CHECK_INPUT(parent_list);
  CHECK_INPUT(selected_index);
  CHECK_INPUT(verified_seq_len);
  CHECK_INPUT(tree_mask);
  CHECK_INPUT(positions);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);

  int64_t batch_size = verified_seq_len.size(0);

  dim3 grid(batch_size);
  dim3 block(draft_token_num);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(parent_list));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  build_tree_efficient_kernel<<<grid, block, 0, stream>>>(
      parent_list.data_ptr<int64_t>(),
      selected_index.data_ptr<int64_t>(),
      verified_seq_len.data_ptr<int64_t>(),
      tree_mask.data_ptr<bool>(),
      positions.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      (int)topk,
      (int)depth,
      (int)draft_token_num,
      (int)tree_mask_mode);
}

// reconstruct_indices_from_tree_mask from ngram_utils.cu
__global__ void reconstruct_indices_from_tree_mask_kernel(
    bool* tree_mask,
    int64_t* verified_seq_len,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int batch_size,
    int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= batch_size || tid >= draft_token_num) {
    return;
  }
  int base_offset = draft_token_num * draft_token_num;
  int token_idx = bid * draft_token_num;
  int tree_mask_offset = bid * base_offset;

  int depth = 0;
  int parent_idx = -1;

  for (int i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; i--) {
    if (tree_mask[start_idx + i]) {
      depth++;
      if (parent_idx == -1) {
        parent_idx = i;
      }
    }
  }
  retrive_index[token_idx + tid] = token_idx + tid;
  positions[token_idx + tid] = depth + verified_seq_len[bid];

  int next_token_idx = -1;
  for (int i = tid + 1; i < draft_token_num; i++) {
    if (tree_mask[tree_mask_offset + i * draft_token_num + tid]) {
      next_token_idx = i;
      break;
    }
  }
  retrive_next_token[token_idx + tid] = next_token_idx;

  int next_sibling_idx = -1;
  if (parent_idx != -1) {
    for (int i = tid + 1; i < draft_token_num; i++) {
      int start_idx = tree_mask_offset + i * draft_token_num + parent_idx;
      if (tree_mask[start_idx]) {
        bool is_sibling = true;
        int end_idx = tree_mask_offset + i * draft_token_num + i;
        for (int j = start_idx + 1; j < end_idx; ++j) {
          if (tree_mask[j]) {
            is_sibling = false;
            break;
          }
        }
        if (is_sibling) {
          next_sibling_idx = i;
          break;
        }
      }
    }
  }
  retrive_next_sibling[token_idx + tid] = next_sibling_idx;
}

void reconstruct_indices_from_tree_mask(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  dim3 grid(batch_size);
  dim3 block(draft_token_num);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tree_mask));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  reconstruct_indices_from_tree_mask_kernel<<<grid, block, 0, stream>>>(
      tree_mask.data_ptr<bool>(),
      verified_seq_len.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      (int)batch_size,
      (int)draft_token_num);
}
