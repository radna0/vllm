/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>

#include <type_traits>

namespace {

template <typename OutT>
__global__ void nvfp4_unpack_fp4x2_kernel(const uint8_t* __restrict__ input,
                                          OutT* __restrict__ output,
                                          int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      static_cast<int64_t>(threadIdx.x);
  if (idx >= n) {
    return;
  }
  const __nv_fp4x2_storage_t packed =
      static_cast<__nv_fp4x2_storage_t>(input[idx]);
  const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1);
  const __half2 h2 = *reinterpret_cast<const __half2*>(&raw);
  if constexpr (std::is_same_v<OutT, half>) {
    output[idx * 2] = __low2half(h2);
    output[idx * 2 + 1] = __high2half(h2);
  } else {
    const float2 f2 = __half22float2(h2);
    output[idx * 2] = static_cast<OutT>(f2.x);
    output[idx * 2 + 1] = static_cast<OutT>(f2.y);
  }
}

}  // namespace

void nvfp4_unpack_fp4x2_sm1xxa(torch::Tensor& output,
                               torch::Tensor const& input) {
  TORCH_CHECK(input.is_cuda(), "nvfp4_unpack_fp4x2: input must be CUDA");
  TORCH_CHECK(output.is_cuda(), "nvfp4_unpack_fp4x2: output must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kByte,
              "nvfp4_unpack_fp4x2: input must be uint8");
  TORCH_CHECK(output.is_contiguous(),
              "nvfp4_unpack_fp4x2: output must be contiguous");
  TORCH_CHECK(input.is_contiguous(),
              "nvfp4_unpack_fp4x2: input must be contiguous");
  const int64_t n = input.numel();
  TORCH_CHECK(output.numel() == n * 2,
              "nvfp4_unpack_fp4x2: output size must be 2x input size");

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (output.scalar_type() == at::kHalf) {
    nvfp4_unpack_fp4x2_kernel<half><<<blocks, threads, 0, stream>>>(
        input.data_ptr<uint8_t>(), reinterpret_cast<half*>(output.data_ptr()),
        n);
  } else if (output.scalar_type() == at::kFloat) {
    nvfp4_unpack_fp4x2_kernel<float><<<blocks, threads, 0, stream>>>(
        input.data_ptr<uint8_t>(), output.data_ptr<float>(), n);
  } else {
    TORCH_CHECK(false,
                "nvfp4_unpack_fp4x2: output must be float16 or float32");
  }
}
