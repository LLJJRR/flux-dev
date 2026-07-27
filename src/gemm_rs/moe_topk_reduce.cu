// Copyright 2025 ByteDance Ltd. and/or its affiliates.

#include "gemm_rs/moe_topk_reduce.h"

#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace bytedance::flux {
namespace {

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <typename Element>
struct MoeTopKReduceArgs {
  Element *inputs[8];
  int32_t *routing_idx;
  Element *output;
  float *row_scale;
  int n_dim;
  int token_start;
  int token_end;
};

template <typename Element, int TOPK, int THREADS, int GROUPS>
__global__ void moe_topk_reduce_kernel(MoeTopKReduceArgs<Element> args) {
  static_assert(TOPK < 32);
  constexpr int kValuesPerThread = 8;
  constexpr int kWarps = THREADS / 32;
  constexpr int kColumnsPerStep = 32 * kValuesPerThread;
  __shared__ int routing[kWarps * TOPK];

  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int row_stride = kWarps * gridDim.x;
  int row = args.token_start + blockIdx.x * kWarps + warp;

  for (; row < args.token_end; row += row_stride) {
    if (lane < TOPK) {
      routing[warp * TOPK + lane] = args.routing_idx[row * TOPK + lane];
    }
    __syncwarp();

#pragma unroll
    for (int column_step = 0; column_step < args.n_dim / kColumnsPerStep; ++column_step) {
      int column = column_step * kColumnsPerStep + lane * kValuesPerThread;
      float accum[kValuesPerThread] = {0.0f};
#pragma unroll
      for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          int64_t source_offset =
              static_cast<int64_t>(routing[warp * TOPK + k]) * args.n_dim + column;
          Element *source = args.inputs[group] + source_offset;
          int routed_row = routing[warp * TOPK + k];
#pragma unroll
          for (int i = 0; i < kValuesPerThread; ++i) {
            Element scaled =
                from_float<Element>(to_float(source[i]) * args.row_scale[routed_row]);
            accum[i] += to_float(scaled);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kValuesPerThread; ++i) {
        int64_t output_offset =
            static_cast<int64_t>(row - args.token_start) * args.n_dim + column + i;
        args.output[output_offset] =
            from_float<Element>(accum[i]);
      }
    }
  }
}

}  // namespace

void moe_topk_reduce_out(
    void **inputs,
    int input_groups,
    DataTypeEnum dtype,
    int32_t *routing_idx,
    int32_t topk,
    void *output,
    int token_count,
    int n_dim,
    int token_start,
    int tokens_to_process,
    float *row_scale,
    cudaStream_t stream) {
  FLUX_CHECK_GE(token_start, 0);
  FLUX_CHECK_GE(tokens_to_process, 0);
  FLUX_CHECK_LE(token_start + tokens_to_process, token_count);
  FLUX_CHECK(row_scale != nullptr);
  FLUX_CHECK_DIV(n_dim, 256);
  constexpr int kThreads = 768;
  dim3 block(kThreads);
  dim3 grid(264);

  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::_4{}, cute::_5{}),
          cute::make_tuple(cute::_1{}, cute::_2{})),
      [&](auto config) {
        auto [config_dtype, config_topk, config_groups] = config;
        return config_dtype == dtype && config_topk == topk && config_groups == input_groups;
      },
      [&](auto config) {
        auto [config_dtype, config_topk, config_groups] = config;
        constexpr int kTopK = decltype(config_topk){};
        constexpr int kGroups = decltype(config_groups){};
        using Element = decltype(to_cuda_dtype(config_dtype));
        MoeTopKReduceArgs<Element> args{};
        for (int i = 0; i < input_groups; ++i) {
          args.inputs[i] = static_cast<Element *>(inputs[i]);
        }
        args.routing_idx = routing_idx;
        args.output = static_cast<Element *>(output);
        args.row_scale = row_scale;
        args.n_dim = n_dim;
        args.token_start = token_start;
        args.token_end = token_start + tokens_to_process;
        moe_topk_reduce_kernel<Element, kTopK, kThreads, kGroups>
            <<<grid, block, 0, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported MoE top-k reduce configuration: dtype=" << dtype
                          << ", topk=" << topk << ", n_dim=" << n_dim
                          << ", input_groups=" << input_groups;
      });
}

template <typename Element>
__global__ void unpack_split_major_kernel(
    Element const *input, Element *output, int token_count, int n_dim, int chunk_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elements = static_cast<int64_t>(token_count) * n_dim;
  if (idx >= elements) return;
  int token = idx / n_dim;
  int column = idx % n_dim;
  int split = column / chunk_n;
  int split_column = column % chunk_n;
  output[idx] = input[(static_cast<int64_t>(split) * token_count + token) * chunk_n +
                      split_column];
}

void moe_unpack_split_major_out(
    void *input,
    void *output,
    DataTypeEnum dtype,
    int token_count,
    int n_dim,
    int n_split,
    cudaStream_t stream) {
  FLUX_CHECK_DIV(n_dim, n_split);
  int64_t elements = static_cast<int64_t>(token_count) * n_dim;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  if (dtype == _FP16{}) {
    unpack_split_major_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<half const *>(input), static_cast<half *>(output), token_count, n_dim,
        n_dim / n_split);
  } else if (dtype == _BF16{}) {
    unpack_split_major_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const *>(input), static_cast<__nv_bfloat16 *>(output),
        token_count, n_dim, n_dim / n_split);
  } else {
    FLUX_CHECK(false) << "unsupported unpack dtype: " << dtype;
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace bytedance::flux
