// Copyright 2025 ByteDance Ltd. and/or its affiliates.
#pragma once

#include "flux/flux.h"

#include <cuda_runtime_api.h>

namespace bytedance::flux {

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
    cudaStream_t stream);

void moe_unpack_split_major_out(
    void *input,
    void *output,
    DataTypeEnum dtype,
    int token_count,
    int n_dim,
    int n_split,
    cudaStream_t stream);

}  // namespace bytedance::flux
