// Copyright 2025 ByteDance Ltd. and/or its affiliates.
#pragma once

#include "flux/ths_op/flux_shm.h"

#include <memory>

#include <torch/torch.h>

namespace bytedance::flux::ths_op {

class MoeGatherRSNccl {
 public:
  // N-split weights are materialized at construction time and treated as immutable.
  MoeGatherRSNccl(
      std::shared_ptr<Group> group,
      torch::Tensor weight,
      int64_t num_experts,
      int64_t topk,
      int64_t n_split = 2);
  ~MoeGatherRSNccl();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor splits_cpu,
      torch::Tensor routing_idx,
      torch::Tensor row_scale);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bytedance::flux::ths_op
