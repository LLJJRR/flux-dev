//===- nccl_signal_reduce_scatter.h ------------------------------ C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "flux/ths_op/flux_shm.h"

#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <torch/torch.h>

namespace bytedance::flux::ths_op {

class NcclSignalReduceScatter {
 public:
  explicit NcclSignalReduceScatter(std::shared_ptr<Group> group);
  ~NcclSignalReduceScatter();

  NcclSignalReduceScatter(const NcclSignalReduceScatter &) = delete;
  NcclSignalReduceScatter &operator=(const NcclSignalReduceScatter &) = delete;

  void run(
      const void *input,
      void *output,
      void *barrier_buffer,
      size_t bytes_per_rank,
      cudaStream_t stream,
      bool emit_signal = true);

 private:
  std::shared_ptr<Group> group_;
  ncclComm_t nccl_comm_ = nullptr;
  torch::Tensor signal_storage_;
  torch::Tensor counter_storage_;
};

}  // namespace bytedance::flux::ths_op
