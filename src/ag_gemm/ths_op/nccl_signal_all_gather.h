//===- nccl_signal_all_gather.h ---------------------------------- C++ ---===//
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
#include <torch/torch.h>
#include <nccl.h>

namespace bytedance::flux::ths_op {

class NcclSignalAllGather {
 public:
  explicit NcclSignalAllGather(std::shared_ptr<Group> group);
  ~NcclSignalAllGather();

  NcclSignalAllGather(const NcclSignalAllGather &) = delete;
  NcclSignalAllGather &operator=(const NcclSignalAllGather &) = delete;

  void run(
      const void *input,
      void *input_buffer,
      void *barrier_buffer,
      size_t bytes_per_rank,
      cudaStream_t stream);

 private:
  std::shared_ptr<Group> group_;
  ncclComm_t nccl_comm_ = nullptr;
  torch::Tensor signal_storage_;
};

}  // namespace bytedance::flux::ths_op
