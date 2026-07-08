//===- nccl_signal_all_gather.cc --------------------------------- C++ ---===//
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

#include "ag_gemm/ths_op/nccl_signal_all_gather.h"

#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace bytedance::flux::ths_op {

namespace {

bool
nccl_signal_debug_enabled() {
  return std::getenv("FLUX_AG_NCCL_DEBUG") != nullptr;
}

void
nccl_signal_debug(int rank, const char *message) {
  if (nccl_signal_debug_enabled()) {
    std::fprintf(stderr, "[FLUX_AG_NCCL_DEBUG] rank=%d %s\n", rank, message);
    std::fflush(stderr);
  }
}

ncclComm_t
create_nccl_comm_with_group(Group *group) {
  ncclComm_t comm = nullptr;
  void *host_unique_id = nullptr;
  nccl_signal_debug(group->get_rank(), "create_nccl_comm begin");
  CUDA_CHECK(cudaMallocHost(&host_unique_id, sizeof(ncclUniqueId)));

  ncclUniqueId &id = *static_cast<ncclUniqueId *>(host_unique_id);
  if (group->get_rank() == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }

  group->broadcast_cpu(host_unique_id, sizeof(ncclUniqueId), 0);
  nccl_signal_debug(group->get_rank(), "ncclCommInitRank begin");
  NCCL_CHECK(ncclCommInitRank(&comm, group->get_size(), id, group->get_rank()));
  nccl_signal_debug(group->get_rank(), "ncclCommInitRank end");
  CUDA_CHECK(cudaFreeHost(host_unique_id));
  return comm;
}

torch::Tensor
make_signal_storage() {
  return torch::empty(
      {static_cast<int64_t>(sizeof(ncclFluxAgSignal_t))},
      torch::TensorOptions()
          .device(torch::kCUDA)
          .device_index(at::cuda::current_device())
          .dtype(torch::kByte));
}

}  // namespace

NcclSignalAllGather::NcclSignalAllGather(std::shared_ptr<Group> group)
    : group_(std::move(group)),
      nccl_comm_(create_nccl_comm_with_group(group_.get())),
      signal_storage_(make_signal_storage()) {
  nccl_signal_debug(group_->get_rank(), "NcclSignalAllGather constructed");
}

NcclSignalAllGather::~NcclSignalAllGather() {
  if (nccl_comm_ != nullptr) {
    nccl_signal_debug(group_->get_rank(), "ncclCommDestroy begin");
    NCCL_CHECK(ncclCommDestroy(nccl_comm_));
    nccl_signal_debug(group_->get_rank(), "ncclCommDestroy end");
  }
}

void
NcclSignalAllGather::run(
    const void *input,
    void *input_buffer,
    void *barrier_buffer,
    size_t bytes_per_rank,
    cudaStream_t stream,
    bool emit_signal) {
  FLUX_CHECK(input != nullptr);
  FLUX_CHECK(input_buffer != nullptr);
  FLUX_CHECK(barrier_buffer != nullptr);
  FLUX_CHECK_GT(bytes_per_rank, 0);

  if (!emit_signal) {
    nccl_signal_debug(group_->get_rank(), "standard ncclAllGather begin");
    NCCL_CHECK(ncclAllGather(
        input,
        input_buffer,
        bytes_per_rank,
        ncclInt8,
        nccl_comm_,
        stream));
    nccl_signal_debug(group_->get_rank(), "standard ncclAllGather end");
    return;
  }

  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync begin");
  ncclFluxAgSignal_t signal = {
      .barrier = static_cast<int *>(barrier_buffer),
      .counters = nullptr,
      .launchSignal = nullptr,
      .split = 1,
  };

  CUDA_CHECK(cudaMemcpyAsync(
      signal_storage_.data_ptr(),
      &signal,
      sizeof(signal),
      cudaMemcpyHostToDevice,
      stream));
  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync end");

  nccl_signal_debug(group_->get_rank(), "ncclAllGatherFluxSignal begin");
  NCCL_CHECK(ncclAllGatherFluxSignal(
      input,
      input_buffer,
      bytes_per_rank,
      ncclInt8,
      static_cast<const ncclFluxAgSignal_t *>(signal_storage_.data_ptr()),
      nccl_comm_,
      stream));
  nccl_signal_debug(group_->get_rank(), "ncclAllGatherFluxSignal end");
}

}  // namespace bytedance::flux::ths_op
