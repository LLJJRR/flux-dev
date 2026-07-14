//===- nccl_signal_reduce_scatter.cc ----------------------------- C++ ---===//
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

#include "gemm_rs/ths_op/nccl_signal_reduce_scatter.h"

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
  return std::getenv("FLUX_RS_NCCL_DEBUG") != nullptr;
}

void
nccl_signal_debug(int rank, const char *message) {
  if (nccl_signal_debug_enabled()) {
    std::fprintf(stderr, "[FLUX_RS_NCCL_DEBUG] rank=%d %s\n", rank, message);
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
make_byte_storage(size_t nbytes) {
  return torch::empty(
      {static_cast<int64_t>(nbytes)},
      torch::TensorOptions()
          .device(torch::kCUDA)
          .device_index(at::cuda::current_device())
          .dtype(torch::kByte));
}

}  // namespace

NcclSignalReduceScatter::NcclSignalReduceScatter(std::shared_ptr<Group> group)
    : group_(std::move(group)),
      nccl_comm_(create_nccl_comm_with_group(group_.get())),
      signal_storage_(make_byte_storage(sizeof(ncclFluxAgSignal_t))),
      counter_storage_(make_byte_storage(sizeof(int) * group_->get_size())) {
  nccl_signal_debug(group_->get_rank(), "NcclSignalReduceScatter constructed");
}

NcclSignalReduceScatter::~NcclSignalReduceScatter() {
  if (nccl_comm_ != nullptr) {
    nccl_signal_debug(group_->get_rank(), "ncclCommDestroy begin");
    NCCL_CHECK(ncclCommDestroy(nccl_comm_));
    nccl_signal_debug(group_->get_rank(), "ncclCommDestroy end");
  }
}

int
NcclSignalReduceScatter::group_size() const {
  return group_->get_size();
}

int
NcclSignalReduceScatter::rank() const {
  return group_->get_rank();
}

void
NcclSignalReduceScatter::run(
    const void *input,
    void *output,
    void *barrier_buffer,
    size_t count_per_rank,
    ncclDataType_t datatype,
    cudaStream_t stream,
    bool emit_signal) {
  FLUX_CHECK(input != nullptr);
  FLUX_CHECK(output != nullptr);
  FLUX_CHECK(barrier_buffer != nullptr);
  FLUX_CHECK_GT(count_per_rank, 0);

  if (!emit_signal) {
    nccl_signal_debug(group_->get_rank(), "standard ncclReduceScatter begin");
    NCCL_CHECK(ncclReduceScatter(
        input,
        output,
        count_per_rank,
        datatype,
        ncclSum,
        nccl_comm_,
        stream));
    nccl_signal_debug(group_->get_rank(), "standard ncclReduceScatter end");
    return;
  }

  CUDA_CHECK(cudaMemsetAsync(
      counter_storage_.data_ptr(),
      0,
      counter_storage_.nbytes(),
      stream));

  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync begin");
  ncclFluxAgSignal_t signal = {
      .barrier = static_cast<int *>(barrier_buffer),
      .counters = static_cast<int *>(counter_storage_.data_ptr()),
      .launchSignal = nullptr,
      .readyCycles = nullptr,
      .split = 1,
  };

  CUDA_CHECK(cudaMemcpyAsync(
      signal_storage_.data_ptr(),
      &signal,
      sizeof(signal),
      cudaMemcpyHostToDevice,
      stream));
  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync end");

  nccl_signal_debug(group_->get_rank(), "ncclReduceScatterFluxSignal begin");
  NCCL_CHECK(ncclReduceScatterFluxSignal(
      input,
      output,
      count_per_rank,
      datatype,
      ncclSum,
      static_cast<const ncclFluxAgSignal_t *>(signal_storage_.data_ptr()),
      nccl_comm_,
      stream));
  nccl_signal_debug(group_->get_rank(), "ncclReduceScatterFluxSignal end");
}

}  // namespace bytedance::flux::ths_op
