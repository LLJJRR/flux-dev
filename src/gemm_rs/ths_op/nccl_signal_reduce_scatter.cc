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
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace bytedance::flux::ths_op {

namespace {

using FluxNcclSignalLayout = ncclFluxAgSignal_t;

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
      signal_storage_(make_byte_storage(sizeof(FluxNcclSignalLayout))),
      counter_storage_(make_byte_storage(sizeof(int) * group_->get_size())),
      producer_ready_storage_(make_byte_storage(sizeof(int) * group_->get_size())) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&comm_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaMemsetAsync(
      producer_ready_storage_.data_ptr(), 0, producer_ready_storage_.nbytes(), comm_stream_));
  nccl_signal_debug(group_->get_rank(), "NcclSignalReduceScatter constructed");
}

NcclSignalReduceScatter::~NcclSignalReduceScatter() {
  if (nccl_comm_ != nullptr) {
    nccl_signal_debug(group_->get_rank(), "ncclCommDestroy begin");
    NCCL_CHECK(ncclCommDestroy(nccl_comm_));
    nccl_signal_debug(group_->get_rank(), "ncclCommDestroy end");
  }
  if (completion_event_ != nullptr) {
    CUDA_CHECK(cudaEventDestroy(completion_event_));
  }
  if (comm_stream_ != nullptr) {
    CUDA_CHECK(cudaStreamDestroy(comm_stream_));
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
  FLUX_CHECK(!emit_signal || barrier_buffer != nullptr);
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
  FluxNcclSignalLayout signal = {
      .barrier = static_cast<int *>(barrier_buffer),
      .counters = static_cast<int *>(counter_storage_.data_ptr()),
      .launchSignal = nullptr,
      .split = 1,
      .preReadyRankToken = 0,
      .readyCycles = nullptr,
      .startCycles = nullptr,
      .endCycles = nullptr,
      .producerReady = nullptr,
      .launchCounter = nullptr,
      .producerEpoch = 0,
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
      reinterpret_cast<const ncclFluxAgSignal_t *>(signal_storage_.data_ptr()),
      nccl_comm_,
      stream));
  nccl_signal_debug(group_->get_rank(), "ncclReduceScatterFluxSignal end");
}

void
NcclSignalReduceScatter::start_overlap(
    const void *input,
    void *output,
    size_t count_per_rank,
    ncclDataType_t datatype,
    int split,
    cudaStream_t compute_stream) {
  (void)compute_stream;
  FLUX_CHECK(input != nullptr);
  FLUX_CHECK(output != nullptr);
  FLUX_CHECK_GT(count_per_rank, 0);
  FLUX_CHECK_GT(split, 0);
  FLUX_CHECK_EQ(count_per_rank % split, 0);
  FLUX_CHECK(!overlap_active_) << "previous NCCL ReduceScatter overlap is still active";
  const char *algo = std::getenv("NCCL_ALGO");
  FLUX_CHECK(algo != nullptr && std::strcmp(algo, "Ring") == 0)
      << "experimental NCCL ReduceScatter overlap requires NCCL_ALGO=Ring";
  const char *proto = std::getenv("NCCL_PROTO");
  FLUX_CHECK(proto != nullptr && std::strcmp(proto, "Simple") == 0)
      << "experimental NCCL ReduceScatter overlap requires NCCL_PROTO=Simple";
  FLUX_CHECK_LT(producer_epoch_, std::numeric_limits<int>::max());
  ++producer_epoch_;
  nccl_signal_debug(group_->get_rank(), "start_overlap enqueue begin");
  active_split_ = split;
  size_t ready_bytes = sizeof(int) * group_->get_size() * split;
  if (producer_ready_storage_.nbytes() < ready_bytes) {
    producer_ready_storage_ = make_byte_storage(ready_bytes);
    CUDA_CHECK(cudaMemsetAsync(
        producer_ready_storage_.data_ptr(), 0, producer_ready_storage_.nbytes(), comm_stream_));
  }

  FluxNcclSignalLayout signal = {
      .barrier = nullptr,
      .counters = nullptr,
      .launchSignal = nullptr,
      .split = split,
      .preReadyRankToken = 0,
      .readyCycles = nullptr,
      .startCycles = nullptr,
      .endCycles = nullptr,
      .producerReady = static_cast<int *>(producer_ready_storage_.data_ptr()),
      .launchCounter = nullptr,
      .producerEpoch = producer_epoch_,
  };
  CUDA_CHECK(cudaMemcpyAsync(
      signal_storage_.data_ptr(),
      &signal,
      sizeof(signal),
      cudaMemcpyHostToDevice,
      comm_stream_));
  NCCL_CHECK(ncclReduceScatterFluxSignal(
      input,
      output,
      count_per_rank,
      datatype,
      ncclSum,
      reinterpret_cast<const ncclFluxAgSignal_t *>(signal_storage_.data_ptr()),
      nccl_comm_,
      comm_stream_));
  CUDA_CHECK(cudaEventRecord(completion_event_, comm_stream_));
  overlap_active_ = true;
  nccl_signal_debug(group_->get_rank(), "start_overlap enqueue end");
}

void
NcclSignalReduceScatter::mark_ready(
    int rank_segment, int split_idx, cudaStream_t compute_stream) {
  FLUX_CHECK(overlap_active_) << "NCCL ReduceScatter overlap has not been started";
  FLUX_CHECK_GE(rank_segment, 0);
  FLUX_CHECK_LT(rank_segment, group_->get_size());
  FLUX_CHECK_GE(split_idx, 0);
  FLUX_CHECK_LT(split_idx, active_split_);
  CU_CHECK(CUStreamWriteValue(
      compute_stream,
      reinterpret_cast<CUdeviceptr>(
          static_cast<int *>(producer_ready_storage_.data_ptr()) +
          rank_segment * active_split_ + split_idx),
      producer_epoch_,
      CU_STREAM_WRITE_VALUE_DEFAULT));
  if (nccl_signal_debug_enabled()) {
    std::fprintf(
        stderr,
        "[FLUX_RS_NCCL_DEBUG] rank=%d mark_ready segment=%d split=%d epoch=%d\n",
        group_->get_rank(),
        rank_segment,
        split_idx,
        producer_epoch_);
    std::fflush(stderr);
  }
}

void
NcclSignalReduceScatter::finish_overlap(cudaStream_t compute_stream) {
  FLUX_CHECK(overlap_active_) << "NCCL ReduceScatter overlap has not been started";
  CUDA_CHECK(cudaStreamWaitEvent(compute_stream, completion_event_, 0));
  overlap_active_ = false;
  nccl_signal_debug(group_->get_rank(), "finish_overlap wait enqueued");
}

}  // namespace bytedance::flux::ths_op
