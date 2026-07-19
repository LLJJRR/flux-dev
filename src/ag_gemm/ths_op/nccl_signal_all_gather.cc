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

#include "coll/ths_op/ag_event_profiler.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace bytedance::flux::ths_op {

namespace {

struct PendingNcclSignalTimeline {
  int rank;
  torch::Tensor timeline;
  size_t bytes_per_rank;
  int nranks;
};

using FluxNcclSignalLayout = ncclFluxAgSignal_t;

constexpr int kFluxAgPreReadyMagic = 0x46580000;

std::vector<PendingNcclSignalTimeline> nccl_signal_timelines;
std::mutex nccl_signal_events_mutex;

bool
nccl_signal_debug_enabled() {
  return std::getenv("FLUX_AG_NCCL_DEBUG") != nullptr;
}

bool
nccl_signal_inplace_enabled() {
  return std::getenv("FLUX_AG_NCCL_INPLACE") != nullptr;
}

bool
ag_timeline_profile_enabled() {
  return std::getenv("FLUX_AG_TIMELINE_PROFILE") != nullptr;
}

bool
ag_timeline_profile_enabled_for_rank(int rank) {
  if (!ag_timeline_profile_enabled()) {
    return false;
  }
  const char *target_rank = std::getenv("FLUX_AG_TIMELINE_PROFILE_RANK");
  return target_rank == nullptr || rank == std::atoi(target_rank);
}

bool
ag_timeline_profile_launch_selected(uint64_t launch_id) {
  const char *target_launch = std::getenv("FLUX_AG_TIMELINE_PROFILE_LAUNCH");
  return target_launch == nullptr || launch_id == std::strtoull(target_launch, nullptr, 10);
}

void
nccl_signal_debug(int rank, const char *message) {
  if (nccl_signal_debug_enabled()) {
    std::fprintf(stderr, "[FLUX_AG_NCCL_DEBUG] rank=%d %s\n", rank, message);
    std::fflush(stderr);
  }
}

void
nccl_signal_timeline_push(int rank, torch::Tensor ready_cycles, size_t bytes_per_rank, int nranks) {
  std::lock_guard<std::mutex> lock(nccl_signal_events_mutex);
  nccl_signal_timelines.push_back(
      PendingNcclSignalTimeline{rank, ready_cycles, bytes_per_rank, nranks});
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

NcclSignalTimeline
consume_nccl_signal_timeline(int rank) {
  torch::Tensor ready_cycles_storage;
  size_t bytes_per_rank = 0;
  int nranks = 0;
  {
    std::lock_guard<std::mutex> lock(nccl_signal_events_mutex);
    for (auto iter = nccl_signal_timelines.begin(); iter != nccl_signal_timelines.end(); ++iter) {
      if (iter->rank != rank) {
        continue;
      }
      ready_cycles_storage = iter->timeline;
      bytes_per_rank = iter->bytes_per_rank;
      nranks = iter->nranks;
      nccl_signal_timelines.erase(iter);
      break;
    }
  }
  if (!ready_cycles_storage.defined()) {
    return {};
  }
  auto ready_cycles_cpu = ready_cycles_storage.cpu();
  auto *timeline = reinterpret_cast<uint64_t *>(ready_cycles_cpu.data_ptr<uint8_t>());
  int n_data_chunks = static_cast<int>(ready_cycles_storage.nbytes() / sizeof(uint64_t)) - 2;
  if (n_data_chunks <= 0 || timeline[0] == 0) {
    return {};
  }
  return NcclSignalTimeline{
      timeline[0],
      timeline[1],
      bytes_per_rank,
      nranks,
      std::vector<uint64_t>(timeline + 2, timeline + 2 + n_data_chunks)};
}

NcclSignalAllGather::NcclSignalAllGather(std::shared_ptr<Group> group)
    : group_(std::move(group)),
      nccl_comm_(create_nccl_comm_with_group(group_.get())),
      signal_storage_(make_byte_storage(sizeof(FluxNcclSignalLayout))),
      counter_storage_(make_byte_storage(sizeof(int) * group_->get_size())) {
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
    AgEventTimer timer("standard_ncclAllGather_total", group_->get_rank(), stream);

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

  uint64_t local_launch_id = this->profile_launch_id_++;
  uint64_t profile_launch_id = ag_profile_context_active()
      ? ag_profile_launch_id()
      : local_launch_id;

  const bool timeline_profile =
      ag_timeline_profile_enabled_for_rank(group_->get_rank()) &&
      ag_timeline_profile_launch_selected(profile_launch_id);
  const bool use_inplace = nccl_signal_inplace_enabled();
  AgEventTimer total_timer("nccl_signal_run_total", group_->get_rank(), stream);

  {
    AgEventTimer timer("counter_memset", group_->get_rank(), stream);
    CUDA_CHECK(cudaMemsetAsync(
        counter_storage_.data_ptr(),
        0,
        counter_storage_.nbytes(),
        stream));
  }
  if (timeline_profile) {
    if (!ready_cycles_storage_.defined() ||
        ready_cycles_storage_.numel() < static_cast<int64_t>(group_->get_size() + 2)) {
      ready_cycles_storage_ = make_byte_storage(sizeof(uint64_t) * (group_->get_size() + 2));
    }
    CUDA_CHECK(cudaMemsetAsync(
        ready_cycles_storage_.data_ptr(),
        0,
        ready_cycles_storage_.nbytes(),
        stream));
  }

  const void *nccl_input = input;
  if (use_inplace) {
    FLUX_CHECK_LE(group_->get_rank(), 0xffff)
        << "FLUX_AG_NCCL_INPLACE supports ranks representable by the signal token";
    auto *local_input = static_cast<uint8_t *>(input_buffer) +
        static_cast<size_t>(group_->get_rank()) * bytes_per_rank;
    if (local_input != input) {
      AgEventTimer timer("inplace_local_copy", group_->get_rank(), stream);
      CUDA_CHECK(cudaMemcpyAsync(
          local_input, input, bytes_per_rank, cudaMemcpyDeviceToDevice, stream));
    }
    {
      AgEventTimer timer("inplace_local_ready", group_->get_rank(), stream);
      CU_CHECK(CUStreamWriteValue(
          stream,
          reinterpret_cast<CUdeviceptr>(static_cast<int *>(barrier_buffer) + group_->get_rank()),
          1,
          CU_STREAM_WRITE_VALUE_DEFAULT));
    }
    nccl_input = local_input;
  }

  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync begin");
  FluxNcclSignalLayout signal = {
      .barrier = static_cast<int *>(barrier_buffer),
      .counters = static_cast<int *>(counter_storage_.data_ptr()),
      .launchSignal = nullptr,
      .split = 1,
      .preReadyRankToken =
          use_inplace ? kFluxAgPreReadyMagic | group_->get_rank() : 0,
      .readyCycles = timeline_profile
          ? static_cast<unsigned long long *>(ready_cycles_storage_.data_ptr()) + 2
          : nullptr,
      .startCycles = timeline_profile
          ? static_cast<unsigned long long *>(ready_cycles_storage_.data_ptr())
          : nullptr,
      .endCycles = timeline_profile
          ? static_cast<unsigned long long *>(ready_cycles_storage_.data_ptr()) + 1
          : nullptr,
  };

  {
    AgEventTimer timer("signal_desc_h2d", group_->get_rank(), stream);
    CUDA_CHECK(cudaMemcpyAsync(
        signal_storage_.data_ptr(),
        &signal,
        sizeof(signal),
        cudaMemcpyHostToDevice,
        stream));
  }
  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync end");

  {
    AgEventTimer timer("ncclAllGatherFluxSignal_total", group_->get_rank(), stream);
    nccl_signal_debug(group_->get_rank(), "ncclAllGatherFluxSignal begin");
    NCCL_CHECK(ncclAllGatherFluxSignal(
        nccl_input,
        input_buffer,
        bytes_per_rank,
        ncclInt8,
        reinterpret_cast<const ncclFluxAgSignal_t *>(signal_storage_.data_ptr()),
        nccl_comm_,
        stream));
  }
  nccl_signal_debug(group_->get_rank(), "ncclAllGatherFluxSignal end");

  if (timeline_profile) {
    nccl_signal_timeline_push(
        group_->get_rank(), ready_cycles_storage_, bytes_per_rank, group_->get_size());
  }
}

}  // namespace bytedance::flux::ths_op
