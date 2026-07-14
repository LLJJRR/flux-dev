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
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace bytedance::flux::ths_op {

namespace {

struct NcclSignalEvent {
  int rank;
  std::string name;
  cudaEvent_t start;
  cudaEvent_t stop;
};

struct NcclSignalTimeline {
  int rank;
  torch::Tensor ready_cycles;
};

struct FluxNcclSignalLayout {
  int *barrier;
  int *counters;
  int *launchSignal;
  int split;
  unsigned long long *readyCycles;
};

std::mutex nccl_signal_events_mutex;
std::vector<NcclSignalEvent> nccl_signal_events;
std::vector<NcclSignalTimeline> nccl_signal_timelines;

bool
nccl_signal_debug_enabled() {
  return std::getenv("FLUX_AG_NCCL_DEBUG") != nullptr;
}

bool
nccl_signal_event_profile_enabled() {
  return std::getenv("FLUX_AG_NCCL_EVENT_PROFILE") != nullptr;
}

bool
ag_timeline_profile_enabled() {
  return std::getenv("FLUX_AG_TIMELINE_PROFILE") != nullptr;
}

void
nccl_signal_debug(int rank, const char *message) {
  if (nccl_signal_debug_enabled()) {
    std::fprintf(stderr, "[FLUX_AG_NCCL_DEBUG] rank=%d %s\n", rank, message);
    std::fflush(stderr);
  }
}

void
nccl_signal_event_create(cudaEvent_t *event) {
  CUDA_CHECK(cudaEventCreate(event));
}

void
nccl_signal_event_record(cudaEvent_t event, cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(event, stream));
}

void
nccl_signal_event_push(int rank, const char *name, cudaEvent_t start, cudaEvent_t stop) {
  std::lock_guard<std::mutex> lock(nccl_signal_events_mutex);
  nccl_signal_events.push_back(NcclSignalEvent{rank, name, start, stop});
}

void
nccl_signal_timeline_push(int rank, torch::Tensor ready_cycles) {
  std::lock_guard<std::mutex> lock(nccl_signal_events_mutex);
  nccl_signal_timelines.push_back(NcclSignalTimeline{rank, ready_cycles});
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

void
flush_nccl_signal_events_after_sync() {
  if (!nccl_signal_event_profile_enabled() && !ag_timeline_profile_enabled()) {
    return;
  }

  std::vector<NcclSignalEvent> events;
  std::vector<NcclSignalTimeline> timelines;
  {
    std::lock_guard<std::mutex> lock(nccl_signal_events_mutex);
    events.swap(nccl_signal_events);
    timelines.swap(nccl_signal_timelines);
  }

  if (nccl_signal_event_profile_enabled()) {
    for (auto &event : events) {
      CUDA_CHECK(cudaEventSynchronize(event.stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, event.start, event.stop));
      std::cout << "[NCCL SIGNAL EVENT] rank=" << event.rank
                << " name=" << event.name
                << " ms=" << ms
                << std::endl;
      CUDA_CHECK(cudaEventDestroy(event.start));
      CUDA_CHECK(cudaEventDestroy(event.stop));
    }
  }

  if (ag_timeline_profile_enabled()) {
    for (auto &timeline : timelines) {
      auto ready_cycles_cpu = timeline.ready_cycles.cpu();
      auto *ready_cycles = reinterpret_cast<uint64_t *>(ready_cycles_cpu.data_ptr<uint8_t>());
      int n_data_chunks = static_cast<int>(timeline.ready_cycles.nbytes() / sizeof(uint64_t));
      for (int i = 0; i < n_data_chunks; ++i) {
        std::cout << "[NCCL SIGNAL TIMELINE] rank=" << timeline.rank
                  << " chunk=" << i
                  << " ready_globaltimer=" << ready_cycles[i]
                  << std::endl;
      }
    }
  }
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
    cudaEvent_t standard_start{};
    cudaEvent_t standard_stop{};
    if (nccl_signal_event_profile_enabled()) {
      nccl_signal_event_create(&standard_start);
      nccl_signal_event_create(&standard_stop);
      nccl_signal_event_record(standard_start, stream);
    }

    nccl_signal_debug(group_->get_rank(), "standard ncclAllGather begin");
    NCCL_CHECK(ncclAllGather(
        input,
        input_buffer,
        bytes_per_rank,
        ncclInt8,
        nccl_comm_,
        stream));
    nccl_signal_debug(group_->get_rank(), "standard ncclAllGather end");

    if (nccl_signal_event_profile_enabled()) {
      nccl_signal_event_record(standard_stop, stream);
      nccl_signal_event_push(
          group_->get_rank(), "standard_ncclAllGather_total", standard_start, standard_stop);
    }
    return;
  }

  cudaEvent_t total_start{};
  cudaEvent_t total_stop{};
  cudaEvent_t counter_start{};
  cudaEvent_t counter_stop{};
  cudaEvent_t desc_start{};
  cudaEvent_t desc_stop{};
  cudaEvent_t allgather_start{};
  cudaEvent_t allgather_stop{};
  if (nccl_signal_event_profile_enabled()) {
    nccl_signal_event_create(&total_start);
    nccl_signal_event_create(&total_stop);
    nccl_signal_event_create(&counter_start);
    nccl_signal_event_create(&counter_stop);
    nccl_signal_event_create(&desc_start);
    nccl_signal_event_create(&desc_stop);
    nccl_signal_event_create(&allgather_start);
    nccl_signal_event_create(&allgather_stop);
    nccl_signal_event_record(total_start, stream);
    nccl_signal_event_record(counter_start, stream);
  }

  CUDA_CHECK(cudaMemsetAsync(
      counter_storage_.data_ptr(),
      0,
      counter_storage_.nbytes(),
      stream));
  if (ag_timeline_profile_enabled()) {
    if (!ready_cycles_storage_.defined()) {
      ready_cycles_storage_ = make_byte_storage(sizeof(uint64_t) * group_->get_size());
    }
    CUDA_CHECK(cudaMemsetAsync(
        ready_cycles_storage_.data_ptr(),
        0,
        ready_cycles_storage_.nbytes(),
        stream));
  }

  if (nccl_signal_event_profile_enabled()) {
    nccl_signal_event_record(counter_stop, stream);
  }

  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync begin");
  FluxNcclSignalLayout signal = {
      .barrier = static_cast<int *>(barrier_buffer),
      .counters = static_cast<int *>(counter_storage_.data_ptr()),
      .launchSignal = nullptr,
      .split = 1,
      .readyCycles = ag_timeline_profile_enabled()
          ? static_cast<unsigned long long *>(ready_cycles_storage_.data_ptr())
          : nullptr,
  };

  if (nccl_signal_event_profile_enabled()) {
    nccl_signal_event_record(desc_start, stream);
  }

  CUDA_CHECK(cudaMemcpyAsync(
      signal_storage_.data_ptr(),
      &signal,
      sizeof(signal),
      cudaMemcpyHostToDevice,
      stream));
  nccl_signal_debug(group_->get_rank(), "signal cudaMemcpyAsync end");

  if (nccl_signal_event_profile_enabled()) {
    nccl_signal_event_record(desc_stop, stream);
    nccl_signal_event_record(allgather_start, stream);
  }

  nccl_signal_debug(group_->get_rank(), "ncclAllGatherFluxSignal begin");
  NCCL_CHECK(ncclAllGatherFluxSignal(
      input,
      input_buffer,
      bytes_per_rank,
      ncclInt8,
      reinterpret_cast<const ncclFluxAgSignal_t *>(signal_storage_.data_ptr()),
      nccl_comm_,
      stream));
  nccl_signal_debug(group_->get_rank(), "ncclAllGatherFluxSignal end");

  if (nccl_signal_event_profile_enabled()) {
    nccl_signal_event_record(allgather_stop, stream);
    nccl_signal_event_record(total_stop, stream);
    nccl_signal_event_push(group_->get_rank(), "counter_memset", counter_start, counter_stop);
    nccl_signal_event_push(group_->get_rank(), "signal_desc_h2d", desc_start, desc_stop);
    nccl_signal_event_push(
        group_->get_rank(), "ncclAllGatherFluxSignal_total", allgather_start, allgather_stop);
    nccl_signal_event_push(
        group_->get_rank(), "nccl_signal_run_total", total_start, total_stop);
  }

  if (ag_timeline_profile_enabled()) {
    nccl_signal_timeline_push(group_->get_rank(), ready_cycles_storage_);
  }
}

}  // namespace bytedance::flux::ths_op
