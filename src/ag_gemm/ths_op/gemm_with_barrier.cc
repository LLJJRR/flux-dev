//===- gemm_with_barrier.cc --------------------------------------- C++ ---===//
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
#include "ag_gemm/ths_op/gemm_with_barrier.h"

#include <c10/cuda/CUDAStream.h>

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_runtime_api.h>

#include "coll/ths_op/all_gather_types.h"
#include "coll/ths_op/ag_event_profiler.h"
#include "ag_gemm/ths_op/nccl_signal_all_gather.h"
#include "flux/args/ag_gemm.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/ag_gemm_split.h"

namespace bytedance::flux {

namespace {

struct GWBRecordedEvent {
  int rank;
  const char *mode;
  uint64_t launch_id;
  const char *name;
  cudaEvent_t start;
  cudaEvent_t stop;
};

static thread_local std::vector<GWBRecordedEvent> g_gwb_events;

inline bool
gwb_event_profile_enabled() {
  static const bool enabled =
      std::getenv("FLUX_GWB_EVENT_PROFILE") != nullptr ||
      std::getenv("FLUX_AG_KERNEL_EVENT_PROFILE") != nullptr;
  return enabled;
}

inline bool
ag_wait_profile_enabled() {
  static const bool enabled = std::getenv("FLUX_AG_WAIT_PROFILE") != nullptr;
  return enabled;
}

inline bool
ag_timeline_profile_enabled() {
  static const bool enabled = std::getenv("FLUX_AG_TIMELINE_PROFILE") != nullptr;
  return enabled;
}

inline bool
ag_timeline_profile_enabled_for_rank(int rank) {
  if (!ag_timeline_profile_enabled()) {
    return false;
  }
  const char *target_rank = std::getenv("FLUX_AG_TIMELINE_PROFILE_RANK");
  return target_rank == nullptr || rank == std::atoi(target_rank);
}

inline bool
ag_timeline_profile_launch_selected(uint64_t launch_id) {
  const char *target_launch = std::getenv("FLUX_AG_TIMELINE_PROFILE_LAUNCH");
  return target_launch == nullptr || launch_id == std::strtoull(target_launch, nullptr, 10);
}

inline bool
ag_print_hparams_enabled() {
  static const bool enabled = std::getenv("FLUX_AG_PRINT_HPARAMS") != nullptr;
  return enabled;
}

inline bool
force_h100_agk_best_hparams_enabled() {
  return std::getenv("FLUX_FORCE_H100_AGK_BEST_HPARAMS") != nullptr;
}

inline void
gwb_event_create(cudaEvent_t *event) {
  CUDA_CHECK(cudaEventCreate(event));
}

inline void
gwb_event_record(cudaEvent_t event, cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(event, stream));
}

inline void
gwb_event_push(int rank, const char *name, cudaEvent_t start, cudaEvent_t stop) {
  g_gwb_events.push_back(GWBRecordedEvent{
      rank, ths_op::ag_profile_mode(), ths_op::ag_profile_launch_id(), name, start, stop});
}

inline std::string
debug_tile_shape_to_string(UnifiedTileShape const &shape) {
  std::ostringstream os;
  os << "(" << cute::get<0>(shape) << "," << cute::get<1>(shape) << ","
     << cute::get<2>(shape) << ")";
  return os.str();
}

inline std::string
debug_impl_hparams_value_to_string(None const &) {
  return "None";
}

inline std::string
debug_impl_hparams_value_to_string(unified_type_t<GemmV2HParams> const &v) {
  std::ostringstream os;
  os << "GemmV2HParams{"
     << "warp_shape=" << debug_tile_shape_to_string(v.warp_shape())
     << ", instruction_shape=" << debug_tile_shape_to_string(v.instruction_shape())
     << ", streamk_mode=" << static_cast<int>(v.streamk_mode()) << "}";
  return os.str();
}

inline std::string
debug_impl_hparams_value_to_string(unified_type_t<GemmV3HParams> const &v) {
  std::ostringstream os;
  os << "GemmV3HParams{"
     << "cluster_shape=" << debug_tile_shape_to_string(v.cluster_shape())
     << ", kernel_schedule=" << enum_to_string(v.kernel_schedule())
     << ", blockscale_M=" << static_cast<int>(v.blockscale_M())
     << ", blockscale_N=" << static_cast<int>(v.blockscale_N()) << "}";
  return os.str();
}

inline std::string
debug_impl_hparams_to_string(UnifiedImplHParams const &impl_spec) {
  return std::visit(
      [](auto const &v) -> std::string { return debug_impl_hparams_value_to_string(v); },
      impl_spec);
}

inline std::string
debug_comm_hparams_value_to_string(None const &) {
  return "None";
}

inline std::string
debug_comm_hparams_value_to_string(unified_type_t<GatherRSHParams> const &v) {
  std::ostringstream os;
  os << "GatherRSHParams{"
     << "gather_rs_ctas=" << v.gather_rs_ctas()
     << ", n_dim_per_split=" << v.n_dim_per_split() << "}";
  return os.str();
}

inline std::string
debug_comm_hparams_to_string(UnifiedCommHParams const &comm_spec) {
  return std::visit(
      [](auto const &v) -> std::string { return debug_comm_hparams_value_to_string(v); },
      comm_spec);
}

inline std::string
debug_hparams_to_string(UnifiedGemmHParams const &hparams) {
  std::ostringstream os;
  os << "{"
     << "impl_spec=" << debug_impl_hparams_to_string(hparams.impl_spec())
     << ", comm_spec=" << debug_comm_hparams_to_string(hparams.comm_spec())
     << ", tile_shape=" << debug_tile_shape_to_string(hparams.tile_shape())
     << ", gemm_kind=" << static_cast<int>(hparams.gemm_kind())
     << ", mainloop_stage=" << hparams.mainloop_stage()
     << ", raster_order=" << static_cast<int>(hparams.raster_order()) << "}";
  return os.str();
}

inline void
print_ag_hparams_once(
    int rank,
    UnifiedGemmHParams const &hparams,
    int m,
    int n,
    int k,
    int world_size,
    int nnodes,
    const char *source) {
  static bool printed = false;
  if (!ag_print_hparams_enabled() || rank != 0 || printed) {
    return;
  }
  std::cout << "[AG HPARAMS] source=" << source
            << " M=" << m
            << " N=" << n
            << " K=" << k
            << " world_size=" << world_size
            << " nnodes=" << nnodes
            << " hparams=" << debug_hparams_to_string(hparams)
            << std::endl;
  printed = true;
}

}  // namespace

void
flush_gwb_events_after_sync() {
  if (!gwb_event_profile_enabled()) {
    return;
  }

  for (auto &e : g_gwb_events) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e.start, e.stop));
    std::ostringstream line;
    line << "[GWB EVENT] rank=" << e.rank
         << " mode=" << e.mode
         << " launch=" << e.launch_id
         << " name=" << e.name
         << " ms=" << ms << '\n';
    ths_op::ag_profile_append(line.str());
    CUDA_CHECK(cudaEventDestroy(e.start));
    CUDA_CHECK(cudaEventDestroy(e.stop));
  }

  g_gwb_events.clear();
}

AGGemmMeta
get_gemm_meta(
    at::ScalarType input_torch_dtype,
    at::ScalarType output_torch_dtype,
    bool transpose_weight,
    bool has_bias,
    bool fast_accum) {
  ArchEnum arch = get_arch();
  SMCoreEnum sm_core = get_sm_core();
  auto input_dtype = ths_op::from_torch_dtype(input_torch_dtype);
  auto output_dtype = ths_op::from_torch_dtype(output_torch_dtype);

  bool is_fp8_gemm = ths_op::is_fp8_torch_dtype(input_torch_dtype);
  bool is_s8_gemm = ths_op::is_s8_torch_dtype(input_torch_dtype);
  DataTypeEnum accum_type = is_s8_gemm ? _S32{}() : _FP32{}();
  DataTypeEnum block_scale_type = _FP32{}();
  auto dtype_config = make_gemm_dtype_config(
      input_dtype,
      input_dtype,
      has_bias ? output_dtype : _Void{}(),
      output_dtype,
      accum_type,
      block_scale_type);

  auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
  UnifiedImplMeta impl_spec = None{};

  bool use_fast_accum = fast_accum && is_fp8_gemm;
  auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
  if (impl == _GemmV2{}) {
    impl_spec = make_gemm_v2_meta(use_fast_accum);
  } else if (impl == _GemmV3{}) {
    impl_spec = unify_type(make_gemm_v3_meta(use_fast_accum));
  }

  auto meta =
      make_gemm_meta(dtype_config, arch, sm_core, _AGKernel{}, gemm_layout, impl, impl_spec);
  return meta;
}

RuntimeConfig
get_rt_config(
    int world_size,
    int nnodes,
    int m,
    int n,
    int k,
    AGRingMode ring_mode) {  // TODO(houqi.1993) what about ring mode
  return make_runtime_config(
      m, n, k, make_all_gather_runtime_config(world_size, nnodes, (int)ring_mode));
}

void
GemmWithBarirer::lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
  if (buffer_size <= 0)
    return;
  buffer_size = (buffer_size + 127) / 128 * 128;
  if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
    auto options = input.options().dtype(c10::ScalarType::Byte);
    this->gemm_buffer = torch::empty({buffer_size}, options);
  }
}


void
GemmWithBarirer::lazy_init_ag_wait_profile(torch::Tensor input, int n_data_chunks) {
  if (n_data_chunks <= 0) {
    return;
  }

  auto options_i64 = input.options().dtype(at::ScalarType::Long);
  auto options_i32 = input.options().dtype(at::ScalarType::Int);

  if (ag_wait_profile_enabled() || this->prof_timeline_active) {
    if (!this->prof_wait_max_cycles_buffer.defined() ||
        this->prof_wait_max_cycles_buffer.numel() < n_data_chunks) {
      this->prof_wait_max_cycles_buffer = torch::empty({n_data_chunks}, options_i64);
    }
  }

  if (ag_wait_profile_enabled()) {
    if (!this->prof_wait_cycles_buffer.defined() ||
        this->prof_wait_cycles_buffer.numel() < n_data_chunks) {
      this->prof_wait_cycles_buffer = torch::empty({n_data_chunks}, options_i64);
    }

    if (!this->prof_wait_count_buffer.defined() ||
        this->prof_wait_count_buffer.numel() < n_data_chunks) {
      this->prof_wait_count_buffer = torch::empty({n_data_chunks}, options_i32);
    }

    if (!this->prof_tile_count_buffer.defined() ||
        this->prof_tile_count_buffer.numel() < n_data_chunks) {
      this->prof_tile_count_buffer = torch::empty({n_data_chunks}, options_i32);
    }
  }

  if (this->prof_timeline_active) {
    if (!this->prof_wait_enter_cycles_buffer.defined() ||
        this->prof_wait_enter_cycles_buffer.numel() < n_data_chunks) {
      this->prof_wait_enter_cycles_buffer = torch::empty({n_data_chunks}, options_i64);
    }

    if (!this->prof_wait_exit_cycles_buffer.defined() ||
        this->prof_wait_exit_cycles_buffer.numel() < n_data_chunks) {
      this->prof_wait_exit_cycles_buffer = torch::empty({n_data_chunks}, options_i64);
    }
  }
}

void
GemmWithBarirer::reset_ag_wait_profile(cudaStream_t stream) {
  if (!ag_wait_profile_enabled() && !this->prof_timeline_active) {
    return;
  }

  if (ag_wait_profile_enabled() && this->prof_wait_cycles_buffer.defined()) {
    CUDA_CHECK(cudaMemsetAsync(
        this->prof_wait_cycles_buffer.data_ptr(),
        0,
        this->prof_wait_cycles_buffer.numel() * sizeof(int64_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        this->prof_wait_count_buffer.data_ptr(),
        0,
        this->prof_wait_count_buffer.numel() * sizeof(int32_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        this->prof_tile_count_buffer.data_ptr(),
        0,
        this->prof_tile_count_buffer.numel() * sizeof(int32_t),
        stream));
  }
  if (this->prof_wait_max_cycles_buffer.defined()) {
    CUDA_CHECK(cudaMemsetAsync(
        this->prof_wait_max_cycles_buffer.data_ptr(),
        0,
        this->prof_wait_max_cycles_buffer.numel() * sizeof(int64_t),
        stream));
  }
  if (this->prof_timeline_active && this->prof_wait_enter_cycles_buffer.defined()) {
    CUDA_CHECK(cudaMemsetAsync(
        this->prof_wait_enter_cycles_buffer.data_ptr(),
        0,
        this->prof_wait_enter_cycles_buffer.numel() * sizeof(int64_t),
        stream));
    CUDA_CHECK(cudaMemsetAsync(
        this->prof_wait_exit_cycles_buffer.data_ptr(),
        0,
        this->prof_wait_exit_cycles_buffer.numel() * sizeof(int64_t),
        stream));
  }
}

void
GemmWithBarirer::dump_ag_wait_profile(cudaStream_t stream) {
  if (!ag_wait_profile_enabled() && !this->prof_timeline_active) {
    return;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  torch::Tensor wait_cycles_cpu;
  torch::Tensor wait_max_cycles_cpu;
  torch::Tensor wait_count_cpu;
  torch::Tensor tile_count_cpu;
  torch::Tensor wait_enter_cpu;
  torch::Tensor wait_exit_cpu;

  int64_t *wait_cycles = nullptr;
  int64_t *wait_max_cycles = nullptr;
  int32_t *wait_count = nullptr;
  int32_t *tile_count = nullptr;
  int64_t *wait_enter = nullptr;
  int64_t *wait_exit = nullptr;
  ths_op::NcclSignalTimeline nccl_timeline;
  uint64_t launch_id = this->prof_current_launch_id;

  int n_data_chunks = 0;
  if (ag_wait_profile_enabled() && this->prof_wait_cycles_buffer.defined()) {
    wait_cycles_cpu = this->prof_wait_cycles_buffer.cpu();
    wait_count_cpu = this->prof_wait_count_buffer.cpu();
    tile_count_cpu = this->prof_tile_count_buffer.cpu();
    wait_cycles = wait_cycles_cpu.data_ptr<int64_t>();
    wait_count = wait_count_cpu.data_ptr<int32_t>();
    tile_count = tile_count_cpu.data_ptr<int32_t>();
    n_data_chunks = static_cast<int>(this->prof_wait_cycles_buffer.numel());
  }

  if (this->prof_timeline_active && this->prof_wait_enter_cycles_buffer.defined()) {
    wait_enter_cpu = this->prof_wait_enter_cycles_buffer.cpu();
    wait_exit_cpu = this->prof_wait_exit_cycles_buffer.cpu();
    wait_enter = wait_enter_cpu.data_ptr<int64_t>();
    wait_exit = wait_exit_cpu.data_ptr<int64_t>();
    n_data_chunks = static_cast<int>(this->prof_wait_enter_cycles_buffer.numel());
    if constexpr (::bytedance::flux::kAGGemmSplit == 1) {
      nccl_timeline = ths_op::consume_nccl_signal_timeline(this->rank);
    }
  }

  if (this->prof_wait_max_cycles_buffer.defined()) {
    wait_max_cycles_cpu = this->prof_wait_max_cycles_buffer.cpu();
    wait_max_cycles = wait_max_cycles_cpu.data_ptr<int64_t>();
    n_data_chunks = static_cast<int>(this->prof_wait_max_cycles_buffer.numel());
  }

  const char *profile_mode = nccl_timeline.ready.empty() ? "native" : "nccl_signal";
  if (!nccl_timeline.ready.empty()) {
    std::ostringstream line;
    line << "[AG NCCL TIMELINE] rank=" << this->rank
         << " mode=" << profile_mode
         << " launch=" << launch_id
         << " start_globaltimer=" << nccl_timeline.start
         << " end_globaltimer=" << nccl_timeline.end
         << " duration_globaltimer="
         << (nccl_timeline.end >= nccl_timeline.start
                 ? nccl_timeline.end - nccl_timeline.start
                 : 0)
         << '\n';
    ths_op::ag_profile_append(line.str());
  }
  std::ostringstream profile_output;
  for (int i = 0; i < n_data_chunks; ++i) {
    if (wait_count != nullptr && (wait_count[i] != 0 || tile_count[i] != 0)) {
      double avg_cycles =
          wait_count[i] > 0 ? static_cast<double>(wait_cycles[i]) / wait_count[i] : 0.0;
      profile_output << "[AG WAIT PROFILE] rank=" << this->rank
                     << " mode=" << profile_mode
                     << " launch=" << launch_id
                     << " chunk=" << i
                     << " tiles=" << tile_count[i]
                     << " waits=" << wait_count[i]
                     << " cycles=" << wait_cycles[i]
                     << " avg_cycles=" << avg_cycles
                     << " max_wait=" << (wait_max_cycles != nullptr ? wait_max_cycles[i] : 0)
                     << '\n';
    }
    if (wait_enter != nullptr && (wait_enter[i] != 0 || wait_exit[i] != 0)) {
      uint64_t ready = i < static_cast<int>(nccl_timeline.ready.size()) ? nccl_timeline.ready[i] : 0;
      uint64_t enter = static_cast<uint64_t>(wait_enter[i]);
      uint64_t exit = static_cast<uint64_t>(wait_exit[i]);
      uint64_t sampled_wait = exit >= enter ? exit - enter : 0;
      uint64_t ready_after_enter = ready > enter ? ready - enter : 0;
      uint64_t exit_after_ready = ready != 0 && exit > ready ? exit - ready : 0;
      profile_output << "[AG SIGNAL TIMELINE] rank=" << this->rank
                     << " mode=" << profile_mode
                     << " launch=" << launch_id
                     << " chunk=" << i
                     << " nccl_ready_globaltimer=" << ready
                     << " wait_enter_globaltimer=" << enter
                     << " wait_exit_globaltimer=" << exit
                     << " sampled_wait=" << sampled_wait
                     << " ready_after_enter=" << ready_after_enter
                     << " exit_after_ready=" << exit_after_ready
                     << " max_wait=" << (wait_max_cycles != nullptr ? wait_max_cycles[i] : 0)
                     << '\n';
    }
  }
  std::string output = profile_output.str();
  if (wait_cycles != nullptr || wait_enter != nullptr) {
    uint64_t total_wait = 0;
    uint64_t max_wait = 0;
    uint64_t total_waits = 0;
    for (int i = 0; i < n_data_chunks; ++i) {
      if (wait_cycles != nullptr) {
        total_wait += static_cast<uint64_t>(wait_cycles[i]);
        total_waits += static_cast<uint64_t>(wait_count[i]);
      } else if (wait_enter[i] != 0 && wait_exit[i] >= wait_enter[i]) {
        total_wait += static_cast<uint64_t>(wait_exit[i] - wait_enter[i]);
        total_waits += 1;
      }
      if (wait_max_cycles != nullptr) {
        max_wait = std::max(max_wait, static_cast<uint64_t>(wait_max_cycles[i]));
      }
    }
    std::ostringstream line;
    line << "[AG GEMM WAIT] rank=" << this->rank
         << " mode=" << profile_mode
         << " launch=" << launch_id
         << " unit=globaltimer_ticks"
         << " total_wait=" << total_wait
         << " max_wait=" << max_wait
         << " waits=" << total_waits << '\n';
    ths_op::ag_profile_append(line.str());
  }
  if (!output.empty()) {
    ths_op::ag_profile_append(std::move(output));
  }
}

GemmWithBarirer::GemmWithBarirer(int rank, int world_size, int32_t nnodes)
    : nnodes(nnodes), world_size(world_size), rank(rank) {}

torch::Tensor
GemmWithBarirer::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    torch::Tensor barrier,
    bool fast_accum,
    int32_t *producer_signal,
    bool transpose_weight) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  return forward(
      input,
      weight,
      bias,
      output,
      input_scale,
      weight_scale,
      output_scale,
      barrier,
      fast_accum,
      transpose_weight,
      c10::nullopt,  // use default hparams
      producer_signal,
      stream);
}

torch::Tensor
GemmWithBarirer::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    torch::Tensor barrier,
    bool fast_accum,
    bool transpose_weight,
    c10::optional<UnifiedGemmHParams> const &hparams,
    int32_t *producer_signal,
    cudaStream_t stream) {
  this->prof_current_launch_id = this->prof_next_launch_id++;
  this->prof_timeline_active =
      ag_timeline_profile_enabled_for_rank(this->rank) &&
      ag_timeline_profile_launch_selected(this->prof_current_launch_id);
  const bool prof = gwb_event_profile_enabled();
  const bool has_producer_signal = producer_signal != nullptr;

  cudaEvent_t fwd_start{}, fwd_stop{};
  cudaEvent_t init_start{}, init_stop{};
  cudaEvent_t signal_wait_start{}, signal_wait_stop{};
  cudaEvent_t run_start{}, run_stop{};

  if (prof) {
    gwb_event_create(&fwd_start);
    gwb_event_create(&fwd_stop);
    gwb_event_create(&init_start);
    gwb_event_create(&init_stop);
    gwb_event_create(&run_start);
    gwb_event_create(&run_stop);
    if (has_producer_signal) {
      gwb_event_create(&signal_wait_start);
      gwb_event_create(&signal_wait_stop);
    }

    gwb_event_record(fwd_start, stream);
    gwb_event_record(init_start, stream);
  }

  auto output_tensor = this->initialize(
      input,
      weight,
      bias,
      output,
      input_scale,
      weight_scale,
      output_scale,
      barrier,
      fast_accum,
      transpose_weight,
      hparams,
      stream);

  if (prof) {
    gwb_event_record(init_stop, stream);
  }

  // if not a nullptr, gemm need to wait producer kernel to be launch.
  if (has_producer_signal) {
    if (prof) {
      gwb_event_record(signal_wait_start, stream);
    }

    CU_CHECK(
        CUStreamWaitValue(stream, (CUdeviceptr)(producer_signal), 1, CU_STREAM_WAIT_VALUE_EQ));

    if (prof) {
      gwb_event_record(signal_wait_stop, stream);
    }
  }

  /// GEMM
  if (prof) {
    gwb_event_record(run_start, stream);
  }

  this->run(stream, /*launch_with_pdl=*/false);

  if (prof) {
    gwb_event_record(run_stop, stream);
    gwb_event_record(fwd_stop, stream);

    // Do not synchronize here. The caller already synchronizes the enclosing
    // AGKernel event at the end of forward/gemm_only. We only save events and
    // let flush_gwb_events_after_sync() print them after that outer sync.
    gwb_event_push(this->rank, "GWB_forward_total", fwd_start, fwd_stop);
    gwb_event_push(this->rank, "GWB_initialize_total", init_start, init_stop);
    if (has_producer_signal) {
      gwb_event_push(
          this->rank, "GWB_producer_signal_wait_total", signal_wait_start, signal_wait_stop);
    }
    gwb_event_push(this->rank, "GWB_run_total", run_start, run_stop);
  }

  if (ag_wait_profile_enabled() || this->prof_timeline_active) {
    this->dump_ag_wait_profile(stream);
  }

  return output_tensor;
}

torch::Tensor
GemmWithBarirer::initialize(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    torch::Tensor barrier,
    bool fast_accum,
    bool transpose_weight,
    c10::optional<UnifiedGemmHParams> const &hparams,
    cudaStream_t stream) {
  at::ScalarType input_dtype = input.scalar_type();
  bool is_fp8_gemm = ths_op::is_fp8_torch_dtype(input_dtype);
  bool is_s8_gemm = ths_op::is_s8_torch_dtype(input_dtype);
  at::ScalarType output_dtype =
      (is_fp8_gemm || is_s8_gemm) ? at::ScalarType::BFloat16 : input_dtype;

  // verify all kinds of shapes
  FLUX_CHECK(!(transpose_weight && is_fp8_gemm)) << "FP8 GEMM does not support transpose weight";

  CHECK_NDIM(input, 2);
  CHECK_CUDA(input);
  CHECK_NDIM(weight, 2);
  CHECK_CUDA(weight);

  int m = input.size(0);
  int n = transpose_weight ? weight.size(1) : weight.size(0);
  int k = transpose_weight ? weight.size(0) : weight.size(1);
  CHECK_2D(input, m, k);
  CHECK_TYPE(weight, input_dtype);

  if (bias.has_value()) {
    CHECK_2D(bias.value(), (is_fp8_gemm || is_s8_gemm) ? 1 : m, n);
    CHECK_TYPE(bias.value(), output_dtype);
    CHECK_CUDA(bias.value());
  }
  if (!is_s8_gemm && !is_fp8_gemm) {
    FLUX_CHECK(!input_scale.has_value());
    FLUX_CHECK(!weight_scale.has_value());
  }

  auto meta = get_gemm_meta(
      input_dtype,
      output_dtype,
      transpose_weight,
      /*has_bias=*/bias.has_value(),
      /*fast_accum=*/fast_accum);
  auto rt_config = get_rt_config(
      this->world_size,
      this->nnodes,
      m,
      n,
      k,
      AGRingMode::All2All);  // TODO(houqi.1993) set this later
  if (hparams.has_value()) {
    print_ag_hparams_once(
        this->rank, hparams.value(), m, n, k, this->world_size, this->nnodes, "explicit");
    this->cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
  } else {
    // Temporary H100 / SM90 AGKernel override for validating the profiled best hparams.
    // Enable with:
    //   FLUX_FORCE_H100_AGK_BEST_HPARAMS=1
    // This only affects the exact BF16 AGKernel shape used in the current experiment:
    //   M=2048, N=24576, K=12288, TP=2, nnodes=1, transpose_weight=true.
    const bool force_h100_agk_best_hparams =
        force_h100_agk_best_hparams_enabled() &&
        get_arch() == ArchEnum::Sm90 &&
        input_dtype == at::ScalarType::BFloat16 &&
        output_dtype == at::ScalarType::BFloat16 &&
        transpose_weight &&
        !bias.has_value() &&
        !fast_accum &&
        this->world_size == 2 &&
        this->nnodes == 1 &&
        m == 2048 &&
        n == 24576 &&
        k == 12288;

    if (force_h100_agk_best_hparams) {
      auto params = unify_type(make_gemm_hparams(
          make_gemm_v3_hparams(cute::make_tuple(2l, 1l, 1l), _Cooperative{}()),
          None{},
          cute::make_tuple(256l, 128l, 64l),
          _GemmDefault{}(),
          4,
          _RasterAlongN{}()));

      static bool printed_force_hparams = false;
      if (this->rank == 0 && !printed_force_hparams) {
        std::cout << "[GWB FORCE HPARAMS] use profiled best hparams for "
                  << "M=" << m
                  << " N=" << n
                  << " K=" << k
                  << " world_size=" << this->world_size
                  << " nnodes=" << this->nnodes
                  << " cluster_shape=(2,1,1)"
                  << " tile_shape=(256,128,64)"
                  << " mainloop_stage=4"
                  << " raster_order=RasterAlongN"
                  << std::endl;
        printed_force_hparams = true;
      }

      print_ag_hparams_once(
          this->rank, params, m, n, k, this->world_size, this->nnodes, "forced");
      this->cutlass_op = OpRegistry::instance().get_op(meta, params);
    } else {
      auto params = OpRegistry::instance().get_hparams(meta, rt_config);
      print_ag_hparams_once(
          this->rank, params, m, n, k, this->world_size, this->nnodes, "registry");
      this->cutlass_op = OpRegistry::instance().get_op(meta, params);
    }
  }
  torch::Tensor output_tensor;

  if (output.has_value()) {
    output_tensor = output.value();
    CHECK_2D(output_tensor, m, n);
    CHECK_TYPE(output_tensor, output_dtype);
  } else {
    output_tensor = torch::empty(
        {m, n},
        at::TensorOptions(output_dtype)
            .device(at::kCUDA)
            .device_index(c10::cuda::current_device()));
  }
  const bool wait_prof = ag_wait_profile_enabled();
  const bool timeline_prof = this->prof_timeline_active;
  int n_data_chunks = this->world_size * ::bytedance::flux::kAGGemmSplit;

  uint64_t *prof_wait_cycles = nullptr;
  uint64_t *prof_wait_max_cycles = nullptr;
  uint32_t *prof_wait_count = nullptr;
  uint32_t *prof_tile_count = nullptr;
  uint64_t *prof_wait_enter_cycles = nullptr;
  uint64_t *prof_wait_exit_cycles = nullptr;

  if (wait_prof || timeline_prof) {
    this->lazy_init_ag_wait_profile(input, n_data_chunks);
    this->reset_ag_wait_profile(stream);
    prof_wait_max_cycles = reinterpret_cast<uint64_t *>(
        this->prof_wait_max_cycles_buffer.data_ptr<int64_t>());
  }

  if (wait_prof) {
    prof_wait_cycles =
        reinterpret_cast<uint64_t *>(this->prof_wait_cycles_buffer.data_ptr<int64_t>());
    prof_wait_count =
        reinterpret_cast<uint32_t *>(this->prof_wait_count_buffer.data_ptr<int32_t>());
    prof_tile_count =
        reinterpret_cast<uint32_t *>(this->prof_tile_count_buffer.data_ptr<int32_t>());
  }

  if (timeline_prof) {
    prof_wait_enter_cycles =
        reinterpret_cast<uint64_t *>(this->prof_wait_enter_cycles_buffer.data_ptr<int64_t>());
    prof_wait_exit_cycles =
        reinterpret_cast<uint64_t *>(this->prof_wait_exit_cycles_buffer.data_ptr<int64_t>());
  }

  std::any gemm_args;
  auto data_ptr_or = [](auto &&t, void *other) -> void * {
    return t.has_value() ? t->data_ptr() : other;
  };
  if (is_fp8_gemm) {
    // TODO(houqi.1993) what about output_scale?
    if (output_scale.has_value()) {
      CHECK_CUDA(output_scale.value());
      CHECK_TYPE(output_scale.value(), at::ScalarType::Float);
    }
    gemm_args = AGFP8KernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .rank = rank,
        .world_size = world_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = 0.0f,
        .A = input.data_ptr(),
        .B = weight.data_ptr(),
        .C = nullptr,
        .Aux = nullptr,
        .D = output_tensor.data_ptr(),
        .barrier_buffer = barrier.data_ptr(),
        .Vector = data_ptr_or(bias, nullptr),
        .abs_max_Aux = nullptr,
        .abs_max_D = nullptr,
        .scaleA = (float *)data_ptr_or(input_scale, nullptr),
        .scaleB = (float *)data_ptr_or(weight_scale, nullptr),
        .scaleC = nullptr,
        .scaleD = (float *)data_ptr_or(output_scale, nullptr),
        .scaleAux = nullptr,
        .prof_wait_cycles = prof_wait_cycles,
        .prof_wait_max_cycles = prof_wait_max_cycles,
        .prof_wait_count = prof_wait_count,
        .prof_tile_count = prof_tile_count,
        .prof_wait_enter_cycles = prof_wait_enter_cycles,
        .prof_wait_exit_cycles = prof_wait_exit_cycles};
  } else if (is_s8_gemm) {
    // check input_scale
    FLUX_CHECK(input_scale.has_value());
    torch::Tensor input_scale_t = input_scale.value();
    CHECK_2D(input_scale_t, m, 1);
    CHECK_TYPE(input_scale_t, at::ScalarType::Float);
    CHECK_CUDA(input_scale_t);
    // check weight_scale
    FLUX_CHECK(weight_scale.has_value());
    torch::Tensor weight_scale_t = weight_scale.value();
    CHECK_2D(weight_scale_t, 1, n);
    CHECK_TYPE(weight_scale_t, at::ScalarType::Float);
    CHECK_CUDA(weight_scale_t);

    gemm_args = AGS8KernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .rank = rank,
        .world_size = world_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .A = input.data_ptr(),
        .B = weight.data_ptr(),
        .bias = data_ptr_or(bias, nullptr),
        .output = output_tensor.data_ptr(),
        .scale_A = (float *)input_scale_t.data_ptr(),
        .scale_B = (float *)weight_scale_t.data_ptr(),
        .barrier_buffer = barrier.data_ptr(),
        .prof_wait_cycles = prof_wait_cycles,
        .prof_wait_max_cycles = prof_wait_max_cycles,
        .prof_wait_count = prof_wait_count,
        .prof_tile_count = prof_tile_count,
        .prof_wait_enter_cycles = prof_wait_enter_cycles,
        .prof_wait_exit_cycles = prof_wait_exit_cycles};
  } else {
    // AG GEMM Arguments
    gemm_args = AGKernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .rank = rank,
        .world_size = world_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = data_ptr_or(bias, nullptr),
        .output = output_tensor.data_ptr(),
        .barrier_buffer = barrier.data_ptr(),
        .prof_wait_cycles = prof_wait_cycles,
        .prof_wait_max_cycles = prof_wait_max_cycles,
        .prof_wait_count = prof_wait_count,
        .prof_tile_count = prof_tile_count,
        .prof_wait_enter_cycles = prof_wait_enter_cycles,
        .prof_wait_exit_cycles = prof_wait_exit_cycles};
  }

  // AG Gemm Workspace
  int64_t workspace_size = this->cutlass_op->get_workspace_size(gemm_args);
  this->lazy_init_gemm_buffer(input, workspace_size);

  /// GEMM initialize
  this->cutlass_op->initialize(
      gemm_args, workspace_size ? this->gemm_buffer.data_ptr() : nullptr, stream);
  return output_tensor;
}

void
GemmWithBarirer::run(cudaStream_t stream, bool launch_with_pdl) {
  this->cutlass_op->run(stream, launch_with_pdl);
}

}  // namespace bytedance::flux
