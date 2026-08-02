//===- all_gather_gemm_op.cc -------------------------------------- C++ ---===//
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

#include "ag_gemm/ths_op/all_gather_gemm_op.h"

#include "ag_gemm/ths_op/nccl_signal_all_gather.h"
#include "coll/ths_op/ag_event_profiler.h"
#include "coll/ths_op/all_gather_types.h"
#include "coll/ths_op/all_gather_op.h"
#include "flux/args/ag_gemm.h"
#include "ag_gemm/ths_op/gemm_with_barrier.h"
#include "flux/ag_gemm_split.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <torch/all.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <variant>

namespace bytedance {
namespace flux {

void flush_gwb_events_after_sync();

namespace ths_op {
using torch::Tensor;

void flush_ag_events_after_sync();

namespace {
inline void *
ptr_offset(void *ptr, ptrdiff_t offset) {
  return static_cast<char *>(ptr) + offset;
}

inline bool
agk_event_profile_enabled() {
  return std::getenv("FLUX_AG_KERNEL_EVENT_PROFILE") != nullptr;
}

inline bool
ag_event_profile_enabled_for_flush() {
  return std::getenv("FLUX_AG_EVENT_PROFILE") != nullptr;
}

inline bool
gwb_event_profile_enabled_for_flush() {
  return std::getenv("FLUX_GWB_EVENT_PROFILE") != nullptr ||
         std::getenv("FLUX_AG_KERNEL_EVENT_PROFILE") != nullptr;
}

inline bool
ag_nccl_signal_enabled() {
  return std::getenv("FLUX_AG_USE_NCCL_SIGNAL") != nullptr;
}

inline bool
ag_nccl_signal_wait_enabled() {
  return std::getenv("FLUX_AG_NCCL_SIGNAL_WAIT") != nullptr;
}

inline bool
ag_nccl_debug_enabled() {
  return std::getenv("FLUX_AG_NCCL_DEBUG") != nullptr;
}

inline bool
ag_timeline_profile_enabled() {
  return std::getenv("FLUX_AG_TIMELINE_PROFILE") != nullptr;
}

inline void
ag_nccl_debug(int rank, const char *message) {
  if (ag_nccl_debug_enabled()) {
    std::fprintf(stderr, "[FLUX_AG_NCCL_DEBUG] rank=%d %s\n", rank, message);
    std::fflush(stderr);
  }
}

inline void
set_ag_barrier_ready_async(torch::Tensor &barrier, int world_size) {
  int nsignals = world_size * ::bytedance::flux::kAGGemmSplit;
  FLUX_CHECK_GE(barrier.numel(), nsignals);
  barrier.fill_(1);
}

inline void
agk_event_create(cudaEvent_t *event) {
  CUDA_CHECK(cudaEventCreate(event));
}

inline void
agk_event_destroy(cudaEvent_t event) {
  CUDA_CHECK(cudaEventDestroy(event));
}

inline void
agk_event_record(cudaEvent_t event, cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(event, stream));
}

inline void
agk_event_print(int rank, const char *name, cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::ostringstream line;
  line << "[AGK EVENT] rank=" << rank
       << " mode=" << ag_profile_mode()
       << " launch=" << ag_profile_launch_id()
       << " name=" << name
       << " ms=" << ms << '\n';
  ag_profile_append(line.str());
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
     << ", kernel_schedule=" << static_cast<int>(v.kernel_schedule())
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

}  // namespace

/// All Gather GEMM Kernel OP
class AllGatherGemmOp::AllGatherGemmOpImpl {
 private:
  using FlagType = int32_t;

 private:
  std::shared_ptr<Group> tp_group;
  int world_size;
  int nnodes;
  cudaStream_t cp_stream;
  cudaEvent_t cp_event;
  cudaEvent_t ready_event;
  cudaEvent_t all_gather_event;

  GemmWithBarirer gemm_op;
  AllGatherOp ag_op;
  std::unique_ptr<NcclSignalAllGather> nccl_signal_ag;
  // Dedicated ordinary CUDA storage for the NCCL path. The native Flux path
  // continues to use AllGatherOp's NVSHMEM-backed symmetric storage.
  torch::Tensor nccl_input_buffer;
  torch::Tensor nccl_barrier;

  bool use_pdl;  // sm90 feature
  bool disable_nccl_signal_for_profiling = false;
  uint64_t profile_launch_id = 0;

 private:
  AllGatherOption
  materialize(const AllGatherOptionWithOptional opt, bool with_input_scale) {
    return AllGatherOption{
        .input_buffer_copied = opt.input_buffer_copied.value_or(false),
        .use_cuda_core_local = opt.use_cuda_core_local.value_or(with_input_scale),
        .use_cuda_core_ag = opt.use_cuda_core_ag.value_or(with_input_scale),
        .fuse_sync = opt.fuse_sync.value_or(with_input_scale),
        .use_read = opt.use_read.value_or(false),
        .mode = opt.mode.value_or(get_default_ag_ring_mode()),
    };
  }

 public:
  AllGatherGemmOpImpl(
      std::shared_ptr<Group> tp_group,
      int32_t nnodes,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool use_pdl)
      : tp_group(tp_group),
        world_size(tp_group->get_size()),
        nnodes(nnodes),
        gemm_op(tp_group->get_rank(), tp_group->get_size(), nnodes),
        ag_op(tp_group, nnodes, full_m, k_dim, input_dtype),
        nccl_input_buffer(torch::empty(
            {full_m, k_dim},
            torch::TensorOptions().device(torch::kCUDA).dtype(input_dtype))),
        nccl_barrier(torch::zeros(
            {tp_group->get_size()},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt))),
        use_pdl(use_pdl) {
    // copy stream
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &this->cp_stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    // create events
    CUDA_CHECK(cudaEventCreateWithFlags(&this->cp_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));

    if (this->use_pdl) {
      FLUX_CHECK(get_arch() == ArchEnum::Sm90);
    }
  }  // AllGatherGemmOpImpl

  ~AllGatherGemmOpImpl() {
    cudaStreamDestroy(cp_stream);
    cudaEventDestroy(cp_event);
    cudaEventDestroy(ready_event);
    cudaEventDestroy(all_gather_event);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllGatherOptionWithOptional opt,
      c10::optional<torch::Tensor> gathered_input) {
    auto stream = at::cuda::getCurrentCUDAStream();

    bool is_s8_gemm = is_s8_torch_dtype(input.scalar_type());
    bool with_input_scale = is_s8_gemm && input_scale.has_value();
    return forward_impl(
        input,
        weight,
        bias,
        output,
        input_scale,
        weight_scale,
        output_scale,
        fast_accum,
        transpose_weight,
        materialize(opt, with_input_scale),
        gathered_input,
        c10::nullopt,
        stream);
  }

  // never mind the result
  torch::Tensor
  gemm_only(
      torch::Tensor input,  // this should be the full input
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,  // this should be the full scale
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int rank = this->tp_group->get_rank();
    bool prof = agk_event_profile_enabled();

    cudaEvent_t gemm_only_start{};
    cudaEvent_t gemm_only_stop{};

    if (prof) {
      agk_event_create(&gemm_only_start);
      agk_event_create(&gemm_only_stop);
      agk_event_record(gemm_only_start, stream);
    }

    torch::Tensor barrier = torch::ones(
        {this->world_size},
        at::TensorOptions(at::ScalarType::Int)
            .device(torch::kCUDA)
            .device_index(at::cuda::current_device()));

    auto result = this->gemm_op.forward(
        input,  // never mind the result
        weight,
        bias,
        output,
        input_scale,
        weight_scale,
        output_scale,
        barrier,
        fast_accum,
        nullptr,
        transpose_weight);

    if (prof) {
      agk_event_record(gemm_only_stop, stream);
      CUDA_CHECK(cudaEventSynchronize(gemm_only_stop));

      ::bytedance::flux::flush_gwb_events_after_sync();

      agk_event_print(rank, "AGK_gemm_only_total", gemm_only_start, gemm_only_stop);
      ag_profile_flush();

      agk_event_destroy(gemm_only_start);
      agk_event_destroy(gemm_only_stop);
    }

    // Wait/timeline profiling can append output without creating CUDA events.
    ag_profile_flush();

    return result;
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> gathered_input,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    int rank = this->tp_group->get_rank();
    bool use_nccl_signal =
        ag_nccl_signal_enabled() && this->world_size > 1 && !this->disable_nccl_signal_for_profiling;
    const char *mode = use_nccl_signal
        ? (ag_nccl_signal_wait_enabled() ? "nccl_signal_wait" : "nccl_signal_fused")
        : "native";
    AgEventProfilerScope profile_scope(mode, rank, profile_launch_id++);

    if (!ag_nccl_signal_enabled() && use_pdl && opt.use_cuda_core_ag) {
      return forward_with_pdl_impl(
          input,
          weight,
          bias,
          output,
          input_scale,
          weight_scale,
          output_scale,
          fast_accum,
          transpose_weight,
          opt,
          gathered_input,
          hparams,
          stream);
    } else {
      return forward_default_impl(
          input,
          weight,
          bias,
          output,
          input_scale,
          weight_scale,
          output_scale,
          fast_accum,
          transpose_weight,
          opt,
          gathered_input,
          hparams,
          stream);
    }
  }

  torch::Tensor
  forward_default_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> gathered_input,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    int rank = this->tp_group->get_rank();
    bool agk_prof = agk_event_profile_enabled();
    bool ag_prof = ag_event_profile_enabled_for_flush();
    bool gwb_prof = gwb_event_profile_enabled_for_flush();
    bool prof = agk_prof || ag_prof || gwb_prof;

    cudaEvent_t fwd_start{};
    cudaEvent_t fwd_stop{};
    cudaEvent_t cp_ag_start{};
    cudaEvent_t cp_ag_stop{};
    cudaEvent_t wait_local_start{};
    cudaEvent_t wait_local_stop{};
    cudaEvent_t gemm_start{};
    cudaEvent_t gemm_stop{};
    cudaEvent_t gathered_copy_start{};
    cudaEvent_t gathered_copy_stop{};

    if (prof) {
      agk_event_create(&fwd_start);
      agk_event_create(&fwd_stop);
      agk_event_create(&cp_ag_start);
      agk_event_create(&cp_ag_stop);
      agk_event_create(&wait_local_start);
      agk_event_create(&wait_local_stop);
      agk_event_create(&gemm_start);
      agk_event_create(&gemm_stop);
      agk_event_create(&gathered_copy_start);
      agk_event_create(&gathered_copy_stop);

      agk_event_record(fwd_start, stream);
    }

    torch::Tensor barrier = ag_op.local_barrier_buffer();
    int M = input.size(0) * this->world_size;
    torch::Tensor input_buffer = ag_op.local_input_buffer().slice(0, 0, M);
    bool is_s8_gemm = is_s8_torch_dtype(input.scalar_type());
    bool use_nccl_signal =
        ag_nccl_signal_enabled() && this->world_size > 1 && !this->disable_nccl_signal_for_profiling;
    if (use_nccl_signal) {
      FLUX_CHECK_LE(M, nccl_input_buffer.size(0))
          << "NCCL AG input exceeds dedicated buffer capacity";
      input_buffer = nccl_input_buffer.slice(0, 0, M);
      barrier = nccl_barrier;
    }
    if (use_nccl_signal) {
      ag_nccl_debug(
          rank,
          ag_nccl_signal_wait_enabled() ? "forward use NCCL wait path"
                                        : "forward use NCCL fused signal path");
    }
    FLUX_CHECK(!use_nccl_signal || !is_s8_gemm)
        << "FLUX_AG_USE_NCCL_SIGNAL does not support S8 input-scale all-gather yet";
    at::optional<torch::Tensor> input_scale_tensor =
        is_s8_gemm
            ? (input_scale.has_value()
                   ? at::optional<torch::Tensor>{ag_op.local_input_scale_buffer().slice(0, 0, M)}
                   : c10::nullopt)
            : input_scale;

    // TODO(houqi.1993)
    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));

    if (prof) {
      agk_event_record(cp_ag_start, this->cp_stream);
    }

    if (use_nccl_signal) {
      if (this->nccl_signal_ag == nullptr) {
        ag_nccl_debug(rank, "create NcclSignalAllGather begin");
        this->nccl_signal_ag = std::make_unique<NcclSignalAllGather>(this->tp_group);
        ag_nccl_debug(rank, "create NcclSignalAllGather end");
      }

      ag_nccl_debug(rank, "barrier memset begin");
      {
        AgEventTimer timer("NCCL_SIGNAL_barrier_memset", rank, stream);
        CUDA_CHECK(cudaMemsetAsync(barrier.data_ptr(), 0, barrier.nbytes(), stream));
      }
      CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
      CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
      ag_nccl_debug(rank, "barrier memset enqueued");

      ag_nccl_debug(rank, "nccl allgather run begin");
      this->nccl_signal_ag->run(
          input.data_ptr(),
          input_buffer.data_ptr(),
          barrier.data_ptr(),
          input.nbytes(),
          this->cp_stream,
          !ag_nccl_signal_wait_enabled());
      ag_nccl_debug(rank, "nccl allgather run end");
      CUDA_CHECK(cudaEventRecord(this->all_gather_event, this->cp_stream));
      ag_nccl_debug(rank, "all_gather_event recorded");
    } else {
      ag_op.run(input, is_s8_gemm ? input_scale : c10::nullopt, opt, this->cp_stream);
    }

    if (prof) {
      agk_event_record(cp_ag_stop, this->cp_stream);
    }

    if (prof) {
      agk_event_record(wait_local_start, stream);
    }

    if (!use_nccl_signal) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ag_op.get_local_prepare_event()));
    } else if (ag_nccl_signal_wait_enabled()) {
      ag_nccl_debug(rank, "wait all_gather_event begin");
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));
      ag_nccl_debug(rank, "wait all_gather_event enqueued");
      ag_nccl_debug(rank, "set barrier ready begin");
      set_ag_barrier_ready_async(barrier, this->world_size);
      ag_nccl_debug(rank, "set barrier ready end");
    }

    if (prof) {
      agk_event_record(wait_local_stop, stream);
    }

    torch::Tensor result;

    if (prof) {
      agk_event_record(gemm_start, stream);
    }

    if (use_nccl_signal) {
      ag_nccl_debug(rank, "gemm_op.forward begin");
    }
    result = this->gemm_op.forward(
        input_buffer,
        std::move(weight),
        std::move(bias),
        std::move(output),
        input_scale_tensor,
        std::move(weight_scale),
        std::move(output_scale),
        barrier,
        fast_accum,
        transpose_weight,
        hparams,
        opt.use_cuda_core_ag && !use_nccl_signal ? this->ag_op.ag_signal_ptr() : nullptr,
        stream);
    if (use_nccl_signal) {
      ag_nccl_debug(rank, "gemm_op.forward end");
    }

    if (prof) {
      agk_event_record(gemm_stop, stream);
    }

    if (gathered_input.has_value()) {
      if (prof) {
        agk_event_record(gathered_copy_start, stream);
      }

      CHECK_INPUT(gathered_input.value(), input.scalar_type());
      CHECK_2D(gathered_input.value(), input.size(0) * this->world_size, input.size(1));
      CUDA_CHECK(cudaMemcpyAsync(
          gathered_input->data_ptr(),
          input_buffer.data_ptr(),
          gathered_input->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));

      if (prof) {
        agk_event_record(gathered_copy_stop, stream);
      }
    }

    if (prof) {
      agk_event_record(fwd_stop, stream);

      // Synchronize once, after the original forward work has been enqueued.
      CUDA_CHECK(cudaEventSynchronize(fwd_stop));
      CUDA_CHECK(cudaEventSynchronize(cp_ag_stop));

      // GWB events are recorded in gemm_with_barrier.cc. They are safe to flush now
      // because fwd_stop guarantees the main stream has reached the end of gemm_op.forward().
      ::bytedance::flux::flush_gwb_events_after_sync();

      // AG events are recorded in all_gather_op.cc. They are safe to flush now
      // because cp_ag_stop guarantees the AG stream has reached the end of ag_op.run().
      flush_ag_events_after_sync();

      if (agk_prof) {
        agk_event_print(rank, "AGK_forward_default_total", fwd_start, fwd_stop);
        agk_event_print(rank, "AGK_cp_stream_ag_op_run_total", cp_ag_start, cp_ag_stop);
        agk_event_print(rank, "AGK_wait_local_prepare_event", wait_local_start, wait_local_stop);
        agk_event_print(rank, "AGK_forward_gemm_op_forward_total", gemm_start, gemm_stop);

        if (gathered_input.has_value()) {
          agk_event_print(
              rank,
              "AGK_gathered_input_copy_total",
              gathered_copy_start,
              gathered_copy_stop);
        }
      }

      agk_event_destroy(fwd_start);
      agk_event_destroy(fwd_stop);
      agk_event_destroy(cp_ag_start);
      agk_event_destroy(cp_ag_stop);
      agk_event_destroy(wait_local_start);
      agk_event_destroy(wait_local_stop);
      agk_event_destroy(gemm_start);
      agk_event_destroy(gemm_stop);
      agk_event_destroy(gathered_copy_start);
      agk_event_destroy(gathered_copy_stop);
    }

    // Wait/timeline profiling may not create CUDA events, but its dump path
    // has already synchronized and appended records to the pending output.
    ag_profile_flush();

    return result;
  }

  torch::Tensor
  forward_with_pdl_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      const AllGatherOption &opt,
      c10::optional<torch::Tensor> gathered_input,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    torch::Tensor barrier = ag_op.local_barrier_buffer();
    int M = input.size(0) * this->world_size;
    torch::Tensor input_buffer = ag_op.local_input_buffer().slice(0, 0, M);
    bool is_s8_gemm = is_s8_torch_dtype(input.scalar_type());
    at::optional<torch::Tensor> input_scale_tensor =
        is_s8_gemm
            ? (input_scale.has_value()
                   ? at::optional<torch::Tensor>{ag_op.local_input_scale_buffer().slice(0, 0, M)}
                   : c10::nullopt)
            : input_scale;

    auto output_buf = this->gemm_op.initialize(
        input_buffer,
        std::move(weight),
        std::move(bias),
        std::move(output),
        input_scale_tensor,
        std::move(weight_scale),
        std::move(output_scale),
        barrier,
        fast_accum,
        transpose_weight,
        hparams,
        stream);

    ag_op.run(input, is_s8_gemm ? input_scale : c10::nullopt, opt, stream);

    this->gemm_op.run(stream, /*launch_with_pdl=*/true);
    if (gathered_input.has_value()) {
      CHECK_INPUT(gathered_input.value(), input.scalar_type());
      CHECK_2D(gathered_input.value(), input.size(0) * this->world_size, input.size(1));
      CUDA_CHECK(cudaMemcpyAsync(
          gathered_input->data_ptr(),
          input_buffer.data_ptr(),
          gathered_input->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    if (ag_event_profile_enabled() ||
        std::getenv("FLUX_AG_KERNEL_EVENT_PROFILE") != nullptr ||
        std::getenv("FLUX_GWB_EVENT_PROFILE") != nullptr ||
        std::getenv("FLUX_AG_WAIT_PROFILE") != nullptr ||
        std::getenv("FLUX_AG_TIMELINE_PROFILE") != nullptr) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ::bytedance::flux::flush_gwb_events_after_sync();
      flush_ag_events_after_sync();
      ag_profile_flush();
    }
    return output_buf;
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllGatherOptionWithOptional option_,
      c10::optional<torch::Tensor> gathered_input,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    at::ScalarType input_dtype = input.scalar_type();
    bool is_fp8_gemm = is_fp8_torch_dtype(input_dtype);
    bool is_s8_gemm = is_s8_torch_dtype(input_dtype);
    at::ScalarType output_dtype =
        is_fp8_gemm || is_s8_gemm ? at::ScalarType::BFloat16 : input_dtype;

    if (is_fp8_gemm || is_s8_gemm) {
      FLUX_CHECK(!transpose_weight) << "FP8/INT8 GEMM does not support transpose_weight";
    }

    AllGatherOption option = materialize(option_, is_s8_gemm && input_scale.has_value());

    // NOTE: input shape is [m, k], where m = M / TP and M is the size for GEMM.
    int M = input.size(0) * this->world_size;
    int n = transpose_weight ? weight.size(1) : weight.size(0);
    int k = transpose_weight ? weight.size(0) : weight.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto meta = unify_type(get_gemm_meta(
        input_dtype,
        output_dtype,
        transpose_weight,
        /*has_bias=*/bias.has_value(),
        /*fast_accum=*/fast_accum));
    auto rt_config = get_rt_config(
        this->world_size,
        this->nnodes,
        M,  // this should be full M for GEMM
        n,
        k,
        AGRingMode::All2All);  // TODO(houqi.1993) set this later

    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = !opt_ctx ? &tmp_ctx : opt_ctx.get();

    // TODO: Add filter
    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };
    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    const bool restore_disable_nccl_signal_for_profiling = this->disable_nccl_signal_for_profiling;
    this->disable_nccl_signal_for_profiling = true;
    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          constexpr int warmup = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          this->tp_group->sync();
          for (int iter = 0; iter < warmup + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto _ [[maybe_unused]] = this->forward_impl(
                input,
                weight,
                bias,
                output,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                transpose_weight,
                option,  // profile with input buffer
                gathered_input,
                hparams,
                stream);
            timer.stop();
            if (iter >= warmup) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          this->tp_group->sync();
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          float reduce_elapsed = all_reduce_max_float(this->tp_group.get(), avg_elapsed);

          if (this->tp_group->get_rank() == 0) {
            std::cout << "[AGK PROFILE] "
                      << "M=" << M
                      << " N=" << n
                      << " K=" << k
                      << " elapsed_ms=" << reduce_elapsed
                      << " hparams=" << debug_hparams_to_string(hparams)
                      << std::endl;
          }

          ctx->add(meta, rt_config, hparams, reduce_elapsed);
        },
        meta);
    auto best_hparams = ctx->record_best(meta, rt_config);

    if (this->tp_group->get_rank() == 0) {
      std::cout << "[AGK PROFILE] "
                << "M=" << M
                << " N=" << n
                << " K=" << k
                << " best_hparams=" << debug_hparams_to_string(best_hparams)
                << std::endl;
    }

    auto result = this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        transpose_weight,
        option,
        std::move(gathered_input),
        std::move(best_hparams),
        stream);
    this->disable_nccl_signal_for_profiling = restore_disable_nccl_signal_for_profiling;
    return result;
  }
};

AllGatherGemmOp::AllGatherGemmOp(
    std::shared_ptr<Group> tp_group,
    int32_t nnodes,
    int32_t full_m,
    int32_t n_dim,
    int32_t k_dim,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool use_pdl)
    : impl_(new AllGatherGemmOpImpl(
          tp_group, nnodes, full_m, n_dim, k_dim, input_dtype, output_dtype, use_pdl)) {}

AllGatherGemmOp::~AllGatherGemmOp() { delete impl_; }

torch::Tensor
AllGatherGemmOp::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    AllGatherOptionWithOptional opt,
    c10::optional<torch::Tensor> gathered_input) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllGatherGemmOp is not initialized";
  return this->impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight,
      opt,
      std::move(gathered_input));
}

torch::Tensor
AllGatherGemmOp::gemm_only(
    torch::Tensor input,  // this should be the full input
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,  // this should be the full scale
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllGatherGemmOp is not initialized";
  return this->impl_->gemm_only(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight);
}

torch::Tensor
AllGatherGemmOp::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    bool transpose_weight,
    AllGatherOptionWithOptional option_,
    c10::optional<torch::Tensor> gathered_input,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(this->impl_ != nullptr) << "AllGatherGemmOp is not initialized";
  return this->impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      transpose_weight,
      option_,
      std::move(gathered_input),
      opt_ctx);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
