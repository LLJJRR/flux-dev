################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import argparse
import contextlib
import os
import time
from functools import partial
from typing import Optional

import torch
import torch.distributed

import flux
from flux.testing import (
    DTYPE_MAP,
    RING_MODE_MAP,
    all_gather_into_tensor_with_fp8,
    generate_data,
    initialize_distributed,
    zeros_with_fp8,
    matmul_int8,
)
import flux.testing
from flux.testing.perf_db_helper import should_log_to_rds, set_global_args, log_perf
from flux.util import bench_func, is_fp8_dtype

try:
    from flux.triton.ag_gemm import AgGemmTriton
except Exception as e:
    print("triton module import failed. skip...")

print = partial(print, flush=True)


def benchmark_barrier() -> None:
    torch.cuda.synchronize()
    try:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    except TypeError:
        torch.distributed.barrier()
    torch.cuda.synchronize()


def flux_ag_impl_name() -> str:
    if os.getenv("FLUX_AG_USE_NCCL_SIGNAL") != "1":
        return "flux"
    if os.getenv("FLUX_AG_NCCL_SIGNAL_WAIT") == "1":
        return "flux_nccl_signal_wait"
    return "flux_nccl_signal_fused"


@contextlib.contextmanager
def ag_nccl_signal_env(enabled: bool, wait: bool = False):
    saved = {
        "FLUX_AG_USE_NCCL_SIGNAL": os.environ.get("FLUX_AG_USE_NCCL_SIGNAL"),
        "FLUX_AG_NCCL_SIGNAL_WAIT": os.environ.get("FLUX_AG_NCCL_SIGNAL_WAIT"),
    }

    try:
        if enabled:
            os.environ["FLUX_AG_USE_NCCL_SIGNAL"] = "1"
            if wait:
                os.environ["FLUX_AG_NCCL_SIGNAL_WAIT"] = "1"
            else:
                os.environ.pop("FLUX_AG_NCCL_SIGNAL_WAIT", None)
        else:
            os.environ.pop("FLUX_AG_USE_NCCL_SIGNAL", None)
            os.environ.pop("FLUX_AG_NCCL_SIGNAL_WAIT", None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PerfResult:
    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        gathered_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.output = output
        self.gathered_output = gathered_output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.time3 = time3
        self.gemm_only_time_ms = gemm_only_time_ms

    def __repr__(self) -> str:
        if self.gemm_only_time_ms == 0.0:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms"
            )
        else:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms, {self.time3} {self.gemm_only_time_ms:.3f} ms"
            )


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup: int,
    iters: int,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()

    torch.distributed.barrier()
    # All gather input tensors from all gpus
    full_input = zeros_with_fp8(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    full_input_scale = (
        torch.zeros(
            (M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        if is_s8_dequant
        else None
    )

    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        input = input.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        full_input = full_input.to(torch.bfloat16)

    if is_s8_dequant:
        assert input_scale is not None
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)

    benchmark_barrier()
    warmup_iters = warmup
    for _ in range(warmup_iters):
        torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        if is_s8_dequant:
            accum = matmul_int8(full_input, weight.t()).to(torch.float32)
            output = full_input_scale * weight_scale * accum
        else:
            output = alpha_scale * torch.matmul(full_input, weight.t())

        if is_fp8 or is_s8_dequant:
            output = output.to(torch.bfloat16)
        if bias is not None:
            output += bias

    benchmark_barrier()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()
        if is_s8_dequant:
            accum = matmul_int8(full_input, weight.t()).to(torch.float32)
            output = full_input_scale * weight_scale * accum
        else:
            output = alpha_scale * torch.matmul(full_input, weight.t())

        if is_fp8 or is_s8_dequant:
            output = output.to(torch.bfloat16)
        if bias is not None:
            output += bias
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
        gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux_no_overlap(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    gather_input: bool = True,  # not used. always as true
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
):
    input_dtype = input.dtype
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input.dtype
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    full_input = zeros_with_fp8(
        (M, input.size(1)),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    ag_gemm_op = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input_dtype,
        output_dtype=output_dtype,
    )

    gemm_only_output = torch.empty(
        [M, N], dtype=output_dtype, device=input.device, requires_grad=False
    )

    warmup_iters = warmup
    benchmark_barrier()
    for _ in range(warmup_iters):
        all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
        gemm_only_output = ag_gemm_op.gemm_only(
            full_input,
            w,
            bias=bias,
            input_scale=full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            transpose_weight=transpose_weight,
        )

    benchmark_barrier()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
        allgather_end_events[i].record()

        gemm_only_output = ag_gemm_op.gemm_only(
            full_input,
            w,
            bias=bias,
            input_scale=full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            transpose_weight=transpose_weight,
        )
        end_events[i].record()

    comm_times = []  # all gather
    gemm_times = []  # gemm
    for i in range(iters):
        allgather_end_events[i].synchronize()
        end_events[i].synchronize()
        comm_times.append(start_events[i].elapsed_time(allgather_end_events[i]) / 1000)
        gemm_times.append(allgather_end_events[i].elapsed_time(end_events[i]) / 1000)

    comm_time = sum(comm_times) / iters * 1000
    gemm_time = sum(gemm_times) / iters * 1000

    return PerfResult(
        name=f"flux(no-overlap) #{TP_GROUP.rank()}",
        output=gemm_only_output,
        gathered_output=full_input,
        total_ms=gemm_time + comm_time,
        time1="gemm",
        gemm_time_ms=gemm_time,
        time2="comm",
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    transpose_weight: bool = True,
    gather_input: bool = False,
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
    verify: bool = False,
):
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)
    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    op = AgGemmTriton(TP_GROUP, weight.dtype, M, K, transpose_weight=transpose_weight)

    full_input = (
        zeros_with_fp8(
            (M, K),
            dtype=input.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        if gather_input
        else None
    )

    ag_option = flux.AllGatherOption()
    ag_option.mode = RING_MODE_MAP[args.ring_mode]

    (output, gathered_output), total_time_ms = bench_func(
        lambda: op.forward(
            input,
            w.t(),
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            gathered_input=full_input,
            fast_accum=fast_acc,
            ag_option=ag_option,
        ),
        iters,
        warmup,
    )
    return PerfResult(
        name=f"triton  #{TP_GROUP.rank()}",
        output=output,
        gathered_output=gathered_output,
        total_ms=total_time_ms,
        time1="gemm",
        gemm_time_ms=0,
        time2="comm",
        comm_time_ms=0,
    )


@torch.no_grad()
def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    transpose_weight: bool = True,
    gather_input: bool = False,
    ring_mode: Optional[flux.AGRingMode] = None,
    warmup: int = 5,
    iters: int = 10,
    fast_acc: bool = False,
    verify: bool = False,
    use_cuda_core_local: bool = False,
    use_cuda_core_ag: bool = False,
    use_pdl: bool = False,
    tune_agk=False,
):
    input_dtype = input.dtype
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input_dtype
    local_M = input.size(0)
    M = local_M * TP_GROUP.size()
    K = input.size(1)

    if transpose_weight:
        w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    torch.distributed.barrier()
    full_input = zeros_with_fp8(
        (M, K),
        dtype=input_dtype,
        device=torch.cuda.current_device(),
    )

    full_input_scale = (
        torch.zeros((M, 1), dtype=input_scale.dtype, device=torch.cuda.current_device())
        if is_s8_dequant
        else None
    )
    all_gather_into_tensor_with_fp8(full_input, input, group=TP_GROUP)
    if is_s8_dequant:
        torch.distributed.all_gather_into_tensor(full_input_scale, input_scale, group=TP_GROUP)

    use_fp8_gemm = True if is_fp8 else False
    gemm_only_op = flux.GemmOnly(
        input_dtype=input_dtype,
        weight_dtype=input_dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=use_fp8_gemm,
    )
    gemm_only_output = torch.empty(
        [M, N], dtype=output_dtype, device=input.device, requires_grad=False
    )

    ag_gemm_output = torch.empty([M, N], dtype=output_dtype, device=input.device)
    ag_option = flux.AllGatherOption()
    ag_option.mode = ring_mode
    all_gather_gemm_kernel = flux.AGKernel(
        TP_GROUP,
        NNODES,
        M,
        N,
        K,
        input_dtype,
        output_dtype=output_dtype,
        use_pdl=use_pdl,
    )

    ag_option.use_cuda_core_local = use_cuda_core_local
    ag_option.use_cuda_core_ag = use_cuda_core_ag

    if tune_agk:
        torch.distributed.barrier()
        torch.cuda.current_stream().synchronize()

        if TP_GROUP.rank() == 0:
            print(
                "[AGK TUNE] start AGKernel profiling "
                f"M={M}, N={N}, K={K}, dtype={input_dtype}, "
                f"transpose_weight={transpose_weight}, "
                f"use_cuda_core_local={use_cuda_core_local}, "
                f"use_cuda_core_ag={use_cuda_core_ag}, "
                f"use_pdl={use_pdl}"
            )

        prof_ctx = flux.ProfilingContext(f"ag_kernel_tune_M{M}_N{N}_K{K}")

        _ = all_gather_gemm_kernel.profiling(
            input,
            w,
            bias=bias,
            output=ag_gemm_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            transpose_weight=transpose_weight,
            all_gather_option=ag_option,
            gathered_input=full_input if gather_input else None,
            prof_ctx=prof_ctx,
        )
        flux.load_tuning_record(prof_ctx.get_latest_record())

        torch.cuda.current_stream().synchronize()
        torch.distributed.barrier()

        if TP_GROUP.rank() == 0:
            print("[AGK TUNE] finish AGKernel profiling")

    warmup_iters = 0 if verify else warmup
    perf_iters = 1 if verify else iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(perf_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(perf_iters)]

    benchmark_barrier()
    for _ in range(warmup_iters):
        gemm_only_output = gemm_only_op.forward(
            full_input,
            w,
            bias=bias,
            output_buf=gemm_only_output,
            input_scale=input_scale if not is_s8_dequant else full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
        )
    benchmark_barrier()
    for i in range(perf_iters):
        start_events[i].record()
        gemm_only_output = gemm_only_op.forward(
            full_input,
            w,
            bias=bias,
            output_buf=gemm_only_output,
            input_scale=input_scale if not is_s8_dequant else full_input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
        )
        end_events[i].record()
    torch.cuda.current_stream().synchronize()

    gemm_times = []
    for i in range(perf_iters):
        end_events[i].synchronize()
        gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_time = sum(gemm_times)

    full_input.zero_()
    time.sleep(1)

    benchmark_barrier()
    ag_option.use_cuda_core_local = use_cuda_core_local
    ag_option.use_cuda_core_ag = use_cuda_core_ag
    for _ in range(warmup_iters):
        all_gather_gemm_kernel.forward(
            input,
            w,
            bias=bias,
            output=ag_gemm_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            gathered_input=full_input if gather_input else None,
            transpose_weight=transpose_weight,
            all_gather_option=ag_option,
        )

    benchmark_barrier()
    for i in range(perf_iters):
        start_events[i].record()
        all_gather_gemm_kernel.forward(
            input,
            w,
            bias=bias,
            output=ag_gemm_output,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=fast_acc,
            gathered_input=full_input if gather_input else None,
            transpose_weight=transpose_weight,
            all_gather_option=ag_option,
        )
        end_events[i].record()

    benchmark_barrier()

    ag_gemm_times = []
    for i in range(perf_iters):
        end_events[i].synchronize()
        ag_gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)

    ag_gemm_time = sum(ag_gemm_times)

    ## signals are already set
    benchmark_barrier()
    for _ in range(warmup_iters):
        if not verify:
            _ = all_gather_gemm_kernel.gemm_only(
                full_input,
                w,
                bias=bias,
                input_scale=full_input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=fast_acc,
                transpose_weight=transpose_weight,
            )

    benchmark_barrier()
    for i in range(perf_iters):
        start_events[i].record()
        if not verify:
            _ = all_gather_gemm_kernel.gemm_only(
                full_input,
                w,
                bias=bias,
                input_scale=full_input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=fast_acc,
                transpose_weight=transpose_weight,
            )
        end_events[i].record()

    benchmark_barrier()

    gemm_only_times = []
    for i in range(perf_iters):
        end_events[i].synchronize()
        gemm_only_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_only_time = sum(gemm_only_times)

    ag_gemm_time_ms = ag_gemm_time / perf_iters * 1000
    gemm_time_ms = gemm_time / perf_iters * 1000
    comm_time_ms = (ag_gemm_time - gemm_time) / perf_iters * 1000
    gemm_only_time_ms = gemm_only_time / perf_iters * 1000

    is_bitwise_match = flux.bitwise_check(gemm_only_output, ag_gemm_output)
    if TP_GROUP.rank() == 0:
        print("is bitwise match: ", is_bitwise_match)

    return PerfResult(
        name=f"{flux_ag_impl_name()}  #{TP_GROUP.rank()}",
        output=ag_gemm_output,
        gathered_output=full_input,
        total_ms=ag_gemm_time_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
        time3="gemm_only",
        gemm_only_time_ms=gemm_only_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--tune-agk",
        default=False,
        action="store_true",
        help="run AGKernel hparams profiling before benchmark",
    )
    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--fastacc",
        default=False,
        action="store_true",
        help="whether to use fast accumulation (FP8 Gemm only)",
    )
    parser.add_argument(
        "--ring_mode",
        default="auto",
        choices=["auto", "all2all", "ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    parser.add_argument(
        "--verify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run once to verify correctness",
    )

    parser.add_argument(
        "--use_cuda_core_local",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl local copy, auto select if not specified",
    )

    parser.add_argument(
        "--use_cuda_core_ag",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl all gather, auto select if not specified",
    )

    parser.add_argument(
        "--use_pdl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use Programmatic Dependent Launch",
    )

    parser.add_argument(
        "--triton",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="run with triton kernels",
    )
    parser.add_argument(
        "--compare_nccl_signal",
        default=False,
        action="store_true",
        help="run native flux and NCCL fused signal flux in the same test process",
    )
    parser.add_argument(
        "--nccl_signal_only", action="store_true",
        help="skip native Flux and test only PyTorch vs NCCL Signal",
    )
    parser.add_argument(
        "--gather_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="gather input",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
    torch.int8: 0,
}

if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()
    if args.nccl_signal_only:
        args.compare_nccl_signal = True

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8

    if args.transpose_weight and (is_fp8 or is_s8_dequant):
        raise ValueError("FP8/S8 GEMM does not support transpose weight (RRR layout).")

    assert args.M % TP_GROUP.size() == 0
    assert args.N % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_M = args.M // TP_GROUP.size()
    local_N = args.N // TP_GROUP.size()

    scale = TP_GROUP.rank() + 1
    if is_s8_dequant:
        data_config = [
            ((local_M, args.K), input_dtype, (127, 0)),  # A
            ((local_N, args.K), input_dtype, (127, 0)),  # B
            None if not args.has_bias else ((1, local_N), torch.bfloat16, (scale, 0)),  # bias
            ((local_M, 1), torch.float32, (1, 0)),  # input_scale
            ((1, local_N), torch.float32, (1, 0)),  # weight_scale
        ]
    elif is_fp8:
        data_config = [
            ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
            ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((1, local_N), torch.bfloat16, (0.1 * scale, 0))
            ),
            ((1, 1), torch.float32, (1, 0)),  # input_scale
            ((1, 1), torch.float32, (1, 0)),  # weight_scale
        ]
    else:
        data_config = [
            ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
            ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((args.M, local_N), input_dtype, (0.1 * scale, 0))
            ),
            None,  # input_scale
            None,  # weight_scale
        ]

    generator = generate_data(data_config)
    input, weight, bias, input_scale, weight_scale = next(generator)

    if args.debug:
        input.zero_()
        input[:, 0].fill_(TP_GROUP.rank() + 1)
        weight.fill_(1)
        if input_scale is not None:
            input_scale.fill_(1)
            weight_scale.fill_(1)
        if bias is not None:
            bias.zero_()
    TP_GROUP.barrier()

    with flux.util.group_profile(
        name="ag_gemm_" + os.environ["TORCHELASTIC_RUN_ID"], do_prof=args.profile, group=TP_GROUP
    ):
        perf_res_torch = perf_torch(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            is_fp8,
            is_s8_dequant,
            args.warmup,
            args.iters,
        )
        perf_res_flux = None
        if args.compare_nccl_signal:
            if not args.nccl_signal_only:
              with ag_nccl_signal_env(enabled=False):
                perf_res_flux = perf_flux(
                    input,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    args.transpose_weight,
                    args.gather_input,
                    RING_MODE_MAP[args.ring_mode],
                    args.warmup,
                    args.iters,
                    args.fastacc,
                    args.verify,
                    args.use_cuda_core_local,
                    args.use_cuda_core_ag,
                    args.use_pdl,
                    args.tune_agk,
                )

            with ag_nccl_signal_env(enabled=True, wait=False):
                perf_res_flux_nccl_fused = perf_flux(
                    input,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    args.transpose_weight,
                    args.gather_input,
                    RING_MODE_MAP[args.ring_mode],
                    args.warmup,
                    args.iters,
                    args.fastacc,
                    args.verify,
                    args.use_cuda_core_local,
                    args.use_cuda_core_ag,
                    args.use_pdl,
                    False,
                )
        else:
            perf_res_flux = perf_flux(
                input,
                weight,
                bias,
                input_scale,
                weight_scale,
                args.transpose_weight,
                args.gather_input,
                RING_MODE_MAP[args.ring_mode],
                args.warmup,
                args.iters,
                args.fastacc,
                args.verify,
                args.use_cuda_core_local,
                args.use_cuda_core_ag,
                args.use_pdl,
                args.tune_agk,
            )
            perf_res_flux_nccl_fused = None

        perf_res_flux_no_overlap = None if args.nccl_signal_only else perf_flux_no_overlap(
            input,
            weight,
            bias,
            input_scale,
            weight_scale,
            args.transpose_weight,
            args.gather_input,  # not used,
            args.warmup,
            args.iters,
            args.fastacc,
        )

        if args.triton:
            perf_res_triton = perf_triton(
                input,
                weight,
                bias,
                input_scale,
                weight_scale,
                args.transpose_weight,
                args.gather_input,
                args.warmup,
                args.iters,
                args.fastacc,
                args.verify,
            )

    if TP_GROUP.rank() == 0:
        flux.testing.print_gemm_sol_time(local_M, args.N, args.K, input_dtype)

    if should_log_to_rds():
        set_global_args("ag_gemm", args)
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            log_perf(perf_res_torch)
            if perf_res_flux is not None:
                log_perf(perf_res_flux)
            if perf_res_flux_nccl_fused is not None:
                log_perf(perf_res_flux_nccl_fused)
            if perf_res_flux_no_overlap is not None:
                log_perf(perf_res_flux_no_overlap)
            if args.triton:
                log_perf(perf_res_triton)
        torch.distributed.barrier()

    torch_output = perf_res_torch.output
    torch_gathered_data = perf_res_torch.gathered_output
    atol = THRESHOLD_MAP[input_dtype]
    rtol = THRESHOLD_MAP[input_dtype]

    flux_results = [] if perf_res_flux is None else [perf_res_flux]
    if perf_res_flux_nccl_fused is not None:
        flux_results.append(perf_res_flux_nccl_fused)

    for perf_res in flux_results:
        flux_output = perf_res.output
        flux_gathered_data = perf_res.gathered_output
        torch.distributed.barrier()
        if flux.bitwise_check(torch_output, flux_output):
            print(f"✅  torch vs {perf_res.name.split()[0]} bitwise match")
        else:
            print(f"❌  torch vs {perf_res.name.split()[0]} not bitwise match")

        if args.gather_input:
            try:
                flux.torch_allclose(flux_gathered_data, torch_gathered_data, atol=1e-9, rtol=1e-9)
            except Exception as e:
                torch.save(
                    flux_gathered_data,
                    f"{perf_res.name.split()[0]}_gathered_data_{TP_GROUP.rank()}.pt",
                )
                torch.save(torch_gathered_data, f"torch_gathered_data_{TP_GROUP.rank()}.pt")
                print(f"❌ {perf_res.name.split()[0]} gathered data check failed")
                raise e
            else:
                print(f"✅ {perf_res.name.split()[0]} gathered data check passed")
        try:
            flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            torch.save(flux_output, f"{perf_res.name.split()[0]}_{TP_GROUP.rank()}.pt")
            torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
            print(f"❌ {perf_res.name.split()[0]} check failed")
            raise e
        else:
            print(f"✅ {perf_res.name.split()[0]} check passed")

    flux_output = perf_res_flux.output if perf_res_flux is not None else None
    flux_gathered_data = perf_res_flux.gathered_output if perf_res_flux is not None else None

    if args.triton:
        triton_output = perf_res_triton.output
        is_bitwise_match = flux.bitwise_check(torch_output, triton_output)
        print("torch vs triton bitwise match: ", is_bitwise_match)
        if args.gather_input:
            triton_gathered_data = perf_res_triton.gathered_output
            try:
                flux.torch_allclose(triton_gathered_data, flux_gathered_data, atol=1e-9, rtol=1e-9)
            except Exception as e:
                torch.save(triton_gathered_data, f"triton_gathered_data_{TP_GROUP.rank()}.pt")
                torch.save(torch_gathered_data, f"torch_gathered_data_{TP_GROUP.rank()}.pt")
                print("❌ triton gathered data check failed")
                raise e
            else:
                print("✅ triton gathered data check passed")
        try:
            flux.torch_allclose(triton_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            torch.save(triton_output, f"triton_{TP_GROUP.rank()}.pt")
            torch.save(torch_output, f"torch_{TP_GROUP.rank()}.pt")
            print("❌ triton check failed")
            raise e
        else:
            print("✅ triton check passed")

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()
