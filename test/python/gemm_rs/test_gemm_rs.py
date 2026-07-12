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

# usage: torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_id=none --master_addr=127.0.0.1 --master_port=23456 test/python/gemm_rs/test_gemm_rs.py 2048 10240 40960
import argparse
import os
import time
from functools import partial
from typing import Optional

import torch
import torch.distributed

import flux
from flux.cpp_mod import ReduceScatterOption
import flux.testing
from flux.testing import DTYPE_MAP, generate_data, initialize_distributed, matmul_int8
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds

try:
    from flux.triton.gemm_rs import GemmRSTritonPCIe as GemmRSTriton
except Exception as e:
    pass

print = partial(print, flush=True)


def benchmark_barrier() -> None:
    torch.cuda.synchronize()
    try:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    except TypeError:
        torch.distributed.barrier()
    torch.cuda.synchronize()


def _option_enabled(option: flux.ReduceScatterOption, name: str) -> bool:
    return getattr(option, name, None) is True


def _ring_mode_name(reduce_scatter_option: flux.ReduceScatterOption) -> Optional[str]:
    ring_mode = getattr(reduce_scatter_option, "ring_mode", None)
    if ring_mode is None:
        return None
    if ring_mode == getattr(flux.RingMode, "Ring1D", object()):
        return "ring1d"
    if ring_mode == getattr(flux.RingMode, "Ring2D", object()):
        return "ring2d"
    if ring_mode == getattr(flux.RingMode, "All2All", object()):
        return "all2all"
    return str(ring_mode).split(".")[-1].lower()


def flux_rs_impl_name(
    reduce_scatter_option: flux.ReduceScatterOption, fuse_reduction: bool, ring_reduction: bool
) -> str:
    tags = []
    if fuse_reduction:
        tags.append("fuse")
    if ring_reduction:
        tags.append("ring_reduce")

    ring_mode = _ring_mode_name(reduce_scatter_option)
    if ring_mode is not None:
        tags.append(ring_mode)
    if _option_enabled(reduce_scatter_option, "use_1d_ring"):
        tags.append("use_1d")
    if _option_enabled(reduce_scatter_option, "use_p2p_read"):
        tags.append("p2p_read")
    if _option_enabled(reduce_scatter_option, "use_cudaMemcpyAsync"):
        tags.append("cudaMemcpyAsync")
    if _option_enabled(reduce_scatter_option, "use_gemmk"):
        tags.append("gemmk")
    if _option_enabled(reduce_scatter_option, "per_tile_flags"):
        tags.append("per_tile_flags")

    return "flux_rs" if not tags else "flux_rs_" + "_".join(tags)


class PerfResult:
    def __init__(
        self, name: str, output: torch.Tensor, gemm_time_ms: float, comm_time_ms: float
    ) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.total_ms = self.gemm_time_ms + self.comm_time_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: gemm {self.gemm_time_ms:.3f} ms, comm {self.comm_time_ms:.3f} ms"
            f", total {self.total_ms:.3f} ms"
        )


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    transpose_weight: bool = False,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
):
    benchmark_barrier()

    is_fp8 = flux.util.is_fp8_dtype(input.dtype)
    is_s8_dequant = input.dtype == torch.int8
    warmup_iters = warmup
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input.dtype
    m = input.size(0)
    with flux.util.with_torch_deterministic(False):
        if transpose_weight:
            w = weight.t().contiguous()
            n = w.size(1)
        else:
            n = weight.size(0)
            w = weight

        full_output = torch.zeros(
            [m, n],
            dtype=output_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        output = torch.zeros(
            [m // WORLD_SIZE, n],
            dtype=output_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

    op = (
        flux.GemmOnly(
            input_dtype=input.dtype,
            weight_dtype=input.dtype,
            output_dtype=output_dtype,
            transpose_weight=transpose_weight,
            use_fp8_gemm=is_fp8,
        )
        if is_fp8
        else None
    )

    for _ in range(warmup_iters):
        if is_fp8:
            full_output = op.forward(
                input,
                w,
                bias=bias,
                output_buf=full_output,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=False,
            )
        elif is_s8_dequant:
            accum = matmul_int8(input, weight.t()).to(torch.float32)
            full_output = input_scale * weight_scale * accum
            full_output = full_output.to(output_dtype)
        else:
            full_output = torch.matmul(input, weight.t())
        if bias is not None and not is_fp8:
            if not is_s8_dequant or (is_s8_dequant and TP_GROUP.rank() == 0):
                full_output += bias
        torch.distributed.reduce_scatter_tensor(output, full_output, group=TP_GROUP)

    benchmark_barrier()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        if is_fp8:
            full_output = op.forward(
                input,
                w,
                bias=bias,
                output_buf=full_output,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=None,
                fast_accum=False,
            )
        elif is_s8_dequant:
            accum = matmul_int8(input, weight.t()).to(torch.float32)
            full_output = input_scale * weight_scale * accum
            full_output = full_output.to(output_dtype)
        else:
            full_output = torch.matmul(input, weight.t())
        if bias is not None and not is_fp8:
            # only apply bias on rank 0 for s8 gemm
            if not is_s8_dequant or (is_s8_dequant and TP_GROUP.rank() == 0):
                full_output += bias
        gemm_end_events[i].record()
        torch.distributed.reduce_scatter_tensor(output, full_output, group=TP_GROUP)
        end_events[i].record()

    gemm_times = []
    comm_times = []
    for i in range(iters):
        gemm_end_events[i].synchronize()
        end_events[i].synchronize()
        gemm_times.append(start_events[i].elapsed_time(gemm_end_events[i]) / 1000)
        comm_times.append(gemm_end_events[i].elapsed_time(end_events[i]) / 1000)
    # print(gemm_times)
    # print(comm_times)
    gemm_time = sum(gemm_times) / iters * 1000
    comm_time = sum(comm_times) / iters * 1000
    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        output=output,
        gemm_time_ms=gemm_time,
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    transpose_weight: bool,
    fuse_reduction: bool,
    ring_reduction: bool,
    warmup: int,
    iters: int,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    reduce_scatter_option: flux.ReduceScatterOption = flux.ReduceScatterOption(),
    tune_gemm_rs: bool = False,
):
    is_fp8 = flux.util.is_fp8_dtype(input.dtype)
    is_s8_dequant = input.dtype == torch.int8
    M = input.size(0)
    # todo: transpose here to avoid TN kernel, which has the worst performence
    if transpose_weight:
        with flux.util.with_torch_deterministic(False):
            w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input.dtype
    gemm_only_op = flux.GemmOnly(
        w.dtype,
        w.dtype,
        output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=is_fp8,
    )
    gemm_rs_op = flux.GemmRS(
        TP_GROUP,
        NNODES,
        (M + 1024 - 1) // 1024 * 1024,
        N,
        input.dtype,
        output_dtype,
        transpose_weight=transpose_weight,
        fuse_reduction=fuse_reduction,
        ring_reduction=ring_reduction,
    )

    if tune_gemm_rs:
        torch.distributed.barrier()
        torch.cuda.current_stream().synchronize()

        if TP_GROUP.rank() == 0:
            print(
                "[GEMM-RS TUNE] start GemmRS profiling "
                f"M={M}, N={N}, K={input.size(1) * TP_GROUP.size()}, "
                f"local_K={input.size(1)}, dtype={input.dtype}, "
                f"transpose_weight={transpose_weight}, "
                f"fuse_reduction={fuse_reduction}, "
                f"ring_reduction={ring_reduction}, "
                f"reduce_scatter_option={reduce_scatter_option}"
            )

        prof_ctx = flux.ProfilingContext(
            f"gemm_rs_tune_M{M}_N{N}_K{input.size(1) * TP_GROUP.size()}"
        )
        _ = gemm_rs_op.profiling(
            input,
            w,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
            prof_ctx=prof_ctx,
            reduce_scatter_option=reduce_scatter_option,
        )
        flux.load_tuning_record(prof_ctx.get_latest_record())

        torch.cuda.current_stream().synchronize()
        torch.distributed.barrier()

        if TP_GROUP.rank() == 0:
            print("[GEMM-RS TUNE] finish GemmRS profiling")

    warmup_iters = warmup
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with flux.util.with_torch_deterministic(False):
        gemm_only_output_buf = torch.empty(
            [M, N], dtype=output_dtype, device=input.device, requires_grad=False
        )

    benchmark_barrier()
    for _ in range(warmup_iters):
        _ = gemm_only_op.forward(
            input,
            w,
            bias=bias,
            output_buf=gemm_only_output_buf,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
        )

    benchmark_barrier()
    for i in range(iters):
        start_events[i].record()
        _ = gemm_only_op.forward(
            input,
            w,
            bias=bias,
            output_buf=gemm_only_output_buf,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
        )
        end_events[i].record()
    torch.cuda.current_stream().synchronize()

    gemm_times = []
    for i in range(iters):
        end_events[i].synchronize()
        gemm_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_time = sum(gemm_times)

    time.sleep(1)

    benchmark_barrier()
    for _ in range(warmup_iters):
        output = gemm_rs_op.forward(
            input,
            w,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
            reduce_scatter_option=reduce_scatter_option,
        )

    benchmark_barrier()
    for i in range(iters):
        start_events[i].record()
        output = gemm_rs_op.forward(
            input,
            w,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
            reduce_scatter_option=reduce_scatter_option,
        )
        end_events[i].record()
    torch.cuda.current_stream().synchronize()

    gemm_rs_times = []
    for i in range(iters):
        end_events[i].synchronize()
        gemm_rs_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    gemm_rs_time = sum(gemm_rs_times)

    gemm_time_ms = gemm_time / iters * 1000
    comm_time_ms = (gemm_rs_time - gemm_time) / iters * 1000

    return PerfResult(
        name=f"{flux_rs_impl_name(reduce_scatter_option, fuse_reduction, ring_reduction)} #{TP_GROUP.rank()}",
        output=output,
        gemm_time_ms=gemm_time_ms,
        comm_time_ms=comm_time_ms,
    )


@torch.no_grad()
def perf_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    transpose_weight: bool,
    warmup: int,
    iters: int,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
):
    is_fp8 = flux.util.is_fp8_dtype(input.dtype)
    is_s8_dequant = input.dtype == torch.int8
    M = input.size(0)
    # todo: transpose here to avoid TN kernel, which has the worst performence
    if transpose_weight:
        with flux.util.with_torch_deterministic(False):
            w = weight.t().contiguous()
        N = w.size(1)
    else:
        w = weight
        N = w.size(0)

    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else input.dtype
    gemm_only_op = flux.GemmOnly(
        w.dtype,
        w.dtype,
        output_dtype,
        transpose_weight=transpose_weight,
        use_fp8_gemm=is_fp8,
    )

    gemm_rs_op = GemmRSTriton(
        TP_GROUP,
        input.dtype,
        output_dtype,
        (M + 1024 - 1) // 1024 * 1024,
        N,
    )

    warmup_iters = warmup
    start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
    end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
    with flux.util.with_torch_deterministic(False):
        gemm_only_output_buf = torch.empty(
            [M, N], dtype=output_dtype, device=input.device, requires_grad=False
        )

    benchmark_barrier()
    for _ in range(warmup_iters):
        _ = gemm_only_op.forward(
            input,
            w,
            bias=bias,
            output_buf=gemm_only_output_buf,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
        )

    benchmark_barrier()
    start_event.record()
    for _ in range(iters):
        _ = gemm_only_op.forward(
            input,
            w,
            bias=bias,
            output_buf=gemm_only_output_buf,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            fast_accum=False,
        )
    end_event.record()
    torch.cuda.current_stream().synchronize()
    gemm_time = start_event.elapsed_time(end_event) / 1000

    time.sleep(1)

    benchmark_barrier()
    for _ in range(warmup_iters):
        output = gemm_rs_op.forward(
            input,
            w,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            transpose_weight=transpose_weight,
            fast_accum=False,
        )

    benchmark_barrier()
    start_event.record()
    for _ in range(iters):
        output = gemm_rs_op.forward(
            input,
            w,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            transpose_weight=transpose_weight,
            fast_accum=False,
        )
    end_event.record()
    torch.cuda.current_stream().synchronize()

    gemm_rs_time = start_event.elapsed_time(end_event) / 1000

    gemm_time_ms = gemm_time / iters * 1000
    comm_time_ms = (gemm_rs_time - gemm_time) / iters * 1000

    return PerfResult(
        name=f"triton #{TP_GROUP.rank()}",
        output=output,
        gemm_time_ms=gemm_time_ms,
        comm_time_ms=comm_time_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument(
        "--tune-gemm-rs",
        default=False,
        action="store_true",
        help="run GemmRS hparams profiling before benchmark",
    )
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )
    parser.add_argument(
        "--fuse_reduction", default=False, action="store_true", help="fuse reduction to gemm"
    )
    parser.add_argument(
        "--ring_reduction",
        default=False,
        action="store_true",
        help="reduce paritial output with ring order",
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument(
        "--debug", action="store_true", help="debug mode. use human read input", default=False
    )
    parser.add_argument(
        "--triton",
        action="store_true",
        help="also perf triton version",
        default=False,
    )
    parser.add_argument(
        "--use_1d_ring",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    parser.add_argument(
        "--use_p2p_read",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    parser.add_argument(
        "--use_cudaMemcpyAsync",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    parser.add_argument(
        "--use_gemmk",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    parser.add_argument(
        "--per_tile_flags",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    parser.add_argument(
        "--reduce_scatter_blocks",
        type=int,
        help="number of blocks for reduce scatter",
    )
    parser.add_argument(
        "--ring_mode",
        choices=["ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    return parser.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()

    input_dtype = DTYPE_MAP[args.dtype]
    is_fp8 = flux.util.is_fp8_dtype(input_dtype)
    is_s8_dequant = input_dtype == torch.int8

    if args.transpose_weight and (is_fp8 or is_s8_dequant):
        raise ValueError("FP8/S8 GEMM does not support RRR layout")

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    # input: [M, K], weight: [N, K]

    scale = TP_GROUP.rank() + 1
    if is_s8_dequant:
        data_config = [
            ((args.M, local_K), input_dtype, (127, 0)),  # A
            ((args.N, local_K), input_dtype, (127, 0)),  # B
            None if not args.has_bias else ((1, args.N), torch.bfloat16, (24, -12)),  # bias
            ((args.M, 1), torch.float32, (1 / 1024.0, 0)),  # input_scale
            ((1, args.N), torch.float32, (1 / (args.K * 4 / 1024.0), 0)),  # weight_scale
        ]
    elif is_fp8:
        data_config = [
            ((args.M, local_K), input_dtype, (0.01 * scale, 0)),  # A
            ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
            None,  # bias. not supported now. ((1, args.N), torch.bfloat16, (0.1 * scale, 0))
            ((1), torch.float32, (1, 0)),  # input_scale
            ((1), torch.float32, (1, 0)),  # weight_scale
        ]
    else:
        data_config = [
            ((args.M, local_K), input_dtype, (0.01 * scale, 0)),  # A
            ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((args.M, args.N), input_dtype, (0.1 * scale, 0))
            ),
            None,  # input_scale
            None,  # weight_scale
        ]

    assert not (args.has_bias and is_fp8), "FP8 does not support bias"
    generator = generate_data(data_config)
    input, weight, bias, input_scale, weight_scale = next(generator)

    if args.debug:
        input.zero_()
        input[:, 0].fill_(TP_GROUP.rank() + 1)
        weight.fill_(1)
        if input_scale is not None:
            input_scale.fill_(1)
        if weight_scale is not None:
            weight_scale.fill_(1)
        if bias is not None:
            bias.fill_(TP_GROUP.rank() + 1)

    if args.triton:
        if is_fp8 and flux.util.get_arch() < 90:
            print(f"warning: triton L20 does not support FP8 now. skip triton...")
            args.triton = False
        if NNODES > 1:
            print(f"warning: triton does not support multi-node now. skip multi-node...")
            args.triton = False
        if args.fuse_reduction:
            print(f"warning: triton does not support fuse_reduction now. skip fuse_reduction...")

    reduce_scatter_option = ReduceScatterOption()
    reduce_scatter_option.use_1d_ring = args.use_1d_ring
    reduce_scatter_option.use_p2p_read = args.use_p2p_read
    reduce_scatter_option.use_cudaMemcpyAsync = args.use_cudaMemcpyAsync
    reduce_scatter_option.use_gemmk = args.use_gemmk
    reduce_scatter_option.per_tile_flags = args.per_tile_flags
    reduce_scatter_option.num_blocks = args.reduce_scatter_blocks
    reduce_scatter_option.ring_mode = {
        "ring1d": flux.RingMode.Ring1D,
        "ring2d": flux.RingMode.Ring2D,
    }.get(args.ring_mode, None)
    with flux.util.group_profile(
        name="gemm_rs_" + os.environ["TORCHELASTIC_RUN_ID"], do_prof=args.profile, group=TP_GROUP
    ):
        perf_res_flux = perf_flux(
            input,
            weight,
            bias,
            args.transpose_weight,
            args.fuse_reduction,
            args.ring_reduction,
            args.warmup,
            args.iters,
            input_scale,
            weight_scale,
            reduce_scatter_option=reduce_scatter_option,
            tune_gemm_rs=args.tune_gemm_rs,
        )
        perf_res_torch = perf_torch(
            input,
            weight,
            bias,
            args.warmup,
            args.iters,
            args.transpose_weight,
            input_scale,
            weight_scale,
        )

        if args.triton:
            perf_res_triton = perf_triton(
                input,
                weight,
                bias,
                args.transpose_weight,
                args.warmup,
                args.iters,
                input_scale,
                weight_scale,
            )

    if TP_GROUP.rank() == 0:
        flux.testing.print_gemm_sol_time(args.M, args.N, local_K, input_dtype)
    if should_log_to_rds():
        set_global_args("gemm_rs", args)
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            log_perf(perf_res_torch)
        TP_GROUP.barrier()
    for i in range(TP_GROUP.size()):
        if i == TP_GROUP.rank():
            log_perf(perf_res_flux)
        TP_GROUP.barrier()
    if args.triton:
        for i in range(TP_GROUP.size()):
            if i == TP_GROUP.rank():
                log_perf(perf_res_triton)
            TP_GROUP.barrier()

    TP_GROUP.barrier()

    flux_output = perf_res_flux.output
    torch_output = perf_res_torch.output
    THRESHOLD_MAP = {
        torch.float16: 1e-2,
        torch.bfloat16: 2e-2,
        torch.float8_e4m3fn: 3e-2,
        torch.float8_e5m2: 3e-2,
        torch.int8: 2e-1,
    }

    flux_output = flux_output.reshape(torch_output.size())
    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
    try:
        flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
    except Exception as e:
        torch.save(flux_output, f"flux_output_{RANK}.pt")
        torch.save(torch_output, f"torch_output_{RANK}.pt")
        print("❌ flux check failed")
        raise e
    else:
        print("✅ flux check passed")
    TP_GROUP.barrier()

    if TP_GROUP.rank() == 0:
        if flux.bitwise_check(torch_output, flux_output):
            print("✅ flux vs torch bitwise check passed")
        else:
            print("❌ flux vs torch bitwise check failed")
    if args.triton:
        triton_output = perf_res_triton.output
        try:
            flux.torch_allclose(torch_output, triton_output, atol=atol, rtol=rtol)
            print("✅ triton check passed")
        except Exception as e:
            print("❌ triton check failed")
            torch.save(triton_output, f"triton_output_{RANK}.pt")
            torch.save(torch_output, f"torch_output_{RANK}.pt")
            raise e

    TP_GROUP.barrier()
    torch.cuda.synchronize()
