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
import datetime
import os
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

import flux
import flux.testing
from flux.cpp_mod import ReduceScatterOption
from flux.testing import DTYPE_MAP, generate_data, initialize_distributed
from flux.testing.perf_db_helper import log_perf

print = partial(print, flush=True)


class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_ms: float, comm_ms: float):
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_ms
        self.comm_time_ms = comm_ms
        self.total_ms = gemm_ms + comm_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: gemm {self.gemm_time_ms:.3f} ms, "
            f"comm {self.comm_time_ms:.3f} ms, total {self.total_ms:.3f} ms"
        )


def benchmark_barrier() -> None:
    torch.cuda.synchronize()
    try:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    except TypeError:
        torch.distributed.barrier()
    torch.cuda.synchronize()


def _primitive_rank_pattern(rank: int, world_size: int, recv_shape: tuple[int, ...]) -> torch.Tensor:
    chunks = []
    for dst_rank in range(world_size):
        value = rank + dst_rank + 1
        chunks.append(torch.full(recv_shape, value, device="cuda", dtype=torch.float32))
    return torch.cat(chunks, dim=0)


def _primitive_expected(rank: int, world_size: int, recv_shape: tuple[int, ...]) -> torch.Tensor:
    value = sum(src_rank + rank + 1 for src_rank in range(world_size))
    return torch.full(recv_shape, value, device="cuda", dtype=torch.float32)


def run_primitive_case(process_group: dist.ProcessGroup, m: int, n: int) -> None:
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    recv_shape = (m, n)
    input = _primitive_rank_pattern(rank, world_size, recv_shape)
    output = torch.empty(recv_shape, device="cuda", dtype=torch.float32)
    nccl_rs = flux.NcclSignalReduceScatter(process_group)

    for name, emit_signal in (("standard", False), ("signal", True)):
        if rank == 0:
            print(f"[NCCL-SIGNAL-RS] start primitive {name}, emit_signal={emit_signal}")
        output.zero_()
        nccl_rs.run(input, output, emit_signal)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, _primitive_expected(rank, world_size, recv_shape), rtol=0, atol=0)
        if rank == 0:
            print(f"[NCCL-SIGNAL-RS] pass primitive {name}")


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    group: dist.ProcessGroup,
) -> PerfResult:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    m = input.size(0)
    n = weight.size(0)
    full_output = torch.empty((m, n), dtype=input.dtype, device=input.device)
    output = torch.empty((m // world_size, n), dtype=input.dtype, device=input.device)

    benchmark_barrier()
    for _ in range(warmup):
        full_output = torch.matmul(input, weight.t())
        if bias is not None:
            full_output += bias
        torch.distributed.reduce_scatter_tensor(output, full_output, group=group)

    benchmark_barrier()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    gemm_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        full_output = torch.matmul(input, weight.t())
        if bias is not None:
            full_output += bias
        gemm_events[i].record()
        torch.distributed.reduce_scatter_tensor(output, full_output, group=group)
        end_events[i].record()

    gemm_ms = 0.0
    comm_ms = 0.0
    for i in range(iters):
        end_events[i].synchronize()
        gemm_ms += start_events[i].elapsed_time(gemm_events[i])
        comm_ms += gemm_events[i].elapsed_time(end_events[i])
    return PerfResult(f"torch #{rank}", output, gemm_ms / iters, comm_ms / iters)


@torch.no_grad()
def perf_flux_rs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    group: dist.ProcessGroup,
    nnodes: int,
    tune: bool,
) -> PerfResult:
    rank = dist.get_rank(group)
    op = flux.GemmRS(
        group,
        nnodes,
        (input.size(0) + 1023) // 1024 * 1024,
        weight.size(0),
        input.dtype,
        input.dtype,
        False,
        False,
        False,
    )
    option = ReduceScatterOption()

    if tune:
        if rank == 0:
            print("[NCCL-SIGNAL-RS] start original Flux GemmRS tuning")
        _ = op.profiling(input, weight, bias=bias, reduce_scatter_option=option)
        benchmark_barrier()
        if rank == 0:
            print("[NCCL-SIGNAL-RS] finish original Flux GemmRS tuning")

    gemm_only = flux.GemmOnly(input.dtype, input.dtype, input.dtype, False, False)
    gemm_buf = torch.empty((input.size(0), weight.size(0)), dtype=input.dtype, device=input.device)

    benchmark_barrier()
    for _ in range(warmup):
        _ = gemm_only.forward(input, weight, bias=bias, output_buf=gemm_buf)
    benchmark_barrier()

    gemm_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    gemm_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        gemm_start[i].record()
        _ = gemm_only.forward(input, weight, bias=bias, output_buf=gemm_buf)
        gemm_end[i].record()
    torch.cuda.synchronize()
    gemm_ms = sum(gemm_start[i].elapsed_time(gemm_end[i]) for i in range(iters)) / iters

    benchmark_barrier()
    for _ in range(warmup):
        output = op.forward(input, weight, bias=bias, reduce_scatter_option=option)
    benchmark_barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        output = op.forward(input, weight, bias=bias, reduce_scatter_option=option)
        end_events[i].record()
    torch.cuda.synchronize()
    total_ms = sum(start_events[i].elapsed_time(end_events[i]) for i in range(iters)) / iters
    return PerfResult(f"flux_rs #{rank}", output, gemm_ms, total_ms - gemm_ms)


@torch.no_grad()
def perf_flux_nccl_signal_rs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    group: dist.ProcessGroup,
) -> PerfResult:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    full_output = torch.empty((input.size(0), weight.size(0)), dtype=input.dtype, device=input.device)
    output = torch.empty((input.size(0) // world_size, weight.size(0)), dtype=input.dtype, device=input.device)
    gemm_only = flux.GemmOnly(input.dtype, input.dtype, input.dtype, False, False)
    nccl_rs = flux.NcclSignalReduceScatter(group)

    benchmark_barrier()
    for _ in range(warmup):
        full_output = gemm_only.forward(input, weight, bias=bias, output_buf=full_output)
        nccl_rs.run(full_output, output, True)

    benchmark_barrier()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    gemm_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        full_output = gemm_only.forward(input, weight, bias=bias, output_buf=full_output)
        gemm_events[i].record()
        nccl_rs.run(full_output, output, True)
        end_events[i].record()

    gemm_ms = 0.0
    comm_ms = 0.0
    for i in range(iters):
        end_events[i].synchronize()
        gemm_ms += start_events[i].elapsed_time(gemm_events[i])
        comm_ms += gemm_events[i].elapsed_time(end_events[i])
    return PerfResult(f"flux_nccl_signal_rs #{rank}", output, gemm_ms / iters, comm_ms / iters)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int, nargs="?", default=2048)
    parser.add_argument("N", type=int, nargs="?", default=49152)
    parser.add_argument("K", type=int, nargs="?", default=12288)
    parser.add_argument("--m", type=int, help="primitive-only receive M")
    parser.add_argument("--n", type=int, help="primitive-only receive N")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--has_bias", action="store_true")
    parser.add_argument("--tune-gemm-rs", action="store_true")
    parser.add_argument("--primitive-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.primitive_only:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=1800),
        )
        run_primitive_case(dist.group.WORLD, args.m or args.M, args.n or args.N)
        dist.barrier()
        dist.destroy_process_group()
        return

    tp_group = initialize_distributed()
    rank = tp_group.rank()
    world_size = tp_group.size()
    nnodes = flux.testing.NNODES()
    assert args.M % world_size == 0
    assert args.K % world_size == 0

    dtype = DTYPE_MAP[args.dtype]
    local_k = args.K // world_size
    scale = rank + 1
    data_config = [
        ((args.M, local_k), dtype, (0.01 * scale, 0)),
        ((args.N, local_k), dtype, (0.01 * scale, 0)),
        None if not args.has_bias else ((args.M, args.N), dtype, (0.1 * scale, 0)),
    ]
    input, weight, bias = next(generate_data(data_config))

    perf_torch_res = perf_torch(input, weight, bias, args.warmup, args.iters, tp_group)
    perf_flux_res = perf_flux_rs(
        input, weight, bias, args.warmup, args.iters, tp_group, nnodes, args.tune_gemm_rs
    )
    perf_signal_res = perf_flux_nccl_signal_rs(input, weight, bias, args.warmup, args.iters, tp_group)

    if rank == 0:
        flux.testing.print_gemm_sol_time(args.M, args.N, local_k, dtype)

    for result in (perf_torch_res, perf_flux_res, perf_signal_res):
        for i in range(world_size):
            if i == rank:
                log_perf(result)
            tp_group.barrier()

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = atol
    torch_output = perf_torch_res.output
    flux_output = perf_flux_res.output.reshape(torch_output.shape)
    signal_output = perf_signal_res.output.reshape(torch_output.shape)
    torch.testing.assert_close(flux_output, torch_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(signal_output, torch_output, atol=atol, rtol=rtol)

    print("✅ flux_rs check passed")
    print("✅ flux_nccl_signal_rs check passed")
    tp_group.barrier()
    if rank == 0:
        print("[NCCL-SIGNAL-RS] all checks passed")


if __name__ == "__main__":
    main()
