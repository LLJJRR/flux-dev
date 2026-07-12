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

import torch
import torch.distributed as dist

import flux

print = partial(print, flush=True)


def _rank_pattern(rank: int, world_size: int, recv_shape: tuple[int, ...]) -> torch.Tensor:
    chunks = []
    for dst_rank in range(world_size):
        value = rank + dst_rank + 1
        chunks.append(torch.full(recv_shape, value, device="cuda", dtype=torch.int8))
    return torch.cat(chunks, dim=0)


def _expected(rank: int, world_size: int, recv_shape: tuple[int, ...]) -> torch.Tensor:
    value = sum(src_rank + rank + 1 for src_rank in range(world_size))
    return torch.full(recv_shape, value, device="cuda", dtype=torch.int8)


def _run_case(
    name: str,
    process_group: dist.ProcessGroup,
    input: torch.Tensor,
    output: torch.Tensor,
    emit_signal: bool,
):
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    if rank == 0:
        print(f"[NCCL-SIGNAL-RS] start {name}, emit_signal={emit_signal}")

    output.zero_()
    flux.test_nccl_signal_reduce_scatter(process_group, input, output, emit_signal)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, _expected(rank, world_size, tuple(output.shape)), rtol=0, atol=0)

    if rank == 0:
        print(f"[NCCL-SIGNAL-RS] pass {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument(
        "--mode",
        default="both",
        choices=("standard", "signal", "both"),
        help="standard uses ncclReduceScatter; signal uses ncclReduceScatterFluxSignal",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
    process_group = dist.group.WORLD

    recv_shape = (args.m, args.n)
    input = _rank_pattern(rank, world_size, recv_shape)
    output = torch.empty(recv_shape, device="cuda", dtype=torch.int8)

    if args.mode in ("standard", "both"):
        _run_case("standard", process_group, input, output, emit_signal=False)
    if args.mode in ("signal", "both"):
        _run_case("signal", process_group, input, output, emit_signal=True)

    dist.barrier()
    if rank == 0:
        print("[NCCL-SIGNAL-RS] all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
