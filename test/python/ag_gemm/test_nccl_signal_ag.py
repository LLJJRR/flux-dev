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

import torch
import torch.distributed as dist

import flux
from flux.testing import DTYPE_MAP


def _rank_pattern(rank: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    data = torch.arange(numel, device="cuda", dtype=torch.float32).reshape(shape)
    data = data * 0.001 + rank
    return data.to(dtype)


def _check_output(output: torch.Tensor, input_shape: tuple[int, ...], dtype: torch.dtype, world_size: int):
    for rank in range(world_size):
        expected = _rank_pattern(rank, input_shape, dtype)
        actual = output.narrow(0, rank * input_shape[0], input_shape[0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _run_case(name: str, process_group: dist.ProcessGroup, input: torch.Tensor, emit_signal: bool):
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    output = torch.empty(
        (world_size * input.shape[0], *input.shape[1:]),
        device=input.device,
        dtype=input.dtype,
    )

    if rank == 0:
        print(f"[NCCL-SIGNAL-AG] start {name}, emit_signal={emit_signal}")

    flux.test_nccl_signal_all_gather(process_group, input, output, emit_signal)
    torch.cuda.synchronize()
    _check_output(output, tuple(input.shape), input.dtype, world_size)

    if rank == 0:
        print(f"[NCCL-SIGNAL-AG] pass {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=12288)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_MAP.keys()))
    parser.add_argument(
        "--mode",
        default="both",
        choices=("standard", "signal", "both"),
        help="standard uses ncclAllGather; signal uses ncclAllGatherFluxSignal",
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

    dtype = DTYPE_MAP[args.dtype]
    input = _rank_pattern(rank, (args.m, args.k), dtype)

    if args.mode in ("standard", "both"):
        _run_case("standard", process_group, input, emit_signal=False)
    if args.mode in ("signal", "both"):
        _run_case("signal", process_group, input, emit_signal=True)

    dist.barrier()
    if rank == 0:
        print("[NCCL-SIGNAL-AG] all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
