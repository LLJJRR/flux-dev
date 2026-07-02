#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$ROOT/patches/nccl_flux_signal.patch"
NCCL_DIR="$ROOT/3rdparty/nccl"

if [[ ! -f "$PATCH" ]]; then
  echo "missing patch: $PATCH" >&2
  exit 1
fi

if [[ ! -d "$NCCL_DIR/.git" && ! -f "$NCCL_DIR/.git" ]]; then
  echo "missing nccl submodule: $NCCL_DIR" >&2
  exit 1
fi

git -C "$NCCL_DIR" apply --whitespace=nowarn "$PATCH"
