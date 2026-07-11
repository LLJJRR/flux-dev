#!/usr/bin/env bash
set -euo pipefail

FLUX_DIR=${FLUX_DIR:-/root/flux}
FLUX_ENV=${FLUX_ENV:-/root/flux_env.sh}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-127.0.0.1:23456}
LOG_DIR=${LOG_DIR:-${FLUX_DIR}/test/python/ag_gemm/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/diagnose_nccl_signal_env.log}

mkdir -p "${LOG_DIR}"

{
  echo "========== basic =========="
  date
  hostname
  echo "FLUX_DIR=${FLUX_DIR}"
  echo "FLUX_ENV=${FLUX_ENV}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"

  if [[ -f "${FLUX_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${FLUX_ENV}"
  else
    echo "ERROR: ${FLUX_ENV} not found"
    exit 1
  fi

  cd "${FLUX_DIR}"

  echo
  echo "========== git =========="
  git log --oneline -5
  git submodule status 3rdparty/nccl || true

  echo
  echo "========== cuda binaries =========="
  command -v python || true
  command -v python3 || true
  command -v nvcc || true
  nvcc --version || true
  readlink -f /usr/local/cuda || true
  readlink -f /usr/local/cuda/lib64/libcudart.so.12 || true

  echo
  echo "========== python torch flux =========="
  python - <<'PY'
import os
import pathlib
import torch

print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())
print("torch nccl:", torch.cuda.nccl.version())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))

try:
    import nvidia.cuda_runtime
    cuda_runtime = pathlib.Path(nvidia.cuda_runtime.__path__[0])
    print("nvidia.cuda_runtime:", cuda_runtime)
    print("nvidia.cuda_runtime libcudart:", cuda_runtime / "lib" / "libcudart.so.12")
except Exception as exc:
    print("nvidia.cuda_runtime: ERROR", repr(exc))

import flux
print("flux import ok")
print("flux helper:", hasattr(flux, "test_nccl_signal_all_gather"))
PY

  echo
  echo "========== environment =========="
  env | sort | grep -E '^(CUDA|CUDART|FLUX|LD_|NCCL|NVSHMEM|PYTHONPATH|TORCH)' || true

  echo
  echo "========== linked libraries =========="
  ldd "${FLUX_DIR}/python/flux/lib/libflux_cuda_ths_op.so" | grep -E 'cudart|cuda|nccl|torch|c10' || true

  echo
  echo "========== loaded libraries on import =========="
  LD_DEBUG=libs python - <<'PY' 2>&1 | grep -E 'libcudart|libcuda|libnccl|libtorch_cuda|libc10_cuda' || true
import flux
PY

  echo
  echo "========== nccl standard init/allgather =========="
  FLUX_AG_NCCL_DEBUG=1 \
  NCCL_DEBUG=TRACE \
  NCCL_DEBUG_SUBSYS=INIT,COLL,GRAPH,ENV \
  torchrun \
    --node_rank=0 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    -- \
    "${FLUX_DIR}/test/python/ag_gemm/test_nccl_signal_ag.py" \
    --m 2048 \
    --k 12288 \
    --dtype bfloat16 \
    --mode standard
} 2>&1 | tee "${LOG_FILE}"

echo
echo "diagnostic log: ${LOG_FILE}"
