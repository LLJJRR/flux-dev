#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_DIR=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)
LOG_DIR=${FLUX_COMPARE_LOG_DIR:-"${FLUX_DIR}/test/python/ag_gemm/logs"}

M=${FLUX_COMPARE_M:-2048}
N=${FLUX_COMPARE_N:-49152}
K=${FLUX_COMPARE_K:-12288}
DTYPE=${FLUX_COMPARE_DTYPE:-bfloat16}
WARMUP=${FLUX_COMPARE_WARMUP:-5}
ITERS=${FLUX_COMPARE_ITERS:-10}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  "${FLUX_COMPARE_TEST:-test/python/ag_gemm/test_ag_kernel.py}"
  "${M}" "${N}" "${K}"
  "--dtype=${DTYPE}"
  "--warmup=${WARMUP}"
  "--iters=${ITERS}"
)

if [[ -n "${FLUX_COMPARE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${FLUX_COMPARE_EXTRA_ARGS})
  COMMON_ARGS+=("${EXTRA_ARGS[@]}")
fi

run_case() {
  local name=$1
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo
  echo "========== ${name} =========="
  echo "log: ${log_file}"
  (
    cd "${FLUX_DIR}"
    "$@" ./launch.sh "${COMMON_ARGS[@]}"
  ) 2>&1 | tee "${log_file}"
}

export CUDA_VISIBLE_DEVICES

run_case "baseline_flux" \
  env -u FLUX_AG_USE_NCCL_SIGNAL -u FLUX_AG_NCCL_SIGNAL_WAIT

run_case "nccl_signal_wait" \
  env \
    FLUX_AG_USE_NCCL_SIGNAL=1 \
    FLUX_AG_NCCL_SIGNAL_WAIT=1 \
    NCCL_ALGO="${NCCL_ALGO:-Ring}" \
    NCCL_PROTO="${NCCL_PROTO:-Simple}" \
    NCCL_IBGDA_ENABLE="${NCCL_IBGDA_ENABLE:-0}" \
    NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
    NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"

run_case "nccl_signal_fused" \
  env -u FLUX_AG_NCCL_SIGNAL_WAIT \
    FLUX_AG_USE_NCCL_SIGNAL=1 \
    NCCL_ALGO="${NCCL_ALGO:-Ring}" \
    NCCL_PROTO="${NCCL_PROTO:-Simple}" \
    NCCL_IBGDA_ENABLE="${NCCL_IBGDA_ENABLE:-0}" \
    NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
    NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"

echo
echo "Logs are under ${LOG_DIR}"
