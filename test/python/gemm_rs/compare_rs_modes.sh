#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_DIR=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)
LOG_DIR=${FLUX_RS_COMPARE_LOG_DIR:-"${FLUX_DIR}/test/python/gemm_rs/logs"}

M=${FLUX_RS_COMPARE_M:-2048}
N=${FLUX_RS_COMPARE_N:-49152}
K=${FLUX_RS_COMPARE_K:-12288}
DTYPE=${FLUX_RS_COMPARE_DTYPE:-bfloat16}
WARMUP=${FLUX_RS_COMPARE_WARMUP:-5}
ITERS=${FLUX_RS_COMPARE_ITERS:-20}
TUNE=${FLUX_RS_COMPARE_TUNE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

mkdir -p "${LOG_DIR}"

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ "${GPU_COUNT}" -lt 2 && "${FLUX_RS_COMPARE_ALLOW_SINGLE_GPU:-0}" != "1" ]]; then
  echo "compare_rs_modes.sh needs at least 2 visible GPUs for GEMM-RS comparison."
  echo "Currently visible GPUs: ${GPU_COUNT}."
  echo "Expose more GPUs, or set FLUX_RS_COMPARE_ALLOW_SINGLE_GPU=1 for a single-rank smoke test."
  exit 1
fi

COMMON_ARGS=(
  "${FLUX_RS_COMPARE_TEST:-test/python/gemm_rs/test_gemm_rs.py}"
  "${M}" "${N}" "${K}"
  "--dtype=${DTYPE}"
  "--warmup=${WARMUP}"
  "--iters=${ITERS}"
)

if [[ "${TUNE}" != "0" ]]; then
  COMMON_ARGS+=("--tune-gemm-rs")
fi

if [[ -n "${FLUX_RS_COMPARE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${FLUX_RS_COMPARE_EXTRA_ARGS})
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

run_case_args() {
  local name=$1
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo
  echo "========== ${name} =========="
  echo "log: ${log_file}"
  (
    cd "${FLUX_DIR}"
    env FLUX_RS_DEBUG="${FLUX_RS_DEBUG:-1}" ./launch.sh "${COMMON_ARGS[@]}" "$@"
  ) 2>&1 | tee "${log_file}"
}

export CUDA_VISIBLE_DEVICES

run_case "baseline_rs" \
  env -u FLUX_RS_DEBUG

run_case_args "rs_ring1d" --ring_mode=ring1d

run_case_args "rs_ring2d" --ring_mode=ring2d

run_case_args "rs_cudaMemcpyAsync" --use_cudaMemcpyAsync

run_case_args "rs_gemmk_per_tile" --use_gemmk --per_tile_flags

echo
echo "Logs are under ${LOG_DIR}"
echo "Best hparams summary:"
grep -h "best_hparams=" "${LOG_DIR}"/*.log 2>/dev/null || true
