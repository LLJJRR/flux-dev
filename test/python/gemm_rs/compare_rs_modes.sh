#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_DIR=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)

M=${FLUX_RS_COMPARE_M:-2048}
N=${FLUX_RS_COMPARE_N:-12288}
K=${FLUX_RS_COMPARE_K:-49152}
DTYPE=${FLUX_RS_COMPARE_DTYPE:-bfloat16}
LOG_DIR=${FLUX_RS_COMPARE_LOG_DIR:-"${FLUX_DIR}/test/python/gemm_rs/logs/M${M}_N${N}_K${K}_${DTYPE}"}
WARMUP=${FLUX_RS_COMPARE_WARMUP:-20}
ITERS=${FLUX_RS_COMPARE_ITERS:-100}
TUNE=${FLUX_RS_COMPARE_TUNE:-1}
BLOCKS=${FLUX_RS_COMPARE_BLOCKS:-"4 6 8 12"}
RESUME=${FLUX_RS_COMPARE_RESUME:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

mkdir -p "${LOG_DIR}"
read -r -a BLOCK_VALUES <<< "${BLOCKS}"
FAILED_CASES=()

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
  local status=0

  if [[ "${RESUME}" != "0" ]] && grep -q "flux vs torch bitwise check passed" "${log_file}" 2>/dev/null; then
    echo "skip completed case: ${name}"
    return
  fi

  echo
  echo "========== ${name} =========="
  echo "log: ${log_file}"
  (
    cd "${FLUX_DIR}"
    "$@" ./launch.sh "${COMMON_ARGS[@]}"
  ) 2>&1 | tee "${log_file}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    FAILED_CASES+=("${name}")
  fi
}

run_case_args() {
  local name=$1
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local status=0

  if [[ "${RESUME}" != "0" ]] && grep -q "flux vs torch bitwise check passed" "${log_file}" 2>/dev/null; then
    echo "skip completed case: ${name}"
    return
  fi

  echo
  echo "========== ${name} =========="
  echo "log: ${log_file}"
  (
    cd "${FLUX_DIR}"
    env -u FLUX_RS_DEBUG ./launch.sh "${COMMON_ARGS[@]}" "$@"
  ) 2>&1 | tee "${log_file}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    FAILED_CASES+=("${name}")
  fi
}

export CUDA_VISIBLE_DEVICES

run_case "baseline_rs" \
  env -u FLUX_RS_DEBUG

run_case_args "rs_ring1d" --ring_mode=ring1d

run_case_args "rs_ring2d" --ring_mode=ring2d

run_case_args "rs_p2p_write" --no-use_p2p_read

run_case_args "rs_cudaMemcpyAsync" --use_cudaMemcpyAsync

run_case_args "rs_gemmk_per_tile" --use_gemmk --per_tile_flags

if [[ "${DTYPE}" == "float16" ]]; then
  run_case_args "rs_fuse_reduction" --fuse_reduction
fi

for blocks in "${BLOCK_VALUES[@]}"; do
  run_case_args \
    "rs_p2p_read_blocks_${blocks}" \
    --use_p2p_read \
    --reduce_scatter_blocks="${blocks}"

  run_case_args \
    "rs_gemmk_per_tile_blocks_${blocks}" \
    --use_gemmk \
    --per_tile_flags \
    --reduce_scatter_blocks="${blocks}"
done

if [[ "${#FAILED_CASES[@]}" -ne 0 ]]; then
  echo
  echo "Failed cases: ${FAILED_CASES[*]}"
fi

echo
echo "Logs are under ${LOG_DIR}"
echo "Best hparams summary:"
for log_file in "${LOG_DIR}"/*.log; do
  [[ -f "${log_file}" ]] || continue
  grep "best_hparams=" "${log_file}" 2>/dev/null | sed "s|^|$(basename "${log_file}"): |" || true
done

echo
echo "Per-rank performance summary:"
for log_file in "${LOG_DIR}"/*.log; do
  [[ -f "${log_file}" ]] || continue
  grep -E "^(torch|flux[^ ]*) #[0-9]+:" "${log_file}" 2>/dev/null \
    | sed "s|^|$(basename "${log_file}"): |" || true
done
