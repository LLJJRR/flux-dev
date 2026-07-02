#!/bin/bash
set -e
set -x

########################################
# CONFIG
########################################

FLUX_DIR=/root/flux
VENV_DIR=/root/venv-flux
JOBS=${JOBS:-16}

########################################
# CHECK
########################################

if [ ! -d "${FLUX_DIR}" ]; then
    echo "ERROR: ${FLUX_DIR} does not exist."
    echo "Please run env_prepare.sh first."
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: ${VENV_DIR} does not exist."
    echo "Please run env_prepare.sh first."
    exit 1
fi

########################################
# PYTHON ENV
########################################

source ${VENV_DIR}/bin/activate

export NVSHMEM_HOME=$(python - <<'PY'
import pathlib
import nvidia.nvshmem
print(pathlib.Path(nvidia.nvshmem.__path__[0]))
PY
)

echo "NVSHMEM_HOME=${NVSHMEM_HOME}"

cd ${NVSHMEM_HOME}/lib
ln -sf libnvshmem_host.so.3 libnvshmem_host.so

########################################
# BUILD ENV
########################################

cd ${FLUX_DIR}

export NVSHMEM_HOME=${NVSHMEM_HOME}
export NCCL_ROOT=${FLUX_DIR}/3rdparty/nccl/build/local

export FLUX_SHM_USE_NVSHMEM=1

export LD_LIBRARY_PATH=\
${NCCL_ROOT}/lib:\
${NVSHMEM_HOME}/lib:\
/usr/local/cuda/lib64:\
${LD_LIBRARY_PATH}

########################################
# CLEAN
########################################

rm -rf build
rm -rf build/lib.*
rm -f python/flux_ths_pybind*.so

########################################
# BUILD
########################################

./build.sh \
    --arch 90 \
    --sm-cores 132 \
    --nvshmem \
    --jobs ${JOBS}

########################################
# ENV SCRIPT
########################################

cat >/root/flux_env.sh <<EOF
source ${VENV_DIR}/bin/activate

export NVSHMEM_HOME=${NVSHMEM_HOME}
export NCCL_ROOT=${NCCL_ROOT}

export PYTHONPATH=${FLUX_DIR}/python:\$PYTHONPATH

export LD_LIBRARY_PATH=${FLUX_DIR}/python/flux/lib:${NCCL_ROOT}/lib:${NVSHMEM_HOME}/lib:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH

export FLUX_SHM_USE_NVSHMEM=1

# Enable this only when comparing the experimental NCCL-signal AG producer.
# export FLUX_AG_USE_NCCL_SIGNAL=1
#
# Conservative correctness mode: GEMM waits for NCCL AllGather completion.
# Leave unset for fused mode, where GEMM waits on NCCL-written barriers.
# export FLUX_AG_NCCL_SIGNAL_WAIT=1

# Recommended first-run NCCL settings for the experimental signal path.
# export NCCL_ALGO=Ring
# export NCCL_PROTO=Simple
# export NCCL_IBGDA_ENABLE=0
# export NCCL_NVLS_ENABLE=0
# export NCCL_COLLNET_ENABLE=0
EOF

########################################
# VERIFY
########################################

source /root/flux_env.sh

python - <<'PY'
import torch
print("torch:", torch.__version__)

import flux
print("flux import ok")
PY

python test/python/gemm_only/test_gemm_only.py \
    4096 12288 6144 \
    --input_dtype bfloat16 \
    --weight_dtype bfloat16 \
    --iters 5

echo ""
echo "===================================="
echo "FLUX BUILD SUCCESS"
echo "source /root/flux_env.sh"
echo "===================================="
