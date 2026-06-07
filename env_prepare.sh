#!/bin/bash
set -e
set -x

########################################
# CONFIG
########################################

FLUX_DIR=/root/flux
VENV_DIR=/root/venv-flux

########################################
# SYSTEM
########################################

apt update

apt install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3.11 \
    python3.11-dev \
    python3.11-venv

########################################
# PYTHON ENV
########################################

if [ ! -d "${VENV_DIR}" ]; then
    python3.11 -m venv ${VENV_DIR}
fi

source ${VENV_DIR}/bin/activate

python -m pip install -U \
    pip \
    setuptools \
    wheel

pip install \
    packaging \
    ninja

########################################
# PYTORCH
########################################

pip install \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

########################################
# NVSHMEM
########################################

pip uninstall -y nvidia-nvshmem-cu12 || true

pip install nvidia-nvshmem-cu12==3.3.9

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
# CLONE FLUX
########################################

if [ ! -d "${FLUX_DIR}" ]; then
    git clone --recursive \
        https://github.com/LLJJRR/flux-dev.git \
        ${FLUX_DIR}
fi

cd ${FLUX_DIR}

git submodule update --init --recursive

########################################
# PATCH BUILD.SH
########################################

# venv 里面不能带 --user
sed -i 's/ --user//g' build.sh

# setup.py develop 会触发 PEP517 隔离环境
sed -i \
's#MAX_JOBS=${JOBS} python3 setup.py develop#MAX_JOBS=${JOBS} python3 -m pip install -e . --no-build-isolation#g' \
build.sh

# ninja merge compile_commands bug
sed -i \
's/^trap merge_compile_commands EXIT/# trap merge_compile_commands EXIT/g' \
build.sh

echo ""
echo "===================================="
echo "FLUX ENV PREPARE SUCCESS"
echo "FLUX_DIR=${FLUX_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "NVSHMEM_HOME=${NVSHMEM_HOME}"
echo "===================================="
