cd /root/flux
source /root/flux_env.sh
hash -r

FLUX_AG_EVENT_PROFILE=1 \
FLUX_AG_KERNEL_EVENT_PROFILE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
timeout 60s ./launch.sh test/python/ag_gemm/test_ag_kernel.py \
    2048 49152 12288 \
    --dtype=bfloat16 \
    --iters=1 