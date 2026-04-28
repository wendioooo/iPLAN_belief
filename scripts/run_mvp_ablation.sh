#!/usr/bin/env bash
# Run 4 MVP ablation experiments on GPUs 6/7/8/9 in parallel.
# Usage: bash scripts/run_mvp_ablation.sh [seed] [env] [difficulty] [t_max]

set -e

SEED=${1:-59582679}
ENV=${2:-highway}
DIFF=${3:-chaotic}
T_MAX=${4:-1500000}
# Warmups default to the paper's 20000; smoke tests should override via env var.
BHW=${BEHAVIOR_WARMUP:-20000}
GHW=${GAT_WARMUP:-20000}
# GPUs override via env var: GPUS="1 2 3 4" bash scripts/run_mvp_ablation.sh ...
GPUS=${GPUS:-"6 7 8 9"}
read -r -a GPU_ARR <<< "${GPUS}"
if [[ ${#GPU_ARR[@]} -ne 4 ]]; then
    echo "ERROR: GPUS must specify exactly 4 GPU indices (got: ${GPUS})"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/results/mvp_ablation_logs"
mkdir -p "${LOG_DIR}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate iplan

COMMON="with env=${ENV} difficulty=${DIFF} seed=${SEED} t_max=${T_MAX} \
Behavior_warmup=${BHW} GAT_warmup=${GHW} \
Behavior_enable=True GAT_enable=True GAT_use_behavior=True \
soft_update_enable=True behavior_fully_connected=False"

launch () {
    local gpu=$1
    local tag=$2
    local extra=$3
    local log="${LOG_DIR}/${tag}_gpu${gpu}_seed${SEED}.log"
    echo "[GPU ${gpu}] ${tag} -> ${log}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main.py ${COMMON} ${extra} \
        label="${tag}_seed${SEED}" \
        > "${log}" 2>&1 &
    echo "  PID=$!"
}

cd "${REPO_DIR}"

# Run 1: baseline (vanilla iPLAN)
launch ${GPU_ARR[0]} baseline \
    "mvp_enable=False"

# Run 2: IB only (constant eta + KL to N(0,I))
launch ${GPU_ARR[1]} ib_only \
    "mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.01"

# Run 3: adaptive eta only (data-dependent Kalman gain, no KL)
launch ${GPU_ARR[2]} eta_only \
    "mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.0"

# Run 4: full MVP (A + B)
launch ${GPU_ARR[3]} mvp_full \
    "mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.01"

echo
echo "All 4 runs launched. Monitor with:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  nvidia-smi"
echo "  tensorboard --logdir ${REPO_DIR}/results/tb_logs"
wait
echo "All runs finished."
