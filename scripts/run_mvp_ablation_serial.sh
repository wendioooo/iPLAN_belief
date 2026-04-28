#!/usr/bin/env bash
# Serialized 4-way MVP ablation: runs one job at a time on a single GPU.
# Use this when GPUs are memory-constrained.
# Usage: GPU=5 BEHAVIOR_WARMUP=500 GAT_WARMUP=500 bash scripts/run_mvp_ablation_serial.sh [seed] [env] [diff] [t_max]

set -e

SEED=${1:-59582679}
ENV=${2:-highway}
DIFF=${3:-chaotic}
T_MAX=${4:-1500000}
BHW=${BEHAVIOR_WARMUP:-20000}
GHW=${GAT_WARMUP:-20000}
GPU=${GPU:-5}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/results/mvp_ablation_logs"
mkdir -p "${LOG_DIR}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate iplan
cd "${REPO_DIR}"

# Reduce GPU memory footprint for shared-GPU environments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BSR=${BATCH_SIZE_RUN:-4}

COMMON="with env=${ENV} difficulty=${DIFF} seed=${SEED} t_max=${T_MAX} \
Behavior_warmup=${BHW} GAT_warmup=${GHW} \
batch_size_run=${BSR} num_test_episodes=${BSR} \
Behavior_enable=True GAT_enable=True GAT_use_behavior=True \
soft_update_enable=True behavior_fully_connected=False"

run_one () {
    local tag=$1
    local extra=$2
    local log="${LOG_DIR}/${tag}_gpu${GPU}_seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] START ${tag} on GPU ${GPU} -> ${log}"
    CUDA_VISIBLE_DEVICES=${GPU} python -u main.py ${COMMON} ${extra} \
        label="${tag}_seed${SEED}" > "${log}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] END   ${tag} (exit=${rc})"
    return ${rc}
}

run_one baseline "mvp_enable=False"
run_one ib_only  "mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.01"
run_one eta_only "mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.0"
run_one mvp_full "mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.01"

echo "All 4 serialized runs finished."
