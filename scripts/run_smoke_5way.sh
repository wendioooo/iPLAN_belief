#!/usr/bin/env bash
# 5-way MVP smoke test (parallel across 5 GPUs, skip GPU 3).
# Aligns train and rollout sampling: all MVP configs use mvp_use_sample_rollout=True.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/results/smoke_logs"
mkdir -p "${LOG_DIR}"

source /data/wendiyu/miniforge3/etc/profile.d/conda.sh
conda activate iplan
cd "${REPO_DIR}"

SEED=${SEED:-42}
T_MAX=${T_MAX:-2000}
BHW=${BHW:-200}
GHW=${GHW:-200}
BSR=${BSR:-8}

COMMON="with env=highway difficulty=chaotic seed=${SEED} t_max=${T_MAX} \
Behavior_warmup=${BHW} GAT_warmup=${GHW} \
batch_size_run=${BSR} num_test_episodes=${BSR} \
Behavior_enable=True GAT_enable=True GAT_use_behavior=True \
soft_update_enable=True behavior_fully_connected=False"

launch () {
    local gpu=$1
    local tag=$2
    local extra=$3
    local log="${LOG_DIR}/smoke_${tag}.log"
    echo "[$(date +%H:%M:%S)] LAUNCH ${tag} on GPU ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main.py ${COMMON} ${extra} \
        label="smoke_${tag}_seed${SEED}" > "${log}" 2>&1 &
    echo $! > "${LOG_DIR}/smoke_${tag}.pid"
}

# C0 baseline (vanilla iPLAN, no MVP)
launch 0 C0_baseline "mvp_enable=False"

# C1 gaussian_only: Gaussian encoder, constant eta, no IB; rollout aligned via sample
launch 1 C1_gaussian_only "mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.0 mvp_use_sample_rollout=True"

# C2 eta_only: + adaptive Kalman gain
launch 2 C2_eta_only "mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.0 mvp_use_sample_rollout=True"

# C3 ib_only: + IB KL
launch 4 C3_ib_only "mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.01 mvp_use_sample_rollout=True"

# C4 mvp_full: A + B together
launch 5 C4_mvp_full "mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.01 mvp_use_sample_rollout=True"

echo
echo "All 5 smoke jobs launched. PIDs:"
for f in "${LOG_DIR}"/smoke_*.pid; do
    tag=$(basename "$f" .pid)
    echo "  $(cat $f)  $tag"
done
echo
echo "Tail logs with: tail -f ${LOG_DIR}/smoke_*.log"
