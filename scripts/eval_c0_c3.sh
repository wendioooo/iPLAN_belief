#!/usr/bin/env bash
set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/results/eval_logs"
MODELS_DIR="${REPO_DIR}/results/models"
mkdir -p "${LOG_DIR}"

source /data/wendiyu/miniforge3/etc/profile.d/conda.sh
conda activate iplan
cd "${REPO_DIR}"

export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

GPU=${GPU:-0}
TEST_NEP=${TEST_NEP:-3}
BSR=${BSR:-8}

C0_CKPT="${MODELS_DIR}/ippo_highway_C0-baseline-seed42_seed=42_04-14-01-56-21"
C3_CKPT="${MODELS_DIR}/ippo_highway_C3-ib-only-seed42_seed=42_04-14-01-56-22"

COMMON="env=highway difficulty=chaotic seed=42 evaluate=True animation_enable=True \
  test_nepisode=${TEST_NEP} batch_size_run=${BSR} num_test_episodes=${BSR} \
  Behavior_enable=True GAT_enable=True GAT_use_behavior=True \
  soft_update_enable=True behavior_fully_connected=False"

stash_gifs () {
    mkdir -p animation_gifs_backup
    find animation -maxdepth 1 -name "*.gif" -exec mv {} animation_gifs_backup/ \; 2>/dev/null || true
}
restore_gifs () {
    if [ -d animation_gifs_backup ]; then
        mv animation_gifs_backup/*.gif animation/ 2>/dev/null || true
        rmdir animation_gifs_backup 2>/dev/null || true
    fi
}
archive_frames () {
    local dst=$1
    rm -rf "${dst}"
    mkdir -p "${dst}"
    find animation -maxdepth 1 -mindepth 1 -type d -exec mv {} "${dst}/" \; 2>/dev/null || true
}

run_eval () {
    local label=$1
    local ckpt=$2
    local extra=$3
    local dst="animation_${label}_seed42"
    local log="${LOG_DIR}/${label}-seed42.log"

    echo "[$(date +%H:%M:%S)] START ${label} on GPU ${GPU}" | tee -a "$log"
    stash_gifs
    rm -rf animation && mkdir -p animation

    CUDA_VISIBLE_DEVICES=${GPU} python -u main.py with \
        ${COMMON} ${extra} \
        checkpoint_paths="[\"${ckpt}\"]" load_step=0 \
        label="eval-${label}-seed42" >> "$log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] END ${label} (exit=${rc})" | tee -a "$log"

    archive_frames "${dst}"
    restore_gifs
    return ${rc}
}

run_eval C0 "${C0_CKPT}" "mvp_enable=False"
run_eval C3 "${C3_CKPT}" "mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.01 mvp_use_sample_rollout=True"

echo "All evaluations done."
