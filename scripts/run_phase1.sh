#!/usr/bin/env bash
# Phase 1: 5 configs x 3 seeds = 15 runs, 6 GPUs in parallel (3 waves).
# Auto-resume from latest checkpoint if it exists.
# Designed to run inside a tmux session; survives ssh disconnect.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/results/phase1_logs"
MODELS_DIR="${REPO_DIR}/results/models"
mkdir -p "${LOG_DIR}"

source /data/wendiyu/miniforge3/etc/profile.d/conda.sh
conda activate iplan
cd "${REPO_DIR}"

# ---- knobs ----
T_MAX=${T_MAX:-1500000}
BHW=${BHW:-20000}
GHW=${GHW:-20000}
BSR=${BSR:-8}
SEEDS=(${SEEDS:-42 123 456})
GPUS=(${GPUS:-0 1 2 3 4 5})
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
TEST_INTERVAL=${TEST_INTERVAL:-25000}

# Config name -> mvp flag string
declare -A CFG_FLAGS
CFG_FLAGS[C0_baseline]="mvp_enable=False"
CFG_FLAGS[C1_gaussian_only]="mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.0 mvp_use_sample_rollout=True"
CFG_FLAGS[C2_eta_only]="mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.0 mvp_use_sample_rollout=True"
CFG_FLAGS[C3_ib_only]="mvp_enable=True mvp_adaptive_eta=False mvp_ib_kl_weight=0.01 mvp_use_sample_rollout=True"
CFG_FLAGS[C4_mvp_full]="mvp_enable=True mvp_adaptive_eta=True mvp_ib_kl_weight=0.01 mvp_use_sample_rollout=True"

CONFIGS=(C0_baseline C1_gaussian_only C2_eta_only C3_ib_only C4_mvp_full)

# Find an existing checkpoint dir for a given (cfg, seed). Returns "" if none.
find_checkpoint () {
    local cfg=$1
    local seed=$2
    local pattern="ippo_highway_${cfg}-seed${seed}_seed=${seed}_*"
    local latest=$(ls -td ${MODELS_DIR}/${pattern} 2>/dev/null | head -1)
    if [ -z "$latest" ]; then echo ""; return; fi
    # Confirm there is at least one numeric step subdir
    if ls -d ${latest}/[0-9]* 2>/dev/null | grep -q .; then
        echo "$latest"
    else
        echo ""
    fi
}

run_one () {
    local gpu=$1
    local cfg=$2
    local seed=$3
    local extra=${CFG_FLAGS[$cfg]}
    local label="${cfg}-seed${seed}"
    local log="${LOG_DIR}/${label}.log"

    local ckpt=$(find_checkpoint "$cfg" "$seed")
    local resume_args=""
    if [ -n "$ckpt" ]; then
        resume_args="checkpoint_paths='[\"${ckpt}\"]' load_step=0"
        echo "[$(date +%H:%M:%S)] RESUME ${label} on GPU ${gpu} from ${ckpt}" | tee -a "$log"
    else
        echo "[$(date +%H:%M:%S)] START  ${label} on GPU ${gpu}" | tee -a "$log"
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python -u main.py with \
        env=highway difficulty=chaotic seed=${seed} t_max=${T_MAX} \
        Behavior_warmup=${BHW} GAT_warmup=${GHW} \
        batch_size_run=${BSR} num_test_episodes=${BSR} \
        save_model_interval=${SAVE_INTERVAL} test_interval=${TEST_INTERVAL} \
        learner_log_interval=5000 \
        Behavior_enable=True GAT_enable=True GAT_use_behavior=True \
        soft_update_enable=True behavior_fully_connected=False \
        ${extra} ${resume_args} \
        label="${label}" >> "${log}" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] END    ${label} (exit=${rc})" | tee -a "$log"
    return ${rc}
}

# Build job list: (cfg, seed) pairs.
# Outer loop = seed, inner loop = cfg.
# Rationale: each wave covers one seed across ALL configs, so partial results
# (after wave N) already give cross-config comparisons at N seeds.
JOBS=()
for seed in "${SEEDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        JOBS+=("${cfg}|${seed}")
    done
done
NJOBS=${#JOBS[@]}
NGPU=${#GPUS[@]}

echo "Phase 1: ${NJOBS} jobs across ${NGPU} GPUs"
echo "Configs : ${CONFIGS[*]}"
echo "Seeds   : ${SEEDS[*]}"
echo "GPUs    : ${GPUS[*]}"
echo "t_max   : ${T_MAX}"
echo

# Wave-based scheduling: launch NGPU jobs at a time, wait, repeat
i=0
wave=0
while [ $i -lt $NJOBS ]; do
    wave=$((wave + 1))
    echo "==== Wave ${wave} ===="
    pids=()
    for g in "${GPUS[@]}"; do
        if [ $i -ge $NJOBS ]; then break; fi
        IFS='|' read -r cfg seed <<< "${JOBS[$i]}"
        run_one "$g" "$cfg" "$seed" &
        pids+=($!)
        i=$((i + 1))
    done
    echo "Wave ${wave} launched ${#pids[@]} jobs, waiting..."
    for p in "${pids[@]}"; do wait $p; done
    echo "Wave ${wave} complete."
done

echo
echo "All Phase 1 jobs finished."
