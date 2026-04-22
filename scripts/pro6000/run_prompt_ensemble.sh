#!/bin/bash
set -uo pipefail
cd ~/UAP-SAM2
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/UAP-SAM2
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONUNBUFFERED=1

run_trial () {
    local UAP=$1
    local MODE=$2
    local TAG=$3
    local SEED=${4:-0}
    echo '==========================================================='
    echo "[trial] $TAG  uap=$UAP  mode=$MODE  prompt_seed=$SEED  start $(date)"
    echo '==========================================================='
    python uap_eval_v2.py         --test_dataset YOUTUBE_VALID         --uap_path "$UAP"         --prompt_mode "$MODE"         --prompt_seed "$SEED"         --results_out "results/heldout_${TAG}_results.json"
    local RC=$?
    echo "[trial] $TAG  end $(date)  rc=$RC"
}

# Center baseline (already have it, but re-run for paired consistency)
run_trial uap_file/YOUTUBE_v2.pth  center      v2_center_repeat  0

# Random FG x5 seeds
for S in 1 2 3 4 5; do
  run_trial uap_file/YOUTUBE_v2.pth  random_fg   v2_randfg_s$S     $S
done

echo '[ensemble] all done' $(date)
