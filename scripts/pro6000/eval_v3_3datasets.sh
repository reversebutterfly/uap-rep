#!/bin/bash
set -uo pipefail
cd ~/UAP-SAM2
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/UAP-SAM2
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONUNBUFFERED=1

run_one () {
  local DS=$1
  local LIMIT=$2
  local TAG=$3
  echo '==========================================================='
  echo "[eval-v3] $TAG ds=$DS start $(date)"
  echo '==========================================================='
  python uap_eval_v2.py       --test_dataset "$DS"       --uap_path uap_file/YOUTUBE_v3.pth       --limit_img "$LIMIT"       --limit_frames 15       --results_out "results/v3_${TAG}_results.json"
  echo "[eval-v3] $TAG end $(date) rc=$?"
}

# YT in-domain (same as training; 100 train videos)
run_one YOUTUBE       100 yt_in_domain
# DAVIS val (all 30 videos)
run_one DAVIS_VAL     30  davis
# MOSE train 100 — three runs for Blackwell variance
run_one MOSE_TRAIN    100 mose_t1
run_one MOSE_TRAIN    100 mose_t2
run_one MOSE_TRAIN    100 mose_t3

echo '[all] done' $(date)
