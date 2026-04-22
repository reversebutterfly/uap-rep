#!/bin/bash
set -uo pipefail
cd ~/UAP-SAM2
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/UAP-SAM2
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONUNBUFFERED=1

run_one () {
  local UAP=$1
  local TAG=$2
  echo '==========================================================='
  echo "[trial] $TAG start $(date)"
  echo '==========================================================='
  python uap_eval_v2.py       --test_dataset DAVIS_VAL       --uap_path "$UAP"       --limit_img 30       --limit_frames 15       --results_out "results/davis_${TAG}_results.json"
  echo "[trial] $TAG end $(date) rc=$?"
}

run_one uap_file/YOUTUBE_v2.pth  v2_t1
run_one uap_file/YOUTUBE_v2.pth  v2_t2
run_one uap_file/YOUTUBE_v2.pth  v2_t3
run_one uap_file/YOUTUBE.pth     v1_t1

echo '[all] done' $(date)
