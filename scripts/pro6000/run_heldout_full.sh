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
  echo "[run] $TAG  uap=$UAP  start $(date)"
  echo '==========================================================='
  python uap_eval_v2.py       --test_dataset YOUTUBE_VALID       --uap_path "$UAP"       --results_out "results/heldout_${TAG}_results.json"
  local RC=$?
  echo "[run] $TAG  end $(date)  rc=$RC"
}

run_one uap_file/YOUTUBE_v2.pth  v2
run_one uap_file/YOUTUBE.pth     v1

echo '[all] done' $(date)
