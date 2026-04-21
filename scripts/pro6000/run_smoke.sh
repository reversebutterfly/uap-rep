#!/bin/bash
set -uo pipefail
cd ~/UAP-SAM2
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/UAP-SAM2
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONUNBUFFERED=1
echo '[smoke] start' $(date)
python uap_eval_v2.py     --test_dataset YOUTUBE_VALID     --uap_path uap_file/YOUTUBE_v2.pth     --limit_img 1     --limit_frames 2
echo '[smoke] end' $(date) 'rc='$?
