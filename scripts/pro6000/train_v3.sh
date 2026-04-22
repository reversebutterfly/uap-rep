#!/bin/bash
set -uo pipefail
cd ~/UAP-SAM2
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /LAI_Data/Anaconda_envs/2025Lv_Zhaoting/UAP-SAM2
export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONUNBUFFERED=1
echo '[train-v3] start' $(date)
python uap_attack_v2.py     --train_dataset YOUTUBE     --loss_t --loss_diff --loss_fea     --weight_fea 1.0     --out_suffix v3
echo '[train-v3] end' $(date) 'rc='$?
