#!/bin/bash
# UAP-SAM2 Protocol-Locked Eval
# Protocol: official-code default, seed=30, limit_img=100, limit_frames=15, train-split
# Logs video IDs to confirm in-sample vs held-out

CONDA_BIN=/IMBR_Data/Student-home/2025M_LvShaoting/miniconda3/bin
eval "$($CONDA_BIN/conda shell.bash hook)"
conda activate UAP-SAM2

REPO_DIR=/IMBR_Data/Student-home/2025M_LvShaoting/UAP-SAM2
LOG=/IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor/refine-logs
cd $REPO_DIR

exec > >(tee -a $LOG/eval_locked.log) 2>&1
echo "=== Protocol-Locked UAP Eval ==="
echo "Date: $(date)"
echo "Commit: $(git rev-parse HEAD)"
echo "Params: limit_img=100, limit_frames=15, seed=30, test_prompts=pt"
echo ""

# Step 1: Save eval video IDs (deterministic: same seed=30, limit_img=100)
echo "[step 1] Saving eval video IDs..."
PYTHONPATH=$REPO_DIR python - <<'PYEOF'
import random, json, sys
from pathlib import Path

def seed_everything(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

seed_everything(30)
DATA_ROOT = Path("./data/YOUTUBE/train/JPEGImages")
video_dirs = [v for v in DATA_ROOT.iterdir() if v.is_dir()]
video_dirs.sort(key=lambda x: x.name, reverse=True)
sampled = random.sample(video_dirs, 100)
eval_ids = sorted([v.name for v in sampled])

# Load train IDs (saved earlier)
import os
train_ids_path = "/IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor/refine-logs/train_video_ids.json"
if os.path.exists(train_ids_path):
    with open(train_ids_path) as f:
        train_data = json.load(f)
    train_ids = set(train_data["video_ids"])
    overlap = set(eval_ids) & train_ids
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Eval IDs:  {len(eval_ids)}")
    print(f"  Overlap:   {len(overlap)}/100")
    print(f"  => {'IN-SAMPLE eval (same videos as training)' if len(overlap)==100 else 'PARTIAL overlap: ' + str(len(overlap))}")
else:
    print("  Train IDs not found — cannot compute overlap")

out = {
    "seed": 30, "limit_img": 100, "split": "train",
    "protocol": "official-code-default",
    "video_ids": eval_ids
}
with open("/IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor/refine-logs/eval_video_ids.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved eval video IDs -> refine-logs/eval_video_ids.json")
PYEOF

echo ""
echo "[step 2] Running protocol-locked eval..."
echo "  CUDA_VISIBLE_DEVICES=$(grep CUDA_VISIBLE ~/UAP-SAM2/attack_setting.py | head -1)"
echo ""

PYTHONPATH=$REPO_DIR python -u uap_atk_test.py \
    --train_dataset YOUTUBE \
    --test_dataset YOUTUBE \
    --test_prompts pt \
    --checkpoints sam2-t \
    --limit_img 100 \
    --limit_frames 15 \
    --seed 30 \
    --P_num 10 \
    --prompts_num 256 \
    > $LOG/eval_locked_stdout.log 2>&1
EXIT_CODE=$?

echo ""
echo "=== EVAL COMPLETE ==="
echo "Exit code: $EXIT_CODE"
echo "Stdout log: $LOG/eval_locked_stdout.log"
echo ""

# Step 3: Extract and display metric
echo "=== RESULTS ==="
grep -E "iou_count|miouimg|::" $LOG/eval_locked_stdout.log | tail -5
echo ""

# Step 4: Check GPU usage
echo "GPU memory peak (current):"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -v "^[^0-2]"

echo ""
echo "=== VERDICT ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Run: SUCCESS (exit 0)"
    MIOUADV=$(grep ':: miouimg' $LOG/eval_locked_stdout.log | grep -o 'miouadv: [0-9.]*' | head -1)
    echo "  Result: $MIOUADV"
    echo "  Protocol: official-code default (train-split, filtered mIoU, JPEG eval, seed=30)"
    echo "  Classification: official-code default-protocol baseline (NOT strict reproduction)"
else
    echo "  Run: FAILED (exit $EXIT_CODE)"
fi

echo "Date: $(date)"
