"""
dataset_DAVIS.py - DAVIS 2017 loader for UAP-SAM2 eval.

Mirrors dataset_YOUTUBE.Dataset_YOUTUBE but reads DAVIS palette-PNG annotations
(unlike YouTube-VOS hardcoded RGBA tuple (236, 95, 103, 255), DAVIS uses
palette index 1 for the first annotated instance).
"""
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from sam2.build_sam import build_sam2
from attack_setting import SamForwarder, make_prompts
from dataset_YOUTUBE import (
    calculate_center,
    calculate_bounding_box,
    _pick_point_prompt,
)

# DAVIS convention: palette index 1 = first annotated instance
# Multi-instance videos (bmx-trees has indices 0/1/2) also use 1 for the first object,
# consistent with standard DAVIS evaluation.
DAVIS_TARGET_INSTANCE = 1


class Dataset_DAVIS(Dataset):
    def __init__(self, sample_ids, data_root, json_path, target_size=(1024, 1024),
                 do_shuffle=False, args=None, start_frames=None,
                 target_instance=DAVIS_TARGET_INSTANCE):
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.json_path = json_path
        self.target_size = target_size
        self.do_shuffle = do_shuffle
        self.args = args
        self.target_instance = target_instance
        self.checkpoint = "../sam2/checkpoints/sam2_hiera_tiny.pt"
        self.model_cfg = "configs/sam2/sam2_hiera_t.yaml"
        self.sam = build_sam2(self.model_cfg, self.checkpoint, device='cuda')
        self.start_frames = start_frames or {}
        combined_data = list(zip(self.sample_ids, self._load_data()))
        if not combined_data:
            raise RuntimeError("Dataset_DAVIS: no samples loaded successfully")
        self.sample_ids, self.data = zip(*combined_data[::-1])

    def _load_data(self):
        data = []
        failed_samples = []
        for sample_id in self.sample_ids:
            image_path = self.data_root / f"{sample_id}.jpg"
            gt_path = Path(self.json_path) / f"{sample_id}.png"
            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image.resize(self.target_size[::-1], Image.BILINEAR))

                gt_palette = np.array(Image.open(gt_path).convert('P'))
                single_channel_mask = np.where(
                    gt_palette == self.target_instance, 255, 0
                ).astype(np.uint8)
                if single_channel_mask.sum() == 0:
                    # target instance not visible in this frame -> skip
                    failed_samples.append(sample_id)
                    continue

                single_mask = Image.fromarray(single_channel_mask)
                gt = np.array(single_mask.resize(self.target_size[::-1], Image.NEAREST))

                sam_fwder = SamForwarder(self.sam)
                X = sam_fwder.transform_image(image)

                if self.args.train_prompts == 'pt':
                    print("pt")
                    x_point, y_point = _pick_point_prompt(gt, self.args)
                    prompt_ann = np.array([[x_point, y_point]], dtype=np.float32)
                    prompts = make_prompts(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                else:
                    # Box prompts on DAVIS not implemented (make_prompts_box is absent
                    # upstream; YOUTUBE path also has a dead bx branch).
                    raise NotImplementedError(
                        f"train_prompts={self.args.train_prompts} not supported "
                        "on DAVIS — only 'pt' is wired.")

                data.append((image, P, sample_id, gt, prompt_ann))
                print(f"from DAVIS Processing sample {sample_id}...")
            except Exception as e:
                print(f"Error loading data for sample {sample_id}: {e}")
                failed_samples.append(sample_id)
                continue
        if failed_samples:
            print(f"[DAVIS] skipped {len(failed_samples)} samples "
                  f"(target instance not visible or load error)")
        return data

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_start_frame_idx(self, video_name):
        return self.start_frames.get(video_name, None)

    def get_img_id(self, idx):
        try:
            if isinstance(idx, int):
                if idx < 0 or idx >= len(self.sample_ids):
                    raise IndexError(
                        f"Index {idx} is out of range. "
                        f"Total samples: {len(self.sample_ids)}"
                    )
                sample_id = self.sample_ids[idx]
            elif isinstance(idx, str):
                sample_id = idx
            else:
                raise TypeError(
                    f"Expected idx to be an integer or string, but got {type(idx)}"
                )
            parts = sample_id.split('/')
            if len(parts) >= 2:
                return parts[0]
            raise ValueError(f"Could not extract image ID from sample_id: {sample_id}")
        except Exception as e:
            print(f"Error extracting image ID from idx {idx}: {e}")
            return None
