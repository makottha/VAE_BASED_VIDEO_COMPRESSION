"""
preprocess_256.py — UCF101 preprocessing for 256×256 MS-VQ-VAE-3 training

Reads raw UCF101 .avi files, extracts 32 frames per clip at 16 FPS,
resizes to 256×256, and saves as float32 .pt tensors.

Output structure:
    dataset_256/
        train/tensors/*.pt
        val/tensors/*.pt
        test/tensors/*.pt

Each .pt file: tensor of shape (3, 32, 256, 256), float32, values in [0, 1]

Features:
  - Resumable: skips clips already saved
  - Progress bar per video (tqdm)
  - Same train/val/test split ratios as 64×64 and 128×128 pipelines (0.70 / 0.15 / 0.15)
  - Estimated output size: ~240-320 GB

Usage:
    conda activate vae_env
    cd C:/Users/kmani/VAE_BASED_VIDEO_COMPRESSION/MS_VQ_VAE_256
    python preprocess_256.py
"""

import os
import random
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
INPUT_DIR       = "../UCF-101"              # raw UCF101 .avi files (after extraction)
OUTPUT_DIR      = "../dataset_256"          # where tensors will be saved
TARGET_FPS      = 16
CLIP_FRAMES     = 32                        # 2 seconds × 16 FPS
FRAME_SIZE      = (256, 256)
SPLITS          = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED            = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

transform = transforms.Compose([
    transforms.ToTensor(),                          # HWC uint8 → CHW float [0,1]
    transforms.Resize(FRAME_SIZE, antialias=True),  # bilinear by default
])


def get_tensor_path(video_name: str, clip_idx: int, split: str) -> str:
    tensor_dir = os.path.join(OUTPUT_DIR, split, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)
    return os.path.join(tensor_dir, f"{video_name}_clip_{clip_idx:04d}.pt")


def process_video(video_path: str) -> int:
    """
    Process a single .avi file into 32-frame clips at TARGET_FPS.
    Returns number of clips saved (0 if skipped entirely).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {video_path}")
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps < TARGET_FPS:
        print(f"[SKIP] FPS too low ({original_fps:.1f}): {video_path}")
        cap.release()
        return 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_interval = max(1, round(original_fps / TARGET_FPS))

    frames = []
    clip_idx = 0
    frame_count = 0
    clips_saved = 0

    # Assign this video's clips to a split deterministically
    split = random.choices(list(SPLITS.keys()), weights=list(SPLITS.values()))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame_rgb))  # (3, 256, 256)

            if len(frames) == CLIP_FRAMES:
                out_path = get_tensor_path(video_name, clip_idx, split)

                if not os.path.exists(out_path):
                    clip_tensor = torch.stack(frames, dim=1)  # (3, 32, 256, 256)
                    clip_tensor = clip_tensor.float().clamp(0.0, 1.0)
                    torch.save(clip_tensor, out_path)
                    clips_saved += 1

                frames = []
                clip_idx += 1

        frame_count += 1

    cap.release()
    return clips_saved


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        print("        Make sure UCF101 has been extracted to that path.")
        return

    # Collect all .avi files recursively
    all_videos = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(".avi"):
                all_videos.append(os.path.join(root, f))

    if not all_videos:
        print(f"[ERROR] No .avi files found under {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(all_videos)} videos. Output → {OUTPUT_DIR}")
    print(f"[INFO] Frame size: {FRAME_SIZE}, FPS: {TARGET_FPS}, Frames/clip: {CLIP_FRAMES}")
    print(f"[INFO] Splits: {SPLITS}")
    print()

    total_saved = 0
    total_skipped = 0

    for video_path in tqdm(all_videos, desc="Videos", unit="vid"):
        try:
            saved = process_video(video_path)
            total_saved += saved
            if saved == 0:
                total_skipped += 1
        except Exception as e:
            print(f"[ERROR] {video_path}: {e}")

    print()
    print(f"[DONE] Clips saved: {total_saved} | Videos skipped: {total_skipped}")

    # Print split counts
    for split in SPLITS:
        tensor_dir = os.path.join(OUTPUT_DIR, split, "tensors")
        if os.path.isdir(tensor_dir):
            n = len([f for f in os.listdir(tensor_dir) if f.endswith(".pt")])
            print(f"       {split}: {n} clips")


if __name__ == "__main__":
    main()
