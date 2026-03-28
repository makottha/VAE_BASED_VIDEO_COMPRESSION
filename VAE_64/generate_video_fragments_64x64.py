"""
generate_video_fragments_64x64.py

Description:
-------------
This script processes raw video files (e.g., from UCF101 dataset),
splits them into fixed-length 2-second video fragments (32 frames at 16 FPS),
resizes each frame to 64x64 pixels,
and saves two outputs:
    1. Human-readable .avi video clips
    2. PyTorch .pt tensor files for training deep learning models.

It organizes the output into "train", "val", and "test" splits automatically.

Intended for use in deep learning pipelines such as Variational Autoencoders (VAE) for video compression research.

Requirements:
-------------
- OpenCV (`cv2`)
- Torch (`torch`)
- torchvision (`transforms`)
- tqdm (optional progress bars)

Usage:
------
Run the script as a standalone:
$ python generate_video_fragments_64x64.py

Configure input/output folders in the __main__ section.

"""

import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# ---- CONFIGURATION ----
CLIP_LENGTH_SECONDS = 2          # Each clip is 2 seconds long
TARGET_FPS = 16                  # Frame sampling rate
CLIP_LENGTH_FRAMES = CLIP_LENGTH_SECONDS * TARGET_FPS  # Number of frames per clip
FRAME_SIZE = (64, 64)             # Resize all frames to 64x64 pixels
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}   # Dataset split ratios
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- FRAME TRANSFORMATION ----
transform = transforms.Compose([
    transforms.ToTensor(),                     # Convert HWC to CHW format
    transforms.Resize(FRAME_SIZE, antialias=True),
])

def save_clip_tensor(frames, out_path_tensor):
    """
    Save a list of frames as a tensor file (.pt).
    Shape: (3, 32, H, W)
    """
    clip_tensor = torch.stack(frames, dim=1)  # Stack along time dimension
    torch.save(clip_tensor.cpu(), out_path_tensor)

def split_and_process_video(video_path, dataset_output_dir, human_output_dir):
    """
    Read a video file, split it into 2-second fragments at TARGET_FPS, resize frames,
    and save both tensor (.pt) and AVI (.avi) formats.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Couldn't open {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps < TARGET_FPS:
        print(f"[SKIP] {video_path} FPS={original_fps:.2f} < target {TARGET_FPS}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_interval = int(original_fps / TARGET_FPS)
    frame_count = 0
    frames = []
    clip_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).to(DEVICE)
            frames.append(frame_tensor)

            if len(frames) == CLIP_LENGTH_FRAMES:
                split = random.choices(list(SPLITS.keys()), weights=SPLITS.values())[0]
                clip_filename = f"{video_name}_clip_{clip_count:04d}"

                # Output directories
                tensor_dir = os.path.join(dataset_output_dir, split, "tensors")
                clip_dir = os.path.join(human_output_dir, split, "clips")
                os.makedirs(tensor_dir, exist_ok=True)
                os.makedirs(clip_dir, exist_ok=True)

                out_tensor_path = os.path.join(tensor_dir, f"{clip_filename}.pt")
                out_clip_path = os.path.join(clip_dir, f"{clip_filename}.avi")

                # Save .pt tensor
                save_clip_tensor(frames, out_tensor_path)

                # Save human-readable .avi clip
                out = cv2.VideoWriter(out_clip_path, cv2.VideoWriter_fourcc(*'XVID'), TARGET_FPS, FRAME_SIZE)
                for f in frames:
                    f_cpu = f.permute(1, 2, 0).cpu().numpy()
                    f_bgr = cv2.cvtColor((f_cpu * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    out.write(f_bgr)
                out.release()

                print(f"[SAVED] {clip_filename} → {split}")
                frames = []
                clip_count += 1

        frame_count += 1

    cap.release()

def batch_process(input_dir, dataset_output_dir, human_output_dir):
    """
    Process all .avi videos in input_dir recursively.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".avi"):
                video_path = os.path.join(root, file)
                try:
                    split_and_process_video(video_path, dataset_output_dir, human_output_dir)
                except Exception as e:
                    print(f"[ERROR] Failed on {video_path}: {e}")

if __name__ == "__main__":
    # Configure paths
    input_dir = "./UCF101"                        # Raw UCF101 videos
    dataset_output_dir = "./dataset_64"              # Model input: .pt tensors
    human_output_dir = "./dataset_human_readable_64" # For human-readable .avi clips

    print(f"[INFO] Using device: {DEVICE} | Target FPS: {TARGET_FPS} | Frame size: {FRAME_SIZE}")
    batch_process(input_dir, dataset_output_dir, human_output_dir)
    print("[DONE] All videos processed.")
