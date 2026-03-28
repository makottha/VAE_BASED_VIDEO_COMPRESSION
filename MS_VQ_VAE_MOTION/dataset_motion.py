# dataset_motion.py
"""
Frame-difference dataset for P-frame VQ-VAE training.

Each clip is transformed into a diff clip:
  frame 0      = original frame 0        (I-frame anchor, [0,1])
  frame t>0    = (frame_t - frame_{t-1}) * 0.5 + 0.5   ([0,1] normalized diff)

The VQ-VAE is trained to reconstruct these diff clips.
At eval time, decoded diffs are denormalized and added to the
previous (reconstructed) frame to recover frame_t.
"""

import os
import random
import torch
from torch.utils.data import Dataset


class FrameDiffDataset(Dataset):
    """
    Wraps a directory of .pt clip tensors and returns frame-diff clips.

    Returns: diff_clip (3, T, H, W) in [0, 1]
      frame 0 : original frame 0 (unchanged)
      frame t : (frame_t - frame_{t-1}) * 0.5 + 0.5
    """

    def __init__(
        self,
        tensor_dir: str,
        subset_fraction: float = 1.0,
        shuffle: bool = True,
        expected_T: int = 32,
    ):
        files = [
            os.path.join(tensor_dir, f)
            for f in os.listdir(tensor_dir)
            if f.endswith(".pt")
        ]
        if shuffle:
            random.shuffle(files)
        n = int(len(files) * float(subset_fraction))
        self.files = sorted(files[:n])
        self.expected_T = int(expected_T)
        print(f"[DATASET] {tensor_dir}: {len(self.files)} samples (diff mode)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        clip = torch.load(self.files[idx], weights_only=True)
        if clip.dim() != 4:
            raise ValueError(f"Expected (C,T,H,W), got {clip.shape}")
        C, T, H, W = clip.shape
        if T != self.expected_T:
            raise ValueError(f"Expected T={self.expected_T}, got {T}")
        if C != 3:
            clip = clip[:3]
        clip = clip.float().clamp(0.0, 1.0)

        # Build diff clip
        diff_clip = clip.clone()
        for t in range(1, T):
            diff = clip[:, t] - clip[:, t - 1]         # [-1, 1]
            diff_clip[:, t] = diff * 0.5 + 0.5          # [0, 1]

        return diff_clip
