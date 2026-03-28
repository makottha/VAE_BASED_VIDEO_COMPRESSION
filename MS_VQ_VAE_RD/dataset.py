# dataset.py
import os
import random
import torch
from torch.utils.data import Dataset


class VideoClipTensorDataset(Dataset):
    """
    Expects .pt files containing a tensor shaped:
      (3, T, H, W) or (C, T, H, W)
    Will return float tensor (3,T,H,W) in [0,1]
    """
    def __init__(self, tensor_dir: str, subset_fraction: float = 1.0, shuffle: bool = True, expected_T: int = 32):
        files = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        if shuffle:
            random.shuffle(files)
        n = int(len(files) * float(subset_fraction))
        self.files = sorted(files[:n])
        self.expected_T = int(expected_T)
        print(f"[DATASET] {tensor_dir}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        clip = torch.load(self.files[idx], weights_only=True)  # tensor
        if clip.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C,T,H,W), got {clip.shape} in {self.files[idx]}")
        C, T, H, W = clip.shape
        if T != self.expected_T:
            raise ValueError(f"Expected T={self.expected_T}, got {T} in {self.files[idx]}")
        if C != 3:
            clip = clip[:3]
        clip = clip.float().clamp(0, 1)
        return clip
