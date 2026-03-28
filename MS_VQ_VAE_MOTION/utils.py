# utils.py
import os
import torch


def save_ckpt(path: str, **state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_ckpt(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
