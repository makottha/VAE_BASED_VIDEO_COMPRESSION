"""
codebook_entropy_256.py — Population-level codebook entropy analysis

For each K (128, 256, 512, 1024), runs a forward pass on the 500 test
clips and accumulates ALL index assignments from every codebook level
into one population array before computing entropy.

NOTE: entropy is computed over the full population distribution, NOT
averaged per-clip — per-clip averaging gives an inflated estimate that
does not reflect how the codebook is used across the dataset.

Outputs:
  outputs_msvqvae_256/codebook_entropy.csv
    K, level, K_size, n_symbols, used, utilization_pct,
    entropy_bits, max_entropy_bits, efficiency_pct

Usage:
    conda activate vae_env
    cd MS_VQ_VAE_256
    python codebook_entropy_256.py
"""

import os
import csv
import math

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "../MS_VQ_VAE_RD")
from dataset import VideoClipTensorDataset

from models_3level import MSVQVAE3Video

# ─── CONFIG (must match eval_codec_256.py) ───────────────────────────────────
TEST_DIR   = "../dataset_256/test/tensors"
OUT_DIR    = "./outputs_msvqvae_256"
OUT_CSV    = os.path.join(OUT_DIR, "codebook_entropy.csv")

K_VALUES   = [128, 256, 512, 1024]
EXPECTED_T = 32
BATCH_SIZE = 2
MAX_CLIPS  = 500

TOP_DIM    = 256
MID_DIM    = 192
BOTTOM_DIM = 128
# ─────────────────────────────────────────────────────────────────────────────


def population_entropy(counts: torch.Tensor) -> dict:
    """
    Given a 1-D integer count array (one entry per codebook index),
    compute population Shannon entropy and related stats.

    Returns:
      used          — number of indices with count > 0
      n_symbols     — total symbol count (sum of counts)
      entropy_bits  — Shannon entropy H in bits
      max_entropy   — log2(K) — entropy if uniform
      efficiency    — entropy / max_entropy (%)
    """
    total = counts.sum().item()
    used  = int((counts > 0).sum().item())
    K     = counts.shape[0]

    if total == 0:
        return dict(used=0, n_symbols=0, entropy_bits=0.0,
                    max_entropy_bits=math.log2(K), efficiency_pct=0.0)

    probs = counts.float() / total
    probs = probs[probs > 0]                          # avoid log(0)
    entropy = float(-(probs * torch.log2(probs)).sum().item())
    max_ent = math.log2(K)

    return dict(
        used=used,
        n_symbols=int(total),
        entropy_bits=entropy,
        max_entropy_bits=max_ent,
        efficiency_pct=100.0 * entropy / max_ent,
    )


@torch.no_grad()
def analyse_k(K: int, device: torch.device):
    ae_ckpt = os.path.join(OUT_DIR, f"ae_K{K}_epoch020.pt")
    if not os.path.exists(ae_ckpt):
        print(f"[SKIP] checkpoint not found: {ae_ckpt}")
        return []

    ae = MSVQVAE3Video(
        top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_mid=K, K_bottom=K,
        use_ema=True, use_vgg=False,
    ).to(device).eval()
    state = torch.load(ae_ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(state["model"], strict=False)

    ds     = VideoClipTensorDataset(TEST_DIR, subset_fraction=1.0, expected_T=EXPECTED_T)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Population count accumulators — one per level
    counts_top = torch.zeros(K, dtype=torch.long)
    counts_mid = torch.zeros(K, dtype=torch.long)
    counts_bot = torch.zeros(K, dtype=torch.long)

    n_clips = 0
    for batch in loader:
        remaining = MAX_CLIPS - n_clips
        if remaining <= 0:
            break
        x = batch[:remaining].to(device)
        out = ae(x)

        counts_top += torch.bincount(out["idx_top"].reshape(-1).cpu(), minlength=K)
        counts_mid += torch.bincount(out["idx_mid"].reshape(-1).cpu(), minlength=K)
        counts_bot += torch.bincount(out["idx_bottom"].reshape(-1).cpu(), minlength=K)

        n_clips += x.shape[0]

    print(f"  K={K}: processed {n_clips} clips")

    rows = []
    for level, counts in [("top", counts_top), ("mid", counts_mid), ("bottom", counts_bot)]:
        stats = population_entropy(counts)
        rows.append(dict(K=K, level=level, K_size=K, **stats))
        print(f"    {level:8s}: used={stats['used']:>5}/{K}  "
              f"util={stats['efficiency_pct'] if False else 100*stats['used']/K:5.1f}%  "
              f"H={stats['entropy_bits']:.4f} bits  "
              f"H_max={stats['max_entropy_bits']:.4f}  "
              f"efficiency={stats['efficiency_pct']:.1f}%")

    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Population entropy over {MAX_CLIPS} test clips\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []

    for K in K_VALUES:
        print(f"\n{'─'*50}")
        print(f"  K = {K}")
        print(f"{'─'*50}")
        rows = analyse_k(K, device)
        all_rows.extend(rows)

    # Write CSV
    fieldnames = ["K", "level", "K_size", "n_symbols", "used",
                  "utilization_pct", "entropy_bits", "max_entropy_bits", "efficiency_pct"]

    # Add utilization_pct field (used/K_size * 100)
    for row in all_rows:
        row["utilization_pct"] = round(100.0 * row["used"] / row["K_size"], 2)
        row["entropy_bits"]     = round(row["entropy_bits"], 4)
        row["max_entropy_bits"] = round(row["max_entropy_bits"], 4)
        row["efficiency_pct"]   = round(row["efficiency_pct"], 2)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[INFO] Results → {OUT_CSV}")

    # Pretty-print summary table
    print(f"\n{'K':>6} {'Level':>8} {'Used/K':>10} {'Util%':>7} {'H (bits)':>10} {'H_max':>8} {'Efficiency%':>12}")
    print("─" * 70)
    for row in all_rows:
        print(f"{row['K']:>6} {row['level']:>8} "
              f"{row['used']:>4}/{row['K_size']:<5} "
              f"{row['utilization_pct']:>6.1f}%  "
              f"{row['entropy_bits']:>9.4f}  "
              f"{row['max_entropy_bits']:>7.4f}  "
              f"{row['efficiency_pct']:>10.1f}%")


if __name__ == "__main__":
    main()
