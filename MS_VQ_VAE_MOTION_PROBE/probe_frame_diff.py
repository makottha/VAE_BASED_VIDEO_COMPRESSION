#!/usr/bin/env python3
"""
Probe 2: Frame Differencing (No Flow)
======================================
Simpler hypothesis test — no RAFT, no warping.

  residual_t = frame_t - frame_{t-1}   (raw temporal difference)

This is the LOWER BOUND of what motion compensation can achieve.
If even this naive approach shows BPP reduction, then:
  - Temporal redundancy IS present in UCF101 at 64x64
  - The probe_motion_compensation.py failure was RAFT quality, not the hypothesis
  - Phase 2 (learned motion VQ) will work

Comparison:
  baseline : encode raw clip with VQ-VAE
  proposed : encode frame-diff clip with VQ-VAE (no warping)
  reconstruct: frame_t = frame_{t-1} + decoded_diff_t
"""

import sys, math, time, json, csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

PROBE_DIR  = Path(__file__).resolve().parent
RD_DIR     = PROBE_DIR.parent / "MS_VQ_VAE_RD"
sys.path.insert(0, str(RD_DIR))
from models import MSVQVAE2Video

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = RD_DIR.parent / "dataset_64" / "test" / "tensors"
AE_CKPT    = RD_DIR / "outputs_msvqvae_rd" / "ae_K1024_epoch020.pt"
RESULTS_DIR = PROBE_DIR / "results_framediff"
PLOTS_DIR  = RESULTS_DIR / "plots"
SAMPLES_DIR = PLOTS_DIR / "samples"

N_CLIPS         = 100
N_SAMPLE_CLIPS  = 5
K = 1024; TOP_DIM = 256; BOTTOM_DIM = 128

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_ae():
    ae = MSVQVAE2Video(
        top_dim=TOP_DIM, bottom_dim=BOTTOM_DIM, K_top=K, K_bottom=K,
        commit_top=0.25, commit_bottom=0.25, norm="gn",
        use_vgg=False, soft_temp=1.0, use_ema=False,
    ).to(DEVICE).eval()
    ckpt = torch.load(str(AE_CKPT), map_location="cpu", weights_only=False)
    ae.load_state_dict(ckpt["ae"], strict=False)
    log(f"AE (K={K}) loaded.")
    return ae

@torch.no_grad()
def encode_decode(ae, clip):
    out = ae(clip.unsqueeze(0).to(DEVICE))
    out["recon"] = out["recon"].clamp(0.0, 1.0)
    return out

def entropy_bpp(out, T, H, W):
    n_top = out["idx_top"].numel()
    n_bot = out["idx_bottom"].numel()
    return (out["entropy_top"] * n_top + out["entropy_bottom"] * n_bot) / (T * H * W)

def psnr(x, y):
    mse = F.mse_loss(x, y).item()
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)

def tensor_to_np(t):
    return (t.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

def save_sample(idx, clip, diff_clip, recon_raw, final_recon, t=15):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Frame Diff Probe — Clip {idx:03d}  t={t}", fontsize=12)
    diff_vis = diff_clip[:, t].clone()  # [0,1] normalized diff
    diff_abs = (diff_vis - 0.5).abs()   # magnitude

    imgs = [
        tensor_to_np(clip[:, t]),
        tensor_to_np(clip[:, t-1]),
        tensor_to_np(diff_vis),
        tensor_to_np(diff_abs * 4),  # amplified for visibility
        tensor_to_np(recon_raw[:, t]),
        tensor_to_np(final_recon[:, t]),
        tensor_to_np((clip[:, t] - recon_raw[:, t]).abs()),
        tensor_to_np((clip[:, t] - final_recon[:, t]).abs()),
    ]
    titles = [
        "Original frame_t",
        "frame_{t-1}",
        "Frame diff (normalized)",
        "Frame diff magnitude (4x)",
        f"VQ-VAE raw recon\nPSNR={psnr(recon_raw[:,t], clip[:,t]):.1f}dB",
        f"Diff+prev recon\nPSNR={psnr(final_recon[:,t], clip[:,t]):.1f}dB",
        "Error: raw recon",
        "Error: diff recon",
    ]
    for ax, img, title in zip(axes.flat, imgs, titles):
        ax.imshow(img); ax.set_title(title, fontsize=8); ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(SAMPLES_DIR / f"clip_{idx:03d}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    for d in [RESULTS_DIR, PLOTS_DIR, SAMPLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    ae = load_ae()

    clip_files = sorted(DATA_DIR.glob("*.pt"))[:N_CLIPS]
    log(f"Processing {len(clip_files)} clips ...")

    rows = []
    fieldnames = [
        "clip", "T", "H", "W",
        "bpp_raw", "bpp_diff", "bpp_reduction_pct",
        "entropy_top_raw", "entropy_top_diff",
        "entropy_bot_raw", "entropy_bot_diff",
        "used_top_raw", "used_top_diff",
        "used_bot_raw", "used_bot_diff",
        "mean_abs_raw", "mean_abs_diff", "magnitude_ratio",
        "psnr_raw", "psnr_diff_recon",
    ]

    with open(RESULTS_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(tqdm(clip_files, desc="Frame-diff probe")):
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and "tensor" in raw:
                raw = raw["tensor"]
            clip = raw.float()
            if clip.max() > 1.0:
                clip = (clip / 255.0).clamp(0, 1)
            C, T, H, W = clip.shape

            # ── baseline: raw clip ─────────────────────────────────────────
            out_raw   = encode_decode(ae, clip)
            recon_raw = out_raw["recon"].squeeze(0).cpu()
            bpp_raw   = entropy_bpp(out_raw, T, H, W)
            psnr_raw  = psnr(recon_raw, clip)

            # ── frame-diff clip ────────────────────────────────────────────
            # frame 0 = frame 0 (I-frame)
            # frames 1..T-1 = (frame_t - frame_{t-1}) * 0.5 + 0.5
            diff_clip = clip.clone()
            raw_diffs = []
            for t in range(1, T):
                diff = clip[:, t] - clip[:, t-1]     # [-1, 1]
                raw_diffs.append(diff.abs().mean().item())
                diff_clip[:, t] = diff * 0.5 + 0.5   # normalize to [0, 1]

            out_diff  = encode_decode(ae, diff_clip)
            recon_diff = out_diff["recon"].squeeze(0).cpu()
            bpp_diff   = entropy_bpp(out_diff, T, H, W)

            # ── reconstruct from diff ──────────────────────────────────────
            final = clip.clone()
            for t in range(1, T):
                decoded_diff = recon_diff[:, t] * 2.0 - 1.0   # back to [-1, 1]
                final[:, t]  = (clip[:, t-1] + decoded_diff).clamp(0, 1)
            psnr_diff_recon = psnr(final, clip)

            # ── stats ──────────────────────────────────────────────────────
            mean_abs_raw  = clip[:, 1:].abs().mean().item()
            mean_abs_diff = float(np.mean(raw_diffs))
            mag_ratio     = mean_abs_diff / (mean_abs_raw + 1e-8)
            reduction     = (bpp_raw - bpp_diff) / (bpp_raw + 1e-8) * 100

            row = dict(
                clip=path.name, T=T, H=H, W=W,
                bpp_raw=round(bpp_raw, 5),
                bpp_diff=round(bpp_diff, 5),
                bpp_reduction_pct=round(reduction, 2),
                entropy_top_raw=round(out_raw["entropy_top"], 4),
                entropy_top_diff=round(out_diff["entropy_top"], 4),
                entropy_bot_raw=round(out_raw["entropy_bottom"], 4),
                entropy_bot_diff=round(out_diff["entropy_bottom"], 4),
                used_top_raw=out_raw["used_top"],
                used_top_diff=out_diff["used_top"],
                used_bot_raw=out_raw["used_bottom"],
                used_bot_diff=out_diff["used_bottom"],
                mean_abs_raw=round(mean_abs_raw, 5),
                mean_abs_diff=round(mean_abs_diff, 5),
                magnitude_ratio=round(mag_ratio, 4),
                psnr_raw=round(psnr_raw, 3),
                psnr_diff_recon=round(psnr_diff_recon, 3),
            )
            writer.writerow(row)
            rows.append(row)

            if i < N_SAMPLE_CLIPS:
                save_sample(i, clip, diff_clip, recon_raw, final)

    def avg(k): return float(np.mean([r[k] for r in rows]))

    summary = {
        "n_clips"              : len(rows),
        "mean_bpp_raw"         : round(avg("bpp_raw"),            5),
        "mean_bpp_diff"        : round(avg("bpp_diff"),           5),
        "mean_bpp_reduction"   : round(avg("bpp_reduction_pct"),  2),
        "mean_magnitude_ratio" : round(avg("magnitude_ratio"),    4),
        "mean_psnr_raw"        : round(avg("psnr_raw"),           3),
        "mean_psnr_diff_recon" : round(avg("psnr_diff_recon"),    3),
        "mean_entropy_top_raw" : round(avg("entropy_top_raw"),    4),
        "mean_entropy_top_diff": round(avg("entropy_top_diff"),   4),
        "mean_entropy_bot_raw" : round(avg("entropy_bot_raw"),    4),
        "mean_entropy_bot_diff": round(avg("entropy_bot_diff"),   4),
        "mean_used_top_raw"    : round(avg("used_top_raw"),       1),
        "mean_used_top_diff"   : round(avg("used_top_diff"),      1),
        "mean_used_bot_raw"    : round(avg("used_bot_raw"),       1),
        "mean_used_bot_diff"   : round(avg("used_bot_diff"),      1),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── quick BPP plot ─────────────────────────────────────────────────────
    bpp_raw_list  = [r["bpp_raw"]  for r in rows]
    bpp_diff_list = [r["bpp_diff"] for r in rows]
    reductions    = [r["bpp_reduction_pct"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Frame Diff vs Raw BPP", fontsize=13)
    axes[0].scatter(bpp_raw_list, bpp_diff_list, alpha=0.5, s=20)
    lim = max(max(bpp_raw_list), max(bpp_diff_list)) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", label="y=x")
    axes[0].set_xlabel("BPP raw"); axes[0].set_ylabel("BPP frame-diff")
    axes[0].set_title("Per-clip scatter"); axes[0].legend()
    axes[1].hist(reductions, bins=30, color="mediumseagreen", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("BPP reduction (%)"); axes[1].set_ylabel("Clips")
    axes[1].set_title(f"Mean reduction: {summary['mean_bpp_reduction']:.1f}%")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "bpp_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 55)
    print("FRAME DIFF PROBE SUMMARY")
    print("=" * 55)
    print(f"  BPP raw              : {summary['mean_bpp_raw']:.5f}")
    print(f"  BPP frame-diff       : {summary['mean_bpp_diff']:.5f}")
    print(f"  BPP reduction        : {summary['mean_bpp_reduction']:.1f}%")
    print(f"  Magnitude ratio      : {summary['mean_magnitude_ratio']:.3f}  (< 1.0 = diffs smaller)")
    print(f"  PSNR raw             : {summary['mean_psnr_raw']:.2f} dB")
    print(f"  PSNR diff recon      : {summary['mean_psnr_diff_recon']:.2f} dB")
    print(f"  Used codes top raw/diff : {summary['mean_used_top_raw']:.0f} / {summary['mean_used_top_diff']:.0f}")
    print(f"  Used codes bot raw/diff : {summary['mean_used_bot_raw']:.0f} / {summary['mean_used_bot_diff']:.0f}")
    print("=" * 55)

    reduction = summary["mean_bpp_reduction"]
    print("\nINTERPRETATION:")
    if reduction >= 20:
        print(f"  ✓ STRONG ({reduction:.1f}% reduction with naive frame diff).")
        print("    Temporal redundancy confirmed. Learned motion VQ will do even better.")
        print("    → Proceed to Phase 2 (dual-codebook architecture).")
    elif reduction >= 5:
        print(f"  ~ MODERATE ({reduction:.1f}% reduction).")
        print("    Temporal redundancy present but limited (UCF101 has fast motion).")
        print("    Learned motion VQ should still improve over raw frame coding.")
    else:
        print(f"  ✗ WEAK ({reduction:.1f}%). UCF101 action clips have high inter-frame change.")
        print("    Temporal redundancy is limited at 64x64 resolution.")
        print("    Consider: (a) I+P frame interleaving rather than all-P, or")
        print("              (b) Group of Pictures (GOP) design with keyframes every N frames.")

    log("Done.")

if __name__ == "__main__":
    main()
