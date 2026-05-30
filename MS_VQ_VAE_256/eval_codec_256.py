"""
eval_codec_256.py — Evaluate trained 256×256 MS-VQ-VAE-3 codec

For each K value, loads the AE + 3 priors and computes per-clip:
  BPP   — bits per pixel from learned priors (top + mid + bottom)
  PSNR  — peak signal-to-noise ratio (dB)
  SSIM  — structural similarity index
  LPIPS — learned perceptual image patch similarity

Results saved to: outputs_msvqvae_256/eval_results.csv

Usage:
    conda activate vae_env
    cd MS_VQ_VAE_256
    python eval_codec_256.py
"""

import os
import csv
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

import sys
sys.path.insert(0, "../MS_VQ_VAE_RD")
from losses import bpp_from_bits
from dataset import VideoClipTensorDataset

from models_3level import MSVQVAE3Video
from priors_3level import (
    TopPrior3D,
    MidPriorConditional3D,
    BottomPriorConditional3D,
    categorical_bits_from_logits,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TEST_DIR   = "../dataset_256/test/tensors"
OUT_DIR    = "./outputs_msvqvae_256"
RESULTS_CSV = os.path.join(OUT_DIR, "eval_results.csv")

K_VALUES   = [128, 256, 512, 1024]
EXPECTED_T = 32
BATCH_SIZE = 1      # evaluate one clip at a time for accurate per-clip stats
MAX_CLIPS  = 500    # match the 500-clip test set used at 64×64 and 128×128
USE_AMP    = True

# Model dims (must match train_256.py)
TOP_DIM    = 256
MID_DIM    = 192
BOTTOM_DIM = 128

TOP_PRIOR_EMB      = 64;  TOP_PRIOR_HIDDEN   = 128;  TOP_PRIOR_RESBLOCKS  = 6
MID_PRIOR_EMB      = 64;  MID_PRIOR_HIDDEN   = 192;  MID_PRIOR_RESBLOCKS  = 8
BOT_PRIOR_EMB      = 64;  BOT_PRIOR_HIDDEN   = 192;  BOT_PRIOR_RESBLOCKS  = 8
# ─────────────────────────────────────────────────────────────────────────────


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def frames_to_2d(x: torch.Tensor) -> torch.Tensor:
    """(B,3,T,H,W) → (B*T,3,H,W) for 2D metrics."""
    B, C, T, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


@torch.no_grad()
def evaluate_k(K: int, device: torch.device, ssim_fn, lpips_fn):
    ae_ckpt     = os.path.join(OUT_DIR, f"ae_K{K}_epoch020.pt")
    prior_ckpt  = os.path.join(OUT_DIR, f"priors_K{K}_epoch015.pt")

    if not os.path.exists(ae_ckpt):
        print(f"[SKIP] AE checkpoint not found: {ae_ckpt}")
        return None
    if not os.path.exists(prior_ckpt):
        print(f"[SKIP] Prior checkpoint not found: {prior_ckpt}")
        return None

    # Load AE
    ae = MSVQVAE3Video(
        top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_mid=K, K_bottom=K,
        use_ema=True, use_vgg=False,
    ).to(device).eval()
    ae_state = torch.load(ae_ckpt, map_location=device, weights_only=False)
    ae.load_state_dict(ae_state["model"], strict=False)

    # Load priors
    prior_state = torch.load(prior_ckpt, map_location=device, weights_only=True)
    top_prior = TopPrior3D(K, emb_dim=TOP_PRIOR_EMB, hidden=TOP_PRIOR_HIDDEN,
                            n_res=TOP_PRIOR_RESBLOCKS).to(device).eval()
    mid_prior = MidPriorConditional3D(K, cond_channels=TOP_DIM, emb_dim=MID_PRIOR_EMB,
                                       hidden=MID_PRIOR_HIDDEN, n_res=MID_PRIOR_RESBLOCKS).to(device).eval()
    bot_prior = BottomPriorConditional3D(K, cond_channels=256, emb_dim=BOT_PRIOR_EMB,
                                          hidden=BOT_PRIOR_HIDDEN, n_res=BOT_PRIOR_RESBLOCKS).to(device).eval()
    top_prior.load_state_dict(prior_state["top_prior"])
    mid_prior.load_state_dict(prior_state["mid_prior"])
    bot_prior.load_state_dict(prior_state["bot_prior"])

    test_ds     = VideoClipTensorDataset(TEST_DIR, subset_fraction=1.0, expected_T=EXPECTED_T)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    total_bpp   = 0.0
    total_psnr  = 0.0
    total_ssim  = 0.0
    total_lpips = 0.0
    n_clips     = 0

    for batch in test_loader:
        if n_clips >= MAX_CLIPS:
            break
        x = batch.to(device)
        B, C, T, H, W = x.shape
        pixels = B * T * H * W

        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out      = ae(x)
            recon    = out["recon"]

            idx_t    = out["idx_top"]
            idx_m    = out["idx_mid"]
            idx_b    = out["idx_bottom"]
            cond_mid = out["z_top_up"]
            cond_bot = out["z_mid_up"]

            logits_t = top_prior(idx_t)
            logits_m = mid_prior(idx_m, cond_mid)
            logits_b = bot_prior(idx_b, cond_bot)

            bits_t = categorical_bits_from_logits(logits_t, idx_t)
            bits_m = categorical_bits_from_logits(logits_m, idx_m)
            bits_b = categorical_bits_from_logits(logits_b, idx_b)
            bits   = bits_t + bits_m + bits_b
            bpp    = bpp_from_bits(bits, x)

        # 2D metrics (frame-level)
        x_2d     = frames_to_2d(x.float().clamp(0, 1))
        recon_2d = frames_to_2d(recon.float().clamp(0, 1))

        clip_psnr  = psnr(recon_2d, x_2d)
        clip_ssim  = ssim_fn(recon_2d, x_2d).item()
        clip_lpips = lpips_fn(recon_2d * 2 - 1, x_2d * 2 - 1).item()  # LPIPS expects [-1,1]
        clip_bpp   = bpp.item()

        total_bpp   += clip_bpp
        total_psnr  += clip_psnr
        total_ssim  += clip_ssim
        total_lpips += clip_lpips
        n_clips     += 1

        if n_clips % 100 == 0:
            print(f"  K={K}  clip {n_clips}  bpp={clip_bpp:.4f}  psnr={clip_psnr:.2f}  "
                  f"ssim={clip_ssim:.4f}  lpips={clip_lpips:.4f}")

    n = max(1, n_clips)
    results = {
        "K":     K,
        "clips": n_clips,
        "BPP":   round(total_bpp  / n, 4),
        "PSNR":  round(total_psnr / n, 2),
        "SSIM":  round(total_ssim / n, 4),
        "LPIPS": round(total_lpips/ n, 4),
    }
    print(f"\n[K={K}] BPP={results['BPP']}  PSNR={results['PSNR']} dB  "
          f"SSIM={results['SSIM']}  LPIPS={results['LPIPS']}  ({n_clips} clips)")
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Test set: {TEST_DIR}")

    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    for K in K_VALUES:
        print(f"\n{'─'*50}")
        print(f"  Evaluating K={K}")
        print(f"{'─'*50}")
        result = evaluate_k(K, device, ssim_fn, lpips_fn)
        if result:
            all_results.append(result)

    # Save CSV
    if all_results:
        fields = ["K", "clips", "BPP", "PSNR", "SSIM", "LPIPS"]
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[SAVED] {RESULTS_CSV}")

        print("\n── Summary ──────────────────────────────────────")
        print(f"{'K':>6}  {'BPP':>7}  {'PSNR':>8}  {'SSIM':>7}  {'LPIPS':>7}")
        for r in all_results:
            print(f"{r['K']:>6}  {r['BPP']:>7.4f}  {r['PSNR']:>7.2f}dB  {r['SSIM']:>7.4f}  {r['LPIPS']:>7.4f}")


if __name__ == "__main__":
    main()
