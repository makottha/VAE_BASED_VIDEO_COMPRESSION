"""
eval_codec_128_T16.py — Evaluate T=16 temporal ablation model

Loads the T=16 / K=512 checkpoint from outputs_msvqvae_128_T16/ and
computes BPP/PSNR/SSIM/LPIPS on the test set using T=16 clips sliced
from the existing T=32 test tensors.

BPP is computed from learned priors over the T=16 latent grids:
  top:    (1,  8,  8) =    64 symbols/clip
  mid:    (2, 16, 16) =   512 symbols/clip
  bottom: (4, 32, 32) = 4,096 symbols/clip
BPP denominator = B * T_CLIP * H * W  (T=16, H=W=128)

Comparison target (T=32 K=512):
  BPP=0.0710  PSNR=25.72  SSIM=0.8393  LPIPS=0.1245

Results saved to: outputs_msvqvae_128_T16/eval_results_T16.csv

Usage:
  conda activate vae_env
  cd MS_VQ_VAE_128
  python eval_codec_128_T16.py
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

from models_3level import MSVQVAE3Video
from priors_3level import (
    TopPrior3D, MidPriorConditional3D, BottomPriorConditional3D,
    categorical_bits_from_logits,
)
from train_128_T16 import VideoClipT16Dataset

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TEST_DIR    = "../dataset_128/test/tensors"
OUT_DIR     = "./outputs_msvqvae_128_T16"
RESULTS_CSV = os.path.join(OUT_DIR, "eval_results_T16.csv")

K       = 512
T_CLIP  = 16
USE_AMP = True

TOP_DIM    = 256
MID_DIM    = 192
BOTTOM_DIM = 128

TOP_PRIOR_EMB = 64;  TOP_PRIOR_HIDDEN = 128;  TOP_PRIOR_RESBLOCKS = 6
MID_PRIOR_EMB = 64;  MID_PRIOR_HIDDEN = 192;  MID_PRIOR_RESBLOCKS = 8
BOT_PRIOR_EMB = 64;  BOT_PRIOR_HIDDEN = 192;  BOT_PRIOR_RESBLOCKS = 8
# ─────────────────────────────────────────────────────────────────────────────


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def frames_to_2d(x: torch.Tensor) -> torch.Tensor:
    """(B,3,T,H,W) -> (B*T,3,H,W) for 2D metrics."""
    B, C, T, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] K={K}  T={T_CLIP}  Test set: {TEST_DIR}")

    ae_ckpt    = os.path.join(OUT_DIR, f"ae_K{K}_epoch020.pt")
    prior_ckpt = os.path.join(OUT_DIR, f"priors_K{K}_epoch015.pt")

    if not os.path.exists(ae_ckpt):
        print(f"[ERROR] AE checkpoint not found: {ae_ckpt}")
        return
    if not os.path.exists(prior_ckpt):
        print(f"[ERROR] Prior checkpoint not found: {prior_ckpt}")
        return

    # Load AE
    ae = MSVQVAE3Video(
        top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_mid=K, K_bottom=K, use_ema=True, use_vgg=False,
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

    # Dataset
    test_ds     = VideoClipT16Dataset(TEST_DIR, t_clip=T_CLIP,
                                       subset_fraction=1.0, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    total_bpp = 0.0; total_psnr = 0.0
    total_ssim = 0.0; total_lpips = 0.0
    n_clips = 0

    for batch in test_loader:
        x = batch.to(device)

        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out      = ae(x)
            recon    = out["recon"]
            idx_t    = out["idx_top"];    cond_mid = out["z_top_up"]
            idx_m    = out["idx_mid"];    cond_bot = out["z_mid_up"]
            idx_b    = out["idx_bottom"]

            logits_t = top_prior(idx_t)
            logits_m = mid_prior(idx_m, cond_mid)
            logits_b = bot_prior(idx_b, cond_bot)

            bits_t = categorical_bits_from_logits(logits_t, idx_t)
            bits_m = categorical_bits_from_logits(logits_m, idx_m)
            bits_b = categorical_bits_from_logits(logits_b, idx_b)
            bits   = bits_t + bits_m + bits_b
            bpp    = bpp_from_bits(bits, x)

        x_2d     = frames_to_2d(x.float().clamp(0, 1))
        recon_2d = frames_to_2d(recon.float().clamp(0, 1))

        clip_psnr  = psnr(recon_2d, x_2d)
        clip_ssim  = ssim_fn(recon_2d, x_2d).item()
        clip_lpips = lpips_fn(recon_2d * 2 - 1, x_2d * 2 - 1).item()
        clip_bpp   = bpp.item()

        total_bpp   += clip_bpp
        total_psnr  += clip_psnr
        total_ssim  += clip_ssim
        total_lpips += clip_lpips
        n_clips     += 1

        if n_clips % 200 == 0:
            print(f"  clip {n_clips}  bpp={clip_bpp:.4f}  psnr={clip_psnr:.2f}  "
                  f"ssim={clip_ssim:.4f}  lpips={clip_lpips:.4f}")

    n = max(1, n_clips)
    result = {
        "K":     K,
        "T":     T_CLIP,
        "clips": n_clips,
        "BPP":   round(total_bpp   / n, 4),
        "PSNR":  round(total_psnr  / n, 2),
        "SSIM":  round(total_ssim  / n, 4),
        "LPIPS": round(total_lpips / n, 4),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "T", "clips", "BPP", "PSNR", "SSIM", "LPIPS"])
        writer.writeheader()
        writer.writerow(result)

    print(f"\n[RESULT] K={K} T={T_CLIP}:")
    print(f"  BPP  = {result['BPP']}")
    print(f"  PSNR = {result['PSNR']} dB")
    print(f"  SSIM = {result['SSIM']}")
    print(f"  LPIPS= {result['LPIPS']}")
    print(f"\n[BASELINE] K={K} T=32:")
    print(f"  BPP  = 0.0710")
    print(f"  PSNR = 25.72 dB")
    print(f"  SSIM = 0.8393")
    print(f"  LPIPS= 0.1245")
    print(f"\n[SAVED] {RESULTS_CSV}")


if __name__ == "__main__":
    main()
