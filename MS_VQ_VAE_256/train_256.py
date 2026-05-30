"""
train_256.py — Three-level MS-VQ-VAE training at 256×256 resolution

Stages:
  Stage A: Train autoencoder (distortion + VQ + perceptual loss)
  Stage B: Freeze AE, train three entropy priors (top, mid, bottom)

Run modes (set STAGE below):
  "AB" — K sweep: runs Stage A then Stage B for each K in K_SWEEP_VALUES
  "A"  — Stage A only
  "B"  — Stage B only (requires AE checkpoint)

Hardware config (RTX 5080 16GB):
  BATCH_SIZE = 2, GRAD_ACCUM_STEPS = 4  →  effective batch = 8
  If Stage B (prior training) runs OOM: set BATCH_SIZE=1, GRAD_ACCUM_STEPS=8

Latent grids at 256×256, T=32:
  top:    (B, 256, 4,  16, 16) —  1024 symbols/clip
  mid:    (B, 192, 8,  32, 32) —  8192 symbols/clip
  bottom: (B, 128, 16, 64, 64) — 65536 symbols/clip

Checkpoints saved to: outputs_msvqvae_256/
  ae_K{K}_epoch{N}.pt
  priors_K{K}_epoch{N}.pt
"""

import csv
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models_3level import MSVQVAE3Video
from priors_3level import (
    TopPrior3D,
    MidPriorConditional3D,
    BottomPriorConditional3D,
    categorical_bits_from_logits,
)

import sys
sys.path.insert(0, "../MS_VQ_VAE_RD")
from losses import recon_loss_fn, vgg_perceptual_loss, bpp_from_bits
from dataset import VideoClipTensorDataset
from utils import save_ckpt, set_requires_grad


# ============================================================
#                     CSV LOGGER
# ============================================================

class CsvLogger:
    def __init__(self, path: str, fieldnames: list):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        print(f"[LOG] → {path}")

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================
#                  HARD-CODED CONFIG
# ============================================================

STAGE = "AB"   # "A", "B", or "AB"

# Dataset (256×256 preprocessed tensors)
TRAIN_DIR = "../dataset_256/train/tensors"
VAL_DIR   = "../dataset_256/val/tensors"

# Output
OUT_DIR = "./outputs_msvqvae_256"

# Data
EXPECTED_T      = 32
SUBSET_FRACTION = 0.5   # use 50% of clips per run (consistent with 128×128 protocol)

# Hardware (RTX 5080 16GB at 256×256)
# NOTE: If Stage B prior training hits OOM, reduce BATCH_SIZE to 1 and
#       increase GRAD_ACCUM_STEPS to 8 to maintain effective batch = 8.
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 4    # effective batch size = 8

# Training
EPOCHS_STAGE_A = 20
EPOCHS_STAGE_B = 15
LR_AE          = 1e-4
LR_PRIOR       = 2e-4
USE_AMP        = True

# Loss weights
GAMMA_PERCEPTUAL = 0.4
BETA_VQ          = 0.25
RECON_LOSS_MODE  = "l1"
PERCEPTUAL_STRIDE = 4   # compute VGG on every 4th frame

# K sweep — same values as 64×64 and 128×128 for fair comparison
K_SWEEP_VALUES = [128, 256, 512, 1024]

# Model dimensions
TOP_DIM    = 256
MID_DIM    = 192
BOTTOM_DIM = 128

# Prior capacity
TOP_PRIOR_EMB      = 64
TOP_PRIOR_HIDDEN   = 128
TOP_PRIOR_RESBLOCKS = 6

MID_PRIOR_EMB      = 64
MID_PRIOR_HIDDEN   = 192
MID_PRIOR_RESBLOCKS = 8

BOT_PRIOR_EMB      = 64
BOT_PRIOR_HIDDEN   = 192
BOT_PRIOR_RESBLOCKS = 8


# ============================================================
#                  VALIDATION HELPERS
# ============================================================

@torch.no_grad()
def validate_ae(ae, val_loader, device) -> float:
    ae.eval()
    total = 0.0
    for batch in val_loader:
        x = batch.to(device)
        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out   = ae(x)
            rec   = recon_loss_fn(out["recon"], x, mode=RECON_LOSS_MODE)
            perc  = (
                vgg_perceptual_loss(ae.vgg, x, out["recon"], stride=PERCEPTUAL_STRIDE)
                if GAMMA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
            )
            loss  = rec + BETA_VQ * out["vq_loss"] + GAMMA_PERCEPTUAL * perc
        total += float(loss.item())
    return total / max(1, len(val_loader))


@torch.no_grad()
def validate_priors(ae, top_prior, mid_prior, bot_prior, val_loader, device) -> Tuple[float, float]:
    ae.eval(); top_prior.eval(); mid_prior.eval(); bot_prior.eval()
    total_bits = 0.0
    total_bpp  = 0.0
    for batch in val_loader:
        x = batch.to(device)
        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out      = ae(x)
            idx_t    = out["idx_top"]
            idx_m    = out["idx_mid"]
            idx_b    = out["idx_bottom"]
            cond_mid = out["z_top_up"].detach()
            cond_bot = out["z_mid_up"].detach()

            logits_t = top_prior(idx_t)
            logits_m = mid_prior(idx_m, cond_mid)
            logits_b = bot_prior(idx_b, cond_bot)

            bits_t = categorical_bits_from_logits(logits_t, idx_t)
            bits_m = categorical_bits_from_logits(logits_m, idx_m)
            bits_b = categorical_bits_from_logits(logits_b, idx_b)
            bits   = bits_t + bits_m + bits_b
            bpp    = bpp_from_bits(bits, x)

        total_bits += float(bits.item())
        total_bpp  += float(bpp.item())
    n = max(1, len(val_loader))
    return total_bits / n, total_bpp / n


# ============================================================
#                     STAGE A — AUTOENCODER
# ============================================================

def run_stage_a(K: int, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  STAGE A — Autoencoder  |  K={K}  |  256×256")
    print(f"{'='*60}")

    train_ds = VideoClipTensorDataset(TRAIN_DIR, subset_fraction=SUBSET_FRACTION, expected_T=EXPECTED_T)
    val_ds   = VideoClipTensorDataset(VAL_DIR,   subset_fraction=1.0,             expected_T=EXPECTED_T)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    ae = MSVQVAE3Video(
        top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_mid=K, K_bottom=K,
        use_ema=True, use_vgg=True,
    ).to(device)

    opt    = torch.optim.Adam(ae.parameters(), lr=LR_AE, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    log_path = os.path.join(OUT_DIR, f"log_stageA_K{K}.csv")
    logger   = CsvLogger(log_path, ["epoch", "step", "loss_recon", "loss_vq", "loss_perc", "loss_total",
                                     "used_top", "used_mid", "used_bottom", "val_loss", "elapsed_s"])
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    for epoch in range(1, EPOCHS_STAGE_A + 1):
        ae.train()
        opt.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)
            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                out  = ae(x)
                rec  = recon_loss_fn(out["recon"], x, mode=RECON_LOSS_MODE)
                perc = (
                    vgg_perceptual_loss(ae.vgg, x, out["recon"], stride=PERCEPTUAL_STRIDE)
                    if GAMMA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
                )
                loss = (rec + BETA_VQ * out["vq_loss"] + GAMMA_PERCEPTUAL * perc) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [A|K={K}] Ep {epoch}/{EPOCHS_STAGE_A}  Step {step}  "
                      f"rec={rec.item():.4f}  vq={out['vq_loss'].item():.4f}  "
                      f"perc={perc.item():.4f}  "
                      f"used=({out['used_top']},{out['used_mid']},{out['used_bottom']})  "
                      f"t={elapsed:.0f}s")
                logger.log(epoch=epoch, step=step,
                           loss_recon=rec.item(), loss_vq=out["vq_loss"].item(),
                           loss_perc=perc.item(), loss_total=loss.item() * GRAD_ACCUM_STEPS,
                           used_top=out["used_top"], used_mid=out["used_mid"],
                           used_bottom=out["used_bottom"], elapsed_s=elapsed)

        val_loss = validate_ae(ae, val_loader, device)
        print(f"  [A|K={K}] Ep {epoch} val_loss={val_loss:.4f}")
        logger.log(epoch=epoch, step="val", val_loss=val_loss)

        ckpt_path = os.path.join(OUT_DIR, f"ae_K{K}_epoch{epoch:03d}.pt")
        save_ckpt(ckpt_path, model=ae.state_dict(), opt=opt.state_dict(), epoch=epoch)

    logger.close()
    print(f"[Stage A done] K={K}")
    return ae


# ============================================================
#                     STAGE B — PRIORS
# ============================================================

def run_stage_b(K: int, ae: MSVQVAE3Video, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  STAGE B — Priors  |  K={K}  |  256×256")
    print(f"{'='*60}")

    train_ds = VideoClipTensorDataset(TRAIN_DIR, subset_fraction=SUBSET_FRACTION, expected_T=EXPECTED_T)
    val_ds   = VideoClipTensorDataset(VAL_DIR,   subset_fraction=1.0,             expected_T=EXPECTED_T)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    set_requires_grad(ae, False)
    ae.eval()

    top_prior = TopPrior3D(K,
                           emb_dim=TOP_PRIOR_EMB,
                           hidden=TOP_PRIOR_HIDDEN,
                           n_res=TOP_PRIOR_RESBLOCKS).to(device)
    mid_prior = MidPriorConditional3D(K, cond_channels=TOP_DIM,
                                      emb_dim=MID_PRIOR_EMB,
                                      hidden=MID_PRIOR_HIDDEN,
                                      n_res=MID_PRIOR_RESBLOCKS).to(device)
    bot_prior = BottomPriorConditional3D(K, cond_channels=256,   # stage1 decoder output channels
                                         emb_dim=BOT_PRIOR_EMB,
                                         hidden=BOT_PRIOR_HIDDEN,
                                         n_res=BOT_PRIOR_RESBLOCKS).to(device)

    all_params = list(top_prior.parameters()) + list(mid_prior.parameters()) + list(bot_prior.parameters())
    opt    = torch.optim.Adam(all_params, lr=LR_PRIOR, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    log_path = os.path.join(OUT_DIR, f"log_stageB_K{K}.csv")
    logger   = CsvLogger(log_path, ["epoch", "step", "bits_top", "bits_mid", "bits_bot",
                                     "bits_total", "bpp", "val_bits", "val_bpp", "elapsed_s"])
    t0 = time.time()

    for epoch in range(1, EPOCHS_STAGE_B + 1):
        top_prior.train(); mid_prior.train(); bot_prior.train()
        opt.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)
            B, C, T, H, W = x.shape
            pixels = B * T * H * W

            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                with torch.no_grad():
                    out = ae(x)
                idx_t    = out["idx_top"]
                idx_m    = out["idx_mid"]
                idx_b    = out["idx_bottom"]
                cond_mid = out["z_top_up"].detach()
                cond_bot = out["z_mid_up"].detach()

                logits_t = top_prior(idx_t)
                logits_m = mid_prior(idx_m, cond_mid)
                logits_b = bot_prior(idx_b, cond_bot)

                bits_t = categorical_bits_from_logits(logits_t, idx_t)
                bits_m = categorical_bits_from_logits(logits_m, idx_m)
                bits_b = categorical_bits_from_logits(logits_b, idx_b)
                bits   = bits_t + bits_m + bits_b
                bpp    = bpp_from_bits(bits, x)
                loss   = bits / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [B|K={K}] Ep {epoch}/{EPOCHS_STAGE_B}  Step {step}  "
                      f"bpp={bpp.item():.4f}  "
                      f"bits=({bits_t.item():.0f},{bits_m.item():.0f},{bits_b.item():.0f})  "
                      f"t={elapsed:.0f}s")
                logger.log(epoch=epoch, step=step,
                           bits_top=bits_t.item(), bits_mid=bits_m.item(), bits_bot=bits_b.item(),
                           bits_total=bits.item(), bpp=bpp.item(), elapsed_s=elapsed)

        val_bits, val_bpp = validate_priors(ae, top_prior, mid_prior, bot_prior, val_loader, device)
        print(f"  [B|K={K}] Ep {epoch} val_bpp={val_bpp:.4f}")
        logger.log(epoch=epoch, step="val", val_bits=val_bits, val_bpp=val_bpp)

        ckpt_path = os.path.join(OUT_DIR, f"priors_K{K}_epoch{epoch:03d}.pt")
        torch.save({
            "K": K,
            "epoch": epoch,
            "top_prior": top_prior.state_dict(),
            "mid_prior": mid_prior.state_dict(),
            "bot_prior": bot_prior.state_dict(),
            "opt": opt.state_dict(),
        }, ckpt_path)
        print(f"  [CKPT] {ckpt_path}")

    logger.close()
    set_requires_grad(ae, True)
    print(f"[Stage B done] K={K}")
    return top_prior, mid_prior, bot_prior


# ============================================================
#                         MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Stage: {STAGE}  |  K sweep: {K_SWEEP_VALUES}")
    print(f"[INFO] Batch: {BATCH_SIZE}  GradAccum: {GRAD_ACCUM_STEPS}  EffBatch: {BATCH_SIZE*GRAD_ACCUM_STEPS}")

    os.makedirs(OUT_DIR, exist_ok=True)

    if STAGE == "AB":
        for K in K_SWEEP_VALUES:
            ae = run_stage_a(K, device)
            run_stage_b(K, ae, device)
            del ae
            torch.cuda.empty_cache()

    elif STAGE == "A":
        for K in K_SWEEP_VALUES:
            run_stage_a(K, device)

    elif STAGE == "B":
        for K in K_SWEEP_VALUES:
            ae_ckpt = os.path.join(OUT_DIR, f"ae_K{K}_epoch{EPOCHS_STAGE_A:03d}.pt")
            assert os.path.exists(ae_ckpt), f"AE checkpoint not found: {ae_ckpt}"
            ae = MSVQVAE3Video(
                top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
                K_top=K, K_mid=K, K_bottom=K,
                use_ema=True, use_vgg=True,
            ).to(device)
            ckpt = torch.load(ae_ckpt, map_location=device, weights_only=True)
            ae.load_state_dict(ckpt["model"])
            run_stage_b(K, ae, device)
            del ae
            torch.cuda.empty_cache()

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
