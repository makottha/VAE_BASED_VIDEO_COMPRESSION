"""
train.py — MS-VQ-VAE2 + Entropy Priors (Rate Modeling) for Video Compression (64x64, 32 frames)

This file provides an end-to-end training script with three stages:

  Stage A: Train the MS-VQ-VAE2 autoencoder (Distortion + VQ + Perceptual)
  Stage B: Freeze AE, train entropy priors for top/bottom indices (Rate only)
  Stage C: Fine-tune end-to-end with Rate–Distortion (RD) objective

The goal is to extend your ISVC baseline (hierarchical VQ-VAE2 + perceptual loss) into
a codec-complete system by learning explicit probability models over discrete indices.

------------------------------------------------------------------------------------
How to run (no CLI flags required; defaults are hard-coded):
------------------------------------------------------------------------------------

1) Stage A (Autoencoder):
    python train.py

2) Stage B (Priors):
    - Set STAGE = "B" below
    - Set AE_CKPT_PATH to your best AE checkpoint
    python train.py

3) Stage C (RD Fine-tuning):
    - Set STAGE = "C" below
    - Set AE_CKPT_PATH and PRIOR_CKPT_PATH
    python train.py

------------------------------------------------------------------------------------
Notes:
- Expects dataset .pt tensors shaped (3, T, H, W), e.g. (3, 32, 64, 64).
- Uses AMP by default if CUDA is available (can disable via USE_AMP).
- Uses expected-bits rate from learned priors, not fixed-length log2(K).
------------------------------------------------------------------------------------
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Import your local modules (from the from-scratch package)
# Make sure these files exist in the same folder:
#   models.py, priors.py, losses.py, dataset.py, utils.py
from models import MSVQVAE2Video
from priors import TopPrior3D, BottomPriorConditional3D, categorical_bits_from_logits
from losses import recon_loss_fn, vgg_perceptual_loss, bpp_from_bits
from dataset import VideoClipTensorDataset
from utils import save_ckpt, set_requires_grad


# ============================================================
#                       CSV LOGGER
# ============================================================

class CsvLogger:
    """
    Writes training metrics to a CSV file so nothing is lost if the terminal
    is closed. One row per call to .log(). File is flushed after every write
    so partial runs are always recoverable.

    Usage:
        logger = CsvLogger(path, fieldnames=["epoch", "step", "loss", ...])
        logger.log(epoch=1, step=50, loss=1.42, ...)
    """

    def __init__(self, path: str, fieldnames: list):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        print(f"[LOG] Writing training log → {path}")

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()   # survive crashes / Ctrl-C

    def close(self):
        self._file.close()


# ============================================================
#                    HARD-CODED DEFAULTS
# ============================================================

# Choose one: "A", "B", "C", or "AB" (K sweep — runs Stage A + B for each K in K_SWEEP_VALUES)
STAGE = "AB"
AE_CKPT_PATH = "./outputs_msvqvae_rd/ae_epoch050.pt"
PRIOR_CKPT_PATH = "./outputs_msvqvae_rd/priors_epoch020.pt"

# Dataset locations (update these paths to your machine)
TRAIN_DIR = "../dataset_64/train/tensors"
VAL_DIR   = "../dataset_64/val/tensors"

# Output folder for checkpoints/logging
OUT_DIR = "./outputs_msvqvae_rd"

# Data settings
EXPECTED_T = 32
SUBSET_FRACTION = 0.5  # e.g. 0.5 for faster experiments

# Training settings
BATCH_SIZE = 8
EPOCHS_STAGE_A = 20
EPOCHS_STAGE_B = 15
EPOCHS_STAGE_C = 20

LR_AE = 1e-4
LR_PRIOR = 2e-4
LR_FT = 5e-5

USE_AMP = True  # AMP helps on RTX 4060; set False if debugging

# Loss weights
GAMMA_PERCEPTUAL = 0.4
BETA_VQ = 0.25
LAMBDA_RD = 0.002  # single-run RD trade-off; used when SWEEP_LAMBDA_RD=False

# --- λ sweep for full RD curve (set SWEEP_LAMBDA_RD=True to enable) ---
# Each value produces a separate Stage C checkpoint; plot all on the same RD curve.
# Rule of thumb: smaller λ → higher quality / higher bitrate point on the curve.
SWEEP_LAMBDA_RD = False
LAMBDA_RD_VALUES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

# Soft-rate temperature for the differentiable rate proxy in Stage C.
# Controls how "soft" the encoder-to-codebook assignment distribution is.
# Lower → harder (near-zero gradient for encoder); Higher → softer (more gradient).
# Default 1.0 works well; increase to 5.0–10.0 if encoder is not learning rate control.
SOFT_TEMP = 10.0

RECON_LOSS_MODE = "l1"   # "l1" or "mse"
PERCEPTUAL_STRIDE = 4     # compute VGG loss on every 4th frame (speed)

# --- K sweep (STAGE="AB") — produces one RD curve point per K value ---
# K=1024 is already trained. Only 128, 256, 512 need to run.
# Each K trains Stage A (50 epochs) then Stage B (30 epochs) from scratch.
# Checkpoints saved as: ae_K{K}_epoch050.pt, priors_K{K}_epoch030.pt
# Estimated time: ~3.5h per K value (Stage A ~2.5h + Stage B ~1h) = ~10.5h total
K_SWEEP_VALUES = [1024]  # retrain K=1024 with same config as K=128/256/512 (EMA, 20/15 epochs, 0.5 subset)

# Model settings (align with your paper; adjust if needed)
K_TOP = 1024
K_BOTTOM = 1024
TOP_DIM = 256
BOTTOM_DIM = 128

# Prior settings (trade-off quality vs speed)
TOP_PRIOR_EMB = 64
TOP_PRIOR_HIDDEN = 128
TOP_PRIOR_RESBLOCKS = 6

BOT_PRIOR_EMB = 64
BOT_PRIOR_HIDDEN = 192
BOT_PRIOR_RESBLOCKS = 8


# ============================================================
#                   VALIDATION HELPERS
# ============================================================

@torch.no_grad()
def validate_autoencoder(
    ae: MSVQVAE2Video,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Validate autoencoder using the same objective as Stage A:
      loss = recon + beta*vq + gamma*perceptual

    Returns:
      avg_loss (float)
    """
    ae.eval()
    total = 0.0

    for batch in val_loader:
        x = batch.to(device)  # (B,3,T,H,W)
        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out = ae(x)
            recon = out["recon"]
            vq_loss = out["vq_loss"]

            rec = recon_loss_fn(recon, x, mode=RECON_LOSS_MODE)
            perc = (
                vgg_perceptual_loss(ae.vgg, x, recon, stride=PERCEPTUAL_STRIDE)
                if GAMMA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
            )

            loss = rec + (BETA_VQ * vq_loss) + (GAMMA_PERCEPTUAL * perc)

        total += float(loss.item())

    return total / max(1, len(val_loader))


@torch.no_grad()
def validate_priors(
    ae: MSVQVAE2Video,
    top_prior: TopPrior3D,
    bot_prior: BottomPriorConditional3D,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate priors by computing expected bits and bpp:
      bits = -log2 p(z_top) + -log2 p(z_bottom | top_condition)

    Returns:
      (avg_bits_per_batch, avg_bpp_per_batch)
    """
    ae.eval()
    top_prior.eval()
    bot_prior.eval()

    total_bits = 0.0
    total_bpp = 0.0

    for batch in val_loader:
        x = batch.to(device)

        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out = ae(x)
            idx_t = out["idx_top"]
            idx_b = out["idx_bottom"]
            cond = out["z_top_up"].detach()

            logits_t = top_prior(idx_t)
            bits_t = categorical_bits_from_logits(logits_t, idx_t)

            logits_b = bot_prior(idx_b, cond)  # FIX
            bits_b = categorical_bits_from_logits(logits_b, idx_b)

            bits = bits_t + bits_b
            bpp = bpp_from_bits(bits, x)

        total_bits += float(bits.item())
        total_bpp += float(bpp.item())

    n = max(1, len(val_loader))
    return total_bits / n, total_bpp / n


# ============================================================
#                   CHECKPOINT LOADERS
# ============================================================

def try_load_autoencoder(ae: MSVQVAE2Video, path: str) -> bool:
    """
    Loads AE weights if checkpoint exists.
    Returns True if loaded, False otherwise.
    """
    if path and os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu")
        ae.load_state_dict(ckpt["ae"])
        print(f"[LOAD] Autoencoder loaded from: {path}")
        return True
    return False


def try_load_priors(top_prior: TopPrior3D, bot_prior: BottomPriorConditional3D, path: str) -> bool:
    """
    Loads prior weights if checkpoint exists.
    Returns True if loaded, False otherwise.
    """
    if path and os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu")
        top_prior.load_state_dict(ckpt["top_prior"])
        bot_prior.load_state_dict(ckpt["bot_prior"])
        print(f"[LOAD] Priors loaded from: {path}")
        return True
    return False


# ============================================================
#                          TRAINERS
# ============================================================

def train_stage_a(
    ae: MSVQVAE2Video,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger: CsvLogger = None,
    ckpt_prefix: str = "",
) -> None:
    """
    Stage A: Train autoencoder with distortion + VQ + perceptual objective.
    ckpt_prefix: e.g. "K256_" → saves as ae_K256_epoch050.pt
    """
    print("\n====================")
    tag = f"[{ckpt_prefix.rstrip('_')}]" if ckpt_prefix else ""
    print(f" Stage A: Train AE  {tag}")
    print("====================\n")

    set_requires_grad(ae, True)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR_AE)
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    for epoch in range(1, EPOCHS_STAGE_A + 1):
        ae.train()
        t0 = time.time()
        running = 0.0

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)

            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                out = ae(x)
                recon = out["recon"]
                vq_loss = out["vq_loss"]

                rec = recon_loss_fn(recon, x, mode=RECON_LOSS_MODE)
                perc = (
                    vgg_perceptual_loss(ae.vgg, x, recon, stride=PERCEPTUAL_STRIDE)
                    if GAMMA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
                )

                loss = rec + (BETA_VQ * vq_loss) + (GAMMA_PERCEPTUAL * perc)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())

            if step % 50 == 0:
                print(
                    f"[A][E{epoch:03d}][{step:05d}] "
                    f"loss={loss.item():.4f} rec={rec.item():.4f} vq={vq_loss.item():.4f} "
                    f"perc={float(perc.item()):.4f} "
                    f"ent_top={out['entropy_top']:.3f} ent_bot={out['entropy_bottom']:.3f} "
                    f"used_top={out['used_top']} used_bot={out['used_bottom']}"
                )
                if logger:
                    logger.log(stage="A", epoch=epoch, step=step,
                               loss=round(loss.item(), 6),
                               rec=round(rec.item(), 6),
                               vq=round(vq_loss.item(), 6),
                               perc=round(float(perc.item()), 6),
                               entropy_top=round(out['entropy_top'], 4),
                               entropy_bot=round(out['entropy_bottom'], 4),
                               used_top=out['used_top'],
                               used_bot=out['used_bottom'])

        val_loss = validate_autoencoder(ae, val_loader, device)
        print(
            f"[A][E{epoch:03d}] train={running/len(train_loader):.4f} "
            f"val={val_loss:.4f} time={time.time()-t0:.1f}s"
        )
        if logger:
            logger.log(stage="A", epoch=epoch, step="epoch_end",
                       loss=round(running / len(train_loader), 6),
                       val_loss=round(val_loss, 6),
                       elapsed=round(time.time() - t0, 1))

        save_ckpt(
            os.path.join(OUT_DIR, f"ae_{ckpt_prefix}epoch{epoch:03d}.pt"),
            ae=ae.state_dict(),
            epoch=epoch,
            stage="A"
        )


def train_stage_b(
    ae: MSVQVAE2Video,
    top_prior: TopPrior3D,
    bot_prior: BottomPriorConditional3D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger: CsvLogger = None,
    ckpt_prefix: str = "",
) -> None:
    """
    Stage B: Freeze AE, train priors to minimize expected bits.
    ckpt_prefix: e.g. "K256_" → saves as priors_K256_epoch030.pt
    """
    print("\n======================")
    tag = f"[{ckpt_prefix.rstrip('_')}]" if ckpt_prefix else ""
    print(f" Stage B: Train Priors  {tag}")
    print("======================\n")

    # Freeze AE; train priors
    set_requires_grad(ae, False)
    set_requires_grad(top_prior, True)
    set_requires_grad(bot_prior, True)

    optimizer = torch.optim.Adam(
        list(top_prior.parameters()) + list(bot_prior.parameters()),
        lr=LR_PRIOR
    )
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    for epoch in range(1, EPOCHS_STAGE_B + 1):
        top_prior.train()
        bot_prior.train()
        t0 = time.time()
        running_bits = 0.0
        running_bpp = 0.0

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)

            with torch.no_grad():
                out = ae(x)
                idx_t = out["idx_top"]
                idx_b = out["idx_bottom"]
                cond = out["z_top_up"].detach()

            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                logits_t = top_prior(idx_t)
                bits_t = categorical_bits_from_logits(logits_t, idx_t)

                logits_b = bot_prior(idx_b, cond)
                bits_b = categorical_bits_from_logits(logits_b, idx_b)

                bits = bits_t + bits_b
                bpp = bpp_from_bits(bits, x)

                loss = bits  # minimize total expected bits



            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_bits += float(bits.item())
            running_bpp += float(bpp.item())

            if step % 50 == 0:
                ut = torch.unique(idx_t).numel()
                ub = torch.unique(idx_b).numel()
                n_sym = idx_t.numel() + idx_b.numel()
                bps = bits.item() / max(1, n_sym)
                print(
                    f"[B][E{epoch:03d}][{step:05d}] "
                    f"bits={bits.item():.1f} bpp={bpp.item():.4f} "
                    f"(top={bits_t.item():.1f}, bot={bits_b.item():.1f}) "
                    f"unique_top={ut} unique_bot={ub} bits/sym={bps:.4f}"
                )
                if logger:
                    logger.log(stage="B", epoch=epoch, step=step,
                               bits=round(bits.item(), 2),
                               bpp=round(bpp.item(), 6),
                               bits_top=round(bits_t.item(), 2),
                               bits_bot=round(bits_b.item(), 2),
                               bits_per_sym=round(bps, 6),
                               unique_top=ut, unique_bot=ub)

        val_bits, val_bpp = validate_priors(ae, top_prior, bot_prior, val_loader, device)
        print(
            f"[B][E{epoch:03d}] train_bits={running_bits/len(train_loader):.1f} "
            f"train_bpp={running_bpp/len(train_loader):.4f} "
            f"val_bits={val_bits:.1f} val_bpp={val_bpp:.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        if logger:
            logger.log(stage="B", epoch=epoch, step="epoch_end",
                       bits=round(running_bits / len(train_loader), 2),
                       bpp=round(running_bpp / len(train_loader), 6),
                       val_bits=round(val_bits, 2),
                       val_bpp=round(val_bpp, 6),
                       elapsed=round(time.time() - t0, 1))

        save_ckpt(
            os.path.join(OUT_DIR, f"priors_{ckpt_prefix}epoch{epoch:03d}.pt"),
            top_prior=top_prior.state_dict(),
            bot_prior=bot_prior.state_dict(),
            epoch=epoch,
            stage="B"
        )


def train_stage_c(
    ae: MSVQVAE2Video,
    top_prior: TopPrior3D,
    bot_prior: BottomPriorConditional3D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lambda_rd: float = LAMBDA_RD,
    logger: CsvLogger = None,
) -> None:
    """
    Stage C: End-to-end RD fine-tuning.

    Loss = D + beta*VQ + lambda_rd * bpp_soft

    Two rate signals are used:
    - bpp_soft (differentiable): from the encoder's soft-assignment entropy over the
      codebook. This flows gradients back to the encoder, so the encoder is
      actually incentivised to reduce its rate. This fixes the fundamental issue
      where discrete VQ indices break the gradient path from rate to encoder.
    - bpp_hard (non-differentiable, monitoring only): from learned priors via
      categorical cross-entropy — gives the true expected bitrate estimate.
    """
    lam_tag = f"{lambda_rd:.5f}".rstrip("0").rstrip(".")
    print(f"\n=================================")
    print(f" Stage C: RD Fine-tuning λ={lam_tag}")
    print(f"=================================\n")

    set_requires_grad(ae, True)
    set_requires_grad(top_prior, True)
    set_requires_grad(bot_prior, True)

    optimizer = torch.optim.Adam(
        list(ae.parameters()) + list(top_prior.parameters()) + list(bot_prior.parameters()),
        lr=LR_FT
    )
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    all_params = list(ae.parameters()) + list(top_prior.parameters()) + list(bot_prior.parameters())

    for epoch in range(1, EPOCHS_STAGE_C + 1):
        ae.train()
        top_prior.train()
        bot_prior.train()
        t0 = time.time()

        running = {"loss": 0.0, "rec": 0.0, "vq": 0.0, "perc": 0.0,
                   "bpp_soft": 0.0, "bpp_hard": 0.0}

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)

            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                out = ae(x)
                recon = out["recon"]
                vq_loss = out["vq_loss"]
                idx_t = out["idx_top"]
                idx_b = out["idx_bottom"]

                # Distortion
                rec = recon_loss_fn(recon, x, mode=RECON_LOSS_MODE)
                perc = (
                    vgg_perceptual_loss(ae.vgg, x, recon, stride=PERCEPTUAL_STRIDE)
                    if GAMMA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
                )
                D = rec + (GAMMA_PERCEPTUAL * perc)

                # Soft rate (differentiable → gradients flow to AE encoder)
                # This is the key fix: VQ indices are discrete, so prior-based bits
                # cannot back-prop to the encoder. The soft_bits from VQ distances can.
                soft_bits = out["soft_bits_top"] + out["soft_bits_bottom"]
                bpp_soft = bpp_from_bits(soft_bits, x)

                # Hard rate (non-differentiable w.r.t. encoder — for priors & monitoring)
                logits_t = top_prior(idx_t)
                bits_t = categorical_bits_from_logits(logits_t, idx_t)
                cond = out["z_top_up"].detach()
                logits_b = bot_prior(idx_b, cond)
                bits_b = categorical_bits_from_logits(logits_b, idx_b)
                bpp_hard = bpp_from_bits(bits_t + bits_b, x)

                # Full RD objective:
                #   - bpp_soft trains the AE encoder to reduce its coding rate
                #   - bits_t + bits_b trains the priors (as in Stage B)
                loss = D + (BETA_VQ * vq_loss) + (lambda_rd * bpp_soft) + (lambda_rd * bpp_hard)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running["loss"] += float(loss.item())
            running["rec"] += float(rec.item())
            running["vq"] += float(vq_loss.item())
            running["perc"] += float(perc.item())
            running["bpp_soft"] += float(bpp_soft.item())
            running["bpp_hard"] += float(bpp_hard.item())

            if step % 50 == 0:
                print(
                    f"[C][λ={lam_tag}][E{epoch:03d}][{step:05d}] "
                    f"loss={loss.item():.4f} D={D.item():.4f} rec={rec.item():.4f} "
                    f"vq={vq_loss.item():.4f} perc={float(perc.item()):.4f} "
                    f"bpp_soft={bpp_soft.item():.4f} bpp_hard={bpp_hard.item():.4f}"
                )
                if logger:
                    logger.log(stage="C", lambda_rd=lambda_rd, epoch=epoch, step=step,
                               loss=round(loss.item(), 6),
                               D=round(D.item(), 6),
                               rec=round(rec.item(), 6),
                               vq=round(vq_loss.item(), 6),
                               perc=round(float(perc.item()), 6),
                               bpp_soft=round(bpp_soft.item(), 6),
                               bpp_hard=round(bpp_hard.item(), 6))

        # Validation
        val_ae_loss = validate_autoencoder(ae, val_loader, device)
        val_bits, val_bpp = validate_priors(ae, top_prior, bot_prior, val_loader, device)

        ntr = max(1, len(train_loader))
        print(
            f"[C][λ={lam_tag}][E{epoch:03d}] "
            f"train_loss={running['loss']/ntr:.4f} "
            f"train_bpp_soft={running['bpp_soft']/ntr:.4f} "
            f"train_bpp_hard={running['bpp_hard']/ntr:.4f} "
            f"val_ae={val_ae_loss:.4f} val_bpp={val_bpp:.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        if logger:
            logger.log(stage="C", lambda_rd=lambda_rd, epoch=epoch, step="epoch_end",
                       loss=round(running['loss'] / ntr, 6),
                       rec=round(running['rec'] / ntr, 6),
                       vq=round(running['vq'] / ntr, 6),
                       perc=round(running['perc'] / ntr, 6),
                       bpp_soft=round(running['bpp_soft'] / ntr, 6),
                       bpp_hard=round(running['bpp_hard'] / ntr, 6),
                       val_ae_loss=round(val_ae_loss, 6),
                       val_bpp=round(val_bpp, 6),
                       elapsed=round(time.time() - t0, 1))

        save_ckpt(
            os.path.join(OUT_DIR, f"rd_lam{lam_tag}_epoch{epoch:03d}.pt"),
            ae=ae.state_dict(),
            top_prior=top_prior.state_dict(),
            bot_prior=bot_prior.state_dict(),
            epoch=epoch,
            lambda_rd=lambda_rd,
            stage="C"
        )


# ============================================================
#                           MAIN
# ============================================================

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device} | STAGE={STAGE}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV logger — one file per run, timestamped so reruns don't overwrite each other
    log_dir = os.path.join(OUT_DIR, "logs")
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_stage{STAGE}_{ts}.csv")
    logger = CsvLogger(log_path, fieldnames=[
        "stage", "lambda_rd", "epoch", "step",
        "loss", "D", "rec", "vq", "perc",
        "bpp_soft", "bpp_hard", "bits", "bits_top", "bits_bot",
        "bits_per_sym", "unique_top", "unique_bot",
        "val_loss", "val_ae_loss", "val_bits", "val_bpp",
        "entropy_top", "entropy_bot", "used_top", "used_bot",
        "elapsed",
    ])

    # Dataset / Dataloaders
    train_ds = VideoClipTensorDataset(TRAIN_DIR, subset_fraction=SUBSET_FRACTION, expected_T=EXPECTED_T)
    val_ds = VideoClipTensorDataset(VAL_DIR, subset_fraction=min(1.0, SUBSET_FRACTION), expected_T=EXPECTED_T)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Autoencoder
    ae = MSVQVAE2Video(
        top_dim=TOP_DIM,
        bottom_dim=BOTTOM_DIM,
        K_top=K_TOP,
        K_bottom=K_BOTTOM,
        commit_top=0.05,
        commit_bottom=0.1,
        norm="gn",
        use_vgg=True,
        vgg_layers=16,
        soft_temp=SOFT_TEMP,
    ).to(device)

    # Optional post-init override (only if your MSVQVAE2Video stores these)
    if hasattr(ae, "commit_top"):
        ae.commit_top = 0.05
    if hasattr(ae, "commit_bottom"):
        ae.commit_bottom = 0.10
    print(f"[INFO] commit_top={getattr(ae, 'commit_top', None)} commit_bottom={getattr(ae, 'commit_bottom', None)}")

    # Move perceptual model to device
    if ae.vgg is not None:
        ae.vgg = ae.vgg.to(device)

    # Priors
    top_prior = TopPrior3D(
        K=K_TOP,
        emb_dim=TOP_PRIOR_EMB,
        hidden=TOP_PRIOR_HIDDEN,
        n_res=TOP_PRIOR_RESBLOCKS
    ).to(device)

    bot_prior = BottomPriorConditional3D(
        Kb=K_BOTTOM,
        cond_channels=TOP_DIM,  # conditioning uses z_top_up with TOP_DIM channels
        emb_dim=BOT_PRIOR_EMB,
        hidden=BOT_PRIOR_HIDDEN,
        n_res=BOT_PRIOR_RESBLOCKS
    ).to(device)

    # Load checkpoints if provided
    if STAGE in ("B", "C"):
        loaded = try_load_autoencoder(ae, AE_CKPT_PATH)
        if not loaded:
            print("[WARN] Stage B/C selected but AE_CKPT_PATH is empty or not found. Training may be invalid.")

    if STAGE == "C":
        loaded = try_load_priors(top_prior, bot_prior, PRIOR_CKPT_PATH)
        if not loaded:
            print("[WARN] Stage C selected but PRIOR_CKPT_PATH is empty or not found. RD fine-tuning may be unstable.")

    # Run
    if STAGE == "A":
        train_stage_a(ae, train_loader, val_loader, device, logger=logger)
    elif STAGE == "B":
        train_stage_b(ae, top_prior, bot_prior, train_loader, val_loader, device, logger=logger)
    elif STAGE == "C":
        if SWEEP_LAMBDA_RD:
            # Train a separate Stage C run for each λ to trace the full RD curve.
            # Each run reloads from the same Stage A/B checkpoints so they are independent.
            print(f"[INFO] Sweeping λ values: {LAMBDA_RD_VALUES}")
            import copy
            for lam in LAMBDA_RD_VALUES:
                print(f"\n{'='*60}")
                print(f" Starting sweep run: λ = {lam}")
                print(f"{'='*60}")
                # Deep-copy models so each λ starts from the same pre-trained weights
                ae_run = copy.deepcopy(ae)
                top_run = copy.deepcopy(top_prior)
                bot_run = copy.deepcopy(bot_prior)
                train_stage_c(ae_run, top_run, bot_run, train_loader, val_loader, device,
                              lambda_rd=lam, logger=logger)
        else:
            train_stage_c(ae, top_prior, bot_prior, train_loader, val_loader, device,
                          lambda_rd=LAMBDA_RD, logger=logger)
    elif STAGE == "AB":
        # K sweep: train Stage A + Stage B from scratch for each K in K_SWEEP_VALUES.
        # K=1024 is already done; this covers K=128, 256, 512 to get 4 RD curve points.
        # Checkpoints: ae_K{K}_epoch050.pt, priors_K{K}_epoch030.pt
        print(f"[INFO] K sweep: {K_SWEEP_VALUES}")
        for K in K_SWEEP_VALUES:
            prefix = f"K{K}_"
            print(f"\n{'='*60}")
            print(f" K sweep: K = {K}  (prefix={prefix})")
            print(f"{'='*60}")

            # Build fresh models for this K value.
            # use_ema=True: EMA codebook updates prevent collapse for small K.
            # Codebook entries track a moving average of assigned encoder outputs
            # instead of being updated by gradient descent — no explosion possible.
            # Dead entries are restarted from random encoder outputs each step.
            ae_k = MSVQVAE2Video(
                top_dim=TOP_DIM,
                bottom_dim=BOTTOM_DIM,
                K_top=K,
                K_bottom=K,
                commit_top=0.25,
                commit_bottom=0.25,
                norm="gn",
                use_vgg=True,
                vgg_layers=16,
                soft_temp=SOFT_TEMP,
                use_ema=True,
            ).to(device)
            if ae_k.vgg is not None:
                ae_k.vgg = ae_k.vgg.to(device)

            top_k = TopPrior3D(
                K=K,
                emb_dim=TOP_PRIOR_EMB,
                hidden=TOP_PRIOR_HIDDEN,
                n_res=TOP_PRIOR_RESBLOCKS,
            ).to(device)

            bot_k = BottomPriorConditional3D(
                Kb=K,
                cond_channels=TOP_DIM,
                emb_dim=BOT_PRIOR_EMB,
                hidden=BOT_PRIOR_HIDDEN,
                n_res=BOT_PRIOR_RESBLOCKS,
            ).to(device)

            # Stage A: train AE from scratch for this K
            train_stage_a(ae_k, train_loader, val_loader, device,
                          logger=logger, ckpt_prefix=prefix)

            # Stage B: freeze AE, train priors for this K
            train_stage_b(ae_k, top_k, bot_k, train_loader, val_loader, device,
                          logger=logger, ckpt_prefix=prefix)

            print(f"[INFO] K={K} done. Checkpoints: ae_{prefix}epoch{EPOCHS_STAGE_A:03d}.pt, "
                  f"priors_{prefix}epoch{EPOCHS_STAGE_B:03d}.pt")
    else:
        raise ValueError(f"Unknown STAGE={STAGE}")

    logger.close()


if __name__ == "__main__":
    main()
