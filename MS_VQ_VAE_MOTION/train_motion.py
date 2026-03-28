"""
train_motion.py — P-frame VQ-VAE training for Motion-Conditioned Codec (Paper 3)
==================================================================================
Trains a dedicated VQ-VAE on frame-difference clips.
Mirrors the K-sweep structure of MS_VQ_VAE_RD/train.py.

Stage A : Train autoencoder on frame diffs (L1 + VQ loss; NO VGG perceptual)
Stage B : Freeze AE, train entropy priors on diff indices

K sweep : K in K_SWEEP_VALUES — each K gives one P-frame operating point.
          Combined with I-frame K=1024 (from MS_VQ_VAE_RD) and GOP size,
          this produces the full Paper 3 RD curve.

Run:
    python train_motion.py
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models import MSVQVAE2Video
from priors import TopPrior3D, BottomPriorConditional3D, categorical_bits_from_logits
from losses import recon_loss_fn, bpp_from_bits
from dataset_motion import FrameDiffDataset
from utils import save_ckpt, set_requires_grad

# ── paths ──────────────────────────────────────────────────────────────────────
TRAIN_DIR = "../dataset_64/train/tensors"
VAL_DIR   = "../dataset_64/val/tensors"
OUT_DIR   = "./outputs"

# ── training config ────────────────────────────────────────────────────────────
STAGE           = "AB"         # "A", "B", or "AB" (runs both per K)
K_SWEEP_VALUES  = [128, 256, 512, 1024]

BATCH_SIZE      = 8
EPOCHS_STAGE_A  = 20
EPOCHS_STAGE_B  = 15
SUBSET_FRACTION = 0.1   # ~4,977 clips → ~620 steps/epoch → ~3 min/epoch on RTX4060
                         # Matches Paper 2 training quality at this scale.
                         # Increase to 0.5 for publication-quality run (overnight ~8h).

LR_AE    = 1e-4
LR_PRIOR = 2e-4
USE_AMP  = True

BETA_VQ         = 0.25
RECON_LOSS_MODE = "l1"
# NOTE: VGG perceptual loss is intentionally disabled for diff training.
# Frame diffs are near-zero sparse signals — VGG features (trained on natural
# images) are not meaningful and would add noise to the gradient signal.

# ── model config ───────────────────────────────────────────────────────────────
TOP_DIM    = 256
BOTTOM_DIM = 128

TOP_PRIOR_EMB       = 64
TOP_PRIOR_HIDDEN    = 128
TOP_PRIOR_RESBLOCKS = 6
BOT_PRIOR_EMB       = 64
BOT_PRIOR_HIDDEN    = 192
BOT_PRIOR_RESBLOCKS = 8

EXPECTED_T = 32


# ── CSV logger ─────────────────────────────────────────────────────────────────
class CsvLogger:
    def __init__(self, path: str, fieldnames: list):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        print(f"[LOG] {path}")

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ── validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_ae(ae, val_loader, device):
    ae.eval()
    total = 0.0
    for batch in val_loader:
        x = batch.to(device)
        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out  = ae(x)
            rec  = recon_loss_fn(out["recon"], x, mode=RECON_LOSS_MODE)
            loss = rec + BETA_VQ * out["vq_loss"]
        total += float(loss.item())
    return total / max(1, len(val_loader))


@torch.no_grad()
def validate_priors(ae, top_prior, bot_prior, val_loader, device):
    ae.eval(); top_prior.eval(); bot_prior.eval()
    total_bpp = 0.0
    for batch in val_loader:
        x = batch.to(device)
        with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            out      = ae(x)
            idx_t    = out["idx_top"]
            idx_b    = out["idx_bottom"]
            cond     = out["z_top_up"].detach()
            bits_t   = categorical_bits_from_logits(top_prior(idx_t), idx_t)
            bits_b   = categorical_bits_from_logits(bot_prior(idx_b, cond), idx_b)
            total_bpp += float(bpp_from_bits(bits_t + bits_b, x).item())
    return total_bpp / max(1, len(val_loader))


# ── Stage A ────────────────────────────────────────────────────────────────────
def train_stage_a(ae, train_loader, val_loader, device, logger, ckpt_prefix=""):
    tag = ckpt_prefix.rstrip("_") or "Stage A"
    print(f"\n{'='*50}\n Stage A — {tag}\n{'='*50}")
    set_requires_grad(ae, True)
    opt    = torch.optim.Adam(ae.parameters(), lr=LR_AE)
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    for epoch in range(1, EPOCHS_STAGE_A + 1):
        ae.train()
        t0 = time.time()
        running = 0.0

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)
            opt.zero_grad()
            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                out  = ae(x)
                rec  = recon_loss_fn(out["recon"], x, mode=RECON_LOSS_MODE)
                loss = rec + BETA_VQ * out["vq_loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())

        val_loss = validate_ae(ae, val_loader, device)
        elapsed  = time.time() - t0
        avg      = running / max(1, len(train_loader))
        print(f"  [A] {ckpt_prefix}ep {epoch:03d}/{EPOCHS_STAGE_A}  "
              f"train={avg:.4f}  val={val_loss:.4f}  {elapsed:.0f}s")

        ckpt_path = os.path.join(OUT_DIR, f"ae_{ckpt_prefix}epoch{epoch:03d}.pt")
        save_ckpt(ckpt_path, ae=ae.state_dict(), epoch=epoch)
        if logger:
            logger.log(stage="A", prefix=ckpt_prefix, epoch=epoch,
                       train_loss=avg, val_loss=val_loss, elapsed=elapsed)


# ── Stage B ────────────────────────────────────────────────────────────────────
def train_stage_b(ae, top_prior, bot_prior, train_loader, val_loader,
                  device, logger, ckpt_prefix=""):
    tag = ckpt_prefix.rstrip("_") or "Stage B"
    print(f"\n{'='*50}\n Stage B — {tag}\n{'='*50}")
    set_requires_grad(ae, False)
    set_requires_grad(top_prior, True)
    set_requires_grad(bot_prior, True)
    ae.eval()

    params = list(top_prior.parameters()) + list(bot_prior.parameters())
    opt    = torch.optim.Adam(params, lr=LR_PRIOR)
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    for epoch in range(1, EPOCHS_STAGE_B + 1):
        top_prior.train(); bot_prior.train()
        t0 = time.time()
        running = 0.0

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)
            opt.zero_grad()
            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                with torch.no_grad():
                    out  = ae(x)
                idx_t  = out["idx_top"]
                idx_b  = out["idx_bottom"]
                cond   = out["z_top_up"].detach()
                bits_t = categorical_bits_from_logits(top_prior(idx_t), idx_t)
                bits_b = categorical_bits_from_logits(bot_prior(idx_b, cond), idx_b)
                loss   = bpp_from_bits(bits_t + bits_b, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())

        val_bpp = validate_priors(ae, top_prior, bot_prior, val_loader, device)
        elapsed = time.time() - t0
        avg     = running / max(1, len(train_loader))
        print(f"  [B] {ckpt_prefix}ep {epoch:03d}/{EPOCHS_STAGE_B}  "
              f"train_bpp={avg:.4f}  val_bpp={val_bpp:.4f}  {elapsed:.0f}s")

        ckpt_path = os.path.join(OUT_DIR, f"priors_{ckpt_prefix}epoch{epoch:03d}.pt")
        save_ckpt(ckpt_path, top_prior=top_prior.state_dict(),
                  bot_prior=bot_prior.state_dict(), epoch=epoch)
        if logger:
            logger.log(stage="B", prefix=ckpt_prefix, epoch=epoch,
                       val_bpp=val_bpp, elapsed=elapsed)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] K sweep: {K_SWEEP_VALUES}")
    print(f"[INFO] Stage A: {EPOCHS_STAGE_A} epochs  Stage B: {EPOCHS_STAGE_B} epochs")
    print(f"[INFO] Subset: {SUBSET_FRACTION}  Batch: {BATCH_SIZE}  AMP: {USE_AMP}")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = FrameDiffDataset(TRAIN_DIR, subset_fraction=SUBSET_FRACTION, shuffle=True)
    val_ds   = FrameDiffDataset(VAL_DIR,   subset_fraction=1.0,             shuffle=False)
    # num_workers=0: Windows multiprocessing DataLoader has high IPC overhead;
    # single-threaded loading is faster in practice on Windows.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    logger = CsvLogger(
        os.path.join(OUT_DIR, "train_log.csv"),
        fieldnames=["stage", "prefix", "epoch", "train_loss", "val_loss", "val_bpp", "elapsed"],
    )

    for K in K_SWEEP_VALUES:
        prefix = f"K{K}_"
        print(f"\n{'#'*60}\n  K = {K}\n{'#'*60}")

        ae = MSVQVAE2Video(
            top_dim=TOP_DIM, bottom_dim=BOTTOM_DIM,
            K_top=K, K_bottom=K,
            commit_top=0.25, commit_bottom=0.25,
            norm="gn",
            use_vgg=False,   # disabled: VGG not meaningful for diff signals
            soft_temp=1.0,
            use_ema=True,    # EMA for stable training at all K values
        ).to(device)

        top_prior = TopPrior3D(
            K=K, emb_dim=TOP_PRIOR_EMB,
            hidden=TOP_PRIOR_HIDDEN, n_res=TOP_PRIOR_RESBLOCKS,
        ).to(device)
        bot_prior = BottomPriorConditional3D(
            Kb=K, cond_channels=TOP_DIM,
            emb_dim=BOT_PRIOR_EMB, hidden=BOT_PRIOR_HIDDEN, n_res=BOT_PRIOR_RESBLOCKS,
        ).to(device)

        if STAGE in ("A", "AB"):
            train_stage_a(ae, train_loader, val_loader, device, logger, ckpt_prefix=prefix)

        if STAGE in ("B", "AB"):
            # Load best AE checkpoint (last epoch of Stage A)
            ae_ckpt = os.path.join(OUT_DIR, f"ae_{prefix}epoch{EPOCHS_STAGE_A:03d}.pt")
            if os.path.isfile(ae_ckpt):
                ckpt = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
                ae.load_state_dict(ckpt["ae"], strict=False)
                print(f"[LOAD] AE loaded: {ae_ckpt}")
            train_stage_b(ae, top_prior, bot_prior, train_loader, val_loader,
                          device, logger, ckpt_prefix=prefix)

    logger.close()
    print("\n[DONE] All K values trained.")
    print(f"Checkpoints in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
