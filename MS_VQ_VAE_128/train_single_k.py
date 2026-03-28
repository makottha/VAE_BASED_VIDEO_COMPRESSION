"""
train_single_k.py — Train one K value (Stage A + B) with checkpoint resume

Called by run_pipeline.py:
    python train_single_k.py 128
    python train_single_k.py 256
    ...

Resume logic:
  - Stage A: finds highest existing ae_K{K}_epoch*.pt, resumes from there
  - Stage B: finds highest existing priors_K{K}_epoch*.pt, resumes from there
  - If both stages complete, exits 0 immediately (idempotent)
"""

import os
import sys
import glob
import time
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, "../MS_VQ_VAE_RD")
from losses import recon_loss_fn, vgg_perceptual_loss, bpp_from_bits
from dataset import VideoClipTensorDataset
from utils import save_ckpt, set_requires_grad

from models_3level import MSVQVAE3Video
from priors_3level import (
    TopPrior3D, MidPriorConditional3D, BottomPriorConditional3D,
    categorical_bits_from_logits,
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_DIR = "../dataset_128/train/tensors"
VAL_DIR   = "../dataset_128/val/tensors"
OUT_DIR   = "./outputs_msvqvae_128"

EXPECTED_T       = 32
SUBSET_FRACTION  = 0.5
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 4
EPOCHS_A         = 20
EPOCHS_B         = 15
LR_AE            = 1e-4
LR_PRIOR         = 2e-4
USE_AMP          = True
GAMMA_PERCEPTUAL = 0.4
BETA_VQ          = 0.25
RECON_LOSS_MODE  = "l1"
PERCEPTUAL_STRIDE = 4

TOP_DIM    = 256
MID_DIM    = 192
BOTTOM_DIM = 128

TOP_PRIOR_EMB = 64;  TOP_PRIOR_HIDDEN = 128;  TOP_PRIOR_RESBLOCKS = 6
MID_PRIOR_EMB = 64;  MID_PRIOR_HIDDEN = 192;  MID_PRIOR_RESBLOCKS = 8
BOT_PRIOR_EMB = 64;  BOT_PRIOR_HIDDEN = 192;  BOT_PRIOR_RESBLOCKS = 8
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def ts():
    return datetime.now().strftime("%H:%M:%S")


class CsvLogger:
    def __init__(self, path, fieldnames):
        exists = os.path.exists(path)
        self._file = open(path, "a", newline="")   # append for resume
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            self._writer.writeheader()
        self._file.flush()

    def log(self, **kwargs):
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self):
        self._file.close()


def latest_checkpoint(pattern: str):
    """Return (path, epoch) of the most recent checkpoint matching glob pattern, or (None, 0)."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None, 0
    path = matches[-1]
    # extract epoch from filename e.g. ae_K128_epoch017.pt → 17
    import re
    m = re.search(r'epoch(\d+)', os.path.basename(path))
    epoch = int(m.group(1)) if m else 0
    return path, epoch


# ─── STAGE A ─────────────────────────────────────────────────────────────────

def run_stage_a(K: int, device: torch.device) -> MSVQVAE3Video:
    ckpt_pattern = os.path.join(OUT_DIR, f"ae_K{K}_epoch*.pt")
    resume_path, start_epoch = latest_checkpoint(ckpt_pattern)

    if start_epoch >= EPOCHS_A:
        print(f"[{ts()}] Stage A K={K}: already complete (epoch {start_epoch}/{EPOCHS_A}). Loading.")
        ae = MSVQVAE3Video(
            top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
            K_top=K, K_mid=K, K_bottom=K, use_ema=True, use_vgg=True,
        ).to(device)
        state = torch.load(resume_path, map_location=device, weights_only=True)
        ae.load_state_dict(state["model"])
        return ae

    print(f"[{ts()}] Stage A K={K}: starting from epoch {start_epoch + 1}/{EPOCHS_A}")

    train_ds = VideoClipTensorDataset(TRAIN_DIR, subset_fraction=SUBSET_FRACTION, expected_T=EXPECTED_T)
    val_ds   = VideoClipTensorDataset(VAL_DIR,   subset_fraction=1.0,             expected_T=EXPECTED_T)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    ae = MSVQVAE3Video(
        top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_mid=K, K_bottom=K, use_ema=True, use_vgg=True,
    ).to(device)

    opt    = torch.optim.Adam(ae.parameters(), lr=LR_AE, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    if resume_path:
        print(f"[{ts()}] Resuming AE from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=True)
        ae.load_state_dict(state["model"])
        if "opt" in state:
            opt.load_state_dict(state["opt"])

    log_path = os.path.join(OUT_DIR, f"log_stageA_K{K}.csv")
    logger = CsvLogger(log_path, ["epoch", "step", "loss_recon", "loss_vq", "loss_perc",
                                   "loss_total", "used_top", "used_mid", "used_bottom",
                                   "val_loss", "elapsed_s", "timestamp"])
    t0 = time.time()

    for epoch in range(start_epoch + 1, EPOCHS_A + 1):
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
                print(f"[{ts()}] A|K={K} Ep{epoch}/{EPOCHS_A} Step{step} "
                      f"rec={rec.item():.4f} vq={out['vq_loss'].item():.4f} "
                      f"perc={perc.item():.4f} "
                      f"used=({out['used_top']},{out['used_mid']},{out['used_bottom']}) "
                      f"t={elapsed:.0f}s", flush=True)
                logger.log(epoch=epoch, step=step,
                           loss_recon=round(rec.item(),5),
                           loss_vq=round(out["vq_loss"].item(),5),
                           loss_perc=round(perc.item(),5),
                           loss_total=round(loss.item()*GRAD_ACCUM_STEPS,5),
                           used_top=out["used_top"], used_mid=out["used_mid"],
                           used_bottom=out["used_bottom"],
                           elapsed_s=round(elapsed,1), timestamp=ts())

        # Validation
        ae.eval()
        val_total = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                xv = vbatch.to(device)
                with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                    out_v = ae(xv)
                    rec_v = recon_loss_fn(out_v["recon"], xv, mode=RECON_LOSS_MODE)
                    perc_v = (
                        vgg_perceptual_loss(ae.vgg, xv, out_v["recon"], stride=PERCEPTUAL_STRIDE)
                        if GAMMA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
                    )
                    val_total += float((rec_v + BETA_VQ * out_v["vq_loss"] + GAMMA_PERCEPTUAL * perc_v).item())
        val_loss = val_total / max(1, len(val_loader))
        print(f"[{ts()}] A|K={K} Ep{epoch} VAL loss={val_loss:.4f}", flush=True)
        logger.log(epoch=epoch, step="VAL", val_loss=round(val_loss,5), timestamp=ts())

        ckpt_path = os.path.join(OUT_DIR, f"ae_K{K}_epoch{epoch:03d}.pt")
        save_ckpt(ckpt_path, model=ae.state_dict(), opt=opt.state_dict(), epoch=epoch)
        print(f"[{ts()}] CKPT saved: {ckpt_path}", flush=True)

    logger.close()
    print(f"[{ts()}] Stage A K={K} COMPLETE", flush=True)
    return ae


# ─── STAGE B ─────────────────────────────────────────────────────────────────

def run_stage_b(K: int, ae: MSVQVAE3Video, device: torch.device):
    ckpt_pattern = os.path.join(OUT_DIR, f"priors_K{K}_epoch*.pt")
    resume_path, start_epoch = latest_checkpoint(ckpt_pattern)

    if start_epoch >= EPOCHS_B:
        print(f"[{ts()}] Stage B K={K}: already complete (epoch {start_epoch}/{EPOCHS_B}).")
        return

    print(f"[{ts()}] Stage B K={K}: starting from epoch {start_epoch + 1}/{EPOCHS_B}")

    set_requires_grad(ae, False)
    ae.eval()

    train_ds = VideoClipTensorDataset(TRAIN_DIR, subset_fraction=SUBSET_FRACTION, expected_T=EXPECTED_T)
    val_ds   = VideoClipTensorDataset(VAL_DIR,   subset_fraction=1.0,             expected_T=EXPECTED_T)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    top_prior = TopPrior3D(K, emb_dim=TOP_PRIOR_EMB, hidden=TOP_PRIOR_HIDDEN,
                            n_res=TOP_PRIOR_RESBLOCKS).to(device)
    mid_prior = MidPriorConditional3D(K, cond_channels=TOP_DIM, emb_dim=MID_PRIOR_EMB,
                                       hidden=MID_PRIOR_HIDDEN, n_res=MID_PRIOR_RESBLOCKS).to(device)
    bot_prior = BottomPriorConditional3D(K, cond_channels=256, emb_dim=BOT_PRIOR_EMB,
                                          hidden=BOT_PRIOR_HIDDEN, n_res=BOT_PRIOR_RESBLOCKS).to(device)

    all_params = list(top_prior.parameters()) + list(mid_prior.parameters()) + list(bot_prior.parameters())
    opt    = torch.optim.Adam(all_params, lr=LR_PRIOR, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    if resume_path:
        print(f"[{ts()}] Resuming priors from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=True)
        top_prior.load_state_dict(state["top_prior"])
        mid_prior.load_state_dict(state["mid_prior"])
        bot_prior.load_state_dict(state["bot_prior"])
        if "opt" in state:
            opt.load_state_dict(state["opt"])

    log_path = os.path.join(OUT_DIR, f"log_stageB_K{K}.csv")
    logger = CsvLogger(log_path, ["epoch", "step", "bits_top", "bits_mid", "bits_bot",
                                   "bits_total", "bpp", "val_bits", "val_bpp",
                                   "elapsed_s", "timestamp"])
    t0 = time.time()

    for epoch in range(start_epoch + 1, EPOCHS_B + 1):
        top_prior.train(); mid_prior.train(); bot_prior.train()
        opt.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            x = batch.to(device)
            B, C, T, H, W = x.shape

            with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                with torch.no_grad():
                    out = ae(x)
                idx_t    = out["idx_top"];    cond_mid = out["z_top_up"].detach()
                idx_m    = out["idx_mid"];    cond_bot = out["z_mid_up"].detach()
                idx_b    = out["idx_bottom"]

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
                print(f"[{ts()}] B|K={K} Ep{epoch}/{EPOCHS_B} Step{step} "
                      f"bpp={bpp.item():.4f} "
                      f"bits=({bits_t.item():.0f},{bits_m.item():.0f},{bits_b.item():.0f}) "
                      f"t={elapsed:.0f}s", flush=True)
                logger.log(epoch=epoch, step=step,
                           bits_top=round(bits_t.item(),2), bits_mid=round(bits_m.item(),2),
                           bits_bot=round(bits_b.item(),2), bits_total=round(bits.item(),2),
                           bpp=round(bpp.item(),5), elapsed_s=round(elapsed,1), timestamp=ts())

        # Validation
        top_prior.eval(); mid_prior.eval(); bot_prior.eval()
        val_bits_total = 0.0; val_bpp_total = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                xv = vbatch.to(device)
                Bv, Cv, Tv, Hv, Wv = xv.shape
                pix = Bv * Tv * Hv * Wv
                with autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
                    ov = ae(xv)
                    lt = top_prior(ov["idx_top"])
                    lm = mid_prior(ov["idx_mid"], ov["z_top_up"])
                    lb = bot_prior(ov["idx_bottom"], ov["z_mid_up"])
                    bt = categorical_bits_from_logits(lt, ov["idx_top"])
                    bm = categorical_bits_from_logits(lm, ov["idx_mid"])
                    bb = categorical_bits_from_logits(lb, ov["idx_bottom"])
                    b  = bt + bm + bb
                    val_bits_total += float(b.item())
                    val_bpp_total  += float(bpp_from_bits(b, xv).item())

        vb = val_bits_total / max(1, len(val_loader))
        vbpp = val_bpp_total / max(1, len(val_loader))
        print(f"[{ts()}] B|K={K} Ep{epoch} VAL bpp={vbpp:.4f}", flush=True)
        logger.log(epoch=epoch, step="VAL", val_bits=round(vb,2), val_bpp=round(vbpp,5), timestamp=ts())

        ckpt_path = os.path.join(OUT_DIR, f"priors_K{K}_epoch{epoch:03d}.pt")
        torch.save({
            "K": K, "epoch": epoch,
            "top_prior": top_prior.state_dict(),
            "mid_prior": mid_prior.state_dict(),
            "bot_prior": bot_prior.state_dict(),
            "opt": opt.state_dict(),
        }, ckpt_path)
        print(f"[{ts()}] CKPT saved: {ckpt_path}", flush=True)

    logger.close()
    set_requires_grad(ae, True)
    print(f"[{ts()}] Stage B K={K} COMPLETE", flush=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_single_k.py <K>")
        sys.exit(1)

    K = int(sys.argv[1])
    assert K in [128, 256, 512, 1024], f"K must be one of 128/256/512/1024, got {K}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}  |  K={K}", flush=True)
    print(f"[{ts()}] Batch={BATCH_SIZE}  GradAccum={GRAD_ACCUM_STEPS}  EffBatch={BATCH_SIZE*GRAD_ACCUM_STEPS}", flush=True)

    ae = run_stage_a(K, device)
    run_stage_b(K, ae, device)

    print(f"[{ts()}] K={K} FULLY COMPLETE", flush=True)
