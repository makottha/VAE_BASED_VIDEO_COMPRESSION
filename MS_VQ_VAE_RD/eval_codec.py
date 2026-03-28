# eval_codec.py
import os
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

from models import MSVQVAE2Video
from priors import TopPrior3D, BottomPriorConditional3D, categorical_bits_from_logits
from losses import vgg_perceptual_loss

# -----------------------------
# Single reusable timestamp logger
# -----------------------------
def log(msg: str, stage: str = "EVAL") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][{stage}] {msg}")


# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalConfig:
    # Data
    DATA_DIR: str = "../dataset_64/test/tensors"  # or "../dataset_64/val/tensors"
    EXPECTED_T: int = 32
    FPS: int = 16  # if 2 sec clips and T=32, FPS=16

    # Checkpoints
    # Prefer RD checkpoint (Stage C) because it contains AE + priors together.
    # For λ-sweep runs use the tagged name, e.g. "rd_lam0.001_epoch020.pt".
    # To evaluate the Stage A autoencoder alone (no rate opt), set:
    #   RD_CKPT_PATH = ""  and  AE_CKPT_PATH = "./outputs_msvqvae_rd/ae_epoch050.pt"
    RD_CKPT_PATH: str = ""         # empty = use AE + priors separately below

    # Stage B evaluation: best AE (Stage A ep50) + best priors (Stage B ep20)
    AE_CKPT_PATH: str = "./outputs_msvqvae_rd/ae_epoch050.pt"
    PRIOR_CKPT_PATH: str = "./outputs_msvqvae_rd/priors_epoch020.pt"

    # Model params (must match training)
    K_TOP: int = 1024
    K_BOTTOM: int = 1024
    TOP_DIM: int = 256
    BOTTOM_DIM: int = 128

    TOP_PRIOR_EMB: int = 64
    TOP_PRIOR_HIDDEN: int = 128
    TOP_PRIOR_RESBLOCKS: int = 6

    BOT_PRIOR_EMB: int = 64
    BOT_PRIOR_HIDDEN: int = 192
    BOT_PRIOR_RESBLOCKS: int = 8

    # Eval controls
    MAX_CLIPS: int = 500  # 0 means all; 500 is sufficient for publication-quality averages
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Perceptual (VGG-based, internal to AE)
    PERCEPTUAL_STRIDE: int = 4
    COMPUTE_PERCEPTUAL: bool = True  # uses ae.vgg if available

    # LPIPS — frame-level perceptual similarity (lower = better, range ~0–1)
    # Uses VGG backbone, evaluates every frame. Requires: pip install lpips
    COMPUTE_LPIPS: bool = True

    # Output
    OUT_CSV: str = "./outputs_msvqvae_rd/metrics_test.csv"


# -----------------------------
# SSIM implementation (PyTorch)
# -----------------------------
def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device: str = "cpu") -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return w

def ssim_2d(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    x, y: (N,C,H,W) in [0,1]
    returns: scalar mean SSIM over N,C
    """
    assert x.shape == y.shape
    device = x.device
    w = _gaussian_window(window_size, sigma, device=str(device))
    C = x.shape[1]
    w = w.repeat(C, 1, 1, 1)  # (C,1,ws,ws)

    # conv2d with groups=C to do per-channel filtering
    mu_x = F.conv2d(x, w, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, w, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=window_size // 2, groups=C) - mu_xy

    # SSIM constants for data range [0,1]
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()


def psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0) -> float:
    mse_val = float(mse.item())
    if mse_val <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse_val)


# -----------------------------
# Checkpoint loading
# -----------------------------
def load_models(cfg: EvalConfig):
    device = torch.device(cfg.DEVICE)
    log(f"Using device: {device}", "EVAL")

    ae = MSVQVAE2Video(
        top_dim=cfg.TOP_DIM,
        bottom_dim=cfg.BOTTOM_DIM,
        K_top=cfg.K_TOP,
        K_bottom=cfg.K_BOTTOM,
        commit_top=0.25,
        commit_bottom=0.25,
        norm="gn",
        use_vgg=True,
        vgg_layers=16,
        soft_temp=1.0,   # soft_bits not used at eval time
        use_ema=False,   # eval only — EMA buffers not needed; use strict=False below
    ).to(device)

    top_prior = TopPrior3D(
        K=cfg.K_TOP,
        emb_dim=cfg.TOP_PRIOR_EMB,
        hidden=cfg.TOP_PRIOR_HIDDEN,
        n_res=cfg.TOP_PRIOR_RESBLOCKS,
    ).to(device)

    bot_prior = BottomPriorConditional3D(
        Kb=cfg.K_BOTTOM,
        cond_channels=cfg.TOP_DIM,
        emb_dim=cfg.BOT_PRIOR_EMB,
        hidden=cfg.BOT_PRIOR_HIDDEN,
        n_res=cfg.BOT_PRIOR_RESBLOCKS,
    ).to(device)

    # Prefer RD checkpoint (contains everything)
    if cfg.RD_CKPT_PATH and os.path.isfile(cfg.RD_CKPT_PATH):
        ckpt = torch.load(cfg.RD_CKPT_PATH, map_location="cpu")
        ae.load_state_dict(ckpt["ae"], strict=False)
        top_prior.load_state_dict(ckpt["top_prior"])
        bot_prior.load_state_dict(ckpt["bot_prior"])
        log(f"Loaded RD checkpoint: {cfg.RD_CKPT_PATH}", "EVAL")
    else:
        # Fallback: AE + priors separately
        if cfg.AE_CKPT_PATH:
            ckpt = torch.load(cfg.AE_CKPT_PATH, map_location="cpu")
            ae.load_state_dict(ckpt["ae"], strict=False)
            log(f"Loaded AE checkpoint: {cfg.AE_CKPT_PATH}", "EVAL")
        if cfg.PRIOR_CKPT_PATH:
            ckpt = torch.load(cfg.PRIOR_CKPT_PATH, map_location="cpu")
            top_prior.load_state_dict(ckpt["top_prior"])
            bot_prior.load_state_dict(ckpt["bot_prior"])
            log(f"Loaded priors checkpoint: {cfg.PRIOR_CKPT_PATH}", "EVAL")

    # Ensure VGG is on device
    if getattr(ae, "vgg", None) is not None:
        ae.vgg = ae.vgg.to(device).eval()

    ae.eval()
    top_prior.eval()
    bot_prior.eval()

    # LPIPS model (VGG backbone, frozen)
    lpips_model = None
    if cfg.COMPUTE_LPIPS:
        if _LPIPS_AVAILABLE:
            lpips_model = lpips_lib.LPIPS(net="vgg").to(device).eval()
            log("LPIPS model loaded (VGG backbone)", "EVAL")
        else:
            log("WARNING: COMPUTE_LPIPS=True but lpips package not found. pip install lpips", "EVAL")

    return ae, top_prior, bot_prior, device, lpips_model


# -----------------------------
# Core evaluation per clip
# -----------------------------
@torch.no_grad()
def eval_one_clip(
    x: torch.Tensor,
    ae: MSVQVAE2Video,
    top_prior: TopPrior3D,
    bot_prior: BottomPriorConditional3D,
    device: torch.device,
    cfg: EvalConfig,
    lpips_model=None,
) -> Dict[str, Any]:
    """
    x: (3,T,H,W) float in [0,1]
    returns metrics dict
    """
    # shape checks
    assert x.ndim == 4 and x.shape[0] == 3, f"Expected (3,T,H,W), got {tuple(x.shape)}"
    if cfg.EXPECTED_T and x.shape[1] != cfg.EXPECTED_T:
        log(f"Warning: clip T={x.shape[1]} != EXPECTED_T={cfg.EXPECTED_T}", "EVAL")

    x = x.unsqueeze(0).to(device)  # (1,3,T,H,W)

    out = ae(x)
    recon = out["recon"].clamp(0, 1)

    idx_t = out["idx_top"]
    idx_b = out["idx_bottom"]
    cond = out["z_top_up"].detach()

    # Rate (expected bits)
    logits_t = top_prior(idx_t)
    bits_t = categorical_bits_from_logits(logits_t, idx_t)

    logits_b = bot_prior(idx_b, cond)
    bits_b = categorical_bits_from_logits(logits_b, idx_b)

    bits = bits_t + bits_b

    # bpp: bits / (B*T*H*W) ; here B=1
    _, _, T, H, W = x.shape
    bpp = float(bits.item()) / float(T * H * W)

    # kbps for 2-second clip
    seconds = float(T) / float(cfg.FPS)
    kbps = (float(bits.item()) / seconds) / 1000.0

    # Distortion metrics
    mse = F.mse_loss(recon, x)
    psnr = psnr_from_mse(mse)

    # SSIM: compute on frames by flattening time into batch
    # x_2d, r_2d: (N,C,H,W) where N = B*T
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, H, W)
    r_2d = recon.permute(0, 2, 1, 3, 4).reshape(-1, 3, H, W)
    ssim = float(ssim_2d(r_2d, x_2d).item())

    l1 = float(F.l1_loss(recon, x).item())

    # VGG perceptual loss (internal AE VGG, strided over frames)
    perc = None
    if cfg.COMPUTE_PERCEPTUAL and getattr(ae, "vgg", None) is not None:
        perc = float(vgg_perceptual_loss(ae.vgg, x, recon, stride=cfg.PERCEPTUAL_STRIDE).item())

    # LPIPS — frame-level perceptual similarity (lower = better)
    # Inputs rescaled from [0,1] to [-1,1] as required by the lpips library.
    lpips_score = None
    if lpips_model is not None:
        x_lp = x_2d * 2.0 - 1.0      # (T, 3, H, W) in [-1, 1]
        r_lp = r_2d * 2.0 - 1.0
        lpips_score = float(lpips_model(r_lp, x_lp).mean().item())

    # bits per symbol (top+bottom)
    n_sym = idx_t.numel() + idx_b.numel()
    bits_per_sym = float(bits.item()) / max(1, n_sym)

    return {
        "bits": float(bits.item()),
        "bits_top": float(bits_t.item()),
        "bits_bottom": float(bits_b.item()),
        "bpp": bpp,
        "kbps": kbps,
        "bits_per_sym": bits_per_sym,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_score,
        "mse": float(mse.item()),
        "l1": l1,
        "vq_loss": float(out["vq_loss"].item()) if "vq_loss" in out else None,
        "perc": perc,
        "entropy_top": out.get("entropy_top", None),
        "entropy_bottom": out.get("entropy_bottom", None),
        "used_top": out.get("used_top", None),
        "used_bottom": out.get("used_bottom", None),
        "T": int(T),
        "H": int(H),
        "W": int(W),
    }


def list_pt_files(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pt"):
            files.append(os.path.join(folder, name))
    files.sort()
    return files


# -----------------------------
# K sweep configs — all 4 RD curve points
# Each entry: (K, ae_ckpt, prior_ckpt, out_csv)
# Set EVAL_MODE = "sweep" to run all; "single" to run just EvalConfig defaults.
# -----------------------------
EVAL_MODE = "sweep"  # "single" or "sweep"

K_SWEEP_CONFIGS = [
    dict(
        K=128,
        AE_CKPT_PATH="./outputs_msvqvae_rd/ae_K128_epoch020.pt",
        PRIOR_CKPT_PATH="./outputs_msvqvae_rd/priors_K128_epoch015.pt",
        OUT_CSV="./outputs_msvqvae_rd/metrics_K128.csv",
    ),
    dict(
        K=256,
        AE_CKPT_PATH="./outputs_msvqvae_rd/ae_K256_epoch020.pt",
        PRIOR_CKPT_PATH="./outputs_msvqvae_rd/priors_K256_epoch015.pt",
        OUT_CSV="./outputs_msvqvae_rd/metrics_K256.csv",
    ),
    dict(
        K=512,
        AE_CKPT_PATH="./outputs_msvqvae_rd/ae_K512_epoch020.pt",
        PRIOR_CKPT_PATH="./outputs_msvqvae_rd/priors_K512_epoch015.pt",
        OUT_CSV="./outputs_msvqvae_rd/metrics_K512.csv",
    ),
    dict(
        K=1024,
        AE_CKPT_PATH="./outputs_msvqvae_rd/ae_K1024_epoch020.pt",
        PRIOR_CKPT_PATH="./outputs_msvqvae_rd/priors_K1024_epoch015.pt",
        OUT_CSV="./outputs_msvqvae_rd/metrics_K1024.csv",
    ),
]


def run_eval(cfg: EvalConfig) -> Dict[str, float]:
    """Run evaluation for one config. Returns summary dict."""
    if not os.path.isdir(cfg.DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {cfg.DATA_DIR}")

    ae, top_prior, bot_prior, device, lpips_model = load_models(cfg)

    files = list_pt_files(cfg.DATA_DIR)
    if cfg.MAX_CLIPS and cfg.MAX_CLIPS > 0:
        files = files[: cfg.MAX_CLIPS]

    log(f"Found {len(files)} tensor clips in: {cfg.DATA_DIR}", "EVAL")
    os.makedirs(os.path.dirname(cfg.OUT_CSV), exist_ok=True)

    fieldnames = [
        "clip",
        "bits", "bits_top", "bits_bottom",
        "bpp", "kbps", "bits_per_sym",
        "psnr", "ssim", "lpips",
        "mse", "l1",
        "vq_loss", "perc",
        "entropy_top", "entropy_bottom",
        "used_top", "used_bottom",
        "T", "H", "W",
    ]

    agg = {k: 0.0 for k in ["bits", "bpp", "kbps", "psnr", "ssim", "mse", "l1", "bits_per_sym"]}
    lpips_agg = 0.0
    n = 0

    with open(cfg.OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(files, 1):
            clip_name = os.path.basename(path)
            x = torch.load(path, map_location="cpu")

            if isinstance(x, dict) and "tensor" in x:
                x = x["tensor"]
            if not torch.is_tensor(x):
                log(f"Skip (not a tensor): {clip_name}", "EVAL")
                continue

            x = x.float()
            if x.max() > 1.0:
                x = (x / 255.0).clamp(0, 1)

            row = eval_one_clip(x, ae, top_prior, bot_prior, device, cfg, lpips_model=lpips_model)
            row["clip"] = clip_name
            writer.writerow(row)

            n += 1
            for k in agg.keys():
                agg[k] += float(row[k])
            if row["lpips"] is not None:
                lpips_agg += float(row["lpips"])

            if i % 50 == 0:
                lpips_avg = lpips_agg / n if lpips_model is not None else float("nan")
                log(
                    f"[{i:05d}/{len(files):05d}] "
                    f"avg_bpp={agg['bpp']/n:.4f} avg_psnr={agg['psnr']/n:.2f} "
                    f"avg_ssim={agg['ssim']/n:.4f} avg_lpips={lpips_avg:.4f}",
                    "EVAL",
                )

    if n == 0:
        log("No clips evaluated.", "EVAL")
        return {}

    lpips_avg = lpips_agg / n if lpips_model is not None else float("nan")
    summary = {
        "bpp": agg["bpp"] / n,
        "kbps": agg["kbps"] / n,
        "psnr": agg["psnr"] / n,
        "ssim": agg["ssim"] / n,
        "lpips": lpips_avg,
        "bits_per_sym": agg["bits_per_sym"] / n,
        "n_clips": n,
    }
    log(
        f"Done. Wrote CSV: {cfg.OUT_CSV}\n"
        f"  bpp={summary['bpp']:.4f}  kbps={summary['kbps']:.2f}  "
        f"psnr={summary['psnr']:.2f}  ssim={summary['ssim']:.4f}  lpips={summary['lpips']:.4f}",
        "EVAL",
    )
    return summary


def main():
    if EVAL_MODE == "sweep":
        # Write a single summary CSV with one row per K value for easy paper table copy-paste
        summary_csv = "./outputs_msvqvae_rd/rd_curve_summary.csv"
        os.makedirs("./outputs_msvqvae_rd", exist_ok=True)
        summary_rows = []

        for entry in K_SWEEP_CONFIGS:
            K = entry["K"]
            log(f"\n{'='*50}", "EVAL")
            log(f"Evaluating K={K}", "EVAL")
            log(f"{'='*50}", "EVAL")

            cfg = EvalConfig(
                K_TOP=K,
                K_BOTTOM=K,
                AE_CKPT_PATH=entry["AE_CKPT_PATH"],
                PRIOR_CKPT_PATH=entry["PRIOR_CKPT_PATH"],
                OUT_CSV=entry["OUT_CSV"],
            )

            summary = run_eval(cfg)
            if summary:
                summary["K"] = K
                summary_rows.append(summary)

        # Write summary table
        if summary_rows:
            with open(summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["K", "bpp", "kbps", "psnr", "ssim", "lpips", "bits_per_sym", "n_clips"])
                writer.writeheader()
                writer.writerows(summary_rows)

            log(f"\nRD curve summary written to: {summary_csv}", "EVAL")
            log("K     BPP      PSNR    SSIM    LPIPS", "EVAL")
            log("-" * 45, "EVAL")
            for r in summary_rows:
                log(f"{r['K']:<6} {r['bpp']:.4f}  {r['psnr']:.2f}  {r['ssim']:.4f}  {r['lpips']:.4f}", "EVAL")

    else:
        cfg = EvalConfig()
        run_eval(cfg)


if __name__ == "__main__":
    main()
