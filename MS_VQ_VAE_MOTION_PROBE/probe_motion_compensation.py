#!/usr/bin/env python3
"""
Phase 1 Probe: Motion-Compensated Residual Analysis
====================================================
Tests the core hypothesis for Paper 3 (Motion-Conditioned VQ-VAE):

  "Motion-compensated residuals are significantly cheaper to encode
   than raw frames, validating the two-stream VQ architecture."

What this script does
---------------------
For each of N_CLIPS test clips:
  1. Encode RAW clip with existing VQ-VAE (K=1024) → bpp_raw, entropy, quality
  2. Compute optical flow (RAFT) between consecutive frames
  3. Warp frame_{t-1} → predicted frame_t
  4. Build residual clip: frame_0 unchanged; frames 1..T-1 = residuals normalized to [0,1]
  5. Encode RESIDUAL clip with same VQ-VAE → bpp_residual, entropy, quality
  6. Reconstruct frame_t = predicted_frame_t + decoded_residual
  7. Measure PSNR of final reconstruction vs baseline

Hypothesis holds if:
  - bpp_residual < bpp_raw (residuals are cheaper)
  - PSNR of motion+residual reconstruction is ≥ baseline reconstruction
  - Residual pixel magnitudes are significantly smaller than raw frame values

Outputs
-------
  results/probe_metrics.csv     — per-clip metrics
  results/probe_summary.json    — aggregate statistics
  results/plots/01_residual_magnitude.png
  results/plots/02_bpp_comparison.png
  results/plots/03_codebook_utilization.png
  results/plots/samples/clip_NNN_overview.png  (for N_SAMPLE_CLIPS clips)
"""

import os, sys, math, time, json, csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

# ── resolve paths ─────────────────────────────────────────────────────────────
PROBE_DIR = Path(__file__).resolve().parent
RD_DIR    = PROBE_DIR.parent / "MS_VQ_VAE_RD"
sys.path.insert(0, str(RD_DIR))
from models import MSVQVAE2Video

# ── config ────────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR       = RD_DIR.parent / "dataset_64" / "test" / "tensors"
AE_CKPT        = RD_DIR / "outputs_msvqvae_rd" / "ae_K1024_epoch020.pt"
RESULTS_DIR    = PROBE_DIR / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"
SAMPLES_DIR    = PLOTS_DIR / "samples"

N_CLIPS        = 100   # number of test clips to probe (100 is fast, 500 for publication)
N_SAMPLE_CLIPS = 5     # number of clips to save visual samples for

# RAFT: upsample 64×64 → 256×256 before flow estimation, then scale flow back.
# RAFT was designed for larger inputs; at 64×64 the feature pyramid collapses.
RAFT_UPSAMPLE  = 4

# VQ-VAE model config — must match K=1024 training
K         = 1024
TOP_DIM   = 256
BOTTOM_DIM = 128

SEED = 42
torch.manual_seed(SEED)


# ── logging ───────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── RAFT optical flow ─────────────────────────────────────────────────────────
def load_raft():
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    weights    = Raft_Small_Weights.DEFAULT
    raft_model = raft_small(weights=weights).to(DEVICE).eval()
    transforms = weights.transforms()
    log("RAFT small loaded.")
    return raft_model, transforms


@torch.no_grad()
def compute_flow(raft_model, transforms, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    frame1, frame2: (3, H, W) float in [0, 1] on any device
    returns flow:   (2, H, W) pixel displacements at original (H, W) resolution
    """
    H, W = frame1.shape[1], frame1.shape[2]
    f1 = frame1.unsqueeze(0)   # (1, 3, H, W)
    f2 = frame2.unsqueeze(0)

    # Upsample for RAFT accuracy at low resolution
    if RAFT_UPSAMPLE > 1:
        f1_up = F.interpolate(f1, scale_factor=RAFT_UPSAMPLE, mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, scale_factor=RAFT_UPSAMPLE, mode="bilinear", align_corners=False)
    else:
        f1_up, f2_up = f1, f2

    # RAFT transforms expect float tensors in [0, 255]
    f1_t, f2_t = transforms(f1_up * 255.0, f2_up * 255.0)
    f1_t = f1_t.to(DEVICE)
    f2_t = f2_t.to(DEVICE)

    predictions = raft_model(f1_t, f2_t)
    flow_up = predictions[-1].squeeze(0)          # (2, H*scale, W*scale)

    # Downsample flow back to original resolution and scale displacements
    if RAFT_UPSAMPLE > 1:
        flow = F.interpolate(
            flow_up.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        flow = flow / RAFT_UPSAMPLE
    else:
        flow = flow_up

    return flow  # (2, H, W)


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Bilinear warp of frame using flow field.
    frame: (3, H, W) float in [0, 1]
    flow:  (2, H, W) in pixels  (dx=flow[0], dy=flow[1])
    returns warped: (3, H, W) float in [0, 1]
    """
    H, W = frame.shape[1], frame.shape[2]
    frame_b = frame.unsqueeze(0)   # (1, 3, H, W)
    flow_b  = flow.unsqueeze(0)    # (1, 2, H, W)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=DEVICE, dtype=torch.float32),
        torch.arange(W, device=DEVICE, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)   # (1, 2, H, W)
    new_grid = grid + flow_b                                    # (1, 2, H, W)

    # Normalize pixel coordinates to [-1, 1] for grid_sample
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
    new_grid = new_grid.permute(0, 2, 3, 1)                    # (1, H, W, 2)

    warped = F.grid_sample(
        frame_b, new_grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return warped.squeeze(0)   # (3, H, W)


# ── VQ-VAE ────────────────────────────────────────────────────────────────────
def load_ae() -> MSVQVAE2Video:
    ae = MSVQVAE2Video(
        top_dim=TOP_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_bottom=K,
        commit_top=0.25, commit_bottom=0.25,
        norm="gn",
        use_vgg=False,      # VGG not needed for probe — saves memory
        soft_temp=1.0,
        use_ema=False,      # eval mode — EMA buffers not required
    ).to(DEVICE).eval()

    ckpt = torch.load(str(AE_CKPT), map_location="cpu", weights_only=False)
    ae.load_state_dict(ckpt["ae"], strict=False)
    log(f"VQ-VAE (K={K}) loaded from {AE_CKPT.name}")
    return ae


@torch.no_grad()
def encode_decode(ae: MSVQVAE2Video, clip: torch.Tensor) -> dict:
    """
    clip: (3, T, H, W) float in [0, 1]
    returns model output dict; recon is clamped to [0, 1]
    """
    x   = clip.unsqueeze(0).to(DEVICE)   # (1, 3, T, H, W)
    out = ae(x)
    out["recon"] = out["recon"].clamp(0.0, 1.0)
    return out


# ── BPP from VQ index entropy ─────────────────────────────────────────────────
def entropy_bpp(model_out: dict, T: int, H: int, W: int) -> float:
    """
    Lower-bound BPP estimate from Shannon entropy of VQ indices.
    Uses the entropy already computed inside MSVQVAE2Video._entropy_stats.
      entropy_top/bottom: bits per symbol
      idx_top:    (B, T/8, H/8, W/8)  → n_top symbols
      idx_bottom: (B, T/4, H/4, W/4)  → n_bot symbols
    """
    ent_t = model_out["entropy_top"]     # bits / symbol
    ent_b = model_out["entropy_bottom"]  # bits / symbol
    n_top = model_out["idx_top"].numel()
    n_bot = model_out["idx_bottom"].numel()
    total_bits = ent_t * n_top + ent_b * n_bot
    return total_bits / (T * H * W)


# ── metrics ───────────────────────────────────────────────────────────────────
def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)


# ── build residual clip ───────────────────────────────────────────────────────
def build_residual_clip(
    clip: torch.Tensor,
    raft_model,
    transforms,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    clip: (3, T, H, W) in [0, 1]  on CPU

    Returns:
      residual_clip:  (3, T, H, W) in [0, 1]
          Frame 0 = original frame 0 (I-frame, no motion compensation)
          Frames 1..T-1 = (actual - warped) * 0.5 + 0.5  (normalized residuals)

      predicted_clip: (3, T, H, W) in [0, 1]
          Frame 0 = frame 0
          Frames 1..T-1 = RAFT-warped predictions

      raw_residuals:  (T-1, H, W)  — absolute residual magnitudes (before normalization)
                      Used only for statistics.
    """
    C, T, H, W = clip.shape
    residual_clip  = clip.clone()
    predicted_clip = clip.clone()
    raw_residuals  = []

    for t in range(1, T):
        f_prev = clip[:, t - 1, :, :].to(DEVICE)   # (3, H, W)
        f_curr = clip[:, t,     :, :].to(DEVICE)   # (3, H, W)

        flow    = compute_flow(raft_model, transforms, f_prev, f_curr)
        f_pred  = warp_frame(f_prev, flow)           # (3, H, W) in [0, 1]
        residual = f_curr - f_pred                   # (3, H, W) in [-1, 1]

        # Track raw residual magnitude for statistics
        raw_residuals.append(residual.abs().mean(dim=0).cpu())   # (H, W)

        # Normalize residual to [0, 1] so VQ-VAE can encode it
        # Decoder output (Sigmoid) is in [0,1]; denormalize at reconstruction time.
        residual_norm = residual * 0.5 + 0.5          # maps [-1,1] → [0,1]

        residual_clip[:, t, :, :] = residual_norm.cpu()
        predicted_clip[:, t, :, :] = f_pred.cpu()

    raw_residuals_t = torch.stack(raw_residuals, dim=0)   # (T-1, H, W)
    return residual_clip, predicted_clip, raw_residuals_t


def reconstruct_from_residual(
    recon_residual_clip: torch.Tensor,
    predicted_clip: torch.Tensor,
) -> torch.Tensor:
    """
    Undo the residual normalization and add prediction back.
    recon_residual_clip: (3, T, H, W) in [0, 1]  — VQ-VAE decoded residuals
    predicted_clip:      (3, T, H, W) in [0, 1]  — warped predictions
    returns final_recon: (3, T, H, W) in [0, 1]
    """
    final = predicted_clip.clone()
    T = recon_residual_clip.shape[1]
    for t in range(1, T):
        res_norm = recon_residual_clip[:, t, :, :]      # [0, 1]
        res      = res_norm * 2.0 - 1.0                  # back to [-1, 1]
        pred     = predicted_clip[:, t, :, :]
        final[:, t, :, :] = (pred + res).clamp(0.0, 1.0)
    return final


# ── visualisation helpers ─────────────────────────────────────────────────────
def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [0,1] → (H, W, 3) uint8"""
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def residual_to_heatmap(res: torch.Tensor) -> np.ndarray:
    """res: (H, W) magnitude → (H, W, 3) heatmap (viridis)"""
    arr  = res.cpu().numpy()
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    cmap = plt.get_cmap("viridis")
    return (cmap(norm)[:, :, :3] * 255).astype(np.uint8)


def save_sample(
    clip_idx: int,
    clip: torch.Tensor,
    predicted_clip: torch.Tensor,
    residual_clip: torch.Tensor,
    recon_raw: torch.Tensor,
    final_recon: torch.Tensor,
    raw_residuals: torch.Tensor,
    t_show: int = 15,
):
    """Save a 2-row overview figure for one clip at frame t_show."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Clip {clip_idx:03d}  |  Frame t={t_show}", fontsize=13)

    titles = [
        "Original frame_t",
        "RAFT prediction (warped t-1)",
        "Residual magnitude",
        "Residual (normalized, fed to VQ-VAE)",
        "VQ-VAE recon (raw baseline)",
        "Motion + residual recon (proposed)",
        f"PSNR raw: {psnr(recon_raw[:, t_show], clip[:, t_show]):.1f} dB",
        f"PSNR proposed: {psnr(final_recon[:, t_show], clip[:, t_show]):.1f} dB",
    ]
    images = [
        tensor_to_np(clip[:, t_show]),
        tensor_to_np(predicted_clip[:, t_show]),
        residual_to_heatmap(raw_residuals[t_show - 1]),
        tensor_to_np(residual_clip[:, t_show]),
        tensor_to_np(recon_raw[:, t_show]),
        tensor_to_np(final_recon[:, t_show]),
        tensor_to_np(torch.abs(clip[:, t_show] - recon_raw[:, t_show])),
        tensor_to_np(torch.abs(clip[:, t_show] - final_recon[:, t_show])),
    ]

    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    path = SAMPLES_DIR / f"clip_{clip_idx:03d}_overview.png"
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(rows: list[dict]):
    bpp_raw  = [r["bpp_raw"]      for r in rows]
    bpp_res  = [r["bpp_residual"] for r in rows]
    mag_raw  = [r["mean_abs_raw_frame"]  for r in rows]
    mag_res  = [r["mean_abs_residual"]   for r in rows]
    psnr_raw = [r["psnr_raw"]    for r in rows]
    psnr_mc  = [r["psnr_mc"]     for r in rows]
    used_t_raw = [r["used_top_raw"]  for r in rows]
    used_t_res = [r["used_top_res"]  for r in rows]
    used_b_raw = [r["used_bot_raw"]  for r in rows]
    used_b_res = [r["used_bot_res"]  for r in rows]

    # ── Plot 1: Residual magnitude distribution ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Residual vs Raw Frame Pixel Magnitude", fontsize=13)
    axes[0].hist(mag_raw, bins=30, alpha=0.6, color="steelblue", label="Raw frames")
    axes[0].hist(mag_res, bins=30, alpha=0.6, color="tomato",    label="Residuals")
    axes[0].set_xlabel("Mean absolute pixel value")
    axes[0].set_ylabel("Number of clips")
    axes[0].set_title("Distribution across clips")
    axes[0].legend()

    ratio = [r / (b + 1e-8) for r, b in zip(mag_res, mag_raw)]
    axes[1].hist(ratio, bins=30, color="seagreen", alpha=0.8)
    axes[1].axvline(1.0, color="red", linestyle="--", label="ratio=1 (no gain)")
    axes[1].set_xlabel("Residual / Raw magnitude ratio  (lower = better)")
    axes[1].set_ylabel("Number of clips")
    axes[1].set_title(f"Magnitude ratio  (mean={np.mean(ratio):.2f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "01_residual_magnitude.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log("  Saved: 01_residual_magnitude.png")

    # ── Plot 2: BPP comparison ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("BPP Comparison: Raw vs Motion-Compensated Residuals", fontsize=13)

    axes[0].scatter(bpp_raw, bpp_res, alpha=0.5, s=20, color="darkorange")
    lim = max(max(bpp_raw), max(bpp_res)) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x (no gain)")
    axes[0].set_xlabel("BPP raw (baseline)")
    axes[0].set_ylabel("BPP residual (proposed)")
    axes[0].set_title("Per-clip BPP scatter")
    axes[0].legend()

    bpp_reduction = [(r - p) / (r + 1e-8) * 100 for r, p in zip(bpp_raw, bpp_res)]
    axes[1].hist(bpp_reduction, bins=30, color="mediumseagreen", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", label="0% reduction")
    axes[1].set_xlabel("BPP reduction (%)  (positive = proposed is cheaper)")
    axes[1].set_ylabel("Number of clips")
    axes[1].set_title(f"BPP reduction  (mean={np.mean(bpp_reduction):.1f}%)")
    axes[1].legend()

    axes[2].scatter(psnr_raw, psnr_mc, alpha=0.5, s=20, color="royalblue")
    lim2 = max(max(psnr_raw), max(psnr_mc)) * 1.02
    axes[2].plot([0, lim2], [0, lim2], "r--", linewidth=1, label="y=x")
    axes[2].set_xlabel("PSNR raw VQ-VAE (dB)")
    axes[2].set_ylabel("PSNR motion+residual (dB)")
    axes[2].set_title("Reconstruction quality comparison")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "02_bpp_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log("  Saved: 02_bpp_comparison.png")

    # ── Plot 3: Codebook utilisation ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Codebook Utilisation (K={K}): Raw vs Residual", fontsize=13)

    axes[0].scatter(used_t_raw, used_t_res, alpha=0.5, s=20, color="purple")
    lim3 = K * 1.05
    axes[0].plot([0, lim3], [0, lim3], "r--", linewidth=1, label="y=x")
    axes[0].set_xlabel("Used top-level codes — raw")
    axes[0].set_ylabel("Used top-level codes — residual")
    axes[0].set_title(f"Top VQ (mean raw={np.mean(used_t_raw):.0f}, res={np.mean(used_t_res):.0f})")
    axes[0].legend()

    axes[1].scatter(used_b_raw, used_b_res, alpha=0.5, s=20, color="darkviolet")
    axes[1].plot([0, lim3], [0, lim3], "r--", linewidth=1, label="y=x")
    axes[1].set_xlabel("Used bottom-level codes — raw")
    axes[1].set_ylabel("Used bottom-level codes — residual")
    axes[1].set_title(f"Bottom VQ (mean raw={np.mean(used_b_raw):.0f}, res={np.mean(used_b_res):.0f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "03_codebook_utilization.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log("  Saved: 03_codebook_utilization.png")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    for d in [RESULTS_DIR, PLOTS_DIR, SAMPLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log(f"Device: {DEVICE}")
    log(f"Data:   {DATA_DIR}")
    log(f"AE:     {AE_CKPT}")

    # Load models
    raft_model, transforms = load_raft()
    ae = load_ae()

    # List test clips
    clip_files = sorted(DATA_DIR.glob("*.pt"))[:N_CLIPS]
    if not clip_files:
        raise FileNotFoundError(f"No .pt files found in {DATA_DIR}")
    log(f"Processing {len(clip_files)} clips ...")

    rows        = []
    sample_count = 0
    csv_path    = RESULTS_DIR / "probe_metrics.csv"

    fieldnames = [
        "clip", "T", "H", "W",
        "bpp_raw", "bpp_residual", "bpp_reduction_pct",
        "entropy_top_raw", "entropy_top_res",
        "entropy_bot_raw", "entropy_bot_res",
        "used_top_raw", "used_top_res",
        "used_bot_raw", "used_bot_res",
        "mean_abs_raw_frame", "mean_abs_residual", "magnitude_ratio",
        "psnr_raw", "psnr_mc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for clip_idx, clip_path in enumerate(tqdm(clip_files, desc="Probe")):
            # ── load clip ──────────────────────────────────────────────────────
            raw = torch.load(str(clip_path), map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and "tensor" in raw:
                raw = raw["tensor"]
            clip = raw.float()
            if clip.max() > 1.0:
                clip = (clip / 255.0).clamp(0.0, 1.0)
            if clip.ndim == 3:                     # (T, H, W) grayscale edge case
                clip = clip.unsqueeze(0).repeat(3, 1, 1, 1)
            assert clip.ndim == 4 and clip.shape[0] == 3, f"Expected (3,T,H,W), got {clip.shape}"

            C, T, H, W = clip.shape

            # ── baseline: encode raw clip ──────────────────────────────────────
            out_raw  = encode_decode(ae, clip)
            recon_raw = out_raw["recon"].squeeze(0).cpu()   # (3, T, H, W)
            bpp_raw   = entropy_bpp(out_raw, T, H, W)
            psnr_raw_val = psnr(recon_raw, clip)

            # ── build residual clip ────────────────────────────────────────────
            residual_clip, predicted_clip, raw_residuals = build_residual_clip(
                clip, raft_model, transforms
            )

            # ── encode residual clip ───────────────────────────────────────────
            out_res   = encode_decode(ae, residual_clip)
            recon_res = out_res["recon"].squeeze(0).cpu()   # decoded residuals in [0,1]
            bpp_res   = entropy_bpp(out_res, T, H, W)

            # ── reconstruct final frames from motion + decoded residual ────────
            final_recon = reconstruct_from_residual(recon_res, predicted_clip)
            psnr_mc_val = psnr(final_recon, clip)

            # ── statistics ────────────────────────────────────────────────────
            mean_abs_raw  = clip[:, 1:, :, :].abs().mean().item()
            mean_abs_res  = raw_residuals.abs().mean().item()
            mag_ratio     = mean_abs_res / (mean_abs_raw + 1e-8)
            bpp_reduction = (bpp_raw - bpp_res) / (bpp_raw + 1e-8) * 100.0

            row = {
                "clip"              : clip_path.name,
                "T": T, "H": H, "W": W,
                "bpp_raw"           : round(bpp_raw,  5),
                "bpp_residual"      : round(bpp_res,  5),
                "bpp_reduction_pct" : round(bpp_reduction, 2),
                "entropy_top_raw"   : round(out_raw["entropy_top"],    4),
                "entropy_top_res"   : round(out_res["entropy_top"],    4),
                "entropy_bot_raw"   : round(out_raw["entropy_bottom"], 4),
                "entropy_bot_res"   : round(out_res["entropy_bottom"], 4),
                "used_top_raw"      : out_raw["used_top"],
                "used_top_res"      : out_res["used_top"],
                "used_bot_raw"      : out_raw["used_bottom"],
                "used_bot_res"      : out_res["used_bottom"],
                "mean_abs_raw_frame": round(mean_abs_raw, 5),
                "mean_abs_residual" : round(mean_abs_res, 5),
                "magnitude_ratio"   : round(mag_ratio,    4),
                "psnr_raw"          : round(psnr_raw_val, 3),
                "psnr_mc"           : round(psnr_mc_val,  3),
            }
            writer.writerow(row)
            rows.append(row)

            # ── save visual sample ─────────────────────────────────────────────
            if sample_count < N_SAMPLE_CLIPS:
                save_sample(
                    clip_idx, clip, predicted_clip,
                    residual_clip, recon_raw, final_recon, raw_residuals,
                )
                sample_count += 1

    log(f"CSV written: {csv_path}")

    # ── aggregate summary ─────────────────────────────────────────────────────
    def avg(key): return float(np.mean([r[key] for r in rows]))

    summary = {
        "n_clips"               : len(rows),
        "mean_bpp_raw"          : round(avg("bpp_raw"),            5),
        "mean_bpp_residual"     : round(avg("bpp_residual"),       5),
        "mean_bpp_reduction_pct": round(avg("bpp_reduction_pct"),  2),
        "mean_magnitude_ratio"  : round(avg("magnitude_ratio"),    4),
        "mean_psnr_raw"         : round(avg("psnr_raw"),           3),
        "mean_psnr_mc"          : round(avg("psnr_mc"),            3),
        "mean_entropy_top_raw"  : round(avg("entropy_top_raw"),    4),
        "mean_entropy_top_res"  : round(avg("entropy_top_res"),    4),
        "mean_entropy_bot_raw"  : round(avg("entropy_bot_raw"),    4),
        "mean_entropy_bot_res"  : round(avg("entropy_bot_res"),    4),
        "mean_used_top_raw"     : round(avg("used_top_raw"),       1),
        "mean_used_top_res"     : round(avg("used_top_res"),       1),
        "mean_used_bot_raw"     : round(avg("used_bot_raw"),       1),
        "mean_used_bot_res"     : round(avg("used_bot_res"),       1),
    }

    summary_path = RESULTS_DIR / "probe_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary written: {summary_path}")

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROBE SUMMARY")
    print("=" * 60)
    print(f"  Clips evaluated      : {summary['n_clips']}")
    print(f"")
    print(f"  BPP (raw baseline)   : {summary['mean_bpp_raw']:.5f}")
    print(f"  BPP (residuals)      : {summary['mean_bpp_residual']:.5f}")
    print(f"  BPP reduction        : {summary['mean_bpp_reduction_pct']:.1f}%")
    print(f"")
    print(f"  Residual/raw mag     : {summary['mean_magnitude_ratio']:.3f}  (< 1.0 = residuals smaller)")
    print(f"")
    print(f"  PSNR raw recon       : {summary['mean_psnr_raw']:.2f} dB")
    print(f"  PSNR motion+residual : {summary['mean_psnr_mc']:.2f} dB")
    print(f"")
    print(f"  Entropy top  raw/res : {summary['mean_entropy_top_raw']:.3f} / {summary['mean_entropy_top_res']:.3f} bits/sym")
    print(f"  Entropy bot  raw/res : {summary['mean_entropy_bot_raw']:.3f} / {summary['mean_entropy_bot_res']:.3f} bits/sym")
    print(f"  Used codes top raw/res : {summary['mean_used_top_raw']:.0f} / {summary['mean_used_top_res']:.0f}")
    print(f"  Used codes bot raw/res : {summary['mean_used_bot_raw']:.0f} / {summary['mean_used_bot_res']:.0f}")
    print("=" * 60)

    # ── interpret result ───────────────────────────────────────────────────────
    reduction = summary["mean_bpp_reduction_pct"]
    print("\nINTERPRETATION:")
    if reduction >= 30:
        print(f"  ✓ STRONG signal ({reduction:.1f}% BPP reduction).")
        print("    Hypothesis confirmed. Proceed to Phase 2 (dual-codebook architecture).")
    elif reduction >= 15:
        print(f"  ~ MODERATE signal ({reduction:.1f}% BPP reduction).")
        print("    Hypothesis holds. Check visual samples — RAFT may be noisy at 64x64.")
        print("    Still worth proceeding to Phase 2.")
    else:
        print(f"  ✗ WEAK signal ({reduction:.1f}% BPP reduction).")
        print("    Investigate: check sample visualizations for flow quality.")
        print("    Consider: RAFT_UPSAMPLE > 4, or alternative flow (UniMatch).")

    # ── generate plots ─────────────────────────────────────────────────────────
    log("\nGenerating plots ...")
    plot_results(rows)
    log(f"All plots saved to: {PLOTS_DIR}")
    log("Done.")


if __name__ == "__main__":
    main()
