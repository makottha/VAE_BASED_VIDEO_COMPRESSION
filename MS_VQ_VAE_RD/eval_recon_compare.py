import os
import glob
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F

# Your local modules
from models import MSVQVAE2Video


# -----------------------------
# Reusable timestamp logger
# -----------------------------
def log(tag: str, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][{tag}] {msg}")


# -----------------------------
# Simple metrics
# -----------------------------
def psnr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> float:
    # x,y: (1,3,T,H,W) in [0,1]
    mse = F.mse_loss(x, y).item()
    mse = max(mse, eps)
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def try_ssim_torchmetrics(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """
    Attempts SSIM via torchmetrics if installed.
    Computes average SSIM over frames using torchmetrics.image.StructuralSimilarityIndexMeasure.
    Returns None if torchmetrics isn't available.
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except Exception:
        return None

    # x,y: (1,3,T,H,W) -> compute per-frame SSIM on (N,3,H,W)
    x2 = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, x.shape[-2], x.shape[-1])
    y2 = y.permute(0, 2, 1, 3, 4).reshape(-1, 3, y.shape[-2], y.shape[-1])

    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(x.device)
    with torch.no_grad():
        s = metric(x2, y2).item()
    return float(s)


# -----------------------------
# Video writer (best-effort)
# -----------------------------
def save_side_by_side_video(
    out_mp4: str,
    gt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    fps: int = 16
) -> None:
    """
    Saves a side-by-side MP4: [GT | A | B] horizontally per frame.
    gt,a,b: (1,3,T,H,W) in [0,1]
    """
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    gt = gt.clamp(0, 1)
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)

    # (T,H,W,3) uint8
    T = gt.shape[2]
    frames = []
    for t in range(T):
        f_gt = gt[0, :, t].permute(1, 2, 0)
        f_a  = a[0, :, t].permute(1, 2, 0)
        f_b  = b[0, :, t].permute(1, 2, 0)
        cat = torch.cat([f_gt, f_a, f_b], dim=1)  # concat W
        frames.append((cat * 255.0).byte().cpu())

    video = torch.stack(frames, dim=0)  # (T,H,3W,3)

    # Try torchvision first (usually available)
    try:
        import torchvision
        torchvision.io.write_video(out_mp4, video, fps=fps, video_codec="h264", options={"crf": "18"})
        return
    except Exception:
        pass

    # Fallback: imageio
    try:
        import imageio
        writer = imageio.get_writer(out_mp4, fps=fps)
        for t in range(video.shape[0]):
            writer.append_data(video[t].numpy())
        writer.close()
        return
    except Exception as e:
        raise RuntimeError(
            f"Could not write video. Install torchvision (with ffmpeg) or imageio. Error: {e}"
        )


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    TEST_DIR: str = "../dataset_64/test/tensors"
    AE_CKPT_PATH: str = "./outputs_msvqvae_rd/ae_epoch050.pt"
    RD_CKPT_PATH: str = "./outputs_msvqvae_rd/rd_epoch020.pt"
    OUT_DIR: str = "./outputs_msvqvae_rd/recon_compare"
    MAX_CLIPS: int = 2        # set -1 for all
    SAVE_VIDEOS: bool = True
    SAVE_EVERY: int = 1        # save 1 video every N clips
    FPS: int = 16

    # Must match training
    TOP_DIM: int = 256
    BOTTOM_DIM: int = 128
    K_TOP: int = 1024
    K_BOTTOM: int = 1024


def load_ae_from_ckpt(path: str, device: torch.device, cfg: Cfg) -> MSVQVAE2Video:
    ae = MSVQVAE2Video(
        top_dim=cfg.TOP_DIM,
        bottom_dim=cfg.BOTTOM_DIM,
        K_top=cfg.K_TOP,
        K_bottom=cfg.K_BOTTOM,
        commit_top=0.05,
        commit_bottom=0.1,
        norm="gn",
        use_vgg=False,   # eval recon only; faster
        vgg_layers=16
    ).to(device).eval()

    ckpt = torch.load(path, map_location="cpu")
    if "ae" in ckpt:
        ae.load_state_dict(ckpt["ae"])
    else:
        # if you ever saved raw state_dict
        ae.load_state_dict(ckpt)
    return ae


def main():
    cfg = Cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("EVAL", f"Using device: {device}")

    # Collect test clips
    paths = sorted(glob.glob(os.path.join(cfg.TEST_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No .pt tensors found in {cfg.TEST_DIR}")
    if cfg.MAX_CLIPS > 0:
        paths = paths[: cfg.MAX_CLIPS]
    log("EVAL", f"Found {len(paths)} test clips")

    # Load baseline AE (Stage A)
    log("LOAD", f"Loading baseline AE: {cfg.AE_CKPT_PATH}")
    ae_base = load_ae_from_ckpt(cfg.AE_CKPT_PATH, device, cfg)

    # Load RD AE (Stage C) – contains updated AE weights
    log("LOAD", f"Loading RD checkpoint: {cfg.RD_CKPT_PATH}")
    ckpt_rd = torch.load(cfg.RD_CKPT_PATH, map_location="cpu")

    log("LOAD", "Building AE for RD weights")
    ae_rd = MSVQVAE2Video(
        top_dim=cfg.TOP_DIM,
        bottom_dim=cfg.BOTTOM_DIM,
        K_top=cfg.K_TOP,
        K_bottom=cfg.K_BOTTOM,
        commit_top=0.05,
        commit_bottom=0.1,
        norm="gn",
        use_vgg=False,
        vgg_layers=16
    ).to(device).eval()

    if "ae" not in ckpt_rd:
        raise KeyError("RD checkpoint missing key 'ae'. Did you save rd checkpoints via save_ckpt(ae=...)?")
    ae_rd.load_state_dict(ckpt_rd["ae"])

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.OUT_DIR, "recon_metrics_compare.csv")

    # CSV header
    rows: List[Dict] = []
    ssim_available = True

    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            x = torch.load(p, map_location="cpu")  # (3,T,H,W)
            x = x.unsqueeze(0).to(device).float().clamp(0, 1)

            r_base = ae_base(x)["recon"]
            r_rd = ae_rd(x)["recon"]

            psnr_base = psnr_torch(x, r_base)
            psnr_rd = psnr_torch(x, r_rd)

            ssim_base = try_ssim_torchmetrics(x, r_base)
            ssim_rd = try_ssim_torchmetrics(x, r_rd)

            if (ssim_base is None) or (ssim_rd is None):
                ssim_available = False

            rows.append({
                "clip": os.path.basename(p),
                "psnr_base": psnr_base,
                "psnr_rd": psnr_rd,
                "ssim_base": (ssim_base if ssim_base is not None else ""),
                "ssim_rd": (ssim_rd if ssim_rd is not None else ""),
            })

            if cfg.SAVE_VIDEOS and (i % cfg.SAVE_EVERY == 0):
                out_mp4 = os.path.join(cfg.OUT_DIR, f"cmp_{i:05d}.mp4")
                save_side_by_side_video(out_mp4, x, r_base, r_rd, fps=cfg.FPS)
                log("VID", f"Wrote {out_mp4}")

            if i % 50 == 0:
                avg_psnr_base = sum(r["psnr_base"] for r in rows) / len(rows)
                avg_psnr_rd = sum(r["psnr_rd"] for r in rows) / len(rows)
                log("EVAL", f"[{i:05d}/{len(paths)}] avg_psnr_base={avg_psnr_base:.2f} avg_psnr_rd={avg_psnr_rd:.2f}")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["clip", "psnr_base", "psnr_rd", "ssim_base", "ssim_rd"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary
    avg_psnr_base = sum(r["psnr_base"] for r in rows) / len(rows)
    avg_psnr_rd = sum(r["psnr_rd"] for r in rows) / len(rows)

    if ssim_available:
        ssim_base_vals = [r["ssim_base"] for r in rows if isinstance(r["ssim_base"], float)]
        ssim_rd_vals = [r["ssim_rd"] for r in rows if isinstance(r["ssim_rd"], float)]
        avg_ssim_base = sum(ssim_base_vals) / len(ssim_base_vals) if ssim_base_vals else None
        avg_ssim_rd = sum(ssim_rd_vals) / len(ssim_rd_vals) if ssim_rd_vals else None
        log("DONE", f"AVG PSNR base={avg_psnr_base:.2f} RD={avg_psnr_rd:.2f} | AVG SSIM base={avg_ssim_base} RD={avg_ssim_rd}")
    else:
        log("DONE", f"AVG PSNR base={avg_psnr_base:.2f} RD={avg_psnr_rd:.2f} | SSIM skipped (torchmetrics not installed)")

    log("DONE", f"Wrote CSV: {csv_path}")
    log("DONE", f"Side-by-side videos (if enabled) are in: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()
