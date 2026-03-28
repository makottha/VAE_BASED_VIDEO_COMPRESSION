#!/usr/bin/env python3
"""
RD points for MS-VQ-VAE, producing rd_msvqvae.csv just like the H.264 script.
- For each standardized clip (default: 64x64, 32f @16fps):
  * runs your model to get recon + discrete indices (top/bottom)
  * computes Proxy bpp and Actual bpp (zlib)
  * computes PSNR/SSIM vs the original
  * writes CSV rows with columns: clip, method, crf, bpp, psnr, ssim, frames
    - crf = "proxy" or "actual" to distinguish the two bpp types
- Also writes a small summary CSV averaged over clips.

Deps: pip install torch opencv-python scikit-image pandas
"""

import argparse, csv, glob, sys, zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ======== import your model class here ========
# CHANGE THIS to match your repo/file layout
from MS_VQ_VAE_2.deeper_vqvae2_entropy_model import MultiScaleVQVAE2   # <-- e.g., from models.vqvae import MultiScaleVQVAE2
# =============================================

# ---------------- IO utils ----------------
def list_inputs(maybe_glob_or_dir: str):
    p = Path(maybe_glob_or_dir)
    if p.exists() and p.is_dir():
        exts = (".avi", ".mp4", ".mov", ".mkv")
        files = []
        for ext in exts:
            files.extend(p.glob(f"*{ext}"))
        return sorted(str(fp) for fp in files)
    else:
        return sorted(glob.glob(maybe_glob_or_dir))

def read_video_rgb01(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(rgb)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames)  # (T,H,W,3)

def clip_metrics(ref_vid, test_vid):
    T = min(len(ref_vid), len(test_vid))
    ref, dec = ref_vid[:T], test_vid[:T]
    if ref.shape[1:3] != dec.shape[1:3]:
        raise ValueError(f"Shape mismatch: ref {ref.shape[1:3]} vs dec {dec.shape[1:3]}")
    psnrs, ssims = [], []
    for r, d in zip(ref, dec):
        psnrs.append(psnr(r, d, data_range=1.0))
        try:
            ssims.append(ssim(r, d, data_range=1.0, channel_axis=2))
        except TypeError:
            ssims.append(ssim(r, d, data_range=1.0, multichannel=True))
    return float(np.mean(psnrs)), float(np.mean(ssims)), T

# ---------------- rate utils ----------------
def proxy_bpp_from_indices(zt_idx, zb_idx, Kt, Kb, T, H, W):
    Nt = int(np.prod(zt_idx.shape)) if zt_idx is not None else 0
    Nb = int(np.prod(zb_idx.shape)) if zb_idx is not None else 0
    if Nt + Nb == 0:
        return None
    bits = Nt * np.log2(Kt) + Nb * np.log2(Kb)
    return float(bits / (T * H * W))

def actual_bpp_from_indices(zt_idx, zb_idx, T, H, W, dtype=np.uint16):
    if (zt_idx is None) or (zb_idx is None):
        return None
    payload = zt_idx.astype(dtype).tobytes(order="C") + zb_idx.astype(dtype).tobytes(order="C")
    comp = zlib.compress(payload, level=9)
    bits = len(comp) * 8
    return float(bits / (T * H * W))

def write_recon_mp4(path, frames_thwc, fps=16):
    T, H, W, _ = frames_thwc.shape
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames_thwc:
        bgr = cv2.cvtColor(np.clip(f*255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

# ---------------- model hooks ----------------
import torch
import torch.nn.functional as F

def load_model(weights_path=None, **ctor_kwargs):
    """Build model, optionally load weights, return (model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiScaleVQVAE2(**ctor_kwargs).to(device).eval()
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        # strip DataParallel prefix if present
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return (model, device)

@torch.no_grad()
def infer_clip(model_device, clip_bcthw):
    """
    Args:
      clip_bcthw: np.float32 [1,3,T,H,W] in [0,1]
    Returns:
      recon_thwc: (T,H,W,3) in [0,1]
      zt_idx:     (Tt,Ht,Wt) int32
      zb_idx:     (Tb,Hb,Wb) int32
    """
    model, device = model_device
    x = torch.from_numpy(clip_bcthw).float().to(device)

    # replicate your forward to access indices explicitly
    z_bottom = model.encoder_bottom(x)                         # [1,Cb,T/4,H/4,W/4]
    z_top    = model.encoder_top(z_bottom)                     # [1,Ct,T/8,H/8,W/8]

    z_top_q,    _, indices_top    = model.vq_top(z_top)        # indices_top: [1,T/8,H/8,W/8]
    z_bottom_q, _, indices_bottom = model.vq_bottom(z_bottom)  # indices_bottom: [1,T/4,H/4,W/4]

    z_top_up   = F.interpolate(z_top_q, size=z_bottom_q.shape[2:], mode='trilinear', align_corners=False)
    z_combined = torch.cat([z_top_up, z_bottom_q], dim=1)
    recon      = model.decoder(z_combined).clamp(0, 1)         # [1,3,T,H,W]

    recon_thwc = recon.squeeze(0).permute(1,2,3,0).cpu().numpy()
    zt_idx     = indices_top.squeeze(0).to(torch.int32).cpu().numpy()
    zb_idx     = indices_bottom.squeeze(0).to(torch.int32).cpu().numpy()
    return recon_thwc, zt_idx, zb_idx

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="./dataset_human_readable_64/test/clips/*.avi",
                    help="Glob or directory of standardized clips")
    ap.add_argument("--outdir", default="./dataset_human_readable_64/test/msvqvae_out/",
                    help="Output directory")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--weights", default=None, help="Path to model weights (optional)")
    ap.add_argument("--write_video", action="store_true", help="Save recon mp4s for inspection")
    ap.add_argument("--csv", default="rd_msvqvae.csv")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N clips (0 = all)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    inputs = list_inputs(args.inputs)
    if not inputs:
        print(f"No input videos found for: {args.inputs}", file=sys.stderr); sys.exit(1)
    if args.limit and args.limit > 0:
        inputs = inputs[:args.limit]
    print(f"Found {len(inputs)} input(s).")

    # build model & read codebook sizes for proxy-bpp automatically
    model_device = load_model(args.weights)
    model, _ = model_device
    Kt = int(getattr(model.vq_top, "num_embeddings"))
    Kb = int(getattr(model.vq_bottom, "num_embeddings"))
    print(f"[info] codebook sizes: top={Kt}, bottom={Kb}")

    csv_path = outdir / args.csv
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["clip", "method", "crf", "bpp", "psnr", "ssim", "frames"])  # same as H.264

        for in_path in inputs:
            # read standardized ref (T,H,W,3) in [0,1]
            ref = read_video_rgb01(in_path, max_frames=args.frames)
            T, H, W = ref.shape[0], ref.shape[1], ref.shape[2]

            # BCHW → BCTHW for your model
            clip_bcthw = np.transpose(ref, (3,0,1,2))[None, ...]   # (1,3,T,H,W)

            # run model
            recon_thwc, zt_idx, zb_idx = infer_clip(model_device, clip_bcthw)

            # metrics
            p, s, usedT = clip_metrics(ref, recon_thwc)

            # rates
            proxy_bpp  = proxy_bpp_from_indices(zt_idx, zb_idx, Kt, Kb, T, H, W)
            actual_bpp = actual_bpp_from_indices(zt_idx, zb_idx, T, H, W)

            # write 2 rows: proxy and actual (crf column carries the tag)
            if proxy_bpp is not None:
                w.writerow([in_path, "MSVQVAE", "proxy",  f"{proxy_bpp:.6f}", f"{p:.4f}", f"{s:.4f}", usedT])
            else:
                w.writerow([in_path, "MSVQVAE", "proxy",  "",                f"{p:.4f}", f"{s:.4f}", usedT])

            if actual_bpp is not None:
                w.writerow([in_path, "MSVQVAE", "actual", f"{actual_bpp:.6f}", f"{p:.4f}", f"{s:.4f}", usedT])
            else:
                w.writerow([in_path, "MSVQVAE", "actual", "",                  f"{p:.4f}", f"{s:.4f}", usedT])

            if args.write_video:
                write_recon_mp4(outdir / f"{Path(in_path).stem}_msvqvae.mp4", recon_thwc, fps=args.fps)

            print(f"[OK] {Path(in_path).name}: PSNR={p:.2f} SSIM={s:.4f} "
                  f"BPP(proxy)={'' if proxy_bpp is None else f'{proxy_bpp:.4f}'} "
                  f"BPP(actual)={'' if actual_bpp is None else f'{actual_bpp:.4f}'}")

    # summary (mean over clips, grouped by 'crf' tag: proxy/actual)
    try:
        df = pd.read_csv(csv_path)
        df["bpp"]  = pd.to_numeric(df["bpp"],  errors="coerce")
        df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
        df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
        summary = df.groupby("crf").agg(bpp=("bpp","mean"),
                                        psnr=("psnr","mean"),
                                        ssim=("ssim","mean"),
                                        n=("clip","count")).reset_index()
        print("\n=== MS-VQ-VAE RD summary (mean over clips) ===")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(summary)
        summary.to_csv(outdir / "rd_msvqvae_summary.csv", index=False)
    except Exception as e:
        print("Summary generation skipped due to error:", e)

if __name__ == "__main__":
    main()
