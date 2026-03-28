# eval_h264.py
"""
H.264 baseline evaluation: PSNR, SSIM, LPIPS, BPP for CRF 28/32/36.

Encodes test AVI clips with H.264 at each CRF, decodes, and computes
all metrics against the original. Results saved to:
  outputs_msvqvae_rd/h264_baseline/
    h264_per_clip.csv       — per-clip rows (clip, crf, bpp, psnr, ssim, lpips)
    h264_summary.csv        — one row per CRF (mean over N clips)  ← use in paper
    h264_eval_log.txt       — timestamped log for future reference

Usage:
    python eval_h264.py

Requires: pip install opencv-python scikit-image imageio-ffmpeg lpips
"""

import os
import csv
import time
import glob
import subprocess
import platform
import tempfile
import math

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from imageio_ffmpeg import get_ffmpeg_exe
import lpips as lpips_lib

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
CLIPS_DIR   = "../dataset_human_readable_64/test/clips"
OUT_DIR     = "./outputs_msvqvae_rd/h264_baseline"
N_CLIPS     = 500       # match model eval count for fair comparison
CRFS        = [28, 32, 36]
WIDTH       = 64
HEIGHT      = 64
T_FRAMES    = 32
FPS         = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
FFMPEG = get_ffmpeg_exe()

def pick_codec():
    nul = "NUL" if platform.system().lower().startswith("win") else "/dev/null"
    for codec in ("libx264", "h264_mf", "h264_nvenc"):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=size=64x64:rate=1:color=black",
               "-t", "1", "-c:v", codec, "-pix_fmt", "yuv420p",
               "-f", "mp4", "-y", nul]
        if subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
            return codec
    raise RuntimeError("No working H.264 encoder found.")

CODEC = pick_codec()
print(f"[INFO] Using codec: {CODEC}  device: {DEVICE}")

# LPIPS model — VGG backbone, frozen
lpips_model = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
print("[INFO] LPIPS model loaded.")

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def read_video(path, max_frames=T_FRAMES):
    """Read video to float32 RGB (T,H,W,3) in [0,1]."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames) if frames else None


def encode_h264(in_path, out_path, crf):
    cmd = [
        FFMPEG, "-y", "-i", str(in_path),
        "-vf", f"fps={FPS}",
        "-c:v", CODEC, "-preset", "medium", "-tune", "psnr",
        "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-frames:v", str(T_FRAMES),
        str(out_path)
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[:300]}")


def compute_metrics(ref, dec):
    """
    ref, dec: (T, H, W, 3) float32 [0,1]
    Returns: psnr, ssim, lpips_score (all averaged over frames)
    """
    T = min(len(ref), len(dec))
    ref, dec = ref[:T], dec[:T]

    psnrs, ssims = [], []
    for r, d in zip(ref, dec):
        psnrs.append(sk_psnr(r, d, data_range=1.0))
        try:
            ssims.append(sk_ssim(r, d, data_range=1.0, channel_axis=2))
        except TypeError:
            ssims.append(sk_ssim(r, d, data_range=1.0, multichannel=True))

    # LPIPS: batch all frames at once for efficiency
    # Convert (T,H,W,3) → (T,3,H,W) tensor in [-1,1]
    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float().to(DEVICE) * 2 - 1
    dec_t = torch.from_numpy(dec).permute(0, 3, 1, 2).float().to(DEVICE) * 2 - 1
    with torch.no_grad():
        lp = float(lpips_model(dec_t, ref_t).mean().item())

    return float(np.mean(psnrs)), float(np.mean(ssims)), lp


def bpp_from_file(path, width, height, frames):
    return os.path.getsize(path) * 8.0 / (width * height * frames)

# -----------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------
all_clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.avi")))[:N_CLIPS]
print(f"[INFO] Evaluating {len(all_clips)} clips at CRFs {CRFS}")

per_clip_csv  = os.path.join(OUT_DIR, "h264_per_clip.csv")
summary_csv   = os.path.join(OUT_DIR, "h264_summary.csv")
log_path      = os.path.join(OUT_DIR, "h264_eval_log.txt")

fieldnames = ["clip", "crf", "bpp", "psnr", "ssim", "lpips", "frames"]

# accumulators: {crf: {metric: total, n: count}}
agg = {crf: {"bpp": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0}
       for crf in CRFS}

run_ts = time.strftime("%Y-%m-%d %H:%M:%S")

with open(per_clip_csv, "w", newline="") as f, \
     open(log_path, "a") as log_f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    log_f.write(f"\n{'='*60}\n")
    log_f.write(f"H.264 Baseline Eval — {run_ts}\n")
    log_f.write(f"Clips: {len(all_clips)}  CRFs: {CRFS}  Dataset: {CLIPS_DIR}\n")
    log_f.write(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, clip_path in enumerate(all_clips, 1):
            clip_name = os.path.basename(clip_path)
            ref = read_video(clip_path)
            if ref is None or len(ref) < 4:
                print(f"  [SKIP] {clip_name}")
                continue

            for crf in CRFS:
                out_path = os.path.join(tmpdir, f"enc_crf{crf}.mp4")
                try:
                    encode_h264(clip_path, out_path, crf)
                    dec = read_video(out_path)
                    if dec is None:
                        continue

                    bpp   = bpp_from_file(out_path, WIDTH, HEIGHT, min(len(ref), len(dec)))
                    psnr, ssim, lp = compute_metrics(ref, dec)

                    writer.writerow({
                        "clip": clip_name, "crf": crf,
                        "bpp": round(bpp, 6), "psnr": round(psnr, 4),
                        "ssim": round(ssim, 6), "lpips": round(lp, 6),
                        "frames": min(len(ref), len(dec)),
                    })
                    f.flush()

                    agg[crf]["bpp"]   += bpp
                    agg[crf]["psnr"]  += psnr
                    agg[crf]["ssim"]  += ssim
                    agg[crf]["lpips"] += lp
                    agg[crf]["n"]     += 1

                except Exception as e:
                    print(f"  [ERR] {clip_name} CRF{crf}: {e}")

            if i % 50 == 0:
                msg = f"[{i:04d}/{len(all_clips)}] "
                for crf in CRFS:
                    n = max(1, agg[crf]["n"])
                    msg += (f"CRF{crf}: bpp={agg[crf]['bpp']/n:.4f} "
                            f"psnr={agg[crf]['psnr']/n:.2f} "
                            f"ssim={agg[crf]['ssim']/n:.4f} "
                            f"lpips={agg[crf]['lpips']/n:.4f}  ")
                print(msg)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'CRF':<6} {'N':>5} {'BPP':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-"*70)

    summary_rows = []
    for crf in CRFS:
        n = max(1, agg[crf]["n"])
        row = {
            "crf":   crf,
            "n":     agg[crf]["n"],
            "bpp":   round(agg[crf]["bpp"]   / n, 6),
            "psnr":  round(agg[crf]["psnr"]  / n, 4),
            "ssim":  round(agg[crf]["ssim"]  / n, 6),
            "lpips": round(agg[crf]["lpips"] / n, 6),
        }
        summary_rows.append(row)
        print(f"{crf:<6} {row['n']:>5} {row['bpp']:>8.4f} {row['psnr']:>8.2f} "
              f"{row['ssim']:>8.4f} {row['lpips']:>8.4f}")

    print("="*70)

    with open(summary_csv, "w", newline="") as sf:
        writer2 = csv.DictWriter(sf, fieldnames=["crf", "n", "bpp", "psnr", "ssim", "lpips"])
        writer2.writeheader()
        writer2.writerows(summary_rows)

    # Append summary to log
    log_f.write(f"\n{'CRF':<6} {'N':>5} {'BPP':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}\n")
    log_f.write("-"*50 + "\n")
    for row in summary_rows:
        log_f.write(f"{row['crf']:<6} {row['n']:>5} {row['bpp']:>8.4f} "
                    f"{row['psnr']:>8.2f} {row['ssim']:>8.4f} {row['lpips']:>8.4f}\n")
    log_f.write(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\n[INFO] Per-clip CSV  -> {per_clip_csv}")
print(f"[INFO] Summary CSV   -> {summary_csv}")
print(f"[INFO] Eval log      -> {log_path}")
