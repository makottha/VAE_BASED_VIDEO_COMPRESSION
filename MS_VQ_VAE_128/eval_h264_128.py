"""
eval_h264_128.py — H.264 baseline evaluation at 128x128

Reads test .pt tensors from dataset_128/test/tensors/, encodes each clip
with H.264 at multiple CRF values, decodes, and computes BPP/PSNR/SSIM/LPIPS.

CRF range extended (28-51) to cover our model's BPP range (0.05-0.12).

Outputs saved to outputs_msvqvae_128/h264_baseline/:
  h264_per_clip.csv   -- per-clip metrics
  h264_summary.csv    -- mean per CRF  (use this in paper)
  h264_eval_log.txt   -- timestamped log

Usage:
  conda activate vae_env
  cd MS_VQ_VAE_128
  python eval_h264_128.py
"""

import os
import csv
import time
import math
import glob
import tempfile
import platform
import subprocess

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from imageio_ffmpeg import get_ffmpeg_exe
import lpips as lpips_lib

# ---- CONFIG -----------------------------------------------------------------
TEST_DIR  = "../dataset_128/test/tensors"
OUT_DIR   = "./outputs_msvqvae_128/h264_baseline"
N_CLIPS   = 500          # keep consistent with 64x64 baseline
CRFS      = [28, 32, 36, 40, 44, 48]   # extended range to reach BPP ~0.05
WIDTH     = 128
HEIGHT    = 128
T_FRAMES  = 32
FPS       = 16
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

FFMPEG = get_ffmpeg_exe()


def pick_codec():
    nul = "NUL" if platform.system().lower().startswith("win") else "/dev/null"
    for codec in ("libx264", "h264_mf", "h264_nvenc"):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"color=size={WIDTH}x{HEIGHT}:rate=1:color=black",
               "-t", "1", "-c:v", codec, "-pix_fmt", "yuv420p",
               "-f", "mp4", "-y", nul]
        if subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
            return codec
    raise RuntimeError("No working H.264 encoder found.")


CODEC = pick_codec()
print(f"[INFO] Codec: {CODEC}  Device: {DEVICE}")

lpips_model = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
print("[INFO] LPIPS model loaded.")


def tensor_to_frames(pt_path: str):
    """Load .pt tensor (3,T,H,W) -> numpy (T,H,W,3) uint8."""
    t = torch.load(pt_path, weights_only=True)   # (3,32,128,128)
    t = t.float().clamp(0, 1)
    # (3,T,H,W) -> (T,H,W,3)
    frames = t.permute(1, 2, 3, 0).numpy()
    return (frames * 255).astype(np.uint8)


def write_avi(frames: np.ndarray, path: str):
    """Write (T,H,W,3) RGB uint8 frames to an AVI file."""
    h, w = frames.shape[1], frames.shape[2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), FPS, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


def encode_h264(in_path: str, out_path: str, crf: int):
    cmd = [
        FFMPEG, "-y", "-i", in_path,
        "-vf", f"fps={FPS}",
        "-c:v", CODEC, "-preset", "medium", "-tune", "psnr",
        "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-frames:v", str(T_FRAMES),
        out_path
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[:300]}")


def read_video(path: str, max_frames: int = T_FRAMES):
    """Read video -> (T,H,W,3) float32 [0,1]."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames) if frames else None


def compute_metrics(ref: np.ndarray, dec: np.ndarray):
    """ref, dec: (T,H,W,3) float32 [0,1]. Returns psnr, ssim, lpips."""
    T = min(len(ref), len(dec))
    ref, dec = ref[:T], dec[:T]

    psnrs, ssims = [], []
    for r, d in zip(ref, dec):
        psnrs.append(sk_psnr(r, d, data_range=1.0))
        try:
            ssims.append(sk_ssim(r, d, data_range=1.0, channel_axis=2))
        except TypeError:
            ssims.append(sk_ssim(r, d, data_range=1.0, multichannel=True))

    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float().to(DEVICE) * 2 - 1
    dec_t = torch.from_numpy(dec).permute(0, 3, 1, 2).float().to(DEVICE) * 2 - 1
    with torch.no_grad():
        lp = float(lpips_model(dec_t, ref_t).mean().item())

    return float(np.mean(psnrs)), float(np.mean(ssims)), lp


def bpp_from_file(path: str, width: int, height: int, frames: int) -> float:
    return os.path.getsize(path) * 8.0 / (width * height * frames)


# ---- MAIN -------------------------------------------------------------------

all_pts = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))[:N_CLIPS]
print(f"[INFO] Evaluating {len(all_pts)} clips  CRFs={CRFS}  Resolution={WIDTH}x{HEIGHT}")

per_clip_csv = os.path.join(OUT_DIR, "h264_per_clip.csv")
summary_csv  = os.path.join(OUT_DIR, "h264_summary.csv")
log_path     = os.path.join(OUT_DIR, "h264_eval_log.txt")

fieldnames = ["clip", "crf", "bpp", "psnr", "ssim", "lpips", "frames"]
agg = {crf: {"bpp": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0}
       for crf in CRFS}

run_ts = time.strftime("%Y-%m-%d %H:%M:%S")

with open(per_clip_csv, "w", newline="") as f, open(log_path, "a") as log_f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    log_f.write(f"\n{'='*60}\n")
    log_f.write(f"H.264 Baseline 128x128 -- {run_ts}\n")
    log_f.write(f"Clips: {len(all_pts)}  CRFs: {CRFS}\n")
    log_f.write(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, pt_path in enumerate(all_pts, 1):
            clip_name = os.path.basename(pt_path)

            try:
                frames_uint8 = tensor_to_frames(pt_path)
                ref = frames_uint8.astype(np.float32) / 255.0

                # Write source AVI once per clip
                src_avi = os.path.join(tmpdir, "src.avi")
                write_avi(frames_uint8, src_avi)

                for crf in CRFS:
                    enc_path = os.path.join(tmpdir, f"enc_crf{crf}.mp4")
                    try:
                        encode_h264(src_avi, enc_path, crf)
                        dec = read_video(enc_path)
                        if dec is None:
                            continue

                        n_frames = min(len(ref), len(dec))
                        bpp  = bpp_from_file(enc_path, WIDTH, HEIGHT, n_frames)
                        psnr, ssim, lp = compute_metrics(ref, dec)

                        writer.writerow({
                            "clip": clip_name, "crf": crf,
                            "bpp": round(bpp, 6), "psnr": round(psnr, 4),
                            "ssim": round(ssim, 6), "lpips": round(lp, 6),
                            "frames": n_frames,
                        })
                        f.flush()

                        agg[crf]["bpp"]   += bpp
                        agg[crf]["psnr"]  += psnr
                        agg[crf]["ssim"]  += ssim
                        agg[crf]["lpips"] += lp
                        agg[crf]["n"]     += 1

                    except Exception as e:
                        print(f"  [ERR] {clip_name} CRF{crf}: {e}")

            except Exception as e:
                print(f"  [SKIP] {clip_name}: {e}")
                continue

            if i % 50 == 0:
                msg = f"[{i:04d}/{len(all_pts)}] "
                for crf in CRFS:
                    n = max(1, agg[crf]["n"])
                    msg += (f"CRF{crf}: bpp={agg[crf]['bpp']/n:.4f} "
                            f"lpips={agg[crf]['lpips']/n:.4f}  ")
                print(msg, flush=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'CRF':<6} {'N':>5} {'BPP':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"{'-'*70}")

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

    print(f"{'='*70}")

    with open(summary_csv, "w", newline="") as sf:
        writer2 = csv.DictWriter(sf, fieldnames=["crf", "n", "bpp", "psnr", "ssim", "lpips"])
        writer2.writeheader()
        writer2.writerows(summary_rows)

    log_f.write(f"\n{'CRF':<6} {'N':>5} {'BPP':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}\n")
    log_f.write(f"{'-'*50}\n")
    for row in summary_rows:
        log_f.write(f"{row['crf']:<6} {row['n']:>5} {row['bpp']:>8.4f} "
                    f"{row['psnr']:>8.2f} {row['ssim']:>8.4f} {row['lpips']:>8.4f}\n")
    log_f.write(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\n[INFO] Per-clip CSV -> {per_clip_csv}")
print(f"[INFO] Summary CSV  -> {summary_csv}")
print(f"[INFO] Log          -> {log_path}")
