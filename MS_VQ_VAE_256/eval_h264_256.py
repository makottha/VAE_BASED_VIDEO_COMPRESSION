"""
eval_h264_256.py — H.264 baseline evaluation at 256×256

Loads the same 500 .pt tensor clips used in eval_codec_256.py,
encodes each with H.264 at CRF 28/32/36, decodes, and computes
PSNR, SSIM, LPIPS, BPP against the original tensor.

Results saved to: outputs_msvqvae_256/h264_baseline/
  h264_per_clip.csv   — per-clip rows (clip, crf, bpp, psnr, ssim, lpips)
  h264_summary.csv    — one row per CRF (mean over N clips)  ← use in paper

Usage:
    conda activate vae_env
    cd MS_VQ_VAE_256
    python eval_h264_256.py
"""

import os
import csv
import time
import glob
import subprocess
import tempfile

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from imageio_ffmpeg import get_ffmpeg_exe
import lpips as lpips_lib

# -----------------------------------------------------------------------
# Config — must match eval_codec_256.py
# -----------------------------------------------------------------------
TEST_DIR = "../dataset_256/test/tensors"
OUT_DIR  = "./outputs_msvqvae_256/h264_baseline"
N_CLIPS  = 500        # match model eval
CRFS     = [28, 32, 36]
WIDTH    = 256
HEIGHT   = 256
T_FRAMES = 32
FPS      = 16
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
FFMPEG = get_ffmpeg_exe()

def pick_codec():
    import platform
    nul = "NUL" if platform.system().lower().startswith("win") else "/dev/null"
    for codec in ("libx264", "h264_mf", "h264_nvenc"):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=size=256x256:rate=1:color=black",
               "-t", "1", "-c:v", codec, "-pix_fmt", "yuv420p",
               "-f", "mp4", "-y", nul]
        if subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
            return codec
    raise RuntimeError("No working H.264 encoder found.")

CODEC = pick_codec()
print(f"[INFO] Using codec:  {CODEC}")
print(f"[INFO] Device:       {DEVICE}")

lpips_model = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
print("[INFO] LPIPS model loaded.")

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def tensor_to_numpy(pt_path):
    """Load .pt clip (3,T,H,W) float32 [0,1] → numpy (T,H,W,3) [0,1]."""
    clip = torch.load(pt_path, weights_only=True).float().clamp(0, 1)
    # (3,T,H,W) → (T,H,W,3)
    return clip.permute(1, 2, 3, 0).numpy()


def write_video(frames_nhwc, out_path, fps=FPS):
    """Write (T,H,W,3) float32 [0,1] to a raw .avi file for H.264 encoding."""
    h, w = frames_nhwc.shape[1], frames_nhwc.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames_nhwc:
        bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()


def encode_h264(in_path, out_path, crf):
    cmd = [
        FFMPEG, "-y", "-i", str(in_path),
        "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
        "-c:v", CODEC, "-preset", "medium", "-tune", "psnr",
        "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-frames:v", str(T_FRAMES),
        str(out_path)
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[:300]}")


def read_video(path):
    """Read encoded video to float32 (T,H,W,3) [0,1]."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < T_FRAMES:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames) if frames else None


def compute_metrics(ref, dec):
    """ref, dec: (T,H,W,3) float32 [0,1]. Returns psnr, ssim, lpips."""
    T = min(len(ref), len(dec))
    ref, dec = ref[:T], dec[:T]

    psnrs, ssims = [], []
    for r, d in zip(ref, dec):
        psnrs.append(sk_psnr(r, d, data_range=1.0))
        ssims.append(sk_ssim(r, d, data_range=1.0, channel_axis=2))

    ref_t = torch.from_numpy(ref).permute(0, 3, 1, 2).float().to(DEVICE) * 2 - 1
    dec_t = torch.from_numpy(dec).permute(0, 3, 1, 2).float().to(DEVICE) * 2 - 1
    with torch.no_grad():
        lp = float(lpips_model(dec_t, ref_t).mean().item())

    return float(np.mean(psnrs)), float(np.mean(ssims)), lp


def bpp_from_file(path, w, h, t):
    return os.path.getsize(path) * 8.0 / (w * h * t)

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
pt_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))[:N_CLIPS]
print(f"[INFO] Evaluating {len(pt_files)} clips  CRFs={CRFS}  {WIDTH}×{HEIGHT}")

per_clip_csv = os.path.join(OUT_DIR, "h264_per_clip.csv")
summary_csv  = os.path.join(OUT_DIR, "h264_summary.csv")
log_path     = os.path.join(OUT_DIR, "h264_eval_log.txt")

agg = {crf: {"bpp": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0}
       for crf in CRFS}

with open(per_clip_csv, "w", newline="") as f, open(log_path, "a") as log_f:
    writer = csv.DictWriter(f, fieldnames=["clip", "crf", "bpp", "psnr", "ssim", "lpips", "frames"])
    writer.writeheader()

    log_f.write(f"\n{'='*60}\n")
    log_f.write(f"H.264 256×256 Eval — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Clips: {len(pt_files)}  CRFs: {CRFS}\n{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_avi = os.path.join(tmpdir, "ref.avi")

        for i, pt_path in enumerate(pt_files, 1):
            clip_name = os.path.basename(pt_path).replace(".pt", "")
            try:
                ref = tensor_to_numpy(pt_path)
            except Exception as e:
                print(f"  [SKIP] {clip_name}: {e}")
                continue

            write_video(ref, raw_avi)

            for crf in CRFS:
                enc_path = os.path.join(tmpdir, f"enc_crf{crf}.mp4")
                try:
                    encode_h264(raw_avi, enc_path, crf)
                    dec = read_video(enc_path)
                    if dec is None:
                        continue
                    t = min(len(ref), len(dec))
                    bpp          = bpp_from_file(enc_path, WIDTH, HEIGHT, t)
                    psnr, ssim, lp = compute_metrics(ref, dec)

                    writer.writerow({
                        "clip": clip_name, "crf": crf,
                        "bpp": round(bpp, 6), "psnr": round(psnr, 4),
                        "ssim": round(ssim, 6), "lpips": round(lp, 6),
                        "frames": t,
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
                msg = f"[{i:04d}/{len(pt_files)}] "
                for crf in CRFS:
                    n = max(1, agg[crf]["n"])
                    msg += (f"CRF{crf}: bpp={agg[crf]['bpp']/n:.4f} "
                            f"psnr={agg[crf]['psnr']/n:.2f} "
                            f"lpips={agg[crf]['lpips']/n:.4f}  ")
                print(msg)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "="*65)
print(f"{'CRF':<6} {'N':>5} {'BPP':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
print("-"*65)

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
print("="*65)

with open(summary_csv, "w", newline="") as sf:
    w2 = csv.DictWriter(sf, fieldnames=["crf", "n", "bpp", "psnr", "ssim", "lpips"])
    w2.writeheader()
    w2.writerows(summary_rows)

print(f"\n[INFO] Per-clip CSV -> {per_clip_csv}")
print(f"[INFO] Summary CSV  -> {summary_csv}")
print(f"[INFO] Log          -> {log_path}")
