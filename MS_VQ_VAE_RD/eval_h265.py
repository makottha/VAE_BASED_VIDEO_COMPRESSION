"""
eval_h265.py — H.265/HEVC baseline evaluation at 64x64

Reads the same 500 AVI test clips used for H.264 evaluation, encodes each
with H.265/HEVC at multiple CRF values via ffmpeg, decodes, and computes
BPP / PSNR / SSIM / LPIPS.

CRF range extended to [28-48] (vs H.264's [28-36]) to cover the low-BPP
region our model operates in — H.265 is ~50% more efficient so higher CRFs
are needed to reach the same bitrate floor.

Outputs saved to outputs_msvqvae_rd/h265_baseline/:
  h265_per_clip.csv   -- per-clip metrics
  h265_summary.csv    -- mean per CRF  (use this in paper)
  h265_eval_log.txt   -- timestamped log

Usage:
  conda activate vae_env
  cd MS_VQ_VAE_RD
  python eval_h265.py
"""

import os
import csv
import time
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
CLIPS_DIR = "../dataset_human_readable_64/test/clips"
OUT_DIR   = "./outputs_msvqvae_rd/h265_baseline"
N_CLIPS   = 500
CRFS      = [28, 32, 36, 40, 44, 48]   # extended vs H.264's [28,32,36]
WIDTH     = 64
HEIGHT    = 64
T_FRAMES  = 32
FPS       = 16
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

FFMPEG = get_ffmpeg_exe()


def pick_codec() -> str:
    """Try H.265 encoders in preference order: libx265 > hevc_nvenc > hevc_mf."""
    nul = "NUL" if platform.system().lower().startswith("win") else "/dev/null"
    candidates = [
        ("libx265",    ["-crf", "35", "-preset", "medium"]),
        ("hevc_nvenc", ["-cq",  "35", "-preset", "p4"]),
        ("hevc_mf",    ["-q:v", "70"]),
    ]
    for codec, extra_flags in candidates:
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=size={WIDTH}x{HEIGHT}:rate=1:color=black",
            "-t", "1", "-c:v", codec, "-pix_fmt", "yuv420p",
            *extra_flags,
            "-f", "mp4", "-y", nul,
        ]
        if subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
            return codec
    raise RuntimeError("No working H.265/HEVC encoder found.")


CODEC = pick_codec()
print(f"[INFO] HEVC codec: {CODEC}  Device: {DEVICE}")

lpips_model = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
print("[INFO] LPIPS model loaded.")


def _crf_flags(codec: str, crf: int) -> list:
    if codec == "libx265":
        return ["-crf", str(crf), "-preset", "medium"]
    if codec == "hevc_nvenc":
        return ["-cq", str(min(crf, 51)), "-preset", "p4"]
    if codec == "hevc_mf":
        q = max(0, min(100, int(100 - (crf - 28) * (60 / 20))))
        return ["-q:v", str(q)]
    raise ValueError(f"Unknown codec: {codec}")


def read_video(path: str, max_frames: int = T_FRAMES) -> np.ndarray | None:
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


def encode_h265(in_path: str, out_path: str, crf: int):
    cmd = [
        FFMPEG, "-y", "-i", in_path,
        "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
        "-c:v", CODEC,
        *_crf_flags(CODEC, crf),
        "-pix_fmt", "yuv420p",
        "-frames:v", str(T_FRAMES),
        "-tag:v", "hvc1",
        out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[:400]}")


def compute_metrics(ref: np.ndarray, dec: np.ndarray) -> tuple[float, float, float]:
    """ref, dec: (T,H,W,3) float32 [0,1]. Returns (psnr, ssim, lpips)."""
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

all_clips = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.avi")))[:N_CLIPS]
print(f"[INFO] Evaluating {len(all_clips)} clips  CRFs={CRFS}  Resolution={WIDTH}x{HEIGHT}")

per_clip_csv = os.path.join(OUT_DIR, "h265_per_clip.csv")
summary_csv  = os.path.join(OUT_DIR, "h265_summary.csv")
log_path     = os.path.join(OUT_DIR, "h265_eval_log.txt")

fieldnames = ["clip", "crf", "bpp", "psnr", "ssim", "lpips", "frames"]
agg = {crf: {"bpp": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "n": 0}
       for crf in CRFS}

run_ts = time.strftime("%Y-%m-%d %H:%M:%S")

with open(per_clip_csv, "w", newline="") as f, open(log_path, "a") as log_f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    log_f.write(f"\n{'='*60}\n")
    log_f.write(f"H.265/HEVC Baseline 64x64 -- {run_ts}\n")
    log_f.write(f"Codec: {CODEC}  Clips: {len(all_clips)}  CRFs: {CRFS}\n")
    log_f.write(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, clip_path in enumerate(all_clips, 1):
            clip_name = os.path.basename(clip_path)

            try:
                ref = read_video(clip_path)
                if ref is None:
                    print(f"  [SKIP] {clip_name}: could not read")
                    continue

                for crf in CRFS:
                    enc_path = os.path.join(tmpdir, f"enc_crf{crf}.mp4")
                    try:
                        encode_h265(clip_path, enc_path, crf)
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
                msg = f"[{i:04d}/{len(all_clips)}] "
                for crf in CRFS:
                    n = max(1, agg[crf]["n"])
                    msg += (f"CRF{crf}: bpp={agg[crf]['bpp']/n:.4f} "
                            f"lpips={agg[crf]['lpips']/n:.4f}  ")
                print(msg, flush=True)

# ---- Summary ----------------------------------------------------------------
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

with open(log_path, "a") as log_f:
    log_f.write(f"\n{'CRF':<6} {'N':>5} {'BPP':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}\n")
    log_f.write(f"{'-'*50}\n")
    for row in summary_rows:
        log_f.write(f"{row['crf']:<6} {row['n']:>5} {row['bpp']:>8.4f} "
                    f"{row['psnr']:>8.2f} {row['ssim']:>8.4f} {row['lpips']:>8.4f}\n")
    log_f.write(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\n[INFO] Per-clip CSV -> {per_clip_csv}")
print(f"[INFO] Summary CSV  -> {summary_csv}")
print(f"[INFO] Log          -> {log_path}")
