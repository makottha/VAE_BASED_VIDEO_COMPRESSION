#!/usr/bin/env python3
"""
Make H.264 RD points (CRF 28/32/36) for standardized clips (W×H, T frames).
- Encodes each input clip with H.264 at CRF {28, 32, 36}
- Computes bpp from output file size
- Computes PSNR/SSIM vs the original (per-frame, averaged)
- Saves per-clip CSV and prints an aggregated summary by CRF

Defaults assume 64×64, 16 fps, 32 frames as in your pipeline.

Install deps:
  pip install opencv-python scikit-image pandas imageio-ffmpeg
"""

import argparse, csv, glob, os, subprocess, sys, platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from imageio_ffmpeg import get_ffmpeg_exe

# ----------------------- FFmpeg + codec detection -----------------------

FFMPEG = get_ffmpeg_exe()  # bundled ffmpeg binary from imageio-ffmpeg

def _codec_works(ffmpeg_path: str, codec: str) -> bool:
    """Try a tiny synthetic encode to check if codec is available."""
    nul = "NUL" if platform.system().lower().startswith("win") else "/dev/null"
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=size=64x64:rate=1:color=black",
        "-t", "1",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-f", "mp4", "-y", nul
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0

def pick_h264_codec(ffmpeg_path: str) -> str:
    """Pick a working H.264 encoder."""
    for c in ("libx264", "h264", "h264_mf", "h264_nvenc"):
        if _codec_works(ffmpeg_path, c):
            print(f"[codec] Using {c}")
            return c
    raise SystemExit("No working H.264 encoder found (tried libx264, h264, h264_mf, h264_nvenc).")

# ------------------------------ IO helpers ------------------------------

def list_inputs(maybe_glob_or_dir: str):
    """Accept either a glob pattern or a directory path."""
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
    """Read a video to float32 RGB [0,1], returns (T,H,W,3)."""
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
    return np.stack(frames)

def clip_metrics(ref_vid, test_vid):
    """Compute mean PSNR/SSIM over min(T_ref, T_test) frames; assumes same H×W."""
    T = min(len(ref_vid), len(test_vid))
    ref = ref_vid[:T]
    dec = test_vid[:T]
    if ref.shape[1:3] != dec.shape[1:3]:
        raise ValueError(f"Shape mismatch: ref {ref.shape[1:3]} vs dec {dec.shape[1:3]}")
    psnrs, ssims = [], []
    for r, d in zip(ref, dec):
        psnrs.append(psnr(r, d, data_range=1.0))
        try:
            ssims.append(ssim(r, d, data_range=1.0, channel_axis=2))  # skimage>=0.19
        except TypeError:
            ssims.append(ssim(r, d, data_range=1.0, multichannel=True))  # older skimage
    return float(np.mean(psnrs)), float(np.mean(ssims)), T

def bytes_to_bpp(num_bytes, width, height, frames):
    return (num_bytes * 8.0) / (width * height * frames)

# ------------------------------ Encoding -------------------------------

def run_ffmpeg_encode(ffmpeg_path, codec, in_path, out_path, crf, fps=None, enforce_frames=None, extra_vf=None):
    """Encode with selected H.264 codec at CRF, optionally forcing fps and frame count."""
    cmd = [ffmpeg_path, "-y", "-i", str(in_path)]
    vf_filters = []
    if fps:
        vf_filters.append(f"fps={fps}")
    if extra_vf:
        vf_filters.append(extra_vf)
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    cmd += [
        "-c:v", codec,
        "-preset", "medium",
        "-tune", "psnr",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf)
    ]
    if enforce_frames is not None:
        cmd += ["-frames:v", str(enforce_frames)]
    cmd += [str(out_path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("FFmpeg command:\n", " ".join(cmd))
        print("FFmpeg stderr:\n", p.stderr)
        raise RuntimeError(f"ffmpeg failed for {in_path}")
    return out_path

# -------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="./dataset_human_readable_64/test/clips/*.avi",
                    help="Glob of input videos or a directory path")
    ap.add_argument("--outdir", default="./dataset_human_readable_64/test/h264_clips/",
                    help="Output directory")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--crfs", type=int, nargs="+", default=[28, 32, 36])
    ap.add_argument("--csv", default="rd_h264.csv")
    ap.add_argument("--skip_metrics", action="store_true",
                    help="Skip PSNR/SSIM (only compute bpp).")
    args = ap.parse_args()

    print("Using ffmpeg at:", FFMPEG)
    CODEC = pick_h264_codec(FFMPEG)

    WIDTH, HEIGHT, T, FPS = args.width, args.height, args.frames, args.fps
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    inputs = list_inputs(args.inputs)
    if not inputs:
        print(f"No input videos found for: {args.inputs}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(inputs)} input(s).")

    csv_path = outdir / args.csv
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["clip", "method", "crf", "bpp", "psnr", "ssim", "frames"])

        # Collect for summary
        sums = {crf: {"bpp": 0.0, "psnr": 0.0, "ssim": 0.0, "n": 0} for crf in args.crfs}

        for in_path in inputs:
            # load reference (standardized) clip
            ref = None
            if not args.skip_metrics:
                ref = read_video_rgb01(in_path, max_frames=T)

            for crf in args.crfs:
                out_path = outdir / f"{Path(in_path).stem}_h264_crf{crf}.mp4"
                run_ffmpeg_encode(FFMPEG, CODEC, in_path, out_path, crf=crf, fps=FPS, enforce_frames=T)

                num_bytes = os.path.getsize(out_path)
                bpp = bytes_to_bpp(num_bytes, WIDTH, HEIGHT, T)

                psnr_val = ssim_val = ""
                used_frames = T
                if not args.skip_metrics:
                    dec = read_video_rgb01(out_path, max_frames=T)
                    psnr_val, ssim_val, used_frames = clip_metrics(ref, dec)

                w.writerow([
                    in_path, "H264", crf, f"{bpp:.6f}",
                    f"{psnr_val:.4f}" if psnr_val != "" else "",
                    f"{ssim_val:.4f}" if ssim_val != "" else "", used_frames
                ])
                print(f"[OK] {Path(in_path).name} CRF{crf}: bpp={bpp:.4f} "
                      f"PSNR={psnr_val if psnr_val=='' else f'{psnr_val:.2f}'} "
                      f"SSIM={ssim_val if ssim_val=='' else f'{ssim_val:.4f}'}")

                # summary accumulators
                if psnr_val != "" and ssim_val != "":
                    sums[crf]["bpp"]  += bpp
                    sums[crf]["psnr"] += float(psnr_val)
                    sums[crf]["ssim"] += float(ssim_val)
                    sums[crf]["n"]    += 1

    # Print mean summary by CRF (if metrics computed)
    try:
        df = pd.read_csv(csv_path)
        df["bpp"] = pd.to_numeric(df["bpp"], errors="coerce")
        if not args.skip_metrics:
            df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
            df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
        summary = df.groupby("crf").agg(bpp=("bpp","mean"),
                                        psnr=("psnr","mean"),
                                        ssim=("ssim","mean"),
                                        n=("clip","count")).reset_index()
        print("\n=== H.264 RD summary (mean over clips) ===")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(summary)
        summary.to_csv(outdir / "rd_h264_summary.csv", index=False)
    except Exception as e:
        print("Summary generation skipped due to error:", e)

if __name__ == "__main__":
    main()
