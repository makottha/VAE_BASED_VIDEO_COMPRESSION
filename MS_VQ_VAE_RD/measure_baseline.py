"""
measure_baselines.py
====================
Runs H.265 baseline encoding + decoding on the same 500 UCF101 test
clips used in the paper, and measures per-stage latency for MS-VQ-VAE.
Outputs two things:
  1. h265_results.csv  — BPP, PSNR, SSIM, LPIPS for H.265 CRF {28,32,36}
  2. latency_results.csv — per-stage wall-clock times for K={512,1024}

Usage
-----
python measure_baselines.py \
    --clips_dir   /path/to/ucf101_tensors \
    --ckpt_k512   /path/to/ckpt_k512.pt  \
    --ckpt_k1024  /path/to/ckpt_k1024.pt \
    --tmp_dir     /tmp/h265_encode        \
    --out_dir     ./baseline_results

Requirements: torch, torchvision, numpy, ffmpeg (system), ffprobe (system)
              scikit-image, lpips
"""

import argparse, csv, os, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import MSVQVAE2Video

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(path, K):
    m = MSVQVAE2Video(
        top_dim=256, bottom_dim=128,
        K_top=K, K_bottom=K,
        commit_top=0.25, commit_bottom=0.25,
        norm="gn", use_vgg=True, vgg_layers=16,
        soft_temp=1.0, use_ema=True,
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    s = ckpt.get("ae", ckpt)
    m.load_state_dict(s, strict=True)
    return m.eval()


def encode_decode(model, clip):
    """Run full forward pass, return reconstructed (1,3,T,H,W)."""
    with torch.no_grad():
        out = model(clip)
    return out["recon"]

def load_clip(path):
    c = torch.load(path, map_location="cpu").float()
    if c.max() > 1.5: c /= 255.0
    if c.ndim == 4: c = c.unsqueeze(0)
    return c.clamp(0, 1)

def clip_to_frames_uint8(clip):
    """clip: (1,3,T,H,W) float [0,1] → list of (H,W,3) uint8"""
    c = clip.squeeze(0).permute(1,2,3,0).numpy()
    return [(c[t]*255).clip(0,255).astype(np.uint8) for t in range(c.shape[0])]

def frames_to_video(frames, path, fps=16):
    """Write frames as raw yuv420p pipe into ffmpeg → mp4."""
    H, W = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "pipe:0", "-vcodec", "libx265", "-an", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

def encode_h265(frames, tmp_dir, crf, fps=16):
    """Encode frames with libx265 at given CRF, return (bpp, decoded_frames)."""
    H, W = frames[0].shape[:2]
    vid_in  = os.path.join(tmp_dir, "in.mp4")
    vid_out = os.path.join(tmp_dir, f"out_crf{crf}.mp4")

    # Write with libx265
    cmd_enc = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
               "-i", "pipe:0",
               "-vcodec", "libx265", "-crf", str(crf), "-preset", "medium",
               "-an", vid_out]
    proc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    for f in frames: proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

    # Get file size → BPP
    nbytes = os.path.getsize(vid_out)
    T = len(frames)
    bpp = (nbytes * 8) / (T * H * W)

    # Decode back to frames
    cmd_dec = ["ffmpeg", "-y", "-i", vid_out,
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.check_output(cmd_dec, stderr=subprocess.DEVNULL)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(T, H, W, 3)
    decoded = [arr[t] for t in range(T)]
    return bpp, decoded

def compute_metrics(orig_frames, rec_frames, lpips_fn):
    """Compute PSNR, SSIM, LPIPS averaged over frames."""
    psnrs, ssims, lpips_vals = [], [], []
    for o, r in zip(orig_frames, rec_frames):
        psnrs.append(psnr_fn(o, r, data_range=255))
        ssims.append(ssim_fn(o, r, data_range=255, channel_axis=2))
        ot = torch.from_numpy(o).float().permute(2,0,1).unsqueeze(0)/127.5-1
        rt = torch.from_numpy(r).float().permute(2,0,1).unsqueeze(0)/127.5-1
        with torch.no_grad():
            lpips_vals.append(lpips_fn(ot, rt).item())
    return np.mean(psnrs), np.mean(ssims), np.mean(lpips_vals)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir",  required=True)
    ap.add_argument("--ckpt_k512",  required=True)
    ap.add_argument("--ckpt_k1024", required=True)
    ap.add_argument("--tmp_dir",    default=os.path.join(tempfile.gettempdir(), "h265_encode"))
    ap.add_argument("--out_dir",    default="./baseline_results")
    ap.add_argument("--n_clips",    type=int, default=500)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Find clips
    clips = sorted(Path(args.clips_dir).rglob("*.pt"))[:args.n_clips]
    print(f"Found {len(clips)} clips")

    lpips_fn = lpips.LPIPS(net="vgg")

    # ── H.265 baseline ────────────────────────────────────────────────────────
    print("\n=== H.265 baseline ===")
    h265_rows = []
    for crf in [36, 32, 28]:
        bpps, psnrs, ssims, lpipss = [], [], [], []
        for i, cp in enumerate(clips):
            clip = load_clip(cp)
            frames = clip_to_frames_uint8(clip)
            bpp, dec = encode_h265(frames, args.tmp_dir, crf)
            p, s, l = compute_metrics(frames, dec, lpips_fn)
            bpps.append(bpp); psnrs.append(p)
            ssims.append(s);  lpipss.append(l)
            if (i+1) % 50 == 0:
                print(f"  CRF {crf}: {i+1}/{len(clips)} done")
        row = {
            "method": f"H.265 CRF {crf}",
            "bpp":   f"{np.mean(bpps):.4f}",
            "psnr":  f"{np.mean(psnrs):.2f}",
            "ssim":  f"{np.mean(ssims):.4f}",
            "lpips": f"{np.mean(lpipss):.4f}",
        }
        h265_rows.append(row)
        print(f"  CRF {crf}: BPP={row['bpp']}  PSNR={row['psnr']}  "
              f"SSIM={row['ssim']}  LPIPS={row['lpips']}")

    h265_path = os.path.join(args.out_dir, "h265_results.csv")
    with open(h265_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","bpp","psnr","ssim","lpips"])
        w.writeheader(); w.writerows(h265_rows)
    print(f"\nH.265 results saved → {h265_path}")

    # ── Latency measurement ───────────────────────────────────────────────────
    print("\n=== Latency measurement (CPU, 100 clips) ===")
    latency_rows = []
    N_LAT = min(100, len(clips))

    for K, ckpt in [(512, args.ckpt_k512), (1024, args.ckpt_k1024)]:
        print(f"\n  Loading K={K}...")
        model = load_model(ckpt, K)

        enc_times, dec_times, total_times = [], [], []
        for cp in clips[:N_LAT]:
            clip = load_clip(cp)

            # Encoder: bottom + top + both VQs
            t0 = time.perf_counter()
            with torch.no_grad():
                z_b = model.enc_bottom(clip)
                z_t = model.enc_top(z_b)
                z_t_q, _, _, _ = model.vq_top(z_t)
                z_b_q, _, _, _ = model.vq_bottom(z_b)
            t_enc = time.perf_counter() - t0

            # Decoder: upsample + concat + decode
            t1 = time.perf_counter()
            with torch.no_grad():
                z_t_up = F.interpolate(z_t_q, size=z_b_q.shape[2:],
                                       mode="trilinear", align_corners=False)
                dec_in = torch.cat([z_t_up, z_b_q], dim=1)
                _ = model.decoder(dec_in)
            t_dec = time.perf_counter() - t1

            enc_times.append(t_enc * 1000)   # ms
            dec_times.append(t_dec * 1000)
            total_times.append((t_enc + t_dec) * 1000)

        row = {
            "K":     K,
            "enc_ms":   f"{np.mean(enc_times):.1f}",
            "dec_ms":   f"{np.mean(dec_times):.1f}",
            "total_ms": f"{np.mean(total_times):.1f}",
        }
        latency_rows.append(row)
        print(f"  K={K}: encode={row['enc_ms']}ms  "
              f"decode={row['dec_ms']}ms  total={row['total_ms']}ms")

    # H.264 decode reference
    print("\n  Measuring H.264 decode time (reference)...")
    clip = load_clip(clips[0])
    frames = clip_to_frames_uint8(clip)
    tmp_vid = os.path.join(args.tmp_dir, "ref_h264.mp4")
    cmd_enc = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "64x64", "-r", "16",
               "-i", "pipe:0", "-vcodec", "libx264", "-crf", "36", "-an", tmp_vid]
    proc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames: proc.stdin.write(f.tobytes())
    proc.stdin.close(); proc.wait()

    h264_dec_times = []
    for _ in range(100):
        t = time.perf_counter()
        subprocess.check_output(
            ["ffmpeg", "-y", "-i", tmp_vid, "-f", "rawvideo",
             "-pix_fmt", "rgb24", "pipe:1"], stderr=subprocess.DEVNULL)
        h264_dec_times.append((time.perf_counter() - t) * 1000)
    h264_ref = np.mean(h264_dec_times)
    print(f"  H.264 CRF36 decode: {h264_ref:.1f}ms")

    lat_path = os.path.join(args.out_dir, "latency_results.csv")
    with open(lat_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K","enc_ms","dec_ms","total_ms"])
        w.writeheader(); w.writerows(latency_rows)
        w.writerow({"K": "H.264_decode_ref", "enc_ms": "-",
                    "dec_ms": f"{h264_ref:.1f}", "total_ms": f"{h264_ref:.1f}"})
    print(f"Latency results saved → {lat_path}")

    print("\n=== DONE ===")
    print("Copy these numbers into Table 1 (H.265 rows) and Table 2 (latency)")
    print(f"  H.265 results:  {h265_path}")
    print(f"  Latency results: {lat_path}")

if __name__ == "__main__":
    main()