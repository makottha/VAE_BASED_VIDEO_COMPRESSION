# plot_rd_curve.py
"""
RD curve plots for the K-sweep VQ-VAE2 video compression paper.

Generates 3 publication-quality figures:
  Fig 1 — BPP vs PSNR  (with H.264 baseline)
  Fig 2 — BPP vs SSIM  (with H.264 baseline)
  Fig 3 — BPP vs LPIPS (with H.264 baseline, lower=better)

H.264 baseline points (CRF 28/32/36, direct Python comparison):
  CRF 28: PSNR=34.46, BPP~0.21
  CRF 32: PSNR=31.87, BPP~0.15
  CRF 36: PSNR=29.51, BPP~0.10

Usage:
    python plot_rd_curve.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
K_VALUES   = [128, 256, 512, 1024]
SUMMARY_CSV = "./outputs_msvqvae_rd/rd_curve_summary.csv"
OUT_DIR     = "./outputs_msvqvae_rd/rd_curves"
os.makedirs(OUT_DIR, exist_ok=True)

# H.264 baseline — measured on 500 UCF101 64×64 test clips (eval_h264.py, 2026-03-16)
H264 = [
    {"crf": 36, "bpp": 0.2105, "psnr": 29.52, "ssim": 0.9051, "lpips": 0.1275},
    {"crf": 32, "bpp": 0.2622, "psnr": 31.87, "ssim": 0.9391, "lpips": 0.0830},
    {"crf": 28, "bpp": 0.3525, "psnr": 34.47, "ssim": 0.9624, "lpips": 0.0494},
]

# -----------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------
MODEL_COLOR  = "#e15759"   # red for our model
H264_COLOR   = "#4e79a7"   # blue for H.264
MODEL_MARKER = "o"
H264_MARKER  = "s"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# -----------------------------------------------------------------------
# Load our model results
# -----------------------------------------------------------------------
rows = list(csv.DictReader(open(SUMMARY_CSV)))
model = {int(r["K"]): r for r in rows}

our_bpp   = [float(model[K]["bpp"])   for K in K_VALUES]
our_psnr  = [float(model[K]["psnr"])  for K in K_VALUES]
our_ssim  = [float(model[K]["ssim"])  for K in K_VALUES]
our_lpips = [float(model[K]["lpips"]) for K in K_VALUES]

h264_bpp   = [p["bpp"]   for p in H264]
h264_psnr  = [p["psnr"]  for p in H264]
h264_ssim  = [p["ssim"]  for p in H264]
h264_lpips = [p["lpips"] for p in H264]
h264_crfs  = [p["crf"]   for p in H264]

# -----------------------------------------------------------------------
# Helper: annotate K labels on model curve
# -----------------------------------------------------------------------
def annotate_k(ax, bpp_vals, metric_vals, offsets=None):
    """offsets: list of (dx, dy) per K value for label nudging."""
    if offsets is None:
        offsets = [(0.002, 0.003)] * len(K_VALUES)
    for K, bpp, val, (dx, dy) in zip(K_VALUES, bpp_vals, metric_vals, offsets):
        ax.annotate(f"K={K}", xy=(bpp, val), xytext=(bpp + dx, val + dy),
                    fontsize=8.5, color=MODEL_COLOR,
                    arrowprops=dict(arrowstyle="-", color=MODEL_COLOR, lw=0.7))

def annotate_h264(ax, bpp_vals, metric_vals, offsets=None):
    if offsets is None:
        offsets = [(0.003, -0.4)] * len(H264)
    for pt, bpp, val, (dx, dy) in zip(H264, bpp_vals, metric_vals, offsets):
        ax.annotate(f"CRF{pt['crf']}", xy=(bpp, val), xytext=(bpp + dx, val + dy),
                    fontsize=8.5, color=H264_COLOR,
                    arrowprops=dict(arrowstyle="-", color=H264_COLOR, lw=0.7))

# -----------------------------------------------------------------------
# Fig 1 — BPP vs PSNR
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(our_bpp, our_psnr, color=MODEL_COLOR, linewidth=2,
        marker=MODEL_MARKER, markersize=7, zorder=3, label="Ours (VQ-VAE2 + Entropy Prior)")
ax.plot(h264_bpp, h264_psnr, color=H264_COLOR, linewidth=2,
        marker=H264_MARKER, markersize=7, linestyle="--", zorder=3, label="H.264")

annotate_k(ax, our_bpp, our_psnr,
           offsets=[(0.002, 0.20), (0.002, 0.20), (-0.012, -0.45), (0.002, 0.20)])
annotate_h264(ax, h264_bpp, h264_psnr,
              offsets=[(0.003, -0.5), (0.003, -0.5), (0.003, -0.5)])

# shade ultra-low bitrate region
ax.axvspan(0.0, 0.08, alpha=0.06, color=MODEL_COLOR, label="Ultra-low bitrate regime")

ax.set_xlabel("Bits Per Pixel (BPP)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Rate–Distortion Curve: BPP vs. PSNR\n(UCF101, 64×64, 32 frames)")
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_rd_psnr.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[DONE] Fig 1 — BPP vs PSNR")

# -----------------------------------------------------------------------
# Fig 2 — BPP vs SSIM
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(our_bpp, our_ssim, color=MODEL_COLOR, linewidth=2,
        marker=MODEL_MARKER, markersize=7, zorder=3, label="Ours (VQ-VAE2 + Entropy Prior)")
ax.plot(h264_bpp, h264_ssim, color=H264_COLOR, linewidth=2,
        marker=H264_MARKER, markersize=7, linestyle="--", zorder=3, label="H.264")

annotate_k(ax, our_bpp, our_ssim,
           offsets=[(0.002, 0.002), (0.002, 0.002), (-0.014, -0.005), (0.002, 0.002)])
annotate_h264(ax, h264_bpp, h264_ssim,
              offsets=[(0.003, -0.004), (0.003, -0.004), (0.003, -0.004)])

ax.axvspan(0.0, 0.08, alpha=0.06, color=MODEL_COLOR, label="Ultra-low bitrate regime")

ax.set_xlabel("Bits Per Pixel (BPP)")
ax.set_ylabel("SSIM")
ax.set_title("Rate–Distortion Curve: BPP vs. SSIM\n(UCF101, 64×64, 32 frames)")
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_rd_ssim.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[DONE] Fig 2 — BPP vs SSIM")

# -----------------------------------------------------------------------
# Fig 3 — BPP vs LPIPS (lower = better)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(our_bpp, our_lpips, color=MODEL_COLOR, linewidth=2,
        marker=MODEL_MARKER, markersize=7, zorder=3, label="Ours (VQ-VAE2 + Entropy Prior)")
ax.plot(h264_bpp, h264_lpips, color=H264_COLOR, linewidth=2,
        marker=H264_MARKER, markersize=7, linestyle="--", zorder=3, label="H.264")

annotate_k(ax, our_bpp, our_lpips,
           offsets=[(0.002, 0.003), (0.002, 0.003), (-0.014, 0.006), (0.002, 0.003)])
annotate_h264(ax, h264_bpp, h264_lpips,
              offsets=[(0.003, 0.004), (0.003, 0.004), (0.003, 0.004)])

ax.axvspan(0.0, 0.08, alpha=0.06, color=MODEL_COLOR, label="Ultra-low bitrate regime")

ax.set_xlabel("Bits Per Pixel (BPP)")
ax.set_ylabel("LPIPS ↓  (lower = better)")
ax.set_title("Rate–Perceptual Curve: BPP vs. LPIPS\n(UCF101, 64×64, 32 frames)")
ax.legend(loc="upper right")
ax.invert_yaxis()   # lower LPIPS is better, so invert for visual clarity
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_rd_lpips.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[DONE] Fig 3 — BPP vs LPIPS")

# -----------------------------------------------------------------------
# Fig 4 — Combined 3-panel figure (paper-ready single figure)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

panels = [
    ("PSNR (dB)",          our_psnr,  h264_psnr,  "lower right", False,
     [(0.002, 0.20), (0.002, 0.20), (-0.013, -0.45), (0.002, 0.20)],
     [(0.003, -0.5), (0.003, -0.5), (0.003, -0.5)]),
    ("SSIM",               our_ssim,  h264_ssim,  "lower right", False,
     [(0.002, 0.002), (0.002, 0.002), (-0.014, -0.005), (0.002, 0.002)],
     [(0.003, -0.004), (0.003, -0.004), (0.003, -0.004)]),
    ("LPIPS ↓",            our_lpips, h264_lpips, "upper right", True,
     [(0.002, 0.003), (0.002, 0.003), (-0.014, 0.006), (0.002, 0.003)],
     [(0.003, 0.004), (0.003, 0.004), (0.003, 0.004)]),
]

for ax, (ylabel, our_y, h264_y, leg_loc, invert, k_off, h_off) in zip(axes, panels):
    ax.plot(our_bpp, our_y, color=MODEL_COLOR, linewidth=2,
            marker=MODEL_MARKER, markersize=7, zorder=3,
            label="Ours (VQ-VAE2 + Entropy Prior)")
    ax.plot(h264_bpp, h264_y, color=H264_COLOR, linewidth=2,
            marker=H264_MARKER, markersize=7, linestyle="--", zorder=3,
            label="H.264")

    # K labels
    for K, bpp, val, (dx, dy) in zip(K_VALUES, our_bpp, our_y, k_off):
        ax.annotate(f"K={K}", xy=(bpp, val), xytext=(bpp + dx, val + dy),
                    fontsize=8, color=MODEL_COLOR,
                    arrowprops=dict(arrowstyle="-", color=MODEL_COLOR, lw=0.6))
    # CRF labels
    for pt, bpp, val, (dx, dy) in zip(H264, h264_bpp, h264_y, h_off):
        ax.annotate(f"CRF{pt['crf']}", xy=(bpp, val), xytext=(bpp + dx, val + dy),
                    fontsize=8, color=H264_COLOR,
                    arrowprops=dict(arrowstyle="-", color=H264_COLOR, lw=0.6))

    ax.axvspan(0.0, 0.08, alpha=0.06, color=MODEL_COLOR)
    ax.set_xlabel("Bits Per Pixel (BPP)")
    ax.set_ylabel(ylabel)
    ax.legend(loc=leg_loc, fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    if invert:
        ax.invert_yaxis()

axes[0].set_title("BPP vs. PSNR")
axes[1].set_title("BPP vs. SSIM")
axes[2].set_title("BPP vs. LPIPS")

fig.suptitle(
    "Rate–Distortion Performance vs. H.264  (UCF101, 64×64, 32 frames @ 16 FPS)",
    fontsize=13, y=1.02
)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_rd_combined.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[DONE] Fig 4 — combined 3-panel")

# -----------------------------------------------------------------------
# Print summary for paper
# -----------------------------------------------------------------------
print("\n" + "="*65)
print(f"{'Model':<30} {'BPP':>6} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
print("-"*65)
for K in K_VALUES:
    print(f"{'Ours (K='+str(K)+')':<30} {float(model[K]['bpp']):>6.4f} "
          f"{float(model[K]['psnr']):>7.2f} {float(model[K]['ssim']):>7.4f} "
          f"{float(model[K]['lpips']):>7.4f}")
print("-"*65)
for pt in H264:
    print(f"{'H.264 (CRF '+str(pt['crf'])+')':<30} {pt['bpp']:>6.4f} "
          f"{pt['psnr']:>7.2f} {pt['ssim']:>7.4f} {pt['lpips']:>7.4f}")
print("="*65)
print(f"\n[INFO] All figures -> {OUT_DIR}/")
