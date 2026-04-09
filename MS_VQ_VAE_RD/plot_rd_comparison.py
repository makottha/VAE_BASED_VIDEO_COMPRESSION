"""
plot_rd_comparison.py
=====================
Rate-distortion curves: MS-VQ-VAE vs H.264 and H.265.

Usage:
    python plot_rd_comparison.py --out paper/figures/rd_comparison.pdf
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data (from metrics_K*.csv and baseline_results/)
# ---------------------------------------------------------------------------

OURS = {
    "label": "MS-VQ-VAE (ours)",
    "bpp":   [0.0427, 0.0487, 0.0550, 0.0635],
    "psnr":  [25.88,  26.28,  26.27,  27.14],
    "ssim":  [0.8780, 0.8855, 0.8966, 0.9043],
    "lpips": [0.1546, 0.1386, 0.1287, 0.1140],
    "k":     [128,    256,    512,    1024],
}

H264 = {
    "label": "H.264",
    "bpp":   [0.2105, 0.2622, 0.3525],
    "psnr":  [29.52,  31.87,  34.47],
    "ssim":  [0.9051, 0.9391, 0.9624],
    "lpips": [0.1275, 0.0830, 0.0494],
    "crf":   [36,     32,     28],
}

H265 = {
    "label": "H.265",
    "bpp":   [0.3266, 0.3887, 0.4952],
    "psnr":  [26.54,  28.90,  31.43],
    "ssim":  [0.8383, 0.8963, 0.9369],
    "lpips": [0.1856, 0.1200, 0.0729],
    "crf":   [36,     32,     28],
}

# Colours
C_OURS = "#d62728"   # red
C_H264 = "#1f77b4"   # blue
C_H265 = "#2ca02c"   # green
SHADE  = "#f0f0f0"   # shaded ultra-low BPP region

ULTRA_LOW_BPP = 0.08   # threshold for shaded region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_panel(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=11, fontweight="bold", pad=6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def shade_ultra_low(ax, ymin, ymax):
    """Shade the ultra-low BPP region."""
    ax.axvspan(0, ULTRA_LOW_BPP, color=SHADE, zorder=0, label="_nolegend_")
    ax.text(ULTRA_LOW_BPP * 0.5, ymin + (ymax - ymin) * 0.03,
            "ultra-low\nbitrate", ha="center", va="bottom",
            fontsize=7, color="#999999", style="italic", zorder=2)


def plot_curve(ax, data, color, marker, zorder=3):
    bpp = data["bpp"]
    # sort by bpp for clean lines
    order = np.argsort(bpp)
    x = np.array(bpp)[order]

    if "psnr" in ax._ylabel_text:
        y = np.array(data["psnr"])[order]
    elif "SSIM" in ax._ylabel_text:
        y = np.array(data["ssim"])[order]
    else:
        y = np.array(data["lpips"])[order]

    ax.plot(x, y, color=color, marker=marker, markersize=7,
            linewidth=1.5, zorder=zorder, label=data["label"])

    # Annotate K or CRF values
    tags = data.get("k", data.get("crf", []))
    tags = np.array(tags)[order]
    for xi, yi, tag in zip(x, y, tags):
        prefix = "K=" if "k" in data else "CRF"
        ax.annotate(f"{prefix}{tag}",
                    xy=(xi, yi), xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5, color=color, zorder=5)


# Monkey-patch to pass ylabel info into the helper
def _ylabel(ax):
    return ax.get_ylabel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor("white")

    # ── Panel 1: BPP vs PSNR ───────────────────────────────────────────────
    ax = axes[0]
    setup_panel(ax, "Bits per Pixel (BPP)", "PSNR (dB) ↑", "BPP vs. PSNR")
    shade_ultra_low(ax, 24, 36)

    for data, color, marker in [
        (OURS, C_OURS, "o"),
        (H264, C_H264, "s"),
        (H265, C_H265, "^"),
    ]:
        bpp   = np.array(data["bpp"])
        psnr  = np.array(data["psnr"])
        order = np.argsort(bpp)
        x, y  = bpp[order], psnr[order]
        ax.plot(x, y, color=color, marker=marker, markersize=7,
                linewidth=1.5, label=data["label"])
        tags = np.array(data.get("k", data.get("crf")))[order]
        for xi, yi, tag in zip(x, y, tags):
            prefix = "K=" if "k" in data else "CRF "
            ax.annotate(f"{prefix}{tag}", xy=(xi, yi), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, color=color)

    ax.set_xlim(left=0)
    ax.set_ylim(24, 36)
    shade_ultra_low(ax, 24, 36)

    # ── Panel 2: BPP vs SSIM ───────────────────────────────────────────────
    ax = axes[1]
    setup_panel(ax, "Bits per Pixel (BPP)", "SSIM ↑", "BPP vs. SSIM")

    for data, color, marker in [
        (OURS, C_OURS, "o"),
        (H264, C_H264, "s"),
        (H265, C_H265, "^"),
    ]:
        bpp   = np.array(data["bpp"])
        ssim  = np.array(data["ssim"])
        order = np.argsort(bpp)
        x, y  = bpp[order], ssim[order]
        ax.plot(x, y, color=color, marker=marker, markersize=7,
                linewidth=1.5, label=data["label"])
        tags = np.array(data.get("k", data.get("crf")))[order]
        for xi, yi, tag in zip(x, y, tags):
            prefix = "K=" if "k" in data else "CRF "
            ax.annotate(f"{prefix}{tag}", xy=(xi, yi), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, color=color)

    ax.set_xlim(left=0)
    ax.set_ylim(0.82, 0.98)
    shade_ultra_low(ax, 0.82, 0.98)

    # ── Panel 3: BPP vs LPIPS (inverted — lower LPIPS = better) ───────────
    ax = axes[2]
    setup_panel(ax, "Bits per Pixel (BPP)", "1 − LPIPS ↑", "BPP vs. LPIPS")

    for data, color, marker in [
        (OURS, C_OURS, "o"),
        (H264, C_H264, "s"),
        (H265, C_H265, "^"),
    ]:
        bpp   = np.array(data["bpp"])
        inv   = 1.0 - np.array(data["lpips"])   # invert so "up = better"
        order = np.argsort(bpp)
        x, y  = bpp[order], inv[order]
        ax.plot(x, y, color=color, marker=marker, markersize=7,
                linewidth=1.5, label=data["label"])
        tags = np.array(data.get("k", data.get("crf")))[order]
        for xi, yi, tag in zip(x, y, tags):
            prefix = "K=" if "k" in data else "CRF "
            ax.annotate(f"{prefix}{tag}", xy=(xi, yi), xytext=(4, -10),
                        textcoords="offset points", fontsize=6.5, color=color)

    ax.set_xlim(left=0)
    ax.set_ylim(0.78, 0.98)
    shade_ultra_low(ax, 0.78, 0.98)

    # ── Shared legend ───────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=C_OURS, label="MS-VQ-VAE (ours)  K∈{128,256,512,1024}"),
        mpatches.Patch(color=C_H264, label="H.264  CRF∈{36,32,28}"),
        mpatches.Patch(color=C_H265, label="H.265  CRF∈{36,32,28}"),
        mpatches.Patch(color=SHADE,  label="Ultra-low bitrate (<0.08 bpp)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, framealpha=0.95,
               edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.04),
               bbox_transform=fig.transFigure)

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    # ── Save ────────────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figures/rd_comparison.pdf")
    return p.parse_args()

if __name__ == "__main__":
    build(parse_args().out)
