"""
plot_motion_rd.py — Paper 3 RD Curve Plots
===========================================
Generates four RD curve figures combining:
  - Paper 2 I-frame-only results (from MS_VQ_VAE_RD/outputs_msvqvae_rd/rd_curve_summary.csv)
  - Paper 3 I+P results (from MS_VQ_VAE_MOTION/results/motion_summary.csv)
  - H.264 baseline (from MS_VQ_VAE_RD/outputs_msvqvae_rd/h264_baseline/h264_summary.csv)

Figures:
  01_rd_psnr.png    — BPP vs PSNR
  02_rd_ssim.png    — BPP vs SSIM
  03_rd_psnr_gop.png — per-GOP-size traces
  04_gain_over_ionly.png — BPP savings of I+P over I-only at matched PSNR
"""

import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
RD_DIR = BASE.parent / "MS_VQ_VAE_RD"

IONLY_CSV  = RD_DIR / "outputs_msvqvae_rd" / "rd_curve_summary.csv"
MOTION_CSV = BASE / "results" / "motion_summary.csv"
H264_CSV   = RD_DIR / "outputs_msvqvae_rd" / "h264_baseline" / "h264_summary.csv"
PLOTS_DIR  = BASE / "results" / "plots"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def flt(rows, key):
    return [float(r[key]) for r in rows]


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    ionly  = sorted(read_csv(IONLY_CSV),  key=lambda r: float(r["bpp"]))
    motion = read_csv(MOTION_CSV)
    h264   = sorted(read_csv(H264_CSV),   key=lambda r: float(r["bpp"]))

    bpp_i   = flt(ionly,  "bpp")
    psnr_i  = flt(ionly,  "psnr")
    ssim_i  = flt(ionly,  "ssim")

    bpp_h   = flt(h264,   "bpp")
    psnr_h  = flt(h264,   "psnr")
    ssim_h  = flt(h264,   "ssim")

    # Group motion results by GOP
    gops = sorted(set(int(r["gop"]) for r in motion))
    motion_by_gop = {}
    for g in gops:
        rows = sorted([r for r in motion if int(r["gop"]) == g], key=lambda r: float(r["bpp"]))
        motion_by_gop[g] = rows

    GOP_COLORS = {4: "royalblue", 8: "darkorange", 32: "mediumseagreen"}

    # ── Plot 1: BPP vs PSNR ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bpp_h, psnr_h, "k--o", label="H.264", linewidth=1.5, markersize=6)
    ax.plot(bpp_i, psnr_i, "r-^",  label="I-only (Paper 2)", linewidth=1.5, markersize=7)
    for g, rows in motion_by_gop.items():
        ax.plot(flt(rows, "bpp"), flt(rows, "psnr"),
                color=GOP_COLORS.get(g, "purple"), marker="s", linewidth=1.5, markersize=7,
                label=f"I+P GOP={g} (Paper 3)")
    ax.set_xlabel("Bits per pixel (BPP)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion: BPP vs PSNR", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "01_rd_psnr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: BPP vs SSIM ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bpp_h, ssim_h, "k--o", label="H.264", linewidth=1.5, markersize=6)
    ax.plot(bpp_i, ssim_i, "r-^",  label="I-only (Paper 2)", linewidth=1.5, markersize=7)
    for g, rows in motion_by_gop.items():
        ax.plot(flt(rows, "bpp"), flt(rows, "ssim"),
                color=GOP_COLORS.get(g, "purple"), marker="s", linewidth=1.5, markersize=7,
                label=f"I+P GOP={g} (Paper 3)")
    ax.set_xlabel("Bits per pixel (BPP)", fontsize=12)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title("Rate-Distortion: BPP vs SSIM", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "02_rd_ssim.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Per-GOP-size traces with K labels ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Paper 3 — I+P Codec per GOP Size", fontsize=13)
    for ax, metric, ylabel in zip(axes, ["psnr", "ssim"], ["PSNR (dB)", "SSIM"]):
        ax.plot(bpp_h, flt(h264, metric), "k--o", label="H.264", linewidth=1.5, markersize=5)
        ax.plot(bpp_i, flt(ionly, metric), "r-^", label="I-only (Paper 2)", linewidth=1.5, markersize=7)
        for g, rows in motion_by_gop.items():
            xs = flt(rows, "bpp"); ys = flt(rows, metric)
            ax.plot(xs, ys, color=GOP_COLORS.get(g, "purple"), marker="s",
                    linewidth=1.5, markersize=7, label=f"I+P GOP={g}")
            for x, y, r in zip(xs, ys, rows):
                ax.annotate(f"K={r['K_p']}", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=GOP_COLORS.get(g, "purple"))
        ax.set_xlabel("BPP"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "03_rd_per_gop.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: BPP savings over I-only at matched PSNR ───────────────────────
    # For each motion point, find nearest I-only point with same PSNR (interpolate)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="I-only (reference)")

    for g, rows in motion_by_gop.items():
        savings = []
        bpp_vals = []
        for r in rows:
            psnr_target = float(r["psnr"])
            bpp_m       = float(r["bpp"])
            # Interpolate I-only BPP at this PSNR
            p_arr = np.array(psnr_i)
            b_arr = np.array(bpp_i)
            if psnr_target < p_arr.min() or psnr_target > p_arr.max():
                continue
            bpp_ionly_at_psnr = float(np.interp(psnr_target, p_arr, b_arr))
            savings.append((bpp_ionly_at_psnr - bpp_m) / bpp_ionly_at_psnr * 100)
            bpp_vals.append(bpp_m)
        if savings:
            ax.plot(bpp_vals, savings, marker="s", color=GOP_COLORS.get(g, "purple"),
                    linewidth=1.5, markersize=7, label=f"GOP={g}")

    ax.set_xlabel("BPP (I+P codec)", fontsize=12)
    ax.set_ylabel("BPP savings over I-only at matched PSNR (%)", fontsize=11)
    ax.set_title("Gain of Motion-Conditioned Codec over I-frame Baseline", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "04_bpp_savings.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("Files:")
    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
