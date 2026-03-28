"""
generate_arch_diagram.py
========================
Architecture diagram for MS-VQ-VAE (MS_VQ_VAE_RD) — styled after the original
paper figure: sharp rectangular boxes, white background, black arrows, math labels.

Correct flow (from models.py forward()):
  Input → Bottom Encoder → z_b ──────────────────────► VQ Bottom → ẑ_b ──┐
                              └──► Top Encoder → z_t ► VQ Top → ẑ_t      ├──► Decoder → Reconstructed Video
                                                              (upsample) ──┘

Usage:
    python generate_arch_diagram.py --out paper/figures/architecture.png
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, label, fontsize=10.5):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="square,pad=0",
        linewidth=1.2, edgecolor="black",
        facecolor="white", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color="black", zorder=4,
            multialignment="center")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color="black",
                        lw=1.2, mutation_scale=12),
        zorder=5)


def alabel(ax, x, y, text, ha="center", va="center", fontsize=10):
    ax.text(x, y, text, ha=ha, va=va,
            fontsize=fontsize, color="black",
            fontstyle="italic", zorder=6)


# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------

def build(out_path):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # ── Dimensions ─────────────────────────────────────────────────────────
    BW, BH = 1.85, 0.82

    # ── Y positions ────────────────────────────────────────────────────────
    TOP_Y = 3.8    # Top Encoder + VQ Top row
    BOT_Y = 1.7    # Bottom Encoder + VQ Bottom row
    MID_Y = 2.75   # Decoder row

    # ── X positions ────────────────────────────────────────────────────────
    INPUT_X  = 1.1
    BENC_X   = 3.2    # Bottom Encoder
    TENC_X   = 5.5    # Top Encoder (receives z_b from Bottom Encoder)
    VQB_X    = 5.5    # VQ Bottom (receives z_b from Bottom Encoder)
    VQT_X    = 7.9    # VQ Top
    DEC_X    = 10.5   # Decoder
    OUT_X    = 12.9   # Reconstructed Video

    # ── Blocks ─────────────────────────────────────────────────────────────
    box(ax, INPUT_X, MID_Y,  1.55, BH + 0.35,
        "Input Video\n($3{\\times}32{\\times}64{\\times}64$)", fontsize=9.5)

    box(ax, BENC_X,  BOT_Y,  BW, BH, "Bottom-Level\nEncoder")
    box(ax, TENC_X,  TOP_Y,  BW, BH, "Top-Level\nEncoder")

    box(ax, VQB_X,   BOT_Y,  BW, BH,
        "Vector Quantizer\n$\\mathcal{E}_b$")
    box(ax, VQT_X,   TOP_Y,  BW, BH,
        "Vector Quantizer\n$\\mathcal{E}_t$")

    box(ax, DEC_X,   MID_Y,  BW, BH, "Decoder")

    box(ax, OUT_X,   MID_Y,  1.55, BH + 0.35,
        "Reconstructed\nVideo", fontsize=9.5)

    # ── Arrows ─────────────────────────────────────────────────────────────

    # Input → Bottom Encoder
    arrow(ax, INPUT_X + 1.55/2, MID_Y,
               BENC_X  - BW/2,  BOT_Y)

    # Bottom Encoder → VQ Bottom  (z_b, same row →)
    arrow(ax, BENC_X + BW/2, BOT_Y,
               VQB_X  - BW/2, BOT_Y)
    alabel(ax, (BENC_X + BW/2 + VQB_X - BW/2)/2,
               BOT_Y + 0.24, "$\\mathbf{z}_b$")

    # Bottom Encoder → Top Encoder  (z_b, branch upward)
    arrow(ax, BENC_X + BW/2 - 0.2, BOT_Y + BH/2,
               TENC_X - BW/2,       TOP_Y)
    alabel(ax, (BENC_X + BW/2 - 0.2 + TENC_X - BW/2)/2 - 0.25,
               (BOT_Y + BH/2 + TOP_Y)/2 + 0.1,
               "$\\mathbf{z}_b$", ha="right")

    # Top Encoder → VQ Top  (z_t)
    arrow(ax, TENC_X + BW/2, TOP_Y,
               VQT_X  - BW/2, TOP_Y)
    alabel(ax, (TENC_X + BW/2 + VQT_X - BW/2)/2,
               TOP_Y + 0.24, "$\\mathbf{z}_t$")

    # VQ Top → Decoder  (ẑ_t, diagonal down — upsample implied)
    arrow(ax, VQT_X + BW/2, TOP_Y,
               DEC_X  - BW/2, MID_Y + 0.15)
    alabel(ax, (VQT_X + BW/2 + DEC_X - BW/2)/2 + 0.05,
               (TOP_Y + MID_Y + 0.15)/2 + 0.22,
               "$\\hat{\\mathbf{z}}_t$ (↑ upsample)")

    # VQ Bottom → Decoder  (ẑ_b, diagonal up)
    arrow(ax, VQB_X + BW/2, BOT_Y,
               DEC_X  - BW/2, MID_Y - 0.15)
    alabel(ax, (VQB_X + BW/2 + DEC_X - BW/2)/2 + 0.05,
               (BOT_Y + MID_Y - 0.15)/2 - 0.22,
               "$\\hat{\\mathbf{z}}_b$")

    # Decoder → Reconstructed Video
    arrow(ax, DEC_X + BW/2, MID_Y,
               OUT_X  - 1.55/2, MID_Y)

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figures/architecture.png")
    return p.parse_args()

if __name__ == "__main__":
    build(parse_args().out)
