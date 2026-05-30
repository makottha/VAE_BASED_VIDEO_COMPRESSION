"""
generate_paper_figures.py
Generates fig_rd_combined.png and fig_lpips_scaling.png
for the NeurIPS 2026 MS-VQ-VAE paper.
Saves to C:/Users/kmani/Downloads/ alongside neurips_main.tex
"""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT_DIR = "C:/Users/kmani/Downloads"

# ── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          10,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.6,
    'ytick.minor.width':  0.6,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# Colorblind-safe palette (tab10 first three: blue, orange, green)
C64  = '#1f77b4'   # blue
C128 = '#ff7f0e'   # orange
C256 = '#2ca02c'   # green

# ── DATA ────────────────────────────────────────────────────────────────────

# Our model — (BPP, PSNR, SSIM, LPIPS) per K
our = {
    64: [
        (0.0427, 25.88, 0.8780, 0.1546),
        (0.0487, 26.28, 0.8855, 0.1386),
        (0.0550, 26.27, 0.8966, 0.1287),
        (0.0635, 27.14, 0.9043, 0.1140),
    ],
    128: [
        (0.0507, 25.50, 0.8302, 0.1399),
        (0.0614, 25.85, 0.8431, 0.1251),
        (0.0710, 25.72, 0.8393, 0.1245),
        (0.0799, 25.99, 0.8492, 0.1154),
    ],
    256: [
        (0.0542, 26.32, 0.8507, 0.1436),
        (0.0591, 26.60, 0.8567, 0.1352),
        (0.0701, 26.61, 0.8640, 0.1309),
        (0.0771, 26.65, 0.8682, 0.1194),
    ],
}

# H.264 baseline — (BPP, PSNR, SSIM, LPIPS) per CRF
h264 = {
    64: [
        (0.3525, 34.47, 0.9624, 0.0494),
        (0.2622, 31.87, 0.9391, 0.0830),
        (0.2105, 29.52, 0.9051, 0.1275),
    ],
    128: [
        (0.2514, 32.20, 0.9376, 0.0949),
        (0.1622, 30.61, 0.9134, 0.1307),
        (0.1102, 28.82, 0.8763, 0.1787),
    ],
    256: [
        (0.1841, 35.13, 0.9509, 0.0943),
        (0.1114, 33.36, 0.9317, 0.1254),
        (0.0702, 31.46, 0.9000, 0.1660),
    ],
}

res_colors = {64: C64, 128: C128, 256: C256}
res_labels = {64: r'64$\times$64', 128: r'128$\times$128', 256: r'256$\times$256'}
K_markers   = {0: 'o', 1: '^', 2: 's', 3: 'D'}   # K=128,256,512,1024
K_labels    = {0: 'K=128', 1: 'K=256', 2: 'K=512', 3: 'K=1024'}

# ============================================================================
# FIGURE 1 — Rate-distortion curves (3 panels)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(7, 3))

metric_idx  = {0: 1, 1: 2, 2: 3}   # PSNR=col1, SSIM=col2, LPIPS=col3
ylabels     = ['PSNR (dB) ↑', 'SSIM ↑', 'LPIPS ↓']
panel_titles = ['PSNR', 'SSIM', 'LPIPS']

for panel, ax in enumerate(axes):
    mi = metric_idx[panel]

    for res, color in res_colors.items():
        pts = our[res]
        # Sort by BPP
        pts_sorted = sorted(pts, key=lambda x: x[0])
        bpp_vals = [p[0] for p in pts_sorted]
        met_vals = [p[mi] for p in pts_sorted]

        # Line connecting our model points
        ax.plot(bpp_vals, met_vals, color=color, linewidth=1.2,
                linestyle='-', zorder=2)

        # Individual markers by K
        for ki, pt in enumerate(pts):
            ax.scatter(pt[0], pt[mi],
                       marker=K_markers[ki], color=color,
                       s=28, zorder=3, linewidths=0.5,
                       edgecolors=color)

        # H.264 dashed line
        h_pts = sorted(h264[res], key=lambda x: x[0])
        bpp_h = [p[0] for p in h_pts]
        met_h = [p[mi] for p in h_pts]
        ax.plot(bpp_h, met_h, color=color, linewidth=1.0,
                linestyle='--', zorder=2)
        ax.scatter(bpp_h, met_h,
                   marker='s', color='none', s=24, zorder=3,
                   edgecolors=color, linewidths=0.8)

    # LPIPS panel: shaded ultra-low bitrate region
    if panel == 2:
        ax.axvspan(0.0, 0.085, color='lightgrey', alpha=0.35, zorder=0)
        ax.text(0.0425, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.04,
                'Ultra-low\nbitrate', ha='center', va='bottom',
                fontsize=7, color='grey', style='italic',
                transform=ax.get_xaxis_transform(),
                clip_on=True)

    ax.set_xlabel('BPP (bits per pixel)', fontsize=10)
    ax.set_ylabel(ylabels[panel], fontsize=10)
    ax.set_title(panel_titles[panel], fontsize=10, pad=4)
    ax.tick_params(labelsize=9)

# ── Legend ───────────────────────────────────────────────────────────────────
# Resolution color patches
res_handles = [
    mpatches.Patch(color=res_colors[r], label=res_labels[r])
    for r in [64, 128, 256]
]
# Line style handles
style_handles = [
    Line2D([0], [0], color='grey', linestyle='-',  linewidth=1.2,
           marker='o', markersize=4, label='MS-VQ-VAE'),
    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.0,
           marker='s', markersize=4, markerfacecolor='none', label='H.264'),
]
# K marker handles
k_handles = [
    Line2D([0], [0], linestyle='none', marker=K_markers[i],
           color='grey', markersize=4, label=K_labels[i])
    for i in range(4)
]

axes[0].legend(
    handles=res_handles + style_handles + k_handles,
    fontsize=8, loc='lower right',
    framealpha=0.9, edgecolor='#cccccc',
    handlelength=1.6, handletextpad=0.5,
    borderpad=0.5, labelspacing=0.3
)

# LPIPS shaded label — apply after axes are finalised
ax_lpips = axes[2]
ylim = ax_lpips.get_ylim()
y_text = ylim[0] + 0.02 * (ylim[1] - ylim[0])
ax_lpips.text(0.0425, y_text,
              'Ultra-low\nbitrate', ha='center', va='bottom',
              fontsize=7, color='dimgrey', style='italic')

plt.tight_layout(pad=0.5, w_pad=0.8)
out1 = f"{OUT_DIR}/fig_rd_combined.png"
plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {out1}")
plt.show()
plt.close()


# ============================================================================
# FIGURE 2 — LPIPS scaling scatter
# ============================================================================

# Scaling law: LPIPS_pred = -0.0094*log2(K) + -0.0009*log2(r) + 0.2169
def pred_lpips(K, r):
    return -0.0094 * math.log2(K) + (-0.0009) * math.log2(r) + 0.2169

# All 12 points
points = [
    (128,  64,  0.1546),
    (256,  64,  0.1386),
    (512,  64,  0.1287),
    (1024, 64,  0.1140),
    (128,  128, 0.1399),
    (256,  128, 0.1251),
    (512,  128, 0.1245),
    (1024, 128, 0.1154),
    (128,  256, 0.1436),
    (256,  256, 0.1352),
    (512,  256, 0.1309),
    (1024, 256, 0.1194),
]

K_list    = [128, 256, 512, 1024]
K_mk_map  = {128: 'o', 256: '^', 512: 's', 1024: 'D'}

fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))

for K, r, actual in points:
    predicted = pred_lpips(K, r)
    color  = res_colors[r]
    marker = K_mk_map[K]
    ax2.scatter(actual, predicted,
                marker=marker, color=color,
                s=40, zorder=3, linewidths=0.6,
                edgecolors='white')

# Identity line y = x
lims = [0.10, 0.16]
ax2.plot(lims, lims, color='grey', linestyle='--',
         linewidth=0.9, zorder=1, label='_nolegend_')

ax2.set_xlim(0.10, 0.16)
ax2.set_ylim(0.10, 0.16)
ax2.set_aspect('equal')

ax2.set_xlabel('Actual LPIPS', fontsize=10)
ax2.set_ylabel('Predicted LPIPS', fontsize=10)
ax2.tick_params(labelsize=9)

# R² annotation
ax2.text(0.105, 0.157, r'$R^2 = 0.826$',
         fontsize=9, ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                   edgecolor='none', alpha=0.8))

# Legend — resolution colors
res_handles2 = [
    mpatches.Patch(color=res_colors[r], label=res_labels[r])
    for r in [64, 128, 256]
]
k_handles2 = [
    Line2D([0], [0], linestyle='none', marker=K_mk_map[K],
           color='grey', markersize=5,
           label=f'K={K}')
    for K in K_list
]
ax2.legend(handles=res_handles2 + k_handles2,
           fontsize=8, loc='lower right',
           framealpha=0.9, edgecolor='#cccccc',
           handlelength=1.2, handletextpad=0.5,
           borderpad=0.5, labelspacing=0.3)

plt.tight_layout(pad=0.4)
out2 = f"{OUT_DIR}/fig_lpips_scaling.png"
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {out2}")
plt.show()
plt.close()

print("Done. Both figures written to", OUT_DIR)
