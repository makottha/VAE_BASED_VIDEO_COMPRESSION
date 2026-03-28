# codebook_analysis.py
"""
Codebook analysis for the K-sweep VQ-VAE2 video compression paper.

Generates 5 publication-quality figures + a summary table:

  Fig 1 — Codebook utilization: active entries / K for top & bottom codebooks
  Fig 2 — Entropy efficiency: measured entropy / log2(K) (0=useless, 1=uniform)
  Fig 3 — Rate breakdown: bits contributed by top vs bottom prior per K
  Fig 4 — Index frequency histogram: power-law code usage for each K (bottom CB)
  Fig 5 — Codebook embedding PCA: 2D projection of learned VQ embeddings

  Table — Codebook analysis summary (for paper Table)

Usage:
    python codebook_analysis.py
"""

import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from sklearn.decomposition import PCA

from models import MSVQVAE2Video
from dataset import VideoClipTensorDataset
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
K_VALUES = [128, 256, 512, 1024]
EVAL_CSV = {
    K: f"./outputs_msvqvae_rd/metrics_K{K}.csv" for K in K_VALUES
}
AE_CKPTS = {
    128:  "./outputs_msvqvae_rd/ae_K128_epoch020.pt",
    256:  "./outputs_msvqvae_rd/ae_K256_epoch020.pt",
    512:  "./outputs_msvqvae_rd/ae_K512_epoch020.pt",
    1024: "./outputs_msvqvae_rd/ae_K1024_epoch020.pt",
}
TEST_DIR = "../dataset_64/test/tensors"
N_HIST_CLIPS = 500       # clips for index histogram (Fig 4)
OUT_DIR = "./outputs_msvqvae_rd/codebook_analysis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_DIM    = 256
BOTTOM_DIM = 128

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Colours
# -----------------------------------------------------------------------
COLORS = {128: "#4e79a7", 256: "#f28e2b", 512: "#59a14f", 1024: "#e15759"}
MARKERS = {128: "o", 256: "s", 512: "^", 1024: "D"}

# -----------------------------------------------------------------------
# Helper: load eval CSV
# -----------------------------------------------------------------------
def load_csv(path):
    return list(csv.DictReader(open(path)))

def col(rows, key):
    return [float(r[key]) for r in rows]


# -----------------------------------------------------------------------
# Step 1: read per-clip stats from eval CSVs
# -----------------------------------------------------------------------
data = {}
for K in K_VALUES:
    rows = load_csv(EVAL_CSV[K])
    log2K = math.log2(K)
    data[K] = {
        "rows": rows,
        "used_top":    np.array(col(rows, "used_top")),
        "used_bot":    np.array(col(rows, "used_bottom")),
        "ent_top":     np.array(col(rows, "entropy_top")),
        "ent_bot":     np.array(col(rows, "entropy_bottom")),
        "bits_top":    np.array(col(rows, "bits_top")),
        "bits_bot":    np.array(col(rows, "bits_bottom")),
        "bits_per_sym":np.array(col(rows, "bits_per_sym")),
        "bpp":         np.array(col(rows, "bpp")),
        "psnr":        np.array(col(rows, "psnr")),
        "ssim":        np.array(col(rows, "ssim")),
        "lpips":       np.array(col(rows, "lpips")),
        "log2K":       log2K,
    }

print("[INFO] Eval CSVs loaded.")


# -----------------------------------------------------------------------
# Fig 1 — Codebook utilisation (mean active entries / K)
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

util_top  = [data[K]["used_top"].mean()  / K * 100 for K in K_VALUES]
util_bot  = [data[K]["used_bot"].mean()  / K * 100 for K in K_VALUES]

x = np.arange(len(K_VALUES))
w = 0.35
bars_top = ax.bar(x - w/2, util_top, w, label="Top codebook",
                  color="#4e79a7", alpha=0.85, edgecolor="white")
bars_bot = ax.bar(x + w/2, util_bot, w, label="Bottom codebook",
                  color="#f28e2b", alpha=0.85, edgecolor="white")

for bar, v in zip(bars_top, util_top):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
for bar, v in zip(bars_bot, util_bot):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"K={K}" for K in K_VALUES])
ax.set_ylabel("Active codebook entries (%)")
ax.set_title("Codebook Utilisation vs. Codebook Size")
ax.set_ylim(0, 115)
ax.legend()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_utilization.png"), dpi=150)
plt.close()
print("[DONE] Fig 1 — utilisation")


# -----------------------------------------------------------------------
# Fig 2 — Entropy efficiency: measured_entropy / log2(K)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, level, title in zip(axes, ["top", "bot"], ["Top Codebook", "Bottom Codebook"]):
    effs = []
    for K in K_VALUES:
        ent = data[K][f"ent_{level}"]
        eff = ent / data[K]["log2K"] * 100   # % of max entropy
        effs.append(eff.mean())
        ax.bar(math.log2(K), eff.mean(), width=0.5,
               color=COLORS[K], alpha=0.85, edgecolor="white",
               label=f"K={K}")
        ax.text(math.log2(K), eff.mean() + 1.0,
                f"{eff.mean():.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([math.log2(K) for K in K_VALUES])
    ax.set_xticklabels([f"K={K}\n(log₂K={int(math.log2(K))})" for K in K_VALUES])
    ax.set_ylabel("Entropy efficiency (% of max)")
    ax.set_title(f"Entropy Efficiency — {title}")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, label="Max (uniform)")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Entropy Efficiency: Measured Entropy / log₂(K)", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_entropy_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[DONE] Fig 2 — entropy efficiency")


# -----------------------------------------------------------------------
# Fig 3 — Rate breakdown: bits_top vs bits_bottom
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

mean_top = [data[K]["bits_top"].mean() for K in K_VALUES]
mean_bot = [data[K]["bits_bot"].mean() for K in K_VALUES]
total    = [t + b for t, b in zip(mean_top, mean_bot)]

x = np.arange(len(K_VALUES))
ax.bar(x, mean_top, label="Top prior bits",    color="#4e79a7", alpha=0.85)
ax.bar(x, mean_bot, bottom=mean_top,
       label="Bottom prior bits", color="#f28e2b", alpha=0.85)

for i, (t, tot) in enumerate(zip(mean_top, total)):
    pct = t / tot * 100
    ax.text(i, tot + 200, f"total={tot:.0f}", ha="center", fontsize=8)
    ax.text(i, t / 2, f"{pct:.0f}%", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"K={K}" for K in K_VALUES])
ax.set_ylabel("Expected bits per clip")
ax.set_title("Rate Breakdown: Top vs. Bottom Codebook")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_rate_breakdown.png"), dpi=150)
plt.close()
print("[DONE] Fig 3 — rate breakdown")


# -----------------------------------------------------------------------
# Fig 4 — Index frequency histogram (bottom codebook, N_HIST_CLIPS clips)
# -----------------------------------------------------------------------
print(f"[INFO] Collecting index histograms from {N_HIST_CLIPS} test clips...")

from dataset import VideoClipTensorDataset

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, K in zip(axes, K_VALUES):
    print(f"  K={K}...")
    ae = MSVQVAE2Video(
        top_dim=TOP_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_bottom=K,
        commit_top=0.25, commit_bottom=0.25,
        norm="gn", use_vgg=False, soft_temp=1.0, use_ema=False,
    ).to(DEVICE)
    ckpt = torch.load(AE_CKPTS[K], map_location="cpu")
    ae.load_state_dict(ckpt["ae"], strict=False)
    ae.eval()

    ds = VideoClipTensorDataset(TEST_DIR, subset_fraction=1.0, expected_T=32)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    hist_bot = torch.zeros(K, dtype=torch.long)
    n_clips = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            out = ae(x)
            idx_b = out["idx_bottom"].reshape(-1).cpu()
            hist_bot += torch.bincount(idx_b, minlength=K)
            n_clips += x.shape[0]
            if n_clips >= N_HIST_CLIPS:
                break

    del ae

    # Sort descending
    counts = hist_bot.numpy().astype(float)
    counts_sorted = np.sort(counts)[::-1]
    total_counts = counts_sorted.sum()
    freq = counts_sorted / total_counts * 100   # percent

    active = int((counts > 0).sum())
    top10_pct = freq[:max(1, K//10)].sum()

    rank = np.arange(1, K + 1)
    ax.bar(rank, freq, color=COLORS[K], alpha=0.8, width=1.0)
    ax.set_title(f"K={K}  |  active={active}/{K} ({100*active/K:.0f}%)  |  top-10%={top10_pct:.0f}% of bits")
    ax.set_xlabel("Code rank (sorted by frequency)")
    ax.set_ylabel("Usage (%)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Bottom Codebook Index Frequency (sorted by rank)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_index_histogram.png"), dpi=150)
plt.close()
print("[DONE] Fig 4 — index histograms")


# -----------------------------------------------------------------------
# Fig 5 — Codebook embedding PCA (bottom codebook)
# -----------------------------------------------------------------------
print("[INFO] Computing codebook embedding PCA...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, K in zip(axes, K_VALUES):
    ckpt = torch.load(AE_CKPTS[K], map_location="cpu")
    # Extract bottom VQ embeddings from state dict
    emb = ckpt["ae"]["vq_bottom.embeddings.weight"].numpy()   # (K, D)

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)   # (K, 2)
    var = pca.explained_variance_ratio_ * 100

    ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
               s=8, alpha=0.6, color=COLORS[K])
    ax.set_title(
        f"K={K}  |  PCA var: PC1={var[0]:.1f}%  PC2={var[1]:.1f}%\n"
        f"Embedding dim={BOTTOM_DIM}",
        fontsize=9
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

plt.suptitle("Bottom Codebook Embeddings — PCA Projection (2D)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_embedding_pca.png"), dpi=150)
plt.close()
print("[DONE] Fig 5 — embedding PCA")


# -----------------------------------------------------------------------
# Summary table (print + CSV)
# -----------------------------------------------------------------------
print("\n" + "="*80)
print(f"{'K':<8} {'Util_Top%':>10} {'Util_Bot%':>10} {'Ent_Top':>9} {'Ent_Eff_T%':>11} "
      f"{'Ent_Bot':>9} {'Ent_Eff_B%':>11} {'Bits/Sym':>10} {'BPP':>8} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
print("-"*80)

summary_rows = []
for K in K_VALUES:
    d = data[K]
    log2K = d["log2K"]
    ut = d["used_top"].mean() / K * 100
    ub = d["used_bot"].mean() / K * 100
    et = d["ent_top"].mean()
    eb = d["ent_bot"].mean()
    eet = et / log2K * 100
    eeb = eb / log2K * 100
    bps = d["bits_per_sym"].mean()
    bpp = d["bpp"].mean()
    psnr = d["psnr"].mean()
    ssim = d["ssim"].mean()
    lpips = d["lpips"].mean()

    print(f"{K:<8} {ut:>10.1f} {ub:>10.1f} {et:>9.3f} {eet:>11.1f} "
          f"{eb:>9.3f} {eeb:>11.1f} {bps:>10.4f} {bpp:>8.4f} {psnr:>7.2f} {ssim:>7.4f} {lpips:>7.4f}")

    summary_rows.append({
        "K": K, "util_top_pct": round(ut, 2), "util_bot_pct": round(ub, 2),
        "entropy_top": round(et, 4), "entropy_eff_top_pct": round(eet, 2),
        "entropy_bot": round(eb, 4), "entropy_eff_bot_pct": round(eeb, 2),
        "bits_per_sym": round(bps, 4), "bpp": round(bpp, 4),
        "psnr": round(psnr, 3), "ssim": round(ssim, 4), "lpips": round(lpips, 4),
    })

print("="*80)

summary_path = os.path.join(OUT_DIR, "codebook_summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\n[INFO] Summary CSV → {summary_path}")
print(f"[INFO] All figures → {OUT_DIR}/")
