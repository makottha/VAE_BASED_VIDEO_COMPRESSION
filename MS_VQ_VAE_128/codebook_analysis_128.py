"""
codebook_analysis_128.py — Codebook analysis for 3-level MS-VQ-VAE at 128x128

Generates 5 publication-quality figures + a summary CSV:

  Fig 1 — Codebook utilisation: active entries % for top, mid, bottom codebooks
  Fig 2 — Entropy efficiency: measured entropy / log2(K) per level
  Fig 3 — Rate breakdown: bits from top / mid / bottom prior per K
  Fig 4 — Index frequency histogram: power-law usage (bottom codebook)
  Fig 5 — Codebook embedding PCA: 2D projection of learned VQ embeddings

  codebook_summary.csv — full table for paper

Usage:
    conda activate vae_env
    cd MS_VQ_VAE_128
    python codebook_analysis_128.py
"""

import os
import csv
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.decomposition import PCA

sys.path.insert(0, "../MS_VQ_VAE_RD")
from dataset import VideoClipTensorDataset
from priors_3level import (
    TopPrior3D, MidPriorConditional3D, BottomPriorConditional3D,
    categorical_bits_from_logits,
)
from models_3level import MSVQVAE3Video

# ---- CONFIG -----------------------------------------------------------------
K_VALUES   = [128, 256, 512, 1024]
TEST_DIR   = "../dataset_128/test/tensors"
OUT_DIR    = "./outputs_msvqvae_128/codebook_analysis"
CKPT_DIR   = "./outputs_msvqvae_128"
N_CLIPS    = 500
BATCH_SIZE = 2
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP    = True

TOP_DIM    = 256
MID_DIM    = 192
BOTTOM_DIM = 128

TOP_PRIOR_EMB = 64;  TOP_PRIOR_HIDDEN = 128;  TOP_PRIOR_RESBLOCKS = 6
MID_PRIOR_EMB = 64;  MID_PRIOR_HIDDEN = 192;  MID_PRIOR_RESBLOCKS = 8
BOT_PRIOR_EMB = 64;  BOT_PRIOR_HIDDEN = 192;  BOT_PRIOR_RESBLOCKS = 8
# -----------------------------------------------------------------------------

COLORS  = {128: "#4e79a7", 256: "#f28e2b", 512: "#59a14f", 1024: "#e15759"}
MARKERS = {128: "o",       256: "s",       512: "^",       1024: "D"}
LEVEL_COLORS = {"top": "#4e79a7", "mid": "#59a14f", "bottom": "#f28e2b"}

os.makedirs(OUT_DIR, exist_ok=True)
print(f"[INFO] Device: {DEVICE}  |  N_CLIPS={N_CLIPS}")


def load_ae(K):
    ae = MSVQVAE3Video(
        top_dim=TOP_DIM, mid_dim=MID_DIM, bottom_dim=BOTTOM_DIM,
        K_top=K, K_mid=K, K_bottom=K,
        use_ema=True, use_vgg=False,
    ).to(DEVICE).eval()
    ckpt = torch.load(f"{CKPT_DIR}/ae_K{K}_epoch020.pt",
                      map_location=DEVICE, weights_only=False)
    ae.load_state_dict(ckpt["model"], strict=False)
    return ae


def load_priors(K):
    ckpt = torch.load(f"{CKPT_DIR}/priors_K{K}_epoch015.pt",
                      map_location=DEVICE, weights_only=False)
    top_p = TopPrior3D(K, emb_dim=TOP_PRIOR_EMB, hidden=TOP_PRIOR_HIDDEN,
                       n_res=TOP_PRIOR_RESBLOCKS).to(DEVICE).eval()
    mid_p = MidPriorConditional3D(K, cond_channels=TOP_DIM, emb_dim=MID_PRIOR_EMB,
                                   hidden=MID_PRIOR_HIDDEN, n_res=MID_PRIOR_RESBLOCKS).to(DEVICE).eval()
    bot_p = BottomPriorConditional3D(K, cond_channels=256, emb_dim=BOT_PRIOR_EMB,
                                      hidden=BOT_PRIOR_HIDDEN, n_res=BOT_PRIOR_RESBLOCKS).to(DEVICE).eval()
    top_p.load_state_dict(ckpt["top_prior"])
    mid_p.load_state_dict(ckpt["mid_prior"])
    bot_p.load_state_dict(ckpt["bot_prior"])
    return top_p, mid_p, bot_p


def get_loader():
    ds = VideoClipTensorDataset(TEST_DIR, subset_fraction=1.0, expected_T=32)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ---- Collect stats per K ----------------------------------------------------

print("[INFO] Collecting stats for all K values...")
stats = {}

for K in K_VALUES:
    print(f"  K={K}...")
    ae = load_ae(K)
    top_p, mid_p, bot_p = load_priors(K)
    loader = get_loader()

    hist_top = torch.zeros(K, dtype=torch.long)
    hist_mid = torch.zeros(K, dtype=torch.long)
    hist_bot = torch.zeros(K, dtype=torch.long)

    ent_top_list, ent_mid_list, ent_bot_list = [], [], []
    used_top_list, used_mid_list, used_bot_list = [], [], []
    bits_top_list, bits_mid_list, bits_bot_list = [], [], []

    n_clips = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE == "cuda")):
                out = ae(x)
                idx_t = out["idx_top"]
                idx_m = out["idx_mid"]
                idx_b = out["idx_bottom"]

                # Prior bits
                lt = top_p(idx_t)
                lm = mid_p(idx_m, out["z_top_up"])
                lb = bot_p(idx_b, out["z_mid_up"])
                bt = categorical_bits_from_logits(lt, idx_t).item()
                bm = categorical_bits_from_logits(lm, idx_m).item()
                bb = categorical_bits_from_logits(lb, idx_b).item()

            # Histograms
            hist_top += torch.bincount(idx_t.reshape(-1).cpu(), minlength=K)
            hist_mid += torch.bincount(idx_m.reshape(-1).cpu(), minlength=K)
            hist_bot += torch.bincount(idx_b.reshape(-1).cpu(), minlength=K)

            # Per-batch entropy & utilization
            for level, idx, hist_list in [
                ("top", idx_t, ent_top_list),
                ("mid", idx_m, ent_mid_list),
                ("bot", idx_b, ent_bot_list),
            ]:
                flat = idx.reshape(-1).cpu()
                h = torch.bincount(flat, minlength=K).float()
                total = h.sum().item()
                if total > 0:
                    p = h / total
                    p = p[p > 0]
                    ent = float(-(p * torch.log2(p)).sum().item())
                    hist_list.append(ent)

            used_top_list.append(int((hist_top > 0).sum().item()))
            used_mid_list.append(int((hist_mid > 0).sum().item()))
            used_bot_list.append(int((hist_bot > 0).sum().item()))

            bits_top_list.append(bt)
            bits_mid_list.append(bm)
            bits_bot_list.append(bb)

            n_clips += x.shape[0]
            if n_clips >= N_CLIPS:
                break

    stats[K] = {
        "hist_top":  hist_top.numpy().astype(float),
        "hist_mid":  hist_mid.numpy().astype(float),
        "hist_bot":  hist_bot.numpy().astype(float),
        "ent_top":   np.mean(ent_top_list),
        "ent_mid":   np.mean(ent_mid_list),
        "ent_bot":   np.mean(ent_bot_list),
        "used_top":  int((hist_top > 0).sum()),
        "used_mid":  int((hist_mid > 0).sum()),
        "used_bot":  int((hist_bot > 0).sum()),
        "bits_top":  np.mean(bits_top_list),
        "bits_mid":  np.mean(bits_mid_list),
        "bits_bot":  np.mean(bits_bot_list),
        "log2K":     math.log2(K),
    }

    del ae, top_p, mid_p, bot_p
    torch.cuda.empty_cache()
    print(f"    used=({stats[K]['used_top']}/{K}, {stats[K]['used_mid']}/{K}, {stats[K]['used_bot']}/{K})  "
          f"bits=({stats[K]['bits_top']:.0f}, {stats[K]['bits_mid']:.0f}, {stats[K]['bits_bot']:.0f})")

print("[INFO] Stats collected.")


# ---- Fig 1 — Codebook utilisation ------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(K_VALUES))
w = 0.25

for i, (level, color, label) in enumerate([
    ("top",    LEVEL_COLORS["top"],    "Top codebook"),
    ("mid",    LEVEL_COLORS["mid"],    "Mid codebook"),
    ("bottom", LEVEL_COLORS["bottom"], "Bottom codebook"),
]):
    vals = [stats[K][f"used_{level if level != 'bottom' else 'bot'}"] / K * 100 for K in K_VALUES]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([f"K={K}" for K in K_VALUES])
ax.set_ylabel("Active codebook entries (%)")
ax.set_title("Codebook Utilisation — 3-Level MS-VQ-VAE (128x128)")
ax.set_ylim(0, 120)
ax.legend()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_utilization.png"), dpi=150)
plt.close()
print("[DONE] Fig 1 - utilisation")


# ---- Fig 2 — Entropy efficiency ---------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
for ax, level, title in zip(axes,
                              ["top", "mid", "bot"],
                              ["Top Codebook", "Mid Codebook", "Bottom Codebook"]):
    for K in K_VALUES:
        ent = stats[K][f"ent_{level}"]
        eff = ent / stats[K]["log2K"] * 100
        bar = ax.bar(math.log2(K), eff, width=0.5,
                     color=COLORS[K], alpha=0.85, edgecolor="white", label=f"K={K}")
        ax.text(math.log2(K), eff + 1.0, f"{eff:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks([math.log2(K) for K in K_VALUES])
    ax.set_xticklabels([f"K={K}" for K in K_VALUES])
    ax.set_ylabel("Entropy efficiency (%)")
    ax.set_title(f"Entropy Efficiency - {title}")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Entropy Efficiency: Measured Entropy / log2(K) per Level", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_entropy_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[DONE] Fig 2 - entropy efficiency")


# ---- Fig 3 — Rate breakdown (top / mid / bottom) ----------------------------

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(K_VALUES))

bt = [stats[K]["bits_top"] for K in K_VALUES]
bm = [stats[K]["bits_mid"] for K in K_VALUES]
bb = [stats[K]["bits_bot"] for K in K_VALUES]
totals = [t + m + b for t, m, b in zip(bt, bm, bb)]

ax.bar(x, bt, label="Top prior bits",    color=LEVEL_COLORS["top"],    alpha=0.85)
ax.bar(x, bm, bottom=bt,
       label="Mid prior bits",    color=LEVEL_COLORS["mid"],    alpha=0.85)
ax.bar(x, bb, bottom=[t + m for t, m in zip(bt, bm)],
       label="Bottom prior bits", color=LEVEL_COLORS["bottom"], alpha=0.85)

for i, (t, m, b, tot) in enumerate(zip(bt, bm, bb, totals)):
    ax.text(i, tot + 500, f"total={tot:.0f}", ha="center", fontsize=8)
    ax.text(i, t / 2,     f"{t/tot*100:.0f}%", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold")
    ax.text(i, t + m / 2, f"{m/tot*100:.0f}%", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold")
    ax.text(i, t + m + b / 2, f"{b/tot*100:.0f}%", ha="center", va="center",
            fontsize=7, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"K={K}" for K in K_VALUES])
ax.set_ylabel("Expected bits per clip")
ax.set_title("Rate Breakdown: Top / Mid / Bottom Prior Bits (128x128)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_rate_breakdown.png"), dpi=150)
plt.close()
print("[DONE] Fig 3 - rate breakdown")


# ---- Fig 4 — Index frequency histogram (bottom codebook) --------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, K in zip(axes, K_VALUES):
    counts = stats[K]["hist_bot"].copy()
    counts_sorted = np.sort(counts)[::-1]
    total = counts_sorted.sum()
    freq = counts_sorted / total * 100

    active = int((counts > 0).sum())
    top10_pct = freq[:max(1, K // 10)].sum()

    rank = np.arange(1, K + 1)
    ax.bar(rank, freq, color=COLORS[K], alpha=0.8, width=1.0)
    ax.set_title(
        f"K={K}  |  active={active}/{K} ({100*active/K:.0f}%)  |  top-10%={top10_pct:.0f}% of bits",
        fontsize=9
    )
    ax.set_xlabel("Code rank (sorted by frequency)")
    ax.set_ylabel("Usage (%)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Bottom Codebook Index Frequency — 128x128 (sorted by rank)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_index_histogram.png"), dpi=150)
plt.close()
print("[DONE] Fig 4 - index histogram")


# ---- Fig 5 — Codebook embedding PCA (bottom codebook) ----------------------

print("[INFO] Computing PCA...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, K in zip(axes, K_VALUES):
    ckpt = torch.load(f"{CKPT_DIR}/ae_K{K}_epoch020.pt",
                      map_location="cpu", weights_only=False)
    emb = ckpt["model"]["vq_bottom.embeddings.weight"].numpy()   # (K, BOTTOM_DIM)

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)
    var = pca.explained_variance_ratio_ * 100

    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=8, alpha=0.6, color=COLORS[K])
    ax.set_title(
        f"K={K}  |  PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  (dim={BOTTOM_DIM})",
        fontsize=9
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

plt.suptitle("Bottom Codebook Embeddings - PCA Projection (128x128)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_embedding_pca.png"), dpi=150)
plt.close()
print("[DONE] Fig 5 - embedding PCA")


# ---- Summary table ----------------------------------------------------------

print(f"\n{'='*90}")
print(f"{'K':<6} {'Util_T%':>8} {'Util_M%':>8} {'Util_B%':>8} "
      f"{'EntEff_T%':>10} {'EntEff_M%':>10} {'EntEff_B%':>10} "
      f"{'Bits_T':>8} {'Bits_M':>8} {'Bits_B':>8} {'Total_bits':>11}")
print(f"{'-'*90}")

summary_rows = []
for K in K_VALUES:
    s = stats[K]
    log2K = s["log2K"]
    ut  = s["used_top"] / K * 100
    um  = s["used_mid"] / K * 100
    ub  = s["used_bot"] / K * 100
    eet = s["ent_top"] / log2K * 100
    eem = s["ent_mid"] / log2K * 100
    eeb = s["ent_bot"] / log2K * 100
    total = s["bits_top"] + s["bits_mid"] + s["bits_bot"]

    print(f"{K:<6} {ut:>8.1f} {um:>8.1f} {ub:>8.1f} "
          f"{eet:>10.1f} {eem:>10.1f} {eeb:>10.1f} "
          f"{s['bits_top']:>8.0f} {s['bits_mid']:>8.0f} {s['bits_bot']:>8.0f} {total:>11.0f}")

    summary_rows.append({
        "K": K,
        "util_top_pct": round(ut, 1), "util_mid_pct": round(um, 1), "util_bot_pct": round(ub, 1),
        "ent_eff_top_pct": round(eet, 1), "ent_eff_mid_pct": round(eem, 1), "ent_eff_bot_pct": round(eeb, 1),
        "bits_top": round(s["bits_top"], 1), "bits_mid": round(s["bits_mid"], 1),
        "bits_bot": round(s["bits_bot"], 1),
        "total_bits": round(total, 1),
    })

print(f"{'='*90}")

summary_csv = os.path.join(OUT_DIR, "codebook_summary.csv")
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\n[INFO] Summary CSV -> {summary_csv}")
print(f"[INFO] All figures -> {OUT_DIR}/")
print("\n[ALL DONE]")
