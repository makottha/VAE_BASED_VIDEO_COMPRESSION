"""
generate_qual_figure.py
=======================
Qualitative comparison figure for MS-VQ-VAE video compression paper.

Layout:  5 rows (clips) × 5 columns (Original | K=128 | K=256 | K=512 | K=1024)
Output:  publication-ready PDF

Usage:
    python generate_qual_figure.py \
        --clips_dir  /path/to/test_clips \
        --ckpt_k128  outputs_msvqvae_rd/ae_K128_epoch020.pt \
        --ckpt_k256  outputs_msvqvae_rd/ae_K256_epoch020.pt \
        --ckpt_k512  outputs_msvqvae_rd/ae_K512_epoch020.pt \
        --ckpt_k1024 outputs_msvqvae_rd/ae_K1024_epoch020.pt \
        --out        paper/figures/qual_figure.pdf \
        --frame_idx  16
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# ---------------------------------------------------------------------------
# Minimal model re-implementation (matches MS_VQ_VAE_RD/models.py exactly)
# ---------------------------------------------------------------------------

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, norm="gn"):
        super().__init__()
        Norm = nn.BatchNorm3d if norm == "bn" else lambda c: nn.GroupNorm(32, c)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            Norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            Norm(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, soft_temp: float = 1.0,
                 use_ema: bool = False, ema_decay: float = 0.99,
                 ema_epsilon: float = 1e-5, restart_threshold: float = 1.0):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.soft_temp = float(soft_temp)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.restart_threshold = float(restart_threshold)

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0 / self.num_embeddings)

        if self.use_ema:
            self.register_buffer("cluster_size", torch.ones(self.num_embeddings))
            self.register_buffer("embed_avg", self.embeddings.weight.data.clone())

    def forward(self, x: torch.Tensor):
        import math
        B, D, T, H, W = x.shape
        x_fp32 = x.float()
        x_flat = x_fp32.permute(0, 2, 3, 4, 1).contiguous()
        flat = x_flat.view(-1, D)
        emb = self.embeddings.weight.float()

        dist = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(emb ** 2, dim=1).unsqueeze(0)
            - 2.0 * flat @ emb.t()
        )

        indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(indices, num_classes=self.num_embeddings).float()
        quantized = (encodings @ emb).view(B, T, H, W, D)

        # straight-through (no grad needed at inference)
        quantized_st = x_flat + (quantized - x_flat).detach()
        indices = indices.view(B, T, H, W).long()
        quantized_st = quantized_st.permute(0, 4, 1, 2, 3).contiguous()

        dist_s = dist - dist.min(dim=1, keepdim=True)[0].detach()
        soft_assign = torch.softmax(-dist_s / self.soft_temp, dim=1)
        log_soft = torch.log_softmax(-dist_s / self.soft_temp, dim=1)
        soft_bits = -(soft_assign * log_soft).sum(1).sum() / math.log(2)

        vq_loss = torch.tensor(0.0)  # not needed at inference
        return quantized_st.to(x.dtype), vq_loss, indices, soft_bits


class MSVQVAE2Video(nn.Module):
    def __init__(self, top_dim=256, bottom_dim=128,
                 K_top=1024, K_bottom=1024,
                 commit_top=0.25, commit_bottom=0.25,
                 norm="gn", use_vgg=True, vgg_layers=16,
                 soft_temp=1.0, use_ema=False):
        super().__init__()
        self.top_dim = int(top_dim)
        self.bottom_dim = int(bottom_dim)

        # Bottom encoder: spatial /4, temporal /4
        self.enc_bottom = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, norm=norm),
            nn.Conv3d(64, self.bottom_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.bottom_dim, norm=norm),
        )

        # Top encoder: /8 from input
        self.enc_top = nn.Sequential(
            nn.Conv3d(self.bottom_dim, self.top_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.top_dim, norm=norm),
        )

        self.vq_top = VectorQuantizer(
            K_top, self.top_dim,
            commitment_cost=commit_top, soft_temp=soft_temp, use_ema=use_ema)
        self.vq_bottom = VectorQuantizer(
            K_bottom, self.bottom_dim,
            commitment_cost=commit_bottom, soft_temp=soft_temp, use_ema=use_ema)

        Norm = nn.BatchNorm3d if norm == "bn" else lambda c: nn.GroupNorm(32, c)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.top_dim + self.bottom_dim, 256,
                               kernel_size=4, stride=2, padding=1),
            Norm(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            Norm(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # VGG needed so state_dict keys match — weights loaded but unused at inference
        self.use_vgg = bool(use_vgg)
        if self.use_vgg:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:vgg_layers]
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.eval()
        else:
            self.vgg = None

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass, returns reconstructed video (B,3,T,H,W) in [0,1]."""
        z_b = self.enc_bottom(x)
        z_t = self.enc_top(z_b)
        z_t_q, _, _, _ = self.vq_top(z_t)
        z_b_q, _, _, _ = self.vq_bottom(z_b)
        z_t_up = F.interpolate(z_t_q, size=z_b_q.shape[2:],
                               mode="trilinear", align_corners=False)
        dec_in = torch.cat([z_t_up, z_b_q], dim=1)
        return self.decoder(dec_in)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = {
    "original": "#2ca02c",   # green
    128:        "#d62728",   # red
    256:        "#ff7f0e",   # orange
    512:        "#1f77b4",   # blue
    1024:       "#9467bd",   # purple
}

K_VALUES = [128, 256, 512, 1024]


def load_model(ckpt_path: str, K: int) -> MSVQVAE2Video:
    """Load a K-sweep checkpoint (saved as {"ae": state_dict, ...})."""
    print(f"  Loading K={K} from {ckpt_path} ...", end=" ", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["ae"] if "ae" in ckpt else ckpt  # fallback: bare state_dict

    model = MSVQVAE2Video(
        top_dim=256, bottom_dim=128,
        K_top=K, K_bottom=K,
        commit_top=0.25, commit_bottom=0.25,
        norm="gn", use_vgg=True, vgg_layers=16,
        soft_temp=1.0, use_ema=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    print("OK")
    return model


def load_clips(clips_dir: str, n: int = 5) -> list:
    """Return up to n .pt clip tensors, each (3,32,64,64) float32 in [0,1]."""
    paths = sorted(Path(clips_dir).glob("*.pt"))
    if len(paths) == 0:
        sys.exit(f"[ERROR] No .pt files found in {clips_dir}")
    if len(paths) < n:
        print(f"[WARN] Only {len(paths)} clip(s) found; requested {n}. Using all.")
        n = len(paths)

    # Pick n evenly-spaced clips for diversity
    indices = np.linspace(0, len(paths) - 1, n, dtype=int)
    clips = []
    for i in indices:
        t = torch.load(paths[i], map_location="cpu")
        if t.dim() == 4:          # (C,T,H,W)  → add batch dim
            t = t.unsqueeze(0)
        t = t.float().clamp(0.0, 1.0)
        clips.append((t, paths[i].stem))
        print(f"  Loaded clip: {paths[i].name}  shape={tuple(t.shape)}")
    return clips


def frame_to_hwc(tensor: torch.Tensor, frame_idx: int) -> np.ndarray:
    """
    Extract one frame from a (B,3,T,H,W) tensor.
    Returns (H,W,3) float32 numpy array in [0,1].
    """
    f = tensor[0, :, frame_idx, :, :].detach().cpu()  # (3,H,W)
    f = f.permute(1, 2, 0).numpy()                    # (H,W,3)
    return np.clip(f, 0.0, 1.0)


def add_border(ax, color: str, linewidth: float = 4.0):
    """Draw a colored border around an axes by adding a rect patch."""
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)
        spine.set_visible(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate qualitative comparison figure for MS-VQ-VAE paper")
    p.add_argument("--clips_dir",   required=True,
                   help="Directory containing .pt clip tensors (C,T,H,W)")
    p.add_argument("--ckpt_k128",   required=True, help="Checkpoint for K=128")
    p.add_argument("--ckpt_k256",   required=True, help="Checkpoint for K=256")
    p.add_argument("--ckpt_k512",   required=True, help="Checkpoint for K=512")
    p.add_argument("--ckpt_k1024",  required=True, help="Checkpoint for K=1024")
    p.add_argument("--out",         default="qual_figure.pdf",
                   help="Output file path (PDF recommended)")
    p.add_argument("--frame_idx",   type=int, default=16,
                   help="Which frame index to display (default: 16)")
    p.add_argument("--n_clips",     type=int, default=5,
                   help="Number of clips to show (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_map = {
        128:  args.ckpt_k128,
        256:  args.ckpt_k256,
        512:  args.ckpt_k512,
        1024: args.ckpt_k1024,
    }

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    print("\n[1/3] Loading checkpoints ...")
    models = {}
    for K in K_VALUES:
        models[K] = load_model(ckpt_map[K], K)

    # ------------------------------------------------------------------
    # 2. Load clips and run inference
    # ------------------------------------------------------------------
    print(f"\n[2/3] Loading {args.n_clips} clips from {args.clips_dir} ...")
    clips = load_clips(args.clips_dir, n=args.n_clips)
    n_clips = len(clips)

    print(f"\n       Running inference (frame_idx={args.frame_idx}) ...")
    # frames[clip_idx] = {"original": ndarray, 128: ndarray, ...}
    all_frames = []
    for clip_tensor, clip_name in clips:
        entry = {"name": clip_name}
        entry["original"] = frame_to_hwc(clip_tensor, args.frame_idx)
        for K in K_VALUES:
            with torch.no_grad():
                recon = models[K].encode_decode(clip_tensor)
            entry[K] = frame_to_hwc(recon, args.frame_idx)
            print(f"    Clip '{clip_name}'  K={K:>4d}  done")
        all_frames.append(entry)

    # ------------------------------------------------------------------
    # 3. Save individual PNG frames
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    frames_dir = out_path.parent / "qual_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/4] Saving individual PNGs → {frames_dir}/")

    col_keys   = ["original"] + K_VALUES
    col_labels = ["original"] + [f"K{K}" for K in K_VALUES]

    for entry in all_frames:
        clip_name = entry["name"]
        for key, label in zip(col_keys, col_labels):
            img_arr = (entry[key] * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            fname = frames_dir / f"{clip_name}__{label}.png"
            img.save(fname)
            print(f"    Saved {fname.name}")

    # ------------------------------------------------------------------
    # 4. Build figure
    # ------------------------------------------------------------------
    print(f"\n[4/4] Rendering figure → {args.out}")

    n_cols = 1 + len(K_VALUES)   # Original + 4 K values
    col_keys    = ["original"] + K_VALUES
    col_labels  = ["Original"] + [f"K = {K}" for K in K_VALUES]
    col_colors  = [COLORS["original"]] + [COLORS[K] for K in K_VALUES]

    # Figure geometry
    cell_w, cell_h = 1.8, 1.8       # inches per cell
    left_margin    = 0.8             # for row labels
    top_margin     = 0.55            # for column headers
    fig_w = left_margin + n_cols * cell_w + 0.2
    fig_h = top_margin  + n_clips * cell_h + 0.1

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    # Axes grid (row=clip, col=method)
    axes = []
    for r in range(n_clips):
        row_axes = []
        for c in range(n_cols):
            left   = (left_margin + c * cell_w) / fig_w
            bottom = 1.0 - (top_margin + (r + 1) * cell_h) / fig_h
            width  = (cell_w * 0.88) / fig_w
            height = (cell_h * 0.88) / fig_h
            ax = fig.add_axes([left, bottom, width, height])
            row_axes.append(ax)
        axes.append(row_axes)

    # Column headers
    for c, (label, color) in enumerate(zip(col_labels, col_colors)):
        left   = (left_margin + c * cell_w + cell_w * 0.44) / fig_w
        top_y  = 1.0 - (top_margin * 0.35) / fig_h
        fig.text(left, top_y, label,
                 ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color,
                 transform=fig.transFigure)

    # Row labels
    for r, entry in enumerate(all_frames):
        bottom = 1.0 - (top_margin + (r + 0.5) * cell_h) / fig_h
        fig.text(left_margin * 0.45 / fig_w, bottom,
                 entry["name"],
                 ha="center", va="center",
                 fontsize=7, color="#333333",
                 transform=fig.transFigure,
                 rotation=0)

    # Fill cells
    border_lw = 3.5
    for r, entry in enumerate(all_frames):
        for c, (key, color) in enumerate(zip(col_keys, col_colors)):
            ax = axes[r][c]
            img = entry[key]
            ax.imshow(img, interpolation="lanczos", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            add_border(ax, color, linewidth=border_lw)

    # Legend strip at the bottom
    legend_y = 0.012
    legend_items = [
        mpatches.Patch(color=COLORS["original"], label="Original"),
    ] + [
        mpatches.Patch(color=COLORS[K], label=f"K = {K}") for K in K_VALUES
    ]
    fig.legend(handles=legend_items,
               loc="lower center",
               ncol=n_cols,
               fontsize=8,
               frameon=True,
               framealpha=0.9,
               edgecolor="#aaaaaa",
               bbox_to_anchor=(0.5, legend_y),
               bbox_transform=fig.transFigure)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=out_path.suffix.lstrip(".") or "pdf",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
