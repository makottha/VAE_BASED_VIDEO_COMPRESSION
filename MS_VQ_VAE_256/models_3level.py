"""
models_3level.py — Three-level hierarchical VQ-VAE for 128×128 video compression

Architecture (128×128 input, T=32 frames):

  Encoder (bottom → mid → top):
    enc_bottom: stride 2×2×2  → h_b  (B, 128, T/2, H/4,  W/4 ) = (B,128,16,32,32)
    enc_mid:    stride 2×2×2  → h_m  (B, 192, T/4, H/8,  W/8 ) = (B,192, 8,16,16)
    enc_top:    stride 2×2×2  → h_t  (B, 256, T/8, H/16, W/16) = (B,256, 4, 8, 8)

  VQ codebooks (each with K entries, EMA updates):
    vq_top    → idx_top    (B, 4,  8,  8)  — 256 symbols/clip
    vq_mid    → idx_mid    (B, 8, 16, 16)  — 2048 symbols/clip
    vq_bottom → idx_bottom (B,16, 32, 32)  — 16384 symbols/clip

  Decoder (top-down cascade with conditioning):
    top_up   → concat with mid_q  → dec_stage1
    stage1   → concat with bot_q  → dec_stage2
    stage2   → pixel output (3, T, H, W)

  Priors (see priors_3level.py):
    TopPrior3D          : p(z_top)
    MidPriorConditional3D  : p(z_mid | z_top)
    BottomPriorConditional3D: p(z_bot | z_mid)

Parameters: ~26M (vs 17.8M for 2-level 64×64 model)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


# ─── Building blocks ────────────────────────────────────────────────────────

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


# ─── Vector Quantizer (identical to MS_VQ_VAE_RD version) ───────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-through VQ with EMA codebook updates and dead-code restart.
    use_ema=True is mandatory for stable training at all K values.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, soft_temp: float = 1.0,
                 use_ema: bool = True, ema_decay: float = 0.99,
                 ema_epsilon: float = 1e-5, restart_threshold: float = 1.0):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim  = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.soft_temp       = float(soft_temp)
        self.use_ema         = bool(use_ema)
        self.ema_decay       = float(ema_decay)
        self.ema_epsilon     = float(ema_epsilon)
        self.restart_threshold = float(restart_threshold)

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0 / self.num_embeddings)

        if self.use_ema:
            self.register_buffer("cluster_size", torch.ones(self.num_embeddings))
            self.register_buffer("embed_avg",    self.embeddings.weight.data.clone())

    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        with torch.no_grad():
            new_counts = encodings.sum(0)
            self.cluster_size.mul_(self.ema_decay).add_(new_counts, alpha=1.0 - self.ema_decay)

            new_sum = encodings.t() @ flat
            self.embed_avg.mul_(self.ema_decay).add_(new_sum, alpha=1.0 - self.ema_decay)

            n = self.cluster_size.sum()
            smoothed = (
                (self.cluster_size + self.ema_epsilon)
                / (n + self.num_embeddings * self.ema_epsilon) * n
            )
            updated = self.embed_avg / smoothed.unsqueeze(1)

            dead = self.cluster_size < self.restart_threshold
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                n_available = flat.shape[0]
                n_restart   = min(n_dead, n_available)
                perm        = torch.randperm(n_available, device=flat.device)[:n_restart]
                restart_vecs   = flat[perm].float()
                dead_indices   = dead.nonzero(as_tuple=False).squeeze(1)[:n_restart]
                updated[dead_indices]          = restart_vecs
                self.cluster_size[dead_indices] = self.restart_threshold
                self.embed_avg[dead_indices]    = restart_vecs * self.restart_threshold

            self.embeddings.weight.data.copy_(updated)

    def forward(self, x: torch.Tensor):
        B, D, T, H, W = x.shape
        x_dtype = x.dtype

        with torch.amp.autocast('cuda', enabled=False):
            x_fp32  = x.float()
            x_flat  = x_fp32.permute(0, 2, 3, 4, 1).contiguous()
            flat    = x_flat.view(-1, D)
            emb     = self.embeddings.weight.float()

            dist = (
                torch.sum(flat ** 2, dim=1, keepdim=True)
                + torch.sum(emb ** 2, dim=1).unsqueeze(0)
                - 2.0 * flat @ emb.t()
            )

            indices   = torch.argmin(dist, dim=1)
            encodings = F.one_hot(indices, num_classes=self.num_embeddings).float()
            quantized = (encodings @ emb).view(B, T, H, W, D)

            if self.use_ema and self.training:
                self._ema_update(flat.detach(), encodings.detach())
                vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), x_flat)
            else:
                codebook_loss = F.mse_loss(quantized, x_flat.detach())
                commit_loss   = F.mse_loss(quantized.detach(), x_flat)
                vq_loss       = codebook_loss + self.commitment_cost * commit_loss

            quantized_st = x_flat + (quantized - x_flat).detach()
            indices      = indices.view(B, T, H, W).long()
            quantized_st = quantized_st.permute(0, 4, 1, 2, 3).contiguous()

            dist_shifted = dist - dist.min(dim=1, keepdim=True)[0].detach()
            soft_assign  = torch.softmax(-dist_shifted / self.soft_temp, dim=1)
            log_soft     = torch.log_softmax(-dist_shifted / self.soft_temp, dim=1)
            soft_bits    = -(soft_assign * log_soft).sum(1).sum() / math.log(2)

        return quantized_st.to(x_dtype), vq_loss, indices, soft_bits


# ─── Three-level MS-VQ-VAE ───────────────────────────────────────────────────

class MSVQVAE3Video(nn.Module):
    """
    Three-level hierarchical VQ-VAE for 128×128 video compression.

    Latent hierarchy (for T=32, H=W=128):
      top:    (B, 256, T/8,  H/16, W/16) = (B,256, 4,  8,  8)  →  256 symbols/clip
      mid:    (B, 192, T/4,  H/8,  W/8 ) = (B,192, 8, 16, 16)  →  2048 symbols/clip
      bottom: (B, 128, T/2,  H/4,  W/4 ) = (B,128,16, 32, 32)  →  16384 symbols/clip

    Constructor args:
      top_dim, mid_dim, bottom_dim  — codebook embedding dimensions
      K_top, K_mid, K_bottom        — codebook sizes (same K used for all three in sweep)
      commit_*                      — commitment loss weights
      norm                          — 'gn' (GroupNorm) or 'bn' (BatchNorm)
      use_vgg                       — enable VGG perceptual loss feature extractor
      vgg_layers                    — VGG feature depth (default 16)
      soft_temp                     — soft assignment temperature for rate proxy
      use_ema                       — EMA codebook updates (MANDATORY — always True)
    """
    def __init__(
        self,
        top_dim=256,
        mid_dim=192,
        bottom_dim=128,
        K_top=1024,
        K_mid=1024,
        K_bottom=1024,
        commit_top=0.25,
        commit_mid=0.25,
        commit_bottom=0.25,
        norm="gn",
        use_vgg=True,
        vgg_layers=16,
        soft_temp=1.0,
        use_ema=True,
    ):
        super().__init__()
        self.top_dim    = int(top_dim)
        self.mid_dim    = int(mid_dim)
        self.bottom_dim = int(bottom_dim)
        self.K_top      = int(K_top)
        self.K_mid      = int(K_mid)
        self.K_bottom   = int(K_bottom)

        Norm = nn.BatchNorm3d if norm == "bn" else lambda c: nn.GroupNorm(32, c)

        # ── Encoders (each stride-2 in all 3 dims) ──────────────────────────
        # Input: (B, 3, T, H, W)
        # h_b:  (B, bottom_dim, T/2, H/4, W/4)
        self.enc_bottom = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, norm=norm),
            nn.Conv3d(64, self.bottom_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.bottom_dim, norm=norm),
        )

        # h_m:  (B, mid_dim, T/4, H/8, W/8)
        self.enc_mid = nn.Sequential(
            nn.Conv3d(self.bottom_dim, self.mid_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.mid_dim, norm=norm),
        )

        # h_t:  (B, top_dim, T/8, H/16, W/16)
        self.enc_top = nn.Sequential(
            nn.Conv3d(self.mid_dim, self.top_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.top_dim, norm=norm),
        )

        # ── VQ codebooks ────────────────────────────────────────────────────
        vq_kwargs = dict(soft_temp=float(soft_temp), use_ema=bool(use_ema))
        self.vq_top    = VectorQuantizer(self.K_top,    self.top_dim,    commit_top,    **vq_kwargs)
        self.vq_mid    = VectorQuantizer(self.K_mid,    self.mid_dim,    commit_mid,    **vq_kwargs)
        self.vq_bottom = VectorQuantizer(self.K_bottom, self.bottom_dim, commit_bottom, **vq_kwargs)

        # ── Decoder (top-down cascade) ───────────────────────────────────────
        # Stage 1: top_q upsampled + mid_q → (B, mid_dim, T/4, H/8, W/8)
        self.dec_stage1 = nn.Sequential(
            nn.ConvTranspose3d(self.top_dim + self.mid_dim, 256, kernel_size=4, stride=2, padding=1),
            Norm(256),
            nn.ReLU(inplace=True),
            ResidualBlock3D(256, norm=norm),
        )

        # Stage 2: stage1 upsampled + bottom_q → (B, 128, T/2, H/4, W/4)
        self.dec_stage2 = nn.Sequential(
            nn.ConvTranspose3d(256 + self.bottom_dim, 128, kernel_size=4, stride=2, padding=1),
            Norm(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128, norm=norm),
        )

        # Stage 3: upsample to pixel space → (B, 3, T, H, W)
        self.dec_out = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            Norm(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # ── Perceptual loss (frozen VGG, frame-wise) ─────────────────────────
        self.use_vgg = bool(use_vgg)
        if self.use_vgg:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:vgg_layers]
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg.eval()
        else:
            self.vgg = None

    @torch.no_grad()
    def _entropy_stats(self, indices: torch.Tensor, K: int):
        flat  = indices.reshape(-1).detach().cpu()
        hist  = torch.bincount(flat, minlength=K).float()
        total = hist.sum().item()
        if total <= 0:
            return -1.0, 0
        probs = hist / total
        probs = probs[probs > 0]
        ent   = float(-(probs * torch.log2(probs)).sum().item())
        used  = int((hist > 0).sum().item())
        return ent, used

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, 3, T, H, W) in [0, 1]

        Returns dict with keys:
          recon, vq_loss,
          idx_top, idx_mid, idx_bottom,
          z_top_q, z_top_up, z_mid_q, z_mid_up,
          soft_bits_top, soft_bits_mid, soft_bits_bottom,
          entropy_top, entropy_mid, entropy_bottom,
          used_top, used_mid, used_bottom
        """
        # ── Encode ──────────────────────────────────────────────────────────
        h_b = self.enc_bottom(x)    # (B, bottom_dim, T/2, H/4,  W/4 )
        h_m = self.enc_mid(h_b)     # (B, mid_dim,    T/4, H/8,  W/8 )
        h_t = self.enc_top(h_m)     # (B, top_dim,    T/8, H/16, W/16)

        # ── Quantize ────────────────────────────────────────────────────────
        z_t_q, vq_top_loss, idx_t, soft_bits_t = self.vq_top(h_t)
        z_m_q, vq_mid_loss, idx_m, soft_bits_m = self.vq_mid(h_m)
        z_b_q, vq_bot_loss, idx_b, soft_bits_b = self.vq_bottom(h_b)
        vq_loss = vq_top_loss + vq_mid_loss + vq_bot_loss

        # ── Decode (top-down cascade) ────────────────────────────────────────
        # upsample top to mid resolution, concat, decode
        z_t_up = F.interpolate(z_t_q, size=z_m_q.shape[2:], mode="trilinear", align_corners=False)
        d1_in  = torch.cat([z_t_up, z_m_q], dim=1)
        d1_out = self.dec_stage1(d1_in)             # (B, 256, T/4, H/8, W/8)

        # upsample stage1 to bottom resolution, concat, decode
        z_m_up = F.interpolate(d1_out, size=z_b_q.shape[2:], mode="trilinear", align_corners=False)
        d2_in  = torch.cat([z_m_up, z_b_q], dim=1)
        d2_out = self.dec_stage2(d2_in)             # (B, 128, T/2, H/4, W/4)

        # final upsample to pixel space
        recon = self.dec_out(d2_out)                # (B, 3, T, H, W)

        # ── Entropy stats (monitoring only) ─────────────────────────────────
        ent_t, used_t = self._entropy_stats(idx_t, self.K_top)
        ent_m, used_m = self._entropy_stats(idx_m, self.K_mid)
        ent_b, used_b = self._entropy_stats(idx_b, self.K_bottom)

        return {
            "recon":             recon,
            "vq_loss":           vq_loss,
            "idx_top":           idx_t,
            "idx_mid":           idx_m,
            "idx_bottom":        idx_b,
            "z_top_q":           z_t_q,
            "z_top_up":          z_t_up,    # top features at mid resolution
            "z_mid_q":           z_m_q,
            "z_mid_up":          z_m_up,    # stage1 features at bottom resolution (for mid prior conditioning)
            "soft_bits_top":     soft_bits_t,
            "soft_bits_mid":     soft_bits_m,
            "soft_bits_bottom":  soft_bits_b,
            "entropy_top":       ent_t,
            "entropy_mid":       ent_m,
            "entropy_bottom":    ent_b,
            "used_top":          used_t,
            "used_mid":          used_m,
            "used_bottom":       used_b,
        }
