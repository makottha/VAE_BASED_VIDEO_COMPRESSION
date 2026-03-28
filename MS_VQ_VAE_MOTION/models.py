# models.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, norm="bn"):
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
    """
    Straight-through VQ with optional EMA codebook updates.

    use_ema=False (default): original gradient-based codebook_loss + commit_loss.
                             Matches the original K=1024 training; checkpoints are
                             compatible (no extra buffers in state_dict).

    use_ema=True:  Codebook updated via exponential moving average of assigned
                   encoder outputs — stable even for small K (128/256/512) where
                   gradient-based updates cause collapse. vq_loss contains only the
                   commitment term; the codebook itself is updated outside the graph.
                   Includes dead-code restart: entries with EMA cluster_size below
                   restart_threshold are reinitialised to a random encoder output
                   from the current batch.

    Returns:
      quantized: (B, D, T, H, W)
      vq_loss:   scalar
      indices:   (B, T, H, W) int64
      soft_bits: scalar (differentiable rate proxy w.r.t. encoder output)
    """
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
            # EMA buffers — not parameters, not updated by the optimizer
            self.register_buffer("cluster_size", torch.ones(self.num_embeddings))
            self.register_buffer("embed_avg", self.embeddings.weight.data.clone())

    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        """
        Update codebook in-place using EMA.
        flat:      (N, D) fp32 encoder outputs (detached)
        encodings: (N, K) fp32 one-hot assignments (detached)
        """
        with torch.no_grad():
            # --- update cluster sizes ---
            new_counts = encodings.sum(0)                                   # (K,)
            self.cluster_size.mul_(self.ema_decay).add_(
                new_counts, alpha=1.0 - self.ema_decay)

            # --- update running embed sums ---
            new_sum = encodings.t() @ flat                                  # (K, D)
            self.embed_avg.mul_(self.ema_decay).add_(
                new_sum, alpha=1.0 - self.ema_decay)

            # --- Laplace-smoothed normalisation ---
            n = self.cluster_size.sum()
            smoothed = (
                (self.cluster_size + self.ema_epsilon)
                / (n + self.num_embeddings * self.ema_epsilon)
                * n
            )                                                               # (K,)
            updated = self.embed_avg / smoothed.unsqueeze(1)                # (K, D)

            # --- dead-code restart ---
            dead = self.cluster_size < self.restart_threshold
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                n_available = flat.shape[0]
                n_restart = min(n_dead, n_available)
                perm = torch.randperm(n_available, device=flat.device)[:n_restart]
                restart_vecs = flat[perm].float()
                dead_indices = dead.nonzero(as_tuple=False).squeeze(1)[:n_restart]
                updated[dead_indices] = restart_vecs
                self.cluster_size[dead_indices] = self.restart_threshold
                self.embed_avg[dead_indices] = restart_vecs * self.restart_threshold

            self.embeddings.weight.data.copy_(updated)

    def forward(self, x: torch.Tensor):
        # x: (B, D, T, H, W)
        B, D, T, H, W = x.shape
        x_dtype = x.dtype

        # IMPORTANT: do VQ math in fp32 for stability under AMP
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            x_flat = x_fp32.permute(0, 2, 3, 4, 1).contiguous()  # (B,T,H,W,D)
            flat = x_flat.view(-1, D)                              # (N, D)

            emb = self.embeddings.weight.float()                   # (K, D)

            # squared L2 distance: |z|^2 + |e|^2 - 2 z.e
            dist = (
                torch.sum(flat ** 2, dim=1, keepdim=True)
                + torch.sum(emb ** 2, dim=1).unsqueeze(0)
                - 2.0 * flat @ emb.t()
            )

            indices = torch.argmin(dist, dim=1)                    # (N,)
            encodings = F.one_hot(indices, num_classes=self.num_embeddings).float()  # (N,K)

            quantized = encodings @ emb                            # (N, D)
            quantized = quantized.view(B, T, H, W, D)

            if self.use_ema and self.training:
                # EMA codebook update (no gradient through codebook)
                self._ema_update(flat.detach(), encodings.detach())
                # Only commitment loss — pulls encoder toward (fixed) codebook entries
                vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), x_flat)
            else:
                # Original gradient-based update
                codebook_loss = F.mse_loss(quantized, x_flat.detach())
                commit_loss = F.mse_loss(quantized.detach(), x_flat)
                vq_loss = codebook_loss + self.commitment_cost * commit_loss

            # straight-through estimator
            quantized_st = x_flat + (quantized - x_flat).detach()

            indices = indices.view(B, T, H, W).long()
            quantized_st = quantized_st.permute(0, 4, 1, 2, 3).contiguous()

            # ------------------------------------------------------------------
            # Soft rate proxy — differentiable w.r.t. encoder output (flat/z).
            # ------------------------------------------------------------------
            dist_shifted = dist - dist.min(dim=1, keepdim=True)[0].detach()  # (N,K), ≥0
            soft_assign = torch.softmax(-dist_shifted / self.soft_temp, dim=1)  # (N,K)
            log_soft = torch.log_softmax(-dist_shifted / self.soft_temp, dim=1)  # (N,K)
            soft_bits = -(soft_assign * log_soft).sum(1).sum() / math.log(2)

        # cast back to original dtype for the rest of the network
        return quantized_st.to(x_dtype), vq_loss, indices, soft_bits


class MSVQVAE2Video(nn.Module):
    """
    Two-level VQ-VAE2-like video autoencoder.
    Returns recon + vq_loss + indices_top/bottom + conditioning tensors for priors.
    """
    def __init__(
        self,
        top_dim=256,
        bottom_dim=128,
        K_top=1024,
        K_bottom=1024,
        commit_top=0.25,
        commit_bottom=0.25,
        norm="gn",
        use_vgg=True,
        vgg_layers=16,
        soft_temp=1.0,
        use_ema=False,
    ):
        super().__init__()
        self.top_dim = int(top_dim)
        self.bottom_dim = int(bottom_dim)
        self.K_top = int(K_top)
        self.K_bottom = int(K_bottom)
        self.commit_top = float(commit_top)
        self.commit_bottom = float(commit_bottom)

        # Bottom encoder: /4
        self.enc_bottom = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, norm=norm),
            nn.Conv3d(64, self.bottom_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.bottom_dim, norm=norm),
        )

        # Top encoder: /8 (relative to input)
        self.enc_top = nn.Sequential(
            nn.Conv3d(self.bottom_dim, self.top_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(self.top_dim, norm=norm),
        )

        self.vq_top = VectorQuantizer(
            self.K_top, self.top_dim,
            commitment_cost=self.commit_top,
            soft_temp=float(soft_temp),
            use_ema=bool(use_ema),
        )
        self.vq_bottom = VectorQuantizer(
            self.K_bottom, self.bottom_dim,
            commitment_cost=self.commit_bottom,
            soft_temp=float(soft_temp),
            use_ema=bool(use_ema),
        )

        # Decoder: current design upsamples by /4 from bottom grid.
        Norm = nn.BatchNorm3d if norm == "bn" else lambda c: nn.GroupNorm(32, c)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.top_dim + self.bottom_dim, 256, kernel_size=4, stride=2, padding=1),
            Norm(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            Norm(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Perceptual model (2D VGG) for frame-wise features
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
        # indices: (B,T,H,W)
        flat = indices.reshape(-1).detach().cpu()
        hist = torch.bincount(flat, minlength=K).float()
        total = hist.sum().item()
        if total <= 0:
            return -1.0, 0
        probs = hist / total
        probs = probs[probs > 0]
        ent = float(-(probs * torch.log2(probs)).sum().item())
        used = int((hist > 0).sum().item())
        return ent, used

    def forward(self, x: torch.Tensor):
        """
        x: (B,3,T,H,W) in [0,1]
        returns dict with recon, vq_loss, indices, and conditioning tensors
        """
        z_b = self.enc_bottom(x)             # (B,Db,T/4,H/4,W/4)
        z_t = self.enc_top(z_b)              # (B,Dt,T/8,H/8,W/8)

        z_t_q, vq_top_loss, idx_t, soft_bits_t = self.vq_top(z_t)       # idx_t: (B,Tt,Ht,Wt)
        z_b_q, vq_bot_loss, idx_b, soft_bits_b = self.vq_bottom(z_b)    # idx_b: (B,Tb,Hb,Wb)
        vq_loss = vq_top_loss + vq_bot_loss

        # Condition bottom on upsampled top quantized embeddings
        z_t_up = F.interpolate(z_t_q, size=z_b_q.shape[2:], mode="trilinear", align_corners=False)
        dec_in = torch.cat([z_t_up, z_b_q], dim=1)
        recon = self.decoder(dec_in)

        # Entropy stats for monitoring only (not used in loss)
        ent_t, used_t = self._entropy_stats(idx_t, self.K_top)
        ent_b, used_b = self._entropy_stats(idx_b, self.K_bottom)

        return {
            "recon": recon,
            "vq_loss": vq_loss,
            "idx_top": idx_t,
            "idx_bottom": idx_b,
            "z_top_q": z_t_q,
            "z_top_up": z_t_up,             # conditioning at bottom resolution
            "soft_bits_top": soft_bits_t,   # differentiable rate proxy for top VQ
            "soft_bits_bottom": soft_bits_b, # differentiable rate proxy for bottom VQ
            "entropy_top": ent_t,
            "entropy_bottom": ent_b,
            "used_top": used_t,
            "used_bottom": used_b,
        }
