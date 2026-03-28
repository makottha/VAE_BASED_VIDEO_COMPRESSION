# models/ms_vqvae2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

# -----------------------
# Norm / Blocks
# -----------------------

def GN(ch: int) -> nn.GroupNorm:
    groups = 32 if ch >= 32 else max(1, ch // 2)
    return nn.GroupNorm(groups, ch)

class ResidualBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            GN(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            GN(ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class UpBlock3D(nn.Module):
    """Deconv upsample + conv refine"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = GN(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.refine = ResidualBlock3D(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.bn(x))
        x = self.refine(x)
        return x

# -----------------------
# Vector Quantizer (gather)
# -----------------------

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B,C,T,H,W)
        B, C, T, H, W = z.shape
        flat = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # (N,C)
        E = self.embeddings.weight  # (K,C)

        d = (flat.pow(2).sum(1, keepdim=True)
             + E.pow(2).sum(1)[None, :]
             - 2 * flat @ E.t())                          # (N,K)

        inds = d.argmin(dim=1)                             # (N,)
        z_q = E.index_select(0, inds).view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        codebook_loss = F.mse_loss(z_q.detach(), z)
        commit_loss   = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through
        z_q = z + (z_q - z).detach()
        inds = inds.view(B, T, H, W)
        return z_q, vq_loss, inds, d  # return distances for optional soft-entropy

# -----------------------
# Multi-Scale VQ-VAE-2 (64x64, 32 frames)
# -----------------------

class MultiScaleVQVAE2(nn.Module):
    """
    - Bottom latent: (B, bottom_dim, 8, 16, 16)
    - Top latent   : (B, top_dim,    4,  8,  8)
    - One skip from bottom encoder feature at (16,32,32)
    - Top->Bottom fused via 1x1x1 conv
    """
    def __init__(
        self,
        top_dim=512,
        bottom_dim=256,
        top_embed=256,
        bottom_embed=1024,
        soft_entropy_beta=0.0,   # set >0 to enable soft entropy reg
        soft_entropy_tau=1.0
    ):
        super().__init__()
        self.soft_entropy_beta = soft_entropy_beta
        self.soft_entropy_tau = soft_entropy_tau

        # ---- Bottom encoder
        self.enc_b_0 = nn.Sequential(           # 32x64x64 -> 16x32x32
            nn.Conv3d(3, 64, 4, 2, 1),
            GN(64), nn.ReLU(inplace=True),
            ResidualBlock3D(64)
        )
        self.enc_b_1 = nn.Sequential(           # 16x32x32 -> 8x16x16
            nn.Conv3d(64, bottom_dim, 4, 2, 1),
            GN(bottom_dim), nn.ReLU(inplace=True),
            ResidualBlock3D(bottom_dim)
        )

        # ---- Top encoder (8x16x16 -> 4x8x8)
        self.enc_t = nn.Sequential(
            nn.Conv3d(bottom_dim, top_dim, 4, 2, 1),
            GN(top_dim), nn.ReLU(inplace=True),
            ResidualBlock3D(top_dim)
        )

        # ---- Quantizers
        self.vq_top = VectorQuantizer(top_embed, top_dim, commitment_cost=0.25)
        self.vq_bot = VectorQuantizer(bottom_embed, bottom_dim, commitment_cost=0.25)

        # ---- Fuse top→bottom
        self.fuse = nn.Conv3d(top_dim + bottom_dim, bottom_dim, kernel_size=1)

        # ---- Decoder with one U-Net style skip from enc_b_0 (16x32x32)
        self.dec_up1 = UpBlock3D(bottom_dim, 256)      # 8x16x16 -> 16x32x32
        self.dec_refine1 = nn.Sequential(              # concat skip (64) -> 256+64
            nn.Conv3d(256 + 64, 256, 3, padding=1),
            GN(256), nn.ReLU(inplace=True),
            ResidualBlock3D(256)
        )

        self.dec_up2 = UpBlock3D(256, 128)             # 16x32x32 -> 32x64x64
        self.dec_out = nn.Sequential(
            nn.Conv3d(128, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # ---- Perceptual model (VGG relu3_3)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.perceptual_model = vgg.eval()  # move to device externally

    @staticmethod
    def estimate_entropy(indices, K: int):
        # hard-index monitoring (no grad)
        flat = indices.detach().reshape(-1)
        hist = torch.bincount(flat, minlength=K).float()
        p = hist / (hist.sum() + 1e-8)
        p = p[p > 0]
        H = -(p * torch.log2(p)).sum().item()
        used = int((hist > 0).sum().item())
        return H, used

    def forward(self, x):
        # x: (B,3,32,64,64)
        f16 = self.enc_b_0(x)          # (B,64,16,32,32)  skip
        z_b = self.enc_b_1(f16)        # (B,bottom_dim,8,16,16)
        z_t = self.enc_t(z_b)          # (B,top_dim,4,8,8)

        z_tq, vq_t, inds_t, dist_t = self.vq_top(z_t)
        z_bq, vq_b, inds_b, dist_b = self.vq_bot(z_b)

        z_t_up = F.interpolate(z_tq, size=z_bq.shape[2:], mode='trilinear', align_corners=False)
        z_comb = self.fuse(torch.cat([z_t_up, z_bq], dim=1))  # -> bottom_dim

        y = self.dec_up1(z_comb)                 # 16x32x32
        y = torch.cat([y, f16], dim=1)          # add skip
        y = self.dec_refine1(y)
        y = self.dec_up2(y)                      # 32x64x64
        recon = self.dec_out(y)

        vq_loss = vq_t + vq_b

        # Optional: soft entropy reg (grad-carrying)
        entropy_loss = torch.tensor(0., device=x.device)
        if self.soft_entropy_beta > 0:
            tau = self.soft_entropy_tau
            p_top = F.softmax(-dist_t / tau, dim=1).mean(0)  # (K_t,)
            p_bot = F.softmax(-dist_b / tau, dim=1).mean(0)  # (K_b,)
            H_top = -(p_top * (p_top + 1e-12).log()).sum()
            H_bot = -(p_bot * (p_bot + 1e-12).log()).sum()
            soft_H = 0.5 * (H_top + H_bot)
            entropy_loss = -self.soft_entropy_beta * soft_H    # encourage spread

        # Hard-entropy monitoring
        Ht, used_t = self.estimate_entropy(inds_t, self.vq_top.num_embeddings)
        Hb, used_b = self.estimate_entropy(inds_b, self.vq_bot.num_embeddings)
        H_avg = (Ht + Hb) / 2.0

        return recon, vq_loss + entropy_loss, {
            "hard_entropy_avg": H_avg,
            "hard_entropy_top": Ht,
            "hard_entropy_bot": Hb,
            "used_top": used_t,
            "used_bot": used_b
        }
