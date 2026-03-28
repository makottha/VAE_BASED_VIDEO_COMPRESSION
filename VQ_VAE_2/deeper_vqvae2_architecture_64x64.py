"""
deeper_vqvae2_architecture_64x64.py

Multi-Scale 3D VQ-VAE-2 for 2-second video clips (32 frames) resized to 64x64 resolution.

- Two-level latent hierarchy with separate quantizers
- Encoder: 3D CNNs with residual blocks
- Decoder: Combines top and bottom embeddings
- Quantization: VectorQuantizer with commitment loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

# ---- Residual Block ----
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

# ---- Vector Quantizer ----
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()
        flat = x_flat.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embeddings.weight).view(x_flat.shape)
        loss = F.mse_loss(quantized.detach(), x_flat) + self.commitment_cost * F.mse_loss(quantized, x_flat.detach())
        quantized = x_flat + (quantized - x_flat).detach()

        return quantized.permute(0, 4, 1, 2, 3).contiguous(), loss

# ---- Multi-Scale VQ-VAE2 ----
class MultiScaleVQVAE2(nn.Module):
    def __init__(self, top_dim=512, bottom_dim=256, top_embed=512, bottom_embed=512):
        super().__init__()

        # ----- Encoder (bottom and top) -----
        self.encoder_bottom = nn.Sequential(
            nn.Conv3d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),
            nn.Conv3d(64, bottom_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(bottom_dim)
        )

        self.encoder_top = nn.Sequential(
            nn.Conv3d(bottom_dim, top_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(top_dim)
        )

        # ----- Quantizers -----
        self.vq_top = VectorQuantizer(top_embed, top_dim)
        self.vq_bottom = VectorQuantizer(bottom_embed, bottom_dim)

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(768, 256, kernel_size=4, stride=2, padding=1),  # 8x16x16 → 16x32x32
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x32x32 → 32x64x64
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # ----- Perceptual Model -----
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.perceptual_model = vgg.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        assert x.shape[-2:] == (64, 64)
        assert x.shape[2] == 32

        z_bottom = self.encoder_bottom(x)
        z_top = self.encoder_top(z_bottom)

        z_top_q, loss_top = self.vq_top(z_top)
        z_bottom_q, loss_bottom = self.vq_bottom(z_bottom)

        z_top_up = F.interpolate(z_top_q, size=z_bottom_q.shape[2:], mode='trilinear', align_corners=False)
        z_combined = torch.cat([z_top_up, z_bottom_q], dim=1)

        recon = self.decoder(z_combined)
        vq_loss = loss_top + loss_bottom
        return recon, vq_loss
