# deeper_vqvae_architecture_64x64.py

"""
Deeper 3D Vector-Quantized Variational Autoencoder (VQ-VAE) for 2-second video clips (32 frames) resized to 64x64.

- Encoder: 3D convolutions + residual blocks
- Decoder: Transposed 3D convolutions + residual blocks
- Vector Quantizer: Straight-through estimator
- Perceptual loss module (VGG16) included

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

# ---- Residual Block for 3D Convolutions ----
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
    def __init__(self, num_embeddings=1024, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        # Flatten input
        x_flattened = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
        flat = x_flattened.view(-1, self.embedding_dim)

        # Compute distances
        distances = (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat, self.embeddings.weight.t())
        )

        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(x_flattened.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x_flattened)
        q_latent_loss = F.mse_loss(quantized, x_flattened.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x_flattened + (quantized - x_flattened).detach()

        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)
        return quantized, loss

# ---- Deeper Video VQ-VAE ----
class DeeperVideoVQVAE(nn.Module):
    def __init__(self, latent_dim=4096, num_embeddings=1024):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- ENCODER ----
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            ResidualBlock3D(256)
        )

        # Dynamically calculate encoder output shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 64, 64)
            dummy_out = self.encoder(dummy)
            self.encoder_out_shape = dummy_out.shape[1:]
            self.embedding_dim = self.encoder_out_shape[0]  # 256

        # ---- VQ Layer ----
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=0.25
        )

        # ---- DECODER ----
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),

            nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # ---- Perceptual Model ----
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.perceptual_model = vgg.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        assert x.shape[-2:] == (64, 64), f"[ERROR] Expected input spatial size (64,64), got {x.shape[-2:]}"
        assert x.shape[2] == 32, f"[ERROR] Expected 32 frames, got {x.shape[2]} frames"

        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, vq_loss
