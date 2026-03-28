"""
deeper_vae_architecture_64x64.py

Description:
-------------
Defines a deeper 3D Convolutional Variational Autoencoder (VAE) for video compression research.

- Input: 2-second video clips (32 frames at 16 FPS), resized to 64x64
- Encoder: 3D convolutions + residual blocks
- Decoder: Transposed 3D convolutions + residual blocks
- Perceptual loss: Pretrained VGG16 up to relu3_3

Logs shapes and stages for debugging during development.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16

# ---- Residual Block for 3D Convolutions ----
class ResidualBlock3D(nn.Module):
    """Applies two 3D convolutions with a skip connection (ResNet-style)."""
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
        out = self.block(x)
        out = self.relu(x + out)
        return out

# ---- Deeper Video VAE ----
class DeeperVideoVAE(nn.Module):
    """Deeper Variational Autoencoder (VAE) for 3D video fragments at 64x64 resolution."""
    def __init__(self, latent_dim=4096):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- ENCODER ----
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # (T/2, H/2, W/2)
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

        # Dynamically determine encoder output shape and flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 64, 64)  # (B, C, T, H, W)
            dummy_output = self.encoder(dummy_input)
            self.encoder_out_shape = dummy_output.shape[1:]  # (C, T, H, W)
            self.encoder_out_dim = dummy_output.numel()


        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        # ---- DECODER ----
        self.decoder_fc = nn.Linear(latent_dim, self.encoder_out_dim)

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

        # ---- Perceptual Model (VGG16) ----
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.perceptual_model = vgg.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, *self.encoder_out_shape)
        x = self.decoder(x)
        return x

    def forward(self, x):
        assert x.shape[-2:] == (64, 64), f"[ERROR] Expected input spatial size (64,64), but got {x.shape[-2:]}"
        assert x.shape[2] == 32, f"[ERROR] Expected input temporal size 32 frames, but got {x.shape[2]} frames"

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

