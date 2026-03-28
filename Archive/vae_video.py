import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class VideoVAE(nn.Module):
    def __init__(self, latent_dim=4096):
        super(VideoVAE, self).__init__()
        self.latent_dim = latent_dim

        # ---- Encoder ----
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=(2, 2, 2), padding=1),   # (32, 16, 64, 64)
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),  # (64, 8, 32, 32)
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1), # (128, 4, 16, 16)
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),# (256, 2, 8, 8)
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.encoder_out_dim = 256 * 2 * 8 * 8
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        # ---- Decoder ----
        self.decoder_fc = nn.Linear(latent_dim, self.encoder_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # ---- Perceptual model (subsampled VGG16) ----
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16]  # Up to relu3_3
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
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 256, 2, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---- Optimized Perceptual Loss ----
def compute_perceptual_loss(model, x, recon, stride=4):
    B, C, T, H, W = x.shape
    # Subsample frames for VGG16 (fewer frames = faster)
    x_sub = x[:, :, ::stride]     # shape: (B, C, T/stride, H, W)
    recon_sub = recon[:, :, ::stride]

    x_2d = x_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    recon_2d = recon_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

    # Feed through VGG16
    feat_x = model(x_2d)
    feat_recon = model(recon_2d)

    return F.mse_loss(feat_recon, feat_x)


# ---- Full VAE Loss ----
def vae_loss(recon, x, mu, logvar, beta=1.0, gamma=0.1, perceptual_model=None):
    recon = recon.clamp(0, 1)

    if recon.shape != x.shape:
        raise ValueError(f"[LOSS] Shape mismatch: recon={recon.shape}, x={x.shape}")

    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    if perceptual_model:
        perceptual_loss = compute_perceptual_loss(perceptual_model, x, recon)
    else:
        perceptual_loss = torch.tensor(0.0, device=x.device)

    total_loss = recon_loss + beta * kl_div + gamma * perceptual_loss
    return total_loss, recon_loss, kl_div, perceptual_loss
