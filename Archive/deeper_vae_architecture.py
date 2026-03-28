import torch
import torch.nn as nn
from torchvision.models import vgg16

# ---- Residual Block for 3D Convolutions ----
class ResidualBlock3D(nn.Module):
    """Applies two 3D convolutions with skip connection (ResNet-style)."""
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
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

# ---- Deeper VAE Architecture for Video ----
class DeeperVideoVAE(nn.Module):
    """Deeper Variational Autoencoder with residual blocks for 3D video input."""
    def __init__(self, latent_dim=4096):
        super(DeeperVideoVAE, self).__init__()
        self.latent_dim = latent_dim

        # ---- ENCODER ----
        # Input shape: (B, 3, 32, 128, 128)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),  # preserve temporal/spatial resolution
            nn.BatchNorm3d(32),
            nn.ReLU(),
            ResidualBlock3D(32),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # downsample
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ResidualBlock3D(64),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            ResidualBlock3D(128),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            ResidualBlock3D(256),

            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),  # Final downsample
            nn.BatchNorm3d(512),
            nn.ReLU()
        )

        # Output shape: (B, 512, 1, 4, 4)
        self.encoder_out_dim = 512 * 2 * 8 * 8
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        # ---- DECODER ----
        self.decoder_fc = nn.Linear(latent_dim, self.encoder_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # (2, 8, 8)
            nn.BatchNorm3d(256),
            nn.ReLU(),
            ResidualBlock3D(256),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # (4, 16, 16)
            nn.BatchNorm3d(128),
            nn.ReLU(),
            ResidualBlock3D(128),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # (8, 32, 32)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ResidualBlock3D(64),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # (16, 64, 64)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            ResidualBlock3D(32),

            nn.ConvTranspose3d(32, 3, kernel_size=3, stride=1, padding=1),     # (32, 128, 128)
            nn.Sigmoid()  # output in [0, 1]
        )

        # ---- Perceptual Model ----
        # Pretrained VGG16 for perceptual loss (frame-wise)
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
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 512, 2, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon[:, :, :32, :128, :128]
        return recon, mu, logvar