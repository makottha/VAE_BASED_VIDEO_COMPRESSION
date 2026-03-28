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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.025):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.normal_(0, 1.0 / self.num_embeddings)


    def forward(self, x):
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, T, H, W) -> (B, T, H, W, D)
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

        indices = encoding_indices.view(x.shape[0], *x.shape[2:])
        return quantized.permute(0, 4, 1, 2, 3).contiguous(), loss, indices


# ---- Entropy Estimation ----
def estimate_entropy(indices, num_embeddings):
    try:
        # Ensure tensor is detached, on CPU, and integer type
        indices_flat = indices.view(-1).detach().to('cpu')

        if not torch.is_floating_point(indices_flat):
            indices_flat = indices_flat.long()
        else:
            print("⚠️ Warning: indices are floating point, skipping entropy.")
            return -1.0, 0

        hist = torch.bincount(indices_flat, minlength=num_embeddings).float()
        total = hist.sum().item()

        if total == 0:
            print("⚠️ Warning: histogram sum is 0, skipping entropy.")
            return -1.0, 0

        probs = hist / total
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        used = torch.sum(hist > 0).item()
        return entropy, used

    except Exception as e:
        print(f"🚨 Entropy calculation error: {e}")
        return -1.0, 0


# ---- Updated Multi-Scale VQ-VAE2 ----
class MultiScaleVQVAE2(nn.Module):
    def __init__(self, top_dim=512, bottom_dim=256, top_embed=128, bottom_embed=512, beta=0.01):
        super().__init__()
        self.beta = beta

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

        self.vq_top = VectorQuantizer(top_embed, top_dim, commitment_cost=0.01)
        self.vq_bottom = VectorQuantizer(bottom_embed, bottom_dim, commitment_cost=0.05)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(top_dim + bottom_dim, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 3, 3, padding=1),
            nn.Sigmoid()
        )

        vgg = vgg16(weights="IMAGENET1K_V1").features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.perceptual_model = vgg.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        z_bottom = self.encoder_bottom(x)
        z_top = self.encoder_top(z_bottom)

        z_top_q, loss_top, indices_top = self.vq_top(z_top)
        z_bottom_q, loss_bottom, indices_bottom = self.vq_bottom(z_bottom)

        z_top_up = F.interpolate(z_top_q, size=z_bottom_q.shape[2:], mode='trilinear', align_corners=False)
        z_combined = torch.cat([z_top_up, z_bottom_q], dim=1)

        recon = self.decoder(z_combined)
        vq_loss = loss_top + loss_bottom

        entropy_top, used_top = estimate_entropy(indices_top, self.vq_top.num_embeddings)
        entropy_bottom, used_bottom = estimate_entropy(indices_bottom, self.vq_bottom.num_embeddings)
        avg_entropy = (entropy_top + entropy_bottom) / 2.0 if entropy_top > 0 and entropy_bottom > 0 else -1.0

        if entropy_top > 0 and entropy_bottom > 0:
            entropy_loss = -avg_entropy * self.beta
        else:
            entropy_loss = 0.0

        total_loss = vq_loss + entropy_loss

        # Debug
        print(f"[ENTROPY] Avg: {avg_entropy:.4f} | Used Codes - Top: {used_top}, Bottom: {used_bottom}")
        # print(f"Top indices histogram: {torch.bincount(indices_top.view(-1).cpu())}")

        return recon, total_loss, avg_entropy, entropy_top, entropy_bottom, used_top, used_bottom



