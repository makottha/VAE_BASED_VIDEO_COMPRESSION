"""
train_deeper_vae_64x64.py

Author: [Your Name]
Created: [Date]

Description:
-------------
Training script for deeper 3D Variational Autoencoder (VAE) on 2-second video clips
(32 frames) resized to 64x64 resolution.

- Encoder: Deep 3D CNN with residual blocks
- Decoder: Deep 3D Transposed CNN with residual blocks
- Loss: MSE + KL Divergence + (optional) Perceptual Loss using VGG16
- Mixed precision (AMP) enabled for faster training on GPU

Training and validation data expected in `.pt` files with shape: (3, 32, 64, 64).

"""

import os
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16
from torch.amp import autocast, GradScaler
from deeper_vae_architecture_64x64 import DeeperVideoVAE

class VideoClipDataset(Dataset):
    """
    PyTorch Dataset for loading video clip tensors saved as .pt files.

    Args:
        tensor_dir (str): Directory containing saved clip tensors.
        subset_fraction (float): Fraction of data to use (e.g., 1.0 = 100%, 0.5 = 50%, 0.25 = 25%).
        shuffle (bool): Whether to shuffle the files before selecting subset.
    """

    def __init__(self, tensor_dir, subset_fraction=1.0, shuffle=True):
        all_files = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        if shuffle:
            random.shuffle(all_files)

        subset_size = int(len(all_files) * subset_fraction)
        self.files = sorted(all_files[:subset_size])

        print(f"[DATASET] Loaded {len(self.files)} samples from {tensor_dir} (Subset Fraction={subset_fraction})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip = torch.load(self.files[idx], weights_only=False)
        if clip.shape[1] != 32:
            raise ValueError(f"Clip length {clip.shape[1]} != 32 frames for file: {self.files[idx]}")
        return clip


# ---- Perceptual Loss ----
def compute_perceptual_loss(model, x, recon, stride=4):
    """Computes perceptual loss using VGG16 on sampled frames."""
    B, C, T, H, W = x.shape
    x_sub = x[:, :, ::stride]
    recon_sub = recon[:, :, ::stride]
    x_2d = x_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    recon_2d = recon_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    with torch.no_grad():
        feat_x = model(x_2d)
        feat_recon = model(recon_2d)
    return F.mse_loss(feat_recon, feat_x)

# ---- Full VAE Loss ----
def vae_loss(recon, x, mu, logvar, beta=1.0, gamma=0.1, perceptual_model=None):
    """Computes total VAE loss (MSE + KL + optional perceptual loss)."""
    recon = recon.clamp(0, 1)
    if recon.shape != x.shape:
        raise ValueError(f"[LOSS] Shape mismatch: recon={recon.shape}, x={x.shape}")
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    perceptual_loss = compute_perceptual_loss(perceptual_model, x, recon) if (gamma > 0 and perceptual_model) else torch.tensor(0.0, device=x.device)
    total_loss = recon_loss + beta * kl_div + gamma * perceptual_loss
    return total_loss, recon_loss, kl_div, perceptual_loss

# ---- Training Config ----
BATCH_SIZE = 20
EPOCHS = 5
LATENT_DIM = 4096
LR = 1e-4
BETA = 1.0
GAMMA = 0.2    # Perceptual loss disabled by default
USE_AMP = True # Mixed precision training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Main Training Function ----
def train():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading datasets...")

    train_dataset = VideoClipDataset("./dataset_64/train/tensors", subset_fraction=0.5)
    val_dataset = VideoClipDataset("./dataset_64/val/tensors", subset_fraction=0.5)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[INFO] Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    model = DeeperVideoVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=USE_AMP)

    print(f"[INFO] Model initialized | latent_dim={LATENT_DIM} | beta={BETA} | gamma={GAMMA} | AMP={USE_AMP}")
    print(f"[INFO] Starting training for {EPOCHS} epochs...\n")

    avg_val_loss = -1
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0
        batch_count = 0

        print(f"🔁 [EPOCH {epoch}] Training...")

        for i, batch in enumerate(train_loader, 1):
            batch_start = time.time()
            batch = batch.to(DEVICE)

            with autocast(device_type="cuda", enabled=USE_AMP):
                recon, mu, logvar = model(batch)
                if recon.shape != batch.shape:
                    print(f"[ERROR] Shape mismatch | recon={recon.shape}, input={batch.shape}")
                    continue
                loss, recon_loss, kl_loss, perceptual_loss = vae_loss(
                    recon, batch, mu, logvar,
                    beta=BETA, gamma=GAMMA,
                    perceptual_model=model.perceptual_model
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            batch_count += 1

            print(f"[TRAIN][BATCH {i}] Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                  f"KL: {kl_loss.item():.4f} | Perceptual: {perceptual_loss.item():.4f} | "
                  f"Time: {time.time() - batch_start:.2f}s")

        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')

        # ---- Validation Phase ----

        model.eval()
        val_loss = 0
        val_batch_count = 0
        print(f"\n🧪 [EPOCH {epoch}] Validating...")

        with torch.no_grad():
            for j, batch in enumerate(val_loader, 1):
                batch = batch.to(DEVICE)
                with autocast(device_type="cuda", enabled=USE_AMP):
                    recon, mu, logvar = model(batch)
                    if recon.shape != batch.shape:
                        print(f"[ERROR] Shape mismatch (Val) | recon={recon.shape}, input={batch.shape}")
                        continue
                    loss, recon_loss, kl_loss, perceptual_loss = vae_loss(
                        recon, batch, mu, logvar,
                        beta=BETA, gamma=GAMMA,
                        perceptual_model=model.perceptual_model
                    )

                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time

        # ---- Save Checkpoint ----
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/deeper_vae_epoch{epoch:03d}.pt")

        print(f"\n📊 [EPOCH SUMMARY] Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss if not torch.isnan(torch.tensor(avg_val_loss)) else 'Skipped'} | "
              f"Epoch Time: {epoch_time:.2f}s\n")

    print("✅ Training complete!")

# ---- Main Entrypoint ----
if __name__ == "__main__":
    train()
