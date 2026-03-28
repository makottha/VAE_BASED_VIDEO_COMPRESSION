# train_deeper_vqvae_64x64.py

"""
Training script for deeper 3D Vector-Quantized Variational Autoencoder (VQ-VAE)
on 2-second video clips (32 frames) resized to 64x64 resolution.

- Encoder: Deep 3D CNN with residual blocks
- Decoder: Deep 3D Transposed CNN with residual blocks
- Loss: MSE + VQ loss + (optional) Perceptual Loss using VGG16
- Mixed precision (AMP) enabled for faster training on GPU

Training and validation data expected in `.pt` files with shape: (3, 32, 64, 64).
"""

import os
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from VQ_VAE.deeper_vqvae_architecture_64x64 import DeeperVideoVQVAE  # <-- Use your new VQ-VAE model

# ---- Dataset Loader ----
class VideoClipDataset(Dataset):
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

# ---- VQ-VAE Loss ----
def vqvae_loss(recon, x, vq_loss, gamma=0.2, perceptual_model=None):
    recon = recon.clamp(0, 1)
    if recon.shape != x.shape:
        raise ValueError(f"[LOSS] Shape mismatch: recon={recon.shape}, x={x.shape}")
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    perceptual_loss = compute_perceptual_loss(perceptual_model, x, recon) if (gamma > 0 and perceptual_model) else torch.tensor(0.0, device=x.device)
    total_loss = recon_loss + vq_loss + gamma * perceptual_loss
    return total_loss, recon_loss, vq_loss, perceptual_loss

# ---- Perceptual Loss ----
def compute_perceptual_loss(model, x, recon, stride=4):
    B, C, T, H, W = x.shape
    x_sub = x[:, :, ::stride]
    recon_sub = recon[:, :, ::stride]
    x_2d = x_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    recon_2d = recon_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    with torch.no_grad():
        feat_x = model(x_2d)
        feat_recon = model(recon_2d)
    return F.mse_loss(feat_recon, feat_x)

# ---- Training Config ----
BATCH_SIZE = 20
EPOCHS = 50
LATENT_DIM = 4096
LR = 1e-4
GAMMA = 0.4  # Perceptual loss weight
USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Training Loop ----
def train():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading datasets...")

    train_dataset = VideoClipDataset("../dataset_64/train/tensors", subset_fraction=0.5)
    val_dataset = VideoClipDataset("../dataset_64/val/tensors", subset_fraction=0.5)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[INFO] Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    model = DeeperVideoVQVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=USE_AMP)

    print(f"[INFO] Model initialized | latent_dim={LATENT_DIM} | gamma={GAMMA} | AMP={USE_AMP}")
    print(f"[INFO] Starting training for {EPOCHS} epochs...\n")

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
                recon, vq_loss = model(batch)
                loss, recon_loss, vq_loss_val, perceptual_loss = vqvae_loss(
                    recon, batch, vq_loss, gamma=GAMMA, perceptual_model=model.perceptual_model
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            batch_count += 1

            print(f"[TRAIN][BATCH {i}/{len(train_loader)}] Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                  f"VQ: {vq_loss_val.item():.4f} | Perceptual: {perceptual_loss.item():.4f} | "
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
                    recon, vq_loss = model(batch)
                    loss, recon_loss, vq_loss_val, perceptual_loss = vqvae_loss(
                        recon, batch, vq_loss, gamma=GAMMA, perceptual_model=model.perceptual_model
                    )
                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time

        # ---- Save Checkpoint ----
        os.makedirs("../checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/deeper_vqvae_epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint to {ckpt_path}")

        print(f"\n📊 [EPOCH SUMMARY] Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Epoch Time: {epoch_time:.2f}s\n")

    print("✅ Training complete!")

# ---- Entrypoint ----
if __name__ == "__main__":
    train()
