# train_multiscale_vqvae_64x64.py

import os
import time
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from VQ_VAE_2.deeper_vqvae2_architecture_64x64 import MultiScaleVQVAE2
from Utils.logger import TrainingLogger  # Or use inline if not modularized


# Dataset
class VideoClipDataset(Dataset):
    def __init__(self, tensor_dir, subset_fraction=1.0, shuffle=True):
        all_files = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        if shuffle:
            random.shuffle(all_files)
        subset_size = int(len(all_files) * subset_fraction)
        self.files = sorted(all_files[:subset_size])
        print(f"[DATASET] Loaded {len(self.files)} samples from {tensor_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip = torch.load(self.files[idx], weights_only=True)
        if clip.shape[1] != 32:
            raise ValueError(f"Expected 32 frames, got {clip.shape[1]} in {self.files[idx]}")
        return clip

# Perceptual Loss
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

# Loss
def vqvae2_loss(recon, x, vq_loss, gamma=0.4, perceptual_model=None):
    recon = recon.clamp(0, 1)
    recon_loss = F.mse_loss(recon, x)
    perceptual_loss = compute_perceptual_loss(perceptual_model, x, recon) if gamma > 0 else 0.0
    total_loss = recon_loss + vq_loss + gamma * perceptual_loss
    return total_loss, recon_loss, vq_loss, perceptual_loss

# Config
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
GAMMA = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True
logger = TrainingLogger()

def train():
    print(f"[INFO] Using {DEVICE}")
    train_dataset = VideoClipDataset("../dataset_64/train/tensors", subset_fraction=0.5)
    val_dataset = VideoClipDataset("../dataset_64/val/tensors", subset_fraction=0.5)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultiScaleVQVAE2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=USE_AMP)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        total_loss = 0
        print(f"🔁 [Epoch {epoch}] Training...")

        for i, batch in enumerate(train_loader, 1):
            batch = batch.to(DEVICE)
            with autocast(device_type="cuda", enabled=USE_AMP):
                recon, vq_loss = model(batch)
                loss, recon_l, vq_l, perc_l = vqvae2_loss(recon, batch, vq_loss, gamma=GAMMA, perceptual_model=model.perceptual_model)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            logger.log_train(epoch, i, loss.item(), recon_l.item(), vq_l.item(),
                             perc_l.item() if hasattr(perc_l, 'item') else perc_l)
            print(f"[TRAIN][BATCH {i}] Loss: {loss.item():.4f} | Recon: {recon_l.item():.4f} | VQ: {vq_l.item():.4f} | Perceptual: {perc_l:.4f}")


        print(f"📊 Epoch {epoch} Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader, 1):
                batch = batch.to(DEVICE)
                with autocast(device_type="cuda", enabled=USE_AMP):
                    recon, vq_loss = model(batch)
                    loss, recon_l, vq_l, perc_l = vqvae2_loss(recon, batch, vq_loss, gamma=GAMMA, perceptual_model=model.perceptual_model)
                val_loss += loss.item()
                logger.log_val(epoch, j, loss.item(), recon_l.item(), vq_l.item(),
                               perc_l.item() if hasattr(perc_l, 'item') else perc_l)
        print(f"🧪 Validation Loss: {val_loss / len(val_loader):.4f}")

        torch.save(model.state_dict(), f"checkpoints/vqvae2_epoch{epoch:03d}.pt")
        print(f"💾 Saved checkpoint to checkpoints/vqvae2_epoch{epoch:03d}.pt | Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    os.makedirs("../checkpoints", exist_ok=True)
    train()
    logger.close()
