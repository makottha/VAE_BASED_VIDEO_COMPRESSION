import torch
from torch.utils.data import DataLoader
from vae_video import VideoVAE, vae_loss
import os
import time
from torch.utils.data import Dataset

class VideoClipDataset(Dataset):
    def __init__(self, tensor_dir):
        self.files = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip = torch.load(self.files[idx], weights_only=False)
        if clip.shape[1] != 32:  # updated for 16 FPS clips (2s = 32 frames)
            raise ValueError(f"Clip length {clip.shape[1]} != 32 for file: {self.files[idx]}")
        return clip

# ---- CONFIG ----
BATCH_SIZE = 8
EPOCHS = 20
LATENT_DIM = 4096
LR = 1e-4
BETA = 1.0
GAMMA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading datasets...")

    train_dataset = VideoClipDataset("./dataset/train/tensors")
    val_dataset = VideoClipDataset("./dataset/val/tensors")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[INFO] Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # ---- MODEL ----
    model = VideoVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[INFO] Model initialized with latent_dim={LATENT_DIM}, beta={BETA}, gamma={GAMMA}")
    print(f"[INFO] Starting training for {EPOCHS} epochs...\n")

    # ---- TRAINING LOOP ----
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0
        batch_count = 0

        print(f"🔁 [EPOCH {epoch}] Training...")

        for i, batch in enumerate(train_loader, 1):
            batch_start = time.time()
            batch = batch.to(DEVICE)

            print(f"\n[TRAIN][BATCH {i}] Input shape: {batch.shape}")

            recon, mu, logvar = model(batch)

            if recon.shape != batch.shape:
                print(f"[ERROR] Shape mismatch | recon={recon.shape}, input={batch.shape}")
                continue

            loss, recon_loss, kl_loss, perceptual_loss = vae_loss(
                recon, batch, mu, logvar,
                beta=BETA,
                gamma=GAMMA,
                perceptual_model=model.perceptual_model
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            print(f"[TRAIN][BATCH {i}] Loss: {loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | "
                  f"Perceptual: {perceptual_loss.item():.4f} | "
                  f"Time: {time.time() - batch_start:.2f}s")

        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        val_batch_count = 0
        print(f"\n🧪 [EPOCH {epoch}] Validating...")

        with torch.no_grad():
            for j, batch in enumerate(val_loader, 1):
                batch_start = time.time()
                batch = batch.to(DEVICE)

                print(f"[VAL][BATCH {j}] Input shape: {batch.shape}")

                recon, mu, logvar = model(batch)

                if recon.shape != batch.shape:
                    print(f"[ERROR] Shape mismatch (Val) | recon={recon.shape}, input={batch.shape}")
                    continue

                loss, recon_loss, kl_loss, perceptual_loss = vae_loss(
                    recon, batch, mu, logvar,
                    beta=BETA,
                    gamma=GAMMA,
                    perceptual_model=model.perceptual_model
                )

                val_loss += loss.item()
                val_batch_count += 1

                print(f"[VAL][BATCH {j}] Loss: {loss.item():.4f} | "
                      f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | "
                      f"Perceptual: {perceptual_loss.item():.4f} | "
                      f"Time: {time.time() - batch_start:.2f}s")

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time

        print(f"\n📊 [EPOCH SUMMARY] Epoch [{epoch}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Epoch Time: {epoch_time:.2f}s\n")

        # ---- SAVE MODEL ----
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/vae_epoch{epoch:03d}.pt")

    print("✅ Training complete!")

if __name__ == "__main__":
    train()
