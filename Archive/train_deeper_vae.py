import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from deeper_vae_architecture import DeeperVideoVAE
import os
import time
from torchvision.models import vgg16

# ---- Dataset Loader ----
class VideoClipDataset(Dataset):
    def __init__(self, tensor_dir):
        self.files = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip = torch.load(self.files[idx], weights_only=False)
        if clip.shape[1] != 32:
            raise ValueError(f"Clip length {clip.shape[1]} != 32 for file: {self.files[idx]}")
        return clip

# ---- Perceptual Loss ----
def compute_perceptual_loss(model, x, recon, stride=4):
    B, C, T, H, W = x.shape
    x_sub = x[:, :, ::stride]
    recon_sub = recon[:, :, ::stride]
    x_2d = x_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    recon_2d = recon_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    feat_x = model(x_2d)
    feat_recon = model(recon_2d)
    return F.mse_loss(feat_recon, feat_x)

# ---- VAE Loss ----
def vae_loss(recon, x, mu, logvar, beta=1.0, gamma=0.1, perceptual_model=None):
    recon = recon.clamp(0, 1)
    if recon.shape != x.shape:
        raise ValueError(f"[LOSS] Shape mismatch: recon={recon.shape}, x={x.shape}")
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    perceptual_loss = compute_perceptual_loss(perceptual_model, x, recon) if perceptual_model else torch.tensor(0.0, device=x.device)
    total_loss = recon_loss + beta * kl_div + gamma * perceptual_loss
    return total_loss, recon_loss, kl_div, perceptual_loss

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

    model = DeeperVideoVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[INFO] Model initialized with latent_dim={LATENT_DIM}, beta={BETA}, gamma={GAMMA}")
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

            print(f"[TRAIN][BATCH {i}] Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                  f"KL: {kl_loss.item():.4f} | Perceptual: {perceptual_loss.item():.4f} | "
                  f"Time: {time.time() - batch_start:.2f}s")

        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')

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

                print(f"[VAL][BATCH {j}] Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                      f"KL: {kl_loss.item():.4f} | Perceptual: {perceptual_loss.item():.4f} | "
                      f"Time: {time.time() - batch_start:.2f}s")

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time

        print(f"\n📊 [EPOCH SUMMARY] Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Epoch Time: {epoch_time:.2f}s\n")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/deeper_vae_epoch{epoch:03d}.pt")

    print("✅ Training complete!")

if __name__ == "__main__":
    train()