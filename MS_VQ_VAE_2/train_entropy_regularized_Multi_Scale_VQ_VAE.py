# train_vqvae2.py
import os, time, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pytorch_msssim import ms_ssim, ssim
from MS_VQ_VAE_2.Entropy_Regularized_Multi_Scale_VQ_VAE import MultiScaleVQVAE2
import torch.nn.functional as F

# Optional external logger; if not present, we no-op
try:
    from Utils.logger import TrainingLogger
    LOGGER_AVAILABLE = True
except Exception:
    LOGGER_AVAILABLE = False

# -----------------------
# Dataset
# -----------------------

class VideoClipDataset(Dataset):
    def __init__(self, tensor_dir, subset_fraction=1.0, shuffle=True):
        files = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        if shuffle:
            random.shuffle(files)
        subset_size = int(len(files) * subset_fraction)
        self.files = files[:subset_size]
        print(f"[DATASET] Loaded {len(self.files)} samples from {tensor_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        clip = torch.load(self.files[idx])  # expect (3,T,64,64), float in [0,1]
        if clip.ndim != 4 or clip.shape[0] != 3:
            raise ValueError(f"Expect (3,T,64,64), got {tuple(clip.shape)} in {self.files[idx]}")
        if clip.shape[1] != 32:
            raise ValueError(f"Expected 32 frames, got {clip.shape[1]} in {self.files[idx]}")
        return clip.float().clamp(0, 1)

# -----------------------
# Losses
# -----------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def _norm_imagenet(x):
    # x: (N,3,H,W) in [0,1]
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std

def perceptual_loss(perceptual_model, x, y, stride=4):
    # x,y: (B,3,T,H,W) in [0,1]; grads through y branch only
    B, C, T, H, W = x.shape
    x_sub, y_sub = x[:, :, ::stride], y[:, :, ::stride]
    x2d = x_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    y2d = y_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    x2d = _norm_imagenet(x2d).detach()
    y2d = _norm_imagenet(y2d)
    with torch.no_grad():
        fx = perceptual_model(x2d)
    fy = perceptual_model(y2d)
    return F.l1_loss(fy, fx)



def _make_ms_weights(n_scales: int, device):
    # simple normalized weights; feel free to tweak
    base = torch.tensor([0.4, 0.3, 0.2, 0.07, 0.03], device=device)
    w = base[:n_scales]
    return w / w.sum()

def _max_scales_for_size(H: int, W: int, win_size: int) -> int:
    # Need: min(H,W) > (win_size-1) * 2**(L-1)
    m = min(H, W)
    for L in (5, 4, 3, 2, 1):
        need = (win_size - 1) * (2 ** (L - 1))
        if m > need:
            return L
    return 1


def _ms_ssim_safe(y2d, x2d, win_size=7):
    """
    Try true MS-SSIM with 3 scales (safe for 64x64). If the package still asserts,
    fall back to a manual 3-scale approximation using single-scale SSIM on downsampled frames.
    Inputs: (N,3,H,W) in [0,1]
    Returns: (1 - MS-SSIM) in [0,1] as a loss-like term
    """
    # preferred: 3 scales with a Python-list weights (not tensor)
    weights = [0.4, 0.3, 0.3]  # 3 levels -> 2 downsamples
    try:
        mss = 1.0 - ms_ssim(
            y2d, x2d,
            data_range=1.0,
            size_average=True,
            win_size=win_size,
            weights=weights,   # length controls #scales in many versions
        )
        return mss
    except AssertionError:
        # Manual 3-scale proxy: SSIM at original, /2, /4; weighted sum of (1-SSIM)
        def _ssim_loss(a, b):
            return 1.0 - ssim(a, b, data_range=1.0, size_average=True, win_size=win_size)

        a0, b0 = y2d, x2d
        a1 = F.avg_pool2d(a0, kernel_size=2, stride=2)
        b1 = F.avg_pool2d(x2d, kernel_size=2, stride=2)
        a2 = F.avg_pool2d(a1, kernel_size=2, stride=2)
        b2 = F.avg_pool2d(b1, kernel_size=2, stride=2)

        l0 = _ssim_loss(a0, b0)
        l1 = _ssim_loss(a1, b1)
        l2 = _ssim_loss(a2, b2)

        # normalize weights to sum to 1 just in case
        ws = torch.tensor(weights, device=y2d.device, dtype=torch.float32)
        ws = ws / ws.sum()

        # weighted sum of SSIM losses ≈ MS-SSIM loss
        return ws[0]*l0 + ws[1]*l1 + ws[2]*l2

def recon_loss(x, y, alpha=0.6, win_size=7):
    """
    Mix (MS-)SSIM with L1, robust for (B,3,T,64,64) inputs.
    Returns a scalar loss.
    """
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    # L1 term
    l1 = F.l1_loss(y, x)

    # Flatten time to frames for SSIM/MS-SSIM
    B, C, T, H, W = x.shape
    x2d = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
    y2d = y.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

    # SSIM/MS-SSIM term (safe)
    mss = _ms_ssim_safe(y2d, x2d, win_size=win_size)

    # Blend (lower alpha → more PSNR; higher alpha → more structural similarity/sharpness)
    return alpha * mss + (1 - alpha) * l1


def temporal_loss(x, y):
    # encourage temporal consistency
    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = y[:, :, 1:] - y[:, :, :-1]
    return F.l1_loss(dy, dx)

def psnr_batch(x, y, eps=1e-8):
    x = x.clamp(0,1); y = y.clamp(0,1)
    mse = F.mse_loss(y, x)
    return 10 * torch.log10(1.0 / (mse + eps))

# -----------------------
# EMA weights
# -----------------------

class ModelEMA:
    def __init__(self, model, decay=0.999):
        # clone architecture and weights
        ema = MultiScaleVQVAE2(
            top_dim=model.enc_t[0].out_channels,
            bottom_dim=model.enc_b_1[0].out_channels,
            top_embed=model.vq_top.num_embeddings,
            bottom_embed=model.vq_bot.num_embeddings,
            soft_entropy_beta=model.soft_entropy_beta,
            soft_entropy_tau=model.soft_entropy_tau
        )
        ema.load_state_dict(model.state_dict())

        # Optional: don’t carry a second VGG (we only use EMA for eval recon)
        ema.perceptual_model = nn.Identity()

        # freeze & move to same device
        for p in ema.parameters():
            p.requires_grad = False
        ema.to(next(model.parameters()).device)

        self.ema = ema
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            # both on same device now; in-place EMA update
            ema_p.mul_(d).add_(p, alpha=1.0 - d)


# -----------------------
# Train
# -----------------------

def set_seed(s=123):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def train():
    set_seed(123)

    # --- Config ---
    TRAIN_DIR = "../dataset_64/train/tensors"
    VAL_DIR   = "../dataset_64/val/tensors"
    CKPT_DIR  = "../checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)

    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True
    PRINT_EVERY = 50

    # loss weights
    ALPHA_MSSSIM = 0.6
    W_PERCEPT = 0.1
    W_TEMP = 0.05

    print(f"[INFO] Device: {DEVICE}")
    if LOGGER_AVAILABLE:
        logger = TrainingLogger()
    else:
        logger = None

    train_ds = VideoClipDataset(TRAIN_DIR, subset_fraction=1.0, shuffle=True)
    val_ds   = VideoClipDataset(VAL_DIR,   subset_fraction=1.0, shuffle=False)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=pin, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin, persistent_workers=True)

    model = MultiScaleVQVAE2(
        top_dim=512, bottom_dim=256,
        top_embed=256, bottom_embed=1024,
        soft_entropy_beta=5e-3, soft_entropy_tau=1.0  # enable soft-entropy gently
    ).to(DEVICE)
    model.perceptual_model.to(DEVICE).eval()  # VGG on device

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler(enabled=USE_AMP)

    # cosine with warmup
    def lr_lambda(it, warm=5 * len(train_loader), total=EPOCHS * len(train_loader)):
        if it < warm:
            return max(1e-3, (it + 1) / max(1, warm))
        progress = (it - warm) / max(1, total - warm)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: 1.0)  # init; we’ll step manually

    ema = ModelEMA(model, decay=0.999)
    autocast_device = "cuda" if DEVICE.type == "cuda" else "cpu"

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        sum_loss = 0.0

        for i, batch in enumerate(train_loader, 1):
            batch = batch.to(DEVICE, non_blocking=True)
            with autocast(device_type=autocast_device, enabled=USE_AMP):
                recon, vq_term, stats = model(batch)

                # ramp perceptual weight early
                perc_w = min(W_PERCEPT, epoch * (W_PERCEPT / 10.0))

                l_rec  = recon_loss(batch, recon, alpha=ALPHA_MSSSIM)
                l_perc = perceptual_loss(model.perceptual_model, batch, recon) if perc_w > 0 else torch.tensor(0.0, device=DEVICE)
                l_temp = temporal_loss(batch, recon) * W_TEMP
                loss   = l_rec + vq_term + perc_w * l_perc + l_temp

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            # cosine schedule (manual step per iter)
            scheduler.base_lrs = [LR]
            for pg in optimizer.param_groups:
                pg['lr'] = LR * lr_lambda(global_step)
            global_step += 1

            sum_loss += float(loss)

            if (i % PRINT_EVERY == 0) or (i == 1):
                p = float(psnr_batch(batch, recon))
                print(f"[E{epoch:02d} B{i:04d}] "
                      f"loss={float(loss):.4f} rec={float(l_rec):.4f} vq={float(vq_term):.4f} "
                      f"perc={float(l_perc):.4f} temp={float(l_temp):.4f} "
                      f"PSNR={p:.2f}dB | H(avg)={stats['hard_entropy_avg']:.2f} "
                      f"used T/B={stats['used_top']}/{stats['used_bot']}")

                if logger:
                    logger.log_train(epoch, i, float(loss), float(l_rec), float(vq_term), float(l_perc))

        train_avg = sum_loss / max(1, len(train_loader))
        print(f"📊 Epoch {epoch} Train Loss: {train_avg:.4f} | time {time.time() - t0:.1f}s")

        # -------- Validation (EMA weights) --------
        model.eval()
        psnr_sum, mss_sum, cnt = 0.0, 0.0, 0
        with torch.no_grad(), autocast(device_type=autocast_device, enabled=USE_AMP):
            ema_model = ema.ema.to(DEVICE)
            for j, batch in enumerate(val_loader, 1):
                batch = batch.to(DEVICE, non_blocking=True)
                recon, _, _ = ema_model(batch)
                psnr_sum += float(psnr_batch(batch, recon))
                mss_sum  += float(ms_ssim(recon.clamp(0,1), batch.clamp(0,1), data_range=1.0, size_average=True))
                cnt += 1

        psnr_val = psnr_sum / max(1, cnt)
        mss_val  = mss_sum  / max(1, cnt)
        print(f"🧪 Val (EMA) PSNR={psnr_val:.2f}dB | MS-SSIM={mss_val:.4f}")

        # Save checkpoint (EMA weights for deployment)
        ckpt_path = os.path.join(CKPT_DIR, f"msvqvae2_ema_epoch{epoch:03d}.pt")
        torch.save(ema.ema.state_dict(), ckpt_path)
        print(f"💾 Saved {ckpt_path}")

    if LOGGER_AVAILABLE:
        logger.close()

if __name__ == "__main__":
    train()
