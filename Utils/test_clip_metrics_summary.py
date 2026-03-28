import os
import cv2
import torch
import numpy as np
from glob import glob
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd

from VQ_VAE_2.deeper_vqvae2_architecture_64x64 import MultiScaleVQVAE2

# --- CONFIG ---
TEST_CLIPS_DIR = "../dataset_human_readable_64/test/clips"
MODEL_PATH = "../checkpoints/MS_VQVAE/vqvae2_epoch050.pt"
EXPECTED_FRAMES = 32
FRAME_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
print("📦 Loading model...")
model = MultiScaleVQVAE2().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# --- Load Video Clip ---
def load_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < EXPECTED_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = ToTensor()(frame_rgb)
        frames.append(frame_tensor)
    cap.release()
    if len(frames) != EXPECTED_FRAMES:
        return None
    return torch.stack(frames, dim=1).unsqueeze(0).to(DEVICE)

# --- Compute Metrics ---
def evaluate_metrics(original, reconstructed):
    original_np = (original.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    reconstructed_np = (reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

    psnr_list, ssim_list = [], []
    for orig, recon in zip(original_np, reconstructed_np):
        psnr = compare_psnr(orig, recon, data_range=255)
        ssim = compare_ssim(orig, recon, channel_axis=2, data_range=255)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    return np.mean(psnr_list), np.mean(ssim_list)

# --- Compute File Sizes ---
def get_file_size(path):
    return os.path.getsize(path) / 1024  # KB

# --- Main Loop ---
results = []
video_paths = sorted(glob(os.path.join(TEST_CLIPS_DIR, "*.avi")))

print(f"🧪 Evaluating {len(video_paths)} clips...")
for path in video_paths:
    clip_tensor = load_clip(path)
    if clip_tensor is None:
        continue
    with torch.no_grad():
        reconstructed, *_ = model(clip_tensor)

    psnr, ssim = evaluate_metrics(clip_tensor, reconstructed)

    # Compression size estimate
    original_size_kb = get_file_size(path)
    # Estimate: 2 codebooks × 2 bytes per code × 32 frames × 8×8 spatial = 8192 bytes
    compressed_estimate_kb = 2 * 2 * 32 * 8 * 8 / 1024  # ~1 KB per clip
    compression_ratio = compressed_estimate_kb / original_size_kb
    reduction = 100 * (1 - compression_ratio)

    results.append({
        "clip": os.path.basename(path),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "original_size_kb": round(original_size_kb, 2),
        "compressed_estimate_kb": round(compressed_estimate_kb, 2),
        "compression_ratio": round(compression_ratio, 2),
        "reduction_percent": round(reduction, 1)
    })
    print(f"{os.path.basename(path)}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, Compression={compression_ratio:.2f}×")

# --- Add Mean Row ---
df = pd.DataFrame(results)
mean_row = {
    "clip": "MEAN",
    "psnr": df["psnr"].mean(),
    "ssim": df["ssim"].mean(),
    "original_size_kb": df["original_size_kb"].mean(),
    "compressed_estimate_kb": df["compressed_estimate_kb"].mean(),
    "compression_ratio": df["compression_ratio"].mean(),
    "reduction_percent": df["reduction_percent"].mean()
}
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

# --- Save ---
df.to_csv("test_clip_metrics.csv", index=False)
print("\n✅ Results saved to test_clip_metrics.csv")

print(f"\n📊 Overall Summary:")
print(f"  ▸ Mean PSNR: {mean_row['psnr']:.2f} dB")
print(f"  ▸ Mean SSIM: {mean_row['ssim']:.4f}")
print(f"  ▸ Mean Compression Ratio: {mean_row['compression_ratio']:.2f}×")
print(f"  ▸ Mean Compression Reduction: {mean_row['reduction_percent']:.1f}%")
