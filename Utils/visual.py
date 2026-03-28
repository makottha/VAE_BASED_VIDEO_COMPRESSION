import os
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from VQ_VAE_2.deeper_vqvae2_architecture_64x64 import MultiScaleVQVAE2

# --- CONFIG ---
VIDEO_PATH = "../dataset_human_readable_64/test/clips/v_BlowingCandles_g05_c01_clip_0000.avi"
MODEL_PATH = "../checkpoints/MS_VQVAE/vqvae2_epoch050.pt"
OUTPUT_PATH = "../side_by_side_recon.avi"
FPS = 16
FRAME_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
print("📦 Loading model...")
model = MultiScaleVQVAE2().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# --- Load and Prepare Clip ---
def load_clip(video_path, expected_frames=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < expected_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = ToTensor()(frame_rgb)  # Converts to [0,1]
        frames.append(frame_tensor)
    cap.release()
    if len(frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {len(frames)}")
    return torch.stack(frames, dim=1).unsqueeze(0).to(DEVICE)  # (1, 3, T, H, W)

# --- Evaluate PSNR & SSIM ---
def evaluate_metrics(original, reconstructed):
    original = original.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    reconstructed = reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    original = (np.clip(original, 0, 1) * 255).astype(np.uint8)
    reconstructed = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)

    psnr_list, ssim_list = [], []
    for orig, recon in zip(original, reconstructed):
        psnr = compare_psnr(orig, recon, data_range=255)
        ssim = compare_ssim(orig, recon, channel_axis=2, data_range=255)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"📊 Average PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

def save_reconstructed_video(reconstructed, output_path, fps):
    reconstructed = (reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    height, width, _ = reconstructed[0].shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame in reconstructed:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"🎞️ Saved reconstructed-only video to: {output_path}")


# --- Save Side-by-Side Video ---
def save_side_by_side_video(original, reconstructed, output_path, fps):
    original = (original.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    reconstructed = (reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

    height, width, _ = original[0].shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width * 2, height))

    for orig, recon in zip(original, reconstructed):
        side_by_side = np.hstack((cv2.cvtColor(orig, cv2.COLOR_RGB2BGR),
                                  cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)))
        writer.write(side_by_side)
    writer.release()
    print(f"🎞️ Saved side-by-side video to: {output_path}")

# --- Compression Rate ---
def compute_compression_rate(original_path, compressed_path):
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    ratio = original_size / compressed_size
    rate = 100 * (1 - compressed_size / original_size)
    print(f"📉 Compression Ratio: {ratio:.2f}× | Compression Rate: {rate:.1f}% reduction")

# --- Main ---
if __name__ == "__main__":
    print("🎬 Loading clip...")
    clip_tensor = load_clip(VIDEO_PATH)

    print("🧠 Running reconstruction...")
    with torch.no_grad():
        reconstructed, *_ = model(clip_tensor)

    evaluate_metrics(clip_tensor, reconstructed)
    save_reconstructed_video(reconstructed, "../reconstructed_only.avi", FPS)
    save_side_by_side_video(clip_tensor, reconstructed, OUTPUT_PATH, FPS)
    compute_compression_rate(VIDEO_PATH, OUTPUT_PATH)
