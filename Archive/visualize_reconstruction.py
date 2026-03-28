"""
visualize_reconstruction_64x64.py

Description:
-------------
Visualizes VAE reconstruction on a 2-second (32-frame) video clip
resized to 64x64 resolution. Computes PSNR and SSIM metrics and
saves a side-by-side comparison video.

Usage:
- Ensure model checkpoint and target clip are configured below.
- Run: python visualize_reconstruction_64x64.py
"""

import cv2
import torch
import numpy as np
from deeper_vae_architecture_64x64 import DeeperVideoVAE
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# --- CONFIG ---
VIDEO_PATH = "./dataset_human_readable/test/clips/v_TrampolineJumping_g16_c02_clip_0000.avi"  # Input clip
MODEL_PATH = "../checkpoints/deeper_vae_epoch005.pt"  # Trained model
OUTPUT_PATH = "side_by_side_recon.avi"                                                        # Output video path
FPS = 16
FRAME_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = DeeperVideoVAE(latent_dim=4096).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
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
        frame_tensor = ToTensor()(frame_rgb)  # (3, H, W)
        frames.append(frame_tensor)

    cap.release()

    if len(frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {len(frames)}")

    clip_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(DEVICE)  # Shape: (1, 3, 32, H, W)
    return clip_tensor

# --- Evaluate PSNR & SSIM ---
def evaluate_metrics(original, reconstructed):
    original = original.squeeze().permute(1, 2, 3, 0).cpu().numpy()       # (T, H, W, C)
    reconstructed = reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy()

    original = (original * 255).astype(np.uint8)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    psnr_list = []
    ssim_list = []

    for orig, recon in zip(original, reconstructed):
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        recon_gray = cv2.cvtColor(recon, cv2.COLOR_RGB2GRAY)

        psnr = compare_psnr(orig, recon, data_range=255)
        ssim = compare_ssim(orig_gray, recon_gray, data_range=255)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"📊 PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

# --- Save Side-by-Side Video ---
def save_side_by_side_video(original, reconstructed, output_path, fps=16):
    original = original.squeeze().permute(1, 2, 3, 0).cpu().numpy()
    reconstructed = reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy()

    original = (original * 255).astype(np.uint8)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    height, width, _ = original[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width * 2, height))

    for orig, recon in zip(original, reconstructed):
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        side_by_side = np.hstack((orig_bgr, recon_bgr))
        out.write(side_by_side)

    out.release()
    print(f"✅ Side-by-side reconstruction saved to: {output_path}")

# --- Main ---
if __name__ == "__main__":
    clip_tensor = load_clip(VIDEO_PATH)

    with torch.no_grad():
        mu, logvar = model.encode(clip_tensor)
        z = model.reparameterize(mu, logvar)
        reconstructed = model.decode(z)

    evaluate_metrics(clip_tensor, reconstructed)
    save_side_by_side_video(clip_tensor, reconstructed, OUTPUT_PATH, fps=FPS)
