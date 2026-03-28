import os
import cv2
import torch
import numpy as np
from glob import glob
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as compare_ssim

from MS_VQ_VAE_2.deeper_vqvae2_entropy_model import MultiScaleVQVAE2

# --- CONFIG ---
TEST_CLIPS_DIR = "./dataset_human_readable_64/test/clips"
MODEL_PATH = "checkpoints/MS_VQVAE/vqvae2_epoch050.pt"
OUTPUT_DIR = "./best_pairs"
EXPECTED_FRAMES = 32
FRAME_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# --- Identify Best Matching Frame ---
def find_best_frame(original, reconstructed):
    original_np = (original.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    reconstructed_np = (reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)

    best_ssim = -1
    best_idx = -1

    for idx, (orig, recon) in enumerate(zip(original_np, reconstructed_np)):
        ssim = compare_ssim(orig, recon, channel_axis=2, data_range=255)
        if ssim > best_ssim:
            best_ssim = ssim
            best_idx = idx

    return best_idx, best_ssim, original_np[best_idx], reconstructed_np[best_idx]

# --- Process All Clips ---
video_paths = sorted(glob(os.path.join(TEST_CLIPS_DIR, "*.avi")))
print(f"🔍 Processing {len(video_paths)} test clips...")

for path in video_paths:
    clip_name = os.path.splitext(os.path.basename(path))[0]
    clip_tensor = load_clip(path)
    if clip_tensor is None:
        print(f"⚠️ Skipping {clip_name} (not enough frames)")
        continue

    with torch.no_grad():
        reconstructed, *_ = model(clip_tensor)

    idx, ssim, orig_img, recon_img = find_best_frame(clip_tensor, reconstructed)

    # Prepare output filenames
    orig_filename = f"{clip_name}_frame{idx:02d}_original.png"
    recon_filename = f"{clip_name}_frame{idx:02d}_reconstructed.png"
    orig_path = os.path.join(OUTPUT_DIR, orig_filename)
    recon_path = os.path.join(OUTPUT_DIR, recon_filename)

    # Save images
    cv2.imwrite(orig_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(recon_path, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

    print(f"✅ {clip_name} | Frame {idx} | SSIM: {ssim:.4f} → Saved to {OUTPUT_DIR}")
