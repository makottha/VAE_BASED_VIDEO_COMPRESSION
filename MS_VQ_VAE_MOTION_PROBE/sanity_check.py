import sys, torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "MS_VQ_VAE_RD"))
from models import MSVQVAE2Video
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# RAFT
weights    = Raft_Small_Weights.DEFAULT
raft_model = raft_small(weights=weights).to(device).eval()
print("RAFT loaded OK")

# AE
ae = MSVQVAE2Video(
    top_dim=256, bottom_dim=128, K_top=1024, K_bottom=1024,
    commit_top=0.25, commit_bottom=0.25, norm="gn",
    use_vgg=False, soft_temp=1.0, use_ema=False,
).to(device).eval()
ckpt_path = Path(__file__).parent.parent / "MS_VQ_VAE_RD" / "outputs_msvqvae_rd" / "ae_K1024_epoch020.pt"
ckpt = torch.load(str(ckpt_path), map_location="cpu")
ae.load_state_dict(ckpt["ae"], strict=False)
print("AE loaded OK")

# One clip
data_dir = Path(__file__).parent.parent / "dataset_64" / "test" / "tensors"
clips    = sorted(data_dir.glob("*.pt"))
clip     = torch.load(str(clips[0]), map_location="cpu").float()
if clip.max() > 1.0:
    clip = clip / 255.0
print(f"Clip shape: {clip.shape}  range: [{clip.min():.3f}, {clip.max():.3f}]")

# AE forward
with torch.no_grad():
    out = ae(clip.unsqueeze(0).to(device))
print(f"Recon shape:   {out['recon'].shape}")
print(f"entropy_top:   {out['entropy_top']:.3f} bits/sym")
print(f"entropy_bot:   {out['entropy_bottom']:.3f} bits/sym")
print(f"used_top:      {out['used_top']} / 1024")
print(f"used_bot:      {out['used_bottom']} / 1024")

# RAFT on frame pair
transforms = weights.transforms()
f1 = clip[:, 0].unsqueeze(0).to(device)
f2 = clip[:, 1].unsqueeze(0).to(device)
f1_up = F.interpolate(f1 * 255, scale_factor=4, mode="bilinear", align_corners=False)
f2_up = F.interpolate(f2 * 255, scale_factor=4, mode="bilinear", align_corners=False)
f1_t, f2_t = transforms(f1_up, f2_up)
with torch.no_grad():
    flow = raft_model(f1_t, f2_t)[-1].squeeze(0)
print(f"Flow shape:    {flow.shape}  range: [{flow.min():.3f}, {flow.max():.3f}] pixels")
print("\nAll checks passed. Ready to run probe.")
