# losses.py
import torch
import torch.nn.functional as F


def recon_loss_fn(recon: torch.Tensor, x: torch.Tensor, mode="l1") -> torch.Tensor:
    recon = recon.clamp(0, 1)
    if mode == "l1":
        return F.l1_loss(recon, x)
    if mode == "mse":
        return F.mse_loss(recon, x)
    raise ValueError(f"Unknown recon loss mode: {mode}")

def vgg_perceptual_loss(vgg, x, recon, stride: int = 4, mode: str = "l1"):
    """
    VGG perceptual loss with correct gradient flow:
      - VGG params are frozen externally (requires_grad=False)
      - Grad flows through VGG(recon) -> recon -> AE
      - Ground-truth VGG(x) is computed under no_grad for efficiency

    Args:
        vgg: a feature extractor (e.g., VGG16.features[:N])
        x, recon: (B,3,T,H,W) tensors in [0,1]
        stride: use every Nth frame
        mode: 'l1' (recommended) or 'mse'
    """
    assert vgg is not None, "vgg_perceptual_loss called with vgg=None"

    B, C, T, H, W = x.shape
    x_sub = x[:, :, ::stride]
    r_sub = recon[:, :, ::stride]

    x_2d = x_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    r_2d = r_sub.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

    # If you did NOT use ImageNet normalization in your baseline, keep it off.
    # If you want it, enable consistently across runs.
    # mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    # std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    # x_2d = (x_2d - mean) / std
    # r_2d = (r_2d - mean) / std

    with torch.no_grad():
        feat_x = vgg(x_2d)

    feat_r = vgg(r_2d)  # IMPORTANT: keep grads

    if mode == "mse":
        return F.mse_loss(feat_r, feat_x)
    return F.l1_loss(feat_r, feat_x)



def bpp_from_bits(bits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    bits: scalar sum over batch
    x: (B,3,T,H,W)
    bpp = bits / (B*T*H*W)
    """
    B, _, T, H, W = x.shape
    denom = B * T * H * W
    return bits / float(denom)
