"""
priors_3level.py — Autoregressive priors for three-level MS-VQ-VAE-3

Three priors, trained in sequence (AE frozen):

  TopPrior3D               : p_θ(z_top)                — unconditional
  MidPriorConditional3D    : p_φ(z_mid  | z_top)       — conditioned on top
  BottomPriorConditional3D : p_ψ(z_bot  | z_mid)       — conditioned on mid

All use masked 3D convolutions (PixelCNN-style) for autoregressive modelling
over the (T, H, W) spatial-temporal grid of each quantized level.

Bit computation:
  bits = TopPrior(idx_top) + MidPrior(idx_mid, z_top_up) + BotPrior(idx_bot, z_mid_up)
  BPP  = bits / (T × H × W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Masked Conv utilities (identical to MS_VQ_VAE_RD) ──────────────────────

def build_mask_3d(weight: torch.Tensor, mask_type: str) -> torch.Tensor:
    assert mask_type in ("A", "B")
    kt, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    ct, ch, cw = kt // 2, kh // 2, kw // 2

    mask = torch.ones_like(weight)
    mask[:, :, ct + 1:, :, :]  = 0
    mask[:, :, ct, ch + 1:, :] = 0
    mask[:, :, ct, ch, cw + 1:] = 0
    if mask_type == "A":
        mask[:, :, ct, ch, cw] = 0
    return mask


class MaskedConv3d(nn.Conv3d):
    def __init__(self, *args, mask_type="A", **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.mask_type = mask_type
        mask = build_mask_3d(self.weight, mask_type=self.mask_type)
        self.register_buffer("mask", mask)
        self.weight.register_hook(lambda g: g * self.mask)

    def forward(self, x):
        w = self.weight * self.mask
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResidualMaskedBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv3d(channels, channels, kernel_size=3, padding=1, mask_type="B"),
            nn.ReLU(inplace=True),
            MaskedConv3d(channels, channels, kernel_size=1, padding=0, mask_type="B"),
        )

    def forward(self, x):
        return x + self.net(x)


# ─── Priors ──────────────────────────────────────────────────────────────────

class TopPrior3D(nn.Module):
    """
    Unconditional autoregressive prior over top-level indices.

    Input:  idx_top  (B, T, H, W)  int64
    Output: logits   (B, K, T, H, W)

    Grid size at 128×128: (4, 8, 8) = 256 symbols/clip → lightweight prior.
    """
    def __init__(self, K: int, emb_dim=64, hidden=128, n_res=6):
        super().__init__()
        self.K = int(K)
        self.emb    = nn.Embedding(self.K, emb_dim)
        self.in_conv = MaskedConv3d(emb_dim, hidden, kernel_size=3, padding=1, mask_type="A")
        self.res    = nn.Sequential(*[ResidualMaskedBlock3D(hidden) for _ in range(n_res)])
        self.out    = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, hidden, kernel_size=1, padding=0, mask_type="B"),
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, self.K,  kernel_size=1, padding=0, mask_type="B"),
        )

    def forward(self, idx_top: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx_top).permute(0, 4, 1, 2, 3).contiguous()
        h = self.in_conv(x)
        h = self.res(h)
        return self.out(h)


class MidPriorConditional3D(nn.Module):
    """
    Autoregressive conditional prior over mid-level indices given top conditioning.

    Input:
      idx_mid : (B, T, H, W)     int64   — mid-level indices
      cond    : (B, Cc, T, H, W) float   — z_top_up (top features upsampled to mid resolution)
    Output:
      logits  : (B, K_mid, T, H, W)

    Grid size at 128×128: (8, 16, 16) = 2048 symbols/clip.
    """
    def __init__(self, K_mid: int, cond_channels: int, emb_dim=64, hidden=192, n_res=8):
        super().__init__()
        self.K_mid = int(K_mid)
        self.emb       = nn.Embedding(self.K_mid, emb_dim)
        self.cond_proj = nn.Conv3d(cond_channels, hidden, kernel_size=1)
        self.in_conv   = MaskedConv3d(emb_dim, hidden, kernel_size=3, padding=1, mask_type="A")
        self.res       = nn.Sequential(*[ResidualMaskedBlock3D(hidden) for _ in range(n_res)])
        self.out       = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, hidden,    kernel_size=1, padding=0, mask_type="B"),
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, self.K_mid, kernel_size=1, padding=0, mask_type="B"),
        )

    def forward(self, idx_mid: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx_mid).permute(0, 4, 1, 2, 3).contiguous()
        h = self.in_conv(x)
        h = h + self.cond_proj(cond)
        h = self.res(h)
        return self.out(h)


class BottomPriorConditional3D(nn.Module):
    """
    Autoregressive conditional prior over bottom-level indices given mid conditioning.

    Input:
      idx_bottom : (B, T, H, W)     int64  — bottom-level indices
      cond       : (B, Cc, T, H, W) float  — z_mid_up (stage1 decoder features at bottom res)
    Output:
      logits     : (B, K_bot, T, H, W)

    Grid size at 128×128: (16, 32, 32) = 16384 symbols/clip.
    This is the most expensive prior — n_res=8 is a reasonable default.
    """
    def __init__(self, K_bot: int, cond_channels: int, emb_dim=64, hidden=192, n_res=8):
        super().__init__()
        self.K_bot = int(K_bot)
        self.emb       = nn.Embedding(self.K_bot, emb_dim)
        self.cond_proj = nn.Conv3d(cond_channels, hidden, kernel_size=1)
        self.in_conv   = MaskedConv3d(emb_dim, hidden, kernel_size=3, padding=1, mask_type="A")
        self.res       = nn.Sequential(*[ResidualMaskedBlock3D(hidden) for _ in range(n_res)])
        self.out       = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, hidden,    kernel_size=1, padding=0, mask_type="B"),
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, self.K_bot, kernel_size=1, padding=0, mask_type="B"),
        )

    def forward(self, idx_bottom: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx_bottom).permute(0, 4, 1, 2, 3).contiguous()
        h = self.in_conv(x)
        h = h + self.cond_proj(cond)
        h = self.res(h)
        return self.out(h)


# ─── Bit computation helper ──────────────────────────────────────────────────

def categorical_bits_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Sum of -log2 p(target) for categorical logits.
    logits:  (B, K, T, H, W)
    targets: (B, T, H, W) int64
    returns: scalar bits
    """
    ce   = F.cross_entropy(logits, targets, reduction="sum")
    bits = ce / torch.log(torch.tensor(2.0, device=logits.device))
    return bits
