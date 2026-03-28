# priors.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_conv3d_weight(conv: nn.Conv3d, mask_type: str):
    """
    Applies a PixelCNN-style mask to a Conv3d weight.
    mask_type: 'A' (exclude current) or 'B' (include current)
    """
    assert mask_type in ("A", "B")
    w = conv.weight
    kt, kh, kw = w.shape[2], w.shape[3], w.shape[4]
    ct, ch, cw = kt // 2, kh // 2, kw // 2

    mask = torch.ones_like(w)

    # Zero out "future" positions (t > ct) or (t==ct and h > ch) or (t==ct and h==ch and w > cw)
    mask[:, :, ct+1:, :, :] = 0
    mask[:, :, ct, ch+1:, :] = 0
    mask[:, :, ct, ch, cw+1:] = 0

    if mask_type == "A":
        # exclude current position
        mask[:, :, ct, ch, cw] = 0

    conv.weight.data *= mask

def build_mask_3d(weight: torch.Tensor, mask_type: str) -> torch.Tensor:
    """
    Build a PixelCNN-style 3D mask for a conv weight tensor.

    weight: (out_ch, in_ch, kt, kh, kw)
    mask_type: 'A' (exclude current position) or 'B' (include current position)
    """
    assert mask_type in ("A", "B")
    kt, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    ct, ch, cw = kt // 2, kh // 2, kw // 2

    mask = torch.ones_like(weight)

    # Future in time
    mask[:, :, ct + 1 :, :, :] = 0
    # Same time, future in height
    mask[:, :, ct, ch + 1 :, :] = 0
    # Same time/height, future in width
    mask[:, :, ct, ch, cw + 1 :] = 0

    if mask_type == "A":
        # Exclude current position
        mask[:, :, ct, ch, cw] = 0

    return mask


class MaskedConv3d(nn.Conv3d):
    """
    Conv3d with a fixed autoregressive mask (PixelCNN-style) that is enforced:
      - on the weights every forward
      - and on the gradients (so masked weights never re-appear after optimizer steps)

    This prevents "future leakage" and the near-zero bits collapse you observed.
    """
    def __init__(self, *args, mask_type="A", **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.mask_type = mask_type

        # Register a persistent mask buffer (same shape as weight)
        mask = build_mask_3d(self.weight, mask_type=self.mask_type)
        self.register_buffer("mask", mask)

        # Mask gradients so forbidden weights cannot grow back
        self.weight.register_hook(lambda g: g * self.mask)

    def forward(self, x):
        # Enforce the mask on weights each forward pass
        # (safe + cheap relative to the rest of the model)
        w = self.weight * self.mask
        return F.conv3d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )



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


class TopPrior3D(nn.Module):
    """
    Autoregressive prior over top indices.
    Input: idx_top (B,T,H,W) integer
    Output: logits (B,K,T,H,W)
    """
    def __init__(self, K: int, emb_dim=64, hidden=128, n_res=6):
        super().__init__()
        self.K = int(K)
        self.emb = nn.Embedding(self.K, emb_dim)

        self.in_conv = MaskedConv3d(emb_dim, hidden, kernel_size=3, padding=1, mask_type="A")
        self.res = nn.Sequential(*[ResidualMaskedBlock3D(hidden) for _ in range(n_res)])
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, hidden, kernel_size=1, padding=0, mask_type="B"),
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, self.K, kernel_size=1, padding=0, mask_type="B"),
        )

    def forward(self, idx_top: torch.Tensor):
        # idx_top: (B,T,H,W)
        x = self.emb(idx_top).permute(0, 4, 1, 2, 3).contiguous()  # (B,emb,T,H,W)
        h = self.in_conv(x)
        h = self.res(h)
        logits = self.out(h)
        return logits


class BottomPriorConditional3D(nn.Module):
    """
    Autoregressive conditional prior over bottom indices given top conditioning feature (z_top_up).
    Input:
      idx_bottom: (B,T,H,W) integer
      cond: (B,Cc,T,H,W) float tensor (e.g., z_top_up from AE)
    Output:
      logits: (B,Kb,T,H,W)
    """
    def __init__(self, Kb: int, cond_channels: int, emb_dim=64, hidden=192, n_res=8):
        super().__init__()
        self.Kb = int(Kb)
        self.emb = nn.Embedding(self.Kb, emb_dim)

        # project conditioning to hidden
        self.cond_proj = nn.Conv3d(cond_channels, hidden, kernel_size=1)

        self.in_conv = MaskedConv3d(emb_dim, hidden, kernel_size=3, padding=1, mask_type="A")
        self.res = nn.Sequential(*[ResidualMaskedBlock3D(hidden) for _ in range(n_res)])
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, hidden, kernel_size=1, padding=0, mask_type="B"),
            nn.ReLU(inplace=True),
            MaskedConv3d(hidden, self.Kb, kernel_size=1, padding=0, mask_type="B"),
        )

    def forward(self, idx_bottom: torch.Tensor, cond: torch.Tensor):
        # idx_bottom: (B,T,H,W)
        x = self.emb(idx_bottom).permute(0, 4, 1, 2, 3).contiguous()  # (B,emb,T,H,W)
        h = self.in_conv(x)
        h = h + self.cond_proj(cond)  # inject conditioning
        h = self.res(h)
        logits = self.out(h)
        return logits


def categorical_bits_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes sum of -log2 p(target) for categorical logits.
    logits: (B,K,T,H,W)
    targets: (B,T,H,W) int64
    returns: scalar bits (sum over all positions and batch)
    """
    # PyTorch CE uses natural log; convert to log2 by / ln(2)
    ce = F.cross_entropy(logits, targets, reduction="sum")
    bits = ce / torch.log(torch.tensor(2.0, device=logits.device))
    return bits
