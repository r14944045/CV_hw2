"""Strong test-time augmentation (CoTTA's `get_tta_transforms`, also used by RoTTA's student).

Runs on the GPU on a whole normalised batch with independent random parameters per sample
(the official code samples one set of parameters per batch). Pipeline, as in CoTTA (ImageNet, soft=False):
colour jitter (brightness, contrast, saturation, hue, gamma) -> random affine (rotation +-15 deg,
translation +-1/16, scale 0.9-1.1, edge padding) -> Gaussian blur (k=5, sigma 0.001-0.5)
-> horizontal flip -> Gaussian noise (std 0.005) -> clip.
"""
import math

import torch
import torch.nn.functional as F

from .common import MEAN, STD

_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _uniform(B, lo, hi, device):
    return torch.empty(B, 1, 1, 1, device=device).uniform_(lo, hi)


def _adjust_hue(x, h):
    """Rotate hue by h (fraction of a full turn, shape (B,1,1,1)) in YIQ space."""
    yiq = torch.einsum("ij,bjhw->bihw", _RGB2YIQ.to(x), x)
    theta = 2 * math.pi * h  # (B, 1, 1, 1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    i, q = yiq[:, 1:2], yiq[:, 2:3]
    yiq = torch.cat([yiq[:, :1], i * cos - q * sin, i * sin + q * cos], 1)
    return torch.einsum("ij,bjhw->bihw", _YIQ2RGB.to(x), yiq)


def _gaussian_blur(x, sigma, k=5):
    """Separable Gaussian blur with a per-sample sigma (shape (B,))."""
    B, C, H, W = x.shape
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    ker = torch.exp(-r[None] ** 2 / (2 * sigma[:, None] ** 2))
    ker = (ker / ker.sum(1, keepdim=True)).repeat_interleave(C, 0)  # (B*C, k)
    y, p = x.reshape(1, B * C, H, W), k // 2
    y = F.conv2d(F.pad(y, (p, p, 0, 0), mode="reflect"), ker[:, None, None, :], groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, p, p), mode="reflect"), ker[:, None, :, None], groups=B * C)
    return y.reshape(B, C, H, W)


def strong_augment(x, noise_std=0.005):
    """x: normalised batch (B, 3, H, W) -> augmented normalised batch."""
    mean, std = MEAN.to(x.device), STD.to(x.device)
    x = (x * std + mean).clamp(0, 1)
    B, dev = x.size(0), x.device

    # colour jitter
    x = (x * _uniform(B, 0.6, 1.4, dev)).clamp(0, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * _uniform(B, 0.7, 1.3, dev) + m).clamp(0, 1)
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    x = ((x - gray) * _uniform(B, 0.5, 1.5, dev) + gray).clamp(0, 1)
    x = _adjust_hue(x, _uniform(B, -0.06, 0.06, dev)).clamp(0, 1)
    x = x.clamp_min(1e-6) ** _uniform(B, 0.7, 1.3, dev)

    # random affine (edge padding)
    ang = _uniform(B, -15, 15, dev).view(B) * math.pi / 180
    scale = _uniform(B, 0.9, 1.1, dev).view(B)
    tx, ty = _uniform(B, -1 / 8, 1 / 8, dev).view(B), _uniform(B, -1 / 8, 1 / 8, dev).view(B)  # 1/16 width
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # blur, flip, noise
    x = _gaussian_blur(x, torch.empty(B, device=dev).uniform_(0.001, 0.5))
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    x = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    return (x - mean) / std
