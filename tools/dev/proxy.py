"""Labelled proxy for the test stream: val images + synthetic versions of the 12 test corruptions.

4 streams; stream s applies corruptions 3s, 3s+1, 3s+2 to val segments [0:776], [776:1500], [1500:2112]
(val label order kept, each val image used once per stream). Output: proxy.pt with x (4*2112 images), y.
"""
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

SP = os.environ.get("HW2_DEV_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")) + "/"
G = torch.Generator().manual_seed(0)


def rnd(*s):
    return torch.rand(*s, generator=G)


def blur(x, s):
    k = int(2 * math.ceil(3 * s) + 1); t = torch.arange(k) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = (g / g.sum()).to(x)
    C = x.size(1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def lum(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def noise(x):  # low light + strong colour noise
    x = x * (0.35 + 0.2 * rnd(len(x), 1, 1, 1))
    return x + 0.10 * torch.randn(x.shape, generator=G)


def wave(x):
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    out = []
    for i in range(B):
        a1, a2 = 0.06 + 0.04 * rnd(1).item(), 0.03 * rnd(1).item(); f1, f2 = 3 + 3 * rnd(1).item(), 2 + 3 * rnd(1).item()
        p1, p2 = 6.28 * rnd(1).item(), 6.28 * rnd(1).item()
        gx = xx + a1 * torch.sin(2 * math.pi * f1 * yy / 2 + p1)
        gy = yy + a2 * torch.sin(2 * math.pi * f2 * xx / 2 + p2)
        out.append(F.grid_sample(x[i:i + 1], torch.stack([gx, gy], -1)[None], padding_mode="reflection", align_corners=False))
    return torch.cat(out)


def chrom(x):
    x = blur(x, 2.0)
    s = 4
    r = torch.roll(x[:, 0:1], shifts=(s, s), dims=(2, 3)); b = torch.roll(x[:, 2:3], shifts=(-s, -s), dims=(2, 3))
    return torch.cat([r, x[:, 1:2], b], 1) * 1.05 + 0.03


def jet(v):
    r = (1.5 - (4 * v - 3).abs()).clamp(0, 1); g = (1.5 - (4 * v - 2).abs()).clamp(0, 1); b = (1.5 - (4 * v - 1).abs()).clamp(0, 1)
    return torch.cat([r, g, b], 1)


def thermal(x):
    v = blur(lum(x), 1.0)
    v = (v - v.amin((2, 3), keepdim=True)) / (v.amax((2, 3), keepdim=True) - v.amin((2, 3), keepdim=True) + 1e-6)
    return jet(v)


def sepia(x):
    M = torch.tensor([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    y = torch.einsum("ij,bjhw->bihw", M, x) * 0.85
    y = y + 0.06 * torch.randn(y.shape[0], 1, *y.shape[2:], generator=G)
    B, _, H, W = y.shape
    for i in range(B):  # vertical scratches
        for _ in range(int(2 + 4 * rnd(1).item())):
            c = int(rnd(1).item() * W); y[i, :, :, c] = 0.85
    return y


def pixel(x):
    B, _, H, W = x.shape
    s = 4
    y = F.interpolate(F.avg_pool2d(x, s), scale_factor=s, mode="nearest")
    y = torch.round(y * 6) / 6  # colour quantisation
    return y + 0.05 * torch.randn(y.shape, generator=G)


def stripes(x):
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    out = []
    for i in range(B):
        per = 10 + 8 * rnd(1).item()
        m = ((torch.sin(2 * math.pi * (xx + yy).float() / per) > 0).float())
        out.append(x[i] * (1 - 0.45 * m) + 0.2 * m * (rnd(1).item() > 0.5))
    return torch.stack(out)


def grid(x):
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    m = ((yy % 3 == 0) | (xx % 3 == 0)).float()
    return x * (1 - 0.45 * m)


def teal(x):
    y = blur(x, 1.2)
    y = torch.cat([y[:, 0:1] * 0.35, y[:, 1:2] * 0.75 + 0.12, y[:, 2:3] * 0.75 + 0.18], 1)
    return torch.round(y * 7) / 7


def halftone(x, per=4):
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    scr = 0.5 + 0.25 * (torch.cos(2 * math.pi * xx / per) + torch.cos(2 * math.pi * yy / per))
    return (blur(x, 0.8) > scr).float() * 0.9 + 0.05


def halft2(x):
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    out = []
    for c, ang in enumerate([0.26, 1.31, 0.79]):
        u = xx * math.cos(ang) + yy * math.sin(ang); v = -xx * math.sin(ang) + yy * math.cos(ang)
        scr = 0.5 + 0.25 * (torch.cos(2 * math.pi * u / 5) + torch.cos(2 * math.pi * v / 5))
        out.append((blur(x[:, c:c + 1], 1.0) > scr).float())
    y = torch.cat(out, 1)
    return 0.5 * y + 0.5 * blur(x, 1.0)


def poster(x):
    y = blur(x, 2.5)
    return torch.round(y * 5) / 5


CORR = [noise, wave, chrom, thermal, sepia, pixel, stripes, grid, teal, halftone, halft2, poster]
NAMES = ["noise", "wave", "chrom", "thermal", "sepia", "pixel", "stripes", "grid", "teal", "halft", "halft2", "poster"]


def apply(fn, x, mix=0.5):
    out = []
    for i in range(0, len(x), 128):
        xb = x[i:i + 128].float() / 255
        y = fn(xb).clamp(0, 1)
        y = mix * y + (1 - mix) * xb
        out.append((y * 255).round().to(torch.uint8))
    return torch.cat(out)


if __name__ == "__main__":
    import sys
    mix = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    v = torch.load(SP + "val.pt"); x, y = v["x"], v["y"]; b = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_bounds.npy"))
    xs, ys, dom = [], [], []
    for s in range(4):
        for j in range(3):
            c = 3 * s + j
            xs.append(apply(CORR[c], x[b[j]:b[j + 1]], mix)); ys.append(y[b[j]:b[j + 1]]); dom += [c] * (b[j + 1] - b[j])
    torch.save(dict(x=torch.cat(xs), y=torch.cat(ys), dom=torch.tensor(dom), stream_len=int(b[-1]), mix=mix), SP + f"proxy{mix}.pt")
    import torchvision
    X = torch.cat(xs); idx = []
    for s in range(4):
        for j in range(3):
            base = s * 2112 + b[j]; idx += [base + k * 37 for k in range(4)]
    torchvision.utils.save_image(torchvision.utils.make_grid(X[idx].float() / 255, nrow=8, padding=2), SP + f"proxy{mix}_grid.png")
    print("saved", X.shape)
