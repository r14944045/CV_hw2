"""Held-out proxy: val images with 6 corruption types that appear nowhere in val, test or the 0.5 proxy, at full
strength. It tests robustness to unseen corruptions (none of these were used to choose any setting).

4 streams; each is the full val stream in val label order, with a different corruption on each of its 3 val
segments; every corruption appears in 2 streams at different positions. CPU only.
Output: cache/heldout.pt (x, y, dom, stream_len, names) and cache/heldout_grid.png.

  python proxy_heldout.py
"""
import io
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.environ.get("HW2_DEV_CACHE", os.path.join(HERE, "cache")) + "/"
G = torch.Generator().manual_seed(1)


def rnd(*s):
    return torch.rand(*s, generator=G)


def jpeg(x, q=7):  # heavy JPEG compression (blocking, ringing, colour bleeding)
    out = []
    for img in x:
        buf = io.BytesIO()
        Image.fromarray((img.permute(1, 2, 0) * 255).round().byte().numpy()).save(buf, "JPEG", quality=q)
        out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"))).permute(2, 0, 1).float() / 255)
    return torch.stack(out)


def fog(x):  # low contrast + smooth uneven haze
    B, _, H, W = x.shape
    haze = F.interpolate(rnd(B, 1, 4, 4), size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
    a = 0.45 + 0.15 * rnd(B, 1, 1, 1)
    return x * (1 - a) + a * (0.6 + 0.4 * haze)


def _line_kernel(length, angle):
    k = torch.zeros(length, length); c = length // 2
    for t in torch.linspace(-c, c, 4 * length):
        i = int(round(c + float(t) * math.sin(angle))); j = int(round(c + float(t) * math.cos(angle)))
        k[i, j] = 1.0
    return k / k.sum()


def motion(x, length=13):  # linear motion blur, one random direction per image
    out = []
    for img in x:
        k = _line_kernel(length, math.pi * rnd(1).item())
        out.append(F.conv2d(F.pad(img[:, None], (length // 2,) * 4, mode="reflect"), k[None, None])[:, 0])
    return torch.stack(out)


def snow(x):  # brightened, desaturated scene with many short bright streaks
    B, _, H, W = x.shape
    g = x.mean(1, keepdim=True)
    y = (0.6 * x + 0.4 * g) * 0.8 + 0.2
    flakes = (rnd(B, 1, H, W) > 0.985).float()
    streak = F.conv2d(F.pad(flakes, (3, 3, 3, 3)), _line_kernel(7, 1.1)[None, None] * 7).clamp(0, 1)
    return torch.maximum(y, streak * (0.85 + 0.15 * rnd(B, 1, 1, 1)))


def solarize(x):  # intensities above a threshold are inverted
    t = 0.45 + 0.1 * rnd(len(x), 1, 1, 1)
    return torch.where(x > t, 1 - x, x)


def saltpepper(x, p=0.08):  # impulse noise, shared across channels
    B, _, H, W = x.shape
    r = rnd(B, 1, H, W)
    return torch.where(r < p / 2, torch.zeros_like(x), torch.where(r > 1 - p / 2, torch.ones_like(x), x))


CORR = [jpeg, fog, motion, snow, solarize, saltpepper]
NAMES = ["jpeg", "fog", "motion", "snow", "solarize", "saltpepper"]
STREAMS = [(0, 1, 2), (3, 4, 5), (1, 5, 0), (2, 3, 4)]  # each corruption twice, at different positions


def apply(fn, x):
    out = []
    for i in range(0, len(x), 128):
        y = fn(x[i:i + 128].float() / 255).clamp(0, 1)
        out.append((y * 255).round().to(torch.uint8))
    return torch.cat(out)


if __name__ == "__main__":
    v = torch.load(SP + "val.pt"); x, y = v["x"], v["y"]
    b = np.load(os.path.join(HERE, "val_bounds.npy"))
    xs, ys, dom = [], [], []
    for s, cs in enumerate(STREAMS):
        for j, c in enumerate(cs):
            xs.append(apply(CORR[c], x[b[j]:b[j + 1]])); ys.append(y[b[j]:b[j + 1]]); dom += [c] * (b[j + 1] - b[j])
            print(f"stream {s} segment {j}: {NAMES[c]}", flush=True)
    X = torch.cat(xs)
    torch.save(dict(x=X, y=torch.cat(ys), dom=torch.tensor(dom), stream_len=int(b[-1]), names=NAMES), SP + "heldout.pt")
    import torchvision
    idx = [s * 2112 + b[j] + k * 41 for s in range(4) for j in range(3) for k in range(3)]
    torchvision.utils.save_image(torchvision.utils.make_grid(X[idx].float() / 255, nrow=9, padding=2), SP + "heldout_grid.png")
    print("saved", X.shape)
