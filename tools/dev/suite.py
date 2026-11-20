"""Corruption suite on val (allowed by the slide-21 FAQ: "adding your own corruptions to val for evaluation").

Rules fixed on 2026-11-19 at about 22:45 CST, before the suite was built or any method was scored on it:
- Corruptions: the 15 ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019) in their published order, at one
  fixed severity, 3. They are taken wholesale from the published list; none was chosen or dropped by resemblance to
  anything. Parameters follow make_imagenet_c.py (github.com/hendrycks/robustness) and the size-independent
  reimplementation in imagecorruptions (github.com/bethgelab/imagecorruptions), rewritten with numpy/scipy because
  new packages could not be installed here.
- Mechanical exclusion: frost (it blends in photos shipped with the original code, which are not available).
- Approximation: glass blur's sequential pixel swaps are replaced by one random displacement (|d| <= 2) per pixel
  and iteration, vectorized.
- Split: even positions of the published order -> tuning half, odd positions -> confirmation half.
- Streams: for each half (7 corruptions c[0..6]), 3 streams; stream i applies c[(3i + d) mod 7] to val domain d
  (val order, the val images keep their own corruption underneath). Every corruption appears at least once.

  python suite.py            # writes cache/suite_{tune,conf}{0,1,2}.pt (uint8 x in val order, y, names)
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from scipy.ndimage import affine_transform, convolve, gaussian_filter, map_coordinates
from scipy.ndimage import zoom as scizoom

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.environ.get("HW2_DEV_CACHE", os.path.join(HERE, "cache")) + "/"
PUBLISHED = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
             "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
             "jpeg_compression"]
EXCLUDED = {"frost"}  # needs the frost photos shipped with the original code
SEVERITY = 3


# ---------------------------------------------------------------- helpers (float HxWx3 in [0, 1])
def disk(radius, alias_blur, rng=None):
    L = np.arange(-8, 9) if radius <= 8 else np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    k = ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float32)
    k /= k.sum()
    return gaussian_filter(k, alias_blur)  # cv2.GaussianBlur(ksize 3) in the original; same role


def clipped_zoom(img, z):
    h = img.shape[0]
    ch = int(np.ceil(h / z))
    top = (h - ch) // 2
    out = scizoom(img[top:top + ch, top:top + ch], (z, z) + (1,) * (img.ndim - 2), order=1)
    t = (out.shape[0] - h) // 2
    return out[t:t + h, t:t + h]


def shift(x, dx, dy):
    h, w = x.shape[:2]
    p = max(abs(dx), abs(dy))
    xp = np.pad(x, ((p, p), (p, p)) + ((0, 0),) * (x.ndim - 2), mode="edge")
    return xp[p - dy:p - dy + h, p - dx:p - dx + w]


def motion_blur_np(x, radius, sigma, angle):
    width = radius * 2 + 1
    k = np.exp(-np.arange(width) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    py, px = width * math.sin(math.radians(angle)), width * math.cos(math.radians(angle))
    hyp = math.hypot(py, px)
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(i * py / hyp - 0.5)
        dx = -math.ceil(i * px / hyp - 0.5)
        if abs(dy) >= x.shape[0] or abs(dx) >= x.shape[1]:
            break
        out += k[i] * shift(x, dx, dy)
    return out


def plasma_fractal(rng, mapsize=256, wibbledecay=3.0):
    a = np.empty((mapsize, mapsize), dtype=np.float64)
    a[0, 0] = 0
    step, wibble = mapsize, 100.0

    def wmean(arr):
        return arr / 4 + wibble * rng.uniform(-wibble, wibble, arr.shape)

    while step >= 2:
        corner = a[0:mapsize:step, 0:mapsize:step]
        sq = corner + np.roll(corner, -1, 0)
        sq += np.roll(sq, -1, 1)
        a[step // 2:mapsize:step, step // 2:mapsize:step] = wmean(sq)
        dr = a[step // 2:mapsize:step, step // 2:mapsize:step]
        ul = a[0:mapsize:step, 0:mapsize:step]
        a[0:mapsize:step, step // 2:mapsize:step] = wmean(dr + np.roll(dr, 1, 0) + ul + np.roll(ul, -1, 1))
        a[step // 2:mapsize:step, 0:mapsize:step] = wmean(dr + np.roll(dr, 1, 1) + ul + np.roll(ul, -1, 0))
        step //= 2
        wibble /= wibbledecay
    a -= a.min()
    return a / a.max()


# ---------------------------------------------------------------- the corruptions, severity index s = SEVERITY - 1
def gaussian_noise(x, rng, s):
    return x + rng.normal(size=x.shape, scale=[.08, .12, .18, .26, .38][s])


def shot_noise(x, rng, s):
    c = [60, 25, 12, 5, 3][s]
    return rng.poisson(x * c) / c


def impulse_noise(x, rng, s):
    amount = [.03, .06, .09, .17, .27][s]
    out = x.copy()
    m = rng.random(x.shape)
    out[m < amount / 2] = 0
    out[(m >= amount / 2) & (m < amount)] = 1
    return out


def defocus_blur(x, rng, s):
    r, a = [(3, .1), (4, .5), (6, .5), (8, .5), (10, .5)][s]
    k = disk(r, a)
    return np.stack([convolve(x[..., c], k, mode="reflect") for c in range(3)], -1)


def glass_blur(x, rng, s):
    sigma, d, iters = [(.7, 1, 2), (.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][s]
    x = gaussian_filter(x, (sigma, sigma, 0))
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(iters):  # approximation: one random displacement per pixel instead of sequential swaps
        dy, dx = rng.integers(-d, d + 1, (2, h, w))
        x = x[np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    return gaussian_filter(x, (sigma, sigma, 0))


def motion_blur(x, rng, s):
    r, sg = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][s]
    return motion_blur_np(x, r, sg, rng.uniform(-45, 45))


def zoom_blur(x, rng, s):
    zs = [np.arange(1, 1.11, .01), np.arange(1, 1.16, .01), np.arange(1, 1.21, .02), np.arange(1, 1.26, .02),
          np.arange(1, 1.31, .03)][s]
    out = np.zeros_like(x)
    for z in zs:
        out += clipped_zoom(x, z)
    return (x + out) / (len(zs) + 1)


def snow(x, rng, s):
    c = [(.1, .3, 3, .5, 10, 4, .8), (.2, .3, 2, .5, 12, 4, .7), (.55, .3, 4, .9, 12, 8, .7),
         (.55, .3, 4.5, .85, 12, 8, .65), (.55, .3, 2.5, .85, 12, 12, .55)][s]
    layer = rng.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    layer = clipped_zoom(layer[..., None], c[2])
    layer[layer < c[3]] = 0
    layer = np.clip(motion_blur_np(np.clip(layer, 0, 1), c[4], c[5], rng.uniform(-135, -45)), 0, 1)
    gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)
    return x + layer + np.rot90(layer, k=2)


def fog(x, rng, s):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][s]
    mx = x.max()
    h, w = x.shape[:2]
    x = x + c[0] * plasma_fractal(rng, wibbledecay=c[1])[:h, :w][..., None]
    return x * mx / (mx + c[0])


def brightness(x, rng, s):
    hsv = rgb_to_hsv(np.clip(x, 0, 1))
    hsv[..., 2] = np.clip(hsv[..., 2] + [.1, .2, .3, .4, .5][s], 0, 1)
    return hsv_to_rgb(hsv)


def contrast(x, rng, s):
    m = x.mean(axis=(0, 1), keepdims=True)
    return (x - m) * [.4, .3, .2, .1, .05][s] + m


def elastic_transform(x, rng, s):
    # make_imagenet_c.py constants for 224 px (244 * ...), scaled to this image size
    scale = x.shape[0] / 224
    alpha, sigma, aff = [(244 * 2, 244 * .7, 244 * .1), (244 * 2, 244 * .08, 244 * .2), (244 * .05, 244 * .01, 244 * .02),
                         (244 * .07, 244 * .01, 244 * .02), (244 * .12, 244 * .01, 244 * .02)][s]
    alpha, sigma, aff = alpha * scale, sigma * scale, aff * scale
    h, w = x.shape[:2]
    cs, ss = np.float32([h, w]) // 2, min(h, w) // 3
    p1 = np.float32([cs + ss, [cs[0] + ss, cs[1] - ss], cs - ss])
    p2 = p1 + rng.uniform(-aff, aff, size=p1.shape).astype(np.float32)
    A = np.c_[p2, np.ones(3)]  # inverse map: output point -> input point (solve p1 = M @ [p2, 1])
    M = np.linalg.solve(A, p1).T  # 2x3, (row, col) coordinates
    x = np.stack([affine_transform(x[..., c], M[:, :2], offset=M[:, 2], mode="mirror", order=1) for c in range(3)], -1)
    dx = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma, mode="reflect", truncate=3) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma, mode="reflect", truncate=3) * alpha
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([map_coordinates(x[..., c], [yy + dy, xx + dx], order=1, mode="reflect") for c in range(3)], -1)


def pixelate(x, rng, s):
    h, w = x.shape[:2]
    c = [.6, .5, .4, .3, .25][s]
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    im = im.resize((int(w * c), int(h * c)), Image.BOX).resize((w, h), Image.BOX)
    return np.asarray(im).astype(np.float32) / 255


def jpeg_compression(x, rng, s):
    buf = BytesIO()
    Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=[25, 18, 15, 10, 7][s])
    return np.asarray(Image.open(buf)).astype(np.float32) / 255


FN = {n: globals()[n] for n in PUBLISHED if n not in EXCLUDED}
SUITE = [n for n in PUBLISHED if n not in EXCLUDED]
HALVES = {"tune": [n for i, n in enumerate(PUBLISHED) if i % 2 == 0 and n not in EXCLUDED],
          "conf": [n for i, n in enumerate(PUBLISHED) if i % 2 == 1 and n not in EXCLUDED]}


def corrupt_one(args):
    name, idx, img = args
    rng = np.random.default_rng((PUBLISHED.index(name) + 1) * 100003 + idx)
    x = img.astype(np.float32) / 255
    y = np.clip(FN[name](x, rng, SEVERITY - 1), 0, 1)
    return (y * 255 + 0.5).astype(np.uint8)


def main():
    d = torch.load(SP + "val.pt")
    X = d["x"].permute(0, 2, 3, 1).numpy()  # N, H, W, 3 uint8, val stream order
    y = d["y"]
    bounds = np.load(os.path.join(HERE, "val_bounds.npy"))
    dom = np.searchsorted(bounds, np.arange(len(X)), side="right") - 1
    with ProcessPoolExecutor(int(os.environ.get("SUITE_WORKERS", 12))) as ex:
        for half, names in HALVES.items():
            for i in range(3):
                assign = [names[(3 * i + k) % len(names)] for k in range(3)]
                jobs = [(assign[dom[j]], j, X[j]) for j in range(len(X))]
                out = np.stack(list(ex.map(corrupt_one, jobs, chunksize=32)))
                torch.save(dict(x=torch.from_numpy(out).permute(0, 3, 1, 2).contiguous(), y=y, names=assign),
                           SP + f"suite_{half}{i}.pt")
                print(half, i, assign, flush=True)


if __name__ == "__main__":
    main()
