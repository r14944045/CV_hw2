"""Offline analysis: how much is per-domain class balance worth? (oracle domain bounds, offline)."""
import os
import sys
import numpy as np
import torch

SP = os.environ.get("HW2_DEV_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")) + "/"


def sinkhorn(P, iters=200, power=1.0):
    """Rescale columns so that each class gets n/C mass; rows sum to 1. Returns balanced soft assignment."""
    Q = P.double().clamp_min(1e-12) ** power
    n, C = Q.shape
    for _ in range(iters):
        Q = Q / Q.sum(0, keepdim=True) * (n / C)
        Q = Q / Q.sum(1, keepdim=True)
    return Q


def floor(pred, C=20):
    n = len(pred); cnt = np.bincount(pred, minlength=C)
    return np.clip(cnt - n / C, 0, None).sum() / n


if __name__ == "__main__":
    tag = sys.argv[1]
    r = torch.load(SP + f"rec_{tag}.pt")
    y = torch.load(SP + "val.pt")["y"].numpy(); vb = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_bounds.npy"))
    v = r["val"]
    for key in ["prob", "z"]:
        P = v[key]
        accs = []
        for a, b in zip(vb[:-1], vb[1:]):
            Pd = P[a:b]; yd = y[a:b]
            base = (Pd.argmax(1).numpy() == yd).mean()
            bal = (sinkhorn(Pd).argmax(1).numpy() == yd).mean()
            accs.append((base, bal, floor(Pd.argmax(1).numpy()), b - a))
        w = np.array([t[3] for t in accs])
        print(f"val {key}: per-domain base/balanced: " + "  ".join(f"{t[0]:.3f}/{t[1]:.3f} (floor {t[2]:.3f})" for t in accs),
              f"| overall {np.average([t[0] for t in accs], weights=w):.4f} -> {np.average([t[1] for t in accs], weights=w):.4f}")
