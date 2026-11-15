"""Simulate a temporal label prior on recorded LAME inputs: prob * (h_prev + eps)^mu, h_prev = mean LAME output
of the previous batch (reset at domain bounds). No self-training feedback."""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../starter"))
from tta.lame import lame  # noqa
SP = os.environ.get("HW2_DEV_CACHE", os.path.join(HERE, "cache")) + "/"

def simulate(rec, bounds, mu, eps, use_bounds=True, decay=0.0):
    prob, feats = rec["prob"], rec["feats"].float(); N = len(prob); out = np.empty(N, int); h = None
    for k in range(0, N, 64):
        if use_bounds and any(k <= b < k + 64 for b in bounds[1:-1]) and h is not None:
            pass  # boundary inside this batch: keep h (unknown online), reset after
        P = prob[k:k+64]
        if h is not None and mu > 0:
            P = P * (h + eps).pow(mu); P = P / P.sum(1, keepdim=True)
        z = lame(P, feats[k:k+64]); out[k:k+64] = z.argmax(1).numpy()
        hz = z.mean(0)
        h = hz if (h is None or decay == 0) else (1 - decay) * hz + decay * h
        if use_bounds and any(k < b <= k + 64 for b in bounds[1:-1]): h = None
    return out

y = torch.load(SP + "val.pt")["y"].numpy(); vb = np.load(os.path.join(HERE, "val_bounds.npy")); tb = np.load(os.path.join(HERE, "test_bounds.npy"))
rv = torch.load(SP + "rec_sp2_val/rec.pt"); rt = torch.load(SP + "rec_sp2_test/rec.pt")
bv = rv["z"].argmax(1).numpy(); bt = rt["z"].argmax(1).numpy()
print("val base", (bv == y).mean())
for mu in [0.1, 0.2, 0.3, 0.5, 1.0]:
    for eps in [0.02, 0.1]:
        for decay in [0.0, 0.5]:
            pv = simulate(rv, vb, mu, eps, decay=decay); pt = simulate(rt, tb, mu, eps, decay=decay)
            print(f"mu {mu} eps {eps} decay {decay}: val {(pv==y).mean():.4f} (chg {(pv!=bv).mean():.3f}, fixed {((pv==y)&(bv!=y)).sum()} broke {((pv!=y)&(bv==y)).sum()})  test chg {(pt!=bt).mean():.3f}")
