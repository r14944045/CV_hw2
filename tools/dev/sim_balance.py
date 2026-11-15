"""Simulate online per-domain balancing on recorded LAME inputs (no self-training feedback).

At batch t of a domain (true or detected bounds), estimate column scalings beta by Sinkhorn on the domain's
history (recorded LAME outputs of previous batches + this batch's LAME input), apply beta^s(k) to this batch's
probabilities, re-run LAME, predict. s(k) = min(1, n_seen / ramp) ramps the correction in.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../starter"))
from tta.lame import lame  # noqa: E402

SP = os.environ.get("HW2_DEV_CACHE", os.path.join(HERE, "cache")) + "/"


def col_scale(P, iters=50, power=1.0):
    """beta such that rows of P*beta (renormalised) have uniform column sums."""
    P = P.double().clamp_min(1e-12) ** power
    n, C = P.shape; beta = torch.ones(C, dtype=P.dtype)
    for _ in range(iters):
        Q = P * beta; Q = Q / Q.sum(1, keepdim=True)
        beta = beta * (n / C) / Q.sum(0).clamp_min(1e-9)
        beta = beta / beta.mean()
    return beta.float()


def simulate(rec, bounds, ramp=640.0, smax=1.0, hist_src="z", knn=5, cap=None):
    prob, feats, z = rec["prob"], rec["feats"].float(), rec["z"]
    N = len(prob); preds = np.empty(N, int); B = 64
    starts = set(int(b) for b in bounds[:-1])
    dom_start = 0
    for k in range(0, N, B):
        # domain start = last bound <= k (a mid-batch boundary counts for the next batch)
        ds = max(b for b in bounds[:-1] if b <= k)
        hist = (z if hist_src == "z" else prob)[ds:k]
        P = prob[k:k + B]
        H = torch.cat([hist, P]) if len(hist) else P
        n_seen = len(H)
        s = min(smax, n_seen / ramp)
        if s > 0 and n_seen >= 64:
            beta = col_scale(H)
            if cap:
                beta = beta.clamp(1 / cap, cap)
            P2 = P * beta.pow(s); P2 = P2 / P2.sum(1, keepdim=True)
        else:
            P2 = P
        zz = lame(P2, feats[k:k + B], knn)
        preds[k:k + B] = zz.argmax(1).numpy()
    return preds


if __name__ == "__main__":
    y = torch.load(SP + "val.pt")["y"].numpy()
    vb = np.load(os.path.join(HERE, "val_bounds.npy")); tb = np.load(os.path.join(HERE, "test_bounds.npy"))
    rv = torch.load(SP + "rec_sp2_val/rec.pt"); rt = torch.load(SP + "rec_sp2_test/rec.pt")
    base_v = rv["z"].argmax(1).numpy(); base_t = rt["z"].argmax(1).numpy()
    lame_v = np.concatenate([lame(rv["prob"][k:k+64], rv["feats"][k:k+64].float()).argmax(1).numpy() for k in range(0, len(base_v), 64)])
    print("val base", (base_v == y).mean(), "re-LAME check", (lame_v == base_v).mean())
    import pandas as pd
    off = pd.read_csv(SP + "probe_sinkhorn/test_pred.csv")["label"].to_numpy()
    for ramp in [320, 640, 1280]:
        for smax in [0.5, 1.0]:
            pv = simulate(rv, vb, ramp, smax); pt = simulate(rt, tb, ramp, smax)
            agree_off = ((pt != base_t) & (pt == off)).sum() / max(1, (pt != base_t).sum())
            print(f"ramp {ramp:5d} smax {smax}: val {(pv == y).mean():.4f} (chg {(pv != base_v).mean():.3f})  test chg {(pt != base_t).mean():.3f}, of which agree w/ offline {agree_off:.2f}")
