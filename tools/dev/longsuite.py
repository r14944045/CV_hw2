"""Long suite stream: the six ImageNet-C suite streams back to back (tune0, conf0, tune1, conf1, tune2, conf2).
12,672 images, 198 batches (about the test stream's 192), 18 domain segments; every image appears six times, each time
under a different corruption, so there are no identical repeats. Val images and val labels only.

Purpose: a check for components that feed predictions back across batches (local class prior, T3A), which the short
val-based metrics could not see. Validity test before use: it must place mixtta_r8 (Kaggle 0.82682) far below
mixtta_valsel (0.92675) and mixtta_local_fixed (0.92789).

  python longsuite.py name '<json cfg>' [seed]     # appends to longsuite_results.jsonl

Result of the validity test (2026-11-20 ~01:35): NOT valid. Every setting collapses onto one class within a few of the
compound (val corruption + ImageNet-C severity 3) segments, and the ranking is inverted against Kaggle:
floor 51.4, valsel 36.6, s5 24.7, ms224 23.0, r8 22.8, local_fixed 21.1 (Kaggle 0.912, 0.927, 0.922, 0.916, 0.827,
0.928). It is not used as a check.
"""
import json
import os
import sys

import numpy as np
import torch

import fe
from tta.mixtta_ms import MixTTAMS

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["tune0", "conf0", "tune1", "conf1", "tune2", "conf2"]


def stream():
    xs = [torch.load(fe.SP + f"suite_{n}.pt")["x"] for n in ORDER]
    y = fe.data("val")["y"].numpy()
    b = np.load(os.path.join(HERE, "val_bounds.npy"))
    seg = np.concatenate([np.searchsorted(b, np.arange(len(y)), side="right") - 1 + 3 * i for i in range(len(ORDER))])
    return torch.cat(xs), np.tile(y, len(ORDER)), seg


def evaluate(cfg, seed=0):
    X, y, seg = stream()
    p = fe.run(lambda model: MixTTAMS(model, **cfg), "val", np.arange(len(X)), seed=seed, x=X)
    ok = p == y
    per_acc = [float(ok[seg == s].mean()) for s in range(seg.max() + 1)]
    per_share = [float(np.bincount(p[seg == s], minlength=20).max() / (seg == s).sum()) for s in range(seg.max() + 1)]
    third = len(y) // 3
    return dict(acc=100 * float(ok.mean()), first_third=100 * float(ok[:third].mean()),
                last_third=100 * float(ok[-third:].mean()), min_seg=100 * min(per_acc),
                max_share=max(per_share), seg_acc=[round(100 * a, 1) for a in per_acc],
                seg_share=[round(s, 3) for s in per_share])


if __name__ == "__main__":
    name, cfg = sys.argv[1], json.loads(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    r = evaluate(cfg, seed)
    with open(os.path.join(HERE, "longsuite_results.jsonl"), "a") as f:
        f.write(json.dumps(dict(name=name, cfg=cfg, seed=seed, **r)) + "\n")
    print(f"{name:14s} acc {r['acc']:.2f} first/last third {r['first_third']:.2f}/{r['last_third']:.2f} "
          f"min seg {r['min_seg']:.1f} max share {r['max_share']:.3f}", flush=True)
