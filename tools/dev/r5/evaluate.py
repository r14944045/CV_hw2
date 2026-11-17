"""Round-5 evaluator: a MixTTAMS subclass (or MixTTAMS itself) on the proxy at 4 boundary offsets + the 7 val
orders, and optionally a label-free check on the test stream. Every run is appended to r5/results.jsonl.

  cd tools/dev && ./gpu_slot.sh python r5/evaluate.py --cls r5.my_idea:MyMethod --kw '{"x": 1}' --tag my_idea_x1
  --cls   module:Class, importable from tools/dev (r5.<file>:<Class>) or from starter (tta.mixtta_ms:MixTTAMS)
  --kw    JSON overrides on top of BASE (the mixtta_ms224sp2cpr settings)
  --seed  RNG seed (default 0)
  --quick smoke test: proxy offset 0 + the original val order only (about 1 min)
  --test  also run the test stream: time, per-domain imbalance floor, largest class share, tree-frog excess

Reference (mixtta_ms224sp2cpr = MixTTAMS with BASE, this evaluator):
  seed 0: proxy offsets 0/16/32/48 = 86.39 86.24 85.96 86.18, mean 86.19; val7 94.00 (scan/wave/fish 94.59 94.10 93.14)
  seed 1: proxy mean 86.09; val7 94.14.  Differences under ~0.3 on the proxy mean are seed noise.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
sys.path.insert(0, DEV)
sys.path.insert(0, os.path.join(DEV, "../../starter"))
import fe  # noqa: E402

BASE = dict(shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0, split_shift=True, refine_batches=2)
OFFSETS = (0, 16, 32, 48)


def factory(cls_spec, kw):
    mod, name = cls_spec.split(":")
    cls = getattr(importlib.import_module(mod), name)
    holder = {}

    def fac(model):
        holder["m"] = cls(model, **kw)
        return holder["m"]
    return fac, holder


def test_check(fac, holder, seed):
    floor = lambda q: np.clip(np.bincount(q, minlength=20) - len(q) / 20, 0, None).sum() / len(q)  # noqa: E731
    torch.cuda.synchronize(); t0 = time.time()
    p = fe.run(fac, "test", seed=seed)
    sec = time.time() - t0
    tb = fe.data("test")["bounds"]
    doms = list(zip(tb[:-1], tb[1:]))
    late = p[tb[9]:]
    out = dict(seconds=round(sec, 1),
               floor_all=round(float(sum(floor(p[a:b]) * (b - a) for a, b in doms) / len(p)), 4),
               floor_dom=[round(float(floor(p[a:b])), 3) for a, b in doms],
               maxshare_dom=[round(float(np.bincount(p[a:b], minlength=20).max() / (b - a)), 3) for a, b in doms],
               maxshare_late=round(float(np.bincount(late, minlength=20).max() / len(late)), 3),
               frog_excess=int((p == 7).sum() - len(p) // 20),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    m = holder.get("m")
    if m is not None and hasattr(m, "shifts"):
        out["shifts"] = list(m.shifts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="tta.mixtta_ms:MixTTAMS")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    kw = dict(BASE, **json.loads(a.kw))
    fac, holder = factory(a.cls, kw)
    t0 = time.time()
    rec = dict(tag=a.tag, cls=a.cls, kw=kw, seed=a.seed, quick=a.quick)
    offs = (0,) if a.quick else OFFSETS
    px = []
    for o in offs:
        r = fe.proxy(fac, mix=0.5, verbose=False, offset=o, seed=a.seed)
        px.append(round(float(r["mean"]), 2))
        if o == 0:
            rec["proxy_per_corruption_off0"] = [float(v) for v in r["per"]]
    rec["proxy_offsets"] = dict(zip(map(str, offs), px))
    rec["proxy_mean"] = round(float(np.mean(px)), 2)
    v = fe.val7(fac, verbose=False, seeds=(a.seed,), orders=["orig"] if a.quick else None)
    rec["val7"] = round(float(v["mean"]), 2)
    rec["val_dom"] = [float(x) for x in v["dom"]]
    if a.test:
        rec["test"] = test_check(fac, holder, a.seed)
    rec["eval_seconds"] = round(time.time() - t0)
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
