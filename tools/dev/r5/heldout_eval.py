"""Evaluate methods on the held-out proxy (proxy_heldout.py: 6 unseen corruption types, full strength).

  short: the 4 streams run separately (33 batches each, model reset between streams)
  long:  the 4 streams back to back (132 batches, 12 domains, no reset): checks drift on a longer stream
Reports mean accuracy, accuracy per corruption, and the largest class share in any domain (collapse check).
Results are appended to r5/heldout_results.jsonl.

  python r5/heldout_eval.py --methods mixtta_ms224sp2cpr,mixtta_ms224sp2cprl3 --seeds 0,1
  python r5/heldout_eval.py --cls r5.module:Class --kw '{...}' --tag name     (MixTTAMS-style class + sp2cpr base kwargs)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, ".."))
import fe  # noqa: E402
from r5.evaluate import BASE, factory  # noqa: E402


def evaluate(fac, seed):
    d = fe.data("heldout"); y, dom = d["y"].numpy(), d["dom"].numpy(); L = d["stream_len"]; names = d["names"]
    out = {}
    for mode in ("short", "long"):
        pred = np.zeros(len(y), int)
        if mode == "short":
            for s in range(4):
                idx = np.arange(s * L, (s + 1) * L); pred[idx] = fe.run(fac, "heldout", idx, seed=seed)
        else:
            pred[:] = fe.run(fac, "heldout", np.arange(len(y)), seed=seed)
        ok = pred == y
        seg = [(s * L + a, s * L + b) for s in range(4) for a, b in zip([0, 776, 1500], [776, 1500, 2112])]
        maxshare = max(np.bincount(pred[a:b], minlength=20).max() / (b - a) for a, b in seg)
        out[mode] = dict(acc=round(100 * ok.mean(), 2), maxshare=round(float(maxshare), 3),
                         per={n: round(100 * ok[dom == c].mean(), 1) for c, n in enumerate(names)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="")
    ap.add_argument("--cls", default=None)
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    jobs = [(m, fe.M(m), m) for m in a.methods.split(",") if m]
    if a.cls:
        fac, _ = factory(a.cls, dict(BASE, **json.loads(a.kw)))
        jobs.append((a.tag or a.cls, fac, a.cls))
    for tag, fac, src in jobs:
        for seed in map(int, a.seeds.split(",")):
            r = evaluate(fac, seed)
            rec = dict(tag=tag, src=src, kw=json.loads(a.kw) if a.cls and tag == (a.tag or a.cls) else None, seed=seed, **r)
            with open(os.path.join(HERE, "heldout_results.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{tag:28s} s{seed} | short {r['short']['acc']:6.2f} max {r['short']['maxshare']:.3f} | long {r['long']['acc']:6.2f} "
                  f"max {r['long']['maxshare']:.3f} | short per: " + " ".join(f"{k} {v}" for k, v in r['short']['per'].items()), flush=True)


if __name__ == "__main__":
    main()
