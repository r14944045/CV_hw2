"""Where inside a domain are a method's errors? Proxy accuracy binned by batch position relative to the domain
boundaries, over the 4 boundary offsets (0, 16, 32, 48 images dropped from each stream).

  straddle: batches that contain a boundary; next1 / next2: the first / second full batch after a boundary;
  first: the first 2 batches of a stream; rest: everything else.

  python bins.py <registered method> [<method> ...]
"""
import sys

import numpy as np

import fe

OFFSETS = (0, 16, 32, 48)


def bins(correct, dom_seq):
    n = len(correct); bset = np.flatnonzero(np.diff(dom_seq)) + 1
    cats = {k: [] for k in ("first", "straddle", "next1", "next2", "rest")}
    for b in range((n + 63) // 64):
        s, e = b * 64, min(n, (b + 1) * 64); ok = correct[s:e]
        if b < 2:
            cats["first"].append(ok); continue
        if any(s < q < e for q in bset):
            cats["straddle"].append(ok); continue
        prev = [q for q in bset if q <= s]
        k = (s - prev[-1]) // 64 if prev else None
        cats["next1" if k == 0 else "next2" if k == 1 else "rest"].append(ok)
    return {k: np.concatenate(v) for k, v in cats.items() if v}


for name in sys.argv[1:]:
    d = fe.data("proxy0.5"); y, dom = d["y"].numpy(), d["dom"].numpy(); L = d["stream_len"]
    agg = {}
    for off in OFFSETS:
        for s in range(4):
            idx = np.arange(s * L + off, (s + 1) * L)
            p = fe.run(fe.M(name), "proxy0.5", idx)
            for k, v in bins(p == y[idx], dom[idx]).items():
                agg.setdefault(k, []).append(v)
    out = {k: (100 * np.concatenate(v).mean(), sum(len(x) for x in v)) for k, v in agg.items()}
    print(name, " | ".join(f"{k} {a:.2f} (n={n})" for k, (a, n) in out.items()), flush=True)
