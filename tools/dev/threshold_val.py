"""Choose the domain-shift threshold (MixTTAMS step 0, `shift_thr`) from val only.

The detector's score for a batch is its conv1-statistics KL divergence divided by the median over the last 8 batches
of the current domain (`ratio`). A batch fires when ratio > shift_thr.

Streams: the 7 val stream orders (fe.val_orders) x in-batch offsets 0, 16, 32, 48 (the first k images of the
stream are dropped, so each boundary falls at 4 different rows of its batch): 28 streams. Domains: val_bounds.npy.

Rule (fixed before looking at the numbers):
  - boundary event: a domain boundary at stream position q. The detector may fire on the batch that contains q or
    on the next one; the event's ratio is the larger of the two.
  - within-domain ratios W: every other batch that has a ratio (the detector needs 3 batches of history).
  - shift_thr = ceil(2 * max(W)): twice the largest ratio ever seen inside a val domain, so that no val batch inside
    a domain comes within a factor 2 of firing. Check: every boundary event must exceed it.
    (Fallback if that check fails: the geometric mean of max(W) and the smallest event ratio.)
  - The ratios depend on earlier detections (a detection resets the domain's history), so the streams are rerun at
    the chosen threshold until the value no longer changes.

  python threshold_val.py [start_thr]

Result (run from start_thr 10, the earlier hand-set value; 28 streams, 596 within-domain batches, 56 boundary events):
  within-domain ratio max 8.12 (99th percentile 4.19, median 0.69); boundary events min 9.2, median 34.2.
  The primary rule (ceil(2 * 8.12) = 17) fails the check, so the fallback applies:
  10 -> 9.5 -> 8.8 -> 8.7 -> 8.7, with 0 missed boundaries and 0 false alarms at every step. shift_thr = 8.7.
  Val separates the two sides only narrowly (8.12 against 9.2), so the margin is thin on both sides.
  The value drifts down because once the detector fires on the mixed boundary batch, the next batch has no
  ratio, which lowers that event's recorded value.
  Accuracy at 8.7 against 10: harness val is identical (95.60 %); val7 (7 orders, seeds 0 and 1) is 94.68 against
  94.70 %, within noise.
"""
import math
import sys

import numpy as np

import fe
from tta.mixtta_ms import MixTTAMS

CFG = dict(alpha_min=0.1, prior_lambda=2.0, split_shift=True, refine_batches=2, stage_lr=(3, 3, 3, 1, 0.1))
OFFSETS = (0, 16, 32, 48)


class Recorder(MixTTAMS):
    """MixTTAMS that records the detector ratio of every batch (None while the domain has < 3 batches)."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.ratios = []

    def _shift_score(self, x):
        d = super()._shift_score(x)
        hist = self.dist_hist[-8:]  # same reference as _tracking_rate
        ref = sorted(hist)[len(hist) // 2] if len(hist) >= 3 else None
        self.ratios.append(None if ref is None else d / max(ref, 1e-4))
        return d


def measure(thr, cfg=None):
    cfg = CFG if cfg is None else cfg
    b = fe.data("val")["bounds"]
    dom = np.zeros(b[-1], int); dom[b[1]:b[2]] = 1; dom[b[2]:] = 2
    W, E, misses, false_alarms, n_streams = [], [], 0, 0, 0
    for name, order in fe.val_orders().items():
        for off in OFFSETS:
            idx = order[off:]
            holder = {}

            def fac(model):
                holder["m"] = Recorder(model, **{**cfg, "shift_thr": thr})
                return holder["m"]

            fe.run(fac, "val", idx)
            m, d = holder["m"], dom[idx]
            q = np.flatnonzero(np.diff(d)) + 1  # first position of each new domain
            event_batches = {int(p // 64): i for i, p in enumerate(q)}
            event_batches.update({int(p // 64) + 1: i for i, p in enumerate(q)})
            ev = [[] for _ in q]
            for t, r in enumerate(m.ratios):
                if t in event_batches:
                    ev[event_batches[t]].append(r)
                elif r is not None:
                    W.append(r)
            for e in ev:
                E.append(max([r for r in e if r is not None], default=float("nan")))
            fired = set(m.shifts)
            misses += sum(1 for p in q if not ({p // 64, p // 64 + 1} & fired))
            false_alarms += sum(1 for t in fired if t not in event_batches)
            n_streams += 1
    return np.array(W), np.array(E), misses, false_alarms, n_streams


def rule(W, E):
    thr = math.ceil(2 * W.max())
    if np.nanmin(E) > thr:
        return float(thr), "ceil(2 * max within-domain)"
    return round(math.sqrt(W.max() * np.nanmin(E)), 1), "fallback: geometric mean"


def derive(cfg=None, thr=10.0, verbose=True):
    """Iterate the rule to a fixed point for the tracking settings in cfg; returns (shift_thr, log lines)."""
    log = []
    for it in range(8):
        W, E, misses, fa, n = measure(thr, cfg)
        new, how = rule(W, E)
        line = (f"[run at thr {thr:g}] {n} streams, {len(W)} within-domain batches, {len(E)} boundary events | "
                f"within: max {W.max():.2f}, 99th pct {np.percentile(W, 99):.2f}, median {np.median(W):.2f} | "
                f"events: min {np.nanmin(E):.1f}, median {np.nanmedian(E):.1f} | misses {misses}, false alarms {fa} "
                f"-> rule gives {new:g} ({how})")
        log.append(line)
        if verbose:
            print(line, flush=True)
        if new == thr:
            break
        thr = new
    return thr, log


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    for it in range(6):
        W, E, misses, fa, n = measure(thr)
        new, how = rule(W, E)
        print(f"[run at thr {thr:g}] {n} streams, {len(W)} within-domain batches, {len(E)} boundary events | "
              f"within: max {W.max():.2f}, 99th pct {np.percentile(W, 99):.2f}, median {np.median(W):.2f} | "
              f"events: min {np.nanmin(E):.1f}, median {np.nanmedian(E):.1f} | misses {misses}, false alarms {fa} "
              f"-> rule gives {new:g} ({how})", flush=True)
        if new == thr:
            break
        thr = new
    print(f"shift_thr = {thr:g}")


if __name__ == "__main__":
    main()
