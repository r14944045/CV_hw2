"""Val-only re-pick of the MixTTA-MS settings. Protocol fixed on 2026-11-19 18:30 CST, before any of its runs.

Why: several settings of the Kaggle-best method (mixtta_ms224sp2cprl3) were chosen with help from an analysis of the
unlabelled test stream outside the harness (offline test segmentation, label-free test checks, a proxy whose
corruptions were modelled on test images). This script re-chooses them from val alone. The candidate ideas still come
from that earlier work; only the selection here is val-only. No test data is read. The test stream is used once, by
the official harness, for the final setting only.

Fixed, not re-picked (each was chosen on val in the original development):
  MixTTAMS core: tracking rate alpha 0.4, source weight w 0.5, flip + 2 random shifts, LAME k 5, input sizes
  (160, 224), Adam lr 1e-3 on BN scale/shift, self-training at 160 px, prior pseudo-count 2000, split_min 4.

Metric (primary), V: mean accuracy over the 7 val stream orders x 4 in-batch offsets (the first 0, 16, 32, 48 images
  of the stream dropped, so each boundary falls at 4 rows of its batch) x seeds 0 and 1 (56 runs).
Collapse veto, L: one long stream, the 7 val orders back to back (14,784 images, 231 batches; every image appears 7
  times, so it is not used for ranking), seed 0. Veto a candidate if, on the last 2,112 images, one class takes more
  than 10 % of the predictions (val domains are class-balanced, 5 % per class), or the accuracy is more than 5 pt below
  that on the first 2,112 images.
Margin m: max(0.3, |V(seed 0) - V(seed 1)|) of the simplest stage-1 candidate, fixed after stage 1.
Shift threshold: every candidate with shift detection gets its own threshold from the threshold_val.py rule for its
  tracking settings (alpha_min, split), iterated from 8.7. The rule's ratios depend only on conv1 and the 160 px
  tracker, so it runs with self-training off and one size (checked: same statistics and 8.7 for the old setting).

Selection, stage by stage in the order below, each starting from the previous picks:
  1. drop vetoed candidates; 2. keep those with V >= best V - m; 3. pick the simplest (first in the listed order).

Stages and candidates (simplest first):
  S1 self-training core, (steps, LAME T): (2,1) (2,2) (2,3) (3,1) (3,2) (3,3) (4,1) (4,2) (4,3)
  S2 tracking: no reset (fixed rate 0.4); shift reset with rate floor alpha_min 0.4, 0.2, 0.1, 0.05
  S3 in-batch split of the shift batch (only if S2 resets): off, on
  S4 second pass on the first R batches of a domain (only if S2 resets): R = 0, 1, 2, 3
  S5 class prior lambda: 0, 1, 2, 3
  S6 per-stage learning rates (stem, layer1, layer2, layer3, layer4): none; then early x {2, 3, 4} (stem to layer2),
     layer3 x {1, 2}, layer4 x {0.25, 0.1}, in that nesting order
  S7 adaptive LAME temperature lame_q: none, 0.4, 0.3, 0.2

The final setting is whatever S7 leaves. It is not revised after a Kaggle score comes back.

  python repick.py            # runs (resumable; results in repick_results.jsonl, decisions in repick_log.txt)
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "repick_results.jsonl")
LOG = os.path.join(HERE, "repick_log.txt")
OFFSETS, SEEDS = (0, 16, 32, 48), (0, 1)
WORKERS = 2  # the GPU is shared with other users' jobs
TRACK_KEYS = ("alpha_min", "split_shift")


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def load():
    out = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            r = json.loads(line)
            out[(r["kind"], r["key"])] = r
    return out


def append(rec):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------- workers (run in subprocesses)
def work_eval(cfg):
    import numpy as np
    import fe
    from tta.mixtta_ms import MixTTAMS
    y = fe.data("val")["y"].numpy()
    fac = lambda model: MixTTAMS(model, **cfg)  # noqa: E731
    per_seed = {s: [] for s in SEEDS}
    for order in fe.val_orders().values():
        for off in OFFSETS:
            idx = order[off:]
            for s in SEEDS:
                p = fe.run(fac, "val", idx, seed=s)
                per_seed[s].append(float((p == y[idx]).mean()))
    allacc = [a for s in SEEDS for a in per_seed[s]]
    idx = np.concatenate(list(fe.val_orders().values()))
    p = fe.run(fac, "val", idx, seed=0)
    ok, L = p == y[idx], len(y)
    first, last = float(ok[:L].mean()), float(ok[-L:].mean())
    share = float(np.bincount(p[-L:], minlength=20).max() / L)
    veto = share > 0.10 or last < first - 0.05
    append(dict(kind="eval", key=key(cfg), cfg=cfg, V=100 * float(np.mean(allacc)),
                V_seed={str(s): 100 * float(np.mean(per_seed[s])) for s in SEEDS},
                L_first=100 * first, L_last=100 * last, L_maxshare=share, veto=bool(veto)))


def work_thr(track):
    import threshold_val as tv
    cfg = {**track, "steps": 0, "sizes": [160]}
    thr, lines = tv.derive(cfg, thr=8.7, verbose=False)
    append(dict(kind="thr", key=key(track), track=track, thr=thr, log=lines))


# ---------------------------------------------------------------- driver
def run_sub(kind, obj):
    """One worker subprocess; retried, since CUDA start-up is flaky under host memory pressure."""
    import time
    for attempt in range(12):
        r = subprocess.run([sys.executable, __file__, kind, json.dumps(obj)], cwd=HERE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            return
        err = r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ""
        log(f"   (worker {kind} failed, attempt {attempt + 1}: {err})")
        time.sleep(60)  # out of memory on the shared GPU: wait for other jobs
    raise SystemExit(f"worker {kind} {json.dumps(obj)} failed 12 times")


def ensure(kind, objs):
    done = load()
    todo = [o for o in objs if (kind, key(o)) not in done]
    if todo:
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda o: run_sub(kind, o), todo))
    done = load()
    return [done[(kind, key(o))] for o in objs]


def with_threshold(cfg):
    """cfg with its own val-derived shift threshold (if it resets at shifts)."""
    if not cfg.get("alpha_min") and not cfg.get("shift_thr"):
        return {k: v for k, v in cfg.items() if k != "shift_thr"}
    track = {k: cfg[k] for k in TRACK_KEYS if k in cfg}
    (t,) = ensure("thr", [track])
    return {**cfg, "shift_thr": t["thr"]}


def select(stage, base, variants, margin):
    cands = [with_threshold({**base, **v}) for v in variants]
    ths = {}
    for v, c in zip(variants, cands):
        if "shift_thr" in c:
            ths[key(v)] = c["shift_thr"]
    res = ensure("eval", cands)
    log(f"== {stage}")
    for v, r in zip(variants, res):
        th = f" thr {ths[key(v)]:g}" if key(v) in ths else ""
        log(f"   {json.dumps(v):70s} V {r['V']:.2f} (s0 {r['V_seed']['0']:.2f} / s1 {r['V_seed']['1']:.2f}) "
            f"| L first {r['L_first']:.1f} last {r['L_last']:.1f} maxshare {r['L_maxshare']:.3f}"
            f"{' VETO' if r['veto'] else ''}{th}")
    ok = [(v, c, r) for v, c, r in zip(variants, cands, res) if not r["veto"]]
    if not ok:
        raise SystemExit(f"{stage}: every candidate vetoed")
    if margin is None:  # stage 1: margin from the simplest candidate's seed spread
        r0 = res[0]
        margin = max(0.3, abs(r0["V_seed"]["0"] - r0["V_seed"]["1"]))
        log(f"   margin m = {margin:.2f}")
    best = max(r["V"] for _, _, r in ok)
    tied = [(v, c, r) for v, c, r in ok if r["V"] >= best - margin]
    v, c, r = tied[0]
    log(f"   -> pick {json.dumps(v)} (V {r['V']:.2f}; best {best:.2f}; tied within m: {len(tied)})")
    return c, margin


def main():
    base = dict(sizes=[160, 224])
    m = None
    base, m = select("S1 self-training (steps, LAME T)", base,
                     [dict(steps=s, lame_T=t) for s in (2, 3, 4) for t in (1.0, 2.0, 3.0)], m)
    base, m = select("S2 tracking", base,
                     [dict()] + [dict(alpha_min=a) for a in (0.4, 0.2, 0.1, 0.05)], m)
    if base.get("alpha_min"):
        base, m = select("S3 in-batch split", base, [dict(), dict(split_shift=True)], m)
        base, m = select("S4 second pass", base, [dict()] + [dict(refine_batches=r) for r in (1, 2, 3)], m)
    base, m = select("S5 class prior", base, [dict()] + [dict(prior_lambda=l) for l in (1.0, 2.0, 3.0)], m)
    base, m = select("S6 per-stage learning rates", base,
                     [dict()] + [dict(stage_lr=[e, e, e, l3, l4]) for e in (2, 3, 4) for l3 in (1, 2) for l4 in (0.25, 0.1)], m)
    base, m = select("S7 adaptive LAME temperature", base, [dict()] + [dict(lame_q=q) for q in (0.4, 0.3, 0.2)], m)
    log(f"FINAL {json.dumps(base, sort_keys=True)}")
    json.dump(base, open(os.path.join(HERE, "repick_final.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        obj = json.loads(sys.argv[2])
        {"eval": work_eval, "thr": work_thr}[sys.argv[1]](obj)
    else:
        main()
