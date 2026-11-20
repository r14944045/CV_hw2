"""Round 8: published practical-TTA components on top of mixtta_repick, judged on a corruption suite added to val.
Protocol fixed on 2026-11-19 before any of its runs (the first run started at 22:55 CST).

Why: the test stream, and the private leaderboard in particular, uses corruption types that val does not contain
(slide 11). The slide-21 FAQ allows adding our own corruptions to val for evaluation, so robustness to unseen
corruptions is measured on the ImageNet-C suite built by suite.py (published list, severity 3, frost excluded for a
mechanical reason; even positions = tuning half, odd = confirmation half). No test data is read.

Metrics (val images and val labels only):
  S_tune: mean accuracy over the 3 tuning-half suite streams x in-batch offsets 0/16/32/48 x seeds 0/1 (24 runs)
  S_conf: the same on the 3 confirmation-half streams
  V:      the re-pick val metric (7 val orders x 4 offsets x 2 seeds); reused from repick_results.jsonl when the
          setting was already evaluated there
  L:      the re-pick collapse veto on the long val stream (one class > 10 % at the end, or accuracy -5 pt)
Margin m = max(0.3, |S_tune(seed 0) - S_tune(seed 1)|) of the base, mixtta_repick (repick_final.json).

Sanity check (not a gate): S_tune / S_conf of the four Kaggle-scored val-only settings (floor, ms224, s5, repick),
compared with their Kaggle order (0.91227 < 0.91585 < 0.92220 < 0.92675).

Candidates, each = base + one component with its published defaults (C = 20 classes, entropy margins 0.4 ln C):
  eata      EATA reliable + non-redundant sample filter and exp(E0 - H) weights (Niu et al., ICML 2022)
  roid_w    ROID certainty x diversity weights (Marsden et al., WACV 2024)
  dot       DELTA's dynamic online reweighting, momentum 0.95 (Zhao et al., ICLR 2023)
  roid_we   ROID weight ensembling with the source, momentum 0.99 after every step
  sar_reset SAR model recovery: reset to source when the EMA (0.9) of the entropy falls below 0.2 (Niu et al., 2023)
  roid_prior ROID prior correction of the prediction
  refine2   re-score of a dropped component: second pass on the first 2 batches of a domain
  lameq3    re-score of a dropped component: adaptive LAME temperature, mean max-probability 0.3
  local     the local class prior from the parallel session (local_context=True)
Rule: a candidate is adopted if S_tune - S_tune(base) >= m, S_conf - S_conf(base) > 0, V >= V(base) - 0.3 and it is
not vetoed. Adopted candidates are then added one at a time in order of S_tune gain; each addition must pass the
same rule against the current combination. The result is the round's final setting. It is not revised after a
Kaggle score comes back.

  python round8.py           # resumable; results in round8_results.jsonl, decisions in round8_log.txt
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "round8_results.jsonl")
REPICK = os.path.join(HERE, "repick_results.jsonl")
LOG = os.path.join(HERE, "round8_log.txt")
OFFSETS, SEEDS = (0, 16, 32, 48), (0, 1)
WORKERS = 2

BASE = json.load(open(os.path.join(HERE, "repick_final.json")))
SANITY = {
    "floor": dict(sizes=[160, 224], steps=2, lame_T=1.0),
    "ms224": dict(sizes=[160, 224], steps=3, lame_T=2.0),
    "s5": dict(alpha_min=0.2, lame_T=2.0, shift_thr=9.0, sizes=[160, 224], split_shift=True, steps=3),
    "repick": BASE,
}
CANDIDATES = {
    "eata": dict(ptta=dict(eata=True)),
    "roid_w": dict(ptta=dict(roid_w=True)),
    "dot": dict(ptta=dict(dot=0.95)),
    "roid_we": dict(ptta=dict(roid_we=0.99)),
    "sar_reset": dict(ptta=dict(sar_reset=True)),
    "roid_prior": dict(ptta=dict(roid_prior=True)),
    "refine2": dict(refine_batches=2),
    "lameq3": dict(lame_q=0.3),
    "local": dict(local_context=True),
}


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("kind") == "eval":
                out[r["key"]] = r
    return out


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def merge(base, add):
    out = dict(base)
    for k, v in add.items():
        out[k] = {**out.get(k, {}), **v} if k == "ptta" else v
    return out


# ---------------------------------------------------------------- worker
def work(cfg):
    import numpy as np
    import torch
    import fe
    from tta.mixtta_ms import MixTTAMS
    fac = lambda model: MixTTAMS(model, **cfg)  # noqa: E731
    rec = dict(kind="eval", key=key(cfg), cfg=cfg)
    y = fe.data("val")["y"].numpy()
    for half in ("tune", "conf"):
        per = {s: [] for s in SEEDS}
        for i in range(3):
            X = torch.load(fe.SP + f"suite_{half}{i}.pt")["x"]
            for off in OFFSETS:
                idx = np.arange(off, len(X))
                for s in SEEDS:
                    p = fe.run(fac, "val", idx, seed=s, x=X)
                    per[s].append(float((p == y[idx]).mean()))
        rec[f"S_{half}"] = 100 * float(np.mean(per[0] + per[1]))
        rec[f"S_{half}_seed"] = {str(s): 100 * float(np.mean(per[s])) for s in SEEDS}
    prev = load(REPICK).get(key(cfg))
    if prev is None:
        sys.path.insert(0, HERE)
        import repick
        prev = repick.evaluate_factory(fac)
    for k in ("V", "V_seed", "L_first", "L_last", "L_maxshare", "veto"):
        rec[k] = prev[k]
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- driver
def run_sub(cfg):
    for attempt in range(12):
        r = subprocess.run([sys.executable, __file__, "eval", json.dumps(cfg)], cwd=HERE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            return
        err = r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ""
        log(f"   (worker failed, attempt {attempt + 1}: {err})")
        time.sleep(60)
    raise SystemExit(f"worker {json.dumps(cfg)} failed 12 times")


def ensure(cfgs):
    done = load(RESULTS)
    todo = [c for c in cfgs if key(c) not in done]
    if todo:
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(run_sub, todo))
    done = load(RESULTS)
    return [done[key(c)] for c in cfgs]


def fmt(name, r):
    return (f"   {name:12s} S_tune {r['S_tune']:.2f} ({r['S_tune_seed']['0']:.2f}/{r['S_tune_seed']['1']:.2f}) "
            f"S_conf {r['S_conf']:.2f} V {r['V']:.2f} L {r['L_first']:.1f}->{r['L_last']:.1f} "
            f"share {r['L_maxshare']:.3f}{' VETO' if r['veto'] else ''}")


def passes(r, ref, m):
    return (r["S_tune"] - ref["S_tune"] >= m and r["S_conf"] - ref["S_conf"] > 0
            and r["V"] >= ref["V"] - 0.3 and not r["veto"])


def main():
    names = list(SANITY)
    res = ensure([SANITY[n] for n in names])
    log("== sanity: Kaggle-scored val-only settings (Kaggle 0.91227 / 0.91585 / 0.92220 / 0.92675)")
    for n, r in zip(names, res):
        log(fmt(n, r))
    base = res[names.index("repick")]
    m = max(0.3, abs(base["S_tune_seed"]["0"] - base["S_tune_seed"]["1"]))
    log(f"   margin m = {m:.2f}")
    cnames = list(CANDIDATES)
    res = ensure([merge(BASE, CANDIDATES[n]) for n in cnames])
    log("== candidates (base + one component)")
    adopted = []
    for n, r in zip(cnames, res):
        ok = passes(r, base, m)
        log(fmt(n, r) + f" | dS_tune {r['S_tune'] - base['S_tune']:+.2f} dS_conf {r['S_conf'] - base['S_conf']:+.2f} "
            f"dV {r['V'] - base['V']:+.2f}{'  ADOPT' if ok else ''}")
        if ok:
            adopted.append((r["S_tune"] - base["S_tune"], n))
    cur_cfg, cur = BASE, base
    for _, n in sorted(adopted, reverse=True):
        cfg = merge(cur_cfg, CANDIDATES[n])
        if cfg == cur_cfg:
            continue
        (r,) = ensure([cfg])
        ok = passes(r, cur, m)
        log(f"== combine + {n}" + fmt("", r) + ("  KEEP" if ok else "  drop"))
        if ok:
            cur_cfg, cur = cfg, r
    log(f"FINAL {json.dumps(cur_cfg, sort_keys=True)}")
    json.dump(cur_cfg, open(os.path.join(HERE, "round8_final.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "eval":
        work(json.loads(sys.argv[2]))
    else:
        main()
