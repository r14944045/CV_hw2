"""Round 10: components on top of mixtta_lfs (Kaggle public 0.93815, the best val-only file).
Protocol fixed on 2026-11-20 at about 01:53 CST, before any of its runs.

Base: mixtta_lfs = MixTTAMS(local_context=True, stage_lr=(2, 2, 2, 1, 0.25)) with the class defaults (fixed tracking
rate 0.4, source weight 0.5, sizes 160 + 224, flip + 2 shifts, LAME T = 2, 3 steps). Fixed tracking only: the local
prior collapsed on the public board with the shift reset / split / running-mean tracking (0.69 and 0.83).
Candidates, each = base + one change:
  dot    DELTA's dynamic online reweighting, momentum 0.95 (Zhao et al., ICLR 2023), a class-balance stabiliser
  lameq3 adaptive LAME temperature, mean max-probability 0.3 (largest gain on unseen corruptions in round 8)
  iabn   NOTE's instance-aware BN, k = 3 (Gong et al., NeurIPS 2022)
  steps4 4 self-training steps instead of 3 (the best stage-1 score on val in the re-pick, within its margin)
Metrics and rule: exactly as round8.py (suite tuning half, confirmation half, val metric V, long-val veto; margin
m = max(0.3, seed spread of the base's S_tune); adopt if S_tune gain >= m, S_conf gain > 0, V drop <= 0.3, not
vetoed; adopted changes are then added one at a time in order of S_tune gain, each re-checked).
A result that changes the base is submitted to Kaggle only as a declared probe; it replaces mixtta_lfs only if its
public score is at least 0.5 pt higher (>= 0.9432). Not revised after a Kaggle score comes back.

  python round10.py          # decisions in round10_log.txt, result in round10_final.json
"""
import json
import os

import round8 as r8

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "round10_log.txt")
BASE = dict(sizes=[160, 224], steps=3, lame_T=2.0, local_context=True, stage_lr=[2, 2, 2, 1, 0.25])
CANDIDATES = {
    "dot": dict(ptta=dict(dot=0.95)),
    "lameq3": dict(lame_q=0.3),
    "iabn": dict(ptta=dict(iabn=3.0)),
    "steps4": dict(steps=4),
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    (base,) = r8.ensure([BASE])
    m = max(0.3, abs(base["S_tune_seed"]["0"] - base["S_tune_seed"]["1"]))
    log(f"== round 10 base mixtta_lfs {json.dumps(BASE, sort_keys=True)}")
    log(r8.fmt("base", base) + f" | margin m = {m:.2f}")
    names = list(CANDIDATES)
    res = r8.ensure([r8.merge(BASE, CANDIDATES[n]) for n in names])
    adopted = []
    for n, r in zip(names, res):
        ok = r8.passes(r, base, m)
        log(r8.fmt(n, r) + f" | dS_tune {r['S_tune'] - base['S_tune']:+.2f} dS_conf {r['S_conf'] - base['S_conf']:+.2f} "
            f"dV {r['V'] - base['V']:+.2f}{'  ADOPT' if ok else ''}")
        if ok:
            adopted.append((r["S_tune"] - base["S_tune"], n))
    cur_cfg, cur = BASE, base
    for _, n in sorted(adopted, reverse=True):
        cfg = r8.merge(cur_cfg, CANDIDATES[n])
        (r,) = r8.ensure([cfg])
        ok = r8.passes(r, cur, m)
        log(f"== combine + {n}" + r8.fmt("", r) + ("  KEEP" if ok else "  drop"))
        if ok:
            cur_cfg, cur = cfg, r
    log(f"FINAL {json.dumps(cur_cfg, sort_keys=True)}")
    json.dump(cur_cfg, open(os.path.join(HERE, "round10_final.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
