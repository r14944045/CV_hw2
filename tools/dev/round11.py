"""Round 11: knobs of the fixed-tracking family on top of mixtta_lfsq (Kaggle public 0.94042, the best file).
Protocol fixed on 2026-11-20 at about 02:38 CST, before any of its runs.

Base: mixtta_lfsq = mixtta_lfs + adaptive LAME temperature 0.3 (round 10's result). This family's own knobs (tracking
rate 0.4, source weight 0.5, 2 shift views) were set long ago on val and never re-chosen for it.
Candidates, each = base + one change:
  shift3  3 random flip + shift views per size instead of 2 (about +10 % time)
  alpha02 / alpha06  fixed tracking rate 0.2 / 0.6 instead of 0.4
  w03 / w07          source weight 0.3 / 0.7 instead of 0.5
Metrics and rule: exactly as round8.py (suite tuning half, confirmation half, val metric V, long-val veto; margin
m = max(0.3, seed spread of the base's S_tune); adopt if S_tune gain >= m, S_conf gain > 0, V drop <= 0.3, not
vetoed; adopted changes are then added one at a time in order of S_tune gain, each re-checked).
A result that changes the base is submitted to Kaggle only as a declared probe; it replaces mixtta_lfs only if its
public score is at least 0.5 pt higher (>= 0.9454). Not revised after a Kaggle score comes back.

  python round11.py          # decisions in round11_log.txt, result in round11_final.json
"""
import json
import os

import round8 as r8

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "round11_log.txt")
BASE = dict(sizes=[160, 224], steps=3, lame_T=2.0, local_context=True, stage_lr=[2, 2, 2, 1, 0.25], lame_q=0.3)
CANDIDATES = {
    "shift3": dict(n_shift=3),
    "alpha02": dict(alpha=0.2),
    "alpha06": dict(alpha=0.6),
    "w03": dict(w=0.3),
    "w07": dict(w=0.7),
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    (base,) = r8.ensure([BASE])
    m = max(0.3, abs(base["S_tune_seed"]["0"] - base["S_tune_seed"]["1"]))
    log(f"== round 11 base mixtta_lfsq {json.dumps(BASE, sort_keys=True)}")
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
    json.dump(cur_cfg, open(os.path.join(HERE, "round11_final.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
