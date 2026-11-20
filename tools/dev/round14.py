"""Round 14: T3A's prototype classifier on top of mixtta_lfsq4 (Kaggle public 0.94319, the best file).
Protocol fixed on 2026-11-20 at about 05:13 CST, before any of its runs.

Base: mixtta_lfsq4 (unchanged by round 13). The last published component on the list:
  t3a20 / t3a100  T3A prototype classifier (Iwasawa & Matsuo, NeurIPS 2021; matsuolab/T3A), filter_K 20 / 100 (values
                  from the official sweep), supports restart at a detected shift (no shift detection in this family:
                  supports persist over the stream)
T3A carries predictions across batches into the targets, the same risk class as the local prior with shift-reset
tracking; an adopted setting must also pass a Kaggle probe before replacing anything.
Metrics and rule: exactly as round8.py (suite tuning half, confirmation half, val metric V, long-val veto; margin
m = max(0.3, seed spread of the base's S_tune); adopt if S_tune gain >= m, S_conf gain > 0, V drop <= 0.3, not
vetoed; adopted changes are then added one at a time in order of S_tune gain, each re-checked).
A result that changes the base is submitted to Kaggle only as a declared probe; it replaces mixtta_lfsq4 only if its
public score is at least 0.5 pt higher (>= 0.9482). Not revised after a Kaggle score comes back.

  python round14.py          # decisions in round14_log.txt, result in round14_final.json
"""
import json
import os

import round8 as r8

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "round14_log.txt")
BASE = dict(sizes=[160, 224], steps=3, lame_T=2.0, local_context=True, stage_lr=[4, 4, 4, 1, 0.1], lame_q=0.3)
CANDIDATES = {
    "t3a20": dict(ptta=dict(t3a=20)),
    "t3a100": dict(ptta=dict(t3a=100)),
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    (base,) = r8.ensure([BASE])
    m = max(0.3, abs(base["S_tune_seed"]["0"] - base["S_tune_seed"]["1"]))
    log(f"== round 14 base mixtta_lfsq4 {json.dumps(BASE, sort_keys=True)}")
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
    json.dump(cur_cfg, open(os.path.join(HERE, "round14_final.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
