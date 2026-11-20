"""Round 9: more published practical-TTA components on top of round 8's final setting.
Protocol fixed on 2026-11-20 at about 01:09 CST, before any of its runs.

Base: round 8b's re-check result (round8b_final_on8.json, i.e. round 8's final setting, mixtta_repick + local class
prior, plus caug only if the re-check adopted it).
Candidates, each = base + one component (official code read for the defaults; see the ptta option of MixTTAMS):
  iabn   NOTE's instance-aware BN, k = 3 (Gong et al., NeurIPS 2022; TaesikGong/NOTE utils/iabn.py)
  t3a    T3A prototype classifier, filter_K = 100 (Iwasawa & Matsuo, NeurIPS 2021; matsuolab/T3A); the official code
         has no single default (it sweeps 1/5/20/50/100/-1); 100 is the value most later papers use. Supports restart
         at a detected domain shift.
  lameq3 adaptive LAME temperature, mean max-probability 0.3: re-tested on the new base because in round 8 it had the
         largest gain on unseen corruptions (+1.14 on the confirmation half) but failed the tuning-half requirement.
Metrics and rule: exactly as round8.py (suite tuning half S_tune, confirmation half S_conf, val metric V, long-stream
veto; margin m = max(0.3, seed spread of the base's S_tune); adopt if S_tune gain >= m, S_conf gain > 0, V drop <= 0.3,
not vetoed; adopted components are then added one at a time in order of S_tune gain, each re-checked).
Not revised after a Kaggle score comes back.

  python round9.py           # waits for round8b_final_on8.json; decisions in round9_log.txt, result in round9_final.json
"""
import json
import os
import time

import round8 as r8

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "round9_log.txt")
CANDIDATES = {
    "iabn": dict(ptta=dict(iabn=3.0)),
    "t3a": dict(ptta=dict(t3a=100)),
    "lameq3": dict(lame_q=0.3),
}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    src = os.path.join(HERE, "round8b_final_on8.json")
    while not os.path.exists(src):
        time.sleep(60)
    base_cfg = json.load(open(src))
    (base,) = r8.ensure([base_cfg])
    m = max(0.3, abs(base["S_tune_seed"]["0"] - base["S_tune_seed"]["1"]))
    log(f"== round 9 base {json.dumps(base_cfg, sort_keys=True)}")
    log(r8.fmt("base", base) + f" | margin m = {m:.2f}")
    names = list(CANDIDATES)
    res = r8.ensure([r8.merge(base_cfg, CANDIDATES[n]) for n in names])
    adopted = []
    for n, r in zip(names, res):
        ok = r8.passes(r, base, m)
        log(r8.fmt(n, r) + f" | dS_tune {r['S_tune'] - base['S_tune']:+.2f} dS_conf {r['S_conf'] - base['S_conf']:+.2f} "
            f"dV {r['V'] - base['V']:+.2f}{'  ADOPT' if ok else ''}")
        if ok:
            adopted.append((r["S_tune"] - base["S_tune"], n))
    cur_cfg, cur = base_cfg, base
    for _, n in sorted(adopted, reverse=True):
        cfg = r8.merge(cur_cfg, CANDIDATES[n])
        (r,) = r8.ensure([cfg])
        ok = r8.passes(r, cur, m)
        log(f"== combine + {n}" + r8.fmt("", r) + ("  KEEP" if ok else "  drop"))
        if ok:
            cur_cfg, cur = cfg, r
    log(f"FINAL {json.dumps(cur_cfg, sort_keys=True)}")
    json.dump(cur_cfg, open(os.path.join(HERE, "round9_final.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
