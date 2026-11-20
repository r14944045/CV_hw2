"""Round 8b: corruption augmentation of the self-training copy (ptta "caug"), judged on unseen corruption types.
Protocol fixed on 2026-11-19 before any of its runs (written while round 8 was still running). Revised at 23:45 CST,
still before any of its runs, at the user's request to test caug first: the base is mixtta_repick instead of round 8's
final setting.

caug: in each self-training step, the randomly shifted training copy of the test batch also gets one ImageNet-C
corruption, drawn at random from the six tuning-half corruptions of the suite (gaussian noise, impulse noise, glass
blur, zoom blur, brightness, elastic transform; severity 3; jpeg left out as it has no fast GPU version). No labels are
used; the target is still the LAME output on the uncorrupted views. The set is the mechanical tuning half of the
published ImageNet-C list (suite.py), not chosen by looking at test images.

Base B8: mixtta_repick (repick_final.json). If round 8 later adopts components, caug is re-checked on top of that
combination with the same rule.
Metrics (round8.py worker): S_conf, the confirmation-half suite score (shot noise, defocus blur, motion blur, snow, fog,
contrast, pixelate: corruption types caug never trains on); V, the re-pick val metric; L, the long-stream veto. S_tune
is reported but not used, since caug trains on exactly those corruption types.
Margin m_c = max(0.3, |S_conf(seed 0) - S_conf(seed 1)|) of B8.
Rule: caug is adopted if S_conf - S_conf(B8) >= m_c, V >= V(B8) - 0.3 and it is not vetoed. Not revised after a
Kaggle score comes back.

  python round8b.py              # decisions in round8b_log.txt, result in round8b_final.json
  python round8b.py --on-round8  # the re-check on round 8's final setting -> round8b_final_on8.json
"""
import json
import os
import sys
import time

import round8 as r8

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "round8b_log.txt")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    base_cfg = r8.BASE
    if "--on-round8" in sys.argv:  # the re-check on top of round 8's final combination (same rule)
        base_cfg = json.load(open(os.path.join(HERE, "round8_final.json")))
    cand_cfg = r8.merge(base_cfg, dict(ptta=dict(caug=True)))
    base, cand = r8.ensure([base_cfg, cand_cfg])
    m = max(0.3, abs(base["S_conf_seed"]["0"] - base["S_conf_seed"]["1"]))
    log(f"== round 8b: caug on top of {json.dumps(base_cfg, sort_keys=True)}")
    log(r8.fmt("B8", base))
    log(r8.fmt("B8+caug", cand))
    ok = (cand["S_conf"] - base["S_conf"] >= m and cand["V"] >= base["V"] - 0.3 and not cand["veto"])
    log(f"   margin m_c = {m:.2f} | dS_conf {cand['S_conf'] - base['S_conf']:+.2f} (unseen corruption types) "
        f"dV {cand['V'] - base['V']:+.2f} dS_tune {cand['S_tune'] - base['S_tune']:+.2f} (not used)"
        f" -> {'ADOPT' if ok else 'drop'}")
    final = cand_cfg if ok else base_cfg
    log(f"FINAL {json.dumps(final, sort_keys=True)}")
    out = "round8b_final_on8.json" if "--on-round8" in sys.argv else "round8b_final.json"
    json.dump(final, open(os.path.join(HERE, out), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
