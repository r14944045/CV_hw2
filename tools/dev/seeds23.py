"""Seed robustness (val only): suite halves and the val metric with seeds 2 and 3, for one setting.
  python seeds23.py name '<json cfg>'      # appends to seeds23_results.jsonl"""
import json, os, sys
import numpy as np, torch
import fe
from tta.mixtta_ms import MixTTAMS
import valsel
HERE = os.path.dirname(os.path.abspath(__file__))
name, cfg = sys.argv[1], json.loads(sys.argv[2])
fac = lambda model: MixTTAMS(model, **cfg)
y = fe.data("val")["y"].numpy()
rec = dict(name=name, cfg=cfg)
for half in ("tune", "conf"):
    per = {2: [], 3: []}
    for i in range(3):
        X = torch.load(fe.SP + f"suite_{half}{i}.pt")["x"]
        for off in (0, 16, 32, 48):
            idx = np.arange(off, len(X))
            for s in (2, 3):
                p = fe.run(fac, "val", idx, seed=s, x=X)
                per[s].append(float((p == y[idx]).mean()))
    rec[f"S_{half}_seed"] = {str(s): 100 * float(np.mean(v)) for s, v in per.items()}
valsel.SEEDS = (2, 3)
m = valsel.evaluate_factory(fac)
rec["V_seed"] = m["V_seed"]
with open(os.path.join(HERE, "seeds23_results.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")
print(name, json.dumps({k: v for k, v in rec.items() if k != "cfg"}), flush=True)
