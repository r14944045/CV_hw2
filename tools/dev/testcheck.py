"""Label-free test-stream check of a MixTTAMS variant (dict literal): detected shifts / splits, per-domain imbalance
floor and largest class share of the final predictions, and the share of predictions that differ from a reference
submission file.   python testcheck.py "dict(...)" [--ref ../../out/mixtta_ms224sp2/test_pred.csv] [--seed N]"""
import argparse, os, sys
import numpy as np, pandas as pd
import fe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../starter"))
from tta import MixTTAMS

ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("--ref", default="../../out/mixtta_ms224sp2/test_pred.csv")
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
kw = eval(a.spec); holder = {}
def fac(model):
    holder["m"] = MixTTAMS(model, **kw); return holder["m"]
p = fe.run(fac, "test", seed=a.seed); m = holder["m"]
tb = fe.data("test")["bounds"]; ids = fe.data("test")["ids"]
ref = pd.read_csv(a.ref).set_index("id").loc[list(ids), "label"].to_numpy()
floor = lambda q: np.clip(np.bincount(q, minlength=20) - len(q) / 20, 0, None).sum() / len(q)
print(a.spec, f"seed {a.seed}")
print("  shifts:", m.shifts, "| splits:", getattr(m, "splits", None))
print("  floor   :", " ".join(f"{floor(p[x:y]):.3f}" for x, y in zip(tb[:-1], tb[1:])),
      f"| all {sum(floor(p[x:y]) * (y - x) for x, y in zip(tb[:-1], tb[1:])) / len(p):.4f}")
print("  maxshare:", " ".join(f"{np.bincount(p[x:y], minlength=20).max() / (y - x):.3f}" for x, y in zip(tb[:-1], tb[1:])),
      f"| frog {np.bincount(p, minlength=20)[7] - len(p) // 20:+d}")
print(f"  changed vs ref: {(p != ref).mean():.4f}", flush=True)
