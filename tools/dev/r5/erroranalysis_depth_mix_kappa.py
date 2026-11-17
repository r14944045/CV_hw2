"""Calibrate gap_kappa for erroranalysis_depth_mix: per-stage median of the tracked-vs-source KL (tracker 0)
over all batches of the 7 val orders, under the given settings (default: the base, w = 0.5 everywhere).
  ./gpu_slot.sh python r5/erroranalysis_depth_mix_kappa.py ['{"w_stage": [...]}']
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "../../starter"))
import fe  # noqa: E402
from r5.evaluate import BASE  # noqa: E402
from r5.erroranalysis_depth_mix import DepthMix  # noqa: E402

kw = dict(BASE, gap_log=True, **(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))
logs = []
for name, idx in fe.val_orders().items():
    h = {}
    fe.run(lambda m: h.setdefault("m", DepthMix(m, **kw)), "val", idx=idx, seed=0)
    logs += h["m"].kl_log
L = np.array(logs)
print("n batches", len(L))
print("median per stage", np.round(np.median(L, 0), 4).tolist())
print("p25", np.round(np.percentile(L, 25, 0), 4).tolist(), "p75", np.round(np.percentile(L, 75, 0), 4).tolist())
