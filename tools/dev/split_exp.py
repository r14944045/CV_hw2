"""Split / second-pass experiments on MixTTAMS: proxy at 4 offsets + val7.  SEED=1 python split_exp.py base,split,split_ref2"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../starter"))
import fe
from tta import MixTTAMS
B = dict(shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0)
CFG = {"base": B, "split": dict(B, split_shift=True), "split_all": dict(B, split_shift=True, split_train="all"),
       "split_m8": dict(B, split_shift=True, split_min=8),
       "ref1": dict(B, refine_batches=1), "ref2": dict(B, refine_batches=2), "ref3": dict(B, refine_batches=3),
       "split_ref2": dict(B, split_shift=True, refine_batches=2)}
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CFG)
SEED = int(os.environ.get("SEED", 0))
offs = (0, 16, 32, 48)
for n in names:
    inst = []
    fac = lambda model, kw=CFG[n]: inst.append(MixTTAMS(model, **kw)) or inst[-1]
    res = []
    for o in offs:
        r = fe.proxy(fac, mix=0.5, verbose=False, offset=o, seed=SEED); res.append(r["mean"])
    v = fe.val7(fac, verbose=False, seeds=(SEED,))
    print(f"{n}: proxy offs {offs} {' '.join(f'{x:.2f}' for x in res)} mean {np.mean(res):.2f} | val7 {v['mean']:.2f} {v['dom']} | splits(last) {getattr(inst[-1], 'splits', None)}", flush=True)
    with open(os.path.join(HERE, "exp_log.txt"), "a") as f:
        f.write(json.dumps(dict(round=f"split_s{SEED}", name=n, kw=CFG[n], proxy_offsets=dict(zip(map(str, offs), [round(x, 2) for x in res])), proxy_off_mean=round(float(np.mean(res)), 2), val7=round(v['mean'], 2))) + "\n")
