import os, sys
import numpy as np, torch
import fe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../starter"))
from tta import MixTTAMS
kw = eval(sys.argv[1]); holder = {}
def fac(model):
    m = MixTTAMS(model, **kw); holder["m"] = m; return m
p = fe.run(fac, "test")
m = holder["m"]
print(kw, "detected", m.shifts)
frog = (p == 7).reshape(-1, 64).mean(1)
print("frog share per 8 batches:", np.round(frog.reshape(-1, 8).mean(1), 2))
g = [l.weight.detach().float() for l in m.layers]; g0 = None
print("gamma mean abs", np.mean([x.abs().mean().item() for x in g]))
