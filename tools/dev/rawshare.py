"""Raw ensemble argmax (before prior / LAME) per test domain: tree-frog share and imbalance floor."""
import os
import sys, numpy as np, torch, fe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../starter"))
from tta import MixTTAMS
names = ["noise","wave","chrom","thermal","sepia","pixel","stripes","grid","teal","halft","halft2","poster"]
tb = fe.data("test")["bounds"]
def floor(p):
    n=len(p); c=np.bincount(p,minlength=20); return np.clip(c-n/20,0,None).sum()/n
for spec in sys.argv[1:]:
    kw = eval(spec); raw = []
    def fac(model, kw=kw):
        m = MixTTAMS(model, **kw); orig = m._scale_forward; buf = []
        def sf(xs, k, alpha=None):
            p, f = orig(xs, k, alpha); buf.append(p)
            if len(buf) == len(m.sizes):
                raw.append(torch.stack(buf).mean(0).argmax(1).cpu()); buf.clear()
            return p, f
        m._scale_forward = sf; return m
    p = fe.run(fac, "test"); r = torch.cat(raw).numpy()
    print(spec)
    print("  raw frog share :", " ".join(f"{(r[a:b]==7).mean():.3f}" for a,b in zip(tb[:-1],tb[1:])))
    print("  raw floor      :", " ".join(f"{floor(r[a:b]):.3f}" for a,b in zip(tb[:-1],tb[1:])), f"| all-dom {sum(floor(r[a:b])*(b-a) for a,b in zip(tb[:-1],tb[1:]))/len(r):.3f}")
    print("  final floor    :", " ".join(f"{floor(p[a:b]):.3f}" for a,b in zip(tb[:-1],tb[1:])), flush=True)
