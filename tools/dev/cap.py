"""Capacity check: val7 (+ optional proxy) for MixTTAMS variants given as dict literals."""
import sys, json, fe
sys.path.insert(0, "../../starter")
from tta import MixTTAMS
for spec in sys.argv[1:]:
    kw = eval(spec); fac = lambda model, kw=kw: MixTTAMS(model, **kw)
    v = fe.val7(fac, verbose=False); o = fe.proxy(fac, mix=0.5, verbose=False)
    print(f"{spec}: val7 {v['mean']:.2f} dom {v['dom']} | proxy {o['mean']:.2f} | {' '.join(f'{x:.1f}' for x in o['per'])}", flush=True)
    with open("exp_log.txt", "a") as f: f.write(json.dumps(dict(round="cap", spec=spec, val7=round(v['mean'], 2), proxy=round(o['mean'], 2), per=o['per'].tolist())) + "\n")
