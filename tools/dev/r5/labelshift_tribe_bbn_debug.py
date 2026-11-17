"""Debug: run proxy stream 0 (offset 16, exercises split + short last batch) with debug stats; print invariants."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DEV = os.path.dirname(HERE)
sys.path.insert(0, DEV); sys.path.insert(0, os.path.join(DEV, "../../starter"))
import fe  # noqa
from r5.evaluate import BASE  # noqa
from r5.labelshift_tribe_bbn import MixTTAMSBal  # noqa

kw = dict(BASE, debug=True, **json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}"))
h = {}
def fac(model):
    h["m"] = MixTTAMSBal(model, **kw); return h["m"]
d = fe.data("proxy0.5"); L = d["stream_len"]; y = d["y"].numpy()
idx = np.arange(16, L)
p = fe.run(fac, "proxy0.5", idx, seed=0)
print("acc", (p == y[idx]).mean())
m = h["m"]
print("shifts", m.shifts, "splits", m.splits)
for i, r in enumerate(m.dbg):
    print(i, {k: round(v, 4) for k, v in r.items()})
