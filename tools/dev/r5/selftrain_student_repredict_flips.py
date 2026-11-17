"""Diagnostic for selftrain_student_repredict: how many predictions does the student re-prediction flip,
and how many of those flips fix or break a label (proxy, offset 0, 4 streams)?
  cd tools/dev && ./gpu_slot.sh python r5/selftrain_student_repredict_flips.py '{"rp_beta":0.5}'
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
sys.path.insert(0, DEV)
sys.path.insert(0, os.path.join(DEV, "../../starter"))
import fe  # noqa: E402
from r5.evaluate import BASE  # noqa: E402
from r5.selftrain_student_repredict import SelfTrainRepredict  # noqa: E402


class Rec(SelfTrainRepredict):
    def _predict_fused(self, out, p_s, f_s):
        z = super()._predict_fused(out, p_s, f_s)
        self.log.append((self.t, self._z_first.argmax(1).cpu().numpy(), z.argmax(1).cpu().numpy()))
        return z

    def _predict(self, out):
        z = super()._predict(out)
        self._z_first = z
        return z


def main():
    kw = dict(BASE, **json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}"))
    d = fe.data("proxy0.5"); y = d["y"].numpy(); L = d["stream_len"]
    tot = dict(batches=0, flips=0, fixed=0, broken=0, images=0)
    for s in range(4):
        idx = np.arange(s * L, (s + 1) * L)
        holder = {}

        def fac(model):
            holder["m"] = Rec(model, **kw); holder["m"].log = []
            return holder["m"]
        fe.run(fac, "proxy0.5", idx, seed=0)
        for t, a, b in holder["m"].log:
            yt = y[idx[(t - 1) * 64:(t - 1) * 64 + len(a)]]
            ch = a != b
            tot["batches"] += 1; tot["images"] += len(a); tot["flips"] += int(ch.sum())
            tot["fixed"] += int((ch & (b == yt)).sum()); tot["broken"] += int((ch & (a == yt)).sum())
    print(json.dumps(dict(kw=kw, **tot)), flush=True)


if __name__ == "__main__":
    main()
