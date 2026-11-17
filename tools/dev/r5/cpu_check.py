"""Reduced check that runs on the CPU only: proxy at offset 0 + the long held-out stream, one seed.
About 25 min per stream on a 12-core CPU (fp32), so ~50 min per configuration. Results: r5/cpu_results.jsonl.

  CUDA_VISIBLE_DEVICES= HW2_DEVICE=cpu OMP_NUM_THREADS=24 python r5/cpu_check.py --seed 1 \
      --cfg base='{}' --cfg q20='{"lame_q": 0.2}'          (overrides on top of sp2cprl3, class r5.round6:Round6)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, ".."))
import fe  # noqa: E402
from r5.evaluate import BASE, factory  # noqa: E402

SP2CPRL3 = dict(BASE, stage_lr=[3, 3, 3, 1, 0.1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="r5.round6:Round6")
    ap.add_argument("--cfg", action="append", required=True, help="name=json overrides")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    assert fe.DEVICE == "cpu", "set HW2_DEVICE=cpu"
    for spec in a.cfg:
        name, kw = spec.split("=", 1)
        fac, _ = factory(a.cls, dict(SP2CPRL3, **json.loads(kw)))
        t0 = time.time()
        px = fe.proxy(fac, mix=0.5, verbose=False, offset=0, seed=a.seed)
        d = fe.data("heldout"); y = d["y"].numpy()
        p = fe.run(fac, "heldout", np.arange(len(y)), seed=a.seed)
        rec = dict(tag=name, kw=json.loads(kw), seed=a.seed, device="cpu", proxy_off0=round(float(px["mean"]), 2),
                   proxy_per=[float(v) for v in px["per"]], heldout_long=round(100 * float((p == y).mean()), 2),
                   minutes=round((time.time() - t0) / 60, 1))
        with open(os.path.join(HERE, "cpu_results.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
