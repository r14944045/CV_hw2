"""Run a registered method through the official harness (same seeding as run.py) and record every batch's
LAME input (probabilities after prior/temperature), features and output. Predictions match run.py exactly.

  python record_harness.py <method> <split> <out_dir>
"""
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../starter"))
import tta.mixtta_ms as mm  # noqa: E402
from harness import run_stream  # noqa: E402
from methods import METHODS  # noqa: E402
from run import load_model  # noqa: E402

name, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
REC = []
_lame = mm.lame


def rec_lame(prob, feats, knn=5, **kw):
    z = _lame(prob, feats, knn, **kw)
    REC.append((prob.float().cpu(), feats.half().cpu(), z.float().cpu()))
    return z


mm.lame = rec_lame
random.seed(0); np.random.seed(0); torch.manual_seed(0)
method = METHODS[name](load_model(os.path.join(HERE, "../../source_resnet50_160.pth")))
log = run_stream(method, os.path.join(HERE, "../../data"), split, out, extra_log=dict(method=name, split=split, seed=0))
torch.save(dict(prob=torch.cat([r[0] for r in REC]), feats=torch.cat([r[1] for r in REC]), z=torch.cat([r[2] for r in REC]),
                shifts=getattr(method, "shifts", None)), os.path.join(out, "rec.pt"))
print("sha", log["pred_sha256"][:16])
