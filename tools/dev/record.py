"""Record per-batch LAME inputs/outputs of a method on val (orig order) and test. Saves rec_<tag>.pt."""
import sys
import numpy as np
import torch
import fe

name = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else name
out = {}
for split in ["val", "test"]:
    p = fe.run(fe.M(name), split, rec=True)
    prob = torch.cat([r[0] for r in fe.REC]); feats = torch.cat([r[1] for r in fe.REC]); z = torch.cat([r[2] for r in fe.REC])
    out[split] = dict(pred=torch.from_numpy(p), prob=prob, feats=feats.half(), z=z)
    if split == "val":
        y = fe.data("val")["y"].numpy(); print(split, "acc", (p == y).mean(), flush=True)
torch.save(out, fe.SP + f"rec_{tag}.pt")
