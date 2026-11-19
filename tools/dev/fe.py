"""Fast evaluation of TTA methods on cached val streams (outside the harness, same batching of 64).

  streams: the 7 val orders (orig, rev, 5 domain permutations). Val only: the test stream is read only by the
  official harness (starter/run.py).
  rec=True records per-batch LAME inputs/outputs (prob after temperature, feats, z) via a lame() wrapper.
"""
import os
import itertools, sys, time, random
import numpy as np
import torch, torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../starter"))
import tta.mixtta_ms as mm  # noqa
from methods import METHODS  # noqa

SP = os.environ.get("HW2_DEV_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")) + "/"
DEVICE = os.environ.get("HW2_DEVICE", "cuda")  # HW2_DEVICE=cpu runs everything on the CPU (fp32)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
_cache = {}


def data(split="val"):
    assert split == "val", "val only"
    if split not in _cache:
        d = torch.load(SP + f"{split}.pt")
        d["bounds"] = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{split}_bounds.npy"))
        _cache[split] = d
    return _cache[split]


def val_orders():
    """7 orders: name -> index array; and domain-id array per position."""
    b = data("val")["bounds"]; segs = [np.arange(b[i], b[i + 1]) for i in range(3)]
    out = {"orig": np.arange(b[-1]), "rev": np.arange(b[-1])[::-1].copy()}
    for p in itertools.permutations(range(3)):
        if p != (0, 1, 2):
            out["p" + "".join(map(str, p))] = np.concatenate([segs[i] for i in p])
    return out


def load_model():
    ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../source_resnet50_160.pth"), map_location="cpu", weights_only=False)
    m = getattr(torchvision.models, ck["arch"])(num_classes=len(ck["wnids"]))
    m.load_state_dict(ck["state_dict"])
    return m.to(DEVICE)


REC = []
_orig_lame = mm.lame


def _rec_lame(prob, feats, knn=5, **kw):
    z = _orig_lame(prob, feats, knn, **kw)
    REC.append((prob.float().cpu(), feats.float().cpu(), z.cpu()))
    return z


def run(factory, split="val", idx=None, seed=0, rec=False, x=None):
    """factory(model) -> method. Returns predictions (in stream order of idx)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    d = data(split); X = d["x"] if x is None else x
    if idx is None:
        idx = np.arange(len(X))
    meth = factory(load_model())
    REC.clear()
    mm.lame = _rec_lame if rec else _orig_lame
    preds = []
    for k in range(0, len(idx), 64):
        xb = X[idx[k:k + 64]].to(DEVICE, non_blocking=True)
        xb = (xb.float() / 255 - MEAN) / STD
        p = meth(xb)
        preds.append(p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p))
    mm.lame = _orig_lame
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(preds)


def val7(factory, seeds=(0,), verbose=True, orders=None):
    d = data("val"); y = d["y"].numpy(); b = d["bounds"]
    dom = np.zeros(len(y), int); dom[b[1]:b[2]] = 1; dom[b[2]:] = 2
    res = []; per_dom = np.zeros(3); t0 = time.time()
    for name, idx in val_orders().items():
        if orders and name not in orders:
            continue
        for s in seeds:
            p = run(factory, "val", idx, seed=s)
            ok = p == y[idx]
            res.append(ok.mean())
            per_dom += np.array([ok[dom[idx] == k].mean() for k in range(3)])
    n = len(res); per_dom /= n
    out = dict(mean=100 * np.mean(res), worst=100 * np.min(res), dom=(100 * per_dom).round(2), sec=time.time() - t0)
    if verbose:
        print(f"val7 mean {out['mean']:.2f} worst {out['worst']:.2f} dom(scan/wave/fish) {out['dom']}  [{out['sec']:.0f}s]", flush=True)
    return out


def M(name, **kw):
    """Factory from a registered method name plus overrides (for partials)."""
    f = METHODS[name]
    return lambda model: f(model, **kw)
