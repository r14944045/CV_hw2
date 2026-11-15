import torch, numpy as np, fe
m = fe.load_model().eval()
def feats(X, idx):
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), 64):
            x = (X[idx[k:k+64]].cuda().float()/255 - fe.MEAN)/fe.STD
            h = m.conv1(x); v, mu = torch.var_mean(h, dim=(0,2,3), unbiased=False); out.append((mu, v))
    return out
def kl(mu, v, mt, vt): return 0.5*((v + (mu-mt)**2)/(vt+1e-5) - 1 - torch.log((v+1e-5)/(vt+1e-5))).mean().item()
def detect(fs, thr, a=0.4, mode="ratio", absthr=None):
    mt, vt = m.bn1.running_mean.clone(), m.bn1.running_var.clone(); hist = []; det = []
    for t, (mu, v) in enumerate(fs):
        d = kl(mu, v, mt, vt)
        ref = np.median(hist[-8:]) if len(hist) >= 3 else None
        shift = (ref is not None and d > thr * max(ref, 1e-4)) if mode == "ratio" else (t > 0 and d > absthr)
        if shift: det.append(t); hist = []; mt, vt = mu.clone(), v.clone()
        else: hist.append(d); mt = (1-a)*mt + a*mu; vt = (1-a)*vt + a*v
    return det
S = {}
d = fe.data("test"); S["test"] = (feats(d["x"], np.arange(len(d["x"]))), sorted(set((d["bounds"][1:-1]//64).tolist())))
d = fe.data("val"); S["val"] = (feats(d["x"], np.arange(len(d["x"]))), sorted(set((d["bounds"][1:-1]//64).tolist())))
p = fe.data("proxy0.5"); L = p["stream_len"]
for s in range(4): S[f"px{s}"] = (feats(p["x"], np.arange(s*L,(s+1)*L)), S["val"][1])
for thr in [5, 10, 20]:
    print("thr", thr)
    for k, (fs, tb) in S.items():
        det = detect(fs, thr); print(f"   {k}: true {tb} det {det}")
for thr in [0.5, 1.0, 2.0]:
    print("abs", thr)
    for k, (fs, tb) in S.items():
        det = detect(fs, None, mode="abs", absthr=thr); print(f"   {k}: true {tb} det {det}")
