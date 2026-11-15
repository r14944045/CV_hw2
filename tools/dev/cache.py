import os, sys, numpy as np, pandas as pd, torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, "../../data")
out = os.environ.get("HW2_DEV_CACHE", os.path.join(HERE, "cache")); os.makedirs(out, exist_ok=True)
for split in ["val", "test"]:
    ids = pd.read_csv(f"{root}/{split}/stream.csv")["id"].tolist()
    def load(i): return np.asarray(Image.open(f"{root}/{split}/images/{i}.png").convert("RGB"))
    with ThreadPoolExecutor(16) as ex: arr = np.stack(list(ex.map(load, ids)))
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    d = dict(ids=ids, x=x)
    lab = f"{root}/{split}/labels.csv"
    import os
    if os.path.exists(lab):
        d["y"] = torch.tensor(pd.read_csv(lab).set_index("id").loc[ids, "label"].to_numpy())
    torch.save(d, os.path.join(out, f"{split}.pt")); print(split, x.shape)
