"""Report Q4: 8 validation images with predicted and ground-truth labels (4 correct, 4 misclassified).

Uses a harness val run (out/val_exp/<method>/val_pred.csv). Images are picked at random (fixed seed), round-robin
over the 3 val domains, so each domain appears in both rows.

  python tools/plot_val_examples.py [--method mixtta_ms224sp2cprl3] [--out figures/val_examples.png]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOMAINS = ["scan-line grid", "wavy warp", "fisheye on black"]  # val domains, in stream order


def pick(pos, dom, n, rng):
    """n stream positions from `pos`, round-robin over domains (random order within each domain)."""
    by_dom = [list(rng.permutation(pos[dom[pos] == d])) for d in range(len(DOMAINS))]
    out, d = [], 0
    while len(out) < n:
        if by_dom[d % len(by_dom)]:
            out.append(by_dom[d % len(by_dom)].pop())
        d += 1
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="mixtta_ms224sp2cprl3")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "val_examples.png"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    stream = pd.read_csv(os.path.join(ROOT, "data/val/stream.csv"))
    labels = pd.read_csv(os.path.join(ROOT, "data/val/labels.csv")).set_index("id")["label"]
    pred = pd.read_csv(os.path.join(ROOT, "out/val_exp", a.method, "val_pred.csv")).set_index("id")["label"]
    names = [line.split()[2].replace("_", " ") for line in open(os.path.join(ROOT, "starter/classes.txt"))]
    ids = stream["id"].to_numpy()
    y, p = labels.loc[ids].to_numpy(), pred.loc[ids].to_numpy()
    bounds = np.load(os.path.join(ROOT, "tools/dev/val_bounds.npy"))
    dom = np.searchsorted(bounds, np.arange(len(ids)), side="right") - 1

    rng = np.random.default_rng(a.seed)
    pos = np.arange(len(ids))
    rows = [("Correct", pick(pos[p == y], dom, 4, rng)), ("Misclassified", pick(pos[p != y], dom, 4, rng))]

    fig, axes = plt.subplots(2, 4, figsize=(12, 7.8))
    for r, (kind, sel) in enumerate(rows):
        for c, i in enumerate(sel):
            ax = axes[r, c]
            ax.imshow(Image.open(os.path.join(ROOT, "data/val/images", ids[i] + ".png")).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            color = "tab:green" if p[i] == y[i] else "tab:red"
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(3)
            ax.set_title(f"pred: {names[p[i]]}\nGT: {names[y[i]]}", fontsize=11, color=color)
            ax.set_xlabel(f"#{i} · {DOMAINS[dom[i]]}", fontsize=9)
        axes[r, 0].set_ylabel(kind, fontsize=13)
    acc = (p == y).mean()
    fig.suptitle(f"Validation predictions of {a.method} (val accuracy {100 * acc:.2f} %)", fontsize=13)
    fig.tight_layout(h_pad=2.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    for kind, sel in rows:
        for i in sel:
            print(f"{kind:13s} #{i:4d} {DOMAINS[dom[i]]:16s} pred {names[p[i]]:16s} GT {names[y[i]]}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
