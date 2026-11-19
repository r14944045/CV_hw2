"""Compare registered methods on shuffled, JPEG-compressed and photometrically perturbed validation images."""
import argparse
from io import BytesIO
import json

import numpy as np
from PIL import Image
import torch

import fe


def jpeg_images(images):
    result = []
    for image in images:
        with BytesIO() as buffer:
            rgb = image.permute(1, 2, 0).numpy()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=35)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                result.append(torch.from_numpy(np.array(decoded.convert("RGB"))).permute(2, 0, 1))
    return torch.stack(result)


def contrast_and_noise(images):
    rng = torch.Generator().manual_seed(113)
    result = []
    for batch in images.split(64):
        pixels = batch.float() / 255
        contrast = 0.55 + 0.65 * torch.rand(len(batch), 1, 1, 1, generator=rng)
        pixels = (pixels - 0.5) * contrast + 0.5
        pixels += 0.04 * torch.randn(pixels.shape, generator=rng)
        result.append((pixels.clamp(0, 1) * 255).round().byte())
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="+", choices=sorted(fe.METHODS))
    args = parser.parse_args()
    data = fe.data("val")
    bounds = data["bounds"]
    rng = np.random.default_rng(17)
    shuffled = np.concatenate([rng.permutation(np.arange(lo, hi)) for lo, hi in zip(bounds, bounds[1:])])
    labels = data["y"].numpy()
    results = {"n": len(labels), "seed": 0, "correct": {}}
    for name, order, transform in (("shuffle", shuffled, None), ("jpeg", None, jpeg_images),
                                   ("photometric", None, contrast_and_noise)):
        images = transform(data["x"]) if transform else None
        expected = labels if order is None else labels[order]
        results["correct"][name] = {}
        for method in args.methods:
            predictions = fe.run(fe.METHODS[method], "val", order, seed=0, x=images)
            results["correct"][name][method] = int((predictions == expected).sum())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
