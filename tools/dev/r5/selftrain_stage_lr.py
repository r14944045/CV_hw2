"""Stage-wise learning rates on the BN affine parameters (surgical adaptation) on top of mixtta_ms224sp2cpr.

The single Adam group (all 53 BN gamma/beta at lr) is split into per-stage groups: stem (bn1), layer1..layer4
(downsample BNs included in their stage). Each stage's lr is lr * stage_lr[stage]; a multiplier of 0 freezes that
stage's affine (SAR-style for layer4). stage_lr=None (default) keeps the base optimiser untouched (off switch).
All multipliers at 1 reproduce the base exactly (Adam is per-parameter). No new random draws.
"""
import torch

from tta.mixbn import MultiMixBN2d
from tta.mixtta_ms import MixTTAMS

STAGES = ("stem", "layer1", "layer2", "layer3", "layer4")


class SelfTrainStageLR(MixTTAMS):
    def __init__(self, model, stage_lr=None, **kw):
        super().__init__(model, **kw)
        self.stage_lr = stage_lr
        if stage_lr is None or self.opt is None:
            return
        assert not kw.get("extra_train"), "stage_lr is not combined with extra_train"
        assert set(stage_lr) <= set(STAGES), stage_lr
        base_lr = kw.get("lr", 1e-3)
        groups = {s: [] for s in STAGES}
        for name, mod in self.model.named_modules():
            if isinstance(mod, MultiMixBN2d):
                stage = "stem" if name == "bn1" else name.split(".")[0]
                groups[stage] += [mod.weight, mod.bias]
        self.stage_sizes = {s: len(ps) for s, ps in groups.items()}
        assert sum(self.stage_sizes.values()) == 2 * len(self.layers), self.stage_sizes
        opt_groups = []
        for s in STAGES:
            m = float(stage_lr.get(s, 1.0))
            if m == 0:
                for p in groups[s]:
                    p.requires_grad_(False)
            elif groups[s]:
                opt_groups.append(dict(params=groups[s], lr=base_lr * m))
        self.opt = torch.optim.Adam(opt_groups, lr=base_lr)
