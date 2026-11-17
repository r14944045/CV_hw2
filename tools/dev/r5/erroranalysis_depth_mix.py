"""Depth-dependent source/test statistics mix per BN layer, on top of mixtta_ms224sp2cpr.

MixTTAMS normalises every MultiMixBN2d layer with (1 - w) * tracked test statistics + w * source statistics, with
one global w = 0.5. Here w depends on the ResNet stage of the layer:

  w_stage = (w_stem, w_layer1, w_layer2, w_layer3, w_layer4)

Early layers (stem, layer1, layer2): statistics dominated by colour / texture / overlay patterns, nearly
class-independent, well estimated from the domain's running mean -> small w (lean on the test statistics).
Deep layers (layer3, layer4): class-dependent statistics, biased by the Dirichlet label bursts -> larger w
(the source statistics are the only class-balanced reference).

Layers are grouped by module name ('bn1' -> stem, 'layerN.*' -> stage N, including 'layerN.0.downsample.1').
MultiMixBN2d.forward reads self.w per instance, so the schedule costs nothing.

Optional gap-adaptive variant (gap_kappa = 5 per-stage constants): before each batch,
  w_l = w_stage * exp(-KL_l / kappa_stage),
where KL_l is the channel-mean Gaussian KL of layer l's tracked statistics (tracker 0, left by the previous batch)
from the source statistics. On a batch where a shift was detected (tracked statistics reset) the plain schedule
is used. w_l stays a GPU tensor (no host sync).

gap_log=True records the per-stage median KL (tracker 0) after every batch, for calibrating kappa.

Off switch: w_stage=None (and gap_kappa=None) -> identical to MixTTAMS.
"""
import math

import torch

from tta.mixtta_ms import MixTTAMS
from tta.mixbn import MultiMixBN2d


def _stage(name):
    if name == "bn1":
        return 0
    if name.startswith("layer"):
        return int(name[5])
    raise ValueError(f"unexpected BN module name {name}")


class DepthMix(MixTTAMS):
    def __init__(self, model, w_stage=None, gap_kappa=None, gap_log=False, **kw):
        super().__init__(model, **kw)
        self.w_stage, self.gap_kappa, self.gap_log = w_stage, gap_kappa, gap_log
        named = [(n, m) for n, m in self.model.named_modules() if isinstance(m, MultiMixBN2d)]
        assert [m for _, m in named] == self.layers
        self.stage_of = [_stage(n) for n, _ in named]
        if w_stage is not None:
            assert len(w_stage) == 5
            for l, s in zip(self.layers, self.stage_of):
                l.w = float(w_stage[s])
        self.kl_log = []

    @torch.no_grad()
    def _layer_kl(self):
        """Per layer: channel-mean KL(N(tracked0) || N(source)), as a list of 0-dim GPU tensors."""
        out = []
        for l in self.layers:
            mt, vt, ms, vs = l.mean0, l.var0, l.src_mean, l.src_var
            kl = 0.5 * ((vt + (mt - ms) ** 2) / (vs + 1e-5) - 1 - torch.log((vt + 1e-5) / (vs + 1e-5)))
            out.append(kl.mean())
        return out

    def _tracking_rate(self, x):
        alpha = super()._tracking_rate(x)
        if self.gap_kappa is not None and self.w_stage is not None:
            shifted = bool(self.shifts) and self.shifts[-1] == self.t - 1
            if shifted:
                for l, s in zip(self.layers, self.stage_of):
                    l.w = float(self.w_stage[s])
            else:
                kls = self._layer_kl()
                for l, s, kl in zip(self.layers, self.stage_of, kls):
                    l.w = self.w_stage[s] * torch.exp(-kl / self.gap_kappa[s])
        return alpha

    def __call__(self, x):
        pred = super().__call__(x)
        if self.gap_log:
            kls = torch.stack(self._layer_kl()).cpu()
            st = torch.tensor(self.stage_of)
            self.kl_log.append([float(kls[st == s].median()) for s in range(5)])
        return pred
