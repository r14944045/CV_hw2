"""Depth-dependent source weight for the mixed BN of MixTTA-MS (idea: normalisation_depth_schedule).

The base normalises every BN layer with mean = (1-w) mean_t + w mean_s (same for var), w = 0.5. Here:
  - w_stage: a fixed w per stage, keyed by the top-level module name: "bn1" (stem), "layer1".."layer4"
    (the downsample BNs of a stage belong to it). Unlisted stages keep the base w.
  - w_stem_mean / w_stem_var: separate weights for the stem's mean and variance (the stem BN's class is
    swapped in place to MVMixBN2d; buffers/parameters are untouched, so the optimiser stays valid).
  - ramp_rows = r0 > 0: a layer whose target weight is below 0.5 uses max(target, 0.5 r0 / (r0 + n_dom)),
    n_dom = rows the tracker has seen since the last detected shift (including the current batch), so a new
    domain whose statistics come from only a few rows keeps about half source statistics.
All options at their defaults (None / 0): nothing is changed, identical to MixTTAMS.
"""
import torch

from tta.mixbn import MultiMixBN2d
from tta.mixtta_ms import MixTTAMS

STAGES = ("bn1", "layer1", "layer2", "layer3", "layer4")


class MVMixBN2d(MultiMixBN2d):
    """MultiMixBN2d with separate source weights for the mean (w_m) and the variance (w_v)."""

    def forward(self, x):
        mean_t, var_t = getattr(self, f"mean{self.k}"), getattr(self, f"var{self.k}")
        if self.training:
            xs = x if self.stat_rows is None else x[:self.stat_rows]
            b_var, b_mean = torch.var_mean(xs.detach().float(), dim=(0, 2, 3), unbiased=False)
            mean_t = (1 - self.alpha) * mean_t + self.alpha * b_mean
            var_t = (1 - self.alpha) * var_t + self.alpha * b_var
            setattr(self, f"mean{self.k}", mean_t)
            setattr(self, f"var{self.k}", var_t)
        mean = (1 - self.w_m) * mean_t + self.w_m * self.src_mean
        var = (1 - self.w_v) * var_t + self.w_v * self.src_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.to(x.dtype).view(1, -1, 1, 1) + shift.to(x.dtype).view(1, -1, 1, 1)


class MixTTADepthW(MixTTAMS):
    def __init__(self, model, w_stage=None, w_stem_mean=None, w_stem_var=None, ramp_rows=0, **kw):
        super().__init__(model, **kw)
        w0 = float(kw.get("w", 0.5))
        w_stage = dict(w_stage or {})
        bad = set(w_stage) - set(STAGES)
        if bad:
            raise ValueError(f"unknown stages {bad}; use {STAGES}")
        self.ramp_rows = float(ramp_rows)
        self._targets = []  # (layer, attribute, target weight)
        counts = dict.fromkeys(STAGES, 0)
        for name, l in self.model.named_modules():
            if not isinstance(l, MultiMixBN2d):
                continue
            stage = name.split(".")[0]
            assert stage in STAGES, name
            counts[stage] += 1
            if stage in w_stage:
                l.w = float(w_stage[stage])
                self._targets.append((l, "w", l.w))
            if stage == "bn1" and (w_stem_mean is not None or w_stem_var is not None):
                ws = float(w_stage.get("bn1", w0))
                l.__class__ = MVMixBN2d
                l.w_m = ws if w_stem_mean is None else float(w_stem_mean)
                l.w_v = ws if w_stem_var is None else float(w_stem_var)
                self._targets = [t for t in self._targets if t[0] is not l]
                self._targets += [(l, "w_m", l.w_m), (l, "w_v", l.w_v)]
        assert sum(counts.values()) == len(self.layers), counts
        self.stage_counts = counts
        self._ramped = [t for t in self._targets if t[2] < 0.5] if self.ramp_rows > 0 else []

    def _tracking_rate(self, x):
        alpha = super()._tracking_rate(x)
        if self._ramped:
            wr = 0.5 * self.ramp_rows / (self.ramp_rows + self.n_dom)
            for l, attr, target in self._ramped:
                setattr(l, attr, max(target, wr))
        return alpha
