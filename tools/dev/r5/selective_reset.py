"""Selective BN-statistics reset at a detected domain shift, on top of MixTTAMS (use with the sp2cprl3 settings).

reset_upto = S: layers in stages 0..S (0 = stem, 1..4 = layer1..layer4) restart at a shift as in the base
(tracking rate 1, then a running mean over the new domain). The deeper layers instead:
  mode="keep": continue their old running mean at the floor rate alpha_min for the whole new domain;
  mode="soft": restart softly, with rate `soft` on the shift batch instead of 1, then the base schedule.
reset_upto=None: identical to MixTTAMS. Adds no time and no random draws.
"""
from tta.mixtta_ms import MixTTAMS


class SelectiveReset(MixTTAMS):
    def __init__(self, model, reset_upto=None, mode="keep", soft=0.5, **kw):
        super().__init__(model, **kw)
        self.reset_upto, self.mode, self.soft = reset_upto, mode, soft
        names = {id(m): n for n, m in self.model.named_modules()}
        self._stage = [0 if names[id(l)] == "bn1" else int(names[id(l)][5]) for l in self.layers]

    def _set_tracker(self, k, stat_rows=None, alpha=None):
        super()._set_tracker(k, stat_rows, alpha)
        if alpha is None or self.reset_upto is None or not self.shifts:
            return
        for l, st in zip(self.layers, self._stage):
            if st <= self.reset_upto:
                continue
            if self.mode == "keep":
                l.alpha = min(alpha, self.alpha_min)
            elif self.mode == "soft" and self.k_dom == 1:
                l.alpha = min(alpha, self.soft)
