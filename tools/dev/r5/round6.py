"""Round 6 ideas on top of MixTTAMS (use with the sp2cprl3 settings). All options off = MixTTAMS exactly.

st224 = n:    the last n of the self-training steps run at 224 px with tracker 1 (instead of 160 px / tracker 0).
fuse_g = g:   the two sizes' probabilities are fused with weights proportional to conf_s^g, where conf_s is the
              running mean (over the current domain's batches) of size s's mean max-probability; the features are
              still averaged equally. Reset at a detected shift.
lame_q = q:   instead of a fixed T=2, each batch's temperature is set (bisection, T in [1, 3]) so that the mean
              max-probability of the softened, prior-corrected probabilities equals q.
No new random draws.
"""
import torch

from tta.lame import lame
from tta.mixtta import random_shift
from tta.mixtta_ms import MixTTAMS


class Round6(MixTTAMS):
    def __init__(self, model, st224=0, fuse_g=None, lame_q=None, **kw):
        super().__init__(model, **kw)
        self.st224, self.fuse_g, self.lame_q = st224, fuse_g, lame_q
        self._conf_sum, self._conf_n, self._fuse_upd, self._last_shifts = None, 0, False, 0

    def __call__(self, x):
        self._fuse_upd = True
        return super().__call__(x)

    def _fuse(self, out):
        probs = torch.stack([p for p, _ in out])
        if self.fuse_g is None:
            return probs.mean(0)
        conf = probs.max(2).values.mean(1)  # per size
        if len(self.shifts) != self._last_shifts:  # new domain: restart the running confidence
            self._conf_sum, self._conf_n, self._last_shifts = None, 0, len(self.shifts)
        if self._fuse_upd:
            self._conf_sum = conf if self._conf_sum is None else self._conf_sum + conf
            self._conf_n += 1
            self._fuse_upd = False
        c = self._conf_sum / max(1, self._conf_n) if self._conf_sum is not None else conf
        w = c.pow(self.fuse_g); w = w / w.sum()
        return (w.view(-1, 1, 1) * probs).sum(0)

    @staticmethod
    def _temper_to(prob, q):
        logp = torch.log(prob + 1e-10)
        lo, hi = 1.0, 3.0
        for _ in range(20):
            T = 0.5 * (lo + hi)
            m = torch.softmax(logp / T, 1).max(1).values.mean()
            if m > q:
                lo = T
            else:
                hi = T
        return torch.softmax(logp / (0.5 * (lo + hi)), 1)

    def _predict(self, out):
        prob = self._fuse(out)
        feats = torch.stack([f for _, f in out]).mean(0)
        C = prob.size(1)
        if self.counts is None:
            self.counts = torch.zeros(C, device=prob.device)
        if self.prior_lambda:
            pi = (self.counts + self.prior_tau / C) / (self.n_seen + self.prior_tau)
            prob = prob * (C * pi).pow(-self.prior_lambda)
            prob = prob / prob.sum(1, keepdim=True)
        if self.lame_q is not None:
            prob = self._temper_to(prob, self.lame_q)
        elif self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        z = lame(prob, feats, self.knn)
        self.h_prev = z.mean(0)
        return z

    def self_train(self, x, target, rows=0, steps=None):
        n = self.steps if steps is None else steps
        if not self.st224:
            return super().self_train(x, target, rows, steps)
        m = self.model.eval()
        for i in range(n):
            k = 1 if i >= n - self.st224 else self.train_scale
            self._set_tracker(k)
            xs = self._resize(x, self.sizes[k])
            xv = random_shift(xs)[rows:]
            with self.amp:
                logits = m(xv.contiguous(memory_format=torch.channels_last))
            loss = -(target[rows:] * logits.float().log_softmax(1)).sum(1).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
