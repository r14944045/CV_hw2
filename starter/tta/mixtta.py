"""MixTTA: forward-only test-time adaptation for label-skewed streams with abrupt domain shifts.
No gradient updates. Every step can be switched off for ablations.

For each incoming batch x:
  1. Statistics (mixbn.MixBN2d): one train-mode forward updates the tracked test statistics with x, then
     normalises x with (1 - w) * tracked + w * source statistics. The update comes before the
     prediction, so the first batch after a domain shift already uses the new domain's statistics.
     (BOSA and RoTTA predict first and update after; with BOSA's statistics alone, on val, that
     one-batch lag cost about 3 points. `update_first=False` measures it for this method.)
  2. Averaging (aug_avg): mean softmax over x, its horizontal flip and `n_shift` random flip + shift copies
     (reflection padding, up to 1/8 of the image). The copies are normalised with the stored stats and
     do not update them.
  3. LAME (lame.lame): corrects the averaged probabilities with a kNN graph on the batch's penultimate
     features. This exploits the few classes present in each batch.
"""
import torch
import torch.nn.functional as F

from .common import setup_cuda
from .lame import lame
from .mixbn import configure_mix_norm


def random_shift(x, frac=1 / 8):
    """Random horizontal flip and translation of up to `frac` of the image size (reflection padding).
    One random offset per batch."""
    pad = int(x.size(-1) * frac)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i, j = torch.randint(0, 2 * pad + 1, (2,)).tolist()
    out = xp[..., i:i + x.size(-2), j:j + x.size(-1)]
    return out.flip(3) if torch.rand(1).item() < 0.5 else out


class MixTTA:
    def __init__(self, model, alpha=0.2, w=0.5, flip=True, n_shift=4, use_lame=True, knn=5,
                 update_first=True):
        setup_cuda()
        self.model = configure_mix_norm(model, alpha, w).requires_grad_(False)
        self.flip, self.n_shift, self.use_lame, self.knn = flip, n_shift, use_lame, knn
        self.update_first = update_first
        self._feats = None
        self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0]))

    @torch.no_grad()
    def __call__(self, x):
        m = self.model
        if self.update_first:
            logits = m.train()(x)  # update the statistics with x, normalise x with them
        else:  # predict with the previous statistics; update after all forwards (BOSA / RoTTA order)
            logits = m.eval()(x)
        feats = self._feats
        m.eval()
        probs = [logits.softmax(1)]
        if self.flip:
            probs.append(m(x.flip(3)).softmax(1))
        for _ in range(self.n_shift):
            probs.append(m(random_shift(x)).softmax(1))
        if not self.update_first:
            m.train()(x)
            m.eval()
        prob = torch.stack(probs).mean(0)
        if self.use_lame:
            prob = lame(prob, feats, self.knn)
        return prob.argmax(1)
