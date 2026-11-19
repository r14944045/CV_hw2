"""MixTTA: test-time adaptation for label-skewed streams with abrupt domain shifts.
Steps 1-3 are forward-only; step 4 (off by default, lr=0) adds self-training. Every step can be
switched off for ablations.

For each incoming batch x:
  1. Statistics (mixbn.MixBN2d): one train-mode forward updates the tracked test statistics with x, then
     normalises x with (1 - w) * tracked + w * source statistics. The update comes before the
     prediction, so the first batch after a domain shift already uses the new domain's statistics.
     (BOSA and RoTTA predict first and update after; with BOSA's statistics alone, on val, that
     one-batch lag cost about 3 points. `update_first=False` measures it for this method.)
  2. Averaging: mean softmax over x, its horizontal flip and `n_shift` random flip + shift copies
     (reflection padding, up to 1/8 of the image). The copies are normalised with the stored stats and
     do not update them.
  3. LAME (lame.lame): corrects the averaged probabilities with a kNN graph on the batch's penultimate
     features (of x, or averaged over all views with lame_feats="mean"). This exploits the few classes
     present in each batch. The argmax of the corrected probabilities is the prediction.
  4. Self-training (lr > 0): after predicting, `train_steps` Adam steps on the BN scale/shift (gamma, beta)
     only. The loss is the soft cross-entropy between the LAME-corrected probabilities (target) and the
     model's output on a random flip + shift copy of x. Training forwards do not update the statistics.
"""
import torch
import torch.nn.functional as F

from .common import setup_cuda
from .lame import lame
from .losses import soft_cross_entropy
from .mixbn import MixBN2d, configure_mix_norm


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
                 update_first=True, lame_feats="orig", lr=0.0, train_steps=1):
        setup_cuda()
        self.model = configure_mix_norm(model, alpha, w).requires_grad_(False)
        self.flip, self.n_shift, self.use_lame, self.knn = flip, n_shift, use_lame, knn
        self.update_first, self.lame_feats, self.train_steps = update_first, lame_feats, train_steps
        self._feats = None
        self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0].detach()))
        self.opt = None
        if lr > 0:
            params = [p for l in self.model.modules() if isinstance(l, MixBN2d) for p in (l.weight, l.bias)]
            for p in params:
                p.requires_grad_(True)
            self.opt = torch.optim.Adam(params, lr=lr)

    def __call__(self, x):
        m = self.model
        with torch.no_grad():
            if self.update_first:
                logits = m.train()(x)  # update the statistics with x, normalise x with them
            else:  # predict with the previous statistics; update after all forwards (BOSA / RoTTA order)
                logits = m.eval()(x)
            feats = [self._feats]
            m.eval()
            probs = [logits.softmax(1)]
            if self.flip:
                probs.append(m(x.flip(3)).softmax(1))
                feats.append(self._feats)
            for _ in range(self.n_shift):
                probs.append(m(random_shift(x)).softmax(1))
                feats.append(self._feats)
            if not self.update_first:
                m.train()(x)
                m.eval()
            prob = torch.stack(probs).mean(0)
            if self.use_lame:
                f = feats[0] if self.lame_feats == "orig" else torch.stack(feats).mean(0)
                prob = lame(prob, f, self.knn)
        if self.opt is not None:
            self.self_train(x, prob)
        return prob.argmax(1)

    def self_train(self, x, target):
        """Step 4: eval mode, so the training forwards use (and do not update) the stored statistics."""
        m = self.model.eval()
        for _ in range(self.train_steps):
            loss = soft_cross_entropy(m(random_shift(x)), target).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
