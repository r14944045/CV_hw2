"""MixTTA-MS: multi-scale MixTTA with self-training, in bf16.

Same steps as MixTTA (mixtta.py), run at several input sizes, with three changes:
  - Scales: the batch is also resized (bilinear) to larger sizes, e.g. 160 -> 192 or 224. The source
    model was trained at 160 px; at a larger input size objects look bigger, closer to the random-resized
    crops it saw in training (the "FixRes" train/test resolution effect). Every scale has its own
    statistics tracker, because the activation statistics depend on the input size.
  - Fused views: per scale, x, its flip and the shifted copies go through the network as one batch. Each
    MultiMixBN2d layer updates its tracker from the rows of x only and normalises all rows with the
    updated statistics (the same result as a train-mode forward of x followed by eval-mode forwards of
    the views, in one call).
  - Mixed precision: forwards and backwards run under bf16 autocast with channels-last tensors.
    Statistics, LAME and the loss stay in float32.

For each incoming batch x:
  1. For each size s: update tracker s with x resized to s, and get the softmax of x, its flip and
     `n_shift` random flip + shift copies (mixbn.MultiMixBN2d normalisation with alpha, w).
  2. Average the probabilities over views and scales, and the penultimate features over views and scales.
  3. LAME on the averaged probabilities with a kNN graph on the averaged features. Predict the argmax.
  4. Self-training: `steps` Adam steps on the BN scale/shift (gamma, beta), soft cross-entropy between
     the LAME output and the model's output on a random flip + shift copy of x at size
     sizes[train_scale] (the native 160 by default; training at a larger size drifted on long streams).
"""
import torch
import torch.nn.functional as F

from .common import setup_cuda
from .lame import lame
from .mixbn import MultiMixBN2d, configure_multi_mix_norm
from .mixtta import random_shift


class MixTTAMS:
    def __init__(self, model, alpha=0.4, w=0.5, sizes=(160, 224), flip=True, n_shift=2, knn=5, lame_T=2.0,
                 lr=1e-3, steps=3, train_scale=0, amp=True, deterministic=True):
        setup_cuda()
        if deterministic:  # fixed cuDNN algorithms: reruns give the same predictions (slower)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.sizes = list(sizes)
        self.model = configure_multi_mix_norm(model, alpha, w, len(self.sizes)).requires_grad_(False)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.layers = [m for m in self.model.modules() if isinstance(m, MultiMixBN2d)]
        self.flip, self.n_shift, self.knn, self.lame_T = flip, n_shift, knn, lame_T
        self.steps, self.train_scale = steps, train_scale
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
        self._feats = None
        self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0].detach()))
        self.opt = None
        if lr > 0 and steps > 0:
            params = [p for l in self.layers for p in (l.weight, l.bias)]
            for p in params:
                p.requires_grad_(True)
            self.opt = torch.optim.Adam(params, lr=lr)

    def _set_tracker(self, k, stat_rows=None):
        for l in self.layers:
            l.k, l.stat_rows = k, stat_rows

    @staticmethod
    def _resize(x, s):
        return x if s == x.size(-1) else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def _scale_forward(self, xs, k):
        """Step 1 at one scale: returns (mean prob over views, mean features over views)."""
        B = xs.size(0)
        views = [xs] + ([xs.flip(3)] if self.flip else []) + [random_shift(xs) for _ in range(self.n_shift)]
        self._set_tracker(k, stat_rows=B)
        with self.amp:
            logits = self.model.train()(torch.cat(views).contiguous(memory_format=torch.channels_last))
        self.model.eval()
        self._set_tracker(k)
        V = len(views)
        return logits.float().softmax(1).view(V, B, -1).mean(0), self._feats.float().view(V, B, -1).mean(0)

    def __call__(self, x):
        out = [self._scale_forward(self._resize(x, s), k) for k, s in enumerate(self.sizes)]
        prob = torch.stack([p for p, _ in out]).mean(0)
        feats = torch.stack([f for _, f in out]).mean(0)
        if self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        z = lame(prob, feats, self.knn)
        if self.opt is not None:
            self.self_train(x, z)
        return z.argmax(1)

    def self_train(self, x, target):
        """Step 4: eval mode (the stored statistics of tracker `train_scale` are used, not updated)."""
        k = self.train_scale
        self._set_tracker(k)
        xs = self._resize(x, self.sizes[k])
        m = self.model.eval()
        for _ in range(self.steps):
            with self.amp:
                logits = m(random_shift(xs).contiguous(memory_format=torch.channels_last))
            loss = -(target * logits.float().log_softmax(1)).sum(1).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
