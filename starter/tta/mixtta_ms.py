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
  0. (shift_thr > 0) Domain-shift detection: the KL divergence between the batch's conv1-output statistics and
     the first BN layer's tracked statistics, divided by its median over the last 8 batches of the current
     domain. Above shift_thr (10), the batch starts a new domain: its statistics replace the tracked ones
     (tracking rate 1) in every layer and at every size. The test domains change abruptly; with a fixed
     rate of 0.4 the first batches of a new domain are normalised with a mix of the old domain's statistics.
  0b. (alpha_min set) Within a domain, the tracking rate is 1/k for the k-th batch since the last detected
     shift (a running mean of the domain's batch statistics, which are stationary within a domain but
     noisy from batch to batch because of the label skew), floored at alpha_min.
  1. For each size s: update tracker s with x resized to s, and get the softmax of x, its flip and
     `n_shift` random flip + shift copies (mixbn.MultiMixBN2d normalisation with alpha, w).
  2. Average the probabilities over views and scales, and the penultimate features over views and scales.
  3. LAME on the averaged probabilities with a kNN graph on the averaged features. Predict the argmax.
  3b. (prior_lambda > 0) Class-prior correction, before LAME: the stream is class-balanced overall, but the
     model over-predicts some classes under corruption (tree frog, container ship) and under-predicts
     others (tabby, sports car). Count the predicted labels so far, pi_c = (count_c + tau/C) / (n + tau), and
     multiply the probabilities by (C * pi_c)^(-prior_lambda). tau (in samples) delays the correction at
     the start of the stream, when the counts follow the label-skewed stream rather than the model's bias.
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
                 lr=1e-3, steps=3, train_scale=0, amp=True, deterministic=True, prior_lambda=0.0, prior_tau=2000.0,
                 shift_thr=0.0, alpha_min=None, extra_train=(), extra_lr=1e-4, temporal_mu=0.0, temporal_eps=0.1):
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
        self.prior_lambda, self.prior_tau = prior_lambda, prior_tau
        self.counts, self.n_seen = None, 0
        self.alpha, self.shift_thr, self.alpha_min = alpha, shift_thr, alpha_min
        self.dist_hist, self.shifts, self.k_dom, self.t = [], [], 0, 0
        self.temporal_mu, self.temporal_eps, self.h_prev = temporal_mu, temporal_eps, None
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
        self._feats = None
        self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0].detach()))
        self.opt = None
        if lr > 0 and steps > 0:
            params = [p for l in self.layers for p in (l.weight, l.bias)]
            # extra_train: names of submodules (e.g. "fc", "layer4") whose remaining weights also train, at extra_lr
            extra = [p for n in extra_train for name, p in getattr(self.model, n).named_parameters()
                     if not any(p is q for q in params)]
            for p in params + extra:
                p.requires_grad_(True)
            groups = [dict(params=params, lr=lr)] + ([dict(params=extra, lr=extra_lr)] if extra else [])
            self.opt = torch.optim.Adam(groups, lr=lr)

    def _set_tracker(self, k, stat_rows=None, alpha=None):
        for l in self.layers:
            l.k, l.stat_rows = k, stat_rows
            l.alpha = self.alpha if alpha is None else alpha

    @torch.no_grad()
    def _shift_score(self, x):
        """Step 0: mean Gaussian KL (over the 64 conv1 channels) of this batch's conv1 output statistics from the
        first BN layer's tracked statistics (tracker 0)."""
        h = self.model.conv1(x.contiguous(memory_format=torch.channels_last)).float()
        v, m = torch.var_mean(h, dim=(0, 2, 3), unbiased=False)
        bn = self.layers[0]
        mt, vt = bn.mean0, bn.var0
        kl = 0.5 * ((v + (m - mt) ** 2) / (vt + 1e-5) - 1 - torch.log((v + 1e-5) / (vt + 1e-5)))
        return kl.mean().item()

    def _tracking_rate(self, x):
        """Steps 0a-0b: detect a domain shift and pick this batch's tracking rate (None: the fixed alpha)."""
        alpha = None
        if self.shift_thr:
            d = self._shift_score(x)
            hist = self.dist_hist[-8:]
            ref = sorted(hist)[len(hist) // 2] if len(hist) >= 3 else None
            if ref is not None and d > self.shift_thr * max(ref, 1e-4):
                self.shifts.append(self.t)
                self.h_prev = None
                self.dist_hist, self.k_dom, alpha = [], 0, 1.0
            else:
                self.dist_hist.append(d)
        if self.alpha_min is not None:
            alpha = max(self.alpha_min, 1.0 / (self.k_dom + 1))
        self.k_dom += 1
        self.t += 1
        return alpha

    @staticmethod
    def _resize(x, s):
        return x if s == x.size(-1) else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def _scale_forward(self, xs, k, alpha=None):
        """Step 1 at one scale: returns (mean prob over views, mean features over views)."""
        B = xs.size(0)
        views = [xs] + ([xs.flip(3)] if self.flip else []) + [random_shift(xs) for _ in range(self.n_shift)]
        self._set_tracker(k, stat_rows=B, alpha=alpha)
        with self.amp:
            logits = self.model.train()(torch.cat(views).contiguous(memory_format=torch.channels_last))
        self.model.eval()
        self._set_tracker(k)
        V = len(views)
        return logits.float().softmax(1).view(V, B, -1).mean(0), self._feats.float().view(V, B, -1).mean(0)

    def __call__(self, x):
        alpha = self._tracking_rate(x)
        out = [self._scale_forward(self._resize(x, s), k, alpha) for k, s in enumerate(self.sizes)]
        prob = torch.stack([p for p, _ in out]).mean(0)
        feats = torch.stack([f for _, f in out]).mean(0)
        C = prob.size(1)
        if self.counts is None:
            self.counts = torch.zeros(C, device=prob.device)
        if self.prior_lambda:
            pi = (self.counts + self.prior_tau / C) / (self.n_seen + self.prior_tau)
            prob = prob * (C * pi).pow(-self.prior_lambda)
            prob = prob / prob.sum(1, keepdim=True)
        if self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        if self.temporal_mu and self.h_prev is not None:  # 3c: classes of the previous batch are likely again
            prob = prob * (self.h_prev + self.temporal_eps).pow(self.temporal_mu)
            prob = prob / prob.sum(1, keepdim=True)
        z = lame(prob, feats, self.knn)
        self.h_prev = z.mean(0)
        pred = z.argmax(1)
        self.counts += torch.bincount(pred, minlength=C).float()
        self.n_seen += pred.numel()
        if self.opt is not None:
            self.self_train(x, z)
        return pred

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
