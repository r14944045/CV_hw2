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
  0c. (split_shift) Most domain boundaries fall inside a batch, and the detector usually fires on that mixed
     batch. On a detected shift, one changepoint is fitted to the per-sample conv1 statistics of the previous
     batch followed by this one (least squares on standardised channel means and log-variances). The first j
     rows (the old domain) are normalised with the previous domain's statistics, the new domain's statistics
     start from the remaining rows only, and the running mean of step 0b counts rows instead of batches.
     Self-training on that batch uses the new rows only.
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
  5. (refine_batches = R) On the first R batches of a domain (not on a split batch), step 4 runs before the
     prediction: the batch is predicted, the model self-trains on that prediction, and the same views are
     predicted again (eval mode) with the updated gamma/beta. Costs one extra forward on those batches only.
"""
import math

import torch
import torch.nn.functional as F

from .common import setup_cuda
from .lame import lame
from .mixbn import MultiMixBN2d, configure_multi_mix_norm
from .mixtta import random_shift


class MixTTAMS:
    def __init__(self, model, alpha=0.4, w=0.5, sizes=(160, 224), flip=True, n_shift=2, knn=5, lame_T=2.0,
                 lr=1e-3, steps=3, train_scale=0, amp=True, deterministic=True, prior_lambda=0.0, prior_tau=2000.0,
                 shift_thr=0.0, alpha_min=None, extra_train=(), extra_lr=1e-4, temporal_mu=0.0, temporal_eps=0.1,
                 split_shift=False, split_min=4, split_train="new", refine_batches=0,
                 st_weight=None, st_pow=1.0, st_ent=None, ema_pred=0.0, refine_steps=None):
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
        self.split_shift, self.split_min, self.split_train = split_shift, split_min, split_train
        self.n_dom, self.f_prev, self.j_old, self.splits = 0, None, 0, []
        self.refine_batches, self._views = refine_batches, {}
        self.st_weight, self.st_pow, self.st_ent, self.ema_pred = st_weight, st_pow, st_ent, ema_pred
        self.refine_steps = refine_steps  # self-training steps on the second-pass batches (None: steps)
        self.temporal_mu, self.temporal_eps, self.h_prev = temporal_mu, temporal_eps, None
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
        self._feats = None
        self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0].detach()))
        self.opt, self._ema, self._live = None, None, None
        if lr > 0 and steps > 0:
            params = [p for l in self.layers for p in (l.weight, l.bias)]
            # extra_train: names of submodules (e.g. "fc", "layer4") whose remaining weights also train, at extra_lr
            extra = [p for n in extra_train for name, p in getattr(self.model, n).named_parameters()
                     if not any(p is q for q in params)]
            for p in params + extra:
                p.requires_grad_(True)
            groups = [dict(params=params, lr=lr)] + ([dict(params=extra, lr=extra_lr)] if extra else [])
            self.opt = torch.optim.Adam(groups, lr=lr)
            self._params = params
            self._ema = [p.detach().clone() for p in params] if ema_pred else None
            self._live = None

    @torch.no_grad()
    def _use_ema(self, on):
        """ema_pred: predict with an exponential moving average of the BN scale/shift, train the live copy."""
        if self._ema is None:
            return
        if on and self._live is None:
            self._live = [p.detach().clone() for p in self._params]
            for p, e in zip(self._params, self._ema):
                p.copy_(e)
        elif not on and self._live is not None:
            for p, q in zip(self._params, self._live):
                p.copy_(q)
            self._live = None

    @torch.no_grad()
    def _update_ema(self):
        if self._ema is not None:
            for p, e in zip(self._params, self._ema):
                e.mul_(self.ema_pred).add_(p.detach(), alpha=1 - self.ema_pred)

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
        if self.split_shift:  # per-sample conv1 statistics for the in-batch changepoint
            vs, ms = torch.var_mean(h, dim=(2, 3), unbiased=False)
            self._f_cur = torch.cat([ms, torch.log(vs + 1e-5)], 1)
        return kl.mean().item()

    def _changepoint(self):
        """Step 0c: number of rows of this batch that still belong to the previous domain. One changepoint is
        fitted (least squares on standardised per-sample conv1 statistics) to the previous batch followed by
        this batch; the previous batch is taken to be all old domain. 0: the whole batch is new."""
        if self.f_prev is None:
            return 0
        W = torch.cat([self.f_prev, self._f_cur]).double()
        W = (W - W.mean(0)) / (W.std(0) + 1e-6)
        n, P, B = W.size(0), self.f_prev.size(0), self._f_cur.size(0)
        S1 = torch.cat([W.new_zeros(1, W.size(1)), W.cumsum(0)])
        S2 = torch.cat([W.new_zeros(1), (W ** 2).sum(1).cumsum(0)])
        c = torch.arange(P, P + B - self.split_min + 1, device=W.device)  # change after row c of the window
        left = S2[c] - (S1[c] ** 2).sum(1) / c
        right = (S2[n] - S2[c]) - ((S1[n] - S1[c]) ** 2).sum(1) / (n - c)
        j = int((left + right).argmin()) 
        return j if j >= self.split_min else 0

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
                self.n_dom = 0
                if self.split_shift:
                    self.j_old = self._changepoint()
                    if self.j_old:
                        self.splits.append((self.t, self.j_old))
            else:
                self.dist_hist.append(d)
            if self.split_shift:
                self.f_prev = self._f_cur
        self.n_dom += x.size(0) - self.j_old
        if self.alpha_min is not None:
            if self.split_shift:  # running mean over the domain's rows
                alpha = max(self.alpha_min, (x.size(0) - self.j_old) / self.n_dom)
            else:
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
        xv = torch.cat(views).contiguous(memory_format=torch.channels_last)
        if self.refine_batches:
            self._views[k] = xv
        with self.amp:
            logits = self.model.train()(xv)
        self.model.eval()
        self._set_tracker(k)
        V = len(views)
        return logits.float().softmax(1).view(V, B, -1).mean(0), self._feats.float().view(V, B, -1).mean(0)

    @torch.no_grad()
    def _scale_forward_split(self, xs, k, j):
        """Step 1 for a batch whose first j rows belong to the previous domain: those rows are normalised with
        the previous domain's statistics (not updated); tracker k restarts from the other rows (rate 1)."""
        B = xs.size(0)
        views = [xs] + ([xs.flip(3)] if self.flip else []) + [random_shift(xs) for _ in range(self.n_shift)]
        V = len(views)
        self._set_tracker(k)
        with self.amp:
            lo = self.model.eval()(torch.cat([v[:j] for v in views]).contiguous(memory_format=torch.channels_last))
        fo = self._feats.float()
        self._set_tracker(k, stat_rows=B - j, alpha=1.0)
        with self.amp:
            ln = self.model.train()(torch.cat([v[j:] for v in views]).contiguous(memory_format=torch.channels_last))
        fn = self._feats.float()
        self.model.eval()
        self._set_tracker(k)
        prob = torch.cat([lo.float().softmax(1).view(V, j, -1), ln.float().softmax(1).view(V, B - j, -1)], 1)
        feats = torch.cat([fo.view(V, j, -1), fn.view(V, B - j, -1)], 1)
        return prob.mean(0), feats.mean(0)

    def __call__(self, x):
        self.j_old = 0
        alpha = self._tracking_rate(x)
        self._use_ema(True)
        if self.j_old:
            out = [self._scale_forward_split(self._resize(x, s), k, self.j_old) for k, s in enumerate(self.sizes)]
        else:
            out = [self._scale_forward(self._resize(x, s), k, alpha) for k, s in enumerate(self.sizes)]
        z = self._predict(out)
        refine = self.refine_batches and not self.j_old and self.k_dom <= self.refine_batches and self.opt is not None
        if refine:  # 5: first batches of a domain: train on this batch, then predict it again with the new gamma/beta
            self._use_ema(False)
            self.self_train(x, z, steps=self.refine_steps)
            self._update_ema()
            self._use_ema(True)
            z = self._predict([self._eval_views(k, x.size(0)) for k in range(len(self.sizes))])
        self._use_ema(False)
        pred = z.argmax(1)
        self.counts += torch.bincount(pred, minlength=z.size(1)).float()
        self.n_seen += pred.numel()
        if self.opt is not None and not refine:
            self.self_train(x, z, rows=self.j_old if self.split_train == "new" else 0)
            self._update_ema()
        return pred

    @torch.no_grad()
    def _eval_views(self, k, B):
        self._set_tracker(k)
        with self.amp:
            logits = self.model.eval()(self._views[k])
        V = logits.size(0) // B
        return logits.float().softmax(1).view(V, B, -1).mean(0), self._feats.float().view(V, B, -1).mean(0)

    def _predict(self, out):
        """Steps 2-3: fuse scales, class prior, temperature, LAME. Returns the LAME output z."""
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
        return z

    def _st_weights(self, t):
        """Per-image weights of the self-training loss (None: uniform). st_weight="conf": (max_c t)^st_pow.
        st_ent: drop images whose target entropy exceeds st_ent * log C."""
        if self.st_weight is None and self.st_ent is None:
            return None
        w = t.max(1).values.pow(self.st_pow) if self.st_weight == "conf" else torch.ones_like(t[:, 0])
        if self.st_ent is not None:
            H = -(t * torch.log(t + 1e-10)).sum(1)
            w = w * (H < self.st_ent * math.log(t.size(1))).float()
        return w

    def self_train(self, x, target, rows=0, steps=None):
        """Step 4: eval mode (the stored statistics of tracker `train_scale` are used, not updated).
        rows > 0: train on rows[rows:] only (the new domain's part of a split batch)."""
        k = self.train_scale
        self._set_tracker(k)
        xs = self._resize(x, self.sizes[k])
        m = self.model.eval()
        for _ in range(self.steps if steps is None else steps):
            xv = random_shift(xs)[rows:]
            with self.amp:
                logits = m(xv.contiguous(memory_format=torch.channels_last))
            ce = -(target[rows:] * logits.float().log_softmax(1)).sum(1)
            w = self._st_weights(target[rows:])
            loss = ce.mean() if w is None else (w * ce).sum() / w.sum().clamp_min(1e-6)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
