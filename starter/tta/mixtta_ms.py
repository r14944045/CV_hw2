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
     domain. Above shift_thr, the batch starts a new domain (tools/dev/threshold_val.py chooses it on val: 8.7): its statistics replace the tracked ones
     (tracking rate 1) in every layer and at every size. The val domains change abruptly; with a fixed
     rate of 0.4 the first batches of a new domain are normalised with a mix of the old domain's statistics.
  0b. (alpha_min set) Within a domain, the tracking rate is 1/k for the k-th batch since the last detected
     shift (a running mean of the domain's batch statistics, which are stationary within a domain but
     noisy from batch to batch because of the label skew), floored at alpha_min.
  0c. (split_shift) A domain boundary usually falls inside a batch (both val boundaries do, at rows 8 and 28),
     and the detector often fires on that mixed batch. On a detected shift, one changepoint is fitted to the per-sample conv1 statistics of the previous
     batch followed by this one (least squares on standardised channel means and log-variances). The first j
     rows (the old domain) are normalised with the previous domain's statistics, the new domain's statistics
     start from the remaining rows only, and the running mean of step 0b counts rows instead of batches.
     Self-training on that batch uses the new rows only.
  1. For each size s: update tracker s with x resized to s, and get the softmax of x, its flip and
     `n_shift` random flip + shift copies (mixbn.MultiMixBN2d normalisation with alpha, w).
  2. Average the probabilities over views and scales, and the penultimate features over views and scales.
  3. LAME on the averaged probabilities with a kNN graph on the averaged features. Predict the argmax.
  3a. Before LAME, the probabilities are softened with temperature lame_T (default 2). With lame_q = q, the
     temperature is instead set per batch (T in [1, 3]) so that the batch's mean max-probability is q: a
     confident batch is softened more and an uncertain one less, so the weight of LAME's neighbour term is
     similar on every domain.
  3b. (prior_lambda > 0) Class-prior correction, before LAME: every val domain is class-balanced, and the
     stream is assumed to be balanced overall, but under corruption the model over-predicts some classes
     (on val, the source model: acoustic guitar, laptop, tree frog) and under-predicts others (steam
     locomotive, sports car, timber wolf). Count the predicted labels so far, pi_c = (count_c + tau/C) / (n + tau), and
     multiply the probabilities by (C * pi_c)^(-prior_lambda). tau (in samples) delays the correction at
     the start of the stream, when the counts follow the label-skewed stream rather than the model's bias.
  4. Self-training: `steps` Adam steps on the BN scale/shift (gamma, beta), soft cross-entropy between
     the LAME output and the model's output on a random flip + shift copy of x at size
     sizes[train_scale] (the native 160 by default).
  4b. (stage_lr = 5 multipliers for stem, layer1 .. layer4) Per-stage learning rates for step 4. The early
     stages see low-level, corruption-specific statistics and can adapt faster; layer4's affine parameters
     act directly on class evidence, and a high rate there destabilises self-training (a global lr of 2e-3
     hurt, while 3x on stem-layer2 with 0.25x on layer4 helps). 0 freezes a stage.
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

# ---------------------------------------------------------------- ptta "caug": corruption augmentation of the
# self-training copy (no labels). Six ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019) at severity 3: the
# even positions of the published list (gaussian noise, impulse noise, glass blur, zoom blur, brightness, elastic
# transform; jpeg left out as it has no fast GPU version), re-implemented in torch. Parameters as in make_imagenet_c.py
# (github.com/hendrycks/robustness), scaled to the image size where the original uses pixel units.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _gauss_blur(x, sigma):
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.size(1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def _c_gaussian_noise(x):
    return x + 0.18 * torch.randn_like(x)


def _c_impulse_noise(x):
    m = torch.rand_like(x)
    return torch.where(m < 0.045, torch.zeros_like(x), torch.where(m < 0.09, torch.ones_like(x), x))


def _c_glass_blur(x):  # sigma 1, |displacement| <= 2, 3 iterations (random displacement instead of sequential swaps)
    x = _gauss_blur(x, 1.0)
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
    for _ in range(3):
        iy = (yy + torch.randint(-2, 3, (B, H, W), device=x.device)).clamp(0, H - 1)
        ix = (xx + torch.randint(-2, 3, (B, H, W), device=x.device)).clamp(0, W - 1)
        idx = (iy * W + ix).view(B, 1, H * W).expand(B, C, H * W)
        x = x.reshape(B, C, H * W).gather(2, idx).view(B, C, H, W)
    return _gauss_blur(x, 1.0)


def _c_zoom_blur(x):  # zoom factors 1.00, 1.02, ..., 1.20
    H = x.size(-1)
    out = torch.zeros_like(x)
    zs = [1 + 0.02 * i for i in range(11)]
    for z in zs:
        ch = int(math.ceil(H / z))
        t = (H - ch) // 2
        out += F.interpolate(x[..., t:t + ch, t:t + ch], size=(H, H), mode="bilinear", align_corners=False)
    return (x + out) / (len(zs) + 1)


def _c_brightness(x):  # HSV value + 0.3 (hue and saturation kept: RGB scales with V)
    v = x.max(1, keepdim=True).values
    nv = (v + 0.3).clamp(max=1)
    return torch.where(v > 0, x * nv / v.clamp_min(1e-6), nv.expand_as(x))


def _c_elastic(x):  # random affine (3 points moved by up to `aff` px) + smoothed displacement field
    B, C, H, W = x.shape
    s = H / 224
    alpha, sigma, aff = 244 * .05 * s, 244 * .01 * s, 244 * .02 * s
    c, q = H // 2, min(H, W) // 3
    p1 = torch.tensor([[c + q, c + q], [c + q, c - q], [c - q, c - q]], dtype=torch.float32)
    p2 = p1 + (torch.rand(B, 3, 2) * 2 - 1) * aff  # tiny 3x3 solves: on the CPU
    A = torch.cat([p2, torch.ones(B, 3, 1)], 2)  # inverse map: output point -> input point
    M = torch.linalg.solve(A, p1.expand(B, 3, 2)).to(x.device)  # (B, 3, 2)
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                            torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
    q_ = torch.stack([yy, xx, torch.ones_like(yy)], -1).view(1, H * W, 3)
    src = (q_ @ M).view(B, H, W, 2)  # (row, col)
    d = _gauss_blur(torch.rand(B, 2, H, W, device=x.device) * 2 - 1, sigma) * alpha
    src = src + d.permute(0, 2, 3, 1)
    grid = torch.stack([src[..., 1] / (W - 1) * 2 - 1, src[..., 0] / (H - 1) * 2 - 1], -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


_CAUG = [_c_gaussian_noise, _c_impulse_noise, _c_glass_blur, _c_zoom_blur, _c_brightness, _c_elastic]


def corrupt_augment(xv):
    """The normalised batch gets one corruption drawn at random from _CAUG (one per training step, so the six kinds
    alternate across steps and batches); returns a normalised batch."""
    mean, std = _MEAN.to(xv.device), _STD.to(xv.device)
    x01 = (xv.float() * std + mean).clamp(0, 1)
    fn = _CAUG[int(torch.randint(0, len(_CAUG), (1,)))]  # CPU draw: no GPU synchronisation
    return ((fn(x01).clamp(0, 1) - mean) / std).to(xv.dtype)


class MixTTAMS:
    def __init__(self, model, alpha=0.4, w=0.5, sizes=(160, 224), flip=True, n_shift=2, knn=5, lame_T=2.0,
                 lr=1e-3, steps=3, train_scale=0, amp=True, deterministic=True, prior_lambda=0.0, prior_tau=2000.0,
                 shift_thr=0.0, alpha_min=None, extra_train=(), extra_lr=1e-4, temporal_mu=0.0, temporal_eps=0.1,
                 split_shift=False, split_min=4, split_train="new", refine_batches=0,
                 st_weight=None, st_pow=1.0, st_ent=None, ema_pred=0.0, refine_steps=None,
                 opt_reset=False, stage_lr=None, lame_q=None, local_context=False, ptta=None):
        setup_cuda()
        if deterministic:  # fixed cuDNN algorithms: reruns give the same predictions (slower)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.sizes = list(sizes)
        self.model = configure_multi_mix_norm(model, alpha, w, len(self.sizes)).requires_grad_(False)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.layers = [m for m in self.model.modules() if isinstance(m, MultiMixBN2d)]
        self.flip, self.n_shift, self.knn, self.lame_T = flip, n_shift, knn, lame_T
        self.lame_q = lame_q  # adaptive temperature: per batch, T in [1, 3] such that the mean max-prob is lame_q
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
        self.opt_reset = opt_reset  # clear Adam's moment estimates at a detected domain shift
        self.temporal_mu, self.temporal_eps, self.h_prev = temporal_mu, temporal_eps, None
        self.local_context = local_context
        self.previous_graph_probs = None
        # ptta: published practical-TTA components on top of self-training (default off, see _ptta_weights):
        #   eata (Niu et al., ICML 2022), dot (DELTA, Zhao et al., ICLR 2023), roid_w / roid_we / roid_prior (ROID,
        #   Marsden et al., WACV 2024), sar_reset (SAR model recovery, Niu et al., ICLR 2023)
        self.ptta = dict(ptta or {})
        self._fused_prob, self._eata_ema, self._roid_ema, self._dot_q, self._sar_ema = None, None, None, None, None
        self.sar_resets = 0
        # iabn: NOTE's instance-aware BN on every layer (Gong et al., NeurIPS 2022), threshold k (NOTE default 3)
        for l in self.layers:
            l.iabn_k = float(self.ptta.get("iabn", 0.0))
            if self.ptta.get("tribe"):  # TRIBE's balanced BN statistics (Su et al., AAAI 2024), momentum = value
                l.enable_tribe(float(self.ptta["tribe"]), self.model.fc.out_features, len(self.sizes))
            if self.ptta.get("unmix"):  # UnMix-TNS (Tomar et al., ICLR 2024): K = value components, momentum 0.1
                l.enable_unmix(int(self.ptta["unmix"]), 0.1, len(self.sizes))
        # t3a: T3A prototype classifier on the fused features (Iwasawa & Matsuo, NeurIPS 2021); value = filter_K
        self._t3a = None
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
            if stage_lr is None:
                groups = [dict(params=params, lr=lr)]
            else:  # 4b: per-stage learning rates, lr x stage_lr[stage] for stem, layer1 .. layer4 (0 freezes a stage)
                names = {id(m): n for n, m in self.model.named_modules()}
                by_stage = [[] for _ in range(5)]
                for l in self.layers:
                    n = names[id(l)]
                    by_stage[0 if n == "bn1" else int(n[5])] += [l.weight, l.bias]
                groups = []
                for mult, ps in zip(stage_lr, by_stage):
                    if mult == 0:
                        for q in ps:
                            q.requires_grad_(False)
                    elif ps:
                        groups.append(dict(params=ps, lr=lr * mult))
            groups += [dict(params=extra, lr=extra_lr)] if extra else []
            self.opt = torch.optim.Adam(groups, lr=lr)
            self._params = params
            # source copies of the trained parameters for ROID weight ensembling and SAR model recovery
            self._src_params = ([p.detach().clone() for p in params]
                                if self.ptta.get("roid_we") or self.ptta.get("sar_reset") or self.ptta.get("restore")
                                else None)
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
                if self.opt_reset and self.opt is not None:
                    self.opt.state.clear()
                if self.ptta.get("restore") and self.opt is not None:
                    # restore the trained parameters to source at a detected shift (Niloy et al., WACV 2024)
                    with torch.no_grad():
                        for p, s in zip(self._params, self._src_params):
                            p.copy_(s)
                    self.opt.state.clear()
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
        if self.ptta.get("tribe"):  # pseudo-labels for the balanced statistics: eval forward with the current statistics
            lab = self._tribe_labels(x)
            for l in self.layers:
                l.label = lab
        if self.j_old:
            out = [self._scale_forward_split(self._resize(x, s), k, self.j_old) for k, s in enumerate(self.sizes)]
        else:
            out = [self._scale_forward(self._resize(x, s), k, alpha) for k, s in enumerate(self.sizes)]
        if self.ptta.get("tribe"):
            for l in self.layers:
                l.label = None
        refine = self.refine_batches and not self.j_old and self.k_dom <= self.refine_batches and self.opt is not None
        z = self._predict(out, remember=not refine)
        if refine:  # 5: first batches of a domain: train on this batch, then predict it again with the new gamma/beta
            self._use_ema(False)
            self.self_train(x, z, steps=self.refine_steps)
            self._update_ema()
            self._use_ema(True)
            z = self._predict([self._eval_views(k, x.size(0)) for k in range(len(self.sizes))])
        self._use_ema(False)
        pred = (self._roid_prior(z) if self.ptta.get("roid_prior") else z).argmax(1)
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

    @staticmethod
    def _temper_to(prob, q):
        """Step 3a with lame_q: soften with the temperature T in [1, 3] (bisection) that brings the batch's mean
        max-probability to q; T = 1 if it is already below q, T = 3 if q is not reached at T = 3."""
        logp = torch.log(prob + 1e-10)
        lo, hi = 1.0, 3.0
        for _ in range(20):
            T = 0.5 * (lo + hi)
            if torch.softmax(logp / T, 1).max(1).values.mean() > q:
                lo = T
            else:
                hi = T
        return torch.softmax(logp / (0.5 * (lo + hi)), 1)

    def _predict(self, out, *, remember=True):
        """Fuse scales and compute soft targets for prediction and self-training."""
        prob = torch.stack([p for p, _ in out]).mean(0)
        feats = torch.stack([f for _, f in out]).mean(0)
        self._fused_prob = prob  # the model's own prediction (views and sizes averaged), used by ptta
        if self.ptta.get("t3a"):  # T3A replaces the linear classifier's output with its prototype classifier
            prob = self._t3a_prob(feats, prob, remember=remember)
        C = prob.size(1)
        if self.counts is None:
            self.counts = torch.zeros(C, device=prob.device)
        if self.prior_lambda:
            pi = (self.counts + self.prior_tau / C) / (self.n_seen + self.prior_tau)
            prob = prob * (C * pi).pow(-self.prior_lambda)
            prob = prob / prob.sum(1, keepdim=True)
        image_prob = prob
        if self.lame_q is not None:
            prob = self._temper_to(prob, self.lame_q)
        elif self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        if self.temporal_mu and self.h_prev is not None:  # 3c: classes of the previous batch are likely again
            prob = prob * (self.h_prev + self.temporal_eps).pow(self.temporal_mu)
            prob = prob / prob.sum(1, keepdim=True)
        z = lame(prob, feats, self.knn)
        self.h_prev = z.mean(0)
        if self.local_context:
            target = self._local_targets(image_prob, z)
            if remember:
                self.previous_graph_probs = z.detach()
            return target
        return z

    def _local_targets(self, image_prob, graph_prob):
        """Estimate each image's class prior from other rows in this batch and the preceding batch."""
        history = self.previous_graph_probs
        evidence = graph_prob if history is None else torch.cat([history, graph_prob])
        positions = torch.arange(len(evidence), device=evidence.device)
        distance = (positions[-len(graph_prob):, None] - positions[None, :]).abs()
        weights = torch.exp(-distance / 12.0).masked_fill_(distance == 0, 0)
        counts = weights @ evidence + 0.2  # pseudocount per class, including when no neighbours exist
        # The prior's row-wise normalizer cancels in softmax. Sharpen the combined target for self-training.
        logits = (2 * image_prob.clamp_min(1e-12).log() + graph_prob.clamp_min(1e-12).log()) / 3
        return (2 * (logits + 0.5 * counts.log())).softmax(1)

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
        pw, denom = self._ptta_weights(rows) if self.ptta else (None, None)
        for _ in range(self.steps if steps is None else steps):
            xv = random_shift(xs)[rows:]
            if self.ptta.get("caug"):  # corruption augmentation of the training copy (no labels)
                xv = corrupt_augment(xv)
            with self.amp:
                logits = m(xv.contiguous(memory_format=torch.channels_last))
            ce = -(target[rows:] * logits.float().log_softmax(1)).sum(1)
            w = self._st_weights(target[rows:])
            loss = ce.mean() if w is None else (w * ce).sum() / w.sum().clamp_min(1e-6)
            if pw is not None:
                loss = ((pw if w is None else pw * w) * ce).sum() / denom
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            if self.ptta.get("roid_we"):  # ROID weight ensembling after every optimizer step
                with torch.no_grad():
                    for p, s in zip(self._params, self._src_params):
                        p.mul_(self.ptta["roid_we"]).add_(s, alpha=1 - self.ptta["roid_we"])
        if self.ptta:
            self._ptta_after(logits.detach().float(), pw, rows)

    # ------------------------------------------------------------------ published practical-TTA components (ptta)
    def _ptta_weights(self, rows):
        """Per-image weights for the self-training loss from the model's fused prediction p, and the denominator.
        eata: keep H(p) < 0.4 ln C and |cos(ema, p)| < 0.05, weight exp(E0 - H) (EATA, eata.py; e_margin, d_margin)
        roid_w: certainty x diversity weights exp(div * cert / (1/3)), samples with below-mean diversity dropped (ROID)
        dot: class weights 1 / q_hat of the predicted class, renormalised to mean 1 (DELTA's DOT)"""
        p = self._fused_prob[rows:].float()
        n, C = p.shape
        H = -(p * torch.log(p + 1e-10)).sum(1)
        w = torch.ones(n, device=p.device)
        denom = float(n)
        if self.ptta.get("eata"):
            e0 = 0.4 * math.log(C)
            keep = H < e0
            if self._eata_ema is not None:
                keep = keep & (F.cosine_similarity(self._eata_ema[None], p, dim=1).abs() < 0.05)
            if keep.any():
                kp = p[keep].mean(0)
                self._eata_ema = kp if self._eata_ema is None else 0.9 * self._eata_ema + 0.1 * kp
            w = w * keep.float() * torch.exp(e0 - H)
            denom = max(1.0, float(keep.sum()))
        if self.ptta.get("roid_w"):
            ema = self._roid_ema if self._roid_ema is not None else torch.full((C,), 1.0 / C, device=p.device)
            div = 1 - F.cosine_similarity(ema[None], p, dim=1)
            div = (div - div.min()) / (div.max() - div.min() + 1e-8)
            cert = -H
            cert = (cert - cert.min()) / (cert.max() - cert.min() + 1e-8)
            rw = torch.exp(div * cert / (1 / 3))
            rw[div < div.mean()] = 0
            w = w * rw
            self._roid_ema = 0.9 * ema + 0.1 * p.mean(0)
        if self.ptta.get("dot"):
            if self._dot_q is None:
                self._dot_q = torch.full((C,), 1.0 / C, device=p.device)
            cw = 1.0 / self._dot_q
            cw = cw / cw.sum()
            sw = cw[p.argmax(1)]
            w = w * (sw / sw.sum() * n)
        return w, denom

    def _ptta_after(self, logits, pw, rows):
        """After the batch's update: DELTA's q_hat update and SAR's model recovery."""
        if self.ptta.get("dot"):
            self._dot_q = self.ptta["dot"] * self._dot_q + (1 - self.ptta["dot"]) * self._fused_prob[rows:].float().mean(0)
        if self.ptta.get("sar_reset"):  # SAR: EMA (0.9) of the trained samples' entropy; reset to source below 0.2
            p = logits.softmax(1)
            H = -(p * torch.log(p + 1e-10)).sum(1)
            if pw is not None and self.ptta.get("eata") and (pw > 0).any():
                H = H[pw > 0]
            e = float(H.mean())
            self._sar_ema = e if self._sar_ema is None else 0.9 * self._sar_ema + 0.1 * e
            if self._sar_ema < 0.2:
                with torch.no_grad():
                    for p_, s in zip(self._params, self._src_params):
                        p_.copy_(s)
                self.opt.state.clear()
                self._sar_ema = None
                self.sar_resets += 1

    @torch.no_grad()
    def _tribe_labels(self, x):
        """Argmax of the softmax averaged over the input sizes, from eval-mode forwards (statistics not updated)."""
        prob = 0
        for k, s in enumerate(self.sizes):
            self._set_tracker(k)
            with self.amp:
                logits = self.model.eval()(self._resize(x, s).contiguous(memory_format=torch.channels_last))
            prob = prob + logits.float().softmax(1)
        return prob.argmax(1)

    @torch.no_grad()
    def _t3a_prob(self, feats, prob, remember=True):
        """T3A (matsuolab/T3A, adapt_algorithms.py): supports start from the classifier weights; each batch adds its
        features with one-hot predicted labels and entropies; per class only the filter_K lowest-entropy supports are
        kept; logits = z @ normalize(normalize(supports).T @ labels). Supports restart at a detected domain shift."""
        K = int(self.ptta["t3a"])
        fc = self.model.fc
        if self._t3a is None or (self.k_dom == 1 and self.shifts):
            w = fc.weight.detach().float()
            p0 = (w @ w.T + fc.bias.detach().float()).softmax(1)
            self._t3a = [w, F.one_hot(p0.argmax(1), w.size(0)).float(), -(p0 * torch.log(p0 + 1e-10)).sum(1)]
        S, L, E = self._t3a
        yh = prob.argmax(1)
        ent = -(prob * torch.log(prob + 1e-10)).sum(1)
        S, L, E = torch.cat([S, feats.float()]), torch.cat([L, F.one_hot(yh, L.size(1)).float()]), torch.cat([E, ent])
        keep = []
        lab = L.argmax(1)
        for c in range(L.size(1)):
            idx = torch.nonzero(lab == c).flatten()
            keep.append(idx[E[idx].argsort()[:K]])
        keep = torch.cat(keep)
        S, L, E = S[keep], L[keep], E[keep]
        if remember:
            self._t3a = [S, L, E]
        weights = F.normalize(S, dim=1).T @ L
        logits = feats.float() @ F.normalize(weights, dim=0)
        return logits.softmax(1)

    @staticmethod
    def _roid_prior(z):
        """ROID's prior correction (roid.py), applied to this method's output probabilities: multiply by the batch's
        smoothed mean prediction."""
        prior = z.mean(0)
        smooth = max(1 / z.size(0), 1 / z.size(1)) / prior.max()
        return z * ((prior + smooth) / (1 + smooth * z.size(1)))
