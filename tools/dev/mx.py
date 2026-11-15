"""Experimental MixTTA-MS variants (scratch). Branches = (size, transform) with one BN tracker each."""
import os
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../starter"))
from tta.common import setup_cuda  # noqa
from tta import mixtta_ms as mm  # noqa
from tta.mixbn import MultiMixBN2d, configure_multi_mix_norm  # noqa
from tta.mixtta import random_shift  # noqa


def gblur(x, s):
    k = int(2 * math.ceil(2.5 * s) + 1); t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = (g / g.sum()).to(x.dtype)
    C = x.size(1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def gray(x):
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    return g.expand(-1, 3, -1, -1).contiguous()


class MX:
    def __init__(self, model, alpha=0.4, w=0.5, branches=((160, "id"), (224, "id")), flip=True, n_shift=2, knn=5,
                 lame_T=2.0, lr=1e-3, steps=3, train_branch=0, fuse="mean", fuse_T=1.0,
                 shift_detect=False, shift_thr=3.0, shift_alpha=1.0, shift_reset_params=False,
                 dom_prior=0.0, dom_tau=200.0, glob_prior=0.0, glob_tau=2000.0,
                 train_first=False, train_params="bn", deterministic=True, ent_w=0.0, div_w=0.0,
                 alpha_cum=False, alpha_min=0.1):
        setup_cuda()
        if deterministic:
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        self.branches = list(branches)
        self.model = configure_multi_mix_norm(model, alpha, w, len(self.branches)).requires_grad_(False)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.layers = [m for m in self.model.modules() if isinstance(m, MultiMixBN2d)]
        self.alpha = alpha
        self.flip, self.n_shift, self.knn, self.lame_T = flip, n_shift, knn, lame_T
        self.steps, self.train_branch, self.fuse, self.fuse_T = steps, train_branch, fuse, fuse_T
        self.shift_detect, self.shift_thr, self.shift_alpha, self.shift_reset_params = shift_detect, shift_thr, shift_alpha, shift_reset_params
        self.dom_prior, self.dom_tau, self.glob_prior, self.glob_tau = dom_prior, dom_tau, glob_prior, glob_tau
        self.train_first, self.ent_w, self.div_w = train_first, ent_w, div_w
        self.alpha_cum, self.alpha_min, self.k_dom = alpha_cum, alpha_min, 0
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16)
        self._feats = None
        self.model.fc.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0].detach()))
        self.gcounts = None; self.gn = 0; self.dcounts = None; self.dn = 0
        self.dist_hist = []; self.shifts = []; self.t = 0
        self.opt = None
        if lr > 0 and steps > 0:
            if train_params == "bn":
                params = [p for l in self.layers for p in (l.weight, l.bias)]
            elif train_params == "bn+fc":
                params = [p for l in self.layers for p in (l.weight, l.bias)] + list(self.model.fc.parameters())
            for p in params:
                p.requires_grad_(True)
            self.params = params
            self.init = [p.detach().clone() for p in params]
            self.opt = torch.optim.Adam(params, lr=lr)
        # first BN layer input stats for shift detection (160 branch)
        self.bn1 = self.model.bn1

    def _set(self, k, rows=None, alpha=None):
        for l in self.layers:
            l.k, l.stat_rows = k, rows
            l.alpha = self.alpha if alpha is None else alpha

    def _prep(self, x, b):
        s, tf = self.branches[b]
        if s != x.size(-1):
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        if tf == "id":
            return x
        if tf.startswith("blur"):
            return gblur(x, float(tf[4:]) if len(tf) > 4 else 1.0)
        if tf == "gray":
            return gray(x)
        raise ValueError(tf)

    @torch.no_grad()
    def _branch(self, xs, k, update=True, alpha=None):
        B = xs.size(0)
        views = [xs] + ([xs.flip(3)] if self.flip else []) + [random_shift(xs) for _ in range(self.n_shift)]
        self._set(k, rows=B if update else None, alpha=alpha)
        with self.amp:
            logits = (self.model.train() if update else self.model.eval())(torch.cat(views).contiguous(memory_format=torch.channels_last))
        self.model.eval(); self._set(k)
        V = len(views)
        return logits.float().softmax(1).view(V, B, -1).mean(0), self._feats.float().view(V, B, -1).mean(0)

    @torch.no_grad()
    def _shift_score(self, x):
        """Gaussian KL between this batch's conv1-output stats and tracker 0's (bn1 input)."""
        h = self.model.conv1(x.contiguous(memory_format=torch.channels_last)).float()
        v, m = torch.var_mean(h, dim=(0, 2, 3), unbiased=False)
        mt, vt = self.bn1.mean0, self.bn1.var0
        kl = 0.5 * ((v + (m - mt) ** 2) / (vt + 1e-5) - 1 - torch.log((v + 1e-5) / (vt + 1e-5)))
        return kl.mean().item()

    def _ensemble(self, x, update=True, alpha=None):
        out = [self._branch(self._prep(x, b), b, update, alpha) for b in range(len(self.branches))]
        probs = torch.stack([p for p, _ in out]); feats = torch.stack([f for _, f in out]).mean(0)
        if self.fuse == "mean":
            prob = probs.mean(0)
        elif self.fuse == "conf":  # weight each branch by its batch-mean max prob
            c = probs.max(2).values.mean(1)  # (nb,)
            wts = torch.softmax(c / (0.05 * self.fuse_T), 0)
            prob = (probs * wts.view(-1, 1, 1)).sum(0)
        elif self.fuse == "ent":  # per-sample entropy weights
            H = -(probs * probs.clamp_min(1e-10).log()).sum(2)  # nb,B
            wts = torch.softmax(-H / self.fuse_T, 0)
            prob = (probs * wts.unsqueeze(2)).sum(0)
        elif self.fuse == "geo":
            prob = torch.softmax(probs.clamp_min(1e-10).log().mean(0), 1)
        return prob, feats

    def _correct(self, prob):
        C = prob.size(1)
        if self.gcounts is None:
            self.gcounts = torch.zeros(C, device=prob.device); self.dcounts = torch.zeros(C, device=prob.device)
        if self.glob_prior:
            pi = (self.gcounts + self.glob_tau / C) / (self.gn + self.glob_tau)
            prob = prob * (C * pi).pow(-self.glob_prior); prob = prob / prob.sum(1, keepdim=True)
        if self.dom_prior:
            pi = (self.dcounts + self.dom_tau / C) / (self.dn + self.dom_tau)
            prob = prob * (C * pi).pow(-self.dom_prior); prob = prob / prob.sum(1, keepdim=True)
        if self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        return prob

    def __call__(self, x):
        alpha = None
        if self.shift_detect:
            d = self._shift_score(x)
            hist = self.dist_hist[-8:]
            ref = sorted(hist)[len(hist) // 2] if len(hist) >= 3 else None
            is_shift = ref is not None and d > self.shift_thr * max(ref, 1e-4)
            if is_shift:
                self.shifts.append(self.t); alpha = self.shift_alpha; self.k_dom = 0
                self.dist_hist = []; self.dcounts.zero_(); self.dn = 0
                if self.shift_reset_params and self.opt is not None:
                    with torch.no_grad():
                        for p, p0 in zip(self.params, self.init):
                            p.copy_(p0)
                    self.opt.state.clear()
            else:
                self.dist_hist.append(d)
        if self.alpha_cum:
            alpha = max(self.alpha_min, 1.0 / (self.k_dom + 1))
        self.k_dom += 1
        prob, feats = self._ensemble(x, True, alpha)
        z = mm.lame(self._correct(prob), feats, self.knn)
        if self.train_first and self.opt is not None:
            self.self_train(x, z)
            prob, feats = self._ensemble(x, False)
            z = mm.lame(self._correct(prob), feats, self.knn)
        pred = z.argmax(1)
        if self.gcounts is None:
            self.gcounts = torch.zeros(z.size(1), device=z.device); self.dcounts = torch.zeros(z.size(1), device=z.device)
        bc = torch.bincount(pred, minlength=z.size(1)).float()
        self.gcounts += bc; self.gn += pred.numel(); self.dcounts += bc; self.dn += pred.numel()
        if self.opt is not None and not self.train_first:
            self.self_train(x, z)
        self.t += 1
        return pred

    def self_train(self, x, target):
        k = self.train_branch; self._set(k)
        xs = self._prep(x, k); m = self.model.eval()
        for _ in range(self.steps):
            with self.amp:
                logits = m(random_shift(xs).contiguous(memory_format=torch.channels_last))
            lp = logits.float().log_softmax(1)
            loss = -(target * lp).sum(1).mean()
            if self.div_w:
                pm = lp.exp().mean(0); loss = loss + self.div_w * (pm * pm.clamp_min(1e-10).log()).sum()
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
