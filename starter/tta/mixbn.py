"""Source-mixed batch normalisation: track the test statistics, normalise with a mix of them and the source's.

  track (train mode):  mu_tr <- (1 - alpha) mu_tr + alpha mu_batch      (same for var)
  normalise:           mu = (1 - w) mu_tr + w mu_s,  var = (1 - w) var_tr + w var_s

Compared with BOSA's aligned BN (align.py), the tracking rate (alpha) and the pull towards the source (w)
are separate knobs. In BOSA one alpha_t sets both, and at alpha_t = 1 the halved step resets the stats to
the source. Compared with Robust BN (alpha = 0.05, w = 0), it tracks faster and keeps part of the source
statistics, which the label-skewed test batches cannot estimate well.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import _replace_bn


class MixBN2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d, alpha=0.2, w=0.5):
        super().__init__()
        self.alpha, self.w, self.eps = alpha, w, bn.eps
        self.register_buffer("src_mean", bn.running_mean.detach().clone())
        self.register_buffer("src_var", bn.running_var.detach().clone())
        self.register_buffer("mean", bn.running_mean.detach().clone())
        self.register_buffer("var", bn.running_var.detach().clone())
        self.weight = nn.Parameter(bn.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bn.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x.detach(), dim=(0, 2, 3), unbiased=False)
            self.mean = (1 - self.alpha) * self.mean + self.alpha * b_mean
            self.var = (1 - self.alpha) * self.var + self.alpha * b_var
        mean = (1 - self.w) * self.mean + self.w * self.src_mean
        var = (1 - self.w) * self.var + self.w * self.src_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)


def configure_mix_norm(model, alpha=0.2, w=0.5):
    """Replace every BatchNorm2d of `model` (in place) by MixBN2d; returns the model."""
    _replace_bn(model, lambda bn: MixBN2d(bn, alpha, w))
    return model


class MultiMixBN2d(nn.Module):
    """MixBN2d with several statistics trackers (one per input scale of MixTTA-MS) and a fused-view mode.

    `k` selects the tracker. With `stat_rows = B` set, a train-mode forward of a batch whose first B rows
    are x computes the batch statistics from those rows only, updates tracker k, and normalises every
    row with the updated statistics: one forward of [x, view_1, ..., view_V] then equals a train-mode
    forward of x followed by eval-mode forwards of the views. Statistics are kept in float32, and the
    output keeps the input dtype (bf16 under autocast).
    """

    def __init__(self, bn: nn.BatchNorm2d, alpha=0.4, w=0.5, n_trackers=1):
        super().__init__()
        self.alpha, self.w, self.eps = alpha, w, bn.eps
        self.register_buffer("src_mean", bn.running_mean.detach().clone().float())
        self.register_buffer("src_var", bn.running_var.detach().clone().float())
        for k in range(n_trackers):
            self.register_buffer(f"mean{k}", bn.running_mean.detach().clone().float())
            self.register_buffer(f"var{k}", bn.running_var.detach().clone().float())
        self.weight = nn.Parameter(bn.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bn.bias.detach().clone(), requires_grad=False)
        self.k, self.stat_rows = 0, None
        self.iabn_k = 0.0  # > 0: NOTE's instance-aware BN on top of the mixed statistics (default off)
        # TRIBE's balanced statistics (default off): momentum `tribe`, per-class statistics for each tracker, and
        # `label` (pseudo-labels of the stat rows) set by the caller before a statistics-updating forward
        self.tribe, self.label = None, None
        self.unmix = None  # UnMix-TNS component statistics (default off): momentum

    def enable_unmix(self, K, momentum, n_trackers):
        """UnMix-TNS (Tomar et al., ICLR 2024; devavratTomar/unmixtns, unmixtns.py): K components per tracker,
        initialised as source mean + noise and (1 - coeff) * source variance, coeff = 0.5."""
        self.unmix = momentum
        coeff = 0.5 if K > 1 else 0.0
        for k in range(n_trackers):
            noise = torch.randn(K, self.src_mean.numel(), device=self.src_mean.device)
            cm = self.src_mean[None] + math.sqrt(coeff * K / max(K - 1, 1)) * self.src_var.sqrt()[None] * noise
            self.register_buffer(f"um_mean{k}", cm)
            self.register_buffer(f"um_var{k}", (1 - coeff) * self.src_var[None].repeat(K, 1))

    def _unmix(self, x):
        """Per-instance statistics: components mixed with the instance statistics by the soft assignment
        softmax(cos(instance mean, component mean) / 0.07), averaged over components (law of total variance).
        Components move towards the instance statistics of the stat rows in statistics-updating forwards."""
        cm, cv = getattr(self, f"um_mean{self.k}"), getattr(self, f"um_var{self.k}")
        i_var, i_mean = torch.var_mean(x.detach().float(), dim=(2, 3), unbiased=False)  # (N, C)
        p = torch.softmax(F.normalize(i_mean, dim=1) @ F.normalize(cm, dim=1).T / 0.07, dim=1)  # (N, K)
        pm = p.unsqueeze(-1)
        mean_mat = (1 - pm) * cm.unsqueeze(0) + pm * i_mean.unsqueeze(1)
        var_mat = (1 - pm) * cv.unsqueeze(0) + pm * i_var.unsqueeze(1)
        mean = mean_mat.mean(1)
        var = var_mat.mean(1) + (mean_mat ** 2).mean(1) - mean ** 2
        if self.training:
            r = x.size(0) if self.stat_rows is None else self.stat_rows
            cm = cm + self.unmix * (pm[:r] * (i_mean[:r].unsqueeze(1) - cm.unsqueeze(0))).mean(0)
            cv = cv + self.unmix * (pm[:r] * (i_var[:r].unsqueeze(1) - cv.unsqueeze(0))).mean(0)
            setattr(self, f"um_mean{self.k}", cm)
            setattr(self, f"um_var{self.k}", cv)
        scale = self.weight * torch.rsqrt(var.clamp_min(0) + self.eps)
        shift = self.bias - mean * scale
        return x * scale.to(x.dtype)[:, :, None, None] + shift.to(x.dtype)[:, :, None, None]

    def enable_tribe(self, momentum, n_classes, n_trackers):
        self.tribe = momentum
        for k in range(n_trackers):
            self.register_buffer(f"cls_mean{k}", self.src_mean[None].repeat(n_classes, 1).clone())
            self.register_buffer(f"cls_var{k}", self.src_var[None].repeat(n_classes, 1).clone())

    def _tribe_update(self, xs):
        """TRIBE (Su et al., AAAI 2024; Gorilla-Lab-SCUT/TRIBE, balanced_bn_pyv.py update_statistics_2d_v5, gamma 0):
        per-class statistics move towards the class's batch statistics at rate 1/count when count > 1/momentum, else
        momentum; the normalisation statistics are the class average (variance: mean variance + variance of means)."""
        cm, cv = getattr(self, f"cls_mean{self.k}"), getattr(self, f"cls_var{self.k}")
        x = xs.detach().float()
        mu_s = x.mean((2, 3))
        m2_s = (x * x).mean((2, 3))
        oh = F.one_hot(self.label, cm.size(0)).float()  # (B, L)
        cnt = oh.sum(0)
        b_mean = (oh.T @ mu_s) / cnt.clamp_min(1)[:, None]
        b_var = ((oh.T @ m2_s) / cnt.clamp_min(1)[:, None] - b_mean ** 2).clamp_min(0)
        m = torch.where(cnt > 1.0 / self.tribe, 1.0 / cnt.clamp_min(1), torch.full_like(cnt, self.tribe))
        m = (m * (cnt > 0).float())[:, None]
        d = b_mean - cm
        cm = cm + m * d
        cv = (1 - m) * cv + m * b_var + m * (1 - m) * d ** 2
        setattr(self, f"cls_mean{self.k}", cm)
        setattr(self, f"cls_var{self.k}", cv)
        return cm.mean(0), cv.mean(0) + cm.var(0, unbiased=False)

    def forward(self, x):
        if self.unmix is not None and x.size(2) * x.size(3) > 1:
            return self._unmix(x)
        mean_t, var_t = getattr(self, f"mean{self.k}"), getattr(self, f"var{self.k}")
        if self.training:
            xs = x if self.stat_rows is None else x[:self.stat_rows]
            if self.tribe is not None and self.label is not None and len(self.label) == xs.size(0):
                mean_t, var_t = self._tribe_update(xs)
            else:
                b_var, b_mean = torch.var_mean(xs.detach().float(), dim=(0, 2, 3), unbiased=False)
                mean_t = (1 - self.alpha) * mean_t + self.alpha * b_mean
                var_t = (1 - self.alpha) * var_t + self.alpha * b_var
            setattr(self, f"mean{self.k}", mean_t)
            setattr(self, f"var{self.k}", var_t)
        mean = (1 - self.w) * mean_t + self.w * self.src_mean
        var = (1 - self.w) * var_t + self.w * self.src_var
        if self.iabn_k > 0 and x.size(2) * x.size(3) > 1:
            return self._iabn(x, mean, var)
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.to(x.dtype).view(1, -1, 1, 1) + shift.to(x.dtype).view(1, -1, 1, 1)

    def _iabn(self, x, mean, var):
        """NOTE (Gong et al., NeurIPS 2022), utils/iabn.py: per-sample statistics = shared statistics plus the
        soft-shrunk deviation of the instance statistics, threshold k times their standard error."""
        hw = x.size(2) * x.size(3)
        i_var, i_mean = torch.var_mean(x.detach().float(), dim=(2, 3), unbiased=False)  # (N, C)
        s_mu = torch.sqrt((var + self.eps) / hw)
        s_var = (var + self.eps) * math.sqrt(2 / (hw - 1))
        mu = mean + _softshrink(i_mean - mean, self.iabn_k * s_mu)
        sg = F.relu(var + _softshrink(i_var - var, self.iabn_k * s_var))
        scale = self.weight * torch.rsqrt(sg + self.eps)  # (N, C)
        shift = self.bias - mu * scale
        return x * scale.to(x.dtype)[:, :, None, None] + shift.to(x.dtype)[:, :, None, None]


def _softshrink(x, lbd):  # per-channel threshold (NOTE's _softshrink): relu(x - lbd) - relu(-(x + lbd))
    return F.relu(x - lbd) - F.relu(-(x + lbd))


def configure_multi_mix_norm(model, alpha=0.4, w=0.5, n_trackers=1):
    """Replace every BatchNorm2d of `model` (in place) by MultiMixBN2d; returns the model."""
    _replace_bn(model, lambda bn: MultiMixBN2d(bn, alpha, w, n_trackers))
    return model
