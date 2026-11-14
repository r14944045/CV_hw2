"""Source-mixed batch normalisation: track the test statistics, normalise with a mix of them and the source's.

  track (train mode):  mu_tr <- (1 - alpha) mu_tr + alpha mu_batch      (same for var)
  normalise:           mu = (1 - w) mu_tr + w mu_s,  var = (1 - w) var_tr + w var_s

Compared with BOSA's aligned BN (align.py), the tracking rate (alpha) and the pull towards the source (w)
are separate knobs. In BOSA one alpha_t sets both, and at alpha_t = 1 the halved step resets the stats to
the source. Compared with Robust BN (alpha = 0.05, w = 0), it tracks faster and keeps part of the source
statistics, which the label-skewed test batches cannot estimate well.
"""
import torch
import torch.nn as nn

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

    def forward(self, x):
        mean_t, var_t = getattr(self, f"mean{self.k}"), getattr(self, f"var{self.k}")
        if self.training:
            xs = x if self.stat_rows is None else x[:self.stat_rows]
            b_var, b_mean = torch.var_mean(xs.detach().float(), dim=(0, 2, 3), unbiased=False)
            mean_t = (1 - self.alpha) * mean_t + self.alpha * b_mean
            var_t = (1 - self.alpha) * var_t + self.alpha * b_var
            setattr(self, f"mean{self.k}", mean_t)
            setattr(self, f"var{self.k}", var_t)
        mean = (1 - self.w) * mean_t + self.w * self.src_mean
        var = (1 - self.w) * var_t + self.w * self.src_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.to(x.dtype).view(1, -1, 1, 1) + shift.to(x.dtype).view(1, -1, 1, 1)


def configure_multi_mix_norm(model, alpha=0.4, w=0.5, n_trackers=1):
    """Replace every BatchNorm2d of `model` (in place) by MultiMixBN2d; returns the model."""
    _replace_bn(model, lambda bn: MultiMixBN2d(bn, alpha, w, n_trackers))
    return model
