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
