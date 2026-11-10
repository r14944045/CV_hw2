"""Normalisation strategies: which statistics BatchNorm uses at test time.

  "source": the source model's running statistics, never updated.
  "batch":  statistics of the current test batch (TENT / CoTTA). Breaks on non-i.i.d. batches.
  "robust": RoTTA's Robust BN, global statistics updated by an EMA of batch statistics.
"""
import torch
import torch.nn as nn


class RobustBN2d(nn.Module):
    """Robust Batch Normalization (RoTTA).

    train mode: stats <- (1 - alpha) * stats + alpha * batch_stats, then normalise with the new stats.
    eval mode:  normalise with the stored stats, no update.
    alpha = 0 gives frozen source statistics.
    """

    def __init__(self, bn: nn.BatchNorm2d, alpha):
        super().__init__()
        self.alpha, self.eps = alpha, bn.eps
        self.register_buffer("mean", bn.running_mean.detach().clone())
        self.register_buffer("var", bn.running_var.detach().clone())
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())

    def forward(self, x):
        if self.training and self.alpha > 0:
            b_var, b_mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
            mean = (1 - self.alpha) * self.mean + self.alpha * b_mean
            var = (1 - self.alpha) * self.var + self.alpha * b_var
            self.mean, self.var = mean.detach(), var.detach()
        else:
            mean, var = self.mean, self.var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)


def _replace_bn(module, fn):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, fn(child))
        else:
            _replace_bn(child, fn)


def _batch_stat_bn(bn):
    bn.track_running_stats = False
    bn.running_mean, bn.running_var = None, None  # forces batch statistics in train and eval mode
    return bn


def configure_norm(model, mode, alpha=0.05):
    """Convert every BatchNorm2d of `model` (in place) to the chosen strategy; returns the model."""
    if mode == "robust":
        _replace_bn(model, lambda bn: RobustBN2d(bn, alpha))
    elif mode == "source":
        _replace_bn(model, lambda bn: RobustBN2d(bn, 0.0))
    elif mode == "batch":
        _replace_bn(model, _batch_stat_bn)
    else:
        raise ValueError(f"unknown norm mode: {mode}")
    return model
