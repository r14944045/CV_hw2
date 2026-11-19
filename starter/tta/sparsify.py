"""BOSA's saliency-guided activation sparsification and gradient reconstruction (Sec. 4.4).

To backpropagate through BN, autograd caches each layer's input x. BOSA caches it only for the most
salient channels. Saliency = |gamma_c| (the BN scale, since normalised channels share one scale).
In each layer, the bottom q fraction of channels by |gamma_c| is not cached.

  kept channels:    exact BN gradient (Eq. 9 when the statistics are batch statistics)
  dropped channels: forward is unchanged (full-resolution output). Backward keeps only the direct term
                    (Eq. 10):  grad_x_c = gamma_c / sqrt(var_c + eps) * grad_y_c.
                    grad_beta_c = sum(grad_y_c), grad_gamma_c = 0 (it needs the uncached normalised input).

Implementation: on dropped channels the output is x * s + t with s, t detached. The product with a
constant saves only s (C values) for backward, not x, and its gradient is exactly s * grad_y.
Channels are reassembled with cat + index_select, which save no activations (index_copy would save its
source). Only x[:, keep] is cached.
"""
import torch


def saliency_split(weight, q):
    """Channel order by decreasing |gamma| and the number of channels to cache (the top 1 - q)."""
    C = weight.numel()
    return weight.detach().abs().argsort(descending=True), C - int(q * C)


def sparse_bn(x, x_keep, mean, var, weight, bias, eps, order, n_keep):
    """y = gamma * (x - mean) / sqrt(var + eps) + beta for every channel (Eq. 8).
    x_keep = x[:, order[:n_keep]] is the only activation cached; gradients on the other channels
    follow Eq. 10."""
    view = (1, -1, 1, 1)
    keep, drop = order[:n_keep], order[n_keep:]
    rstd = torch.rsqrt(var + eps)
    k_scale = weight[keep] * rstd[keep]
    y_keep = x_keep * k_scale.view(view) + (bias[keep] - mean[keep] * k_scale).view(view)
    with torch.no_grad():
        d_scale = weight[drop] * rstd[drop]
        d_shift = -mean[drop] * d_scale
    y_drop = x.index_select(1, drop) * d_scale.view(view) + d_shift.view(view) + bias[drop].view(view)
    return torch.cat([y_keep, y_drop], 1).index_select(1, order.argsort())
