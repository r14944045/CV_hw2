"""BOSA's bias-resistant online statistics alignment (Bai et al., IJCAI 2026, Sec. 4.3).

The running BN statistics of every layer follow the statistics of the features of the class-balanced
memory bank (not of the biased incoming batch), with a rate that adapts to how far they are from them:

  Eq. 4   alpha_t = 1 - exp(-D(N(mu_{t-1}, var_{t-1}), N(mu_hat_t, var_hat_t)))
          D(N1, N2) = 1 / (2C) * sum_c [KL(N1_c || N2_c) + KL(N2_c || N1_c)]   (symmetric KL)
  Eq. 2-3 mu_t  = (1 - alpha_t) mu_{t-1}  + alpha_t mu_hat_t
          var_t = (1 - alpha_t) var_{t-1} + alpha_t var_hat_t
  Eq. 5-7 adaptive update constraint: one gradient step on L_reg = alpha_t W2^2(N(mu_t, var_t), N(mu_s, var_s))
          mu_t    <- mu_t    - 2 alpha_t (mu_t    - mu_s)
          sigma_t <- sigma_t - 2 alpha_t (sigma_t - sigma_s)       (sigma = standard deviation)

mu_hat, var_hat are the statistics of the layer input in the forward that performs the update (the memory
bank); mu_s, sigma_s are the source model's running statistics.

Caveat: for alpha_t > 0.5 the literal step 2 * alpha_t exceeds 1 and overshoots past the source. At
alpha_t = 1 it gives mu = 2 mu_s - mu_hat, a reflection. The reflected statistics can be far from the
next batch's, which keeps alpha_t at 1. `reg` scales
the step: reg = 0.5 gives 2 * reg * alpha_t = alpha_t <= 1, a convex pull towards the source that cannot
overshoot. Normalisation (Eq. 8) uses the aligned
statistics. During a gradient forward with q > 0, activations are cached for the most salient channels only
(sparsify.sparse_bn, Sec. 4.4).
"""
import torch

from .norm import RobustBN2d, _replace_bn
from .sparsify import saliency_split, sparse_bn


def symmetric_kl(mean1, var1, mean2, var2):
    """1/(2C) * sum_c [KL(N(m1,v1) || N(m2,v2)) + KL(N(m2,v2) || N(m1,v1))] for per-channel Gaussians.
    The log terms cancel: KL12 + KL21 = 0.5 * [(v1 + d^2) / v2 + (v2 + d^2) / v1] - 1, d = m1 - m2."""
    d2 = (mean1 - mean2) ** 2
    sym = 0.5 * ((var1 + d2) / var2 + (var2 + d2) / var1) - 1
    return sym.mean() / 2


class AlignedBN2d(RobustBN2d):
    """Batch norm with BOSA's statistics alignment.

    train mode: stats are aligned to the input batch's statistics (Eqs. 2-7), then used to normalise.
    eval mode:  normalise with the stored stats, no update (same as RobustBN2d).

    adaptive=False uses the fixed rate `alpha` instead of Eq. 4 (= RoTTA's Robust BN).
    reg scales the step of the adaptive update constraint (Eqs. 6-7): 1 = paper, 0 = no constraint.
    q = fraction of channels (lowest |gamma|) whose activations are not cached for backprop.
    """

    def __init__(self, bn, alpha=0.05, adaptive=True, reg=1.0, q=0.0):
        super().__init__(bn, alpha)
        self.register_buffer("src_mean", bn.running_mean.detach().clone())
        self.register_buffer("src_std", bn.running_var.detach().clone().sqrt())
        self.adaptive, self.reg, self.q = adaptive, reg, q
        self.alpha_t = None  # last alignment rate (detached scalar tensor), for diagnostics

    def align(self, b_mean, b_var):
        """Eqs. 2-7. Returns the aligned (mean, var); b_mean / b_var may carry gradients."""
        if self.adaptive:
            with torch.no_grad():
                D = symmetric_kl(self.mean, self.var + self.eps, b_mean, b_var + self.eps)
                a = 1 - torch.exp(-D)
        else:
            a = self.alpha
        self.alpha_t = a
        mean = (1 - a) * self.mean + a * b_mean
        var = (1 - a) * self.var + a * b_var
        if self.reg > 0:
            step = 2 * self.reg * a
            mean = mean - step * (mean - self.src_mean)
            std = var.clamp_min(1e-12).sqrt()
            std = std - step * (std - self.src_std)
            var = std ** 2
        return mean, var

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        grad_path = torch.is_grad_enabled() and self.weight.requires_grad
        sparse = grad_path and self.q > 0
        if grad_path and not sparse:
            b_var, b_mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
        else:
            with torch.no_grad():
                b_var, b_mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
        if sparse:
            # Kept channels' statistics carry gradients (as in standard BN); dropped ones are constants.
            # x_keep is the only activation of this layer cached for backprop.
            order, n_keep = saliency_split(self.weight, self.q)
            keep = order[:n_keep]
            x_keep = x.index_select(1, keep)
            k_var, k_mean = torch.var_mean(x_keep, dim=(0, 2, 3), unbiased=False)
            b_var, b_mean = b_var.index_copy(0, keep, k_var), b_mean.index_copy(0, keep, k_mean)
        mean, var = self.align(b_mean, b_var)
        self.mean, self.var = mean.detach(), var.detach()
        if sparse:
            return sparse_bn(x, x_keep, mean, var, self.weight, self.bias, self.eps, order, n_keep)
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)


def configure_aligned_norm(model, **kw):
    """Replace every BatchNorm2d of `model` (in place) by AlignedBN2d(bn, **kw); returns the model."""
    _replace_bn(model, lambda bn: AlignedBN2d(bn, **kw))
    return model
