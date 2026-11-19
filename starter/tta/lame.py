"""LAME: Laplacian Adjusted Maximum-likelihood Estimation (Boudiaf et al., CVPR 2022).
Follows the official implementation (github.com/fiveai/LAME): kNN affinity on L2-normalised features,
symmetrised, bound_lambda = 1.

Corrects the predictions of one batch without touching the model. It minimises
  sum_i KL(z_i || p_i) - sum_ij W_ij z_i^T z_j
so that samples with similar features get similar labels. A batch that contains only a few classes (the
HW2 streams have 5-17 of 20 per batch) is then pushed towards those classes. Fixed-point updates:
  z <- softmax(log p + W z).
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def lame(prob, feats, knn=5, max_iter=100, tol=1e-8):
    """prob: (B, C) probabilities, feats: (B, D) features of the same batch. Returns corrected (B, C)."""
    B = feats.size(0)
    if B < 2:
        return prob
    f = F.normalize(feats.float(), dim=1)
    k = min(knn, B - 1)
    idx = (f @ f.t()).topk(k + 1, dim=1).indices[:, 1:]  # nearest neighbours, excluding self
    W = torch.zeros(B, B, device=f.device).scatter_(1, idx, 1.0)
    W = (W + W.t()) / 2
    logp = torch.log(prob.float() + 1e-10)
    z = prob.float()
    for _ in range(max_iter):
        z_new = torch.softmax(logp + W @ z, 1)
        if (z_new - z).abs().max() < tol:
            return z_new
        z = z_new
    return z
