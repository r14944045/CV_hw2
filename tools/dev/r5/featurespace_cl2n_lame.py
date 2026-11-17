"""Domain-centred (CL2N) LAME affinity on top of mixtta_ms224sp2cpr.

LAME's 5-NN graph is built on L2-normalised post-ReLU pooled features (fc input). All of them are non-negative,
so every cosine is high and dominated by one shared mean direction, which under a corruption also carries the
corruption's texture. Here the graph is built on normalize(f - mu_d) instead (SimpleShot's CL2N), where mu_d is
a per-domain running mean of the features at each input size, kept on the same schedule as the BN tracker
(running mean over the domain's rows, rate floored at alpha_min, restarted at a detected shift; the old-domain
rows of a split batch use the old domain's snapshot). k, T, the unary term and the row mass are unchanged.

Options:
  centre=None        off: exactly the base method.
  centre="domain"    centre each size's features on its domain mean, fuse, then LAME as before.
  zscore=True        also divide by a per-domain running std (var = E[f^2] - mu^2, std floored at
                     zs_floor * mean std over dimensions, so near-dead dimensions are not blown up).
  per_size_graph     one centred 5-NN graph per size, W = mean_k (A_k + A_k^T) / 2; edges both sizes agree on
                     weigh 1, edges one size finds weigh 0.5.
  _debug_copy        (test only) per_size_graph code path with the fused uncentred graph: must equal the base.
"""
import torch
import torch.nn.functional as F

import tta.mixtta_ms as mm
from tta.mixtta_ms import MixTTAMS


@torch.no_grad()
def lame_W(prob, W, max_iter=100, tol=1e-8):
    """Same fixed point as tta.lame.lame, with a given affinity matrix W."""
    logp = torch.log(prob.float() + 1e-10)
    z = prob.float()
    for _ in range(max_iter):
        z_new = torch.softmax(logp + W @ z, 1)
        if (z_new - z).abs().max() < tol:
            return z_new
        z = z_new
    return z


def knn_graph(feats, knn):
    B = feats.size(0)
    f = F.normalize(feats.float(), dim=1)
    k = min(knn, B - 1)
    idx = (f @ f.t()).topk(k + 1, dim=1).indices[:, 1:]
    A = torch.zeros(B, B, device=f.device).scatter_(1, idx, 1.0)
    return (A + A.t()) / 2


class CL2NLame(MixTTAMS):
    def __init__(self, model, centre="domain", zscore=False, per_size_graph=False, mu_n0=0, zs_floor=0.1,
                 _debug_copy=False, **kw):
        super().__init__(model, **kw)
        self.centre, self.zscore, self.per_size_graph = centre, zscore, per_size_graph
        self.mu_n0, self.zs_floor, self._debug_copy = mu_n0, zs_floor, _debug_copy  # mu_n0: unused (no-op)
        n = len(self.sizes)
        self.mu, self.m2 = [None] * n, [None] * n
        self.mu_old, self.m2_old = None, None
        self._mu_t, self._rate = -1, None

    def _tracking_rate(self, x):
        n0 = len(self.shifts)
        a = super()._tracking_rate(x)
        if len(self.shifts) > n0:  # new domain: snapshot the old domain's centre for the split rows
            self.mu_old = [m.clone() if m is not None else None for m in self.mu]
            self.m2_old = [m.clone() if m is not None else None for m in self.m2]
            self._rate = 1.0
        else:
            nb = x.size(0) - self.j_old
            self._rate = max(self.alpha_min or 0.1, nb / self.n_dom) if self.alpha_min is not None else \
                max(self.alpha, 1.0 / self.k_dom)
        return a

    @torch.no_grad()
    def _update_centre(self, out):
        r = self.j_old
        for k, (_, f) in enumerate(out):
            new = f[r:].float()
            rate = 1.0 if self.mu[k] is None else self._rate
            m1, s2 = new.mean(0), (new ** 2).mean(0)
            self.mu[k] = m1 if rate >= 1.0 else (1 - rate) * self.mu[k] + rate * m1
            if self.zscore:
                self.m2[k] = s2 if rate >= 1.0 or self.m2[k] is None else (1 - rate) * self.m2[k] + rate * s2
        self._mu_t = self.t

    def _transform(self, f, k):
        """Centre (and optionally z-score) the features of size k; old-domain rows of a split batch use the
        old domain's snapshot."""
        f = f.float()
        r = self.j_old

        def tr(x, mu, m2):
            x = x - mu
            if self.zscore and m2 is not None:
                sd = (m2 - mu ** 2).clamp_min(0).sqrt()
                x = x / (sd + self.zs_floor * sd.mean() + 1e-6)
            return x
        cur = tr(f, self.mu[k], self.m2[k])
        if r and self.mu_old is not None and self.mu_old[k] is not None:
            old = tr(f[:r], self.mu_old[k], self.m2_old[k] if self.m2_old is not None else None)
            cur = torch.cat([old, cur[r:]])
        return cur

    def _predict(self, out):
        if self.centre is None and not self._debug_copy:
            return super()._predict(out)
        if self.centre is not None and self._mu_t != self.t:  # once per batch (refine batches predict twice)
            self._update_centre(out)
        if self.centre is not None:
            fc = [self._transform(f, k) for k, (_, f) in enumerate(out)]
        else:
            fc = [f for _, f in out]
        if not self.per_size_graph and not self._debug_copy:
            return super()._predict([(p, f) for (p, _), f in zip(out, fc)])
        # copied body of MixTTAMS._predict with a per-size graph
        prob = torch.stack([p for p, _ in out]).mean(0)
        C = prob.size(1)
        if self.counts is None:
            self.counts = torch.zeros(C, device=prob.device)
        if self.prior_lambda:
            pi = (self.counts + self.prior_tau / C) / (self.n_seen + self.prior_tau)
            prob = prob * (C * pi).pow(-self.prior_lambda)
            prob = prob / prob.sum(1, keepdim=True)
        if self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        if self.temporal_mu and self.h_prev is not None:
            prob = prob * (self.h_prev + self.temporal_eps).pow(self.temporal_mu)
            prob = prob / prob.sum(1, keepdim=True)
        if prob.size(0) < 2:
            z = prob
        elif self.per_size_graph:
            W = torch.stack([knn_graph(f, self.knn) for f in fc]).mean(0)
            z = lame_W(prob, W)
        else:  # _debug_copy: fused graph through lame_W
            z = lame_W(prob, knn_graph(torch.stack(fc).mean(0), self.knn))
        self.h_prev = z.mean(0)
        return z
