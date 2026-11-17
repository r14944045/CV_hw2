"""labelshift_tribe_bbn: per-domain class-balanced BN statistics from pseudo-labels (TRIBE-style balanced BN),
deferred and shrunk to the plain (pooled) tracker, on top of mixtta_ms224sp2cpr.

The plain tracker of each MultiMixBN2d is a row-weighted running mean of the batch statistics of the domain seen
so far, i.e. the moments of the domain's partial (label-skewed) class mixture. Here every BN layer from
`start_layer` on additionally keeps, per size tracker k, class-conditional running accumulators (same rate a_t as
the tracker, so they reset at a detected shift where a = 1):
    N_c  <- (1-a) N_c  + a * sum_i z_ic / B'
    S1_c <- (1-a) S1_c + a * sum_i z_ic (m_i - mu_src) / B'
    S2_c <- (1-a) S2_c + a * sum_i z_ic [v_i + (m_i - mu_src)^2] / B'
(m_i, v_i: per-sample spatial mean / variance of the stat rows; z: LAME output). After z is known:
    mu_c = (n_eff S1_c + kappa d_pool) / (n_eff N_c + kappa),  E2_c likewise with e_pool,   n_eff = min(n_dom, B'/alpha_min)
    d_pool = plain_mean - mu_src,  e_pool = plain_var + d_pool^2   (kappa -> inf gives exactly the plain tracker)
    mu_bal = mean_c mu_c + mu_src,  var_bal = mean_c E2_c - (mean_c mu_c)^2
    tracker k <- (1-beta) plain + beta * bal          (var only if var_mode == "meanvar")
The plain tracker is kept as a shadow, updated update-first from the tracker in use:
plain_t = (1-a) * tracker_{t-1} + a * batch_t (the same value the base layer normalises the batch with).
_balance is idempotent: it always recomputes from the pre-batch accumulators saved at the train forward, so the
second _predict call of a refine batch recomputes with the new z rather than accumulating twice.

Switches: bal=False -> exactly MixTTAMS (no class swap). beta=0 -> all machinery runs but the tracker is unchanged.
"""
import torch

from tta.mixbn import MultiMixBN2d
from tta.mixtta_ms import MixTTAMS


class _Stash:
    """Per-tracker flat buffers shared by the balanced layers of one method object."""

    def __init__(self, n_trackers, B, ch_tot, C, device):
        f = dict(device=device, dtype=torch.float32)
        self.M = [torch.zeros(B, ch_tot, **f) for _ in range(n_trackers)]   # m_i - mu_src
        self.Q = [torch.zeros(B, ch_tot, **f) for _ in range(n_trackers)]   # v_i + (m_i - mu_src)^2
        self.pm = [torch.zeros(ch_tot, **f) for _ in range(n_trackers)]     # plain post-update mean
        self.pv = [torch.zeros(ch_tot, **f) for _ in range(n_trackers)]     # plain post-update var
        self.src_mean = torch.zeros(ch_tot, **f)
        self.alpha = [1.0] * n_trackers
        self.rows = [0] * n_trackers
        self.fresh = [False] * n_trackers  # a train forward stashed this batch (reset at each __call__)
        # committed accumulators (state before the current batch) and the candidate from the latest _balance
        self.acc = [(torch.zeros(C, **f), torch.zeros(C, ch_tot, **f), torch.zeros(C, ch_tot, **f))
                    for _ in range(n_trackers)]
        self.new = [None] * n_trackers


class BalMixBN2d(MultiMixBN2d):
    """MultiMixBN2d that also stashes per-sample statistics of the stat rows (train mode only)."""

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        k = self.k
        mean_t, var_t = getattr(self, f"mean{k}"), getattr(self, f"var{k}")
        xs = x if self.stat_rows is None else x[:self.stat_rows]
        xf = xs.detach().float()
        b_var, b_mean = torch.var_mean(xf, dim=(0, 2, 3), unbiased=False)  # same op as the base (exact match)
        mean_t = (1 - self.alpha) * mean_t + self.alpha * b_mean
        var_t = (1 - self.alpha) * var_t + self.alpha * b_var
        setattr(self, f"mean{k}", mean_t)
        setattr(self, f"var{k}", var_t)
        st, o, Ch = self._st, self._off, mean_t.numel()
        v, m = torch.var_mean(xf, dim=(2, 3), unbiased=False)
        B = m.size(0)
        d = m - self.src_mean
        st.M[k][:B, o:o + Ch] = d
        st.Q[k][:B, o:o + Ch] = v + d * d
        st.pm[k][o:o + Ch] = mean_t
        st.pv[k][o:o + Ch] = var_t
        st.alpha[k], st.rows[k], st.fresh[k] = float(self.alpha), B, True
        mean = (1 - self.w) * mean_t + self.w * self.src_mean
        var = (1 - self.w) * var_t + self.w * self.src_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.to(x.dtype).view(1, -1, 1, 1) + shift.to(x.dtype).view(1, -1, 1, 1)


class MixTTAMSBal(MixTTAMS):
    def __init__(self, model, bal=True, kappa=16.0, beta=1.0, hard=False, var_mode="meanvar", start_layer=1,
                 n_classes=20, debug=False, **kw):
        super().__init__(model, **kw)
        self.bal, self.kappa, self.beta, self.hard, self.var_mode = bal, kappa, beta, hard, var_mode
        self.debug, self.dbg = debug, []
        if not bal:
            return
        if isinstance(start_layer, str):  # e.g. "layer3": first BN inside that stage
            names = {id(m): n for n, m in self.model.named_modules()}
            start_layer = next(i for i, l in enumerate(self.layers) if names[id(l)].startswith(start_layer + "."))
        start_layer = max(1, int(start_layer))  # layers[0] stays plain (the shift detector reads it)
        self.bal_layers = self.layers[start_layer:]
        offs, o = [], 0
        for l in self.bal_layers:
            offs.append(o)
            o += l.src_mean.numel()
        dev = self.layers[0].src_mean.device
        self._st = _Stash(len(self.sizes), 64, o, n_classes, dev)
        for l, off in zip(self.bal_layers, offs):
            l.__class__ = BalMixBN2d
            l._st, l._off = self._st, off
            self._st.src_mean[off:off + l.src_mean.numel()] = l.src_mean
        self._offs = offs
        self._st_srcvar = torch.cat([l.src_var.float() for l in self.bal_layers])
        names = {id(m): n for n, m in self.model.named_modules()}
        self._stage = {}
        for l, off in zip(self.bal_layers, offs):
            g = names[id(l)].split(".")[0]
            lo, hi = self._stage.get(g, (off, off))
            self._stage[g] = (lo, off + l.src_mean.numel())
        self._chs = [l.src_mean.numel() for l in self.bal_layers]

    def __call__(self, x):
        if self.bal:
            st = self._st
            for k in range(len(self.sizes)):  # commit the previous batch's accumulators
                if st.new[k] is not None:
                    st.acc[k] = st.new[k]
                    st.new[k] = None
                st.fresh[k] = False
        return super().__call__(x)

    @torch.no_grad()
    def _balance(self, z):
        st = self._st
        rows = z[self.j_old:].float()
        if self.hard:
            rows = torch.nn.functional.one_hot(rows.argmax(1), rows.size(1)).float()
        for k in range(len(self.sizes)):
            if not st.fresh[k]:
                continue
            Bp, a = st.rows[k], st.alpha[k]
            assert Bp == rows.size(0), (Bp, rows.size(0))
            P = rows.t() / Bp                                    # (C, B')
            N0, S10, S20 = st.acc[k]
            N = (1 - a) * N0 + a * P.sum(1)
            S1 = (1 - a) * S10 + a * (P @ st.M[k][:Bp])
            S2 = (1 - a) * S20 + a * (P @ st.Q[k][:Bp])
            st.new[k] = (N, S1, S2)
            amin = self.alpha_min if self.alpha_min else 1.0
            n_eff = min(float(self.n_dom), Bp / amin)
            pm, pv = st.pm[k], st.pv[k]
            d_pool = pm - st.src_mean
            e_pool = pv + d_pool * d_pool
            den = (n_eff * N + self.kappa).unsqueeze(1)          # (C, 1)
            mu_c = (n_eff * S1 + self.kappa * d_pool) / den
            mu_b = mu_c.mean(0)
            mean_new = (1 - self.beta) * pm + self.beta * (mu_b + st.src_mean)
            if self.var_mode == "meanvar":
                e2_c = (n_eff * S2 + self.kappa * e_pool) / den
                var_b = e2_c.mean(0) - mu_b * mu_b
                var_b = torch.where(var_b > 0, var_b, pv)  # numerical guard (near-dead channels)
                var_new = (1 - self.beta) * pv + self.beta * var_b
            else:
                var_new = pv
            if self.debug and k == 0:
                sd = (0.5 * pv + 0.5 * self._st_srcvar).sqrt() + 1e-3  # std actually used for normalisation
                rel = (mean_new - pm).abs() * 0.5 / sd
                seen = (n_eff * N) > 4
                spread = ((mu_c[seen] - d_pool).abs() * 0.5 / sd).mean(0) if seen.any() else rel * 0
                r = dict(sumN=float(N.sum()), a=a, n_eff=n_eff, nseen=int(seen.sum()),
                         vr=float((var_new / pv.clamp_min(1e-6)).median()))
                for g, (lo, hi) in self._stage.items():
                    r[g] = round(float(rel[lo:hi].mean()), 4)
                    r[g + "_cls"] = round(float(spread[lo:hi].mean()), 3)
                self.dbg.append(r)
            for l, o, Ch in zip(self.bal_layers, self._offs, self._chs):
                setattr(l, f"mean{k}", mean_new[o:o + Ch].clone())
                setattr(l, f"var{k}", var_new[o:o + Ch].clone())

    def _predict(self, out):
        z = super()._predict(out)
        if self.bal:
            self._balance(z)
        return z
