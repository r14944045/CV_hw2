"""selftrain_student_repredict: cheap train-then-repredict on every steady-state batch.

On every batch that is neither a split batch nor a refine (second-pass) batch:
  1. first-pass ensemble (160 + 224 px, 4 views each), prior, T=2, LAME -> z (unchanged);
  2. the usual self-training steps on gamma/beta with target z (unchanged);
  3. NEW: one no-grad eval forward of the first `rp_views` cached 160 px views (x, flip(x)) with the
     just-updated gamma/beta and tracker 0's stored statistics -> student probabilities p_s, features f_s;
  4. re-predict: prob = (1 - rp_beta) * mean(p160, p224) + rp_beta * p_s, then the same prior, T and LAME
     (features: first-pass mean, or (2 * first + f_s) / 3 with rp_feats='mix'); argmax; the class counts
     follow this final prediction.
rp_beta = 0 switches the feature off (identical to MixTTAMS). No new RNG draws.
"""
import torch

from tta.lame import lame
from tta.mixtta_ms import MixTTAMS


class SelfTrainRepredict(MixTTAMS):
    def __init__(self, model, rp_beta=0.5, rp_views=2, rp_feats="first", **kw):
        super().__init__(model, **kw)
        self.rp_beta, self.rp_views, self.rp_feats = rp_beta, rp_views, rp_feats
        self._last_out = None
        self.n_repredict, self.n_flips = 0, 0

    def _predict(self, out):
        self._last_out = out
        return super()._predict(out)

    def _predict_fused(self, out, p_s, f_s):
        """Copy of MixTTAMS._predict with the student's probabilities (and optionally features) fused in."""
        b = self.rp_beta
        prob = (1 - b) * torch.stack([p for p, _ in out]).mean(0) + b * p_s
        feats = torch.stack([f for _, f in out]).mean(0)
        if self.rp_feats == "mix":
            feats = (2 * feats + f_s) / 3
        C = prob.size(1)
        if self.prior_lambda:
            pi = (self.counts + self.prior_tau / C) / (self.n_seen + self.prior_tau)
            prob = prob * (C * pi).pow(-self.prior_lambda)
            prob = prob / prob.sum(1, keepdim=True)
        if self.lame_T != 1.0:
            prob = torch.softmax(torch.log(prob + 1e-10) / self.lame_T, 1)
        if self.temporal_mu and self.h_prev is not None:
            prob = prob * (self.h_prev + self.temporal_eps).pow(self.temporal_mu)
            prob = prob / prob.sum(1, keepdim=True)
        z = lame(prob, feats, self.knn)
        self.h_prev = z.mean(0)
        return z

    @torch.no_grad()
    def _student_view(self, B):
        """Eval forward of the first rp_views cached 160 px views with the updated gamma/beta (tracker 0)."""
        self._set_tracker(0)
        V = self.rp_views
        with self.amp:
            logits = self.model.eval()(self._views[0][:V * B])
        return logits.float().softmax(1).view(V, B, -1).mean(0), self._feats.float().view(V, B, -1).mean(0)

    def __call__(self, x):
        if not self.rp_beta or self.opt is None:
            return super().__call__(x)
        self.j_old = 0
        alpha = self._tracking_rate(x)
        self._use_ema(True)
        if self.j_old:
            out = [self._scale_forward_split(self._resize(x, s), k, self.j_old) for k, s in enumerate(self.sizes)]
        else:
            out = [self._scale_forward(self._resize(x, s), k, alpha) for k, s in enumerate(self.sizes)]
        z = self._predict(out)
        refine = self.refine_batches and not self.j_old and self.k_dom <= self.refine_batches
        if refine:  # base second pass: train, then re-predict all views
            self._use_ema(False)
            self.self_train(x, z, steps=self.refine_steps)
            self._update_ema()
            self._use_ema(True)
            z = self._predict([self._eval_views(k, x.size(0)) for k in range(len(self.sizes))])
            self._use_ema(False)
        else:
            self._use_ema(False)
            self.self_train(x, z, rows=self.j_old if self.split_train == "new" else 0)
            self._update_ema()
            if self.j_old == 0:  # split batches: _views is stale, keep the first-pass prediction
                z0 = z
                p_s, f_s = self._student_view(x.size(0))
                z = self._predict_fused(self._last_out, p_s, f_s)
                self.n_repredict += 1
                self.n_flips += int((z.argmax(1) != z0.argmax(1)).sum())
        pred = z.argmax(1)
        self.counts += torch.bincount(pred, minlength=z.size(1)).float()
        self.n_seen += pred.numel()
        return pred
