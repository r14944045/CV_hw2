"""CoTTA's stochastic restoration: after each update, reset a random fraction p of the trainable
weights (element-wise) to their source values, to prevent drift / catastrophic forgetting."""
import torch


class StochasticRestore:
    def __init__(self, model, p):
        self.p = p
        self.params = [q for q in model.parameters() if q.requires_grad]
        self.source = [q.detach().clone() for q in self.params]

    @torch.no_grad()
    def __call__(self):
        for q, s in zip(self.params, self.source):
            q.copy_(torch.where(torch.rand_like(q) < self.p, s, q))
