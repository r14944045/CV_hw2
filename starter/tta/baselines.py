"""Non-training baselines."""
import torch

from .common import forward, setup_cuda
from .norm import configure_norm


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(1)


class NormOnly:
    """Adapt normalisation statistics only, no gradient updates.
    norm="batch": test-batch BN (BN-adapt); norm="robust": RoTTA's Robust BN alone."""

    def __init__(self, model, norm="robust", alpha=0.05):
        setup_cuda()
        self.model = configure_norm(model, norm, alpha)

    @torch.no_grad()
    def __call__(self, x):
        return forward(self.model, x, update_stats=True).argmax(1)
