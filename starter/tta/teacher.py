"""Teacher-side components.

MeanTeacher                    : EMA teacher (RoTTA's teacher; CoTTA's weight-averaged pseudo-labels).
augmentation_averaged_logits   : CoTTA's augmentation-averaged pseudo-labels.
low_confidence                 : CoTTA's gate deciding when to augment (anchor = source model).
"""
from copy import deepcopy

import torch

from .augment import strong_augment
from .common import forward


class MeanTeacher:
    """Teacher whose weights are an exponential moving average of the student's:
    theta_t <- keep * theta_t + (1 - keep) * theta_s.
    Norm statistics are buffers, not parameters: the teacher keeps its own, updated by its own forwards."""

    def __init__(self, student, keep):
        self.model = deepcopy(student).requires_grad_(False)
        self.keep = keep

    def __call__(self, x, update_stats):
        return forward(self.model, x, update_stats)

    @torch.no_grad()
    def update(self, student):
        for t, s in zip(self.model.parameters(), student.parameters()):
            t.mul_(self.keep).add_(s.detach(), alpha=1 - self.keep)


@torch.no_grad()
def augmentation_averaged_logits(teacher, x, n_aug):
    """Mean teacher logits over `n_aug` strongly augmented copies of x. Augmented forwards do not
    update norm statistics (matters only for Robust BN; batch-stat BN ignores the mode)."""
    return torch.stack([teacher(strong_augment(x), update_stats=False) for _ in range(n_aug)]).mean(0)


@torch.no_grad()
def low_confidence(anchor_logits, threshold):
    """CoTTA's domain-gap proxy: True if the anchor (source) model's mean max-softmax < threshold."""
    return anchor_logits.softmax(1).max(1).values.mean().item() < threshold
