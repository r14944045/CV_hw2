"""Loss functions and per-sample weights (all return per-sample values; reduce in the caller)."""
import torch


def entropy(prob, eps=1e-6):
    """Prediction entropy from probabilities, shape (B,). Used as uncertainty in RoTTA's CSTU."""
    return -(prob * torch.log(prob + eps)).sum(1)


def soft_cross_entropy(student_logits, teacher_prob):
    """Cross-entropy between teacher probabilities (target) and student logits, shape (B,).
    The consistency loss of both RoTTA and CoTTA."""
    return -(teacher_prob * student_logits.log_softmax(1)).sum(1)


def timeliness_weight(ages):
    """RoTTA timeliness reweighting: E(A) = exp(-A) / (1 + exp(-A)), A = age / capacity.
    Newer samples (small age) get larger weight (max 0.5 at age 0)."""
    return torch.exp(-ages) / (1 + torch.exp(-ages))
