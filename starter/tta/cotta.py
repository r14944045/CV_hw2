"""CoTTA: Continual Test-Time Domain Adaptation (Wang et al., CVPR 2022).
Follows the official implementation (github.com/qinenergy/cotta), ImageNet-C config by default.

Components (each can be switched off for ablations):
  1. Weight-averaged pseudo-labels        teacher.MeanTeacher                (always on; mt=0 -> no EMA)
  2. Augmentation-averaged pseudo-labels  teacher.augmentation_averaged_logits,
     gated by anchor confidence           teacher.low_confidence             aug_avg, n_aug, ap
  3. Stochastic restoration               restore.StochasticRestore          restore, rst
Model setup: all parameters trainable, BN on test-batch statistics (norm="batch"); norm="robust"
swaps in RoTTA's Robust BN for non-i.i.d. streams.
"""
from copy import deepcopy

import torch

from .common import forward, set_trainable, setup_cuda
from .losses import soft_cross_entropy
from .norm import configure_norm
from .restore import StochasticRestore
from .teacher import MeanTeacher, augmentation_averaged_logits, low_confidence


class CoTTA:
    """For each incoming batch (one SGD step per batch):
      1. Teacher logits on x. If the anchor (frozen source model) is not confident
         (mean max-softmax < ap), replace them with the mean over n_aug augmented copies.
      2. Student logits on x; loss = CE(student, softmax(teacher logits)); SGD step.
      3. Teacher <- EMA(student, mt). 4. Stochastic restoration of the student (prob rst).
    Returns argmax of the (possibly augmentation-averaged) teacher logits.
    """

    def __init__(self, model, lr=0.01, momentum=0.9, mt=0.999, rst=0.01, ap=0.9, n_aug=32,
                 norm="batch", alpha=0.05, train_params="all", aug_avg=True, restore=True):
        setup_cuda()
        self.student = configure_norm(model, norm, alpha)
        params = set_trainable(self.student, train_params)
        self.anchor = deepcopy(self.student).requires_grad_(False)
        self.teacher = MeanTeacher(self.student, keep=mt)
        self.opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        self.restorer = StochasticRestore(self.student, rst) if restore else None
        self.aug_avg, self.ap, self.n_aug = aug_avg, ap, n_aug
        self.n_batches = self.n_augmented = 0  # how often the augmentation gate fired

    def __call__(self, x):
        with torch.no_grad():
            t_logits = self.teacher(x, update_stats=True)
            if self.aug_avg and low_confidence(forward(self.anchor, x, True), self.ap):
                t_logits = augmentation_averaged_logits(self.teacher, x, self.n_aug)
                self.n_augmented += 1
        self.n_batches += 1

        loss = soft_cross_entropy(forward(self.student, x, True), t_logits.softmax(1)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.teacher.update(self.student)
        if self.restorer is not None:
            self.restorer()
        return t_logits.argmax(1)
