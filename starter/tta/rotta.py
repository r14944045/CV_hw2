"""RoTTA: Robust Test-Time Adaptation in Dynamic Scenarios (Yuan et al., CVPR 2023).
Follows the official implementation (github.com/BIT-DA/RoTTA).

Components (each can be switched off for ablations):
  1. Robust BN                          norm.RobustBN2d          norm="robust" (alt: "batch", "source")
  2. CSTU memory bank                   memory.CSTU              use_memory
  3. Teacher-student + timeliness       teacher.MeanTeacher,     use_timeliness
     reweighting                        losses.timeliness_weight
Student sees strong augmentations (augment.strong_augment); only norm-layer affine params are trained.
"""
import torch

from .augment import strong_augment
from .common import NUM_CLASSES, forward, set_trainable, setup_cuda
from .losses import entropy, soft_cross_entropy, timeliness_weight
from .memory import CSTU
from .norm import configure_norm
from .teacher import MeanTeacher


class RoTTA:
    """For each incoming batch:
      1. The teacher predicts it in eval mode (no stat update); these predictions are returned.
      2. Each sample goes to the memory bank with the teacher's pseudo-label and entropy.
      3. Every `update_every` samples: teacher(memory) -> soft targets (updates teacher stats);
         student(strong_aug(memory)) -> loss = mean(timeliness(age) * CE); Adam step on the student;
         teacher <- EMA(student).
    Without memory, step 3 runs on the current batch after each batch.
    """

    def __init__(self, model, lr=1e-3, alpha=0.05, nu=0.001, memory_size=64, update_every=64,
                 lambda_t=1.0, lambda_u=1.0, norm="robust", use_memory=True, use_timeliness=True):
        setup_cuda()
        self.student = configure_norm(model, norm, alpha)
        params = set_trainable(self.student, "norm")
        self.teacher = MeanTeacher(self.student, keep=1 - nu)
        self.opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        self.memory = CSTU(memory_size, NUM_CLASSES, lambda_t, lambda_u) if use_memory else None
        self.use_timeliness, self.update_every, self.seen = use_timeliness, update_every, 0

    def __call__(self, x):
        with torch.no_grad():
            prob = self.teacher(x, update_stats=False).softmax(1)
        pseudo = prob.argmax(1)
        if self.memory is None:
            self.update(x, ages=None)
            return pseudo
        labels, uncs = pseudo.tolist(), entropy(prob).tolist()
        for i in range(x.size(0)):
            self.memory.add(x[i].clone(), labels[i], uncs[i])
            self.seen += 1
            if self.seen % self.update_every == 0:
                xs, ages = self.memory.get()
                if xs:
                    self.update(torch.stack(xs), torch.tensor(ages, device=x.device))
        return pseudo

    def update(self, x, ages):
        with torch.no_grad():
            p_teacher = self.teacher(x, update_stats=True).softmax(1)
        loss = soft_cross_entropy(forward(self.student, strong_augment(x), True), p_teacher)
        if ages is not None and self.use_timeliness:
            loss = loss * timeliness_weight(ages)
        self.opt.zero_grad(set_to_none=True)
        loss.mean().backward()
        self.opt.step()
        self.teacher.update(self.student)
