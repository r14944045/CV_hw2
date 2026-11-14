"""BOSA: On-Device Realistic Test-Time Adaptation via Bias-Resistant Statistical Alignment
(Bai et al., IJCAI 2026). No official code is available; this follows the paper (Secs. 4.2-4.4, 5.1).

Components (each can be switched off for ablations, cf. the paper's Table 2):
  1. Self-training with a class-balanced memory bank   memory.CSTU, teacher.MeanTeacher,
     (Eq. 1; teacher = EMA of the student)             losses.soft_cross_entropy     use_memory, loss
  2. Bias-resistant online statistics alignment        align.AlignedBN2d             adaptive, reg
     (adaptive alpha from symmetric KL, Eqs. 2-4;
      Wasserstein-2 update constraint, Eqs. 5-7)
  3. Saliency-guided activation sparsification and     sparsify.sparse_bn            q
     gradient reconstruction (Eqs. 8-10)

Hyperparameters (paper, ImageNet-C / ResNet-50): Adam lr 1e-3, batch 64, memory capacity 64, q = 0.7,
teacher EMA keep rate v_m = 0.99. Only the student's BN affine parameters are trained.

Assumptions where the paper is silent (supplementary material not available):
  - The memory bank is RoTTA's CSTU ("a design similar to [Yuan et al., 2023]"); its uncertainty
    score is the "low-entropy filtering". No extra entropy threshold.
  - The paper writes the EMA as teacher <- (1 - v_m) teacher + v_m student, which with v_m = 0.99
    would make the teacher a copy of the student. We read v_m as the keep rate (as in RoTTA's 0.999).
  - As in RoTTA, teacher and student each align their own BN statistics during their forward on the
    memory bank. Predictions come from the teacher in eval mode.
  - The augmented view is RoTTA's / CoTTA's strong augmentation; the target is the teacher's softmax.

Warning: with the default reg=1.0 (Eqs. 6-7 as printed, unit step) the statistics lock into an
oscillation on the HW2 test stream; see the caveat in align.py. `bosa_stable` (reg=0.5) avoids it.
"""
import torch

from .align import configure_aligned_norm
from .augment import strong_augment
from .common import NUM_CLASSES, forward, set_trainable, setup_cuda
from .losses import entropy, soft_cross_entropy
from .memory import CSTU
from .teacher import MeanTeacher


class BOSA:
    """For each incoming batch:
      1. The teacher predicts it in eval mode (no stat update); these predictions are returned.
      2. Each sample goes to the memory bank with the teacher's pseudo-label and entropy.
      3. Every `update_every` samples (once per batch), on the memory bank:
         teacher(bank) -> soft targets, and the teacher's BN statistics are aligned to the bank;
         student(aug(bank)) -> aligned statistics + sparsified BN backward; loss = mean CE (Eq. 1);
         Adam step on the student's BN affine parameters; teacher <- EMA(student).
    Without memory, step 3 runs on the current batch.
    """

    def __init__(self, model, lr=1e-3, keep=0.99, memory_size=64, update_every=64, q=0.7,
                 adaptive=True, reg=1.0, alpha=0.05, use_memory=True, loss="self_training"):
        setup_cuda()
        self.student = configure_aligned_norm(model, alpha=alpha, adaptive=adaptive, reg=reg, q=q)
        params = set_trainable(self.student, "norm")
        self.teacher = MeanTeacher(self.student, keep=keep)
        self.opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        self.memory = CSTU(memory_size, NUM_CLASSES) if use_memory else None
        self.loss, self.update_every, self.seen = loss, update_every, 0

    def __call__(self, x):
        with torch.no_grad():
            prob = self.teacher(x, update_stats=False).softmax(1)
        pseudo = prob.argmax(1)
        if self.memory is None:
            self.update(x)
            return pseudo
        labels, uncs = pseudo.tolist(), entropy(prob).tolist()
        for i in range(x.size(0)):
            self.memory.add(x[i].clone(), labels[i], uncs[i])
            self.seen += 1
            if self.seen % self.update_every == 0:
                xs, _ = self.memory.get()
                if xs:
                    self.update(torch.stack(xs))
        return pseudo

    def update(self, x):
        with torch.no_grad():
            p_teacher = self.teacher(x, update_stats=True).softmax(1)
        if self.loss == "self_training":
            loss = soft_cross_entropy(forward(self.student, strong_augment(x), True), p_teacher)
        elif self.loss == "entropy":  # Table 2 "w/o Self-Training": entropy minimisation instead
            loss = entropy(forward(self.student, x, True).softmax(1))
        else:
            raise ValueError(f"unknown loss: {self.loss}")
        self.opt.zero_grad(set_to_none=True)
        loss.mean().backward()
        self.opt.step()
        self.teacher.update(self.student)
