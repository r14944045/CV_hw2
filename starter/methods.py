"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NUM_CLASSES = 20


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(1)


# ----------------------------------------------------------------------------------------------
# RoTTA: Robust Test-Time Adaptation in Dynamic Scenarios (Yuan et al., CVPR 2023)
# Follows the official implementation (github.com/BIT-DA/RoTTA).
# ----------------------------------------------------------------------------------------------

class RobustBN2d(nn.Module):
    """Robust BN: normalise with global statistics, updated by an EMA of batch statistics.

    train mode: stats <- (1 - alpha) * stats + alpha * batch_stats, then normalise with the new stats.
    eval mode:  normalise with the stored stats, no update.
    Stats start from the source model's running mean/var.
    """

    def __init__(self, bn: nn.BatchNorm2d, alpha):
        super().__init__()
        self.alpha, self.eps = alpha, bn.eps
        self.register_buffer("mean", bn.running_mean.detach().clone())
        self.register_buffer("var", bn.running_var.detach().clone())
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())

    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)
            mean = (1 - self.alpha) * self.mean + self.alpha * b_mean
            var = (1 - self.alpha) * self.var + self.alpha * b_var
            self.mean, self.var = mean.detach(), var.detach()
        else:
            mean, var = self.mean, self.var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)


def convert_bn(module, alpha):
    """Replace every BatchNorm2d in `module` (in place) with RobustBN2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, RobustBN2d(child, alpha))
        else:
            convert_bn(child, alpha)
    return module


class CSTU:
    """Category-balanced memory bank with Timeliness and Uncertainty.

    Each slot holds [x, uncertainty, age]. Per-class quota = capacity / num_classes. When full, the
    item with the highest heuristic score (old and/or uncertain) is evicted if its score exceeds the
    new sample's score; otherwise the new sample is dropped.
    """

    def __init__(self, capacity, num_classes, lambda_t=1.0, lambda_u=1.0):
        self.capacity, self.num_classes = capacity, num_classes
        self.per_class = capacity / num_classes
        self.lambda_t, self.lambda_u = lambda_t, lambda_u
        self.items = [[] for _ in range(num_classes)]

    def score(self, age, uncertainty):
        return (self.lambda_t / (1 + math.exp(-age / self.capacity))
                + self.lambda_u * uncertainty / math.log(self.num_classes))

    def occupancy(self):
        return sum(len(c) for c in self.items)

    def _evict(self, classes, new_score):
        best = None  # (score, class, index)
        for c in classes:
            for i, (_, unc, age) in enumerate(self.items[c]):
                s = self.score(age, unc)
                if best is None or s >= best[0]:
                    best = (s, c, i)
        if best is None:
            return True
        if best[0] > new_score:
            self.items[best[1]].pop(best[2])
            return True
        return False

    def add(self, x, label, uncertainty):
        new_score = self.score(0, uncertainty)
        if len(self.items[label]) < self.per_class:
            if self.occupancy() < self.capacity:
                accept = True
            else:  # bank full: try to evict from the most populated class(es)
                sizes = [len(c) for c in self.items]
                majority = [c for c, n in enumerate(sizes) if n == max(sizes)]
                accept = self._evict(majority, new_score)
        else:  # this class is at quota: try to evict from the same class
            accept = self._evict([label], new_score)
        if accept:
            self.items[label].append([x, uncertainty, 0])
        for c in self.items:
            for item in c:
                item[2] += 1

    def get(self):
        xs, ages = [], []
        for c in self.items:
            for x, _, age in c:
                xs.append(x)
                ages.append(age / self.capacity)
        return xs, ages


def strong_augment(x):
    """Per-sample strong augmentation on a normalised batch (CoTTA/RoTTA-style, run on the GPU).

    Colour jitter (brightness, contrast, saturation, gamma), random affine (rotation +-15 deg,
    translation +-1/16, scale 0.9-1.1), horizontal flip, and small Gaussian noise.
    """
    mean, std = MEAN.to(x.device), STD.to(x.device)
    x = (x * std + mean).clamp(0, 1)
    B, dev = x.size(0), x.device

    def U(lo, hi):
        return torch.empty(B, 1, 1, 1, device=dev).uniform_(lo, hi)

    x = (x * U(0.6, 1.4)).clamp(0, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * U(0.7, 1.3) + m).clamp(0, 1)
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    x = ((x - gray) * U(0.5, 1.5) + gray).clamp(0, 1)
    x = x.clamp_min(1e-6) ** U(0.7, 1.3)

    ang = U(-15, 15).view(B) * math.pi / 180
    scale = U(0.9, 1.1).view(B)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    tx, ty = U(-1 / 8, 1 / 8).view(B), U(-1 / 8, 1 / 8).view(B)  # 1/16 of width in [-1, 1] coords
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    x = (x + 0.005 * torch.randn_like(x)).clamp(0, 1)
    return (x - mean) / std


class RoTTA:
    """RoTTA = Robust BN + CSTU memory bank + EMA teacher / student with timeliness reweighting.

    For each incoming batch:
      1. The teacher (eval mode) predicts it; these predictions are returned.
      2. Each sample is offered to the memory bank with the teacher's pseudo-label and entropy.
      3. Every `update_every` samples, the student is trained on the strongly augmented memory
         against teacher targets on the clean memory (loss weighted by sample age), then the
         teacher is updated as an EMA of the student. Only BN affine parameters are trained.
    """

    def __init__(self, model, lr=1e-3, alpha=0.05, nu=0.001, memory_size=64, update_every=64,
                 lambda_t=1.0, lambda_u=1.0):
        torch.backends.cudnn.benchmark = True
        self.student = convert_bn(model, alpha).requires_grad_(False)
        self.student_params = []
        for m in self.student.modules():
            if isinstance(m, RobustBN2d):
                m.requires_grad_(True)
                self.student_params += [m.weight, m.bias]
        self.teacher = deepcopy(self.student).requires_grad_(False)
        self.teacher_params = [p for m in self.teacher.modules() if isinstance(m, RobustBN2d)
                               for p in (m.weight, m.bias)]
        self.opt = torch.optim.Adam(self.student_params, lr=lr, betas=(0.9, 0.999))
        self.mem = CSTU(memory_size, NUM_CLASSES, lambda_t, lambda_u)
        self.nu, self.update_every, self.seen = nu, update_every, 0

    def __call__(self, x):
        with torch.no_grad():
            self.teacher.eval()
            prob = self.teacher(x).softmax(1)
            pseudo = prob.argmax(1)
            entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
        labels, uncs = pseudo.tolist(), entropy.tolist()
        for i in range(x.size(0)):
            self.mem.add(x[i].clone(), labels[i], uncs[i])
            self.seen += 1
            if self.seen % self.update_every == 0:
                self.update()
        return pseudo

    def update(self):
        xs, ages = self.mem.get()
        if not xs:
            return
        x = torch.stack(xs)
        self.teacher.train()
        self.student.train()
        with torch.no_grad():
            p_teacher = self.teacher(x).softmax(1)
        logp_student = self.student(strong_augment(x)).log_softmax(1)
        ages = torch.tensor(ages, device=x.device)
        weight = torch.exp(-ages) / (1 + torch.exp(-ages))  # timeliness: newer samples weigh more
        loss = (-(p_teacher * logp_student).sum(1) * weight).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for pt, ps in zip(self.teacher_params, self.student_params):
                pt.mul_(1 - self.nu).add_(ps, alpha=self.nu)


METHODS = {"source": Source, "rotta": RoTTA}
