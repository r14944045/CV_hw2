"""Shared helpers used by every method."""
import torch
import torch.nn as nn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NUM_CLASSES = 20


def setup_cuda():
    torch.backends.cudnn.benchmark = True


def forward(model, x, update_stats):
    """Run `model` on x. update_stats=True -> train mode (norm layers may use / update test statistics),
    False -> eval mode. ResNet-50 has no dropout, so the mode only affects normalisation layers."""
    model.train(update_stats)
    return model(x)


def norm_layers(model):
    from .norm import RobustBN2d
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, RobustBN2d))]


def set_trainable(model, which):
    """Freeze everything, then unfreeze `which`: "norm" (affine params of norm layers: RoTTA, TENT)
    or "all" (every parameter: CoTTA). Returns the list of trainable parameters."""
    model.requires_grad_(False)
    if which == "all":
        model.requires_grad_(True)
    elif which == "norm":
        for m in norm_layers(model):
            m.requires_grad_(True)
    else:
        raise ValueError(f"unknown trainable set: {which}")
    return [p for p in model.parameters() if p.requires_grad]
