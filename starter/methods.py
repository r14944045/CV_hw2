"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.

Components live in tta/ (one file per component, see tta/__init__.py). Variants are registered
below with functools.partial, so each ablation is just `--method <name>`.
"""
from functools import partial

from tta import CoTTA, NormOnly, RoTTA, Source

METHODS = {
    # baselines
    "source": Source,                                  # no adaptation
    "bn": partial(NormOnly, norm="batch"),             # test-batch BN statistics
    "rbn": partial(NormOnly, norm="robust"),           # Robust BN only (RoTTA component 1)

    # RoTTA and ablations
    "rotta": RoTTA,
    "rotta_bn": partial(RoTTA, norm="batch"),          # Robust BN -> test-batch BN
    "rotta_nomem": partial(RoTTA, use_memory=False),   # no CSTU: train on the current batch
    "rotta_notime": partial(RoTTA, use_timeliness=False),  # uniform weights instead of timeliness

    # CoTTA and ablations
    "cotta": CoTTA,
    "cotta_noaug": partial(CoTTA, aug_avg=False),      # no augmentation-averaged pseudo-labels
    "cotta_norst": partial(CoTTA, restore=False),      # no stochastic restoration
    "cotta_fast": partial(CoTTA, n_aug=4),             # fewer augmentations (90 s budget)
    "cotta_rbn": partial(CoTTA, norm="robust"),        # CoTTA + Robust BN (non-i.i.d. stream)
}
