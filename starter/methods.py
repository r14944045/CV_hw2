"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.

Components live in tta/ (one file per component, see tta/__init__.py). Variants are registered
below with functools.partial, so each ablation is just `--method <name>`.
"""
from functools import partial

from tta import BOSA, CoTTA, MixTTA, NormOnly, RoTTA, Source

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

    # BOSA and ablations (cf. the paper's Table 2)
    "bosa": BOSA,
    "bosa_noreg": partial(BOSA, reg=0.0),              # w/o adaptive update constraint (Eqs. 5-7)
    "bosa_noalign": partial(BOSA, adaptive=False, reg=0.0),  # w/o statistics alignment: fixed alpha=0.05 (RBN)
    "bosa_nomem": partial(BOSA, use_memory=False),     # w/o balanced memory bank: current batch
    "bosa_nosparse": partial(BOSA, q=0.0),             # w/o activation sparsification
    "bosa_ent": partial(BOSA, loss="entropy"),         # w/o self-training: entropy minimisation
    # deviation from the paper: halve the constraint step (2 * 0.5 * alpha_t <= 1). The literal step
    # 2 * alpha_t > 1 for alpha_t > 0.5 reflects the stats through the source and locks alpha_t at 1 on test.
    "bosa_stable": partial(BOSA, reg=0.5),
    "bosa_stable_nomem": partial(BOSA, reg=0.5, use_memory=False),

    # MixTTA (forward-only) and ablations
    "mixtta": MixTTA,                                  # alpha=0.2, w=0.5, flip + 4 shifts, LAME
    "mixtta_w7": partial(MixTTA, alpha=0.4, w=0.7),    # more source statistics, faster tracking
    "mixtta_nolame": partial(MixTTA, use_lame=False),  # no LAME
    "mixtta_noaug": partial(MixTTA, flip=False, n_shift=0),  # no flip/shift averaging
    "mixtta_stats": partial(MixTTA, flip=False, n_shift=0, use_lame=False),  # mixed statistics only
    "mixtta_lag": partial(MixTTA, update_first=False),  # predict before updating the statistics
}
