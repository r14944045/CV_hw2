"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.

Components live in tta/ (one file per component, see tta/__init__.py). Variants are registered
below with functools.partial, so each ablation is just `--method <name>`.
"""
from functools import partial

from tta import BOSA, CoTTA, MixTTA, MixTTAMS, NormOnly, RoTTA, Source

_repick_s5 = partial(MixTTAMS, sizes=(160, 224), steps=3, lame_T=2.0, alpha_min=0.2,
                     shift_thr=9.0, split_shift=True)

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
    # 2 * alpha_t > 1 for alpha_t > 0.5 reflects the stats through the source and can lock alpha_t at 1.
    "bosa_stable": partial(BOSA, reg=0.5),
    "bosa_stable_nomem": partial(BOSA, reg=0.5, use_memory=False),

    # MixTTA (forward-only) and ablations
    "mixtta": MixTTA,                                  # alpha=0.2, w=0.5, flip + 4 shifts, LAME
    "mixtta_nolame": partial(MixTTA, use_lame=False),  # no LAME
    "mixtta_noaug": partial(MixTTA, flip=False, n_shift=0),  # no flip/shift averaging
    "mixtta_lag": partial(MixTTA, update_first=False),  # predict before updating the statistics
    # MixTTA with other normalisation statistics (MixBN2d settings; views use the stored statistics)
    "mixtta_srcbn": partial(MixTTA, alpha=0.0),               # frozen source statistics
    "mixtta_tbn": partial(MixTTA, alpha=1.0, w=0.0),          # current test batch only
    "mixtta_rbn": partial(MixTTA, alpha=0.05, w=0.0),         # Robust BN (RoTTA's EMA), no source mix
    "mixtta_w0": partial(MixTTA, w=0.0),                      # mixtta's alpha=0.2 without the source mix
    # faster tracking + LAME on view-averaged features, then self-training on the LAME output (BN gamma/beta, Adam 1e-3)
    "mixtta_st_notrain": partial(MixTTA, alpha=0.4, w=0.5, lame_feats="mean"),
    "mixtta_st2": partial(MixTTA, alpha=0.4, w=0.5, lame_feats="mean", lr=1e-3, train_steps=2),
    # MixTTA-MS (tta/mixtta_ms.py): MixTTA + self-training run at several input sizes (own statistics per
    # size), fused views, bf16, deterministic cuDNN. Self-training at 160 only. Defaults: sizes (160, 224),
    # flip + 2 shifts, LAME on T=2-softened probabilities, 3 Adam steps per batch.
    "mixtta_ms224": MixTTAMS,
    # Fixed BN tracking as in ms224, plus the validation-derived local class prior.
    "mixtta_local_fixed": partial(MixTTAMS, local_context=True),
    # mixtta_local_fixed + per-stage self-training learning rates x(2, 2, 2, 1, 0.25) (stem, layer1-layer4)
    "mixtta_lfs": partial(MixTTAMS, local_context=True, stage_lr=(2, 2, 2, 1, 0.25)),
    # mixtta_lfs + adaptive LAME temperature (mean max-probability 0.3)
    "mixtta_lfsq": partial(MixTTAMS, local_context=True, stage_lr=(2, 2, 2, 1, 0.25), lame_q=0.3),
    # FINAL SUBMISSION: mixtta_lfsq with per-stage rates x(4, 4, 4, 1, 0.1)
    "mixtta_lfsq4": partial(MixTTAMS, local_context=True, stage_lr=(4, 4, 4, 1, 0.1), lame_q=0.3),

    # side branch (report Q3): val-only re-pick with shift reset, running-mean tracking and in-batch split
    # after stage 5: 3 steps, LAME T 2, rate floor 0.2, val-rule threshold 9.0, in-batch split
    "mixtta_repick_s5": _repick_s5,
    # + per-stage self-training learning rates x(2, 2, 2, 1, 0.25) (stage 6)
    "mixtta_repick": partial(MixTTAMS, sizes=(160, 224), steps=3, lame_T=2.0, alpha_min=0.2, shift_thr=9.0,
                             split_shift=True, stage_lr=(2, 2, 2, 1, 0.25)),
    # side branch + the local class prior: both collapsed on the public leaderboard (kept for the report's table)
    "mixtta_local": partial(_repick_s5, local_context=True),
    "mixtta_r8": partial(MixTTAMS, sizes=(160, 224), steps=3, lame_T=2.0, alpha_min=0.2, shift_thr=9.0,
                         split_shift=True, stage_lr=(2, 2, 2, 1, 0.25), local_context=True),
}
