"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.

Components live in tta/ (one file per component, see tta/__init__.py). Variants are registered
below with functools.partial, so each ablation is just `--method <name>`.
"""
from functools import partial

from tta import BOSA, CoTTA, MixTTA, MixTTAMS, NormOnly, RoTTA, Source

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
    # MixTTA with other normalisation statistics (MixBN2d settings; views use the stored statistics)
    "mixtta_srcbn": partial(MixTTA, alpha=0.0),               # frozen source statistics
    "mixtta_tbn": partial(MixTTA, alpha=1.0, w=0.0),          # current test batch only
    "mixtta_rbn": partial(MixTTA, alpha=0.05, w=0.0),         # Robust BN (RoTTA's EMA), no source mix
    "mixtta_w0": partial(MixTTA, w=0.0),                      # mixtta's alpha=0.2 without the source mix
    # MixTTA + self-training on the LAME-corrected predictions (BN gamma/beta, Adam 1e-3)
    "mixtta_st": partial(MixTTA, alpha=0.4, w=0.5, lame_feats="mean", lr=1e-3, train_steps=1),
    "mixtta_st2": partial(MixTTA, alpha=0.4, w=0.5, lame_feats="mean", lr=1e-3, train_steps=2),
    "mixtta_st_notrain": partial(MixTTA, alpha=0.4, w=0.5, lame_feats="mean"),  # same, without step 4
    # MixTTA-MS (tta/mixtta_ms.py): MixTTA + self-training run at several input sizes (own statistics per
    # size), fused views, bf16, deterministic cuDNN. Self-training at 160 only. Defaults: sizes (160, 224),
    # flip + 2 shifts, LAME on T=2-softened probabilities, 3 Adam steps per batch.
    "mixtta_ms224": MixTTAMS,
    "mixtta_ms3": partial(MixTTAMS, sizes=(160, 192, 224), steps=2),
    "mixtta_ms3s3": partial(MixTTAMS, sizes=(160, 192, 224)),
    # + class-prior correction from the counts of predicted labels so far (tta/mixtta_ms.py, step 3b)
    "mixtta_ms224p": partial(MixTTAMS, prior_lambda=1.0),
    "mixtta_ms224p2": partial(MixTTAMS, prior_lambda=2.0),
    # + domain-shift detection (tracker reset) and running-mean tracking within a domain (steps 0, 0b)
    "mixtta_ms224s": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1),  # collapses on test (tree frog)
    "mixtta_ms224sp2": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0),
    # mixtta_ms224sp2 (Kaggle 0.92724) variants: stronger / faster prior, a third input size, a temporal label prior
    "mixtta_ms224sp3": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1, prior_lambda=3.0),
    "mixtta_ms224sp2f": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0, prior_tau=1000.0),
    "mixtta_ms3sp2": partial(MixTTAMS, sizes=(160, 192, 224), shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0),
    "mixtta_ms224sp2t": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0, temporal_mu=0.2),
    # + in-batch split of the shift batch (step 0c): rows before the changepoint keep the old domain's statistics
    "mixtta_ms224sp2cp": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0, split_shift=True),
    # + on the first 2 batches of a domain: self-train first, then predict again with the updated gamma/beta (step 5)
    "mixtta_ms224sp2cpr": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.1, prior_lambda=2.0, split_shift=True,
                                  refine_batches=2),
    # domain-shift reset only: rate 1 on the shift batch, 1/2 on the next, then the usual 0.4
    "mixtta_ms224r": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.4),
    "mixtta_ms224rp2": partial(MixTTAMS, shift_thr=10.0, alpha_min=0.4, prior_lambda=2.0),
    # ablations
    "mixtta_ms224_t1": partial(MixTTAMS, lame_T=1.0),  # mixtta_ms224 without the LAME temperature
    "mixtta_ms": partial(MixTTAMS, sizes=(160, 192), n_shift=4, lame_T=1.0, steps=2),  # first multi-scale version
    "mixtta_ms_notrain": partial(MixTTAMS, sizes=(160, 192), n_shift=4, lame_T=1.0, steps=0),
    "mixtta_bf16": partial(MixTTAMS, sizes=(160,), n_shift=4, lame_T=1.0, steps=2),  # = mixtta_st2 in bf16
}
