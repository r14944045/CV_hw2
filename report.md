## Q1 Introduce two test-time adaptation (TTA) papers of different TTA settings, for image classification (10% each).

What problem does it address? What is the method? What are its pros and cons?
It would be fine to copy the figure from the referred paper.

1. Robust Test-Time Adaptation in Dynamic Scenarios

- Problem addressed:
  1. continually changing distribution: distribution shift overtime
  2. correlative sampling: the data distribution in a batch differs from global distribution
- Method:
  1. Robust Batch Normalization (RBN): instead of using statistics from the current test batch for normalization, they maintain global mean and variance. During adaptation, these statistics are updated using an exponential moving average (EMA) of statistics computed from samples in the memory bank. Since the memory bank maintains a more balanced representation of the data, normalization becomes less sensitive to class imbalance and correlations within the current batch.
  2. Category-balanced Sampling with Timeliness and Uncertainty (CSTU): maintains a memory bank with an equal memory quota for each category. When a new sample arrives, the teacher model predicts its category and uncertainty, and a heuristic score is calculated based on its age and prediction uncertainty. If replacement is needed, RoTTA selects a candidate from either the most populated category or the incoming sample's category. The candidate is replaced only if its heuristic score is higher than that of the new sample. This favors recent, confident samples while maintaining category balance.
  3. Training with timeliness: we first calculate the cross entropy between prediction of teacher model and that of student model (which predicts on strong-sugmented samples). Then, the student model updates weights using the loss weighted by timeliness (where newer sample gets larger weight), and the teacher model updates weights using EMA from its own weight and student's weight.
- Pros:
  - Explicitly addresses both continual distribution shifts and non-IID sampling through robust batch normalization and category-balanced memory.
  - Reduces error accumulation using an EMA teacher and timeliness-weighted training.
  - Updates only BN parameters, reducing computational cost and limiting changes to pretrained knowledge.

- Cons:
  - Relies on potentially incorrect pseudo-labels for memory management and self-training.
  - Updating only BN parameters limits adaptation capacity under severe distribution shifts.
  - Slowly updated statistics may struggle with abrupt domain changes, while maintaining a memory bank introduces additional overhead.

2. Continual Test-Time Domain Adaptation

- Problem: error accumulation & catastrophic forgetting of source knowledge
- Method:
  1. Weight-and-augmentation-averaged pseudo-labels: instead of using only final weights, the teacher model uses weights averaged over training steps to generate pseudo-labels. To further reduce error accumulation, they apply augmentation when domain difference is large. Specifically, they approximate domain difference with confidence score, and apply random augmentations when confidence is low to improve the quality of pseudo-labels.

  2. Stochastic Restoration: apply a mask tensor to restore some of the trainable model weight back to the weight of original pre-trained model. This avoid the model to drift too far from the source model. Compared to RoTTA that only updates BN weights, CoTTA updates all trainable weights, bringing more capacity for adaption.

- Pros:
  - Reduces error accumulation through EMA teacher predictions and augmentation averaging.
  - Mitigates catastrophic forgetting through stochastic restoration of pretrained weights.
  - Updates all trainable parameters, providing greater adaptation capacity than BN-only methods.

- Cons:
  - Computationally expensive due to multiple augmented forward passes and full-model parameter updates.
  - Does not explicitly address non-IID or class-imbalanced sampling.
  - Incorrect pseudo-labels can still accumulate errors, while stochastic restoration may interfere with adaptation to the target domain.

## Q2 Method

Final model: `mixtta_lfsq4` (`starter/tta/mixtta_ms.py`), Kaggle public **0.94319**, val 95.93 %. Its settings were chosen on val, on ImageNet-C corruptions applied to the val images, and on the public leaderboard, as the slide-21 FAQ allows.

- **Model:** the provided ResNet-50 (160 px). Every BN layer becomes a source-mixed BN layer: it normalises with 50 % tracked test statistics and 50 % source statistics. The tracked statistics are an exponential moving average with rate 0.4, one per input size (160 and 224 px).
- **Trained parameters and loss:** only the BN scale and shift. The loss is the soft cross-entropy between the target of step 3 and the model's output on a randomly flipped and shifted copy at 160 px. Adam, 3 steps per batch, learning rate 1e-3 scaled per ResNet stage: ×4 for the stem, layer1 and layer2, ×1 for layer3, ×0.1 for layer4.

For each batch of 64:

1. **Views.** At 160 and 224 px, update the tracked statistics with the batch, then average the softmax over the batch, its flip and 2 random flip + shift copies. Averaging both sizes gives the image probabilities p.
2. **LAME with an adaptive temperature.** Soften p with the temperature T, between 1 and 3, that brings the batch's mean max-probability to 0.3, then apply LAME (k = 5 nearest neighbours in feature space). This gives z.
3. **Local class prior.** For each image, sum the LAME outputs z of the other images in this batch and the previous one, weighted by exp(−d / 12) for a distance of d stream positions, and add 0.2 per class. This gives prior counts c. The target is softmax(2 · [(2 log p + log z) / 3 + ½ log c]); its argmax is the prediction.
4. **Self-training** towards the target, as above.

**How the settings were chosen** (logs in `tools/dev/`):

1. **Val-only re-pick** (`repick.py`, rules fixed before any run). The score V is the mean accuracy over 7 val stream orders × 4 positions of the domain boundaries in the batch × 2 seeds, and a 231-batch stream (the 7 orders back to back) vetoes collapse. Stage by stage, it kept the simplest setting within 0.3 pt of the best: 3 steps and T = 2 (as in `mixtta_ms224`), shift reset with an in-batch split, and per-stage rates ×(2, 2, 2, 1, 0.25). The result, `mixtta_repick`, scored 0.92675. All 35 evaluations are in [docs/repick_all_evaluations.md](docs/repick_all_evaluations.md).
2. **Local class prior** (designed on val, [docs/local_prior_experiment.md](docs/local_prior_experiment.md)). On the re-pick branch it raised V by about 1 pt but collapsed on the public board (0.68554 and 0.82682). On fixed tracking it scored 0.92789, and 0.93815 with the re-pick's per-stage rates (`mixtta_lfs`). Fixed tracking over shift reset was therefore decided on the public leaderboard.
3. **ImageNet-C suite.** The private board uses other corruption types, so later changes were judged on the 14 ImageNet-C corruptions (Hendrycks & Dietterich, 2019; severity 3), applied to the val images with our own reimplementation. Frost was left out because it needs external photos. The published list was split by position into a tuning and a confirmation half, each with 3 streams. The rule, fixed before each round: adopt a change if it gains at least 0.3 pt on the tuning half (or the seed spread, if larger), gains on the confirmation half, costs at most 0.3 pt of V, and does not collapse on the long stream.

| Round (base) | Change | Tuning half | Confirmation half | V | Kept |
|------------------|-------------------------------------------|---------:|---------:|---------:|---|
| 10 (`mixtta_lfs`) | adaptive LAME temperature (mean max-prob. 0.3) | +0.67 | +1.76 | −0.21 | yes |
| 10 | 4 self-training steps | +0.77 | −0.51 | +0.28 | no |
| 11 (`mixtta_lfsq`) | tracking rate 0.6 / source weight 0.3 | +0.29 / +0.32 | +0.72 / +1.14 | −0.29 / −0.74 | no |
| 12 (`mixtta_lfsq`) | per-stage rates ×(4, 4, 4, 1, 0.1) | +1.57 | +0.38 | +0.31 | yes |
| 13 (`mixtta_lfsq4`) | ×(5, 5, 5, 1, 0.1) / ×(4, 4, 4, 2, 0.1) | +0.16 / −0.94 | −0.51 / −2.33 | +0.05 / +0.04 | no |

- **Seed check.** Over seeds 0–3, `mixtta_lfsq4` beats `mixtta_lfsq` on every seed, on the tuning half (mean 76.82 vs 75.07) and on V (94.58 vs 94.22). It gains +0.30 on the confirmation half, and no seed collapses.
- **Kaggle.** The public gains of `mixtta_lfsq` (+0.23) and `mixtta_lfsq4` (+0.28) are below the 0.5 pt we set in advance as noise. Both were adopted on the suite rule and the seed check, not on Kaggle.

## Q3 Strategy

The val stream has three properties that break standard TTA, and we assume the test stream shares them:

- **Distribution shift:** corruption domains change abruptly, and both val boundaries fall inside a batch. The private leaderboard uses different corruption types from the public one.
- **Non-i.i.d. sampling:** each val batch of 64 holds only 5–17 of the 20 classes, and neighbouring images often share a label.
- **Length:** the test stream is 192 batches long, and over that many steps self-training can drift.

| Strategy (`--method`) | Val (%) | Kaggle public |
|--------------------------------------------------------------------|----------------:|------------------:|
| No adaptation (`source`) | 54.55 | — |
| Test-batch BN (`bn`) / Robust BN (`rbn`) | 72.16 / 76.23 | 0.72037 / 0.70231 |
| RoTTA (`rotta`) / without its memory bank | 65.01 / 72.92 | 0.65983 / — |
| CoTTA (`cotta`) / without stochastic restoration | 76.99 / 77.60 | 0.79117 / 0.79085 |
| BOSA (`bosa`) / without its memory bank | 60.70 / 77.51 | — |
| **Ours, forward-only** (`mixtta`): source-mixed BN updated before predicting, flip/shift averaging, LAME | 89.06 | 0.86246 |
| – without LAME / without view averaging / predicting before updating | 86.41 / 86.93 / 84.66 | — |
| – source statistics only / current batch only / Robust BN / no source mix | 55.82 / 80.68 / 82.24 / 86.51 | — |
| + faster tracking (rate 0.4), LAME on view-averaged features (`mixtta_st_notrain`) | 90.10 | 0.88981 |
| + self-training, 2 steps per batch (`mixtta_st2`) | 91.90 | 0.91 |
| + input sizes 160 and 224, T = 2, 3 steps (`mixtta_ms224`) | 93.99 | 0.91585 |
| + local class prior (`mixtta_local_fixed`) | 95.12 | 0.92789 |
| + per-stage learning rates ×(2, 2, 2, 1, 0.25) (`mixtta_lfs`) | 95.31 | 0.93815 |
| + adaptive LAME temperature (`mixtta_lfsq`) | 95.27 | 0.94042 |
| + per-stage learning rates ×(4, 4, 4, 1, 0.1) (`mixtta_lfsq4`, **final**) | 95.93 | **0.94319** |
| Side branch of `mixtta_ms224`: shift reset + in-batch split (`mixtta_repick_s5`) / + rates ×(2, 2, 2, 1, 0.25) (`mixtta_repick`) | 94.74 / 95.08 | 0.92220 / 0.92675 |
| Side branch + local class prior (`mixtta_local` / `mixtta_r8`), collapsed | 96.07 / 96.69 | 0.68554 / 0.82682 |

Every ablation and variant is listed in [docs/experiments_full.md](docs/experiments_full.md).

**Distribution shift**

- **Source-mixed BN.** The tracked half of the statistics follows the domain, and the source half is an anchor no single batch can move. Against Robust BN this gives +4.3 pt from faster tracking and +2.6 pt from the source mix. On the ImageNet-C suite, slower tracking (0.2) or a heavier source weight (0.7) loses 6–18 pt; faster tracking (0.6) or a lighter source weight (0.3) helps the suite but costs val, so 0.4 and 0.5 stayed.
- **Update before predicting.** Each batch updates the statistics before it is predicted, so a new domain's first batch is not normalised with the old domain's statistics (+4.4 pt).
- **Adaptive LAME temperature.** With a fixed T = 2, a confident domain is softened too little and an uncertain one too much, so LAME's neighbour term carries a different weight on each domain. Choosing T per batch so that the mean max-probability is 0.3 equalises it. It gains most on the unseen half of the suite (+1.76 pt).
- **Prediction averaging.** Flip and shift views add +3.2 pt. The second input size, 224 px, adds +2.1 pt on val and +0.6 on Kaggle.

**Non-i.i.d. sampling**

- **No statistics from one batch alone.** A skewed batch's statistics depend on which classes it holds; averaging over the stream and the source keeps that bias small. CoTTA's current-batch statistics and RoTTA's slow EMA (about 20 batches to follow a shift) both do worse.
- **Laplacian Adjusted Maximum-likelihood Estimation (LAME, CVPR 2022).** Each prediction is corrected and moves towards those of its 5 nearest neighbours in feature space using cosine similarity. The idea is that samples with similar feature representation should also have similar prediction. In a highly skewed batch, the skew becomes extra information for LAME, since an image's 5 nearest neighbours are very likely the same class. Averaging an image's prediction with its neighbours' therefore corrects it more often than it corrupts it (+2.7 pt).
- **Local class prior.** Neighbours in the stream often share a label: 24.8 % of adjacent val pairs, against 5.5 % at a distance of 64. The prior estimated from the nearby LAME outputs turns this correlation into evidence, at no extra forward pass. On `mixtta_ms224` it adds +1.1 pt on val and +1.2 on Kaggle.
- **Remove memory bank.** In our experiments, RoTTA's and BOSA's class-balanced memory banks decrease the performance here (removing them: +7.9 and +16.8 pt).

**Error accumulation and collapse**

- **Small, soft, depth-aware updates.** Only the BN scale and shift are trained, towards soft targets. The rate is 4× on the low-level stages, which adapt to each corruption, and 0.1× on layer4, which acts directly on class evidence. Per-stage rates added +1.0 on Kaggle on top of the local prior (0.92789 → 0.93815). A higher rate on layer3 hurts on unseen corruptions (−2.33 pt on the confirmation half).
- **Feedback loops need fixed tracking.** The local prior feeds the predictions back into the next batch's target. Combined with shift reset, the in-batch split and running-mean tracking, it collapsed on the public board (0.68554 and 0.82682), while every val-based check rated it +1 to +2 pt better. Those checks reuse val's 33-batch label order and cannot show slow, self-reinforcing drift over the 192 batches of the test stream. With fixed tracking it stayed stable: four public scores from 0.92789 to 0.94319, and no collapse on any seed.

**Tried and not kept**

- **From our own method:**
  - shift detection with reset and an in-batch split (alone: +0.40 and +0.39 pt on V, Kaggle 0.91585 → 0.92220; unsafe with the local prior);
  - a second pass on a domain's first batches and a global class prior (within the re-pick margin);
  - 4 self-training steps (−0.51 on the confirmation half);
  - T3A's prototype classifier (+0.48 on V, but −1.62 / −0.56 on the suite).
- **Published components on the suite, with their official defaults:**
  - EATA's sample filtering, and ROID's certainty and diversity weights, weight ensembling and prior correction (all worse);
  - DELTA's reweighting (+0.18, below the margin);
  - SAR's model recovery (never triggered);
  - NOTE's instance-aware BN (−10 pt);
  - TRIBE's balanced BN and UnMix-TNS (89 % and 17 % on the first 20 val batches, against 95 %);
  - ImageNet-C augmentation of the self-training copy (worse);
  - restoring the source weights at each detected shift (Niloy et al., WACV 2024; −5 pt per suite domain).


## Q4 Visualization

![8 validation images with predicted and ground-truth labels](figures/val_examples.png){width=100%}

**Setup.** The figure shows `mixtta_lfsq4` from its harness val run (95.93 %, 86 of 2,112 images wrong). Green images are correct and red ones misclassified; under each image are its stream position and val domain. The images were picked at random with a fixed seed: each row has one image per val domain plus a second from the scan-line grid domain.

**Correct (top row).** Each survives a heavy corruption: dense scan lines over a small strawberry (#523) and over a red fox camouflaged against grey rocks (#633), a wave warp that folds a fire engine's front into a symmetric mask (#913), and a husky squeezed inside the fisheye's black border (#1567).

**Misclassified (bottom row).**

- **#612, laptop → grand piano.** The scan lines cut the open laptop into dark slabs and bright bands, which look like a piano's raised lid and keys.
- **#626, sports car → flamingo.** A dark car seen from behind at night; the only strong colour left is the pink-red of its tail lights and their reflection on the road.
- **#828, golden retriever → red fox.** The warp stretches the dog's head into a narrow, pointed shape, and its orange coat against a forest matches a fox.
- **#1990, steam locomotive → mountain bike.** The fisheye shrinks the train to a small object on a track through fields, seen from above, a scene closer to a bike trail.

**Error distribution.** The errors are spread over the val domains: 30 of 776 (scan-line grid), 31 of 724 (wavy warp) and 25 of 612 (fisheye). Against `mixtta_repick` (104 errors), 38 errors were fixed and 20 introduced, most of the gain on the scan-line grid (43 → 30). The most-missed classes are sports car (16) and timber wolf (15). The most frequent confusions are timber wolf ↔ Siberian husky (8), laptop ↔ sports car (7), and timber wolf → red fox and tabby → timber wolf (3 each).
