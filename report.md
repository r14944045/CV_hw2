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
Final model: `mixtta_ms224sp2cprl3v` (`starter/tta/mixtta_ms.py`), val 95.60 %.
- `mixtta_ms224sp2cprl3` (Kaggle public 0.93831) is the same method with a hand-set shift threshold of 10 instead of the val-chosen 8.7 (step 0). On the val stream the two give identical predictions.
- `mixtta_ms224sp2cpr` (0.93277) also uses one learning rate for all BN layers (step 5).
- `mixtta_ms224sp2` (0.92724) also lacks the in-batch split (step 0, last part) and the second pass (step 6).
- `mixtta_ms224` (0.91585) also lacks the shift detection and the class prior (steps 0 and 3).

- Backbone model: ResNet-50, the provided 160 px source checkpoint. Every BN layer is replaced by a source-mixed BN layer (w = 0.5) with one statistics tracker per input size.
- Learning rate: constant (Adam), on the BN scale and shift only. The base rate of 1e-3 is scaled per ResNet stage: 3× for the stem, layer1 and layer2, 1× for layer3 and 0.1× for layer4.

For each batch of 64:
0. Domain-shift detection. If the batch's conv1 statistics diverge from the tracked ones by more than 8.7× the domain's recent median, a new domain starts.
   - The factor 8.7 is set on val only (`tools/dev/threshold_val.py`). Over the 7 val stream orders, each with the boundaries moved to 4 positions in the batch, the largest ratio inside a domain is 8.12 and the smallest at a boundary is 9.2. The threshold is their geometric mean, and it detects every val boundary with no false alarms.
   - A domain boundary usually falls inside a batch (both val boundaries do). A changepoint fitted to the per-image conv1 statistics of the previous and current batch finds the first image of the new domain.
   - The images before it are normalised with the old domain's statistics. The tracked statistics are replaced by those of the images after it (tracking rate 1).
   - Within a domain, the statistics are a running mean over its images, with the tracking rate floored at 0.1.
1. At input sizes 160 and 224, each with its own BN statistics: update the statistics with the batch, then average the softmax over the batch, its flip and 2 random flip + shift copies.
2. Average over both sizes.
3. Class prior: divide the probabilities by (20 × the share of each class predicted so far)^2, with a pseudo-count of 2000 images. Every val domain is class-balanced, and the test stream is assumed to be balanced overall too.
4. Apply LAME (k = 5, temperature T = 2) and predict the argmax.
5. Train: soft cross-entropy between the LAME output and the model's output on a shifted copy at 160 px, 3 steps, with the per-stage learning rates above.
6. Second pass, on the first 2 batches of each domain only: train first (step 5), then predict the same views again with the updated BN scale and shift.

## Experiments

| Method (`--method`) | What it changes | Val acc (%) | Δ vs source | Val time (s) | Peak VRAM (MiB) | Kaggle public (%) |
|---|---|---:|---:|---:|---:|---:|
| **Baselines** | | | | | | |
| `source` | No adaptation (source BN statistics, frozen weights) | 54.55 | — | 0.7 | 1256 | — |
| `bn` | Test-batch BN statistics, no training | 72.16 | +17.61 | 1.1 | 1152 | 0.72037 |
| `rbn` | Robust BN only (EMA of batch statistics, α = 0.05), no training | 76.23 | +21.69 | 1.2 | 1266 | 0.70231 |
| **RoTTA** | | | | | | |
| `rotta` | Full RoTTA: Robust BN + CSTU memory + teacher–student with timeliness weights | 65.01 | +10.46 | 3.8 | 4208 | 0.65983 |
| `rotta_bn` | Robust BN → test-batch BN | 72.30 | +17.76 | 3.2 | 4078 | — |
| `rotta_nomem` | No CSTU memory: train on the current batch | 72.92 | +18.37 | 2.9 | 4156 | — |
| `rotta_notime` | No timeliness: uniform sample weights | 64.96 | +10.42 | 3.8 | 4202 | — |
| **CoTTA** | | | | | | |
| `cotta` | Full CoTTA: EMA teacher + augmentation-averaged pseudo-labels (32 aug.) + stochastic restoration, test-batch BN | 76.99 | +22.44 | 12.5 | 4510 | 0.79117 |
| `cotta_noaug` | No augmentation-averaged pseudo-labels | 72.16 | +17.61 | 3.5 | 4484 | — |
| `cotta_norst` | No stochastic restoration | 77.60 | +23.06 | 18.4 | 4390 | 0.79085 |
| `cotta_fast` | 4 augmentations instead of 32 | 73.30 | +18.75 | 5.6 | 4446 | — |
| `cotta_rbn` | Test-batch BN → Robust BN | 59.42 | +4.88 | 22.0 | 4546 | — |
| **BOSA** | | | | | | |
| `bosa` | Full BOSA as printed: memory-bank self-training + discrepancy-aware BN alignment (adaptive α, Wasserstein-2 constraint with unit step) + activation sparsification (q = 0.7) | 60.70 | +6.15 | 4.9 | 2916 | — |
| `bosa_noreg` | No adaptive update constraint (Eqs. 5–7) | 58.62 | +4.07 | 4.9 | 2862 | — |
| `bosa_noalign` | No statistics alignment: fixed α = 0.05, no constraint (= Robust BN) | 64.68 | +10.13 | 4.7 | 2886 | — |
| `bosa_nomem` | No memory bank: align and train on the current batch | 77.51 | +22.96 | 4.1 | 2760 | — |
| `bosa_nosparse` | No activation sparsification (q = 0) | 60.56 | +6.01 | 4.3 | 4204 | — |
| `bosa_ent` | Entropy minimisation instead of self-training | 59.19 | +4.64 | 5.0 | 3518 | — |
| `bosa_stable` | Constraint step halved (2 · 0.5 · α_t ≤ 1); not in the paper | 60.51 | +5.96 | 5.1 | 3006 | — |
| `bosa_stable_nomem` | `bosa_stable` without memory bank; not in the paper | 79.50 | +24.95 | 4.1 | 2906 | 0.78662 |
| **MixTTA** (ours) | | | | | | |
| `mixtta` | Source-mixed BN statistics (tracking rate α = 0.2, source weight w = 0.5), updated before predicting + softmax averaged over flip and 4 random shifts + LAME (k = 5) | 89.06 | +34.51 | 3.3 | 1366 | 0.86246 |
| `mixtta_w7` | α = 0.4, w = 0.7 | 89.49 | +34.94 | 3.3 | 1366 | — |
| `mixtta_nolame` | No LAME | 86.41 | +31.86 | 3.2 | 1366 | — |
| `mixtta_noaug` | No flip/shift averaging | 86.93 | +32.38 | 1.5 | 3366 | — |
| `mixtta_stats` | Mixed statistics only (no averaging, no LAME) | 83.19 | +28.64 | 1.4 | 2150 | — |
| `mixtta_lag` | Predict with the previous statistics, update afterwards (BOSA / RoTTA order) | 84.66 | +30.11 | 3.6 | 1366 | — |
| `mixtta_srcbn` | Source BN: frozen training statistics (α = 0) | 55.82 | +1.27 | — | — | — |
| `mixtta_tbn` | Test-batch BN: current batch only (α = 1, w = 0); the flip and shift copies use the original batch's statistics | 80.68 | +26.13 | — | — | — |
| `mixtta_rbn` | Robust BN: slow EMA, no source mix (α = 0.05, w = 0) | 82.24 | +27.69 | — | — | — |
| `mixtta_w0` | No source mix (w = 0), α = 0.2 kept | 86.51 | +31.96 | — | — | — |
| `mixtta_st_notrain` | α = 0.4, w = 0.5; LAME on features averaged over all 6 views (still forward-only) | 90.10 | +35.55 | 4.3 | 2882 | 0.88981 |
| `mixtta_st` | `mixtta_st_notrain` + self-training: 1 Adam step (lr 1e-3) per batch on BN γ/β, soft cross-entropy to the LAME output on a shifted copy | 91.15 | +36.60 | 5.6 | 3846 | — |
| `mixtta_st2` | `mixtta_st` with 2 steps per batch | 91.90 | +37.35 | 6.7 | 3798 | 0.91 |
| **MixTTA-MS** (ours) | | | | | | |
| `mixtta_ms224` | `mixtta_st2` at input sizes 160 and 224, each with its own BN statistics, averaged; flip + 2 shifts per size, LAME on T = 2-softened probabilities, 3 Adam steps per batch at 160; bf16, deterministic cuDNN | 93.99 | +39.44 | 4.8 | 4150 | 0.91585 |
| `mixtta_ms224p2` | + class prior: divide by (20 × each class's share of the predictions so far)^2, pseudo-count 2000 (step 3) | 94.08 | +39.53 | 4.6 | 4150 | 0.92366 |
| `mixtta_ms224sp2` | + domain-shift reset (conv1 statistics jump > 10× the domain's recent median) and running-mean statistics within a domain, rate floored at 0.1 (step 0) | 94.46 | +39.91 | — | 4172 | 0.92724 |
| `mixtta_ms224sp2cp` | + in-batch split of the shift batch at a changepoint; rows before it keep the old domain's statistics (step 0) | 94.84 | +40.29 | — | 4090 | 0.93180 |
| `mixtta_ms224sp2cpr` | + second pass on the first 2 batches of each domain: self-train, then predict again (step 6) | 94.93 | +40.38 | — | 4602 | 0.93277 |
| `mixtta_ms224sp2cprl` | + per-stage learning rates ×(2, 2, 2, 1, 0.25) for stem, layer1–layer4 (step 5) | 95.17 | +40.62 | — | 4624 | — |
| `mixtta_ms224sp2cprl3` | per-stage learning rates ×(3, 3, 3, 1, 0.1). Best Kaggle score | 95.60 | +41.05 | — | 4624 | **0.93831** |
| `mixtta_ms224sp2cprl3v` | shift threshold 8.7, set on val (step 0) instead of 10. **Final** | 95.60 | +41.05 | 5.4 | 4602 | — |

## Q3 Strategy

The val stream has three properties that break standard TTA, and we assume the test stream shares them:
- **Distribution shift:** corruption domains change abruptly, and both val boundaries fall inside a batch.
- **Non-i.i.d. sampling:** each val batch of 64 holds only 5–17 of the 20 classes.
- **Length:** the test stream is 192 batches (12,288 images) long, and over that many steps self-training can drift.

The final method, `mixtta_ms224sp2cprl3v`, addresses each as follows. Gains are on val unless marked Kaggle; the Kaggle gains were measured on the earlier variants in the Experiments table.

**Distribution shift**
- **Source-mixed BN.** Each BN layer normalises with 50 % tracked test statistics and 50 % source statistics: the tracked half follows the domain, and the source half is an anchor no single batch can move (against Robust BN: +4.3 pt from faster tracking, +2.6 pt from the source mix).
- **Update before predicting.** Each batch updates the statistics before it is predicted, so a new domain's first batch is not normalised with the old domain's statistics (+4.4 pt).
- **Shift detection and reset.** A jump in the conv1 statistics (KL divergence > 8.7× the domain's recent median; the factor is set on val) starts a new domain: the tracked statistics restart from it, then follow a running mean (Kaggle +0.36).
- **Splitting the boundary batch.** A changepoint on the per-image conv1 statistics finds where the new domain starts inside the batch. Images before it keep the old domain's statistics.
- **Self-training and a second pass.** After each batch, 3 Adam steps on the BN scale and shift fit the model to its own corrected predictions (Kaggle 0.88981 → 0.91). On the first 2 batches of a domain, the model trains first and then predicts the batch again (with the split: Kaggle +0.55).
- **Prediction averaging.** The softmax is averaged over flip and shift views (+3.2 pt) at input sizes 160 and 224, each size with its own BN statistics.
**Non-i.i.d. sampling**
- **No statistics from one batch alone.** A skewed batch's statistics depend on which classes it holds; averaging over the domain and the source keeps that bias small. CoTTA's current-batch statistics and RoTTA's slow EMA (about 20 batches to follow a shift) both do worse.
- **Laplacian Adjusted Maximum-likelihood Estimation(LAME, CVPR 2022).** Each prediction is corrected and moves towards those of its 5 nearest neighbours in feature space using cosine similarity. The idea is that samples with similar feature representation should also have similar prediction. In highly skewed batch, the skew becomes extra information for LAME, since an image's 5 nearest neighbours are very likely the same class. Averaging an image's prediction with its neighbours' therefore corrects it more often than it corrupts it (+2.7 pt).
- **Per-stage learning rates.** Each ResNet stage gets its own self-training learning rate:
  - the stem, layer1 and layer2 at 3× the base rate of 1e-3: their statistics are low-level and specific to the corruption, so they adapt faster to each new domain;
  - layer4 at 0.1×: its scale and shift act directly on class evidence, where fast updates would amplify pseudo-label errors.
  - Kaggle 0.93277 → 0.93831, with no extra time.

- **Remove memory bank.** In our experiments, RoTTA's and BOSA's class-balanced memory banks decrease the performance here (removing them: +7.9 and +19 pt). 

**Error accumulation and collapse**
- **Global class prior.** Every val domain is class-balanced, and we assume the test stream is too, so each probability is divided by (20 × that class's share of the predictions so far)². Under corruption the model favours some classes (on val, the source model over-predicts acoustic guitar, laptop and tree frog), and self-training on its own predictions would reinforce that bias. The prior pushes back on it (Kaggle 0.91585 → 0.92366).
- **Small, depth-aware updates.** Only the BN scale and shift are trained, on soft targets, with layer4 nearly frozen.

## Q4 Visualization

![8 validation images with predicted and ground-truth labels](figures/val_examples.png)

**Setup.**
- The figure shows the final method, `mixtta_ms224sp2cprl3v`, from its harness val run: 95.60 %, 93 of 2,112 images wrong. Its val predictions are identical to `mixtta_ms224sp2cprl3`'s.
- Green images are correct and red ones misclassified. Under each image are its position in the stream and its val domain.
- The images were picked at random with a fixed seed. Each row takes one image per val domain plus a second from the scan-line grid domain.
- `python tools/plot_val_examples.py` regenerates the figure.

**Correct (top row).** Each survives a heavy corruption:
- dense scan lines over the whole image (#526 flamingo, #636 red fox);
- a wave warp that smears the locomotive into its smoke (#1345);
- a husky squeezed inside the fisheye's black border (#1567).

**Misclassified (bottom row).**
- **#173, red fox → timber wolf.** The scan lines wash out the fox's orange coat, leaving a wolf-like canid shape in a forest.
- **#606, sports car → teapot.** A night shot: under the scan lines only the headlights and a curved, shiny body remain.
- **#896, tabby → steam locomotive.** The warp turns the cat's stripes into wavy dark streaks, like smoke and metal.
- **#1857, laptop → sports car.** The laptop is small, seen from the side in a classroom, and shrunk by the fisheye.

**Error distribution.**
- The 93 errors are spread evenly over the val domains: 34 of 776, 32 of 724 and 27 of 612 images.
- The most frequent confusions are among vehicles and the laptop (sports car → airliner, sports car ↔ laptop, sports car ↔ fire engine) and between the timber wolf and the Siberian husky (both directions).
