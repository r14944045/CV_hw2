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
- Pros & Cons:
  - Pros:
  - Cons: only update BN weights, offers limited adaptation.

2. Continual Test-Time Domain Adaptation

- Problem: error accumulation & catastrophic forgetting of source knowledge
- Method:
  1. Weight-and-augmentation-averaged pseudo-labels: instead of using only final weights, the teacher model uses weights averaged over training steps to generate pseudo-labels. To further reduce error accumulation, they apply augmentation when domain difference is large. Specifically, they approximate domain difference with confidence score, and apply random augmentations when confidence is low to improve the quality of pseudo-labels.

  2. Stochastic Restoration: apply a mask tensor to restore some of the trainable model weight back to the weight of original pre-trained model. This avoid the model to drift too far from the source model. Compared to RoTTA that only updates BN weights, CoTTA updates all trainable weights, bringing more capacity for adaption.

- Pros & Cons:
  - Pros:
  - Cons:

## Experiments

All runs go through the official harness (`python run.py --method <name> --split val ...`) on the validation stream: 2,112 images in 33 batches of 64, seed 0, one run per method, RTX 5090. One image is 0.05 pt. Two runs of `rotta` with the same seed differed by one image (GPU nondeterminism). Variance across seeds was not measured, so small gaps should not be read as real differences. Times are for the val stream; the 90 s limit applies to the test stream (192 batches) on an RTX 3090, so they are not comparable. Kaggle public is the accuracy of that method's test-stream predictions on the public half of the test set (— = not submitted).

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
| `mixtta_st_notrain` | α = 0.4, w = 0.5; LAME on features averaged over all 6 views (still forward-only) | 90.10 | +35.55 | 4.3 | 2882 | 0.88981 |
| `mixtta_st` | `mixtta_st_notrain` + self-training: 1 Adam step (lr 1e-3) per batch on BN γ/β, soft cross-entropy to the LAME output on a shifted copy | 91.15 | +36.60 | 5.6 | 3846 | — |
| `mixtta_st2` | `mixtta_st` with 2 steps per batch | 91.90 | +37.35 | 6.7 | 3798 | 0.91 |

Notes:

- The val stream is label-correlated: each batch of 64 contains 5–17 of the 20 classes (a shuffled stream would contain about 19).
- RoTTA with `lr=0` (BN statistics and memory only, no gradient step) scores 64.96 %, within one image of full `rotta` (65.01). On val, the gap between `rotta` and `rbn` therefore comes from RoTTA's statistics path (statistics updated from the memory bank after each batch is predicted), not from the gradient updates.
- Val gives only 33 updates. With EMA keep rates of 0.999, both teachers stay about 97 % source weights, and `cotta_noaug` matches `bn` exactly (72.16). Rankings may differ on the longer test stream.

BOSA notes:

- The implementation follows the paper's Secs (Bai et al., IJCAI 2026). 4.2–4.4 and its ImageNet-C / ResNet-50 settings: Adam with lr 1e-3, memory capacity 64, q = 0.7, teacher keep rate 0.99. The supplementary material was not available, so these choices are assumptions:
  - the memory bank is RoTTA's CSTU;
  - the paper's momentum v_m is read as the teacher's keep rate;
  - teacher and student each align their own BN statistics on the memory bank.

  As a check, `bosa` with its additions switched off (fixed α = 0.05, no constraint, q = 0, keep 0.999) matched a `rotta_notime` run prediction for prediction. Reruns of either differ by 4–6 predictions from GPU nondeterminism.
- `bosa` and `bosa_nomem` collapse on the test stream, so their val accuracies are not meaningful. For α_t > 0.5, the constraint step 2α_t (Eqs. 6–7 with a unit step) overshoots past the source. At α_t = 1 it reflects the statistics through the source (μ = 2μ_s − μ̂). The next discrepancy then stays large, which keeps α_t near 1.
  - In `bosa` on test, α_t ≈ 1 in all 53 BN layers for 77 of the 192 batches, including batches 1–20.
  - Their test predictions agree with `cotta`'s on 16 % and 6 % of images; `rotta` agrees on 54 % and `bn` on 75 %.
  - On val, α_t exceeds 0.5 in only 2–3 layers (median about 0.06), and the lock never spreads to all layers.
- `bosa_stable` halves the step, so the constraint becomes a weighted average with the source statistics and cannot overshoot. The lock-in disappears for every scale tested (0.05, 0.1, 0.25 and 0.5). The median α_t on the test stream is then about 0.03, and no batch has all layers near 1. Some layer still exceeds 0.5 in 10–16 of the 192 batches. Agreement with `cotta` is 56 % for `bosa_stable` and 70 % for `bosa_stable_nomem`. Agreement only detects collapse; it does not estimate accuracy.
- As with RoTTA, the memory bank hurts on this data. Removing it gives the largest gain, the opposite of the paper's Table 2, where it matters most. `bosa_stable_nomem` with `lr=0` scores 79.36 %, so its gain comes from the statistics path, not from self-training. That path is the adaptive α on the current batch plus the halved pull to the source; with the halved step the statistics settle near ((1 − α)μ̂ + μ_s)/(2 − α), about half source and half batch.
- Sparsification leaves val accuracy unchanged within noise (60.70 vs 60.56). The nvidia-smi peak is noisy: 2760–4124 MiB across runs that all use sparsification. A standalone measurement of one 64-image update is more reliable: the autograd cache falls from 2747 to 1782 MiB (−35 %) and the peak from 2783 to 1742 MiB. The paper reports over 60 % because it counts only BN activations. PyTorch still saves each ReLU output, which the following frozen conv also needs.
- The same configuration (`bosa_stable`) scored 60.98 and 60.51 in two runs, so gaps under about 1 pt between BOSA variants are noise. On the test stream, the four variants run took 19–25 s on the RTX 5090 (not checked on a 3090).

MixTTA notes:

- Motivation. On this data, the gains of the best methods come from forward passes, not from gradient updates:
  - `bosa_stable_nomem` scores the same with `lr=0` (79.36 %); its gain is the mix of source and test statistics.
  - `cotta_noaug` equals `bn`; CoTTA's gain is the augmentation averaging.

  MixTTA combines forward-only parts (steps 1–3); only the `_st` variants add gradient updates (step 4). For each batch:
  1. Statistics. Every BN layer tracks the test statistics with an EMA (rate α) and normalises with (1 − w) · tracked + w · source statistics. Unlike BOSA, the tracking rate and the pull towards the source are separate settings. The current batch updates the statistics before it is predicted.
  2. Averaging. The softmax is averaged over the batch, its horizontal flip and 4 random flip + shift copies (up to 1/8 of the image, reflection padding). The copies are normalised with the stored statistics and do not update them.
  3. LAME (Boudiaf et al., CVPR 2022). The averaged probabilities are corrected with a kNN graph (k = 5) on the batch's penultimate features (`mixtta_st*`: averaged over all views). This exploits the 5–17 classes present in each batch.
  4. Self-training (`mixtta_st`, `mixtta_st2` only). After predicting, 1 or 2 Adam steps (lr 1e-3) on the BN scale and shift. The loss is the soft cross-entropy between the LAME-corrected probabilities and the model's output on a random flip + shift copy. Training forwards do not update the statistics.
- How the settings were chosen. α, w, the augmentation type and k were picked by a grid on val. That grid ran outside the harness, with the same images fed in the same batches of 64. It reproduced the harness's val accuracy exactly for `bn`, `rbn`, `cotta` and `bosa_stable_nomem` at `lr=0`. Each setting ran on 7 stream orders: original, reversed, and the 3 val domains reordered. The 7 orders test robustness to order, not to new domains, so the settings may fit val's 3 domains.

  | Method | Mean over 7 orders (%) | Worst order (%) | Scan-line grid / wavy warp / fisheye on black (%) |
  |---|---:|---:|---|
  | `cotta` | 76.24 | 74.01 | 82.2 / 77.1 / 67.6 |
  | `bosa_stable_nomem` | 79.40 | 77.04 | 84.3 / 76.4 / 76.8 |
  | `mixtta_stats` | 83.25 | 82.01 | 86.0 / 85.1 / 77.6 |
  | `mixtta` | 88.69 | 87.83 | 91.7 / 91.1 / 82.0 |
  | `mixtta_w7` | 89.53 | 88.45 | 93.4 / 90.8 / 83.1 |
  | `mixtta_st_notrain` | 89.81 | 88.78 | — |
  | `mixtta_st` | 90.35 | 89.16 | — |
  | `mixtta_st2` | 90.89 | 89.68 | — |
- A labelled proxy for test. Val has 3 domains and test has 12, so I applied synthetic versions of the 11 other test corruptions to 768-image chunks of the val stream. The corruptions were blended 50/50 with the original image, and the chunks keep the val label order. It ranks the 6 methods already scored on Kaggle in almost the same order as Kaggle (Spearman 0.94; only `bn` and `bosa_stable_nomem` swap). It is harder than test (`mixtta`: 74.2 % on the proxy, 86.2 % on Kaggle). It preferred faster tracking (α = 0.4) and view-averaged LAME features (`mixtta_st_notrain`, proxy 78.2 %). Kaggle confirmed the gain: 0.86246 → 0.88981.
- Self-training carries over to test. `mixtta_st2` scores 0.91 on Kaggle, against 0.88981 for `mixtta_st_notrain`; on val the gains are 1.05 pt (`mixtta_st`) and 1.80 pt (`mixtta_st2`). The proxy could not have predicted this. It reuses each val image about 4 times, and all of its training gain is on repeated images. On an image's first appearance, training is neutral to slightly negative (−0.3 and −0.9 pt). Neither collapses on the test stream (no domain's most-predicted class exceeds 9.2 %). Training all parameters instead (SGD, lr 1e-3) scored 90.31 % on val but collapsed to 33 % on the proxy.
- Updating the statistics before predicting is worth 4.4 pt (`mixtta_lag` 84.66 → 89.06). With the opposite order, the first batch after each domain shift is normalised with the previous domain's statistics.
- Strong CoTTA-style augmentations hurt here. Normalised with the stored statistics, they drop the scan-line domain from 86 % to 39–56 %. Normalised with each copy's own statistics, they are still worse than flip alone (83.9 vs 84.6 % mean over 7 orders). Mild flips and shifts help.
- Label skew. Restricting each batch's predictions to the classes truly present (an oracle) gives 91.0 % mean over 7 orders, against 86.0 % without LAME. LAME with k = 5 reaches 88.7 %. It is sensitive to the graph strength: k = 3 gives 87.7 % and k = 10 gives 85.5 %, no gain. Re-estimating each batch's class prior with EM (3 iterations) gives 88.2 %.
- Test-stream checks (no labels).
  - Time on the RTX 5090: 20–21 s for `mixtta` and `mixtta_st_notrain`, 27 s for `mixtta_st`, 33 s for `mixtta_st2`. `cotta` takes 68 s and may exceed the 90 s limit on a 3090.
  - No collapse: I split the 12 test domains by image statistics. In every domain, the most-predicted class has at most 10.6 % of predictions (the source model reaches 100 % on halftone).
  - Agreement: `mixtta` and `mixtta_w7` agree with `cotta` on 75–76 % of test images and with `bosa_stable_nomem` on 79–80 %, more than those two agree with each other (70 %). This rules out collapse but does not estimate accuracy.
  - LAME changes 11.1 % of test predictions and 8.2 % of val predictions. On val, the predictions it changes go from 22 % correct to 55 % correct.
- Noise. The random shifts move the same configuration between 89.30 % and 89.06 % on val under different random states. The 0.4 pt gap between `mixtta` and `mixtta_w7` is therefore not a real difference. The nvidia-smi peak is noisy: 1366–3426 MiB for the same configuration. The forward-only variants reproduce their test predictions exactly. `mixtta_st2` does not: three test reruns each differed from the submitted file on 9–12 of 12,288 predictions, because the gradient steps amplify GPU nondeterminism.

Why MixTTA outperforms RoTTA and CoTTA:

MixTTA is built around three properties of this stream:
- Short domains: 12 abrupt domain shifts of about 16 batches each.
- Skewed batches: each batch of 64 holds only 5–17 of the 20 classes.
- Training adds nothing here: removing the weight updates leaves RoTTA, BOSA (`lr=0`) and CoTTA (`cotta_noaug` = `bn`) at the same accuracy.

It handles all three with forward passes only.

| | RoTTA | CoTTA | MixTTA |
|---|---|---|---|
| BN statistics | Slow EMA (α = 0.05), updated from the memory bank | Current batch only | Faster EMA (α = 0.2) mixed 50/50 with source statistics |
| When statistics update | After predicting | Before predicting | Before predicting |
| Prediction averaging | None | 32 strong augmentations, only when the source model is unsure | Flip + 4 small shifts, always |
| Uses label skew at prediction | No (the memory bank uses it for training only) | No | Yes (LAME) |
| Weight updates | BN affine parameters | All parameters | None |
| Test time, RTX 5090 (s) | 15 | 68 | 21 |
| Kaggle public | 0.65983 | 0.79117 | 0.86246 (`mixtta_st_notrain`: 0.88981; `mixtta_st2`: 0.91) |

1. BN statistics (val 83.19 % with this step alone, against 72.16 % for `bn` and 76.23 % for `rbn`).
   - CoTTA normalises with the current batch. With about 10 classes per batch, those statistics are noisy and depend on which classes happen to be present.
   - RoTTA's EMA rate of 0.05 needs about 20 batches to follow a shift, longer than a test domain lasts; this is why `rbn` scores below `bn` on Kaggle.
   - MixTTA tracks about 4× faster but keeps half the source statistics, so one skewed batch cannot pull them far.
   - It also updates before predicting. In the RoTTA order, the first batch after each shift is normalised with the previous domain's statistics, which costs 4.4 pt on val (`mixtta_lag`).
2. Mild averaging (+3.2 pt, to 86.41 %). CoTTA's strong augmentations change image statistics and hurt on this data. Flips and small shifts keep the gain at 6 forward passes per batch instead of CoTTA's 33.
3. LAME (+2.7 pt, to 89.06 %). Within a batch, similar-looking images should share a label. LAME pushes predictions towards the few classes the batch actually contains.
4. No training in the submitted versions. The weight updates of RoTTA, BOSA and CoTTA brought no gain on this stream, only drift and time. Without them, MixTTA is fast and stable across the 12 domain shifts. Self-training on MixTTA's own LAME-corrected predictions does help. These targets are much more accurate than RoTTA's or CoTTA's pseudo-labels. `mixtta_st2` gains 1.80 pt on val and reaches 0.91 on Kaggle.
5. Faster tracking and view-averaged LAME features (`mixtta_st_notrain`: +1.0 pt on val, +4.0 pt on the proxy, Kaggle 0.86246 → 0.88981).

Compared with BOSA, MixTTA keeps the mix of source and test statistics but separates the tracking rate from the pull towards the source. In BOSA both are set by one adaptive α_t, and that coupling caused the collapse at α_t ≈ 1.

These choices fit this stream. On a long, single-domain, class-balanced stream, LAME and the source mix would help less, and self-training might matter more.
