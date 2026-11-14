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
| `bosa_stable_nomem` | `bosa_stable` without memory bank; not in the paper | 79.50 | +24.95 | 4.1 | 2906 | — |

Notes:

- The val stream is label-correlated: each batch of 64 contains 5–17 of the 20 classes (a shuffled stream would contain about 19).
- RoTTA with `lr=0` (BN statistics and memory only, no gradient step) scores 64.96 %, within one image of full `rotta` (65.01). On val, the gap between `rotta` and `rbn` therefore comes from RoTTA's statistics path (statistics updated from the memory bank after each batch is predicted), not from the gradient updates.
- Val gives only 33 updates. With EMA keep rates of 0.999, both teachers stay about 97 % source weights, and `cotta_noaug` matches `bn` exactly (72.16). Rankings may differ on the longer test stream.

BOSA notes:

- BOSA (Bai et al., IJCAI 2026) has no official code. The implementation follows the paper's Secs. 4.2–4.4 and its ImageNet-C / ResNet-50 settings: Adam with lr 1e-3, memory capacity 64, q = 0.7, teacher keep rate 0.99. The supplementary material was not available, so these choices are assumptions:
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
