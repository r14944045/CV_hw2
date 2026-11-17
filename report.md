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

## Q2 Method
Best model: `mixtta_ms224sp2cprl3` (`starter/tta/mixtta_ms.py`), Kaggle public 0.93831, val 95.60 %.
- The previous best, `mixtta_ms224sp2cpr` (0.93277), is the same method with one learning rate for all BN layers (step 5).
- `mixtta_ms224sp2` (0.92724) also lacks the in-batch split (step 0, last part) and the second pass (step 6).
- `mixtta_ms224` (0.91585) also lacks the shift detection and the class prior (steps 0 and 3).

- Backbone model: ResNet-50, the provided 160 px source checkpoint. Every BN layer is replaced by a source-mixed BN layer (w = 0.5) with one statistics tracker per input size.
- Learning rate: constant (Adam), on the BN scale and shift only. The base rate of 1e-3 is scaled per ResNet stage: 3× for the stem, layer1 and layer2, 1× for layer3 and 0.1× for layer4.

For each batch of 64:
0. Domain-shift detection. If the batch's conv1 statistics diverge from the tracked ones by more than 10× the domain's recent median, a new domain starts.
   - Most domain boundaries fall inside a batch. A changepoint fitted to the per-image conv1 statistics of the previous and current batch finds the first image of the new domain.
   - The images before it are normalised with the old domain's statistics. The tracked statistics are replaced by those of the images after it (tracking rate 1).
   - Within a domain, the statistics are a running mean over its images, with the tracking rate floored at 0.1.
1. At input sizes 160 and 224, each with its own BN statistics: update the statistics with the batch, then average the softmax over the batch, its flip and 2 random flip + shift copies.
2. Average over both sizes.
3. Class prior: divide the probabilities by (20 × the share of each class predicted so far)^2, with a pseudo-count of 2000 images. The test stream is class-balanced overall.
4. Apply LAME (k = 5, temperature T = 2) and predict the argmax.
5. Train: soft cross-entropy between the LAME output and the model's output on a shifted copy at 160 px, 3 steps, with the per-stage learning rates above.
6. Second pass, on the first 2 batches of each domain only (18 of the 192 test batches): train first (step 5), then predict the same views again with the updated BN scale and shift.

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
| `mixtta_ms224sp2cprl3` | per-stage learning rates ×(3, 3, 3, 1, 0.1). **Best** | 95.60 | +41.05 | — | 4624 | 0.93831 |

Notes:

- The val stream is label-correlated: each batch of 64 contains 5–17 of the 20 classes (a shuffled stream would contain about 19).
- Val times for `mixtta_ms224sp2`, `mixtta_ms224sp2cp` and `mixtta_ms224sp2cpr` are not reported. Their logged harness runs (13.9, 8.8 and 9.7 s) are slower than `mixtta_ms224`'s 4.8 s, but on the test stream they are no slower (24.6–26.2 s), so the GPU was busy during those val runs. Test-stream times are in the MixTTA-MS notes below.
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
  3. LAME (Boudiaf et al., CVPR 2022). Each image's averaged probabilities move towards those of its k = 5 nearest neighbours in the batch (by penultimate features; `mixtta_st*`: averaged over all views). This favours the batch's 5–17 classes.
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
- A labelled proxy for test (`tools/dev/proxy.py`).
  - Val has 3 domains and test has 12, so the proxy applies synthetic versions of the 12 test corruptions to the val images, blended 50/50 with the original.
  - It has 4 streams. Each is the full val stream in val label order, with a different corruption on each of its 3 segments.
  - It ranks the 5 Kaggle-scored MixTTA methods in the Kaggle order: `bn` 62.7 %, `mixtta` 73.5 %, `mixtta_st_notrain` 77.6 %, `mixtta_st2` 79.1 %, `mixtta_ms224` 80.1 %. It is harder than test.
  - It preferred faster tracking (α = 0.4) and view-averaged LAME features, and Kaggle confirmed the gain (0.86246 → 0.88981).
  - It overstated the gains of the later MixTTA-MS changes about 4×.
- Self-training carries over to test. `mixtta_st2` scores 0.91 on Kaggle, against 0.88981 for `mixtta_st_notrain`; on val the gains are 1.05 pt (`mixtta_st`) and 1.80 pt (`mixtta_st2`). Neither collapses on the test stream (no domain's most-predicted class exceeds 9.2 %). Training all parameters instead (SGD, lr 1e-3) scored 90.31 % on val but collapsed to 33 % on the proxy.
- Normalisation (`mixtta_srcbn`, `mixtta_tbn`, `mixtta_rbn`, `mixtta_w0`: `mixtta` with only the BN statistics changed). Robust BN → source-mixed BN gains 6.8 pt: +4.3 from faster tracking (α 0.05 → 0.2) and +2.6 from the source mix (w 0 → 0.5). With frozen source statistics, averaging and LAME add only 1.3 pt. These runs have the same compute as `mixtta`; their time and VRAM are not reported because the GPU was shared with other jobs.
- Updating the statistics before predicting is worth 4.4 pt (`mixtta_lag` 84.66 → 89.06). With the opposite order, the first batch after each domain shift is normalised with the previous domain's statistics.
- Strong CoTTA-style augmentations hurt here. Normalised with the stored statistics, they drop the scan-line domain from 86 % to 39–56 %. Normalised with each copy's own statistics, they are still worse than flip alone (83.9 vs 84.6 % mean over 7 orders). Mild flips and shifts help.
- Label skew (mean over 7 orders). Without LAME: 86.0 %. LAME, k = 5: 88.7 %. An oracle that keeps only the classes truly present in each batch: 91.0 %. LAME is sensitive to k: k = 3 gives 87.7 %, k = 10 gives 85.5 % (no gain over 86.0 %). Re-estimating each batch's class prior with EM (3 iterations) gives 88.2 %.
- Test-stream checks (no labels).
  - Time on the RTX 5090: 20–21 s for `mixtta` and `mixtta_st_notrain`, 27 s for `mixtta_st`, 33 s for `mixtta_st2`. `cotta` takes 68 s and may exceed the 90 s limit on a 3090.
  - No collapse: I split the 12 test domains by image statistics. In every domain, the most-predicted class has at most 10.6 % of predictions (the source model reaches 100 % on halftone).
  - Agreement: `mixtta` and `mixtta_w7` agree with `cotta` on 75–76 % of test images and with `bosa_stable_nomem` on 79–80 %, more than those two agree with each other (70 %). This rules out collapse but does not estimate accuracy.
  - LAME changes 11.1 % of test predictions and 8.2 % of val predictions. On val, the predictions it changes go from 22 % correct to 55 % correct.
- Noise. The random shifts move the same configuration between 89.30 % and 89.06 % on val under different random states. The 0.4 pt gap between `mixtta` and `mixtta_w7` is therefore not a real difference. The nvidia-smi peak is noisy: 1366–3426 MiB for the same configuration. The forward-only variants reproduce their test predictions exactly. `mixtta_st2` does not: three test reruns each differed from the submitted file on 9–12 of 12,288 predictions, because the gradient steps amplify GPU nondeterminism.

MixTTA-MS: domain-shift reset and class prior (`mixtta_ms224sp2`: Kaggle public 0.92724, +1.14 pt over `mixtta_ms224`; `mixtta_ms224rp2` not scored):

- Stream structure.
  - Dynamic-programming segmentation on early-layer statistics splits test into 12 domains of 676–1252 images: low-light noise, wave warp, chromatic blur, thermal colormap, sepia with scratches, pixelation, diagonal stripes, screen grid, teal posterize, two halftones and flat posterize. 7 of the 11 boundaries fall inside a batch.
  - With the same segmentation, every val domain is exactly class-balanced (38–39, 36–37 and 30–31 images per class).
  - Within a domain, the label order follows RoTTA-style Dirichlet slots. Each class's quota comes in a few bursts, so consecutive batches share classes (lag-1 correlation of batch histograms +0.29) and batches 5 apart avoid them (−0.15).
- Where the errors are.
  - If test domains are balanced too, `mixtta_ms224`'s predicted class counts imply at least 5.9 % error per domain, against about 8.4 % total error. Per domain the floor is: thermal 13.8 %, pixelation 9.8 %, noise 9.5 %, the other domains 2–7 %.
  - Tabby is under-predicted in every domain (0.3–0.9× its share).
  - Offline per-domain balancing (Sinkhorn on the LAME output, with oracle boundaries) gains only 0.5 pt on val, where the floors are small.
- Shift reset (step 0).
  - The detector compares the KL divergence of the batch's conv1 statistics from the tracked ones with its median over the domain's last 8 batches. Above 10×, the batch's statistics replace the tracked ones.
  - It fires at all 11 test boundaries plus once inside a domain, and at all val and proxy boundaries. At two test boundaries (batches 35 and 118, where only a few images of the batch belong to the new domain) it fires one batch late.
  - The threshold of 10 was chosen while looking at the unlabelled test stream's boundaries, as well as val and the proxy.
  - Proxy: +2.1 pt alone. With running-mean statistics inside the domain (rate 1/k, floor 0.1): +3.4 pt.
  - Resetting the BN affine parameters at a shift as well is worse: +0.8 pt.
- Self-training drift.
  - Faster adaptation exposes a collapse attractor on the test stream: after the thermal domain, the model drifts to predicting tree frog.
  - `mixtta_ms224s` (shift reset + running mean) ends at 100 % tree frog in the last domain. `mixtta_ms224r` (shift reset only) reaches +1120 tree-frog predictions.
  - `mixtta_ms224` itself drifts for 1 of 4 seeds (+1039).
  - The short proxy (33 batches) and the long proxy (132 batches, repeated images) did not predict which variants drift.
- The class prior as a stabiliser (step 3, λ = 2).
  - The pseudo-labels then stay balanced, and so does self-training.
  - With the prior, both shift variants are stable for 4 seeds. The raw ensemble argmax before the prior also stays balanced (tree frog at 4.3–5.4 % in the late domains).
  - Results: proxy 85.24 / 85.29 % (`mixtta_ms224`: 80.13), long proxy 89.11 / 89.58 % (84.62), val7 93.51 / 93.88 % (93.34).
- Tried and dropped on the proxy:
  - a blurred-input branch with its own statistics (mixed per domain, +50 % compute);
  - training on every batch before predicting it (+1.2 pt for +70 % time). The final method does this only on the first 2 batches of each domain (step 6);
  - 5 self-training steps or lr 2e-3 (worse);
  - a per-domain prior at λ = 2 (−2.3 pt).
- Kaggle decomposition:
  - the class prior alone (`mixtta_ms224p2`): 0.91585 → 0.92366 (+0.78);
  - the shift reset with running-mean statistics on top (`mixtta_ms224sp2`): 0.92724 (+0.36, within seed noise).
  - The proxy had predicted +3.6 and +1.5 pt, so it overstates gains by roughly 4×.
- Balancing per domain, after the fact and online.
  - After the fact, with the domain boundaries known and the whole domain balanced at once (Sinkhorn on `mixtta_ms224sp2`'s LAME outputs), val rises from 94.46 to 95.27 % and 3.2 % of the test predictions change.
  - Online, balancing each batch against the domain seen so far drops val to 79.8–93.7 %. A partial domain is far from class-balanced because of the Dirichlet bursts. A per-domain model bias cannot be separated from that skew until most of the domain has passed, so this gain is not available online.
- Other ideas that did not help, on top of `mixtta_ms224sp2`:
  - A prior from the previous batch's predicted classes. Consecutive batches share classes, but this gains at most +0.3 pt on val in simulation, and stronger settings collapse.
  - Training more parameters: the fc layer, layer4 or layer3 + layer4 at lr 1e-5 is neutral, and layer4 at lr 1e-4 collapses (proxy 67.9 %).
  - A third input size, 192: val7 93.81 %, proxy 85.00 %.
  - Re-tuning on the `mixtta_ms224sp2` base:
    - source weight w = 0.3 / 0.7: val7 93.10 / 94.12 %, proxy 85.49 / 82.62 %;
    - lr 2e-3 / 5e-4: 94.09 / 93.34 %, 84.61 / 84.40 %;
    - LAME T = 1.5 / 3: 94.07 / 92.95 %, 85.23 / 80.68 %;
    - 2 self-training steps: 93.63 %, 84.85 %;
    - LAME with 10 neighbours collapses (val7 74.1 %).
    None is better on both measures (base: 93.88 %, 85.29 %).

MixTTA-MS: splitting the shift batch (`mixtta_ms224sp2cp`: Kaggle public 0.93180; `mixtta_ms224sp2cpr`: 0.93277, +0.55 pt over `mixtta_ms224sp2`):

- Where `mixtta_ms224sp2`'s errors are inside a domain (proxy, 4 streams, seed 0):
  - batches that contain a domain boundary: 72.3 % accuracy;
  - the first full batch after a boundary: 81.1 %;
  - the second: 84.2 %;
  - the rest of the stream: 87.1 %.
  The test stream, which has no labels, shows the same pattern in the mean confidence: 0.175, 0.177, 0.195 and 0.206.
- Why the boundary batch hurts.
  - At 5 of the 7 test boundaries that fall inside a batch, the shift detector fires on that mixed batch. The new domain's statistics then start from a batch that is partly the old domain, and that batch keeps weight 1/k in the running mean.
  - Where the boundary falls in the batch matters. Dropping the first 16, 32 or 48 images of each proxy stream moves its boundaries within the batch, and `mixtta_ms224sp2` scores 85.29, 85.50, 83.64 and 82.18 % on the four versions (offsets 0–48).
- The split (step 0).
  - When the detector fires, one changepoint is fitted to the per-image conv1 statistics (channel means and log-variances) of the previous batch followed by the current one.
  - This uses only the past and the current batch.
  - Measured on the method's own splits, it places every proxy split within ±1 image of the true boundary. On val, 7 of 9 splits are within ±1 and 2 are 10 images early. On test, all 7 are within ±2 of the offline segmentation.
  - When only a few images of the batch belong to the new domain, the detector usually fires one batch late. That batch is all new domain, so it needs no split.
  - Rows before the changepoint are normalised with the old domain's statistics. The new domain's statistics start from the rows after it, and the running mean counts images instead of batches.
- The second pass (step 6). On the first 2 batches of a domain, the model self-trains first, then predicts the same views again. This adds one forward pass on 18 of the 192 test batches.
- Results (seed 0 / seed 1):

  | | Proxy, mean of 4 offsets (%) | Proxy, offset 0 (%) | Val7 (%) | Test time (s) |
  |---|---:|---:|---:|---:|
  | `mixtta_ms224sp2` | 84.15 / 84.15 | 85.29 / 85.37 | 93.88 / 93.78 | 24.6 |
  | + split (`sp2cp`) | 85.60 / 85.59 | 86.01 / 86.17 | 94.00 / 94.01 | 24.9 |
  | + split + second pass (`sp2cpr`) | 86.19 / 86.09 | 86.39 / 86.49 | 94.00 / 94.14 | 26.2 |

  - Harness val run (the original order): 94.46 % (`sp2`), 94.84 % (`sp2cp`), 94.93 % (`sp2cpr`).
  - The second pass alone gives 84.43 / 84.82 / 84.87 % for 1 / 2 / 3 batches (seed 0).
  - Self-training on all rows of the split batch instead of the new rows only gives the same result (85.56 %).
- On test, without labels:
  - the imbalance floor falls from 0.0420 (`sp2`) to 0.0377 (`sp2cp`) and 0.0375 (`sp2cpr`), and is lower for each of seeds 1–3;
  - the predictions stay balanced: tree frog +23 to +55, at most 5.6 % per class in the late domains;
  - against `mixtta_ms224sp2`, 2.4 % (`sp2cp`) and 2.7 % (`sp2cpr`) of the predictions change, and `sp2cp` and `sp2cpr` differ from each other by 0.85 %. Both use the same random draws as `sp2`, so a Kaggle score for either is a paired comparison with 0.92724. `sp2cp` is identical to `sp2` until the first split at batch 18; `sp2cpr` differs from the first batch on, because the second pass also covers the start of the stream.
- Kaggle result.
  - `mixtta_ms224sp2cpr` scored 0.93277 against 0.92724 for `mixtta_ms224sp2`: +0.55 pt, about 34 more correct images on the public part.
  - Expected beforehand: +0.2–0.4 pt, with about ±0.15–0.2 pt of paired noise for 2.7 % changed predictions, so the gain is larger than predicted and about 3× the noise.
  - The offset proxy predicted +2.0 pt, so it overstated this change by about 4×, as it did for the shift reset and the class prior.
  - The imbalance floor on test has tracked Kaggle for the scored MixTTA-MS files, with one exception:

    | Method | Test imbalance floor | Kaggle public |
    |---|---:|---:|
    | `mixtta_ms224` | 0.059 | 0.91585 |
    | `mixtta_ms224p2` | 0.047 | 0.92366 |
    | `mixtta_ms224sp2` | 0.042 | 0.92724 |
    | `mixtta_ms224sp3` (prior λ = 3) | 0.040 | 0.92740 |
    | `mixtta_ms224sp2cp` | 0.0377 | 0.93180 |
    | `mixtta_ms224sp2cpr` | 0.0375 | 0.93277 |

  - The exception is `mixtta_ms224sp3`, which has a lower floor than `mixtta_ms224sp2` but the same Kaggle score.
    - A change that acts on the predicted class counts, such as a stronger class prior, lowers the floor by construction.
    - The floor can therefore rank changes to the statistics or the training, but not changes to the prior.

  - `mixtta_ms224sp2cp` (split only) scored 0.93180. Of the +0.55 pt:
    - the split gives +0.46;
    - the second pass gives +0.10, about 6 images of the public half. That is at the edge of the ±0.1 pt noise for two files that differ on 0.85 % of the predictions, so the second pass's gain is not clearly measurable.

MixTTA-MS: per-stage learning rates (`mixtta_ms224sp2cprl3`: Kaggle public 0.93831, +0.55 pt over `mixtta_ms224sp2cpr`):

- How it was found. A multi-agent search proposed 18 ideas from 6 angles, and 3 judges picked 6 to implement:
  - feature-centred LAME;
  - two depth-dependent source weights;
  - re-predicting with the trained student;
  - TRIBE-style class-balanced BN;
  - per-stage learning rates for self-training.
  Only the last improved on both the proxy and val. The others were flat or worse.
- The change. Self-training keeps the same 3 Adam steps on the BN scale and shift, but each ResNet stage gets its own learning rate. It adds no time and no random draws, so its Kaggle score is a paired comparison with `mixtta_ms224sp2cpr`.
- Results (proxy mean of 4 offsets and val7, seeds 0 / 1; `mixtta_ms224sp2cpr`: 86.19 / 86.09, 94.00 / 94.14):

  | Multipliers (stem, layer1, layer2, layer3, layer4) | Proxy | Val7 | Largest class share in the thermal test domain |
  |---|---:|---:|---:|
  | 1, 1, 1, 1, 1 (`sp2cpr`) | 86.19 / 86.09 | 94.00 / 94.14 | 0.110 |
  | 2, 2, 2, 1, 0.25 (`sp2cprl`) | 86.98 / 86.84 | 94.39 / 94.46 | 0.095 |
  | 3, 3, 3, 1, 0.1 (`sp2cprl3`) | 87.53 / 87.57 | 94.66 / 94.74 | 0.098 |
  | 3, 3, 3, 2, 0.25 | 87.75 | 94.78 | 0.118 |
  | 3, 3, 3, 3, 0.25 | 87.19 | 94.07 | 0.239 |
  | 4, 4, 4, 2, 0.25 | 87.82 | 94.66 | 0.202 |

- Why. The early stages see low-level, corruption-specific statistics and can adapt faster. Layer4's scale and shift act directly on class evidence: a high rate there amplifies pseudo-label errors, and a global lr of 2e-3 had hurt.
- Limit. A rate of 2× or more on layer3 partly collapses the thermal domain on test (up to 24 % of it predicted as one class). The proxy cannot show this because its streams are only 33 batches long, so every setting was also checked on the test stream.
- Stability. `sp2cprl3` is stable for seeds 0–3 on test:
  - the imbalance floor is 0.035–0.037, against 0.035–0.038 for `sp2cpr`;
  - the largest class share is 0.104 in thermal and at most 0.071 in the late domains (timber wolf in the last domain).
- Kaggle. `mixtta_ms224sp2cprl3` scored 0.93831 against 0.93277, with 3.7 % of the predictions changed. The proxy predicted +1.4, so it again overstated the gain, by about 2.5×.

Why MixTTA outperforms RoTTA and CoTTA:

MixTTA is built around three properties of this stream:
- Short domains: 12 abrupt domain shifts of about 16 batches each.
- Skewed batches: each batch of 64 holds only 5–17 of the 20 classes.
- The baselines' training adds nothing here: removing the weight updates leaves RoTTA, BOSA (`lr=0`) and CoTTA (`cotta_noaug` = `bn`) at the same accuracy.

Its first version (`mixtta`) handled all three with forward passes only. The final version adds self-training on its own LAME-corrected predictions and explicit handling of domain shifts.

| | RoTTA | CoTTA | `mixtta` (first version) | `mixtta_ms224sp2cpr` (final) |
|---|---|---|---|---|
| BN statistics | Slow EMA (α = 0.05), updated from the memory bank | Current batch only | Faster EMA (α = 0.2) mixed 50/50 with source statistics | Running mean over the current domain, reset at detected shifts, mixed 50/50 with source statistics; one set per input size |
| When statistics update | After predicting | Before predicting | Before predicting | Before predicting |
| Prediction averaging | None | 32 strong augmentations, only when the source model is unsure | Flip + 4 small shifts, always | Flip + 2 small shifts at 160 and 224 px, always |
| Uses label skew at prediction | No (the memory bank uses it for training only) | No | Yes (LAME) | Yes (LAME, plus a global class prior) |
| Weight updates | BN affine parameters, on memory-bank samples | All parameters | None | BN affine parameters, on its own LAME-corrected predictions |
| Test time, RTX 5090 (s) | 15 | 68 | 21 | 26 |
| Kaggle public | 0.65983 | 0.79117 | 0.86246 | 0.93277 |

1. BN statistics (val 83.19 % with this step alone, against 72.16 % for `bn` and 76.23 % for `rbn`).
   - CoTTA normalises with the current batch. With about 10 classes per batch, those statistics are noisy and depend on which classes happen to be present.
   - RoTTA's EMA rate of 0.05 needs about 20 batches to follow a shift, longer than a test domain lasts; this is why `rbn` scores below `bn` on Kaggle.
   - MixTTA tracks about 4× faster but keeps half the source statistics, so one skewed batch cannot pull them far.
   - It also updates before predicting. In the RoTTA order, the first batch after each shift is normalised with the previous domain's statistics, which costs 4.4 pt on val (`mixtta_lag`).
2. Mild averaging (+3.2 pt, to 86.41 %). CoTTA's strong augmentations change image statistics and hurt on this data. Flips and small shifts keep the gain at 6 forward passes per batch instead of CoTTA's 33.
3. LAME (+2.7 pt, to 89.06 %). Within a batch, similar-looking images should share a label. LAME pushes predictions towards the few classes the batch actually contains.
4. Self-training on better targets. The weight updates of RoTTA, BOSA and CoTTA brought no gain on this stream, only drift and time. Self-training on MixTTA's own LAME-corrected predictions does help, because these targets are much more accurate than RoTTA's or CoTTA's pseudo-labels. `mixtta_st2` gains 1.80 pt on val and reaches 0.91 on Kaggle.
5. Faster tracking and view-averaged LAME features (`mixtta_st_notrain`: +1.0 pt on val, Kaggle 0.86246 → 0.88981).
6. MixTTA-MS (Kaggle 0.91585 → 0.93277):
   - a second input size;
   - the global class prior, which keeps self-training stable;
   - shift detection with a running mean per domain;
   - the split of boundary batches and the second pass.

Compared with BOSA, MixTTA keeps the mix of source and test statistics but separates the tracking rate from the pull towards the source. In BOSA both are set by one adaptive α_t, and that coupling caused the collapse at α_t ≈ 1.

These choices fit this stream. On a long, single-domain, class-balanced stream, LAME and the source mix would help less, and self-training might matter more.

## Q3 Strategy

The test stream has three properties that break standard TTA:
- **Distribution shift:** 12 corruption domains change abruptly, and 7 of the 11 boundaries fall inside a batch.
- **Non-i.i.d. sampling:** each batch of 64 holds only 5–17 of the 20 classes.
- **Length:** over 192 batches, self-training can drift.

The final method, `mixtta_ms224sp2cpr` (Kaggle 0.93277), addresses each as follows. Gains are on val unless marked Kaggle.

**Distribution shift**
- **Source-mixed BN.** Each BN layer normalises with 50 % tracked test statistics and 50 % source statistics: the tracked half follows the domain, and the source half is an anchor no single batch can move (against Robust BN: +4.3 pt from faster tracking, +2.6 pt from the source mix).
- **Update before predicting.** Each batch updates the statistics before it is predicted, so a new domain's first batch is not normalised with the old domain's statistics (+4.4 pt).
- **Shift detection and reset.** A jump in the conv1 statistics (KL divergence > 10× the domain's recent median) starts a new domain: the tracked statistics restart from it, then follow a running mean (Kaggle +0.36).
- **Splitting the boundary batch.** A changepoint on the per-image conv1 statistics finds where the new domain starts inside the batch. Images before it keep the old domain's statistics.
- **Self-training and a second pass.** After each batch, 3 Adam steps on the BN scale and shift fit the model to its own corrected predictions (Kaggle 0.88981 → 0.91). On the first 2 batches of a domain, the model trains first and then predicts the batch again (with the split: Kaggle +0.55).
- **Prediction averaging.** The softmax is averaged over flip and shift views (+3.2 pt) at input sizes 160 and 224, each size with its own BN statistics.

**Non-i.i.d. sampling**
- **No statistics from one batch alone.** A skewed batch's statistics depend on which classes it holds; averaging over the domain and the source keeps that bias small. CoTTA's current-batch statistics and RoTTA's slow EMA (about 20 batches to follow a shift) both do worse.
- **LAME.** LAME uses the skew rather than fighting it. Each prediction moves towards those of its 5 nearest neighbours in feature space, so the batch concentrates on the few classes it contains (+2.7 pt).
- **No memory bank.** RoTTA's and BOSA's class-balanced memory banks hurt here (removing them: +7.9 and +19 pt).

**Error accumulation and collapse**
- **Global class prior.** The stream is class-balanced overall, so each probability is divided by (20 × that class's share of the predictions so far)². Without it, self-training can drift into one class (up to 100 % "tree frog"); with it, every variant is stable across 4 seeds (Kaggle +0.78).
- **Small, depth-aware updates.** Only the BN scale and shift are trained, on soft targets: early stages at 3× the base rate of 1e-3, layer4 at 0.1× (Kaggle +0.55). Training all weights collapsed to 33 % on the proxy.

**Time.** bf16, fused views, training at 160 px only: 26 s on an RTX 5090 (limit: 90 s on a 3090).

## Q4 Visualization

![8 validation images with predicted and ground-truth labels](figures/val_examples.png)

**Setup.**
- The figure shows the final method, `mixtta_ms224sp2cprl3`, from its harness val run: 95.60 %, 93 of 2,112 images wrong.
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
