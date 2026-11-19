# Practical test-time adaptation

## Q1 Two settings for image classification

### Robust Test-Time Adaptation in Dynamic Scenarios (RoTTA)

- Problem addressed:
  1. continually changing distribution: distribution shift over time
  2. correlative sampling: the data distribution in a batch differs from global distribution
- Method:
  1. Robust Batch Normalization (RBN): instead of using statistics from the current test batch for normalization, they maintain global mean and variance. During adaptation, these statistics are updated using an exponential moving average (EMA) of statistics computed from samples in the memory bank. Since the memory bank maintains a more balanced representation of the data, normalization becomes less sensitive to class imbalance and correlations within the current batch.
  2. Category-balanced Sampling with Timeliness and Uncertainty (CSTU): maintains a memory bank with an equal memory quota for each category. When a new sample arrives, the teacher model predicts its category and uncertainty, and a heuristic score is calculated based on its age and prediction uncertainty. If replacement is needed, RoTTA selects a candidate from either the most populated category or the incoming sample's category. The candidate is replaced only if its heuristic score is higher than that of the new sample. This favors recent, confident samples while maintaining category balance.
  3. Training with timeliness: we first calculate the cross entropy between prediction of teacher model and that of student model (which predicts on strongly augmented samples). Then, the student model updates weights using the loss weighted by timeliness (where newer sample gets larger weight), and the teacher model updates weights using EMA from its own weight and student's weight.
- Pros:
  - Explicitly addresses both continual distribution shifts and non-IID sampling through robust batch normalization and category-balanced memory.
  - Reduces error accumulation using an EMA teacher and timeliness-weighted training.
  - Updates only BN parameters, reducing computational cost and limiting changes to pretrained knowledge.

- Cons:
  - Relies on potentially incorrect pseudo-labels for memory management and self-training.
  - Updating only BN parameters limits adaptation capacity under severe distribution shifts.
  - Slowly updated statistics may struggle with abrupt domain changes, while maintaining a memory bank introduces additional overhead.

### Continual Test-Time Domain Adaptation (CoTTA)

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

The CSV on this branch comes from `mixtta_local_fixed`. It uses the supplied 20-class ResNet-50 and an online batch size of 64. The only state required for the local class prior is the previous batch’s 20-way class probabilities.

1. **Track the image statistics.** Resize each batch to 160 and 224 pixels. Each size has its own batch-normalization (BN) statistics. Before prediction, update the tracked mean and variance with a fixed rate of 0.4. Mix the updated statistics equally with the source checkpoint’s statistics.
2. **Average views and refine by visual similarity.** At each size, run the original batch, its horizontal flip, and two random flip-and-shift views. Average their probabilities and penultimate features over both sizes. Let $p_i$ be the averaged class probabilities. Soften them with temperature 2 and apply LAME on a five-nearest-neighbor graph of batch features to get $z_i$.
3. **Estimate a local class prior.** For image $i$, count the LAME probabilities from the current and preceding batches. Give a row at stream distance $d$ weight $\exp(-d/12)$, exclude row $i$, and add 0.2 to every class count. Calling the counts $h_i$, the final soft target for class $c$ is proportional to $p_{ic}^{4/3} z_{ic}^{2/3} h_{ic}$. Return its argmax.
4. **Adapt for the next batch.** Use the soft targets for three Adam updates at learning rate 0.001. Minimize soft cross-entropy on shifted 160-pixel views, updating only the BN scale and bias. The targets come from model predictions. We used validation labels for tuning and scoring.

LAME follows Boudiaf et al., *Parameter-free Online Test-time Adaptation*, CVPR 2022. [Implementation and code attribution](starter/tta/lame.py) · [Local class prior](starter/tta/mixtta_ms.py)

The reported Kaggle public score for this CSV is **0.92789**, up from 0.91585 for the matched fixed-tracking baseline (+1.20 percentage points). The historical best in the repository was 0.93831.

The official harness generated the 12,288 submission predictions in one online pass. It recorded 25.9 seconds and a peak of 4,150 MiB on an RTX 5090. The submission checker verified the CSV against its run log. The 90-second limit on the TAs’ RTX 3090 remains unverified. [Run log](out/mixtta_local_fixed/test_run_log.json)

## Q3 Strategies and validation results

**Local class prior.** In validation, adjacent images share a class 24.8% of the time. At a separation of 64 images the rate falls to 5.5%. This motivated using nearby predictions to estimate which classes are common around each image. On the same fixed-tracking baseline, the local prior corrected 35 errors and introduced 11. Accuracy rose from **93.99% to 95.12%** (1,985 to 2,009 correct out of 2,112). Both official runs used seed 0.

**Domain shift and self-training.** Mixing the source and tracked BN statistics anchors the model during class-skewed batches. Updating before prediction raised single-scale MixTTA accuracy from 84.66% to 89.06%. Two self-training steps improved on the forward-only version by 1.80 points. The submitted preset combines multi-scale prediction with three updates per batch.

**Method selection.** We developed the local prior on validation using the original stream (seed 0), its reverse (offset 32, seed 1), and domain order 2–0–1 (offset 16, seed 0). The stage-5 version with a shift detector, `mixtta_local`, reached 96.07% on the official validation stream. Its ancestor’s development record says offline test analysis influenced candidate ideas before the val-only re-selection. Under the slide 21 FAQ we used the earlier fixed-tracking baseline for the branch CSV. [Development record](tools/dev/repick.py)

### Validation checks

The fixed-tracking submission averaged **93.92%** over seven stream orders, four offsets and two seeds. Its matched baseline averaged **93.12%**. On the 231-batch stream, accuracy rose from 94.70% on the first 2,112 images to 97.77% on the last 2,112. The largest class share in the final window was 5.26%, passing the collapse check. [Fixed-tracking results by order](out/mixtta_local_fixed/val_protocol.json)

| Validation measure | Stage 5 | Stage 5 + local prior |
|---|---:|---:|
| Official order | 94.74% | 96.07% |
| Mean of 7 orders × 4 offsets × 2 seeds | 93.91% | 94.84% |
| Shuffled within each domain | 80.73% | 85.98% |
| Additional JPEG compression, quality 35 | 87.88% | 89.77% |

The shuffled control used NumPy seed 17 for the within-domain permutations and adaptation seed 0. Raising the source BN weight from 0.5 to 0.7 gave 96.02% in the official order with the local prior, but only 62.83% in that shuffled control. The branch preset retains the original source weight of 0.5. The JPEG control was run after choosing the local prior. Each check reuses validation images. [Stage-5 results by order](out/mixtta_local/val_protocol.json)

### Earlier stage-wise validation sweep

A separate experiment scored each setting over 56 validation runs. It chose the simplest setting within 0.3 percentage points of the best after a 231-batch collapse check. The score in this table is the mean over orders, offsets and seeds. The CSV uses the fixed-tracking method described in Q2.

| Stage | Candidates | Kept | Score (%) |
|---|---|---|---:|
| 1 | self-training steps 2/3/4 × LAME T 1/2/3 | 3 steps, T = 2 (4 steps, T = 2: 93.33, within the margin) | 93.12 |
| 2 | no reset; reset with rate floor 0.4/0.2/0.1/0.05 | reset, floor 0.2 (0.1: 93.57, within the margin) | 93.52 |
| 3 | in-batch split off/on | on | 93.91 |
| 4 | second pass on a domain's first 0/1/2/3 batches | none (3 batches: 94.09, within the margin) | 93.91 |
| 5 | class prior λ = 0/1/2/3 | none (λ = 1: 93.96, within the margin) | 93.91 |
| 6 | 12 per-stage learning-rate settings and the uniform baseline | Multipliers (2, 2, 2, 1, 0.25) from stem to layer 4, within the margin of the best score of 94.41 | 94.16 |
| 7 | fixed temperature or adaptive confidence targets of 0.4, 0.3, 0.2 | Fixed T = 2. All three adaptive targets were within the 0.3-point margin (94.30, 94.31, 94.18) | 94.16 |

The sweep finished with the stage-6 learning rates and fixed temperature. Its selected setting passed the long-stream check. [Final configuration and scores](out/val_exp/repick_final.json)

### Method comparisons

Each value is top-1 accuracy on the official validation stream. The full per-method predictions and official logs are under [out/](out/).

| Strategy | Validation accuracy (%) |
|:-----------------------------------------|:-----------------------------------------|
| Frozen source model | 54.55 |
| Normalization alone | Current batch 72.16, Robust BN 76.23 |
| RoTTA | Full 65.01, current-batch BN 72.30, remove memory 72.92, remove timeliness 64.96 |
| CoTTA | Full 76.99, four augmentations 73.30, remove augmentation averaging 72.16, remove restoration 77.60, Robust BN 59.42 |
| BOSA | Full 60.70, remove memory 77.51, remove alignment 64.68, remove constraint 58.62, remove sparsification 60.56, entropy loss 59.19 |
| Single-scale MixTTA | Full 89.06, previous batch’s statistics 84.66, remove LAME 86.41, remove augmented views 86.93, mixed statistics alone 83.19 |
| BN ablations with MixTTA | Frozen source 55.82, current batch 80.68, robust batch statistics 82.24, fast tracking with zero source weight 86.51 |
| Self-training MixTTA at 160 px | Forward only 90.10, one update 91.15, two updates 91.90 |
| Multi-scale MixTTA with fixed tracking | Baseline 93.99, **local prior 95.12** |
| Multi-scale MixTTA with shift detection | Stage 5 baseline 94.74, local prior 96.07 |

## Q4 Validation examples

![Four correct and four incorrect predictions from the branch submission method](figures/val_examples.png)

The top row shows four correct predictions and the bottom row shows four errors. Each image carries its predicted class, ground truth, stream position, and validation domain. We sampled with seed 0, taking one example per domain and a second scan-line example in each row.

| Validation domain | Images | Baseline errors | Errors with local prior |
|---|---:|---:|---:|
| Scan-line grid | 776 | 47 | 36 |
| Wavy warp | 724 | 40 | 32 |
| Fisheye | 612 | 40 | 35 |
| **All** | **2,112** | **127** | **103** |

The fox and husky remain recognizable through the scan lines, and the eagle’s white head is visible inside the fisheye border. Three illustrated errors involve vehicles. The warp bends the yellow school bus (#842) into a curved shape that the model calls an acoustic guitar. The scan lines obscure the outlines of the fire engine (#37) and sports car (#356). In the fisheye example (#1861), the laptop screen reflects the landscape and the model predicts sports car. Regenerate the figure with `python tools/plot_val_examples.py --method mixtta_local_fixed`.
