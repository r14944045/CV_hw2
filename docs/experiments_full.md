# All experiments (val and Kaggle)

Moved out of report.md to keep the report within the 6-page limit.


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
| `mixtta_valsel_floor` | 2 Adam steps and T = 1 instead of 3 and 2 (interim val-only submission) | 94.03 | +39.48 | — | 4150 | 0.91227 |
| `mixtta_valsel_s5` | + shift detection (threshold 9.0 from the val rule) with reset, running-mean statistics floored at 0.2, in-batch split at a changepoint (step 0) | 94.74 | +40.19 | — | 4090 | 0.92220 |
| `mixtta_valsel` | + per-stage self-training learning rates ×(2, 2, 2, 1, 0.25) for stem, layer1–layer4 (stage 6) | 95.08 | +40.53 | 4.7 | 4090 | 0.92675 |
| `mixtta_local` | Stage 5 with a local class prior for prediction and self-training. Collapsed on the public board | 96.07 | +41.52 | 17.0 | 4090 | 0.68554 |
| `mixtta_r8` | `mixtta_valsel` + the local class prior. Collapsed on the public board | 96.69 | +42.14 | 14.8 | 4090 | 0.82682 |
| **Final branch** (fixed tracking) | | | | | | |
| `mixtta_local_fixed` | Fixed-tracking `mixtta_ms224` with the same local class prior | 95.12 | +40.58 | 15.1 | 4148 | 0.92789 |
| `mixtta_lfs` | + per-stage self-training learning rates ×(2, 2, 2, 1, 0.25) | 95.31 | +40.76 | 10.3 | 4150 | 0.93815 |
| `mixtta_lfsq` | + adaptive LAME temperature: per batch, T in [1, 3] with mean max-probability 0.3 (round 10) | 95.27 | +40.72 | 4.6 | 4150 | 0.94042 |
| `mixtta_lfsq4` | per-stage rates ×(4, 4, 4, 1, 0.1) instead (round 12). **Final** | **95.93** | +41.38 | 4.6 | 4148 | **0.94319** |

Val times were measured on a shared RTX 5090, so they vary with load. The ImageNet-C suite rounds (8, 10–14) behind the final branch are logged in `tools/dev/round*_log.txt`; their rules are in each script's docstring.

| Round (base) | Candidate | Tuning half | Confirmation half | V | Decision |
|---|---|---:|---:|---:|---|
| 8 (`mixtta_valsel`, margin 0.35) | local class prior | +1.12 | +1.89 | +1.08 | adopted; collapsed on Kaggle (`mixtta_r8`) |
| 8 | second pass on 2 batches | +0.51 | +0.39 | +0.19 | passed; dropped in the combination |
| 8 | adaptive LAME temperature 0.3 | −0.17 | +1.14 | +0.15 | no |
| 8 | DELTA reweighting | +0.16 | +0.17 | +0.05 | no |
| 8 | EATA filtering | −3.77 | −0.81 | −1.72 | no |
| 8 | ROID weights / weight ensembling / prior correction | −1.20 / −1.38 / −0.93 | −0.54 / −0.40 / −0.99 | −0.61 / −0.03 / −1.79 | no |
| 8 | SAR recovery | ±0 (never triggered) | ±0 | ±0 | no |
| 8b (`mixtta_valsel`) | ImageNet-C augmentation of the self-training copy | −1.97 | −0.20 | −0.74 | no |
| 10 (`mixtta_lfs`) | adaptive LAME temperature 0.3 | +0.67 | +1.76 | −0.21 | **adopted** |
| 10 | DELTA reweighting | +0.18 | +0.11 | −0.01 | no |
| 10 | 4 self-training steps | +0.77 | −0.51 | +0.28 | no |
| 10 | NOTE instance-aware BN | −10.09 | −2.43 | −3.87 | no |
| 11 (`mixtta_lfsq`) | 3 shift views | +0.06 | −0.25 | +0.12 | no |
| 11 | tracking rate 0.2 / 0.6 | −7.14 / +0.29 | −6.35 / +0.72 | −0.50 / −0.29 | no |
| 11 | source weight 0.3 / 0.7 | +0.32 / −18.05 | +1.14 / −16.17 | −0.74 / −0.18 | no |
| 12 (`mixtta_lfsq`) | per-stage rates ×(3, 3, 3, 1, 0.1) | +1.12 | +0.29 | +0.20 | passed; dropped in the combination |
| 12 | per-stage rates ×(4, 4, 4, 1, 0.1) | +1.57 | +0.38 | +0.31 | **adopted** |
| 12 | per-stage rates ×(2, 2, 2, 2, 0.25) | −0.98 | −1.91 | +0.02 | no |
| 12 | temperature target 0.2 / 0.4 | −0.12 / −0.08 | −0.31 / −0.24 | +0.15 / −0.14 | no |
| 12 | LAME k = 3 / 8 | −0.37 / −5.83 | −0.68 / −5.68 | −0.72 / −1.35 | no |
| 13 (`mixtta_lfsq4`, margin 2.16) | per-stage rates ×(5, 5, 5, 1, 0.1) | +0.16 | −0.51 | +0.05 | no |
| 13 | per-stage rates ×(4, 4, 4, 1, 0.05) / ×(4, 4, 4, 2, 0.1) | −0.04 / −0.94 | +0.00 / −2.33 | +0.03 / +0.04 | no |
| 13 | DELTA reweighting | +0.16 | +0.18 | +0.01 | no |
| 14 (`mixtta_lfsq4`) | T3A prototype classifier, K = 20 / 100 | −1.62 / −1.93 | −0.56 / −1.50 | +0.48 / +0.47 | no |

TRIBE's balanced BN and UnMix-TNS were screened on the first 20 val batches only (89.3 % and 17.3 %, against 95.31 % for `mixtta_lfsq`) and not run further.
