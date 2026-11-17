# CVPDL HW2: test-time adaptation

Assignment spec: [HW2.md](HW2.md). Experiments and method write-ups: [report.md](report.md).

## Current best submission

| | |
|---|---|
| Method | `mixtta_ms224sp2cpr` (MixTTA-MS with domain-shift reset, an in-batch split of the shift batch, a class prior and a second pass after each shift). It is self-training MixTTA at input sizes 160 and 224, with separate BN statistics per size, flip + 2-shift averaging, LAME on T = 2-softened probabilities and 3 Adam steps per batch on BN γ/β at 160. Additions: online domain-shift detection that resets the tracked BN statistics, starting from the images after a changepoint found inside the batch; running-mean statistics within a domain; a global class-prior correction (λ = 2) that keeps self-training from drifting; and, on the first 2 batches of each domain, self-training before predicting the batch again. bf16, deterministic cuDNN |
| Kaggle public | **0.93277** |
| Val accuracy | 94.93 % (mean over 7 val stream orders: 94.00 %) |
| Test-stream time | 26.2 s on an RTX 5090 (limit: 90 s on a 3090); peak VRAM 4.6 GB |
| Code | [starter/tta/mixtta_ms.py](starter/tta/mixtta_ms.py), registered in [starter/methods.py](starter/methods.py) |

### Generate the submission file

Run from the repo root on a CUDA machine. It needs `data/` (unzipped from `ntu-cvpdl-2026-hw-2.zip`) and `source_resnet50_160.pth`.

```bash
METHOD=mixtta_ms224sp2cpr
cd starter
python run.py --method $METHOD --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/$METHOD
cd ..
python tools/check_submission.py out/$METHOD     # format, run-log hash, time and VRAM checks
sha256sum out/$METHOD/test_pred.csv
```

The file to upload is `out/mixtta_ms224sp2cpr/test_pred.csv`.

The submitted file's SHA-256 is `fb587264ef394f0121a348f7dff3f29ac28a16968291fe73684b123d1cc25c93`. It was produced with torch 2.13.0+cu129, seed 0, on an RTX 5090.

The method uses deterministic cuDNN algorithms, so a rerun on the same GPU and software reproduces this hash exactly (checked with two harness runs). Reruns with seeds 1–3:
- change 3.8–3.9 % of the test predictions;
- stay class-balanced: tree frog +23 to +33 predictions, no class above 5.5 % in the late domains.

A different GPU (such as the TAs' 3090) should behave like another seed.

The previous best, `mixtta_ms224sp2` (0.92724, SHA-256 `30dc0a8b…`), is the same method without the in-batch split and the second pass. Regenerate it with `METHOD=mixtta_ms224sp2`.

An earlier best, `mixtta_ms224` (0.91585, SHA-256 `75b92de5…`), is not stable across seeds: seed 3 drifts into predicting tree frog (+1039).

`bash tools/run_gpu.sh $METHOD` runs val (prints accuracy) and test, then runs the checker.

## Submission history (Kaggle public)

| Method | Kaggle public | Val (%) |
|---|---:|---:|
| `mixtta_ms224sp2cpr` | **0.93277** | 94.93 |
| `mixtta_ms224sp3` | 0.92740 | — |
| `mixtta_ms224sp2` | 0.92724 | 94.46 |
| `mixtta_ms224p2` | 0.92366 | 94.08 |
| `mixtta_ms224` | 0.91585 | 93.99 |
| `mixtta_st2` | 0.91 | 91.90 |
| `mixtta_st_notrain` | 0.88981 | 90.10 |
| `mixtta` | 0.86246 | 89.06 |
| `cotta` | 0.79117 | 76.99 |
| `cotta_norst` | 0.79085 | 77.60 |
| `bosa_stable_nomem` | 0.78662 | 79.50 |
| `bn` | 0.72037 | 72.16 |
| `rbn` | 0.70231 | 76.23 |
| `rotta` | 0.65983 | 65.01 |

Copies of the scored files are kept in [out/_submitted/](out/_submitted/) (except `mixtta_ms224`, `mixtta_ms224sp2` and `mixtta_ms224sp2cpr`, which reproduce exactly).

- `mixtta_st2` (SHA-256 `2359887ab175d886…`) does not reproduce bit-for-bit. Reruns differ on 9–12 of the 12,288 predictions, so its reference copy is `out/_submitted/mixtta_st2_kaggle0.91.csv`.
- `cotta` and `cotta_norst` take about 68 s on the test stream on an RTX 5090, so they may exceed the 90 s limit on a 3090.

### Shift-reset and class-prior candidates

| # | Method | Val, 7 orders (%) | Proxy (%) | Long proxy (%) | Test imbalance floor | Changed vs `mixtta_ms224` | SHA-256 (first 16) |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `mixtta_ms224p2` (control: class prior only): Kaggle 0.92366 | 93.20 | 83.74 | 87.99 | 0.047 | 3.7 % | `44d79b0853c74e69` |
| 2 | `mixtta_ms224rp2` (shift reset + class prior) | 93.51 | 85.24 | 89.11 | 0.043 | 5.0 % | `e3c14c40f36d1e05` |
| 3 | `mixtta_ms224sp2` (shift reset + running-mean statistics + class prior): **Kaggle 0.92724** | 93.88 | 85.29 | 89.58 | 0.042 | 6.2 % | `30dc0a8be78faf3d` |
| — | `mixtta_ms224` (Kaggle 0.91585, for reference) | 93.34 | 80.13 | 84.62 | 0.059 | — | `75b92de51e2742b5` |

On Kaggle the class prior alone is worth +0.78 pt (`mixtta_ms224p2` 0.92366 against `mixtta_ms224` 0.91585). The shift reset with running-mean statistics adds +0.36 pt (`mixtta_ms224sp2` 0.92724), which is 22 images of the public half and within seed noise. `mixtta_ms224rp2` is not scored.

- **Shift reset (steps 0–0b in [starter/tta/mixtta_ms.py](starter/tta/mixtta_ms.py)).** A batch whose conv1 statistics jump by more than 10× the domain's recent median starts a new domain, and its statistics replace the tracked ones.
  - On test the detector fires at all 11 domain boundaries, plus once inside the halftone domain (batch 148).
  - The 12 test domains are 676–1252 images long. Their boundaries are listed in [tools/dev/test_bounds.npy](tools/dev/test_bounds.npy).
- **The class prior is needed for stability, not just for score.**
  - Without it, BN self-training on the test stream can drift into predicting tree frog after the thermal-colormap domain.
  - `mixtta_ms224` itself does this for seed 3: +1039 tree-frog predictions, and 18 % of the last three domains predicted as one class. Seeds 0–2 do not drift.
  - `mixtta_ms224rp2` and `mixtta_ms224sp2` stay balanced for seeds 0–3: tree frog +30 to +59, at most 5.7 % per class in the late domains.
  - Before the prior is applied, their raw ensemble argmax also stays balanced, so the prior is not hiding a degrading model.
- **Never submit `mixtta_ms224r` or `mixtta_ms224s`.** They are the same methods without the prior, and both drift into tree frog on test (+1120 and +3349 predictions). Their files are kept in `out/` as ablations.
- **Metrics.**
  - Proxy = val images with synthetic versions of the 12 test corruptions, 4 streams of 3 domains ([tools/dev/](tools/dev/)). It ranks the 5 scored MixTTA methods in the Kaggle order, but its absolute accuracy is lower.
  - Long proxy = the 4 streams back to back (132 batches).
  - Imbalance floor = the minimum error rate implied by the predicted class counts if every test domain is exactly class-balanced, as every val domain is.
  - Reruns with another seed change about 4 % of the test predictions, so `rp2` and `sp2` (3.3 % apart) cannot be ranked offline.
- **Time.** On a quiet RTX 5090 back to back, `mixtta_ms224rp2` and `mixtta_ms224sp2` take 28.5 s and `mixtta_ms224` takes 28.2 s. Peak VRAM is 4.2 GB.
  - Those reruns reproduced each file's SHA-256 exactly, and their run logs are the ones in `out/`.
  - Run time has not been measured on a 3090.

### Splitting the shift batch (current best)

| Method | Change | Proxy, mean of 4 offsets (%), seeds 0 / 1 | Val, 7 orders (%), seeds 0 / 1 | Val, harness (%) | Test imbalance floor | Changed vs `sp2` | Test time on RTX 5090 (s) | SHA-256 (first 16) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `mixtta_ms224sp2cp` | In-batch split of the shift batch | 85.60 / 85.59 | 94.00 / 94.01 | 94.84 | 0.0377 | 2.4 % | 24.9 | `c5d8e6cdcb6432af` |
| `mixtta_ms224sp2cpr` (best): **Kaggle 0.93277** | Split + second pass on the first 2 batches of a domain | 86.19 / 86.09 | 94.00 / 94.14 | 94.93 | 0.0375 | 2.7 % | 26.2 | `fb587264ef394f01` |
| `mixtta_ms224sp2` (Kaggle 0.92724) | — | 84.15 / 84.15 | 93.88 / 93.78 | 94.46 | 0.0420 | — | 24.6 | `30dc0a8be78faf3d` |

- **Why.** `mixtta_ms224sp2` loses most accuracy on batches that contain a domain boundary (proxy: 72 % against 87 % elsewhere). 7 of the 11 test boundaries fall inside a batch, and at 5 of them the shift detector fires on that mixed batch. The new domain's statistics then start from a batch that is partly the old domain.
- **Split (step 0c in [starter/tta/mixtta_ms.py](starter/tta/mixtta_ms.py)).**
  - On a detected shift, a changepoint fitted to the per-image conv1 statistics of the previous and current batch finds where the new domain starts. Measured on the method's own splits:
    - proxy: all splits within ±1 image of the true boundary;
    - val (7 orders): 7 of 9 within ±1, and 2 are 10 images early;
    - test: all 7 within ±2 of the offline segmentation, plus one at the detector's extra firing inside the halftone domain.
  - When only a few images of the batch are new, the detector usually fires one batch late. That batch is all new domain, so it is not split.
  - Rows before it keep the old domain's statistics, and the new domain's statistics start from the rows after it.
  - The step uses only past and current images, and it changes no random draws.
- **Second pass (step 5).** On the first 2 batches of each domain, the model self-trains on the batch first, then predicts the same views again. It runs on 18 of the 192 test batches and adds 6.5 % to the time.
- **Proxy offsets.** "Mean of 4 offsets" averages the proxy with the first 0, 16, 32 and 48 images of each stream dropped, so each boundary falls at four different positions in its batch.
- **Stability.** Both are stable on test for seeds 0–3: imbalance floor 0.034–0.038 against 0.038–0.042 for `sp2`, tree frog +23 to +55, and at most 5.6 % per class in the late domains.
- **Paired comparisons.** Both follow `sp2`'s random sequence, so a Kaggle score for either, against 0.92724, measures the change itself. `sp2cp` is identical to `sp2` up to batch 18.
- **Kaggle.** `mixtta_ms224sp2cpr` scored 0.93277, +0.55 pt over `mixtta_ms224sp2` (expected +0.2–0.4; the offset proxy predicted +2.0, again about 4× too high). `mixtta_ms224sp2cp` is not scored, so the gain is not yet divided between the split and the second pass.

### Variants of `mixtta_ms224sp2`

| Method | Change | Val, 7 orders (%) | Proxy (%) | Test imbalance floor | Changed vs `sp2` | SHA-256 (first 16) |
|---|---|---:|---:|---:|---:|---|
| `mixtta_ms224sp3`: Kaggle 0.92740 | Class prior λ = 3 | — | 85.42 | 0.040 | 1.3 % | `090a4c960788f4e9` |
| `mixtta_ms224sp2f` | Faster prior: pseudo-count 1000 | — | — | 0.041 | 1.2 % | `45dd36cb6cd0bae6` |
| `mixtta_ms224sp2t` | Temporal prior: × (previous batch's mean LAME output + 0.1)^0.2 | 93.96 | 85.12 | 0.042 | 1.6 % | `889489681935c8c4` |
| `mixtta_ms3sp2` | Third input size, 192 | 93.81 | 85.00 | 0.043 | 3.9 % | `5ea61220011130ee` |
| `mixtta_ms224sp2` | — | 93.88 | 85.29 | 0.042 | — | `30dc0a8be78faf3d` |

- **Offline evaluation cannot judge the prior settings.** The proxy predicted +3.6 pt for the class prior and Kaggle gave +0.78. Val is too short for the prior to act.
- **`sp3` and `sp2f` are clean comparisons.** They follow the same random sequence as `sp2`, so they differ from it only through the prior setting, on 1.2–1.3 % of the predictions. A Kaggle score for either, compared with 0.92724, is a paired measurement of that setting. `sp3` scored 0.92740 (+0.016 pt, about one image of the public half), so λ = 2 and λ = 3 are equivalent.
- `sp2t` and `ms3sp2` show no offline gain. `ms3sp2` is also about 18 % slower.
- All four are stable on test: no class above 5.6 % of the predictions in the late domains.

Generated but not submitted (older):

| Method | Val, 7 orders (%) | Test time on RTX 5090 (s) | Changed vs `mixtta_ms224` | SHA-256 (first 16) |
|---|---:|---:|---:|---|
| `mixtta_ms224p` (class-prior correction, λ = 1) | 93.17 | 24.7 | 2.6 % | `826dc3dc6aacc643` |
| `mixtta_ms3s3` | 93.53 | 32.5 | 4.3 % | `3b12773d9d323d53` |
| `mixtta_ms224_t1` (control: LAME T=1) | 93.02 | 26.2 | 5.0 % | `14ee137f2d0c0a80` |
| `mixtta_ms3` | 93.31 | 31.0 | 4.8 % | `8361707ae3ebcea4` |
| `mixtta_st` | — | 27.2 | — | — |

- `mixtta_ms224p` and `mixtta_ms224p2` are `mixtta_ms224` plus an online class-prior correction (step 3b in [starter/tta/mixtta_ms.py](starter/tta/mixtta_ms.py)).
  - The test stream is class-balanced overall, but the model over-predicts some classes (tree frog +126, container ship +100) and under-predicts others (tabby −173, sports car −106).
  - Each batch's probabilities are divided by (C × the share of each label predicted so far)^λ, with a pseudo-count of 2000 samples so the correction starts slowly.
  - It does not change val, which is too short for it to act. On a long class-balanced proxy it stops self-training drifting in late domains (74.5 → 82.2 / 83.8 % for λ = 1 / 2).
  - It also lowers the test stream's per-domain imbalance floor from 6.5 % to 5.7 / 5.3 %.
- `mixtta_ms3` and `mixtta_ms3s3` add a third input size, 192. `mixtta_ms3s3` takes 3 self-training steps per batch and `mixtta_ms3` takes 2.
- "Val, 7 orders" is the mean over the 7 val stream orders and 2 seeds. Differences under about 0.3 pt are noise.
- Most of the multi-scale val gain comes from the val fisheye domain (small objects inside black borders), which the test stream does not have. On Kaggle, `mixtta_ms224` gained 0.6 pt over `mixtta_st2`; val showed 2.3 pt.
- `mixtta_st` takes 1 self-training step per batch instead of 2.

Generate any of these with the commands above by setting `METHOD=<name>`.
