# CVPDL HW2: test-time adaptation

Assignment spec: [HW2.md](HW2.md). Experiments and method write-ups: [report.md](report.md).

## Setup

Python 3.10 and a CUDA GPU with bf16 support (Ampere or newer). The pinned versions in [requirements.txt](requirements.txt) are the ones every run here used.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
```

Unzip the dataset into `data/` (from `ntu-cvpdl-2026-hw-2.zip`) and put `source_resnet50_160.pth` in the repo root.

## Data use

- Only the official harness ([starter/harness.py](starter/harness.py), unmodified) reads the test stream, in one online pass per submission (`starter/run.py --split test`).
- Settings are chosen on val ([tools/dev/](tools/dev/), val only) and with Kaggle public scores.

## Best submission

| | |
|---|---|
| Method | `mixtta_ms224sp2cprl3` (details below). Reproduce it with `mixtta_ms224sp2cprl3v`, which gives a byte-identical test file |
| Kaggle public | **0.93831** |
| Val accuracy | 95.60 % (harness) |
| Test-stream time | 27.3 s on an RTX 5090. The limit is 90 s on a 3090, where it has not been measured. Peak VRAM 4.6 GB |
| Submitted file | `out/mixtta_ms224sp2cprl3v/test_pred.csv`, SHA-256 `4b5217d9336d2c875e5bff54be22fe9dbab42c54ec6c02f984e6d576ff793d8c` (torch 2.13.0+cu129, seed 0, RTX 5090) |
| Code | [starter/tta/mixtta_ms.py](starter/tta/mixtta_ms.py), registered in [starter/methods.py](starter/methods.py) |

The method is MixTTA-MS: self-training MixTTA at input sizes 160 and 224, with separate BN statistics per size.
- **Predictions:** flip and 2-shift averaging, then LAME on T = 2-softened probabilities. bf16, deterministic cuDNN.
- **Self-training:** 3 Adam steps per batch on BN γ/β at 160, with per-stage learning rates of 3× the base 1e-3 for the stem, layer1 and layer2, 1× for layer3 and 0.1× for layer4.
- **Domain-shift reset:** online detection resets the tracked BN statistics. The threshold, 8.7, is chosen on val only by [tools/dev/threshold_val.py](tools/dev/threshold_val.py). The reset starts from the images after a changepoint found inside the batch.
- **Within a domain:** running-mean statistics.
- **Global class prior:** a correction with λ = 2.
- **Second pass:** on the first 2 batches of each domain, the model self-trains first, then predicts the batch again.

`mixtta_ms224sp2cprl3` uses a hand-set threshold of 10, and `mixtta_ms224sp2cprl3v` the val-chosen 8.7. Otherwise they are the same method, and they give identical predictions on val and on test (same SHA-256).

### Generate the submission file

Run from the repo root on a CUDA machine. It needs `data/` (unzipped from `ntu-cvpdl-2026-hw-2.zip`) and `source_resnet50_160.pth`.

```bash
METHOD=mixtta_ms224sp2cprl3v
cd starter
python run.py --method $METHOD --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/$METHOD
cd ..
python tools/check_submission.py out/$METHOD     # format, run-log hash, time and VRAM checks
sha256sum out/$METHOD/test_pred.csv
```

The file to upload is `out/$METHOD/test_pred.csv`. The method uses deterministic cuDNN algorithms, so a rerun on the same GPU and software gives the same file.

`bash tools/run_gpu.sh $METHOD` runs val (prints accuracy) and test, then runs the checker.

## Val-only re-pick of the settings

[tools/dev/repick.py](tools/dev/repick.py) re-chooses every MixTTA-MS setting from val alone. Its protocol is written in its docstring and was fixed before any run.
- **Score:** each candidate runs on the 7 val stream orders × 4 positions of the domain boundaries within the batch × 2 seeds.
- **Collapse veto:** the 7 orders back to back. It vetoes candidates that collapse but never ranks them.
- **Choice:** in each of 7 stages, the simplest candidate within 0.3 pt of the best (stages: self-training steps and LAME temperature, statistics tracking, in-batch split, second pass, class prior, per-stage learning rates, adaptive temperature).
- **Shift threshold:** each tracking setting gets its own, from the val rule in [tools/dev/threshold_val.py](tools/dev/threshold_val.py).

It reads no test data.

```bash
cd tools/dev
python cache.py      # decode the val images once into cache/val.pt (val only)
python repick.py     # resumable: results in repick_results.jsonl, decisions in repick_log.txt
```

The final setting is written to `tools/dev/repick_final.json`. The run takes about 41 candidate evaluations of roughly 5 minutes each on an idle RTX 5090 with one worker (`WORKERS` in the script, about 4.7 GB of VRAM each). To produce a submission, register that setting in `starter/methods.py` and run the harness commands above with its name.

## Submission history (Kaggle public)

| Method | Kaggle public | Val (%) |
|---|---:|---:|
| `mixtta_ms224sp2cprl3` (reproduced by `mixtta_ms224sp2cprl3v`) | **0.93831** | 95.60 |
| `mixtta_ms224sp2cpr` | 0.93277 | 94.93 |
| `mixtta_ms224sp2cp` | 0.93180 | 94.84 |
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

Every scored file is a harness output: its SHA-256 matches the `pred_sha256` in its `test_run_log.json`. Copies are kept in [out/_submitted/](out/_submitted/) or reproduce exactly from the commands above.

- `mixtta_st2` does not reproduce bit-for-bit (non-deterministic cuDNN). Its reference copy is `out/_submitted/mixtta_st2_kaggle0.91.csv`.
- `cotta` and `cotta_norst` take about 68 s on the test stream on an RTX 5090, so they may exceed the 90 s limit on a 3090.

## What each step is worth on Kaggle

Each step is measured by comparing two scored submissions that differ only in that step.

| Step | Submissions compared | Kaggle gain (pt) |
|---|---|---:|
| Class prior | `mixtta_ms224` → `mixtta_ms224p2` | +0.78 |
| Shift reset and running-mean statistics | `mixtta_ms224p2` → `mixtta_ms224sp2` | +0.36 (within seed noise) |
| In-batch split of the shift batch | `mixtta_ms224sp2` → `mixtta_ms224sp2cp` | +0.46 |
| Second pass | `mixtta_ms224sp2cp` → `mixtta_ms224sp2cpr` | +0.10 (edge of noise) |
| Per-stage learning rates | `mixtta_ms224sp2cpr` → `mixtta_ms224sp2cprl3` | +0.55 (val 94.93 → 95.60 %) |
| Class-prior strength λ = 3 instead of 2 | `mixtta_ms224sp2` → `mixtta_ms224sp3` | +0.016, so the two are equivalent |
| Second input size, 224 | `mixtta_st2` → `mixtta_ms224` | +0.6 (val +2.3) |

Other registered variants, generated but not submitted: `mixtta_ms224sp2cprl`, `mixtta_ms224sp2cprl3q3`, `mixtta_ms224rp2`, `mixtta_ms224sp2f`, `mixtta_ms224sp2t`, `mixtta_ms3sp2`, `mixtta_ms224p`, `mixtta_ms3`, `mixtta_ms3s3`, `mixtta_ms224_t1`, `mixtta_st`. Generate any of them by setting `METHOD=<name>`.
