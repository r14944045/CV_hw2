# CVPDL HW2: test-time adaptation

Assignment spec: [HW2.md](HW2.md). Experiments and method write-ups: [report.md](report.md).

## Current best submission

| | |
|---|---|
| Method | `mixtta_st2` (MixTTA: source-mixed BN statistics with α = 0.4 and w = 0.5, flip + 4-shift averaging, LAME on view-averaged features, then 2 Adam steps per batch on BN γ/β towards the LAME output) |
| Kaggle public | **0.91** |
| Val accuracy | 91.90 % |
| Test-stream time | 33.0 s on an RTX 5090 (limit: 90 s on a 3090); peak VRAM 3.8 GB |
| Code | [starter/tta/mixtta.py](starter/tta/mixtta.py), registered in [starter/methods.py](starter/methods.py) |

### Generate the submission file

Run from the repo root on a CUDA machine. It needs `data/` (unzipped from `ntu-cvpdl-2026-hw-2.zip`) and `source_resnet50_160.pth`.

```bash
METHOD=mixtta_st2
cd starter
python run.py --method $METHOD --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/$METHOD
cd ..
python tools/check_submission.py out/$METHOD     # format, run-log hash, time and VRAM checks
sha256sum out/$METHOD/test_pred.csv
```

The file to upload is `out/mixtta_st2/test_pred.csv`.

The submitted file's SHA-256 is `2359887ab175d886c0231e6f5079635ecec84612c04680d47070b692125bb8f5`. It was produced with torch 2.13.0+cu129, seed 0, on an RTX 5090.

A rerun will not reproduce this hash exactly. The gradient steps amplify GPU nondeterminism (cuDNN autotuning), so three reruns each differed from the submitted file on 9–12 of the 12,288 predictions (under 0.1 %), and from each other. Keep the submitted file itself as the reference.

The forward-only `mixtta_st_notrain` (Kaggle 0.88981) does reproduce exactly. Its SHA-256 is `614548cc5b4e0d5e168e546c11b4cf18724975dc3607024f6c33739914b1fc32`.

`bash tools/run_gpu.sh $METHOD` runs val (prints accuracy) and test, then runs the checker.

## Submission history (Kaggle public)

| Method | Kaggle public | Val (%) |
|---|---:|---:|
| `mixtta_st2` | **0.91** | 91.90 |
| `mixtta_st_notrain` | 0.88981 | 90.10 |
| `mixtta` | 0.86246 | 89.06 |
| `cotta` | 0.79117 | 76.99 |
| `cotta_norst` | 0.79085 | 77.60 |
| `bosa_stable_nomem` | 0.78662 | 79.50 |
| `bn` | 0.72037 | 72.16 |
| `rbn` | 0.70231 | 76.23 |
| `rotta` | 0.65983 | 65.01 |

`cotta` and `cotta_norst` take about 68 s on the test stream on an RTX 5090, so they may exceed the 90 s limit on a 3090.

Generated but not submitted:

| Method | Val, 7 orders (%) | Test time on RTX 5090 (s) | Changed vs the 0.91 file | SHA-256 (first 16) |
|---|---:|---:|---:|---|
| `mixtta_ms224` | 93.18 | 26.6 | 8.4 % | `75b92de51e2742b5` |
| `mixtta_ms3s3` | 93.53 | 32.5 | 8.7 % | `3b12773d9d323d53` |
| `mixtta_ms224_t1` (control: LAME T=1) | 93.02 | 26.2 | 6.8 % | `14ee137f2d0c0a80` |
| `mixtta_ms3` | 93.31 | 31.0 | 8.5 % | `8361707ae3ebcea4` |
| `mixtta_st` | — | 27.2 | — | — |

- The `mixtta_ms*` methods ([starter/tta/mixtta_ms.py](starter/tta/mixtta_ms.py)) run MixTTA with self-training at several input sizes: 160 plus 224, or 160 plus 192 and 224. Each size has its own BN statistics, and self-training runs at 160 only.
- They use bf16 and deterministic cuDNN, so a rerun on the same GPU and software reproduces the hash. A different GPU (such as the TAs' 3090) gives a slightly different file.
- "Val, 7 orders" is the mean over the 7 val stream orders and 2 seeds. Differences under about 0.3 pt are noise.
- Most of the multi-scale val gain comes from the val fisheye domain (small objects inside black borders), which the test stream does not have. Expect a smaller gain on Kaggle.
- `mixtta_st` takes 1 self-training step per batch instead of 2.

Generate any of these with the commands above by setting `METHOD=<name>`.
