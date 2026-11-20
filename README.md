# CVPDL HW2: test-time adaptation

The [assignment slides](CVPDL%202026%20hw2%20slides.pptx.pdf) cover grading and submission on slides 15–17 and the adaptation rules on slides 19–21. The [starter instructions](HW2.md) describe the stream format. Results are in [report.md](report.md).

## Setup

Python 3.10 and a CUDA GPU with bf16 support (Ampere or newer). The pinned versions in [requirements.txt](requirements.txt) are the ones every run here used.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
```

Unzip the dataset into `data/` (from `ntu-cvpdl-2026-hw-2.zip`) and put `source_resnet50_160.pth` in the repo root.

### Generate the final file (`mixtta_lfsq4`, Kaggle public 0.94319)

`mixtta_lfsq4` is fixed-tracking `mixtta_ms224` with the local class prior, the adaptive LAME temperature and per-stage learning rates ×(4, 4, 4, 1, 0.1). Its settings were chosen on val, on ImageNet-C corruptions applied to val, and on the public leaderboard; see [report.md](report.md) (Q2) and [docs/experiments_full.md](docs/experiments_full.md).

```bash
METHOD=mixtta_lfsq4
cd starter
python run.py --method $METHOD --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/$METHOD
cd ..
python tools/check_submission.py out/$METHOD     # format, run-log hash, time and VRAM checks
sha256sum out/$METHOD/test_pred.csv              # 48c04860a92e2eae3ff4b3858876df7bd22d365d25709718c1d490ffb0326dec on an RTX 5090
```

On an RTX 3090 the test stream takes 58.0 s with 3.4 GB peak VRAM, within the limits of 90 s and 10 GB. On a quiet RTX 5090 it takes 24.8 s.

| Submission | What it adds | Val (%) | Kaggle public |
|---|---|---:|---:|
| `mixtta_lfsq4` (final) | per-stage rates ×(4, 4, 4, 1, 0.1) instead | 95.93 | **0.94319** |
| `mixtta_lfsq` | + adaptive LAME temperature | 95.27 | 0.94042 |
| `mixtta_lfs` | + per-stage rates ×(2, 2, 2, 1, 0.25) | 95.31 | 0.93815 |
| `mixtta_local_fixed` | `mixtta_ms224` + local class prior | 95.12 | 0.92789 |
| `mixtta_repick` | val-only re-pick, all 7 stages | 95.08 | 0.92675 |
| `mixtta_repick_s5` | re-pick stages 1–5 | 94.74 | 0.92220 |
| `mixtta_ms224` | re-pick stage 1 | 93.99 | 0.91585 |
| `mixtta_repick_floor` | interim: 2 steps, T = 1 | 94.03 | 0.91227 |
| `mixtta_r8` / `mixtta_local` | local class prior with shift reset: collapsed | 96.69 / 96.07 | 0.82682 / 0.68554 |

### Reproduce the validation result

`mixtta_local_fixed` adds a local class prior to the fixed-tracking `mixtta_ms224` baseline. It scores **95.12%** on official validation, up from **93.99%**. Run from the repository root with only `data/val/` and the source checkpoint.

```bash
python starter/run.py --method mixtta_local_fixed --split val --data data --ckpt source_resnet50_160.pth --out out/mixtta_local_fixed
```

The predictions and run log are in [out/mixtta_local_fixed](out/mixtta_local_fixed). The earlier `mixtta_local` experiment reached **96.07%** with the stage-5 shift detector. Its design history matters under the slide 21 FAQ. [Experiment details and provenance](docs/local_prior_experiment.md)

### Generate the submission file for the earlier preset

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
