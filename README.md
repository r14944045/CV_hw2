# CVPDL HW2 — practical test-time adaptation

The latest CSV uses `mixtta_local_calibrated`. It got **2,025 of 2,112 correct (95.88%)** on official validation, up from 2,013 (95.31%) for the previous smoothed model. Its mean over 56 validation runs rose from **94.84% to 95.22%**. It estimates a local class prior from neighboring predictions and calibrates class scores online. The original fixed-tracking submission scored **0.92789** on Kaggle. The latest CSV awaits a Kaggle score. [Method and validation results](report.md)

## Submission files

| Preset | Prediction CSV | Official run log |
|---|---|---|
| Latest submission | [Download CSV](out/mixtta_local_calibrated/test_pred.csv) | [Run log](out/mixtta_local_calibrated/test_run_log.json) |
| Previous smoothed model | [Download CSV](out/mixtta_local_smooth192/test_pred.csv) | [Run log](out/mixtta_local_smooth192/test_run_log.json) |
| Previous submission, public 0.92789 | [Scored CSV](out/mixtta_local_fixed/test_pred.csv) | [Run log](out/mixtta_local_fixed/test_run_log.json) |

The checker verified all 12,288 predictions. The single test pass took 84.5 seconds and peaked at 5,374 MiB on a shared RTX 5090. Runtime on the grading RTX 3090 is unverified.

## Reproduce the CSV

Use Python 3.10 and a CUDA GPU with bf16 support. Place the provided `source_resnet50_160.pth` and extracted `data/` in the repository root. Install the versions in [requirements.txt](requirements.txt).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
```

Run from the repository root. The first command makes one online pass through the official harness. The checker verifies the prediction hash against the run log.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python starter/run.py --method mixtta_local_calibrated --seed 0 --split test --data data --ckpt source_resnet50_160.pth --out out/mixtta_local_calibrated
python tools/check_submission.py out/mixtta_local_calibrated
```

## Reproduce the validation results

```bash
python starter/run.py --method mixtta_local_calibrated --seed 0 --split val --data data --ckpt source_resnet50_160.pth --out out/mixtta_local_calibrated
python tools/plot_val_examples.py --method mixtta_local_calibrated
```

The longer validation protocol evaluates seven stream orders, four offsets, and two seeds, followed by a 231-batch collapse check.

```bash
python tools/dev/cache.py
python tools/dev/repick.py method mixtta_local_calibrated
python tools/dev/val_controls.py mixtta_local_smooth192 mixtta_local_readout mixtta_local_calibrated
python tools/dev/val_controls.py mixtta_local_smooth192 mixtta_local_readout mixtta_local_calibrated --controls low_resolution
```

The course slides (15–21) specify the grading, submission contents, and online adaptation rules. The full test stream has a hard limit of 90 seconds and 10 GB of GPU memory on the TAs’ RTX 3090. [Starter instructions](HW2.md)
