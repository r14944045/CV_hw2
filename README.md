# CVPDL HW2 — practical test-time adaptation

The reported Kaggle public score for `mixtta_local_fixed` is **0.92789**. It classified **2,009 of 2,112 validation images correctly (95.12%)**, 24 more than the fixed-tracking MixTTA-MS baseline. It uses the supplied ResNet-50 and estimates a local class prior from the predictions of the current and preceding batches. [Method, results, and validation examples](report.md)

## Submission files

- [Kaggle prediction CSV](out/mixtta_local_fixed/test_pred.csv)
- [Official harness run log](out/mixtta_local_fixed/test_run_log.json)

The checker verified all 12,288 predictions against the run log. The single test pass took 25.9 seconds and peaked at 4,150 MiB on an RTX 5090.

## Reproduce the CSV

Use Python 3.10 and a CUDA GPU with bf16 support. Place the provided `source_resnet50_160.pth` and extracted `data/` in the repository root. Install the versions in [requirements.txt](requirements.txt).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
```

Run from the repository root. The first command makes one online pass through the official harness. The checker verifies the prediction hash against the run log.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python starter/run.py --method mixtta_local_fixed --seed 0 --split test --data data --ckpt source_resnet50_160.pth --out out/mixtta_local_fixed
python tools/check_submission.py out/mixtta_local_fixed
```

## Reproduce the validation results

```bash
python starter/run.py --method mixtta_local_fixed --seed 0 --split val --data data --ckpt source_resnet50_160.pth --out out/mixtta_local_fixed
python tools/plot_val_examples.py --method mixtta_local_fixed
```

The longer validation protocol evaluates seven stream orders, four offsets, and two seeds, followed by a 231-batch collapse check.

```bash
python tools/dev/cache.py
python tools/dev/repick.py method mixtta_local_fixed
```

The course slides (15–21) specify the grading, submission contents, and online adaptation rules. The full test stream has a hard limit of 90 seconds and 10 GB of GPU memory on the TAs’ RTX 3090. [Starter instructions](HW2.md)
