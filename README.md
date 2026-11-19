# CVPDL HW2: test-time adaptation

The [assignment slides](CVPDL%202026%20hw2%20slides.pptx.pdf) cover grading and submission on slides 15–17 and the adaptation rules on slides 19–21. The [starter instructions](HW2.md) describe the stream format. Results are in [report.md](report.md).

## Setup

Python 3.10 and a CUDA GPU with bf16 support (Ampere or newer). The pinned versions in [requirements.txt](requirements.txt) are the ones every run here used.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129
```

Unzip the dataset into `data/` (from `ntu-cvpdl-2026-hw-2.zip`) and put `source_resnet50_160.pth` in the repo root.

### Generate the submitted file (`mixtta_repick`, Kaggle public 0.92675)

Every setting of `mixtta_repick` was chosen on val alone by [tools/dev/repick.py](tools/dev/repick.py); see [report.md](report.md#val-only-re-pick-all-evaluations).

```bash
METHOD=mixtta_repick
cd starter
python run.py --method $METHOD --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/$METHOD
cd ..
python tools/check_submission.py out/$METHOD     # format, run-log hash, time and VRAM checks
sha256sum out/$METHOD/test_pred.csv              # a6ab4f7074a81a692cef0af7dcc94e5dcaa63cb9b01a1f7d90c747b6e6b58e23 on an RTX 5090
```

On a quiet RTX 5090 the test stream takes 25.1 s, with 4.1 GB peak VRAM. The limit is 90 s on an RTX 3090, which has not been measured.

| Val-only submission | Re-pick stages | Val (%) | Kaggle public |
|---|---|---:|---:|
| `mixtta_repick` (submitted) | all 7 | 95.08 | **0.92675** |
| `mixtta_repick_s5` | 1–5 (no per-stage learning rates) | 94.74 | 0.92220 |
| `mixtta_ms224` | 1 | 93.99 | 0.91585 |
| `mixtta_repick_floor` | interim: 2 steps, T = 1 | 94.03 | 0.91227 |

### Reproduce the validation result

`mixtta_local_fixed` adds a local class prior to the fixed-tracking `mixtta_ms224` baseline. It scores **95.12%** on official validation, up from **93.99%**. Run from the repository root with only `data/val/` and the source checkpoint.

```bash
python starter/run.py --method mixtta_local_fixed --split val --data data --ckpt source_resnet50_160.pth --out out/mixtta_local_fixed
```

The predictions and run log are in [out/mixtta_local_fixed](out/mixtta_local_fixed). The earlier `mixtta_local` experiment reached **96.07%** with the stage-5 shift detector. Its design history matters under the slide 21 FAQ. [Experiment details and provenance](report.md#local-class-prior-validation-experiment)

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
