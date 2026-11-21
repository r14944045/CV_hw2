# CVPDL HW2: reproduce the submitted `test_pred.csv`

Final method `mixtta_lfsq4` (Kaggle public 0.94319). Python 3.10, CUDA GPU with bf16 (Ampere or newer).

```bash
# environment (the versions used for every run)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129

# inputs: the Kaggle dataset in ./data (val/, test/) and the checkpoint at ./source_resnet50_160.pth

# test stream, one online pass through the official harness -> out/mixtta_lfsq4/test_pred.csv
# (RTX 3090: 58.0 s and 82.4 s in two runs, 3.4 GB peak VRAM)
cd starter
python run.py --method mixtta_lfsq4 --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/mixtta_lfsq4

# optional: validation accuracy (95.93 %)
python run.py --method mixtta_lfsq4 --split val --data ../data --ckpt ../source_resnet50_160.pth --out ../out/mixtta_lfsq4
cd ..

# optional: format / run-log hash / VRAM / time checks and the file hash
python tools/check_submission.py out/mixtta_lfsq4
sha256sum out/mixtta_lfsq4/test_pred.csv  # 48c04860a92e2eae3ff4b3858876df7bd22d365d25709718c1d490ffb0326dec (RTX 5090)
# RTX 3090: 083d2c23ffc3e83ad6a4b54148ea02387d8b76f19e4db6a5c23080a736c4131a; bf16 kernels differ between GPU
# generations, so 88 of 12,288 predictions (0.72 %) differ from the RTX 5090 file

# NTU COOL zip (needs hw2_report_R14944045.pdf in the repo root)
python tools/package.py out/mixtta_lfsq4 --id R14944045
```
