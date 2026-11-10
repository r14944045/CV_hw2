#!/usr/bin/env bash
# Run one method on val (accuracy) and test (Kaggle file) on a CUDA machine, then validate.
#   bash tools/run_gpu.sh rotta            # -> out/rotta/{val,test}_pred.csv + run logs
#   bash tools/run_gpu.sh rotta --seed 1   # extra args are passed to run.py
set -euo pipefail
METHOD=${1:?usage: bash tools/run_gpu.sh <method> [run.py args]}; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/starter"
for SPLIT in val test; do
  python run.py --method "$METHOD" --split "$SPLIT" --data ../data --ckpt ../source_resnet50_160.pth \
    --out "../out/$METHOD" "$@"
done
python "$ROOT/tools/check_submission.py" "$ROOT/out/$METHOD"
echo
echo "Submit to Kaggle: upload out/$METHOD/test_pred.csv (or: kaggle competitions submit -c <competition> -f out/$METHOD/test_pred.csv -m \"$METHOD\")"
