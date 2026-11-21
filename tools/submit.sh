#!/usr/bin/env bash
# Submit a harness test run to Kaggle after the format / hash / VRAM / time checks.
#   bash tools/submit.sh mixtta_valsel "val-only selection, final"
# Needs the Kaggle CLI (pip install kaggle) and an API token in ~/.kaggle/kaggle.json (chmod 600).
set -euo pipefail
METHOD=${1:?usage: bash tools/submit.sh <method> "<message>"}
MSG=${2:-$METHOD}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FILE="$ROOT/out/$METHOD/test_pred.csv"
python "$ROOT/tools/check_submission.py" "$ROOT/out/$METHOD"
echo "SHA-256: $(sha256sum "$FILE" | cut -d' ' -f1)"
kaggle competitions submit -c ntu-cvpdl-2026-hw-2 -f "$FILE" -m "$METHOD: $MSG"
