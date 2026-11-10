"""Validate a run directory (test_pred.csv + test_run_log.json) against the HW2 rules.

  python tools/check_submission.py out/rotta

Checks: CSV format (header id,label, 12,288 rows, ids == test stream order, labels int 0..19),
run log (test split, sha256 of the CSV matches the harness-written hash, VRAM <= 10 GB, time),
and that starter/harness.py is byte-identical to the official one. Exit code 1 on any error.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS_SHA256 = "405992f868962e5c9643f4a89d3e6995112a942a0a8c86a655b249ddf0e48928"  # official harness.py
N_TEST, N_CLASSES, TIME_LIMIT_S, VRAM_LIMIT_MIB = 12288, 20, 90, 10 * 1024


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(run_dir, data=ROOT / "data", code=ROOT / "starter", verbose=True):
    """Return (errors, warnings, log)."""
    run_dir, errors, warnings = Path(run_dir), [], []
    pred, log_path = run_dir / "test_pred.csv", run_dir / "test_run_log.json"
    for p in (pred, log_path):
        if not p.exists():
            errors.append(f"missing {p}")
    if errors:
        return errors, warnings, None

    # --- CSV format ---
    with open(pred, newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["id", "label"]:
        errors.append(f"header must be 'id,label', got {rows[0]}")
    body = rows[1:]
    if len(body) != N_TEST:
        errors.append(f"expected {N_TEST} rows, got {len(body)}")
    labels = []
    for i, r in enumerate(body):
        try:
            v = int(r[1])
            assert 0 <= v < N_CLASSES and str(v) == r[1]
            labels.append(v)
        except Exception:
            errors.append(f"row {i + 2}: bad label {r!r}")
            break
    stream = data / "test" / "stream.csv"
    if stream.exists():
        ids = [line.strip() for line in stream.read_text().splitlines()[1:] if line.strip()]
        if [r[0] for r in body] != ids:
            errors.append("ids do not match data/test/stream.csv (order or content)")
    else:
        warnings.append(f"{stream} not found, skipped id check")

    # --- run log ---
    log = json.loads(log_path.read_text())
    if log.get("split") != "test" or log.get("n") != N_TEST:
        errors.append(f"run log is not a full test run: split={log.get('split')} n={log.get('n')}")
    csv_bytes = pred.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(csv_bytes).hexdigest() != log.get("pred_sha256"):
        errors.append("test_pred.csv does not match pred_sha256 in the run log (edited or from another run)")
    if not log.get("vram_ok"):
        errors.append(f"VRAM check failed: peak {log.get('peak_vram_mib')} MiB (limit {VRAM_LIMIT_MIB})")
    secs, gpu = log.get("seconds", 0), log.get("gpu", "?")
    if secs > TIME_LIMIT_S:
        msg = f"run took {secs}s > {TIME_LIMIT_S}s on {gpu}"
        (errors if "3090" in gpu else warnings).append(msg + ("" if "3090" in gpu else
                                                              " (not a 3090, compare with care)"))
    elif "3090" not in gpu and secs > 0.6 * TIME_LIMIT_S:
        warnings.append(f"run took {secs}s on {gpu}; limit is {TIME_LIMIT_S}s on an RTX 3090, leave margin")

    # --- code ---
    h = code / "harness.py"
    if not h.exists() or sha256(h) != HARNESS_SHA256:
        errors.append(f"{h} is missing or modified (must be the official file)")
    method = log.get("config", {}).get("method")
    if method and f'"{method}"' not in (code / "methods.py").read_text():
        warnings.append(f"method '{method}' not found in METHODS of {code / 'methods.py'}")

    # --- sanity: prediction distribution (test stream is class-balanced overall) ---
    if labels and len(labels) == N_TEST:
        cnt = Counter(labels)
        top, n_top = cnt.most_common(1)[0]
        if n_top > 3 * N_TEST / N_CLASSES or len(cnt) < N_CLASSES:
            warnings.append(f"skewed predictions: class {top} has {n_top} preds, {len(cnt)}/20 classes used")

    if verbose:
        val = run_dir / "val_run_log.json"
        val_acc = json.loads(val.read_text()).get("acc") if val.exists() else None
        print(f"run      : {run_dir}")
        print(f"config   : {log.get('config')}")
        print(f"gpu      : {gpu}, torch {log.get('torch')}")
        print(f"time     : {secs}s (limit {TIME_LIMIT_S}s on 3090)   peak VRAM: {log.get('peak_vram_mib')} MiB")
        if val_acc is not None:
            print(f"val acc  : {val_acc:.4f}")
        for w in warnings:
            print(f"WARNING  : {w}")
        for e in errors:
            print(f"ERROR    : {e}")
        print("RESULT   : " + ("FAIL" if errors else "OK"))
    return errors, warnings, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing test_pred.csv and test_run_log.json")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--code", default=str(ROOT / "starter"))
    a = ap.parse_args()
    errors, _, _ = check(a.run_dir, Path(a.data), Path(a.code))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
