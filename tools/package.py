"""Build hw2_<student_id>.zip for NTU COOL.

  python tools/package.py out/rotta                 # needs hw2_report_<id>.pdf in the project root
  python tools/package.py out/rotta --draft         # no report yet: builds the zip without it

Zip layout (slide 18):
  hw2_<id>.zip
  ├── test_pred.csv
  ├── test_run_log.json
  ├── Readme.md               generated: exact command to reproduce test_pred.csv
  ├── code/                   the whole starter/ folder (harness.py, run.py, methods.py, ...)
  └── hw2_report_<id>.pdf
The dataset and the source checkpoint are never included.
"""
import argparse
import sys
import zipfile
from pathlib import Path

from check_submission import ROOT, check

SKIP_NAMES = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pth", ".pt", ".ckpt", ".png", ".jpg", ".zip"}
PATH_ARGS = {"data", "ckpt", "out", "split"}


def reproduce_command(config):
    extra = " ".join(f"--{k} {v}" for k, v in config.items() if k not in PATH_ARGS | {"method"})
    return (f"cd code\npython run.py --method {config['method']} --split test "
            f"--data ../data --ckpt ../source_resnet50_160.pth --out ../out {extra}").rstrip()


PINNED = ("torch==2.13.0 torchvision==0.28.0 numpy==2.2.6 pandas==2.3.3 pillow==12.2.0 "
          "--extra-index-url https://download.pytorch.org/whl/cu129")


def readme(student_id, log, val_log):
    cmd = reproduce_command(log["config"])
    return f"""# CVPDL HW2 ({student_id}): reproduce `test_pred.csv`

```bash
# environment: Python 3.10, CUDA GPU with bf16 (Ampere or newer); the versions used for the submitted run
pip install {PINNED}

# inputs next to code/: the Kaggle dataset in ./data (val/, test/) and ./source_resnet50_160.pth

# one online pass through the official harness -> out/test_pred.csv, out/test_run_log.json
{cmd}

# expected: sha256sum ../out/test_pred.csv = {log['pred_sha256']} ({log['gpu']})
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory with test_pred.csv / test_run_log.json (e.g. out/rotta)")
    ap.add_argument("--id", default="R14944045", help="student id")
    ap.add_argument("--report", default=None, help="report pdf (default: hw2_report_<id>.pdf in project root)")
    ap.add_argument("--draft", action="store_true", help="allow building without the report pdf")
    ap.add_argument("--out", default=None, help="zip path (default: hw2_<id>.zip in project root)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    errors, _, log = check(run_dir)
    if errors:
        sys.exit("Not packaging: fix the errors above first.")

    report = Path(a.report) if a.report else ROOT / f"hw2_report_{a.id}.pdf"
    if not report.exists() and not a.draft:
        sys.exit(f"Report not found: {report}  (use --draft to build without it)")

    val_path = run_dir / "val_run_log.json"
    val_log = __import__("json").loads(val_path.read_text()) if val_path.exists() else None

    zip_path = Path(a.out) if a.out else ROOT / f"hw2_{a.id}.zip"
    code_dir = ROOT / "starter"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run_dir / "test_pred.csv", "test_pred.csv")
        z.write(run_dir / "test_run_log.json", "test_run_log.json")
        z.writestr("Readme.md", readme(a.id, log, val_log))
        for f in sorted(code_dir.rglob("*")):
            rel = f.relative_to(code_dir)
            if f.is_file() and not (set(rel.parts) & SKIP_NAMES) and f.suffix not in SKIP_SUFFIXES:
                z.write(f, Path("code") / rel)
        if report.exists():
            z.write(report, f"hw2_report_{a.id}.pdf")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        size = sum(i.file_size for i in z.infolist())
    print(f"\nWrote {zip_path} ({size / 1e6:.2f} MB uncompressed):")
    for n in names:
        print("  " + n)
    if not report.exists():
        print(f"NOTE: report missing ({report.name}); draft zip is NOT ready to submit.")


if __name__ == "__main__":
    main()
