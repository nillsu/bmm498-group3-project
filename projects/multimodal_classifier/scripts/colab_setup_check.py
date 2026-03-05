"""
Environment and data setup checker for Colab / local runs.

Run this BEFORE training to catch configuration problems early.

Usage:
    python colab_setup_check.py --csv_path <path> --data_root <path>
    python colab_setup_check.py --csv_path <path> --data_root <path> --fold_mode
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    ("pytorch_lightning", "pytorch-lightning"),
    ("timm",             "timm"),
    ("torchmetrics",     "torchmetrics"),
    ("pandas",           "pandas"),
    ("PIL",              "pillow"),
    ("torchvision",      "torchvision"),
]


def check_env() -> None:
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"Python  : {sys.version}")
    try:
        import torch
        print(f"PyTorch : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA    : {cuda_ok}")
        if cuda_ok:
            n = torch.cuda.device_count()
            print(f"GPUs    : {n}")
            for i in range(n):
                print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        else:
            print("  WARNING: No GPU detected — training will be very slow on CPU.")
    except ImportError:
        print("ERROR: torch is not installed.")
    print()


def check_packages() -> None:
    print("=" * 60)
    print("PACKAGE CHECK")
    print("=" * 60)
    missing_pip = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"  OK      : {import_name}")
        except ImportError:
            print(f"  MISSING : {import_name}")
            missing_pip.append(pip_name)
    if missing_pip:
        print(f"\nTo install missing packages run:")
        print(f"  pip install {' '.join(missing_pip)}")
    print()


def check_data(data_root: str, csv_path: str, fold_mode: bool) -> None:
    print("=" * 60)
    print("DATA CHECK")
    print("=" * 60)

    dr = Path(data_root)
    if not dr.exists():
        print(f"  ERROR   : data_root not found: {dr}")
        print("  Hint    : On Colab, mount Drive first:")
        print("            from google.colab import drive; drive.mount('/content/drive')")
    else:
        try:
            entries = sum(1 for _ in dr.iterdir())
            print(f"  OK      : data_root exists ({entries} entries)")
        except PermissionError:
            print(f"  ERROR   : data_root exists but is not readable: {dr}")

    csv = Path(csv_path)
    if not csv.exists():
        print(f"  ERROR   : CSV not found: {csv}")
        print("  Hint    : Check path; ensure Drive is mounted if on Colab.")
        print()
        return

    print(f"  OK      : CSV found: {csv}")

    try:
        import pandas as pd
        df = pd.read_csv(csv)
        print(f"  Rows    : {len(df)}")
        print(f"  Columns : {list(df.columns)}")

        required_base = {"sample_id", "fundus_rel", "oct_rel", "DR_pos", "DME"}
        required = required_base | (set() if fold_mode else {"split"})
        missing_cols = required - set(df.columns)
        if missing_cols:
            print(f"  ERROR   : Missing required columns: {sorted(missing_cols)}")
        else:
            print(f"  OK      : All required columns present")

        print("\n  First 3 rows:")
        print(df.head(3).to_string(index=False))

        if "split" in df.columns:
            print("\n  Split counts:")
            for split_val, count in df["split"].value_counts().items():
                print(f"    {split_val:<8}: {count}")

    except Exception as exc:
        print(f"  ERROR   : Could not read CSV — {exc}")

    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Environment and data setup checker")
    p.add_argument("--csv_path",  required=True,
                   help="Path to splits_clean.csv or any fold CSV")
    p.add_argument("--data_root", required=True,
                   help="Root directory for image data")
    p.add_argument("--fold_mode", action="store_true",
                   help="If set, do not require the 'split' column (fold-CSV mode)")
    args = p.parse_args()

    check_env()
    check_packages()
    check_data(args.data_root, args.csv_path, args.fold_mode)
    print("Setup check complete.")


if __name__ == "__main__":
    main()
