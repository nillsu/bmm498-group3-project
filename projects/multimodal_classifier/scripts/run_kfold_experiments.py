"""
A7 — K-fold experiment runner.

Detects fold_* directories inside --splits_root, runs the full ablation
(fundus / oct / fusion) for each fold via run_ablation.py, then writes a
results_summary.csv with one row per (fold, mode).

Usage:
    python run_kfold_experiments.py \
        --data_root  ./bmm498_data \
        --splits_root ./splits \
        --output_root ./runs/kfold_experiment \
        --epochs 20 --batch_size 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODES = ["fundus", "oct", "fusion"]
_ABLATION_SCRIPT = Path(__file__).resolve().parent / "run_ablation.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-fold ablation experiment runner")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--folds",         default="all",
                   help="'all' or comma-separated fold indices, e.g. '0,2,4'.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Passed through to run_ablation.py.")
    p.add_argument("--dry_run",       action="store_true",
                   help="Passed through to run_ablation.py.")
    return p.parse_args()


def _detect_folds(splits_root: Path) -> list[Path]:
    folds = sorted(
        p for p in splits_root.iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    )
    if not folds:
        raise FileNotFoundError(
            f"No fold_* directories found in {splits_root}. "
            "Expected structure: splits_root/fold_0/train.csv, val.csv ..."
        )
    # Safety check: each fold must have train.csv and val.csv
    invalid = [f for f in folds
               if not (f / "train.csv").exists() or not (f / "val.csv").exists()]
    if invalid:
        raise FileNotFoundError(
            f"The following folds are missing train.csv or val.csv: "
            f"{[f.name for f in invalid]}"
        )
    return folds


def _build_ablation_cmd(args: argparse.Namespace, fold_dir: Path, output_dir: Path) -> list[str]:
    train_csv = fold_dir / "train.csv"
    val_csv   = fold_dir / "val.csv"
    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(f"Expected fold file not found: {p}")
    cmd = [
        sys.executable, str(_ABLATION_SCRIPT),
        "--data_root",   args.data_root,
        "--output_root", str(output_dir),
        "--train_csv",   str(train_csv),
        "--val_csv",     str(val_csv),
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--image_size",  str(args.image_size),
        "--num_workers", str(args.num_workers),
        "--seed",        str(args.seed),
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def _find_best_ckpt(mode_dir: Path) -> str:
    ckpts = list((mode_dir / "checkpoints").glob("best-*.ckpt"))
    if not ckpts:
        return "not found"
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1])


def main() -> None:
    args = parse_args()
    splits_root = Path(args.splits_root)
    output_root = Path(args.output_root)

    all_folds = _detect_folds(splits_root)

    # Filter folds if --folds was specified
    if args.folds == "all":
        folds = all_folds
    else:
        requested = {f"fold_{i.strip()}" for i in args.folds.split(",")}
        folds = [f for f in all_folds if f.name in requested]
        if not folds:
            raise ValueError(
                f"No folds matched --folds={args.folds!r}. "
                f"Available: {[f.name for f in all_folds]}"
            )

    print(f"Running {len(folds)} fold(s): {[f.name for f in folds]}\n")

    rows: list[dict] = []

    for fold_dir in folds:
        fold_name  = fold_dir.name
        output_dir = output_root / fold_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"FOLD {fold_name}  →  {output_dir}")
        print(f"{'#' * 60}\n")

        fold_status = "ok"
        try:
            subprocess.run(
                _build_ablation_cmd(args, fold_dir, output_dir), check=True
            )
            if args.dry_run:
                fold_status = "dry_run"
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: fold {fold_name} failed (exit code {exc.returncode}). Continuing.")
            fold_status = "failed"

        for mode in MODES:
            mode_dir = output_dir / mode
            ckpt = _find_best_ckpt(mode_dir)
            status = fold_status if fold_status != "ok" else ("ok" if ckpt != "not found" else "no_ckpt")
            rows.append({
                "fold":       fold_name,
                "mode":       mode,
                "checkpoint": ckpt,
                "output_dir": str(mode_dir),
                "status":     status,
            })
            print(f"  {fold_name}/{mode}: {status}  {ckpt}")

    summary_path = output_root / "results_summary.csv"
    pd.DataFrame(rows, columns=["fold", "mode", "checkpoint", "output_dir", "status"]).to_csv(
        summary_path, index=False
    )
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
