"""
A6 — Ablation runner.

Trains fundus / oct / fusion sequentially for the same fold, each in its own
output sub-directory.  Delegates to train.py via subprocess so no logic is
duplicated and training output streams live to the terminal in real time.

Usage:
    python run_ablation.py --data_root <path> --output_root <path> [options]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODES = ["fundus", "oct", "fusion"]
_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation runner: fundus / oct / fusion")
    p.add_argument("--data_root",    required=True)
    p.add_argument("--output_root",  required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--epochs",       type=int, default=10)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--image_size",   type=int, default=224)
    p.add_argument("--num_workers",  type=int, default=4)
    # k-fold inputs
    p.add_argument("--train_csv",    default=None)
    p.add_argument("--val_csv",      default=None)
    p.add_argument("--test_csv",     default=None)
    # single-CSV fallback
    p.add_argument("--csv_path",     default=None)
    # run control
    # Default False: always retrain. Pass --skip_existing to resume interrupted runs.
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip modes that already have a best-*.ckpt in checkpoints/.")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print commands without executing them.")
    return p.parse_args()


def _build_cmd(args: argparse.Namespace, mode: str, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(_TRAIN_SCRIPT),
        "--data_root",  args.data_root,
        "--output_dir", str(run_dir),
        "--mode",       mode,
        "--seed",       str(args.seed),
        "--epochs",     str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--image_size", str(args.image_size),
        "--num_workers", str(args.num_workers),
    ]
    if args.train_csv and args.val_csv:
        cmd += ["--train_csv", args.train_csv, "--val_csv", args.val_csv]
    if args.test_csv:
        cmd += ["--test_csv", args.test_csv]
    if args.csv_path:
        cmd += ["--csv_path", args.csv_path]
    return cmd


def _find_best_ckpt(run_dir: Path) -> str:
    ckpts = list((run_dir / "checkpoints").glob("best-*.ckpt"))
    if not ckpts:
        return "not found"
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1])


def _find_log_dir(run_dir: Path) -> str:
    # Prefer version_* subdirs (CSVLogger convention); fall back to any subdir.
    log_dirs = list((run_dir / "logs").glob("version_*/"))
    if not log_dirs:
        log_dirs = list((run_dir / "logs").glob("*/"))
    if not log_dirs:
        return "not found"
    log_dirs.sort(key=lambda p: p.stat().st_mtime)
    return str(log_dirs[-1])


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    # Fast-fail: require at least one CSV source
    if not ((args.train_csv and args.val_csv) or args.csv_path):
        raise ValueError(
            "Must provide either --csv_path (single-CSV mode) "
            "or both --train_csv and --val_csv (fold mode)."
        )

    for mode in MODES:
        run_dir = output_root / mode
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and _find_best_ckpt(run_dir) != "not found":
            print(f"SKIP mode={mode}  (checkpoint exists in {run_dir / 'checkpoints'})")
            continue

        cmd = _build_cmd(args, mode, run_dir)
        print(f"\n{'=' * 60}")
        print(f"Starting  mode={mode}  output_dir={run_dir}")
        print(f"{'=' * 60}\n")

        if args.dry_run:
            print(f"DRY-RUN: {' '.join(cmd)}")
        else:
            subprocess.run(cmd, check=True)
            best_ckpt = _find_best_ckpt(run_dir)
            log_dir   = _find_log_dir(run_dir)
            print(f"\nDONE mode={mode} best_ckpt={best_ckpt} logs={log_dir}")

    print(f"\nAll ablation runs complete. Results in: {output_root}")


if __name__ == "__main__":
    main()
