"""
make_pseudo_folds.py
--------------------
Filter existing k-fold split CSVs using the pseudo-available pool so that
pseudo experiments use the SAME fold assignments as the real-OCT experiments.

Operation
---------
For each fold_X/train.csv and fold_X/val.csv (and the shared test.csv):
  1. Build a lookup set of sample_ids from the fold CSV.
  2. Filter the pseudo-available CSV to keep only those sample_ids.
  3. Write the result to splits_pseudo/fold_X/{train,val,test}.csv.

The output rows carry all columns from the pseudo-available CSVs, which
includes oct_pseudo_rel and pseudo_exists — required by pseudo_oct and
fusion_pseudo training modes.

Match key
---------
  sample_id  (derived as patient_id + "_" + eye if absent from either CSV)

Pseudo-available pools
----------------------
  Fold train/val rows originate from the combined original train+val pool,
  so both pairs_train_pseudo_available.csv AND pairs_val_pseudo_available.csv
  are unioned into a single lookup for fold train and fold val CSVs.

  Test rows come only from the original test split, so only
  pairs_test_pseudo_available.csv is used for test.

Usage (Colab)
-------------
  !python make_pseudo_folds.py \\
      --splits_root    splits \\
      --pseudo_dir     dataset_csv \\
      --output_root    splits_pseudo

Run k-fold pseudo_oct / fusion_pseudo using the generated pseudo splits:
  !python projects/multimodal_classifier/scripts/run_kfold_experiments.py \\
      --data_root   /content/bmm498_data \\
      --splits_root splits_pseudo \\
      --output_root runs/kfold_pseudo \\
      --epochs 20 --batch_size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add sample_id column if absent (patient_id + '_' + eye)."""
    if "sample_id" not in df.columns:
        if "patient_id" not in df.columns or "eye" not in df.columns:
            raise ValueError(
                "Cannot derive sample_id: DataFrame is missing 'patient_id' and/or 'eye' columns.\n"
                f"  Found columns: {list(df.columns)}"
            )
        df = df.copy()
        df["sample_id"] = df["patient_id"].astype(str) + "_" + df["eye"].astype(str)
    return df


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Required CSV not found ({label}): {path}\n"
            f"  Ensure add_pseudo_oct_column.py has been run against the data_root first."
        )
    df = pd.read_csv(path)
    df = _derive_sample_id(df)
    return df


def _filter_pseudo(fold_csv: pd.DataFrame, pseudo_pool: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep rows from pseudo_pool whose sample_id appears in fold_csv."""
    fold_ids = set(fold_csv["sample_id"])
    result = pseudo_pool[pseudo_pool["sample_id"].isin(fold_ids)].reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pseudo-fold CSVs by filtering existing folds with pseudo-available rows."
    )
    p.add_argument(
        "--splits_root", default="splits",
        help="Directory containing fold_0..N sub-directories and test.csv. (default: splits)"
    )
    p.add_argument(
        "--pseudo_dir", default="dataset_csv",
        help=(
            "Directory containing pairs_{train,val,test}_pseudo_available.csv. "
            "(default: dataset_csv)"
        ),
    )
    p.add_argument(
        "--output_root", default="splits_pseudo",
        help="Output directory for pseudo fold CSVs. (default: splits_pseudo)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits_root  = Path(args.splits_root)
    pseudo_dir   = Path(args.pseudo_dir)
    output_root  = Path(args.output_root)

    # ------------------------------------------------------------------
    # 1. Validate splits_root
    # ------------------------------------------------------------------
    if not splits_root.exists():
        print(f"[ERROR] splits_root not found: {splits_root}", file=sys.stderr)
        sys.exit(1)

    fold_dirs = sorted(
        p for p in splits_root.iterdir()
        if p.is_dir() and p.name.startswith("fold_")
    )
    if not fold_dirs:
        print(
            f"[ERROR] No fold_* directories found in {splits_root}.\n"
            f"  Expected structure: {splits_root}/fold_0/train.csv, val.csv ...",
            file=sys.stderr,
        )
        sys.exit(1)

    test_csv_path = splits_root / "test.csv"
    has_test = test_csv_path.exists()

    # ------------------------------------------------------------------
    # 2. Load pseudo-available CSVs
    # ------------------------------------------------------------------
    pseudo_train_path = pseudo_dir / "pairs_train_pseudo_available.csv"
    pseudo_val_path   = pseudo_dir / "pairs_val_pseudo_available.csv"
    pseudo_test_path  = pseudo_dir / "pairs_test_pseudo_available.csv"

    print("=" * 65)
    print("  make_pseudo_folds.py")
    print("=" * 65)
    print(f"\nLoading pseudo-available CSVs from: {pseudo_dir}")

    pseudo_train = _load_csv(pseudo_train_path, "pairs_train_pseudo_available")
    print(f"  pairs_train_pseudo_available : {len(pseudo_train):>5} rows")

    pseudo_val = _load_csv(pseudo_val_path, "pairs_val_pseudo_available")
    print(f"  pairs_val_pseudo_available   : {len(pseudo_val):>5} rows")

    pseudo_test = _load_csv(pseudo_test_path, "pairs_test_pseudo_available")
    print(f"  pairs_test_pseudo_available  : {len(pseudo_test):>5} rows")

    # The fold pool covers the original train+val universe
    pseudo_pool_trainval = pd.concat(
        [pseudo_train, pseudo_val], ignore_index=True
    )
    # Sanity: no duplicate sample_ids in pool (warn, don't fail)
    dup_ids = pseudo_pool_trainval[
        pseudo_pool_trainval.duplicated("sample_id", keep=False)
    ]["sample_id"].unique()
    if len(dup_ids) > 0:
        print(
            f"\n[WARNING] {len(dup_ids)} duplicate sample_id(s) in combined "
            f"train+val pseudo pool. First 5: {list(dup_ids[:5])}"
        )

    # Verify oct_pseudo_rel is present (required by pseudo modes)
    for label, df in [
        ("pairs_train_pseudo_available", pseudo_train),
        ("pairs_val_pseudo_available",   pseudo_val),
        ("pairs_test_pseudo_available",  pseudo_test),
    ]:
        if "oct_pseudo_rel" not in df.columns:
            print(
                f"[ERROR] 'oct_pseudo_rel' column missing from {label}.\n"
                f"  Run add_pseudo_oct_column.py with --data_root to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Process each fold
    # ------------------------------------------------------------------
    print(f"\nProcessing {len(fold_dirs)} fold(s): {[d.name for d in fold_dirs]}")
    print(f"Output root: {output_root}\n")

    summary_rows: list[dict] = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        out_fold_dir = output_root / fold_name
        out_fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {'─' * 55}")
        print(f"  {fold_name}")
        print(f"  {'─' * 55}")

        for split_name in ("train", "val"):
            src_path = fold_dir / f"{split_name}.csv"
            if not src_path.exists():
                print(
                    f"[ERROR] Expected fold file not found: {src_path}",
                    file=sys.stderr,
                )
                sys.exit(1)

            fold_df = _load_csv(src_path, f"{fold_name}/{split_name}")

            n_original = len(fold_df)
            pseudo_filtered = _filter_pseudo(fold_df, pseudo_pool_trainval, split_name)
            n_retained = len(pseudo_filtered)
            n_dropped  = n_original - n_retained

            out_path = out_fold_dir / f"{split_name}.csv"
            pseudo_filtered.to_csv(out_path, index=False)

            print(
                f"    {split_name:5s}: original={n_original:>4}  "
                f"retained={n_retained:>4}  dropped={n_dropped:>4}  -> {out_path}"
            )
            summary_rows.append({
                "fold": fold_name, "split": split_name,
                "original": n_original, "retained": n_retained, "dropped": n_dropped,
            })

    # ------------------------------------------------------------------
    # 4. Process shared test set
    # ------------------------------------------------------------------
    print(f"\n  {'─' * 55}")
    print("  test  (shared across all folds)")
    print(f"  {'─' * 55}")

    if has_test:
        test_df = _load_csv(test_csv_path, "test")
        n_original = len(test_df)
        pseudo_filtered_test = _filter_pseudo(test_df, pseudo_test, "test")
        n_retained = len(pseudo_filtered_test)
        n_dropped  = n_original - n_retained

        out_test_path = output_root / "test.csv"
        pseudo_filtered_test.to_csv(out_test_path, index=False)

        print(
            f"    test : original={n_original:>4}  "
            f"retained={n_retained:>4}  dropped={n_dropped:>4}  -> {out_test_path}"
        )
        # Also copy into each fold dir so run_kfold can find it via splits_pseudo/test.csv
        for fold_dir in fold_dirs:
            fold_test_out = output_root / fold_dir.name / "test.csv"
            pseudo_filtered_test.to_csv(fold_test_out, index=False)

        summary_rows.append({
            "fold": "shared", "split": "test",
            "original": n_original, "retained": n_retained, "dropped": n_dropped,
        })
    else:
        print(f"    [WARNING] No test.csv found at {test_csv_path} — skipping test split.")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'fold':<12} {'split':<6} {'original':>8} {'retained':>8} {'dropped':>8}")
    print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
    for row in summary_rows:
        print(
            f"  {row['fold']:<12} {row['split']:<6} "
            f"{row['original']:>8} {row['retained']:>8} {row['dropped']:>8}"
        )

    total_orig = sum(r["original"] for r in summary_rows if r["split"] != "test")
    total_ret  = sum(r["retained"] for r in summary_rows if r["split"] != "test")
    total_drop = sum(r["dropped"]  for r in summary_rows if r["split"] != "test")
    print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'TOTAL (train+val)':<19} {total_orig:>8} {total_ret:>8} {total_drop:>8}")
    print()

    if total_drop > 0:
        pct = total_drop / total_orig * 100
        print(f"  {total_drop} rows dropped ({pct:.1f}%) — samples without a pseudo-OCT image.")
    else:
        print("  All rows retained — every sample has a pseudo-OCT image.")

    print(f"\n  Pseudo fold CSVs written to: {output_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()
