"""
add_pseudo_oct_column.py
Extends existing split CSVs with oct_pseudo_rel + pseudo_exists columns,
then writes a filtered _pseudo_available.csv containing only rows where
the pseudo image was found on disk.

Derivation rule (split-aware)
------------------------------
  train / val : <split>/pseudo-oct/<base_id>_fundus_oct.png
  test        : <split>/pseudo-oct/<base_id>_fake_B.png

  base_id is the real OCT stem with _oct_v2 stripped:
    train/oct_real/1221_OD_000000_oct_v2.jpg  ->  1221_OD_000000

Usage (Colab)
-------------
# Process all three splits, validate against data_root:
!python add_pseudo_oct_column.py \\
    --csv dataset_csv/pairs_train_oct_v2.csv \\
          dataset_csv/pairs_val_oct_v2.csv   \\
          dataset_csv/pairs_test_oct_v2.csv  \\
    --data_root /content/bmm498_data

# Custom output paths:
!python add_pseudo_oct_column.py \\
    --csv dataset_csv/pairs_train_oct_v2.csv \\
    --out dataset_csv/pairs_train_with_pseudo.csv \\
    --data_root /content/bmm498_data

# Derive paths only, no disk check, no files written:
!python add_pseudo_oct_column.py \\
    --csv dataset_csv/pairs_train_oct_v2.csv \\
    --dry_run

Output files (when not --dry_run and --data_root provided)
-----------------------------------------------------------
  pairs_train_with_pseudo.csv          -- all rows + oct_pseudo_rel + pseudo_exists
  pairs_train_pseudo_available.csv     -- only rows where pseudo_exists == True
  (same for val / test)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import pandas as pd


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

def _base_id(oct_rel: str) -> str:
    """Strip suffix from real OCT stem to get the shared base identifier."""
    stem = PurePosixPath(oct_rel.replace("\\", "/")).stem  # e.g. 1221_OD_000000_oct_v2
    if stem.endswith("_oct_v2"):
        return stem[: -len("_oct_v2")]
    # Fallback: drop last underscore-delimited token
    return "_".join(stem.split("_")[:-1])


def derive_pseudo_path(oct_rel: str, split: str) -> str:
    """
    Derive pseudo-OCT relative path from real OCT path and split name.

    train/val: <split>/pseudo-oct/<base_id>_fundus_oct.png
    test     : <split>/pseudo-oct/<base_id>_fake_B.png
    """
    split_dir = PurePosixPath(oct_rel.replace("\\", "/")).parts[0]
    base      = _base_id(oct_rel)
    suffix    = "_fake_B.png" if split == "test" else "_fundus_oct.png"
    return f"{split_dir}/pseudo-oct/{base}{suffix}"


# ---------------------------------------------------------------------------
# Column normalisation (mirrors datamodule._normalize_columns)
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        if "patient_id" in df.columns and "eye" in df.columns:
            df["sample_id"] = df["patient_id"].astype(str) + "_" + df["eye"].astype(str)
    if "fundus_rel" not in df.columns and "fundus_preprocessed" in df.columns:
        df["fundus_rel"] = df["fundus_preprocessed"]
    if "oct_rel" not in df.columns and "oct_preprocessed_v2" in df.columns:
        df["oct_rel"] = df["oct_preprocessed_v2"]
    return df


# ---------------------------------------------------------------------------
# Per-CSV processing
# ---------------------------------------------------------------------------

def process_csv(
    csv_path: Path,
    out_path: Path,
    data_root: Path | None,
    dry_run: bool,
) -> bool:
    """
    Load csv_path, derive oct_pseudo_rel, optionally validate existence,
    save all-rows CSV and (when data_root given) the available-only CSV.
    Returns True if all pseudo files were found (or data_root not provided).
    """
    print(f"\nProcessing : {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    if "oct_rel" not in df.columns:
        print(f"  [ERROR] 'oct_rel'/'oct_preprocessed_v2' not found in {csv_path}.")
        print(f"  Columns present: {list(df.columns)}")
        return False

    if "split" not in df.columns:
        print(f"  [ERROR] 'split' column not found in {csv_path}.")
        return False

    # Derive oct_pseudo_rel row-by-row (split-aware)
    df["oct_pseudo_rel"] = df.apply(
        lambda r: derive_pseudo_path(r["oct_rel"], str(r["split"])),
        axis=1,
    )

    # Show 5 example derivations
    print("  Sample derivations (first 5 rows):")
    for _, row in df.head(5).iterrows():
        print(f"    [{row['split']}] {row['oct_rel']!r}")
        print(f"           -> {row['oct_pseudo_rel']!r}")

    # ------------------------------------------------------------------
    # File-existence check
    # ------------------------------------------------------------------
    if data_root is not None:
        found_mask   = df["oct_pseudo_rel"].apply(
            lambda p: (data_root / p).exists()
        )
        df["pseudo_exists"] = found_mask

        n_total   = len(df)
        n_found   = int(found_mask.sum())
        n_missing = n_total - n_found

        print(f"\n  Existence check against: {data_root}")
        print(f"    Total    : {n_total}")
        print(f"    Found    : {n_found}")
        print(f"    Missing  : {n_missing}")

        missing_ids = df.loc[~found_mask, "sample_id"].tolist()
        if missing_ids:
            print(f"    First 10 missing sample_ids:")
            for sid in missing_ids[:10]:
                print(f"      - {sid}")
            if len(missing_ids) > 10:
                print(f"      ... and {len(missing_ids) - 10} more")

        all_ok = n_missing == 0
    else:
        df["pseudo_exists"] = False      # unknown — not checked
        print("  [INFO] --data_root not provided; pseudo_exists set to False (unvalidated).")
        print("         Pass --data_root to validate file existence.")
        all_ok = True   # nothing to fail on

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if dry_run:
        print(f"  [dry_run] Would save all-rows CSV to:       {out_path}")
        if data_root is not None:
            avail_path = _available_path(out_path, df)
            print(f"  [dry_run] Would save available-only CSV to: {avail_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved all-rows CSV     -> {out_path}  ({len(df)} rows)")

        if data_root is not None:
            df_avail   = df[df["pseudo_exists"]].reset_index(drop=True)
            avail_path = _available_path(out_path, df)
            df_avail.to_csv(avail_path, index=False)
            print(f"  Saved available CSV    -> {avail_path}  ({len(df_avail)} rows)")

    return all_ok


def _available_path(out_path: Path, df: pd.DataFrame) -> Path:
    """
    Derive the _pseudo_available.csv output path.
    Uses the split value from the first row to produce the canonical name:
        pairs_{split}_pseudo_available.csv
    in the same directory as out_path.
    """
    split_name = str(df["split"].iloc[0]).strip().lower()
    return out_path.parent / f"pairs_{split_name}_pseudo_available.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add oct_pseudo_rel and pseudo_exists columns to split CSVs, "
            "and write filtered _pseudo_available.csv files."
        )
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="One or more input CSV files (train / val / test).",
    )
    parser.add_argument(
        "--out", nargs="+", default=None,
        help=(
            "Output paths for the all-rows CSVs, one per input. "
            "Defaults to <stem>_with_pseudo.csv in the same directory."
        ),
    )
    parser.add_argument(
        "--data_root", default=None,
        help=(
            "Image root directory. When provided: validates file existence, "
            "fills pseudo_exists, and writes _pseudo_available.csv files."
        ),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Derive paths and validate existence but do NOT write any files.",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]

    if args.out is not None:
        if len(args.out) != len(csv_paths):
            print(
                f"[ERROR] --out must have the same number of entries as --csv "
                f"({len(args.out)} vs {len(csv_paths)})."
            )
            sys.exit(1)
        out_paths = [Path(p) for p in args.out]
    else:
        out_paths = [
            p.parent / (p.stem + "_with_pseudo" + p.suffix)
            for p in csv_paths
        ]

    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and not data_root.exists():
        print(f"[WARNING] data_root not found: {data_root}  (skipping file validation)")
        data_root = None

    print("=" * 65)
    print("  add_pseudo_oct_column.py")
    print("=" * 65)

    results = []
    for csv_path, out_path in zip(csv_paths, out_paths):
        if not csv_path.exists():
            print(f"\n[ERROR] CSV not found: {csv_path}")
            results.append(False)
            continue
        ok = process_csv(csv_path, out_path, data_root, args.dry_run)
        results.append(ok)

    print("\n" + "=" * 65)
    if all(results):
        print("  DONE - all pseudo-OCT paths accounted for.")
    else:
        print("  DONE with WARNINGS - some pseudo-OCT files were missing.")
        print("  Use the _pseudo_available.csv files for training.")
    print("=" * 65)


if __name__ == "__main__":
    main()
