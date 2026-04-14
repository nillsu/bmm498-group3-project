"""
add_pseudo_oct_column.py
────────────────────────
Extends existing split CSVs with an oct_pseudo_rel column derived from the
real OCT path (oct_preprocessed_v2 / oct_rel).

Derivation rule
───────────────
Real OCT  :  <split>/oct_real/<base_id>_oct_v2.jpg
Pseudo-OCT:  <split>/pseudo-oct/<base_id>_fake_B.png

where base_id is the filename stem with _oct_v2 stripped.

Example
───────
oct_rel = "train/oct_real/1984_OD_000054_oct_v2.jpg"
  → oct_pseudo_rel = "train/pseudo-oct/1984_OD_000054_fake_B.png"

Usage (Colab)
─────────────
# Process all three CSVs at once:
!python add_pseudo_oct_column.py \\
    --csv dataset_csv/pairs_train_oct_v2.csv \\
          dataset_csv/pairs_val_oct_v2.csv   \\
          dataset_csv/pairs_test_oct_v2.csv  \\
    --out dataset_csv/pairs_train_with_pseudo.csv \\
          dataset_csv/pairs_val_with_pseudo.csv   \\
          dataset_csv/pairs_test_with_pseudo.csv  \\
    --data_root /content/bmm498_data

# Validate only (no output written):
!python add_pseudo_oct_column.py \\
    --csv dataset_csv/pairs_train_oct_v2.csv \\
    --data_root /content/bmm498_data \\
    --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import pandas as pd


# ── path derivation ──────────────────────────────────────────────────────────

def derive_pseudo_path(oct_rel: str) -> str:
    """
    Derive the pseudo-OCT relative path from the real OCT relative path.

    Input  : "train/oct_real/1984_OD_000054_oct_v2.jpg"
    Output : "train/pseudo-oct/1984_OD_000054_fake_B.png"
    """
    p = PurePosixPath(oct_rel.replace("\\", "/"))
    # The base_id is the stem of the real OCT file, with _oct_v2 removed.
    # Handles both _oct_v2 (standard) and any other suffix gracefully.
    stem = p.stem  # e.g. "1984_OD_000054_oct_v2"
    if stem.endswith("_oct_v2"):
        base_id = stem[: -len("_oct_v2")]
    else:
        # Fallback: strip last underscore-delimited token and hope for the best.
        base_id = "_".join(stem.split("_")[:-1])
    split_dir = p.parts[0]  # "train" / "val" / "test"
    return f"{split_dir}/pseudo-oct/{base_id}_fake_B.png"


# ── validation ───────────────────────────────────────────────────────────────

def validate_pseudo_paths(
    df: pd.DataFrame,
    data_root: Path,
    source_label: str,
) -> tuple[int, list[str]]:
    """
    Check that every oct_pseudo_rel path exists under data_root.
    Returns (n_missing, list_of_missing_sample_ids).
    """
    missing = []
    for _, row in df.iterrows():
        full = data_root / str(row["oct_pseudo_rel"])
        if not full.exists():
            missing.append(str(row.get("sample_id", row.name)))
    if missing:
        print(
            f"  [{source_label}] MISSING {len(missing)}/{len(df)} pseudo-OCT files."
        )
        for sid in missing[:10]:
            print(f"    - {sid}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more.")
    else:
        print(f"  [{source_label}] All {len(df)} pseudo-OCT files found.")
    return len(missing), missing


# ── column normalisation (mirrors datamodule._normalize_columns) ─────────────

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


# ── main ─────────────────────────────────────────────────────────────────────

def process_csv(
    csv_path: Path,
    out_path: Path,
    data_root: Path | None,
    dry_run: bool,
) -> bool:
    """Process one CSV file. Returns True if all pseudo files were found."""
    print(f"\nProcessing : {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    # Source column for deriving pseudo path
    if "oct_rel" not in df.columns:
        print(f"  [ERROR] Neither 'oct_rel' nor 'oct_preprocessed_v2' found in {csv_path}.")
        print(f"  Available columns: {list(df.columns)}")
        return False

    # Derive oct_pseudo_rel
    df["oct_pseudo_rel"] = df["oct_rel"].apply(derive_pseudo_path)

    # Show a few examples
    print("  Sample derivations:")
    for _, row in df.head(3).iterrows():
        print(f"    {row['oct_rel']!r}")
        print(f"    -> {row['oct_pseudo_rel']!r}")

    # Validate files if data_root given
    all_ok = True
    if data_root is not None:
        label = csv_path.name
        n_missing, _ = validate_pseudo_paths(df, data_root, label)
        all_ok = n_missing == 0
    else:
        print("  [INFO] --data_root not provided; skipping file-existence validation.")

    # Save
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}  ({len(df)} rows, columns: {list(df.columns)})")
    else:
        print(f"  [dry_run] Would save to {out_path}")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add oct_pseudo_rel column to existing split CSVs."
    )
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="One or more input CSV files (train / val / test).",
    )
    parser.add_argument(
        "--out", nargs="+", default=None,
        help=(
            "Output paths, one per input CSV. "
            "Defaults to <input_stem>_with_pseudo.csv in the same directory."
        ),
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Image root directory. When provided, validates that every pseudo-OCT file exists.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Derive and validate paths but do NOT write output files.",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]

    if args.out is not None:
        if len(args.out) != len(csv_paths):
            print(
                f"[ERROR] --out must have the same number of entries as --csv "
                f"(got {len(args.out)} vs {len(csv_paths)})."
            )
            sys.exit(1)
        out_paths = [Path(p) for p in args.out]
    else:
        # Default: same directory, _with_pseudo suffix
        out_paths = [
            p.parent / (p.stem + "_with_pseudo" + p.suffix)
            for p in csv_paths
        ]

    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and not data_root.exists():
        print(f"[WARNING] data_root does not exist: {data_root}  (skipping file validation)")
        data_root = None

    print("=" * 60)
    print("  add_pseudo_oct_column.py")
    print("=" * 60)

    results = []
    for csv_path, out_path in zip(csv_paths, out_paths):
        if not csv_path.exists():
            print(f"\n[ERROR] Input CSV not found: {csv_path}")
            results.append(False)
            continue
        ok = process_csv(csv_path, out_path, data_root, args.dry_run)
        results.append(ok)

    print("\n" + "=" * 60)
    if all(results):
        print("  DONE — all pseudo-OCT paths validated successfully.")
    else:
        print("  DONE with WARNINGS — some pseudo-OCT files were missing (see above).")
        print("  Rows with missing pseudo-OCT files are still included in the output CSV.")
        print("  Re-run after generating the missing pseudo images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
