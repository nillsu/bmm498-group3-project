"""
Compute pos_weight for BCEWithLogitsLoss from a dataset CSV.

pos_weight[label] = N_neg / N_pos

Usage:
    python compute_pos_weight.py --csv_path splits_clean.csv
    python compute_pos_weight.py --csv_path splits_clean.csv --output_json weights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compute_pos_weight(df: pd.DataFrame) -> dict:
    results = {}
    for col in ("DR_pos", "DME"):
        n_pos = int((df[col] == 1).sum())
        n_neg = int((df[col] == 0).sum())
        weight = n_neg / n_pos if n_pos > 0 else float("nan")
        results[col] = {"n_pos": n_pos, "n_neg": n_neg, "pos_weight": weight}
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute pos_weight for BCEWithLogitsLoss")
    p.add_argument("--csv_path",    required=True)
    p.add_argument("--output_json", default=None,
                   help="Optional path to write JSON output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)

    stats = compute_pos_weight(df)

    w_dr  = stats["DR_pos"]["pos_weight"]
    w_dme = stats["DME"]["pos_weight"]

    for col, s in stats.items():
        print(f"{col}: n_pos={s['n_pos']}  n_neg={s['n_neg']}  pos_weight={s['pos_weight']:.4f}")

    print(f"\npos_weight = [{w_dr:.4f}, {w_dme:.4f}]")

    if args.output_json:
        out = {
            "pos_weight": [w_dr, w_dme],
            "counts": {
                col: {"n_pos": s["n_pos"], "n_neg": s["n_neg"]}
                for col, s in stats.items()
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
