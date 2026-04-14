"""
A5 — Prediction export.

Usage:
    python predict.py --csv_path <path> --data_root <path> \
        --checkpoint <path.ckpt> --mode fusion --output_csv preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _print_env() -> None:
    print(f"Python  : {sys.version}")
    try:
        import torch
        print(f"PyTorch : {torch.__version__}")
        print(f"CUDA    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch : not installed")

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule import MultimodalDataModule
from src.model import MultimodalClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export predictions from a trained checkpoint")
    p.add_argument("--csv_path",   default=None)
    p.add_argument("--data_root",  required=True)
    p.add_argument("--train_csv",  default=None)
    p.add_argument("--val_csv",    default=None)
    p.add_argument("--test_csv",   default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode",       required=True,
                   choices=["fundus", "oct", "fusion",
                            "fusion_cross_attention", "fusion_bi_cross_attention",
                            "pseudo_oct", "fusion_pseudo"])
    p.add_argument("--output_csv", required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--split",       default="test", choices=["test", "val"])
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Probability threshold for binary predictions (default 0.5).")
    p.add_argument("--print_env",   action="store_true",
                   help="Print Python/CUDA environment info at startup.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_env:
        _print_env()

    dm = MultimodalDataModule(
        csv_path=args.csv_path,
        data_root=args.data_root,
        mode=args.mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )
    dm.setup()

    model = MultimodalClassifier.load_from_checkpoint(
        args.checkpoint,
        mode=args.mode,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    include_labels = False
    if args.split == "val":
        loader = dm.val_dataloader()
        include_labels = True
    else:
        loader = dm.test_dataloader()
        if not loader:
            print("No test split found in CSV; using validation split for prediction export.")
            loader = dm.val_dataloader()
            include_labels = True

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            # move image tensors to device; leave strings as-is
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits     = model(batch_dev)                  # (B, 2)
            probs      = torch.sigmoid(logits).cpu()       # (B, 2)
            sample_ids = batch["sample_id"]                # list[str]
            labels     = batch["labels"].cpu() if include_labels else None

            for i, (sid, p) in enumerate(zip(sample_ids, probs.tolist())):
                row = {
                    "sample_id": sid,
                    "DR_prob":   round(p[0], 6),
                    "DME_prob":  round(p[1], 6),
                    "DR_pred":   int(p[0] > args.threshold),
                    "DME_pred":  int(p[1] > args.threshold),
                }
                if include_labels:
                    row["DR_true"]  = int(labels[i, 0].item())
                    row["DME_true"] = int(labels[i, 1].item())
                rows.append(row)

    base_cols = ["sample_id", "DR_prob", "DME_prob", "DR_pred", "DME_pred"]
    all_cols  = base_cols + (["DR_true", "DME_true"] if include_labels else [])
    df = pd.DataFrame(rows, columns=all_cols)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")

    if include_labels and len(df) > 0:
        import numpy as np

        def _f1_binary(y_true: "np.ndarray", y_pred: "np.ndarray") -> float:
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            denom = 2 * tp + fp + fn
            return 2 * tp / denom if denom > 0 else float("nan")

        dr_t, dr_p   = df["DR_true"].values, df["DR_pred"].values
        dme_t, dme_p = df["DME_true"].values, df["DME_pred"].values
        print(f"\nEvaluation (threshold={args.threshold}):")
        print(f"  DR  — accuracy={float((dr_t  == dr_p ).mean()):.4f}  f1={_f1_binary(dr_t,  dr_p ):.4f}")
        print(f"  DME — accuracy={float((dme_t == dme_p).mean()):.4f}  f1={_f1_binary(dme_t, dme_p):.4f}")


if __name__ == "__main__":
    main()
