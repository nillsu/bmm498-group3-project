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
    p.add_argument("--mode",       required=True, choices=["fundus", "oct", "fusion"])
    p.add_argument("--output_csv", required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size",  type=int, default=224)
    p.add_argument("--split",       default="test", choices=["test", "val"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    if args.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()
        if not loader:
            print("No test split found in CSV; using validation split for prediction export.")
            loader = dm.val_dataloader()

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            # move image tensors to device; leave strings as-is
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_dev)                      # (B, 2)
            probs  = torch.sigmoid(logits).cpu()           # (B, 2)
            sample_ids = batch["sample_id"]                # list[str]
            for sid, p in zip(sample_ids, probs.tolist()):
                rows.append({"sample_id": sid, "DR_prob": p[0], "DME_prob": p[1]})

    df = pd.DataFrame(rows, columns=["sample_id", "DR_prob", "DME_prob"])
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
