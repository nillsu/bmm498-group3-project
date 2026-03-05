"""
A4 — Training entrypoint.

Usage:
    python train.py --csv_path <path> --data_root <path> --output_dir <path> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule import MultimodalDataModule
from src.model import MultimodalClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MultimodalClassifier")
    p.add_argument("--csv_path",     default=None)
    p.add_argument("--data_root",    required=True)
    p.add_argument("--train_csv",    default=None)
    p.add_argument("--val_csv",      default=None)
    p.add_argument("--test_csv",     default=None)
    p.add_argument("--mode",         default="fusion", choices=["fundus", "oct", "fusion"])
    p.add_argument("--image_size",   type=int,   default=224)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--precision",    default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--backbone",     default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save CLI config for reproducibility
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to : {config_path}")

    dm = MultimodalDataModule(
        csv_path=args.csv_path,
        data_root=args.data_root,
        mode=args.mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )

    model = MultimodalClassifier(
        mode=args.mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        backbone=args.backbone,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}-{val_dr_auc:.4f}",
        monitor="val_dr_auc",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_dr_auc",
        mode="max",
        patience=5,
    )
    logger = CSVLogger(save_dir=str(output_dir), name="logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        deterministic=True,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)

    print(f"\nBest checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Logs written to : {logger.log_dir}")


if __name__ == "__main__":
    main()
