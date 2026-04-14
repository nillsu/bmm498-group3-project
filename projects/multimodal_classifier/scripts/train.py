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

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule import MultimodalDataModule
from src.model import MultimodalClassifier


class NaNLossCallback(Callback):
    def on_before_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: torch.Tensor) -> None:
        if torch.isnan(loss):
            print(f"\nWARNING: NaN loss detected at step {trainer.global_step}. Stopping training.")
            trainer.should_stop = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MultimodalClassifier")
    p.add_argument("--csv_path",     default=None)
    p.add_argument("--data_root",    required=True)
    p.add_argument("--train_csv",    default=None)
    p.add_argument("--val_csv",      default=None)
    p.add_argument("--test_csv",     default=None)
    p.add_argument("--mode",         default="fusion",
                   choices=["fundus", "oct", "fusion",
                            "fusion_cross_attention", "fusion_bi_cross_attention",
                            "pseudo_oct", "fusion_pseudo",
                            "fusion_cross_attention_pseudo",
                            "fusion_bi_cross_attention_pseudo"])
    p.add_argument("--image_size",   type=int,   default=224)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--precision",    default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--backbone",     default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0"])
    p.add_argument("--pretrained",    action="store_true",  default=True,
                   help="Use pretrained backbone (default: True).")
    p.add_argument("--no_pretrained", action="store_false", dest="pretrained",
                   help="Disable pretrained backbone weights.")
    p.add_argument("--fast_dev_run",  action="store_true",
                   help="Run 1 train+val batch only (sanity check).")
    p.add_argument("--gradient_clip_val",       type=float, default=1.0)
    p.add_argument("--deterministic",           action="store_true",
                   help="Enable torch.use_deterministic_algorithms(True).")
    p.add_argument("--accumulate_grad_batches", type=int,   default=1)
    p.add_argument("--log_every_n_steps",       type=int,   default=10)
    p.add_argument("--patience",                type=int,   default=5)
    p.add_argument("--limit_train_batches",     type=float, default=1.0)
    p.add_argument("--limit_val_batches",       type=float, default=1.0)
    p.add_argument("--monitor",      default="val_dr_auc",
                   choices=["val_dr_auc", "val_dme_auc", "val/loss"],
                   help="Metric to monitor for checkpointing and early stopping.")
    p.add_argument("--pos_weight",   type=float, nargs=2, default=None,
                   metavar=("W_DR", "W_DME"),
                   help="pos_weight for BCEWithLogitsLoss: [DR_pos, DME]. "
                        "Compute with compute_pos_weight.py. Example: --pos_weight 3.2 5.1")
    p.add_argument("--alpha_fundus", type=float, default=0.3,
                   help="Weight for auxiliary fundus loss in multimodal modes (default: 0.3).")
    p.add_argument("--alpha_oct",    type=float, default=0.3,
                   help="Weight for auxiliary OCT loss in multimodal modes (default: 0.3).")
    p.add_argument("--model_variant", default="auxloss", choices=["baseline", "auxloss"],
                   help="'auxloss': auxiliary unimodal losses active (default). "
                        "'baseline': standard single-loss training (no aux heads).")
    p.add_argument("--print_env",    action="store_true",
                   help="Print Python/CUDA environment info at startup.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_env:
        _print_env()

    print(f"Torch   : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Mode    : {args.mode}")
    print(f"Variant : {args.model_variant}")
    print(f"Batch   : {args.batch_size}")

    if args.precision == "16-mixed":
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

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
        pretrained=args.pretrained,
        pos_weight=args.pos_weight,
        alpha_fundus=args.alpha_fundus,
        alpha_oct=args.alpha_oct,
        model_variant=args.model_variant,
    )

    # Pre-flight: setup data and print one batch's shapes to catch path issues early
    print("\n--- Pre-flight data check ---")
    dm.setup("fit")
    for split_name, loader_fn in [("train", dm.train_dataloader), ("val", dm.val_dataloader)]:
        batch = next(iter(loader_fn()))
        print(f"  {split_name} batch:")
        if "fundus" in batch:
            print(f"    fundus : {tuple(batch['fundus'].shape)}")
        if "oct" in batch:
            print(f"    oct    : {tuple(batch['oct'].shape)}")
        print(f"    labels : {tuple(batch['labels'].shape)}")
    print("--- Pre-flight OK ---\n")

    # Dynamic monitor mode and checkpoint filename
    monitor_mode  = "min" if args.monitor == "val/loss" else "max"
    monitor_token = args.monitor.replace("/", "_")   # safe filename token

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"best-{{epoch}}-{{{monitor_token}:.4f}}",
        monitor=args.monitor,
        mode=monitor_mode,
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor=args.monitor,
        mode=monitor_mode,
        patience=args.patience,
    )
    logger = CSVLogger(save_dir=str(output_dir), name="logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        deterministic=args.deterministic,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_cb, early_stop_cb, NaNLossCallback()],
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)

    print(f"\nBest checkpoint : {checkpoint_cb.best_model_path}")
    print(f"Logs written to : {logger.log_dir}")


if __name__ == "__main__":
    main()
