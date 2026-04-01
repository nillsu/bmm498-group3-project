"""
Bootstrap confidence-interval evaluation for MultimodalClassifier.

Usage:
    python bootstrap_eval.py \
        --ckpt_path  path/to/checkpoint.ckpt \
        --csv_path   path/to/splits_clean.csv \
        --data_root  path/to/data_root \
        --mode       fundus|oct|fusion \
        --output_dir ./outputs \
        --bootstrap_iters 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datamodule import MultimodalDataModule
from src.model import MultimodalClassifier


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary AUC via trapezoidal rule (no sklearn dependency)."""
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    try:
        return float(np.trapezoid(tpr, fpr))   # NumPy >= 2.0
    except AttributeError:
        return float(np.trapz(tpr, fpr))       # NumPy < 2.0


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Inference  (mirrors predict.py, single forward pass)
# ---------------------------------------------------------------------------

def collect_predictions(
    model: MultimodalClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    probs  : (N, 2) float32  — sigmoid probabilities
    preds  : (N, 2) int      — binary predictions at threshold
    labels : (N, 2) int      — ground-truth labels
    """
    all_probs:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_dev)                   # (B, 2)
            probs  = torch.sigmoid(logits).cpu().numpy()
            lbls   = batch["labels"].cpu().numpy()      # (B, 2)
            all_probs.append(probs)
            all_labels.append(lbls)

    probs_arr  = np.concatenate(all_probs,  axis=0).astype(np.float32)
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.int32)
    preds_arr  = (probs_arr > threshold).astype(np.int32)
    return probs_arr, preds_arr, labels_arr


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_iters: int,
    rng: np.random.Generator,
) -> dict:
    """
    Parameters
    ----------
    probs  : (N, 2)
    preds  : (N, 2)
    labels : (N, 2)

    Returns
    -------
    dict with keys "DR_pos" and "DME", each containing mean + CI95 for
    AUC, Accuracy, F1.
    """
    label_names = ["DR_pos", "DME"]
    N = len(labels)

    # Collect per-label per-metric bootstrap distributions
    # structure: {label: {metric: [values]}}
    dists: dict[str, dict[str, list[float]]] = {
        name: {"AUC": [], "Accuracy": [], "F1": []}
        for name in label_names
    }

    for _ in range(n_iters):
        idx = rng.integers(0, N, size=N)
        for col, name in enumerate(label_names):
            y_true = labels[idx, col]
            y_prob = probs[idx, col]
            y_pred = preds[idx, col]
            dists[name]["AUC"].append(_auc(y_true, y_prob))
            dists[name]["Accuracy"].append(_accuracy(y_true, y_pred))
            dists[name]["F1"].append(_f1(y_true, y_pred))

    results: dict = {}
    for name in label_names:
        entry: dict = {}
        for metric, values in dists[name].items():
            arr = np.array(values, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                entry[f"{metric}_mean"] = float("nan")
                entry[f"{metric}_CI95"] = [float("nan"), float("nan")]
            else:
                entry[f"{metric}_mean"] = float(np.mean(valid))
                entry[f"{metric}_CI95"] = [
                    float(np.percentile(valid, 2.5)),
                    float(np.percentile(valid, 97.5)),
                ]
            entry[f"{metric}_valid_bootstrap_samples"] = int(len(valid))
            entry[f"{metric}_total_bootstrap_iters"]   = n_iters
        results[name] = entry
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap CI evaluation for MultimodalClassifier"
    )
    p.add_argument("--ckpt_path",       required=True)
    p.add_argument("--csv_path",        default=None)
    p.add_argument("--data_root",       required=True)
    p.add_argument("--mode",            required=True,
                   choices=["fundus", "oct", "fusion",
                            "fusion_cross_attention", "fusion_bi_cross_attention"])
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--bootstrap_iters", type=int,   default=1000)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--image_size",      type=int,   default=224)
    p.add_argument("--split",           default="val", choices=["val", "test"])
    p.add_argument("--threshold",       type=float, default=0.5,
                   help="Probability threshold for binary predictions (default 0.5).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ---------------------------------------------------------------
    model: MultimodalClassifier = MultimodalClassifier.load_from_checkpoint(
        args.ckpt_path,
        mode=args.mode,
        map_location=device,
    )
    model.to(device)

    # ---- datamodule ----------------------------------------------------------
    dm = MultimodalDataModule(
        csv_path=args.csv_path,
        data_root=args.data_root,
        mode=args.mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    if args.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()
        if not loader:
            print("No test split found; falling back to val split.")
            loader = dm.val_dataloader()

    # ---- single forward pass -------------------------------------------------
    print("Running inference...")
    probs, preds, labels = collect_predictions(model, loader, device, threshold=args.threshold)
    print(f"Collected {len(labels)} samples.")

    # ---- bootstrap -----------------------------------------------------------
    print(f"Bootstrapping ({args.bootstrap_iters} iterations)...")
    rng = np.random.default_rng(seed=42)
    results = bootstrap_metrics(probs, preds, labels, args.bootstrap_iters, rng)

    # ---- save ----------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    for label, metrics in results.items():
        print(f"\n  {label}")
        for metric_key, value in metrics.items():
            print(f"    {metric_key}: {value}")


if __name__ == "__main__":
    main()
