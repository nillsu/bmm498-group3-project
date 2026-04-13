"""
debug_augmentation.py
─────────────────────
Standalone script to verify that data augmentation is actually applied
during training by fetching the same sample multiple times and comparing
the resulting image tensors visually.

Run from the repo root:
    python debug_augmentation.py [--data_root /path/to/data] [--index 10]

In Colab:
    !python debug_augmentation.py --data_root /content/drive/MyDrive/bmm498_data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── make repo importable ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
SRC = REPO_ROOT / "projects" / "multimodal_classifier" / "src"
sys.path.insert(0, str(SRC))

from dataset import MultimodalEyeDataset
from transforms import get_fundus_transforms


# ── helpers ──────────────────────────────────────────────────────────────────

FUNDUS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
FUNDUS_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def unnormalize_fundus(tensor):
    """CHW float32 tensor → HWC uint8 numpy array (display-ready)."""
    img = tensor.numpy().transpose(1, 2, 0)          # CHW → HWC
    img = img * FUNDUS_STD[None, None, :] + FUNDUS_MEAN[None, None, :]
    img = np.clip(img, 0.0, 1.0)
    return img


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy CSV columns to the names expected by MultimodalEyeDataset."""
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

def main():
    parser = argparse.ArgumentParser(description="Verify data augmentation is applied.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory that contains train/fundus/... sub-paths. "
             "Defaults to <repo_root>/bmm498_data (local) or /content/bmm498_data (Colab).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to training CSV. Defaults to <repo_root>/dataset_csv/pairs_train_oct_v2.csv.",
    )
    parser.add_argument(
        "--index", type=int, default=10,
        help="Dataset index to sample repeatedly (default: 10).",
    )
    parser.add_argument(
        "--n_samples", type=int, default=6,
        help="How many times to fetch the same index (default: 6).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="If provided, save the figure to this path instead of showing it.",
    )
    args = parser.parse_args()

    # ── resolve paths ────────────────────────────────────────────────────────
    if args.data_root is None:
        candidates = [
            REPO_ROOT / "bmm498_data",
            Path("/content/bmm498_data"),
            Path("/content/drive/MyDrive/bmm498_data"),
        ]
        for c in candidates:
            if c.exists():
                args.data_root = str(c)
                break
        if args.data_root is None:
            print("[ERROR] Could not auto-detect data_root. Pass --data_root explicitly.")
            sys.exit(1)

    if args.csv is None:
        args.csv = str(REPO_ROOT / "dataset_csv" / "pairs_train_oct_v2.csv")

    print(f"\n{'='*60}")
    print(f"  Data root : {args.data_root}")
    print(f"  CSV       : {args.csv}")
    print(f"  Index     : {args.index}")
    print(f"  Samples   : {args.n_samples}")
    print(f"{'='*60}\n")

    # ── load CSV ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df = _normalize_columns(df)
    # keep only training rows (CSV may be train-only, but guard anyway)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)

    print(f"Training samples in CSV: {len(df)}")
    if args.index >= len(df):
        print(f"[WARNING] index {args.index} >= dataset size {len(df)}. Using index 0.")
        args.index = 0

    # ── build transform (TRAIN mode → augmentations active) ──────────────────
    train_transform = get_fundus_transforms(train=True, image_size=224)

    print("\n" + "─"*60)
    print("TRAIN TRANSFORM PIPELINE (fundus):")
    print(train_transform)
    print("─"*60)

    # ── identify random operations ────────────────────────────────────────────
    random_ops = []
    for t in train_transform.transforms:
        name = type(t).__name__
        if any(kw in name for kw in ("Random", "ColorJitter", "Jitter")):
            random_ops.append(name)

    print("\nRANDOM (stochastic) operations detected:")
    if random_ops:
        for op in random_ops:
            print(f"  ✓  {op}")
    else:
        print("  [none found — augmentation may be disabled]")
    print("─"*60 + "\n")

    # ── instantiate dataset in FUNDUS mode ───────────────────────────────────
    dataset = MultimodalEyeDataset(
        df=df,
        data_root=args.data_root,
        mode="fundus",
        transform_fundus=train_transform,
        verify_files=False,
    )

    # ── fetch the same index N times ─────────────────────────────────────────
    print(f"Fetching dataset[{args.index}] × {args.n_samples} times ...\n")
    images = []
    for i in range(args.n_samples):
        sample = dataset[args.index]
        img_np = unnormalize_fundus(sample["fundus"])
        images.append(img_np)

    # ── pixel-level comparison ────────────────────────────────────────────────
    diffs = []
    for j in range(1, args.n_samples):
        diff = np.mean(np.abs(images[0].astype(np.float64) - images[j].astype(np.float64)))
        diffs.append(diff)

    max_diff   = max(diffs)
    mean_diff  = sum(diffs) / len(diffs)
    augmented  = max_diff > 1e-4   # threshold: any non-trivial pixel change

    # ── plot ──────────────────────────────────────────────────────────────────
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, args.n_samples, figsize=(4 * args.n_samples, 4))
    fig.suptitle(
        f"Augmentation verification — dataset[{args.index}] fetched {args.n_samples}×\n"
        f"(TRAIN transform, fundus modality)",
        fontsize=13,
    )
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.set_title(f"sample {i+1}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()

    # ── interpretation ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print(f"  Mean pixel diff vs sample 1 : {mean_diff:.6f}")
    print(f"  Max  pixel diff vs sample 1 : {max_diff:.6f}")
    print()
    if augmented:
        print("  ✅  AUGMENTATION IS WORKING")
        print("      Images differ across fetches → random transforms are applied")
        print("      each time __getitem__ is called.")
    else:
        print("  ❌  AUGMENTATION IS NOT APPLIED  (or not random)")
        print("      All fetches returned identical pixel values.")
        print("      Check that train=True was passed to get_fundus_transforms()")
        print("      and that the Dataset is NOT caching tensors.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
