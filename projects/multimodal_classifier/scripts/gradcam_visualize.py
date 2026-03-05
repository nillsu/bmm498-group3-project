"""
Grad-CAM visualization script for MultimodalClassifier.

Usage:
    python gradcam_visualize.py \
        --ckpt_path path/to/checkpoint.ckpt \
        --csv_path  path/to/splits_clean.csv \
        --data_root path/to/data_root \
        --mode      fundus|oct|fusion \
        --output_dir ./outputs \
        --num_images 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import MultimodalEyeDataset
from model import MultimodalClassifier
from transforms import get_fundus_transforms, get_oct_transforms


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def _get_last_conv(encoder: nn.Module) -> nn.Module:
    last_conv = None
    for module in encoder.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in encoder.")
    return last_conv


class GradCAM:
    def __init__(self, target_layer: nn.Module) -> None:
        self._features: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._hook_fwd)
        self._bwd = target_layer.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, module, inp, out) -> None:
        self._features = out.detach()

    def _hook_bwd(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def compute(self) -> torch.Tensor:
        """Return (B, H, W) ReLU-weighted CAM."""
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam = (weights * self._features).sum(dim=1)                 # (B, H, W)
        return F.relu(cam)

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) normalised tensor → (H, W, 3) float32 [0, 1]."""
    img = tensor.cpu().numpy()
    if img.shape[0] == 1:
        img = np.concatenate([img, img, img], axis=0)
    img = img.transpose(1, 2, 0)
    return _normalize(img).astype(np.float32)


def _overlay(image_np: np.ndarray, cam_np: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap on image_np (H, W, 3) [0, 1]."""
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam_np, (w, h))
    cam_resized = _normalize(cam_resized)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(0.5 * image_np + 0.5 * heatmap, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM for MultimodalClassifier")
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--csv_path",   required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--mode",       required=True, choices=["fundus", "oct", "fusion"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ---------------------------------------------------------------
    model: MultimodalClassifier = MultimodalClassifier.load_from_checkpoint(
        args.ckpt_path, map_location=device
    )
    model.eval()
    model.to(device)

    # ---- dataset -------------------------------------------------------------
    df = pd.read_csv(args.csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "val"].reset_index(drop=True)
    df = df.head(args.num_images).reset_index(drop=True)

    tf_fundus = (
        get_fundus_transforms(train=False, image_size=224)
        if args.mode in ("fundus", "fusion") else None
    )
    tf_oct = (
        get_oct_transforms(train=False, image_size=224)
        if args.mode in ("oct", "fusion") else None
    )

    dataset = MultimodalEyeDataset(
        df=df,
        data_root=args.data_root,
        mode=args.mode,
        transform_fundus=tf_fundus,
        transform_oct=tf_oct,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # ---- Grad-CAM setup ------------------------------------------------------
    # For fusion mode use the fundus encoder for visualization
    if args.mode == "oct":
        encoder = model.oct_encoder
    else:
        encoder = model.fundus_encoder

    target_layer = _get_last_conv(encoder)
    gradcam = GradCAM(target_layer)

    # ---- output dir ----------------------------------------------------------
    out_dir = Path(args.output_dir) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = ["DR_pos", "DME"]

    # ---- inference loop ------------------------------------------------------
    for i, batch in enumerate(loader):
        if i >= args.num_images:
            break

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        sample_id: str = batch["sample_id"][0]
        gt_labels = batch["labels"][0]  # (2,)

        # Display image: use fundus for fundus/fusion, oct otherwise
        if args.mode in ("fundus", "fusion"):
            display_tensor = batch["fundus"][0]
        else:
            display_tensor = batch["oct"][0]
        img_np = _tensor_to_rgb(display_tensor)

        for label_idx, label_name in enumerate(label_names):
            model.zero_grad(set_to_none=True)
            logits = model(batch)            # (1, 2)
            logits[:, label_idx].sum().backward()

            cam = gradcam.compute()          # (1, H, W)
            cam_np = cam[0].cpu().numpy()

            # resize cam to image size for display
            h, w = img_np.shape[:2]
            cam_display = cv2.resize(cam_np, (w, h))
            cam_display = _normalize(cam_display)
            overlay = _overlay(img_np, cam_np)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_np)
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(cam_display, cmap="jet")
            axes[1].set_title(f"Grad-CAM ({label_name})")
            axes[1].axis("off")

            gt_val = int(gt_labels[label_idx].item())
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay  gt={gt_val}")
            axes[2].axis("off")

            fig.suptitle(f"Sample: {sample_id}  |  {label_name}", fontsize=11)
            plt.tight_layout()

            fname = f"{sample_id}_{label_name}_gradcam.png"
            fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"[{i + 1}/{len(dataset)}] saved: {sample_id}")

    gradcam.remove()
    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
