"""
MultimodalClassifier — PyTorch Lightning module for fundus + OCT classification.

Supports three modes: fundus, oct, fusion.
Mid-level fusion: two ResNet-18 encoders, per-branch heads plus a fusion head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

_VALID_MODES = {"fundus", "oct", "fusion"}


def _make_encoder(in_chans: int = 3) -> nn.Module:
    return timm.create_model(
        "resnet18",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
        in_chans=in_chans,
    )


def _infer_dim(encoder: nn.Module, in_chans: int, image_size: int = 224) -> int:
    with torch.no_grad():
        device = next(encoder.parameters()).device
        dummy = torch.zeros(1, in_chans, image_size, image_size, device=device)
        return encoder(dummy).shape[1]


class MultimodalClassifier(pl.LightningModule):
    def __init__(
        self,
        mode: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode={mode!r} is not valid. Expected one of {_VALID_MODES}.")
        self.save_hyperparameters()
        self.mode = mode

        # Build encoders only for needed branches
        if mode in {"fundus", "fusion"}:
            self.fundus_encoder = _make_encoder(in_chans=3)
            fundus_dim = _infer_dim(self.fundus_encoder, in_chans=3)
            self.fundus_head = nn.Linear(fundus_dim, 2)
        else:
            fundus_dim = 0

        if mode in {"oct", "fusion"}:
            self.oct_encoder = _make_encoder(in_chans=1)
            oct_dim = _infer_dim(self.oct_encoder, in_chans=1)
            self.oct_head = nn.Linear(oct_dim, 2)
        else:
            oct_dim = 0

        if mode == "fusion":
            fused_dim = fundus_dim + oct_dim
            self.fusion_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim // 2, 2),
            )

        self.loss_fn = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        if self.mode == "fundus":
            if "fundus" not in batch:
                raise KeyError("mode='fundus' requires batch['fundus'] but key is missing.")
            return self.fundus_head(self.fundus_encoder(batch["fundus"]))

        if self.mode == "oct":
            if "oct" not in batch:
                raise KeyError("mode='oct' requires batch['oct'] but key is missing.")
            return self.oct_head(self.oct_encoder(batch["oct"]))

        # fusion
        if "fundus" not in batch:
            raise KeyError("mode='fusion' requires batch['fundus'] but key is missing.")
        if "oct" not in batch:
            raise KeyError("mode='fusion' requires batch['oct'] but key is missing.")
        f_feat = self.fundus_encoder(batch["fundus"])
        o_feat = self.oct_encoder(batch["oct"])
        return self.fusion_head(torch.cat([f_feat, o_feat], dim=1))

    # ------------------------------------------------------------------
    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        labels = batch["labels"].float()
        if labels.shape == (2,):
            labels = labels.unsqueeze(0)          # (1,2) — single sample edge case
        if labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError(
                f"labels must have shape (B, 2) but got {tuple(labels.shape)}."
            )
        logits = self(batch)
        loss = self.loss_fn(logits, labels)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"),
                 on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
