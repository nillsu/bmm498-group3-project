"""
MultimodalClassifier — PyTorch Lightning module for fundus + OCT classification.

Supports four modes: fundus, oct, fusion, fusion_cross_attention.
Mid-level fusion: two ResNet-18 encoders, per-branch heads plus a fusion head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score

_VALID_MODES     = {"fundus", "oct", "fusion", "fusion_cross_attention", "fusion_bi_cross_attention"}
_VALID_BACKBONES = {"resnet18", "resnet34", "efficientnet_b0"}


def _make_encoder(backbone: str, in_chans: int = 3, pretrained: bool = True) -> nn.Module:
    if backbone not in _VALID_BACKBONES:
        raise ValueError(f"backbone={backbone!r} must be one of {_VALID_BACKBONES}.")
    try:
        return timm.create_model(
            backbone, pretrained=pretrained, num_classes=0,
            global_pool="avg", in_chans=in_chans,
        )
    except Exception as exc:
        if pretrained:
            print(
                f"WARNING: Failed to download pretrained weights for '{backbone}' "
                f"({exc}). Falling back to random initialisation (pretrained=False)."
            )
            return timm.create_model(
                backbone, pretrained=False, num_classes=0,
                global_pool="avg", in_chans=in_chans,
            )
        raise


def _infer_dim(encoder: nn.Module, in_chans: int, image_size: int = 224) -> int:
    with torch.no_grad():
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        dummy = torch.zeros(1, in_chans, image_size, image_size, device=device)
        return encoder(dummy).shape[1]


def _infer_feat_dim(encoder: nn.Module, in_chans: int, image_size: int = 224) -> int:
    """Infer channel dim from forward_features() (pre-pool spatial features)."""
    with torch.no_grad():
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        dummy = torch.zeros(1, in_chans, image_size, image_size, device=device)
        feat = encoder.forward_features(dummy)
        if feat.ndim == 4:   # (B, C, H, W)
            return feat.shape[1]
        if feat.ndim == 3:   # (B, N, C)
            return feat.shape[2]
        return feat.shape[-1]


def _feat_to_tokens(feat: torch.Tensor) -> torch.Tensor:
    """Convert feature tensor to (B, N, C) token sequence."""
    if feat.ndim == 4:                              # (B, C, H, W)
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)      # (B, H*W, C)
    if feat.ndim == 2:                              # (B, C)
        return feat.unsqueeze(1)                    # (B, 1, C)
    return feat                                     # already (B, N, C)


class CrossAttentionFusion(nn.Module):
    """Single cross-attention block: Q from fundus, K/V from OCT."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        q_tokens  : (B, N_q,  C)
        kv_tokens : (B, N_kv, C)
        returns   : (B, N_q,  C)
        """
        attn_out, _ = self.attn(q_tokens, kv_tokens, kv_tokens)
        return self.norm(q_tokens + attn_out)


class MultimodalClassifier(pl.LightningModule):
    def __init__(
        self,
        mode: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        pos_weight: list[float] | None = None,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in _VALID_MODES:
            raise ValueError(f"mode={mode!r} is not valid. Expected one of {_VALID_MODES}.")
        if backbone not in _VALID_BACKBONES:
            raise ValueError(f"backbone={backbone!r} must be one of {_VALID_BACKBONES}.")
        self.save_hyperparameters()
        self.mode = mode

        # Build encoders only for needed branches
        if mode in {"fundus", "fusion", "fusion_cross_attention", "fusion_bi_cross_attention"}:
            self.fundus_encoder = _make_encoder(backbone, in_chans=3, pretrained=pretrained)
            fundus_dim = _infer_dim(self.fundus_encoder, in_chans=3)
            self.fundus_head = nn.Linear(fundus_dim, 2)
        else:
            fundus_dim = 0

        if mode in {"oct", "fusion", "fusion_cross_attention", "fusion_bi_cross_attention"}:
            self.oct_encoder = _make_encoder(backbone, in_chans=1, pretrained=pretrained)
            oct_dim = _infer_dim(self.oct_encoder, in_chans=1)
            if mode in {"oct", "fusion"}:
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

        if mode == "fusion_cross_attention":
            f_feat_dim = _infer_feat_dim(self.fundus_encoder, in_chans=3)
            o_feat_dim = _infer_feat_dim(self.oct_encoder,    in_chans=1)
            attn_dim   = f_feat_dim
            self.f_attn_proj = nn.Identity() if f_feat_dim == attn_dim else nn.Linear(f_feat_dim, attn_dim)
            self.o_attn_proj = nn.Identity() if o_feat_dim == attn_dim else nn.Linear(o_feat_dim, attn_dim)
            self.cross_attn  = CrossAttentionFusion(
                embed_dim=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
            )

        if mode == "fusion_bi_cross_attention":
            f_feat_dim = _infer_feat_dim(self.fundus_encoder, in_chans=3)
            o_feat_dim = _infer_feat_dim(self.oct_encoder,    in_chans=1)
            attn_dim   = f_feat_dim
            self.bi_f_proj = nn.Identity() if f_feat_dim == attn_dim else nn.Linear(f_feat_dim, attn_dim)
            self.bi_o_proj = nn.Identity() if o_feat_dim == attn_dim else nn.Linear(o_feat_dim, attn_dim)
            self.cross_attn_f2o = CrossAttentionFusion(
                embed_dim=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
            )
            self.cross_attn_o2f = CrossAttentionFusion(
                embed_dim=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
            )

        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor(pos_weight))
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Per-label metrics: index 0 = DR_pos, index 1 = DME
        # validate_args=False avoids crashes on single-class batches / empty epochs
        self.train_dr_auc  = BinaryAUROC(validate_args=False)
        self.train_dme_auc = BinaryAUROC(validate_args=False)
        self.val_dr_auc    = BinaryAUROC(validate_args=False)
        self.val_dme_auc   = BinaryAUROC(validate_args=False)
        self.val_dr_f1     = BinaryF1Score(validate_args=False)
        self.val_dme_f1    = BinaryF1Score(validate_args=False)
        self.val_dr_acc    = BinaryAccuracy(validate_args=False)
        self.val_dme_acc   = BinaryAccuracy(validate_args=False)

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

        if "fundus" not in batch:
            raise KeyError(f"mode='{self.mode}' requires batch['fundus'] but key is missing.")
        if "oct" not in batch:
            raise KeyError(f"mode='{self.mode}' requires batch['oct'] but key is missing.")

        if self.mode == "fusion_cross_attention":
            f_feat = self.fundus_encoder.forward_features(batch["fundus"])
            o_feat = self.oct_encoder.forward_features(batch["oct"])
            q  = self.f_attn_proj(_feat_to_tokens(f_feat))  # (B, N_q,  attn_dim)
            kv = self.o_attn_proj(_feat_to_tokens(o_feat))  # (B, N_kv, attn_dim)
            fused  = self.cross_attn(q, kv)                 # (B, N_q,  attn_dim)
            pooled = fused.mean(dim=1)                      # (B, attn_dim)
            return self.fundus_head(pooled)

        if self.mode == "fusion_bi_cross_attention":
            f_feat = self.fundus_encoder.forward_features(batch["fundus"])
            o_feat = self.oct_encoder.forward_features(batch["oct"])
            f_tok = self.bi_f_proj(_feat_to_tokens(f_feat))  # (B, N_f, attn_dim)
            o_tok = self.bi_o_proj(_feat_to_tokens(o_feat))  # (B, N_o, attn_dim)
            f2o = self.cross_attn_f2o(f_tok, o_tok)          # (B, N_f, attn_dim)
            o2f = self.cross_attn_o2f(o_tok, f_tok)          # (B, N_o, attn_dim)
            f_vec = f2o.mean(dim=1)                           # (B, attn_dim)
            o_vec = o2f.mean(dim=1)                           # (B, attn_dim)
            return self.fundus_head(0.5 * (f_vec + o_vec))

        # fusion (concat)
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

        probs    = torch.sigmoid(logits)
        prob_dr  = probs[:, 0]
        prob_dme = probs[:, 1]
        lbl_dr   = labels[:, 0].long()
        lbl_dme  = labels[:, 1].long()

        # Accumulate metrics; epoch-level logging happens in on_*_epoch_end
        if stage == "train":
            self.train_dr_auc.update(prob_dr, lbl_dr)
            self.train_dme_auc.update(prob_dme, lbl_dme)
        else:
            self.val_dr_auc.update(prob_dr,   lbl_dr)
            self.val_dme_auc.update(prob_dme, lbl_dme)
            self.val_dr_f1.update(prob_dr,    lbl_dr)
            self.val_dme_f1.update(prob_dme,  lbl_dme)
            self.val_dr_acc.update(prob_dr,   lbl_dr)
            self.val_dme_acc.update(prob_dme, lbl_dme)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    def _safe_compute_log(self, metric, name: str, prog_bar: bool = False) -> None:
        """Compute metric, log it, reset it. Logs NaN if compute fails (e.g. single-class epoch)."""
        try:
            val = metric.compute()
            self.log(name, val, prog_bar=prog_bar, sync_dist=True)
        except Exception:
            self.log(name, float("nan"), prog_bar=prog_bar, sync_dist=True)
        finally:
            metric.reset()

    def on_train_epoch_end(self) -> None:
        self._safe_compute_log(self.train_dr_auc,  "train_dr_auc",  prog_bar=False)
        self._safe_compute_log(self.train_dme_auc, "train_dme_auc", prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        self._safe_compute_log(self.val_dr_auc,  "val_dr_auc",  prog_bar=True)
        self._safe_compute_log(self.val_dme_auc, "val_dme_auc", prog_bar=True)
        self._safe_compute_log(self.val_dr_f1,   "val_dr_f1",   prog_bar=True)
        self._safe_compute_log(self.val_dme_f1,  "val_dme_f1",  prog_bar=True)
        self._safe_compute_log(self.val_dr_acc,  "val_dr_acc",  prog_bar=False)
        self._safe_compute_log(self.val_dme_acc, "val_dme_acc", prog_bar=False)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        encoder_params = [p for n, p in self.named_parameters() if "encoder" in n]
        head_params    = [p for n, p in self.named_parameters() if "encoder" not in n]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.hparams.lr * 0.1},
                {"params": head_params,    "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
