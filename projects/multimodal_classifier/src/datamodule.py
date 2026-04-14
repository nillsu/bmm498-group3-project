"""
MultimodalDataModule — PyTorch Lightning DataModule for fundus + OCT pairs.

Reads splits_clean.csv once in setup(); splits by the "split" column into
train / val / test.  Passes pre-built transforms and the correct mode to
MultimodalEyeDataset.

Usage:
    dm = MultimodalDataModule(
        csv_path="...splits_clean.csv",
        data_root="...bmm498_data",
        mode="fusion",
    )
    dm.setup("fit")
    for batch in dm.train_dataloader():
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducible augmentation."""
    import random
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy column names to the canonical names expected by MultimodalEyeDataset.

    Legacy → canonical mappings applied only when the canonical name is absent:
        patient_id + eye    → sample_id
        fundus_preprocessed → fundus_rel   (values: {split}/fundus/{file})
        oct_preprocessed_v2 → oct_rel      (values: {split}/oct_real/{file}  — real OCT)

    Note: oct_preprocessed (values: {split}/oct/{file}) is an older v1 preprocessing
    of the real OCT that is no longer used in training and is intentionally not mapped.
    batch["oct"] holds real OCT for real-OCT modes and pseudo-OCT for pseudo modes.
    """
    if df.empty:
        return df
    df = df.copy()
    if "sample_id" not in df.columns:
        if "patient_id" in df.columns and "eye" in df.columns:
            df["sample_id"] = df["patient_id"].astype(str) + "_" + df["eye"].astype(str)
    if "fundus_rel" not in df.columns and "fundus_preprocessed" in df.columns:
        df["fundus_rel"] = df["fundus_preprocessed"]
    if "oct_rel" not in df.columns and "oct_preprocessed_v2" in df.columns:
        df["oct_rel"] = df["oct_preprocessed_v2"]
    return df


from .dataset import MultimodalEyeDataset
from .transforms import get_fundus_transforms, get_oct_transforms


class MultimodalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        mode: str,
        csv_path: str | Path | None = None,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        pin_memory: bool = True,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
    ) -> None:
        super().__init__()
        _VALID_MODES = {
            "fundus", "oct", "fusion",
            "fusion_cross_attention", "fusion_bi_cross_attention",
            "pseudo_oct", "fusion_pseudo",
            "fusion_cross_attention_pseudo", "fusion_bi_cross_attention_pseudo",
        }
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode={mode!r} is not valid. Expected one of {_VALID_MODES}."
            )
        self._use_fold_csvs = train_csv is not None and val_csv is not None
        if not self._use_fold_csvs:
            if csv_path is None:
                raise ValueError(
                    "Provide csv_path for single-CSV mode, or both train_csv and val_csv for fold mode."
                )
            if not Path(csv_path).exists():
                raise FileNotFoundError(
                    f"CSV not found: {csv_path!r}. Check that the path is correct and the file exists."
                )
        self.csv_path  = Path(csv_path)  if csv_path  is not None else None
        self.data_root = Path(data_root)
        self.mode = mode
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # Only pin memory when CUDA is available; pinning on CPU is wasteful
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.train_csv = Path(train_csv) if train_csv is not None else None
        self.val_csv   = Path(val_csv)   if val_csv   is not None else None
        self.test_csv  = Path(test_csv)  if test_csv  is not None else None

        # populated in setup()
        self._df: Optional[pd.DataFrame] = None
        self.train_ds: Optional[MultimodalEyeDataset] = None
        self.val_ds: Optional[MultimodalEyeDataset] = None
        self.test_ds: Optional[MultimodalEyeDataset] = None

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        # Columns always required regardless of mode
        _ALWAYS_COLS = {"sample_id", "DR_pos", "DME"}
        # Per-mode extra column requirements
        _MODE_COLS: dict[str, set] = {
            "fundus":                            {"fundus_rel"},
            "oct":                               {"oct_rel"},
            "fusion":                            {"fundus_rel", "oct_rel"},
            "fusion_cross_attention":            {"fundus_rel", "oct_rel"},
            "fusion_bi_cross_attention":         {"fundus_rel", "oct_rel"},
            "pseudo_oct":                        {"oct_pseudo_rel"},
            "fusion_pseudo":                     {"fundus_rel", "oct_pseudo_rel"},
            "fusion_cross_attention_pseudo":     {"fundus_rel", "oct_pseudo_rel"},
            "fusion_bi_cross_attention_pseudo":  {"fundus_rel", "oct_pseudo_rel"},
        }

        def _check_cols(df: pd.DataFrame, extra: set, source: str) -> None:
            required = _ALWAYS_COLS | _MODE_COLS.get(self.mode, set()) | extra
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"{source} is missing required columns: {sorted(missing)}. "
                    f"Found columns: {list(df.columns)}"
                )

        if self._use_fold_csvs:
            df_train = _normalize_columns(pd.read_csv(self.train_csv))
            df_val   = _normalize_columns(pd.read_csv(self.val_csv))
            df_test  = (
                _normalize_columns(pd.read_csv(self.test_csv))
                if self.test_csv is not None and self.test_csv.exists()
                else pd.DataFrame()
            )
            _check_cols(df_train, set(), "train_csv")
            _check_cols(df_val,   set(), "val_csv")
        else:
            # Read CSV exactly once; re-entrant if called multiple times
            if self._df is None:
                self._df = _normalize_columns(pd.read_csv(self.csv_path))
            _check_cols(self._df, {"split"}, "csv_path")
            df = self._df
            df_train = df[df["split"] == "train"].reset_index(drop=True)
            df_val   = df[df["split"] == "val"].reset_index(drop=True)
            df_test  = df[df["split"] == "test"].reset_index(drop=True)

        tf_fundus_train = get_fundus_transforms(train=True,  image_size=self.image_size)
        tf_fundus_eval  = get_fundus_transforms(train=False, image_size=self.image_size)
        tf_oct_train    = get_oct_transforms   (train=True,  image_size=self.image_size)
        tf_oct_eval     = get_oct_transforms   (train=False, image_size=self.image_size)

        # resolve per-mode transform kwargs once
        def _transform_kwargs(is_train: bool) -> dict:
            tf_f = tf_fundus_train if is_train else tf_fundus_eval
            tf_o = tf_oct_train    if is_train else tf_oct_eval
            if self.mode == "fundus":
                return {"transform_fundus": tf_f, "transform_oct": None}
            if self.mode in ("oct", "pseudo_oct"):
                return {"transform_fundus": None, "transform_oct": tf_o}
            # fusion, fusion_cross_attention, fusion_bi_cross_attention, fusion_pseudo
            return {"transform_fundus": tf_f, "transform_oct": tf_o}

        self.train_ds = MultimodalEyeDataset(
            df_train, self.data_root, self.mode, **_transform_kwargs(True)
        )
        self.val_ds = MultimodalEyeDataset(
            df_val, self.data_root, self.mode, **_transform_kwargs(False)
        )
        self.test_ds = (
            MultimodalEyeDataset(
                df_test, self.data_root, self.mode, **_transform_kwargs(False)
            )
            if len(df_test) > 0
            else None
        )

    # ------------------------------------------------------------------
    def _make_loader(self, dataset: MultimodalEyeDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        # persistent_workers is unreliable on Windows; disable it there
        persistent = self.num_workers > 0 and sys.platform != "win32"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent,
            worker_init_fn=_worker_init_fn if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("train_ds is None — call setup() before train_dataloader().")
        return self._make_loader(self.train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("val_ds is None — call setup() before val_dataloader().")
        return self._make_loader(self.val_ds, shuffle=False, drop_last=False)

    def test_dataloader(self):
        if self.test_ds is None:
            return []
        return self._make_loader(self.test_ds, shuffle=False, drop_last=False)
