"""
MultimodalEyeDataset — loads fundus + OCT image pairs from a pre-filtered DataFrame.

The caller is responsible for:
  - reading splits_clean.csv
  - filtering rows to the desired split
  - building transforms via transforms.get_fundus_transforms / get_oct_transforms

CSV columns expected in df:
    sample_id, split, fundus_rel, oct_rel, DR_pos, DME

__getitem__ returns:
    {
        "sample_id": str,
        "labels":    FloatTensor (2,)   # [DR_pos, DME]
        "fundus":    FloatTensor (3,H,W)  # present when mode in {"fundus", "fusion"}
        "oct":       FloatTensor (1,H,W)  # present when mode in {"oct",    "fusion"}
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_VALID_MODES = {"fundus", "oct", "fusion"}


class MultimodalEyeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        mode: Literal["fundus", "oct", "fusion"],
        transform_fundus: Optional[Callable] = None,
        transform_oct: Optional[Callable] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'")

        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(
                f"data_root not found: {data_root}\n"
                "Check that Google Drive is mounted and the path is correct."
            )

        if mode in ("fundus", "fusion") and transform_fundus is None:
            raise ValueError(f"transform_fundus is required for mode='{mode}'")
        if mode in ("oct", "fusion") and transform_oct is None:
            raise ValueError(f"transform_oct is required for mode='{mode}'")

        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.mode = mode
        self.transform_fundus = transform_fundus
        self.transform_oct = transform_oct

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample_id: str = str(row["sample_id"])

        # --- labels ----------------------------------------------------------
        dr_val, dme_val = row["DR_pos"], row["DME"]
        if pd.isna(dr_val) or pd.isna(dme_val):
            raise ValueError(
                f"[{sample_id}] NaN label detected: DR_pos={dr_val}, DME={dme_val}"
            )
        labels = torch.tensor(
            [float(dr_val), float(dme_val)],
            dtype=torch.float32,
        )

        out: dict = {"sample_id": sample_id, "labels": labels}

        # --- fundus ----------------------------------------------------------
        if self.mode in ("fundus", "fusion"):
            fundus_path = self.data_root / str(row["fundus_rel"])
            if not fundus_path.exists():
                raise FileNotFoundError(
                    f"[{sample_id}] Fundus image not found: {fundus_path}"
                )
            img = Image.open(fundus_path).convert("RGB")
            out["fundus"] = self.transform_fundus(img)

        # --- OCT -------------------------------------------------------------
        if self.mode in ("oct", "fusion"):
            oct_path = self.data_root / str(row["oct_rel"])
            if not oct_path.exists():
                raise FileNotFoundError(
                    f"[{sample_id}] OCT image not found: {oct_path}"
                )
            img = Image.open(oct_path).convert("L")
            out["oct"] = self.transform_oct(img)

        return out
