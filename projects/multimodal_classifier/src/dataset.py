"""
MultimodalEyeDataset — loads fundus + OCT image pairs from a pre-filtered DataFrame.

The caller is responsible for:
  - reading splits_clean.csv
  - filtering rows to the desired split
  - building transforms via transforms.get_fundus_transforms / get_oct_transforms

CSV columns expected in df:
    sample_id, split, fundus_rel, oct_rel, DR_pos, DME
    oct_pseudo_rel   (required only for modes: pseudo_oct, fusion_pseudo)

__getitem__ returns:
    {
        "sample_id": str,
        "labels":    FloatTensor (2,)   # [DR_pos, DME]
        "fundus":    FloatTensor (3,H,W)  # present when mode in {"fundus", "fusion*", "fusion_pseudo"}
        "oct":       FloatTensor (1,H,W)  # present when mode in {"oct", "fusion*", "pseudo_oct", "fusion_pseudo"}
    }

Pseudo-OCT modes
────────────────
  pseudo_oct      — loads pseudo-OCT only (oct_pseudo_rel column), stored in batch["oct"]
  fusion_pseudo   — loads fundus (fundus_rel) + pseudo-OCT (oct_pseudo_rel)

The batch key is "oct" in both pseudo modes so downstream model code is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_VALID_MODES = {
    "fundus",
    "oct",
    "fusion",
    "fusion_cross_attention",
    "fusion_bi_cross_attention",
    "pseudo_oct",
    "fusion_pseudo",
}

# Modes that load a fundus image
_FUNDUS_MODES = {"fundus", "fusion", "fusion_cross_attention", "fusion_bi_cross_attention", "fusion_pseudo"}
# Modes that load a real OCT image (oct_rel column)
_REAL_OCT_MODES = {"oct", "fusion", "fusion_cross_attention", "fusion_bi_cross_attention"}
# Modes that load a pseudo-OCT image (oct_pseudo_rel column)
_PSEUDO_OCT_MODES = {"pseudo_oct", "fusion_pseudo"}


class MultimodalEyeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        mode: Literal[
            "fundus", "oct", "fusion",
            "fusion_cross_attention", "fusion_bi_cross_attention",
            "pseudo_oct", "fusion_pseudo",
        ],
        transform_fundus: Optional[Callable] = None,
        transform_oct: Optional[Callable] = None,
        verify_files: bool = False,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'")

        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(
                f"data_root not found: {data_root}\n"
                "Check that Google Drive is mounted and the path is correct.\n"
                "  from google.colab import drive; drive.mount('/content/drive')"
            )

        if mode in _FUNDUS_MODES and transform_fundus is None:
            raise ValueError(f"transform_fundus is required for mode='{mode}'")
        if mode in (_REAL_OCT_MODES | _PSEUDO_OCT_MODES) and transform_oct is None:
            raise ValueError(f"transform_oct is required for mode='{mode}'")

        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.mode = mode
        self.transform_fundus = transform_fundus
        self.transform_oct = transform_oct

        if verify_files:
            self._verify_files()

    def _verify_files(self) -> None:
        """Scan the dataframe for missing image files and print a summary (does not raise)."""
        missing_fundus:      list[str] = []
        missing_oct:         list[str] = []
        missing_pseudo_oct:  list[str] = []
        for _, row in self.df.iterrows():
            sid = str(row["sample_id"])
            if self.mode in _FUNDUS_MODES:
                if not (self.data_root / str(row["fundus_rel"])).exists():
                    missing_fundus.append(sid)
            if self.mode in _REAL_OCT_MODES:
                if not (self.data_root / str(row["oct_rel"])).exists():
                    missing_oct.append(sid)
            if self.mode in _PSEUDO_OCT_MODES:
                if not (self.data_root / str(row["oct_pseudo_rel"])).exists():
                    missing_pseudo_oct.append(sid)
        total = len(self.df)
        print(
            f"[verify_files] total={total}  "
            f"missing_fundus={len(missing_fundus)}  "
            f"missing_oct={len(missing_oct)}  "
            f"missing_pseudo_oct={len(missing_pseudo_oct)}"
        )
        if missing_fundus:
            print(f"  First 5 missing fundus      : {missing_fundus[:5]}")
        if missing_oct:
            print(f"  First 5 missing oct         : {missing_oct[:5]}")
        if missing_pseudo_oct:
            print(f"  First 5 missing pseudo_oct  : {missing_pseudo_oct[:5]}")

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
        if self.mode in _FUNDUS_MODES:
            fundus_path = self.data_root / str(row["fundus_rel"])
            if not fundus_path.exists():
                raise FileNotFoundError(
                    f"[{sample_id}] Fundus image not found.\n"
                    f"  data_root  : {self.data_root}\n"
                    f"  fundus_rel : {row['fundus_rel']!r}\n"
                    f"  full path  : {fundus_path}\n"
                    "  Hint: verify Drive is mounted and 'fundus_rel' paths are correct."
                )
            img = Image.open(fundus_path).convert("RGB")
            out["fundus"] = self.transform_fundus(img)

        # --- real OCT --------------------------------------------------------
        if self.mode in _REAL_OCT_MODES:
            oct_path = self.data_root / str(row["oct_rel"])
            if not oct_path.exists():
                raise FileNotFoundError(
                    f"[{sample_id}] OCT image not found.\n"
                    f"  data_root : {self.data_root}\n"
                    f"  oct_rel   : {row['oct_rel']!r}\n"
                    f"  full path : {oct_path}\n"
                    "  Hint: verify Drive is mounted and 'oct_rel' paths are correct."
                )
            img = Image.open(oct_path).convert("L")
            out["oct"] = self.transform_oct(img)

        # --- pseudo-OCT ------------------------------------------------------
        if self.mode in _PSEUDO_OCT_MODES:
            # Fast pre-check: if the CSV already tells us this file is absent, fail
            # clearly instead of hitting a FileNotFoundError at open time.
            if "pseudo_exists" in self.df.columns:
                pe = row["pseudo_exists"]
                try:
                    is_absent = not bool(pe)
                except (TypeError, ValueError):
                    is_absent = False   # pd.NA / unknown — let the file check below decide
                if is_absent:
                    raise ValueError(
                        f"[{sample_id}] pseudo_exists=False — no pseudo-OCT image for this sample.\n"
                        f"  Use pairs_{{split}}_pseudo_available.csv to train only on available samples.\n"
                        f"  Run: python add_pseudo_oct_column.py --data_root <root> to regenerate."
                    )

            pseudo_path = self.data_root / str(row["oct_pseudo_rel"])
            if not pseudo_path.exists():
                raise FileNotFoundError(
                    f"[{sample_id}] Pseudo-OCT image not found.\n"
                    f"  data_root      : {self.data_root}\n"
                    f"  oct_pseudo_rel : {row['oct_pseudo_rel']!r}\n"
                    f"  full path      : {pseudo_path}\n"
                    "  Hint: use pairs_{split}_pseudo_available.csv (only verified rows).\n"
                    "  Run: python add_pseudo_oct_column.py --data_root <root> to regenerate."
                )
            img = Image.open(pseudo_path).convert("L")
            out["oct"] = self.transform_oct(img)

        return out
