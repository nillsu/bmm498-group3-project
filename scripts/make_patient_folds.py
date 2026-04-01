import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

K = 5
SEED = 42

# ZIP'ten gelen v2 CSV'leri repo içindeki dataset_csv klasörüne koyacağız
TRAIN_CSV = os.path.join("dataset_csv", "pairs_train_oct_v2.csv")
VAL_CSV   = os.path.join("dataset_csv", "pairs_val_oct_v2.csv")
TEST_CSV  = os.path.join("dataset_csv", "pairs_test_oct_v2.csv")

OUT_DIR = "splits"

def patient_set(df):
    return set(df["patient_id"].astype(str).unique().tolist())


def _add_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns required by MultimodalEyeDataset / MultimodalDataModule."""
    df = df.copy()
    df["sample_id"]  = df["patient_id"].astype(str) + "_" + df["eye"].astype(str)
    df["fundus_rel"] = df["fundus_preprocessed"]
    df["oct_rel"]    = df["oct_preprocessed_v2"]
    return df

def main():
    # --- Dosyaları oku
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # pool = train + val
    df_pool = pd.concat([df_train, df_val], ignore_index=True)

    # --- Gerekli kolon kontrolü
    if "patient_id" not in df_pool.columns:
        raise RuntimeError("CSV must include column: patient_id")

    # Label kolon adı: EĞER farklıysa burayı değiştir
    LABEL_COL = "y3_id"
    if LABEL_COL not in df_pool.columns:
        raise RuntimeError(f"CSV must include label column: {LABEL_COL}. If your label column name is different, edit LABEL_COL.")

    # --- Leakage check: test patients pool ile çakışmasın
    pool_pat = patient_set(df_pool)
    test_pat = patient_set(df_test)
    overlap_test = pool_pat.intersection(test_pat)
    if overlap_test:
        raise RuntimeError(f"DATA LEAKAGE: test patients overlap with train/val pool: {len(overlap_test)}")

    y = df_pool[LABEL_COL].astype(int)
    groups = df_pool["patient_id"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    # sabit test
    _add_required_cols(df_test).to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_pool, y, groups)):
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df_tr = df_pool.iloc[tr_idx].reset_index(drop=True)
        df_va = df_pool.iloc[va_idx].reset_index(drop=True)

        # train/val patient overlap check
        tr_pat = patient_set(df_tr)
        va_pat = patient_set(df_va)
        overlap = tr_pat.intersection(va_pat)
        if overlap:
            raise RuntimeError(f"DATA LEAKAGE: fold {fold} train/val patients overlap: {len(overlap)}")

        _add_required_cols(df_tr).to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        _add_required_cols(df_va).to_csv(os.path.join(fold_dir, "val.csv"), index=False)

        print(f"Fold {fold} OK | train rows={len(df_tr)} val rows={len(df_va)} "
              f"| train patients={len(tr_pat)} val patients={len(va_pat)}")

    print("All folds created under:", OUT_DIR)

if __name__ == "__main__":
    main()
