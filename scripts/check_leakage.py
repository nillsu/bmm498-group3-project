import pandas as pd
from pathlib import Path

splits = Path("splits")

# --- Load test patients
test = pd.read_csv(splits / "test.csv")
test_pat = set(test["patient_id"].astype(str))

ok = True

for k in range(5):
    tr = pd.read_csv(splits / f"fold_{k}" / "train.csv")
    va = pd.read_csv(splits / f"fold_{k}" / "val.csv")

    tr_pat = set(tr["patient_id"].astype(str))
    va_pat = set(va["patient_id"].astype(str))

    inter_tr_va = tr_pat & va_pat
    if inter_tr_va:
        ok = False
        print(f"[LEAK] fold_{k}: train ∩ val patient overlap = {len(inter_tr_va)}")

    inter_with_test = (tr_pat | va_pat) & test_pat
    if inter_with_test:
        ok = False
        print(f"[LEAK] fold_{k}: (train ∪ val) ∩ test overlap = {len(inter_with_test)}")

    print(f"fold_{k}: train_pat={len(tr_pat)} val_pat={len(va_pat)} test_pat={len(test_pat)}")

print("\nRESULT:", "NO PATIENT LEAKAGE [OK]" if ok else "LEAKAGE FOUND [FAIL]")

# Optional: check val sets overlap across folds (ideally 0 for strict group-kfold)
val_sets = []
for k in range(5):
    va = pd.read_csv(splits / f"fold_{k}" / "val.csv")
    val_sets.append(set(va["patient_id"].astype(str)))

overlaps = 0
for i in range(5):
    for j in range(i + 1, 5):
        overlaps += len(val_sets[i] & val_sets[j])

print("Val set pairwise overlaps (ideal 0):", overlaps)
