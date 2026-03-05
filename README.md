# bmm498-group3-project

OCT generator, fundus-only pseudo-OCT pipeline, and multimodal fusion classification for diabetic retinopathy (DR) and diabetic macular oedema (DME) detection.

---

## Project overview

The classifier supports three **modes**:

| Mode | Input | Use case |
|------|-------|----------|
| `fundus` | Colour fundus image (3-ch) | Fundus-only baseline |
| `oct` | OCT volume slice (1-ch) | OCT-only baseline |
| `fusion` | Both | Mid-level fusion (default) |

Each mode trains an independent ResNet-18 encoder per branch; the fusion mode concatenates both feature vectors before a shared head. All modes predict two binary labels: **DR\_pos** and **DME**.

---

## Folder structure

```
bmm498-group3-project/
├── projects/
│   └── multimodal_classifier/
│       ├── src/
│       │   ├── dataset.py        # MultimodalEyeDataset
│       │   ├── datamodule.py     # MultimodalDataModule (Lightning)
│       │   ├── model.py          # MultimodalClassifier (Lightning)
│       │   └── transforms.py     # fundus / OCT augmentations
│       └── scripts/
│           ├── train.py              # single-fold training
│           ├── predict.py            # checkpoint → predictions CSV
│           ├── run_ablation.py       # fundus+oct+fusion in one shot
│           └── run_kfold_experiments.py  # full k-fold loop
├── splits/                       # fold CSVs (gitignored)
├── external/                     # third-party code (do not edit)
└── README.md
```

> **Datasets, checkpoints, and logs are listed in `.gitignore`** and will not be committed.

---

## Data expectations

### Single-CSV mode (`splits_clean.csv`)

Required columns:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique patient-eye identifier (e.g. `1221_OD`) |
| `fundus_rel` | Relative path to fundus image from `data_root` |
| `oct_rel` | Relative path to OCT slice from `data_root` |
| `DR_pos` | Binary label — diabetic retinopathy (0/1) |
| `DME` | Binary label — diabetic macular oedema (0/1) |
| `split` | `train` / `val` / `test` (**required in single-CSV mode only**) |

> The current `splits_clean.csv` may not include a `test` split. Use `--split val` in `predict.py` to export from the validation set instead.

### Fold-CSV mode

Each fold directory contains two files with **no `split` column needed**:

```
splits/
  fold_0/train.csv
  fold_0/val.csv
  fold_1/train.csv
  fold_1/val.csv
  ...
```

---

## Quickstart

```bash
pip install torch torchvision pytorch-lightning timm pandas
```

---

## How to run

All commands are run from the **repo root**.

### Single-fold training (single CSV)

```bash
python projects/multimodal_classifier/scripts/train.py \
  --csv_path   ./bmm498_data/splits_clean.csv \
  --data_root  ./bmm498_data \
  --output_dir ./runs/fusion_run1 \
  --mode       fusion \
  --epochs     20 \
  --batch_size 16
```

### Single-fold training (fold CSVs)

```bash
python projects/multimodal_classifier/scripts/train.py \
  --data_root  ./bmm498_data \
  --train_csv  ./splits/fold_0/train.csv \
  --val_csv    ./splits/fold_0/val.csv \
  --output_dir ./runs/fold_0_fusion \
  --mode       fusion \
  --epochs     20
```

### Ablation (fundus + oct + fusion, one fold)

```bash
python projects/multimodal_classifier/scripts/run_ablation.py \
  --data_root   ./bmm498_data \
  --output_root ./runs/ablation_fold0 \
  --train_csv   ./splits/fold_0/train.csv \
  --val_csv     ./splits/fold_0/val.csv \
  --epochs 20 --batch_size 16
```

Skip already-trained modes (resume):

```bash
python projects/multimodal_classifier/scripts/run_ablation.py \
  --data_root   ./bmm498_data \
  --output_root ./runs/ablation_fold0 \
  --train_csv   ./splits/fold_0/train.csv \
  --val_csv     ./splits/fold_0/val.csv \
  --skip_existing
```

### K-fold experiment runner

Run all folds:

```bash
python projects/multimodal_classifier/scripts/run_kfold_experiments.py \
  --data_root   ./bmm498_data \
  --splits_root ./splits \
  --output_root ./runs/kfold \
  --epochs 20 --batch_size 16
```

Run selected folds only:

```bash
python projects/multimodal_classifier/scripts/run_kfold_experiments.py \
  --data_root   ./bmm498_data \
  --splits_root ./splits \
  --output_root ./runs/kfold \
  --folds 0,2 --skip_existing
```

Dry-run (print commands, no execution):

```bash
python projects/multimodal_classifier/scripts/run_kfold_experiments.py \
  --data_root   ./bmm498_data \
  --splits_root ./splits \
  --output_root ./runs/kfold \
  --dry_run
```

Results are written to `runs/kfold/results_summary.csv` with columns: `fold, mode, checkpoint, output_dir, status`.

### Prediction export

Export from test split (falls back to val if test is absent):

```bash
python projects/multimodal_classifier/scripts/predict.py \
  --csv_path   ./bmm498_data/splits_clean.csv \
  --data_root  ./bmm498_data \
  --checkpoint ./runs/fusion_run1/checkpoints/best-epoch=07-val_loss=0.3412.ckpt \
  --mode       fusion \
  --output_csv ./runs/fusion_run1/predictions.csv
```

Export explicitly from val split:

```bash
python projects/multimodal_classifier/scripts/predict.py \
  ... \
  --split val
```

Output CSV columns: `sample_id, DR_prob, DME_prob`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: timm / pytorch_lightning` | Select the correct Python interpreter in VS Code (`Ctrl+Shift+P` → *Python: Select Interpreter*) |
| `No test split found in CSV` | Pass `--split val` to `predict.py`, or use fold CSVs which don't need a test split |
| `No fold_* directories found` | Check `--splits_root` points to the folder containing `fold_0/`, `fold_1/`, … |
| Checkpoint shows `not found` after training | Early stopping may have triggered before any checkpoint was saved; increase `--patience` or `--epochs` |
