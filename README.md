# DEE-GNN Quick Run (logP)

DEE-GNN is a machine learning framework for predicting molecular **logP** using graph neural networks (GNNs). It builds molecular graphs from coarse-grained `.itp` structures, trains a GNN, and enables reproducible inference with organized results.

`main.py` is the top-level entrypoint:
- `train`: builds graphs from training data, trains the GNN, and saves all outputs in a timestamped subfolder under `results/`.
- `predict`: loads a trained model and writes logP predictions to a CSV file, using outputs from a specific results subfolder.

---

## Setup

Python 3.9 or newer is recommended.

```bash
git clone <your-repository-url>
cd dee_gnn_logp_release

python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Run from the project root. Three data-splitting modes are supported.

### Mode 1: Auto-Split

Random 80/10/10 split (default):

```bash
python main.py train \
	--config config/config.json \
	--training-csv data/logp_values.csv \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_739
```

logP-stratified 80/10/10 split using 10 equal-frequency target bins:

```bash
python main.py train \
	--training-csv data/logp_values.csv \
	--data-dir data/ee_itp_739 \
	--split-method stratified \
	--stratify-bins 10 \
	--seed 121
```

Stratification keeps low, middle, and high logP ranges represented across all
three partitions. Each run saves `split_assignments.csv` with the compound,
logP, assigned split, and stratum so the partition can be audited.

### Mode 2: Custom Split

```bash
python main.py train \
	--train-data data/train.csv \
	--val-data data/val.csv \
	--test-data data/test.csv \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_739
```

You may omit `--test-data` or `--val-data` (see train-only below).

### Mode 3: Train-Only (no splitting)

```bash
python main.py train \
	--train-data data/logp_values.csv \
	--epochs 500 \
	--nbfix data/NBFIX_table \
	--data-dir data/ee_itp_739
```

### Common Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes (auto-supplied by `main.py`) | Path to config JSON |
| `--nbfix` | Yes (auto-supplied by `main.py`) | NBFIX table file |
| `--data-dir` | Yes | Directory with compound folders |
| `--training-csv` | Mode 1 only | Single CSV — auto-split 80/10/10 |
| `--split-method` | No | `random` (default) or `stratified`; applies to `--training-csv` |
| `--stratify-bins` | No | Equal-frequency logP bins for stratification (default: 10) |
| `--train-data` | Modes 2 & 3 | CSV for training set |
| `--val-data` | Optional (Mode 2) | CSV for validation set |
| `--test-data` | Optional (Mode 2) | CSV for test set |
| `--epochs` | Required for Mode 3 | Fixed epoch count (overrides config `max_epochs`) |
| `--results-dir` | No (default: `results`) | Output directory |
| `--seed` | No (default: 121) | Random seed |

> `--training-csv` and `--train-data` are mutually exclusive.

After training, outputs are saved under `results/YYYYMMDD_HHMMSS/` (`model.pth`, scalers, metrics, pred-vs-true plots, etc.).

## Included Default Model

The repository includes a ready-to-use model in `results/default_model/`. It
was trained with the default random 80/10/10 split and seed 121:

- Training: R² 0.9774, MAE 0.3020
- Validation: R² 0.7482, MAE 0.6477
- Test: R² 0.6719, MAE 0.6419

The directory contains the checkpoint, feature scalers, bead-type mapping,
configuration, split assignments, metrics, prediction CSVs, and evaluation
plots required to reproduce and inspect the run.

## Inference / Prediction

```bash
# Predict every compound in a folder
python main.py predict \
	--use-model results/default_model \
	--folder data/ee_itp_739 \
	--output predictions.csv

# Predict from explicit compound IDs
python main.py predict \
	--use-model results/default_model \
	--compounds HEaSC00031 HEaSC00033 \
	--output predictions.csv

# Predict from a CSV with a 'compound' column
python main.py predict \
	--use-model results/default_model \
	--file data/logp_values.csv \
	--output predictions.csv
```

For `--compounds` and `--file`, structures are read from `data/ee_itp_739` by
default. Use `--data-dir PATH` to select another structure directory.

Predictions are written with columns `compound,predicted_logP`. If `--output`
is omitted, they are saved as `predictions.csv` inside the selected model
directory.

---

## Bead Type Mapping Consistency

The bead type ID mapping used during training is saved as `bead_type_to_id.json` in the results subfolder. Using `--use-model` ensures this mapping is used for inference.

# Utility

## Bead Count vs logP Plot

```bash
python utils/plot_bead_count_vs_logp.py --csv data/logp_values.csv --data data/ee_itp_739/
```
