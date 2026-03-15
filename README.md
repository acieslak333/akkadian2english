# Akkadian to English Translation Model

Fine-tuning NLLB-200 for Old Assyrian transliteration to English translation.

## 🚀 Usage Pipeline

Follow these steps in order to prepare data and train the model:

### 1. Data Cleaning
Normalizes transliterations (gap markers, subscripts, special characters) and cleans translations.
```powershell
python scripts/clean_data.py
```
*Output: `data/processed/train_cleaned.csv`, `val_cleaned.csv`, `test_cleaned.csv`*

### 2. Lexicon Normalization
Standardizes proper nouns (PN, GN, DN) and common terms using the `OA_Lexicon_eBL.csv` lexicon.
```powershell
python scripts/normalize_lexicon.py
```
*Output: `data/processed/train_cleaned_norm.csv`, `val_cleaned_norm.csv`, `test_cleaned_norm.csv`*

### 3. Model Training
Fine-tunes the NLLB-200 model using configurations from a YAML file.
```powershell
python scripts/train.py --config configs/training_config.yaml --experiment_name "my_run"
```
*Monitoring: Use TensorBoard (`tensorboard --logdir experiments`) or check `metrics_history.csv` in the experiment folder.*

---

## 📂 Scripts Overview

| Script | Description |
| :--- | :--- |
| `scripts/clean_data.py` | Core cleaning logic. Standardizes `<gap>` tags and aligns them between sources. |
| `scripts/normalize_lexicon.py` | Applies lexicon-based entity normalization to the cleaned data. |
| `scripts/train.py` | Main training script. Optimized for RTX 3060 Ti (Mixed Precision, Gradient Accumulation). |
| `scripts/verify_model.py` | Simple health check to ensure model/tokenizer loading works. |
| `scripts/analyze_chars.py` | Scans the dataset for all unique characters/tokens to identify special Akkadian symbols. |
| `scripts/analyze_published.py` | Analyzes `published_texts.csv` to find additional transliteration context. |
| `scripts/check_context.py` | Compares segment lengths between `train.csv` and `published_texts.csv`. |

## ⚙️ Configuration

Training parameters are managed in `configs/training_config.yaml`.
- **Model**: `facebook/nllb-200-distilled-600M`
- **Effective Batch Size**: 16 (4 per device × 4 accumulation steps)
- **Epochs**: 5 (default)
- **Saving/Eval**: Every 50 steps.

## 📊 Experiment Tracking

Each training run creates a unique directory in `experiments/` named:
`YYYYMMDD_HHMMSS_<experiment_name>_<model>_lr<lr>_bs<bs>`

Inside, you will find:
- `logs/`: TensorBoard logs.
- `val_samples_step_N.csv`: Decoded predictions vs actual translations for inspection.
- `metrics_history.csv`: Step-by-step BLEU, chrF++, and Geometric Mean progress.
- `checkpoint-N/`: Model weights saved periodically.
