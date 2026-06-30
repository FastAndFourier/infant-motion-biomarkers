# infant-motion-biomarker

Thesis code for CP outcome prediction from infant wearable IMU data. Compares hand-engineered signal features, VQ-VAE symbolic tokens, and foundation-model embeddings under nested cross-validation at window / session / infant scales.

See [`src/DATASET_README.md`](src/DATASET_README.md) for the full data format spec and [`experimental_plan.md`](experimental_plan.md) for the evaluation protocol.

---

## Setup

```bash
uv sync
export CP_DATA_DIR=/path/to/your/data
```

Requires Python ≥ 3.11.

---

## Pipeline

The full pipeline runs in three stages: VQ-VAE pretraining → tokenization → outcome experiments.

### Stage 1 — Train the VQ-VAE

The VQ-VAE learns a discrete tokenization of raw IMU windows. Train on your unlabeled sessions:

```bash
uv run python train_vqvae.py train --grouped \
    --ckpt models/vqvae_grouped.pt \
    --epochs 200 --batch-size 64
```

> **Note:** `train_vqvae.py` requires `UnlabeledDataset` and `LabeledDataset` to be implemented in `src/dataset.py`. These are sliding-window loaders over your `aligned/` and `unlabeled/` directories. See [`src/DATASET_README.md`](src/DATASET_README.md) for the expected data format.

Key architecture flags:

| Flag | Description |
|------|-------------|
| `--grouped` | One codebook per sensor group (acc / gyr / pres) — used in the thesis |
| `--cross-group` | Grouped TCN + cross-group attention before VQ |
| `--patch-mode grouped` | Patch-based encoder with 3 codebooks |
| `--codebook-size 512` | Codes per group (total vocab = 1536) |
| `--n-downsample 3` | Temporal stride: 8× downsampling at 50 Hz → 6.25 tokens/s/group |

### Stage 2 — Export token sequences

After training, encode each session and save token arrays + a manifest:

```bash
uv run python train_vqvae.py analyze \
    --ckpt models/vqvae_grouped.pt --grouped
```

Token files go in `$CP_DATA_DIR/tokenized/vqvae_grouped/` (one `.npy` per session + `manifest.csv`). Set `VQVAE_GROUPED_DIR` if you use a different subdirectory name.

### Stage 3 — Run outcome experiments

```bash
# Experiment 1: representation × scale comparison (nested CV)
uv run python new_experiments.py --exp 1

# Experiment 2: early prediction (AUC vs. data budget)
uv run python new_experiments.py --exp 2

# Experiment 3: token biomarkers (Mann-Whitney + FDR)
uv run python new_experiments.py --exp 3
```

Speed tip — use `--model lr` for a fast smoke test (LR ≈ 1 s / 50 folds vs RF ≈ 75 s).

Useful flags for Exp 1:

| Flag | Effect |
|------|--------|
| `--candidate signal` | Signal features only — no VQ-VAE or FM caches needed |
| `--candidate grouped` | One VQ-VAE candidate only |
| `--scale infant` | Infant-level scale only |

All results save to `results/new_experiments/`.

---

## Foundation-model representations (optional)

Pre-compute HuBERT / MANTIS embeddings before running Exp 1/2/3 with FM representations:

```bash
uv run python new_experiments.py --precompute-embeddings
uv run python new_experiments.py --precompute-tokens hubert_win5s
uv run python new_experiments.py --precompute-tokens mantis_win5s
```

---

## Repository layout

```
train_vqvae.py          # VQ-VAE training + analysis (Stage 1)
new_experiments.py      # Outcome experiments Exp 1 / 2 / 3 (Stage 3)
run_experiment.py       # Single nested-CV run
src/
├── DATASET_README.md   # Data format specification
├── vqvae.py            # GroupedVQVAE architecture
├── fused_vqvae.py      # Patch-based VQ-VAE variants
├── dataset.py          # OutcomeDataset (infant-level pools)
├── path.py             # Path configuration
├── exp_config.py       # Shared constants (windows, vocab, CV params)
├── signal_outcome.py   # 140-dim hand-engineered features
├── tfidf_outcome.py    # LognormSmoothTfidf
├── repr_signal.py      # Signal feature matrices
├── repr_symbolic.py    # TF-IDF matrices + token biomarkers
├── repr_fm.py          # FM embedding / token matrices
├── outcome_cv.py       # Nested GroupKFold CV + HPO
├── embedding_extractor.py  # FM embedding extraction
└── utils.py            # Metadata parsing
```
