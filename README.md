# CP Outcome Prediction — Deep Biomarkers

Code accompanying the thesis *"Deep biomarkers for cerebral palsy outcome prediction from infant wearable sensor data"*. The pipeline compares three families of representations — hand-engineered signal features, VQ-VAE symbolic tokens, and foundation-model embeddings — under a rigorous nested cross-validation protocol at three temporal scales (window, session, infant).

## Setup

```bash
uv sync
```

Requires Python ≥ 3.11. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

---

## Dataset Structure

Point `src/path.py` at your data root by setting the environment variable:

```bash
export CP_DATA_DIR=/path/to/your/data
```

Or edit `YOUR_DATA_DIR` directly in `src/path.py`.

The expected directory layout is:

```
$CP_DATA_DIR/
├── aligned/                         # IMU sessions with known outcome
│   └── {infant}_seq-aligned_{session}.npz
├── unlabeled/                       # (optional) IMU sessions without outcome label
│   └── {infant}_seq-aligned_{session}.npz
├── tokenized/                       # VQ-VAE token sequences (one dir per candidate)
│   └── {candidate_dir}/
│       ├── manifest.csv
│       └── {infant}_{session}.npy
└── clinical_outcome.csv             # outcome labels and infant metadata
```

### IMU session files (`aligned/`, `unlabeled/`)

Each `.npz` file contains one recording session for one infant. Arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `acc` | `(N, 3)` float32 | Accelerometer — x, y, z axes at 100 Hz |
| `gyr` | `(N, 3)` float32 | Gyroscope — x, y, z axes at 100 Hz |
| `pressure` | `(N,)` float32 | Pressure sensor at 100 Hz |

**Filename format:** `{infant_id}_seq-aligned_{YYYYMMDD-HHMM}.npz`
Example: `I001_seq-aligned_20240115-1000.npz`

### Clinical outcome file (`clinical_outcome.csv`)

Required columns (exact names, including the trailing space on the T0 date column):

| Column | Description |
|--------|-------------|
| `ID` | Infant identifier — matches `{infant_id}` in session filenames |
| `outcome: A,T` | `T` = typical, `A` = atypical (CP risk) |
| `gender` | `m` or `f` |
| `Gestational Age (weeks); no prematurity if ≥37` | e.g. `39+0` |
| `date of assessment (T0) ` | DD/MM/YY — **note the trailing space in the column name** |
| `correct age at T0 (months)` | Comma decimal, e.g. `4,0` |

### VQ-VAE token files (`tokenized/`)

Produced by running the VQ-VAE tokenisation pipeline (not included here). The default subdirectory names expected under `tokenized/` are `vqvae_grouped`, `vqvae_cm`, and `vqvae_cross_group`. Override with env vars if your directories have different names:

```bash
export VQVAE_GROUPED_DIR=my_grouped_run
export VQVAE_CM_DIR=my_cm_run
export VQVAE_CROSS_GROUP_DIR=my_cross_group_run
```

Each candidate subdirectory contains:

**`manifest.csv`** — one row per session:

| Column | Description |
|--------|-------------|
| `infant` | Infant identifier |
| `session` | Session string (`YYYYMMDD-HHMM`) |
| `outcome` | `1` = atypical, `0` = typical |
| `file` | Filename of the `.npy` token array, relative to this directory |

**`{infant}_{session}.npy`** — 1-D `int64` array of interleaved token indices.
Token encoding: position `i` belongs to sensor group `g = i % 3` (acc / gyr / pres) and carries a value in `[g × 512,  (g+1) × 512)`. Total vocabulary: 1536 (3 groups × 512 codes). Token rate: ≈ 18.75 tokens/second.

### Foundation-model caches (`results/new_experiments/`)

Pre-computed by the commands in step 4 below. Stored as compressed `.npz` files:

| File | Keys | Value shape per infant |
|------|------|----------------------|
| `hubert_embeddings.npz` | `{infant_id}` | `(N_windows, 768)` float32 |
| `mantis_embeddings.npz` | `{infant_id}` | `(N_windows, 256)` float32 |
| `hubert_win5s_tokens.npz` | `{infant_id}`, `_centroids_` | `(N_windows × 3,)` int32 ; `(1536, 64)` float32 |
| `mantis_win5s_tokens.npz` | `{infant_id}`, `_centroids_` | `(N_windows × 3,)` int32 ; `(1536, 64)` float32 |

`N_windows` = number of non-overlapping 30 s windows in the recording (for 30 s hop).

### VQ-VAE checkpoints (optional — Exp 3 only)

Exp 3 biomarker analysis uses the VQ-VAE codebook to compute trajectory-based metrics. Point to your trained checkpoints via env vars:

```bash
export VQVAE_GROUPED_CKPT=/path/to/vqvae_grouped.pt
export VQVAE_CM_CKPT=/path/to/vqvae_cm.pt
export VQVAE_CROSS_GROUP_CKPT=/path/to/vqvae_cross_group.pt
```

If unset, Exp 3 runs with a placeholder codebook (diversity and dynamics metrics are unaffected; only codebook-trajectory metrics are dummy).

---

## Step-by-Step Guide

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure data path

```bash
export CP_DATA_DIR=/path/to/your/data
```

Or edit `YOUR_DATA_DIR` in `src/path.py`.

### 3. Verify data loading

```bash
uv run python - <<'EOF'
from src.dataset import OutcomeDataset
ds = OutcomeDataset()
print(f"Loaded {len(ds)} infants")
for p in ds._pools[:3]:
    print(f"  {p['infant']}  label={p['label']}  chunks={p['chunks'].shape}")
EOF
```

### 4. Pre-compute foundation-model caches  *(skip for signal/symbolic only)*

```bash
# 30 s window embeddings used by Exp 1/2/3 (FM-embedding track)
uv run python new_experiments.py --precompute-embeddings

# 5 s window FM tokens used by Exp 1/2/3 (FM-token track)
uv run python new_experiments.py --precompute-tokens hubert_win5s
uv run python new_experiments.py --precompute-tokens mantis_win5s
```

Results are saved to `results/new_experiments/`.

### 5. Run Experiment 1 — representation comparison

Nested CV (N_OUTER=5, N_INNER=3, N_SEEDS=10) across all representations and scales.

```bash
uv run python new_experiments.py --exp 1
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--candidate signal` | Signal features only (no FM caches needed) |
| `--candidate grouped` | One VQ-VAE candidate only |
| `--model lr` | Logistic regression only (fast smoke test) |
| `--scale infant` | Infant-level scale only |

Results: `results/new_experiments/exp1_*.csv`

### 6. Run Experiment 2 — early prediction

AUC as a function of growing data budget (fraction of sessions, age cutoff, duration cutoff).

```bash
uv run python new_experiments.py --exp 2
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--exp2-config best` | Use best model/params from Exp 1 |
| `--model lr` | Logistic regression only |

Results: `results/new_experiments/exp2_all.csv`

### 7. Run Experiment 3 — token biomarkers

Session-level biomarkers (diversity, dynamics, intrinsic dimensionality) compared between outcome groups via Mann-Whitney U + Benjamini-Hochberg FDR correction.

```bash
uv run python new_experiments.py --exp 3
```

Results: `results/new_experiments/exp3_bm_*.csv`, `exp3_stats_*.csv`

### 8. Collect and aggregate results

After running all experiments (or partial slices), merge CSVs and generate summary plots:

```bash
uv run python new_experiments.py --exp 1 --collect
uv run python new_experiments.py --exp 2 --collect
uv run python new_experiments.py --exp 3 --collect
```

Outputs: merged CSVs + figures in `results/new_experiments/figs/`.

---

## Repository Layout

```
new_experiments.py          # Exp 1 / 2 / 3 entry point
run_experiment.py           # Single nested-CV run (signal or symbolic)
src/
├── dataset.py              # OutcomeDataset — loads .npz sessions, returns infant pools
├── path.py                 # Data path configuration (edit or set CP_DATA_DIR)
├── exp_config.py           # Shared constants: windows, vocab, CV params, paths
├── signal_outcome.py       # 140-dim hand-engineered feature extraction
├── tfidf_outcome.py        # LognormSmoothTfidf on VQ-VAE token sequences
├── repr_signal.py          # Build signal feature matrices (all 3 scales)
├── repr_symbolic.py        # Build TF-IDF matrices + session token biomarkers
├── repr_fm.py              # Build FM embedding / token matrices + biomarkers
├── outcome_cv.py           # Nested GroupKFold CV loop + HPO
├── embedding_extractor.py  # FM embedding extraction and cache I/O
├── vqvae.py                # GroupedVQVAE architecture + codebook utilities
└── utils.py                # Infant metadata parsing (age, gestational age)
```
