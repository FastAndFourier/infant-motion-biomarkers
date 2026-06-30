# Dataset Format

All data lives under `$CP_DATA_DIR` (set via env var or `src/path.py`).

```
$CP_DATA_DIR/
├── aligned/                 # labeled IMU sessions (known CP outcome)
├── unlabeled/               # unlabeled IMU sessions (used for VQ-VAE pretraining)
├── tokenized/               # VQ-VAE token sequences, one subdir per model
│   ├── vqvae_grouped/
│   ├── vqvae_cm/
│   └── vqvae_cross_group/
└── clinical_outcome.csv
```

---

## IMU session files

**Location:** `aligned/` and `unlabeled/`

**Filename:** `{infant_id}_seq-aligned_{YYYYMMDD-HHMM}.npz`
Example: `I001_seq-aligned_20240115-1000.npz`

Each file is a NumPy `.npz` archive with:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `acc` | `(N, 3)` | float32 | Accelerometer x/y/z at 100 Hz |
| `gyr` | `(N, 3)` | float32 | Gyroscope x/y/z at 100 Hz |
| `pressure` | `(N,)` | float32 | Pressure sensor at 100 Hz |

`N` = number of samples (90 s session at 100 Hz → N = 9000).

---

## Clinical outcome file

**Location:** `clinical_outcome.csv`

Required columns — **exact names matter**, including the trailing space on the T0 date column:

| Column name | Example | Description |
|-------------|---------|-------------|
| `ID` | `I001` | Infant identifier, must match `{infant_id}` in filenames |
| `outcome: A,T` | `T` | `T` = typical, `A` = atypical (CP risk) |
| `gender` | `m` | `m` or `f` |
| `Gestational Age (weeks); no prematurity if ≥37` | `39+0` | Format: `WW+D` |
| `date of assessment (T0) ` | `15/01/24` | DD/MM/YY — **note trailing space** |
| `correct age at T0 (months)` | `4,0` | Comma as decimal separator |

---

## VQ-VAE token files

**Location:** `tokenized/{candidate_dir}/`

Default directory names: `vqvae_grouped`, `vqvae_cm`, `vqvae_cross_group`.
Override with env vars `VQVAE_GROUPED_DIR`, `VQVAE_CM_DIR`, `VQVAE_CROSS_GROUP_DIR`.

### `manifest.csv`

One row per session:

| Column | Description |
|--------|-------------|
| `infant` | Infant identifier |
| `session` | Session string (`YYYYMMDD-HHMM`) |
| `outcome` | `1` = atypical, `0` = typical |
| `file` | Filename of the `.npy` token array, relative to this directory |

### `{infant}_{session}.npy`

1-D `int64` array of interleaved token indices. Token encoding:

- Position `i` belongs to sensor group `g = i % 3` (0=acc, 1=gyr, 2=pres)
- Value at position `i` ∈ `[g × 512, (g+1) × 512)`
- Total vocabulary: **1536** (3 groups × 512 codes)
- Token rate: **18.75 tokens/second** (6.25 tokens/s/group at 50 Hz with 8× stride)

---

## Foundation-model embedding caches

**Location:** `results/new_experiments/` (auto-created)

Pre-computed via `new_experiments.py --precompute-*`. Compressed `.npz` files:

| File | Key | Value shape |
|------|-----|-------------|
| `hubert_embeddings.npz` | `{infant_id}` | `(N_windows, 768)` float32 |
| `mantis_embeddings.npz` | `{infant_id}` | `(N_windows, 256)` float32 |
| `hubert_win5s_tokens.npz` | `{infant_id}` | `(N_windows × 3,)` int32 |
| `hubert_win5s_tokens.npz` | `_centroids_` | `(1536, 64)` float32 |
| `mantis_win5s_tokens.npz` | `{infant_id}` | `(N_windows × 3,)` int32 |
| `mantis_win5s_tokens.npz` | `_centroids_` | `(1536, 64)` float32 |

`N_windows` = number of non-overlapping 30 s windows per recording.

---

## VQ-VAE checkpoints (Exp 3 codebook metrics)

Exp 3 computes trajectory-based biomarkers using the VQ-VAE codebook. Point to trained `.pt` checkpoints:

```bash
export VQVAE_GROUPED_CKPT=/path/to/vqvae_grouped.pt
export VQVAE_CM_CKPT=/path/to/vqvae_cm.pt
export VQVAE_CROSS_GROUP_CKPT=/path/to/vqvae_cross_group.pt
```

If unset, Exp 3 uses a placeholder codebook — diversity and dynamics metrics still run correctly, but codebook-trajectory metrics (speed, smoothness) are not meaningful.
