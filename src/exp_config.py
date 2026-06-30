"""Shared constants for experiments and representation builders."""

import os

# Each candidate needs:
#   dir  — subdirectory name under $CP_DATA_DIR/tokenized/ where .npy token files live
#   ckpt — path to the trained VQ-VAE checkpoint (.pt); used only for biomarker analysis
#           (Exp 3 codebook lookup). Leave unset to skip codebook-based biomarkers.
# Override via env vars: VQVAE_GROUPED_DIR, VQVAE_GROUPED_CKPT, etc.
CANDIDATES = {
    "grouped": dict(
        dir=os.environ.get("VQVAE_GROUPED_DIR", "vqvae_grouped"),
        ckpt=os.environ.get("VQVAE_GROUPED_CKPT"),
    ),
    "cm": dict(
        dir=os.environ.get("VQVAE_CM_DIR", "vqvae_cm"),
        ckpt=os.environ.get("VQVAE_CM_CKPT"),
    ),
    "cross_group": dict(
        dir=os.environ.get("VQVAE_CROSS_GROUP_DIR", "vqvae_cross_group"),
        ckpt=os.environ.get("VQVAE_CROSS_GROUP_CKPT"),
        load_kwargs=dict(hidden_dim=64, latent_dim=128, n_tcn_layers=4),
    ),
}

ENCODERS    = ["hubert", "mantis"]
ENCODERS_5S = ["hubert_win5s", "mantis_win5s"]

WIN_SEC        = 30.0
HOP_SEC        = 30.0
VOCAB_SIZE     = 1536
N_GROUPS       = 3
GROUP_VOCAB    = VOCAB_SIZE // N_GROUPS
GROUP_NAMES    = ["acc", "gyr", "pres"]
GROUP_OFFSETS  = [g * GROUP_VOCAB for g in range(N_GROUPS)]
TOKENS_PER_SEC = 6.25 * N_GROUPS

TOKENISE_CHOICES = ENCODERS + ["mantis_svdg", "mantis_svdc"]

THRESHOLD     = 0.5
N_OUTER       = 5
N_INNER       = 3
N_SEEDS       = 10
SEED          = 42
SEEDS         = list(range(10))
MODELS        = ["lr", "svc", "rf", "xgb"]
MODELS_WINDOW = ["lr", "svc", "rf", "xgb"]

OUT_DIR = "results/new_experiments"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)


def session_age(infant: str, session: str) -> float:
    from src.utils import get_infant_info
    try:
        return float(get_infant_info(infant, session)[1])
    except Exception:
        return float("nan")


def cohen_d(x, y):
    import numpy as np
    nx, ny = len(x), len(y)
    pool = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return (x.mean() - y.mean()) / (pool + 1e-8)
