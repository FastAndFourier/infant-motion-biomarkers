"""
Frozen feature extraction from pretrained HuBERT and MANTIS models.

Both extract per-window embeddings from raw IMU pools (OutcomeDataset._pools).
Each channel is passed independently (univariate).

Two output modes:
  - Mean-pooled across channels: {infant: (N, emb_dim)}  — for exp1/exp2
  - Per-group (acc/gyr/pres):    {infant: (N, 3, emb_dim)} — for tokenisation

Tokenisation pipeline (CPU):
  Per-group embeddings → PCA per group → k-means per group → interleaved tokens
  Produces token sequences compatible with _infant_bm() from new_experiments.py.

Usage:
    python new_experiments.py --precompute hubert mantis
    python new_experiments.py --precompute-tokens hubert mantis
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


HUBERT_DIM = 768
MANTIS_DIM = 256
MANTIS_SEQ_LEN = 512
N_CHANNELS = 7

SENSOR_GROUPS = [(0, 3), (3, 6), (6, 7)]   # acc, gyr, pressure
GROUP_NAMES = ["acc", "gyr", "pres"]
N_GROUPS = len(SENSOR_GROUPS)


# ── Per-channel extraction (internal) ────────────────────────────────────────

HUBERT_UPSAMPLE = 10  # 100Hz * 10 = 1kHz — gives ~93 HuBERT time steps per 30s window


@torch.no_grad()
def _extract_hubert_perchannel(
    pools: list[dict],
    device: str = "cpu",
    model_name: str = "facebook/hubert-base-ls960",
    batch_size: int = 4,
    upsample_factor: int = HUBERT_UPSAMPLE,
) -> dict[str, np.ndarray]:
    """Returns {infant: (N_windows, 7, 768)}.

    Upsamples each channel by upsample_factor before HuBERT to give
    the CNN feature extractor enough temporal resolution.
    """
    from transformers import HubertModel

    model = HubertModel.from_pretrained(model_name).to(device).eval()

    result: dict[str, np.ndarray] = {}
    for pool in tqdm(pools, desc="HuBERT"):
        infant = pool["infant"]
        chunks = pool["chunks"]
        N, T, _ = chunks.shape
        target_len = T * upsample_factor
        embs = np.zeros((N, N_CHANNELS, HUBERT_DIM), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = chunks[start:end]
            for ch in range(N_CHANNELS):
                sig = torch.tensor(batch[:, :, ch], dtype=torch.float32)
                sig = F.interpolate(sig.unsqueeze(1), size=target_len,
                                    mode="linear", align_corners=False).squeeze(1)
                out = model(sig.to(device)).last_hidden_state
                embs[start:end, ch] = out.mean(dim=1).cpu().numpy()
                del sig, out

        result[infant] = embs
    return result


def _load_mantis(device="cpu", ckpt_path=None):
    from mantis.architecture import Mantis8M
    if ckpt_path is not None:
        network = Mantis8M(pre_training=False, device=device)
        state = torch.load(ckpt_path, weights_only=True, map_location=device)
        network.load_state_dict(state, strict=False)
    else:
        network = Mantis8M(device=device)
        network = network.from_pretrained("paris-noah/Mantis-8M")
        network.pre_training = False
    return network.to(device).eval()


@torch.no_grad()
def _encode_mantis_signals(network, signals: np.ndarray,
                           device: str, batch_size: int = 64) -> np.ndarray:
    """Encode K univariate signals through MANTIS.

    Args:
        signals: (N, K, T) raw signals — K independent channels
    Returns:
        (N, K, 256) embeddings
    """
    N, K, T = signals.shape
    embs = np.zeros((N, K, MANTIS_DIM), dtype=np.float32)
    signals_t = torch.tensor(signals, dtype=torch.float32)
    signals_resized = F.interpolate(signals_t.reshape(N * K, 1, T),
                                    size=MANTIS_SEQ_LEN, mode="linear",
                                    align_corners=False).reshape(N, K, MANTIS_SEQ_LEN)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = signals_resized[start:end]  # (B, K, 512)
        for k in range(K):
            out = network(batch[:, k:k+1, :].to(device))  # (B, 256)
            embs[start:end, k] = out.cpu().numpy()
    return embs


@torch.no_grad()
def _extract_mantis_perchannel(
    pools: list[dict],
    device: str = "cpu",
    ckpt_path: str | None = None,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Returns {infant: (N_windows, 7, 256)}."""
    network = _load_mantis(device, ckpt_path)
    result: dict[str, np.ndarray] = {}
    for pool in tqdm(pools, desc="MANTIS"):
        chunks = pool["chunks"]  # (N, T, 7)
        signals = chunks.transpose(0, 2, 1)  # (N, 7, T)
        result[pool["infant"]] = _encode_mantis_signals(
            network, signals, device, batch_size)
    return result


# ── Public: mean-pooled embeddings (exp1/exp2) ──────────────────────────────

def extract_hubert(pools, device="cpu", **kwargs) -> dict[str, np.ndarray]:
    """Returns {infant: (N_windows, 768)}."""
    perchannel = _extract_hubert_perchannel(pools, device, **kwargs)
    return {k: v.mean(axis=1) for k, v in perchannel.items()}


def extract_mantis(pools, device="cpu", ckpt_path=None, **kwargs) -> dict[str, np.ndarray]:
    """Returns {infant: (N_windows, 256)}."""
    perchannel = _extract_mantis_perchannel(pools, device, ckpt_path, **kwargs)
    return {k: v.mean(axis=1) for k, v in perchannel.items()}


# ── Public: per-group embeddings (for tokenisation) ─────────────────────────

def _perchannel_to_grouped(perchannel: dict[str, np.ndarray],
                           pool_mode: str = "mean") -> dict[str, np.ndarray]:
    """(N, 7, D) → (N, 3, D') grouped by sensor.

    pool_mode="mean":   mean-pool within group → D' = D
    pool_mode="concat": concatenate within group, zero-pad to max group size → D' = max_ch * D
    """
    result = {}
    max_ch = max(end - start for start, end in SENSOR_GROUPS)
    for infant, emb in perchannel.items():
        N, _, D = emb.shape
        if pool_mode == "concat":
            groups = []
            for start, end in SENSOR_GROUPS:
                g = emb[:, start:end, :].reshape(N, -1)  # (N, n_ch * D)
                pad = max_ch * D - g.shape[1]
                if pad > 0:
                    g = np.pad(g, ((0, 0), (0, pad)))
                groups.append(g)
            result[infant] = np.stack(groups, axis=1)  # (N, 3, max_ch * D)
        else:
            grouped = np.stack([
                emb[:, start:end, :].mean(axis=1)
                for start, end in SENSOR_GROUPS
            ], axis=1)
            result[infant] = grouped
    return result


def extract_hubert_grouped(pools, device="cpu", pool_mode="mean",
                           **kwargs) -> dict[str, np.ndarray]:
    """Returns {infant: (N_windows, 3, D')}.
    pool_mode='concat' (default): D' = 3*768 = 2304
    pool_mode='mean':              D' = 768
    """
    return _perchannel_to_grouped(
        _extract_hubert_perchannel(pools, device, **kwargs), pool_mode)


def extract_mantis_grouped(pools, device="cpu", ckpt_path=None,
                           pool_mode="mean", **kwargs) -> dict[str, np.ndarray]:
    """Returns {infant: (N_windows, 3, D')}.
    pool_mode='mean' (default): D' = 256
    pool_mode='concat':          D' = 3*256 = 768
    """
    return _perchannel_to_grouped(
        _extract_mantis_perchannel(pools, device, ckpt_path, **kwargs), pool_mode)


# ── SVD channel combination before MANTIS ───────────────────────────────────
#
# Instead of passing each channel independently and mean-pooling embeddings,
# combine channels in signal space via SVD/PCA, then pass fused signals to MANTIS.
# Two modes:
#   "groups" — SVD within each sensor group (acc 3→1, gyr 3→1, pres 1→1) → 3 signals
#   "channels" — SVD across all 7 channels → k signals

def _fit_svd_groups(pools: list[dict]) -> list[np.ndarray]:
    """Fit PCA on raw signal within each sensor group. Returns list of 3 projection vectors."""
    from sklearn.decomposition import PCA
    all_data = np.concatenate([p["chunks"].reshape(-1, N_CHANNELS) for p in pools])
    projections = []
    for start, end in SENSOR_GROUPS:
        n_ch = end - start
        if n_ch == 1:
            projections.append(np.ones((1, 1), dtype=np.float32))
        else:
            pca = PCA(n_components=1, random_state=42)
            pca.fit(all_data[:, start:end])
            projections.append(pca.components_.astype(np.float32))  # (1, n_ch)
    return projections


def _fit_svd_channels(pools: list[dict], k: int = 3) -> np.ndarray:
    """Fit PCA across all 7 channels → k components. Returns (k, 7)."""
    from sklearn.decomposition import PCA
    all_data = np.concatenate([p["chunks"].reshape(-1, N_CHANNELS) for p in pools])
    pca = PCA(n_components=k, random_state=42)
    pca.fit(all_data)
    return pca.components_.astype(np.float32)  # (k, 7)


@torch.no_grad()
def extract_mantis_svd_groups(
    pools: list[dict], device="cpu", ckpt_path=None, batch_size=64,
) -> dict[str, np.ndarray]:
    """SVD within groups → 3 fused signals → MANTIS → (N, 3, 256)."""
    projections = _fit_svd_groups(pools)
    network = _load_mantis(device, ckpt_path)

    result: dict[str, np.ndarray] = {}
    for pool in tqdm(pools, desc="MANTIS-SVD-groups"):
        chunks = pool["chunks"]  # (N, T, 7)
        N, T, _ = chunks.shape
        fused = np.zeros((N, N_GROUPS, T), dtype=np.float32)
        for g, (start, end) in enumerate(SENSOR_GROUPS):
            # (N, T, n_ch) @ (n_ch, 1) → (N, T, 1) → (N, T)
            fused[:, g, :] = (chunks[:, :, start:end] @ projections[g].T).squeeze(-1)
        result[pool["infant"]] = _encode_mantis_signals(
            network, fused, device, batch_size)
    return result


@torch.no_grad()
def extract_mantis_svd_channels(
    pools: list[dict], device="cpu", ckpt_path=None,
    k: int = 3, batch_size=64,
) -> dict[str, np.ndarray]:
    """SVD across all 7 channels → k fused signals → MANTIS → (N, k, 256)."""
    proj = _fit_svd_channels(pools, k)  # (k, 7)
    network = _load_mantis(device, ckpt_path)

    result: dict[str, np.ndarray] = {}
    for pool in tqdm(pools, desc="MANTIS-SVD-channels"):
        chunks = pool["chunks"]  # (N, T, 7)
        # (N, T, 7) @ (7, k) → (N, T, k) → (N, k, T)
        fused = (chunks @ proj.T).transpose(0, 2, 1)  # (N, k, T)
        result[pool["infant"]] = _encode_mantis_signals(
            network, fused, device, batch_size)
    return result


# ── Tokenisation: PCA + k-means per group ───────────────────────────────────

def tokenise_grouped(
    grouped_embs: dict[str, np.ndarray],
    n_components: int = 64,
    n_clusters: int = 512,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Per-group PCA → k-means → interleaved token sequences.

    Args:
        grouped_embs: {infant: (N, 3, D)} per-group embeddings
        n_components: PCA output dim per group
        n_clusters:   codebook size per group (total vocab = 3 * n_clusters)

    Returns:
        tokens: {infant: (N * 3,)} interleaved int32 with group offsets
                [acc_0, gyr_0, pres_0, acc_1, ...]
        centroids: (3 * n_clusters, n_components) stacked codebook
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    infants = sorted(grouped_embs)
    emb_dim = grouped_embs[infants[0]].shape[2]
    n_comp = min(n_components, emb_dim)

    # Collect all windows per group for fitting
    per_group_all = [[] for _ in range(N_GROUPS)]
    for inf in infants:
        emb = grouped_embs[inf]  # (N, 3, D)
        for g in range(N_GROUPS):
            per_group_all[g].append(emb[:, g, :])
    per_group_all = [np.concatenate(x) for x in per_group_all]

    # Fit PCA + k-means per group
    pcas, kmeans_models, centroids_list = [], [], []
    for g in range(N_GROUPS):
        pca = PCA(n_components=n_comp, random_state=seed)
        X_pca = pca.fit_transform(per_group_all[g])
        pcas.append(pca)

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             n_init=3, batch_size=1024)
        km.fit(X_pca)
        kmeans_models.append(km)
        centroids_list.append(km.cluster_centers_)

    # Tokenise each infant
    tokens: dict[str, np.ndarray] = {}
    for inf in infants:
        emb = grouped_embs[inf]  # (N, 3, D)
        N = emb.shape[0]
        group_labels = []
        for g in range(N_GROUPS):
            X_pca = pcas[g].transform(emb[:, g, :])
            labels = kmeans_models[g].predict(X_pca) + g * n_clusters
            group_labels.append(labels)
        # Interleave: [acc_0, gyr_0, pres_0, acc_1, ...]
        interleaved = np.stack(group_labels, axis=1).reshape(-1).astype(np.int32)
        tokens[inf] = interleaved

    centroids = np.vstack(centroids_list)  # (3 * n_clusters, n_comp)
    return tokens, centroids


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_embeddings(embs: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **embs)


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def save_tokens(tokens: dict[str, np.ndarray], centroids: np.ndarray,
                path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, _centroids_=centroids, **tokens)


def load_tokens(path: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data = np.load(path)
    centroids = data["_centroids_"]
    tokens = {k: data[k] for k in data.files if k != "_centroids_"}
    return tokens, centroids
