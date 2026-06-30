"""Signal statistics representation (140-dim per window/session/infant)."""

import numpy as np

from src.signal_outcome import feature_extraction
from src.exp_config import WIN_SEC, session_age


def build_signal(scale: str, pools: list[dict]):
    """Build feature matrix from raw IMU signal.

    Window:  feature_extraction on each 30s chunk.
    Session: concatenate chunks within session, feature_extraction on full signal.
    Infant:  concatenate all chunks, feature_extraction on full signal.
    Returns (X, y, groups).
    """
    if scale == "window":
        Xs, ys, gs = [], [], []
        for p in pools:
            F = feature_extraction(p["chunks"])
            Xs.append(F)
            ys.extend([p["label"]] * len(F))
            gs.extend([p["infant"]] * len(F))
        return np.vstack(Xs).astype(np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "session":
        Xs, ys, gs = [], [], []
        for p in pools:
            for sess in np.unique(p["sessions"]):
                mask = p["sessions"] == sess
                concat = np.concatenate(p["chunks"][mask], axis=0)[np.newaxis]
                Xs.append(feature_extraction(concat)[0])
                ys.append(p["label"]); gs.append(p["infant"])
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "infant":
        Xs, ys, gs = [], [], []
        for p in pools:
            concat = np.concatenate(p["chunks"], axis=0)[np.newaxis]
            Xs.append(feature_extraction(concat)[0])
            ys.append(p["label"]); gs.append(p["infant"])
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    raise ValueError(scale)


def signal_window_dur(pools, max_minutes):
    """Exp2: window-level features limited to first max_minutes per infant."""
    max_chunks = int(max_minutes * 60 / WIN_SEC)
    Xs, ys, gs = [], [], []
    for p in pools:
        n = min(len(p["chunks"]), max_chunks)
        F = feature_extraction(p["chunks"][:n])
        Xs.append(F)
        ys.extend([p["label"]] * len(F))
        gs.extend([p["infant"]] * len(F))
    return np.vstack(Xs).astype(np.float32), np.array(ys, np.int64), np.array(gs)


def signal_infant_frac(pools, frac):
    """Exp2: features from first frac fraction of sessions per infant."""
    Xs, ys = [], []
    for p in pools:
        sessions = sorted(np.unique(p["sessions"]))
        k = max(1, int(np.ceil(frac * len(sessions))))
        sel = set(sessions[:k])
        mask = np.array([s in sel for s in p["sessions"]])
        concat = np.concatenate(p["chunks"][mask], axis=0)[np.newaxis]
        Xs.append(feature_extraction(concat)[0])
        ys.append(p["label"])
    return np.array(Xs, np.float32), np.array(ys, np.int64)


def signal_infant_age(pools, age_max):
    """Exp2: features from sessions recorded before age_max months."""
    Xs, ys = [], []
    for p in pools:
        sessions = sorted(np.unique(p["sessions"]))
        ages = [session_age(p["infant"], s) for s in sessions]
        sel = set(s for s, a in zip(sessions, ages) if a <= age_max) or {sessions[0]}
        mask = np.array([s in sel for s in p["sessions"]])
        concat = np.concatenate(p["chunks"][mask], axis=0)[np.newaxis]
        Xs.append(feature_extraction(concat)[0])
        ys.append(p["label"])
    return np.array(Xs, np.float32), np.array(ys, np.int64)
