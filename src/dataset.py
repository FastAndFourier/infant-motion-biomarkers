import numpy as np
from collections import defaultdict
import os

import torch
from torch.utils.data import Dataset

import pandas as pd

from src.path import OUTPUT_ALIGN_DIR, OUTPUT_UNLABELED_DIR, OUTCOME_PATH
from src.utils import get_infant_info

SR = 100


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_raw(d) -> np.ndarray:
    n    = min(len(d["acc"]), len(d["gyr"]), len(d["pressure"]))
    acc  = d["acc"][:n].astype(np.float32)           # (n, 3)
    gyr  = d["gyr"][:n].astype(np.float32)           # (n, 3)
    pres = d["pressure"][:n].astype(np.float32)      # (n,) → (n, 1)
    if pres.ndim == 1:
        pres = pres[:, None]
    return np.concatenate([acc, gyr, pres], axis=1)  # (n, 7)


def _load_outcome_map() -> dict[str, int]:
    """Return {infant_id: 0/1} from OUTCOME_PATH. Missing infants map to -1."""
    try:
        df = pd.read_csv(OUTCOME_PATH)
        df["ID"] = df["ID"].astype(str)
        return {row["ID"]: int(row["outcome: A,T"].strip() == "A")
                for _, row in df.iterrows()}
    except Exception:
        return {}


# ── Outcome dataset for MIL prediction ───────────────────────────────────────

class OutcomeDataset(Dataset):
    """
    One item = one infant.

    Returns (chunks, ages, mask, label) where:
      chunks : (bag_size, T, 7)  float32 — sampled windows from the infant's sessions
      ages   : (bag_size,)       float32 — normalised corrected age per chunk
      mask   : (bag_size,)       bool    — False for real chunks, True for padding
      label  : scalar int64              — 0=Typical, 1=Atypical

    During training (deterministic=False): `bag_size` chunks are sampled randomly
    from the infant's full pool (with replacement when pool < bag_size).
    During val/test (deterministic=True): all chunks are returned, padded to the
    length of the largest pool in the dataset.

    All sessions for each infant are used — both annotated (OUTPUT_ALIGN_DIR) and
    unannotated (OUTPUT_UNLABELED_DIR). Sessions appearing in both directories are
    deduplicated (aligned copy preferred). Only infants present in OUTCOME_PATH are
    included.
    """

    def __init__(
        self,
        win:           float,
        hop:           float,
        bag_size:      int   = 16,
        deterministic: bool  = False,
        seed:          int   = 42,
        norm:          str   = "infant",
    ):
        super().__init__()
        assert norm in ("infant", "session", "none"), f"Unknown norm: {norm!r}"
        self.win          = int(win * SR)
        self.hop          = int(hop * SR)
        self.bag_size     = bag_size
        self.deterministic = deterministic
        self._rng         = np.random.default_rng(seed)
        self.norm         = norm

        df = pd.read_csv(OUTCOME_PATH)
        df["ID"] = df["ID"].astype(str)
        df["_label"] = (df["outcome: A,T"].str.strip() == "A").astype(int)
        outcome_map = dict(zip(df["ID"], df["_label"]))

        seen: dict[tuple[str, str], str] = {}   # (infant, session) → file path
        for data_dir in (OUTPUT_ALIGN_DIR, OUTPUT_UNLABELED_DIR):
            for fname in sorted(os.listdir(data_dir)):
                if fname.startswith(".") or not fname.endswith(".npz"):
                    continue
                infant, _, session = fname.split("_", 2)
                session = session.removesuffix(".npz")
                if infant not in outcome_map:
                    continue
                key = (infant, session)
                if key not in seen:
                    seen[key] = os.path.join(data_dir, fname)

        raw_by_infant: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for (infant, session), fpath in sorted(seen.items()):
            d   = np.load(fpath)
            raw = _load_raw(d)
            raw_by_infant[infant].append((session, raw))

        all_ages_raw: list[float] = []
        infant_pools: list[dict]  = []

        for infant, sessions in raw_by_infant.items():
            if norm == "infant":
                all_raw = np.concatenate([r for _, r in sessions], axis=0)
                mu    = all_raw.mean(0).astype(np.float32)
                sigma = (all_raw.std(0) + 1e-8).astype(np.float32)

            chunks, chunk_ages, chunk_sessions = [], [], []
            for session, raw in sessions:
                if norm == "infant":
                    data = (raw - mu) / sigma
                elif norm == "session":
                    mu_s    = raw.mean(0, keepdims=True).astype(np.float32)
                    sigma_s = (raw.std(0, keepdims=True) + 1e-8).astype(np.float32)
                    data    = (raw - mu_s) / sigma_s
                else:
                    data = raw
                L = 1 + int(np.floor((len(data) - self.win) / self.hop))
                for i in range(0, L * self.hop, self.hop):
                    chunks.append(data[i: i + self.win])
                try:
                    _, age_val, _ = get_infant_info(infant, session)
                    chunk_ages.extend([age_val] * L)
                except Exception:
                    chunk_ages.extend([0.0] * L)
                chunk_sessions.extend([session] * L)

            if not chunks:
                continue

            all_ages_raw.extend(chunk_ages)
            infant_pools.append({
                "infant":   infant,
                "label":    outcome_map[infant],
                "chunks":   np.stack(chunks).astype(np.float32),  # (N, T, 7)
                "ages":     np.array(chunk_ages, dtype=np.float32),
                "sessions": np.array(chunk_sessions),
            })

        age_mean = float(np.mean(all_ages_raw))
        age_std  = float(np.std(all_ages_raw) + 1e-8)
        for pool in infant_pools:
            pool["ages"] = (pool["ages"] - age_mean) / age_std

        self._pools   = infant_pools
        self._max_N   = max(len(p["chunks"]) for p in infant_pools)
        self.infants  = [p["infant"] for p in infant_pools]
        self.labels   = np.array([p["label"] for p in infant_pools], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._pools)

    def __getitem__(self, idx: int):
        pool   = self._pools[idx]
        chunks = pool["chunks"]   # (N, T, 7)
        ages   = pool["ages"]     # (N,)
        label  = pool["label"]

        N = len(chunks)

        if self.deterministic:
            pad   = self._max_N - N
            if pad > 0:
                chunks = np.concatenate([chunks, np.zeros((pad, *chunks.shape[1:]), dtype=np.float32)])
                ages   = np.concatenate([ages,   np.zeros(pad, dtype=np.float32)])
            mask = np.array([False] * N + [True] * pad, dtype=bool)
        else:
            idx_s  = self._rng.choice(N, size=self.bag_size, replace=(N < self.bag_size))
            chunks = chunks[idx_s]
            ages   = ages[idx_s]
            mask   = np.zeros(self.bag_size, dtype=bool)

        t_label = torch.tensor(label, dtype=torch.long)
        info = {
            "infant":  pool["infant"],
            "session": "",
            "outcome": label,
            "mask":    torch.tensor(mask, dtype=torch.bool),
        }
        return (
            torch.tensor(chunks, dtype=torch.float32),
            info,
            torch.tensor(ages,   dtype=torch.float32),
            t_label,
        )
