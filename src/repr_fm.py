"""Foundation model representations: continuous embeddings + action tokens."""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, entropy as scipy_entropy
from statsmodels.stats.multitest import multipletests

from src.exp_config import (
    ENCODERS, N_GROUPS, GROUP_NAMES, GROUP_VOCAB, GROUP_OFFSETS,
    VOCAB_SIZE, OUT_DIR, session_age, cohen_d,
)
from src.embedding_extractor import (
    extract_hubert, extract_mantis, save_embeddings, load_embeddings,
    extract_hubert_grouped, extract_mantis_grouped,
    extract_mantis_svd_groups, extract_mantis_svd_channels,
    tokenise_grouped, save_tokens, load_tokens,
)
from src.repr_symbolic import infant_token_biomarkers, twonn_id


# ── Device ───────────────────────────────────────────────────────────────────

def default_device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ── Cache paths ──────────────────────────────────────────────────────────────

def emb_cache_path(encoder: str) -> str:
    return os.path.join(OUT_DIR, f"{encoder}_embeddings.npz")


def token_cache_path(encoder: str) -> str:
    return os.path.join(OUT_DIR, f"{encoder}_tokens.npz")


# ── Precompute (GPU) ─────────────────────────────────────────────────────────

def precompute_embeddings(encoders: list[str],
                          mantis_ckpt: str | None = None,
                          device: str | None = None,
                          pools: list[dict] | None = None):
    if device is None:
        device = default_device()
    print(f"Precomputing embeddings on {device}")
    if pools is None:
        from src.dataset import OutcomeDataset
        from src.exp_config import WIN_SEC, HOP_SEC
        pools = OutcomeDataset(win=WIN_SEC, hop=HOP_SEC, deterministic=True)._pools
    for enc in encoders:
        path = emb_cache_path(enc)
        if os.path.exists(path):
            print(f"  {enc}: already cached at {path}, skipping")
            continue
        print(f"  {enc}: extracting …")
        if enc == "hubert":
            embs = extract_hubert(pools, device=device)
        elif enc == "mantis":
            embs = extract_mantis(pools, device=device, ckpt_path=mantis_ckpt)
        else:
            raise ValueError(f"Unknown encoder: {enc}")
        save_embeddings(embs, path)
        _EMB_CACHE[enc] = embs
        print(f"  {enc}: saved → {path}  ({len(embs)} infants)")


def precompute_tokens(encoders: list[str],
                      mantis_ckpt: str | None = None,
                      device: str | None = None,
                      n_components: int = 64,
                      n_clusters: int = 512,
                      pools: list[dict] | None = None):
    if device is None:
        device = default_device()
    print(f"Precomputing tokens on {device}")
    if pools is None:
        from src.dataset import OutcomeDataset
        from src.exp_config import WIN_SEC, HOP_SEC
        pools = OutcomeDataset(win=WIN_SEC, hop=HOP_SEC, deterministic=True)._pools
    for enc in encoders:
        tok_path = token_cache_path(enc)
        if os.path.exists(tok_path):
            print(f"  {enc}: tokens already cached at {tok_path}, skipping")
            continue
        grp_path = os.path.join(OUT_DIR, f"{enc}_grouped_embeddings.npz")
        if os.path.exists(grp_path):
            print(f"  {enc}: loading grouped embeddings from {grp_path}")
            grouped = load_embeddings(grp_path)
        else:
            print(f"  {enc}: extracting grouped embeddings …")
            if enc == "hubert":
                grouped = extract_hubert_grouped(pools, device=device)
            elif enc == "mantis":
                grouped = extract_mantis_grouped(pools, device=device, ckpt_path=mantis_ckpt)
            elif enc == "mantis_svdg":
                grouped = extract_mantis_svd_groups(pools, device=device, ckpt_path=mantis_ckpt)
            elif enc == "mantis_svdc":
                grouped = extract_mantis_svd_channels(pools, device=device,
                                                       ckpt_path=mantis_ckpt, k=N_GROUPS)
            else:
                raise ValueError(f"Unknown encoder: {enc}")
            save_embeddings(grouped, grp_path)
            print(f"  {enc}: grouped embeddings → {grp_path}")
        print(f"  {enc}: tokenising (PCA={n_components}, k={n_clusters}) …")
        tokens, centroids = tokenise_grouped(grouped, n_components, n_clusters)
        save_tokens(tokens, centroids, tok_path)
        print(f"  {enc}: tokens → {tok_path}  ({len(tokens)} infants, "
              f"vocab={3 * n_clusters})")


# ── In-memory caches ─────────────────────────────────────────────────────────

_EMB_CACHE: dict[str, dict[str, np.ndarray]] = {}
_TOKEN_CACHE: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}


def get_tokens(encoder: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if encoder in _TOKEN_CACHE:
        return _TOKEN_CACHE[encoder]
    path = token_cache_path(encoder)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokens for '{encoder}' not found at {path}.\n"
            f"Run:  python new_experiments.py --precompute-tokens {encoder}")
    tokens, centroids = load_tokens(path)
    _TOKEN_CACHE[encoder] = (tokens, centroids)
    return tokens, centroids


def get_embeddings(encoder: str, pools: list[dict]) -> dict[str, np.ndarray]:
    if encoder in _EMB_CACHE:
        return _EMB_CACHE[encoder]
    path = emb_cache_path(encoder)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embeddings for '{encoder}' not found at {path}.\n"
            f"Run:  python new_experiments.py --precompute {encoder}")
    print(f"  Loading {encoder} embeddings from {path}")
    embs = load_embeddings(path)
    _EMB_CACHE[encoder] = embs
    return embs


# ── Feature builders (exp1) ─────────────────────────────────────────────────

def build_embedding(scale: str, encoder: str, pools: list[dict]):
    """Build continuous embedding feature matrix. Returns (X, y, groups)."""
    embs = get_embeddings(encoder, pools)
    if scale == "window":
        Xs, ys, gs = [], [], []
        for p in pools:
            e = embs[p["infant"]]
            Xs.append(e)
            ys.extend([p["label"]] * len(e))
            gs.extend([p["infant"]] * len(e))
        return np.vstack(Xs), np.array(ys, np.int64), np.array(gs)

    if scale == "session":
        Xs, ys, gs = [], [], []
        for p in pools:
            e = embs[p["infant"]]
            for sess in np.unique(p["sessions"]):
                mask = p["sessions"] == sess
                Xs.append(e[mask].mean(0))
                ys.append(p["label"]); gs.append(p["infant"])
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "infant":
        Xs, ys, gs = [], [], []
        for p in pools:
            e = embs[p["infant"]]
            Xs.append(e.mean(0))
            ys.append(p["label"]); gs.append(p["infant"])
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    raise ValueError(scale)


def build_tfidf_fm(scale: str, encoder: str, pools: list[dict]):
    """Build token histogram matrix from FM tokens. Returns (X, y, groups)."""
    tokens_dict, centroids = get_tokens(encoder)
    vocab = len(centroids)

    if scale == "window":
        from src.exp_config import WIN_SEC
        fm_tok_rate = N_GROUPS / 5.0  # 3 tokens per 5s = 0.6 tok/s
        win_toks = int(round(WIN_SEC * fm_tok_rate))
        win_toks = (win_toks // N_GROUPS) * N_GROUPS
        Xs, ys, gs = [], [], []
        for p in pools:
            inf = p["infant"]
            if inf not in tokens_dict:
                continue
            seq = tokens_dict[inf].astype(int)
            for s in range(0, len(seq) - win_toks + 1, win_toks):
                Xs.append(np.bincount(seq[s:s+win_toks], minlength=vocab))
                ys.append(p["label"]); gs.append(inf)
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "session":
        Xs, ys, gs = [], [], []
        for p in pools:
            inf = p["infant"]
            if inf not in tokens_dict:
                continue
            seq = tokens_dict[inf].astype(int)
            unique_sess = np.unique(p["sessions"])
            toks_per_win = N_GROUPS
            offset = 0
            for sess in unique_sess:
                nw = int((p["sessions"] == sess).sum())
                chunk = seq[offset:offset + nw * toks_per_win]
                offset += nw * toks_per_win
                if len(chunk) > 0:
                    Xs.append(np.bincount(chunk, minlength=vocab).astype(np.float32))
                    ys.append(p["label"]); gs.append(inf)
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "infant":
        Xs, ys, gs = [], [], []
        for p in pools:
            inf = p["infant"]
            if inf not in tokens_dict:
                continue
            seq = tokens_dict[inf].astype(int)
            Xs.append(np.bincount(seq, minlength=vocab).astype(np.float32))
            ys.append(p["label"]); gs.append(inf)
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    raise ValueError(scale)


# ── Exp2 helpers ─────────────────────────────────────────────────────────────

def emb_window_dur(encoder, pools, max_minutes):
    """Exp2: window-level embeddings limited to first max_minutes per infant."""
    from src.exp_config import WIN_SEC
    embs = get_embeddings(encoder, pools)
    max_chunks = int(max_minutes * 60 / WIN_SEC)
    Xs, ys, gs = [], [], []
    for p in pools:
        e = embs[p["infant"]]
        n = min(len(e), max_chunks)
        Xs.append(e[:n])
        ys.extend([p["label"]] * n)
        gs.extend([p["infant"]] * n)
    return np.vstack(Xs), np.array(ys, np.int64), np.array(gs)


def fm_tok_window_dur(encoder, pools, max_minutes):
    """Exp2: window-level FM token histograms limited to first max_minutes per infant."""
    from src.exp_config import WIN_SEC
    tokens_dict, centroids = get_tokens(encoder)
    vocab = len(centroids)
    fm_tok_rate = N_GROUPS / 5.0
    win_toks = int(round(WIN_SEC * fm_tok_rate))
    win_toks = (win_toks // N_GROUPS) * N_GROUPS
    max_windows = int(max_minutes * 60 / WIN_SEC)
    Xs, ys, gs = [], [], []
    for p in pools:
        inf = p["infant"]
        if inf not in tokens_dict:
            continue
        seq = tokens_dict[inf].astype(int)
        n_win = 0
        for s in range(0, len(seq) - win_toks + 1, win_toks):
            if n_win >= max_windows:
                break
            Xs.append(np.bincount(seq[s:s+win_toks], minlength=vocab))
            ys.append(p["label"]); gs.append(inf)
            n_win += 1
    return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)


def emb_infant_frac(encoder, pools, frac):
    embs = get_embeddings(encoder, pools)
    Xs, ys = [], []
    for p in pools:
        e = embs[p["infant"]]
        sessions = sorted(np.unique(p["sessions"]))
        k = max(1, int(np.ceil(frac * len(sessions))))
        sel = set(sessions[:k])
        mask = np.array([s in sel for s in p["sessions"]])
        Xs.append(e[mask].mean(0)); ys.append(p["label"])
    return np.array(Xs, np.float32), np.array(ys, np.int64)


def emb_infant_age(encoder, pools, age_max):
    embs = get_embeddings(encoder, pools)
    Xs, ys = [], []
    for p in pools:
        e = embs[p["infant"]]
        sessions = sorted(np.unique(p["sessions"]))
        ages = [session_age(p["infant"], s) for s in sessions]
        sel = set(s for s, a in zip(sessions, ages) if a <= age_max) or {sessions[0]}
        mask = np.array([s in sel for s in p["sessions"]])
        Xs.append(e[mask].mean(0)); ys.append(p["label"])
    return np.array(Xs, np.float32), np.array(ys, np.int64)


def fm_tok_infant_frac(encoder, pools, frac):
    tokens_dict, centroids = get_tokens(encoder)
    vocab = len(centroids)
    Xs, ys = [], []
    for p in pools:
        inf = p["infant"]
        if inf not in tokens_dict:
            continue
        seq = tokens_dict[inf].astype(int)
        sessions = sorted(np.unique(p["sessions"]))
        k = max(1, int(np.ceil(frac * len(sessions))))
        sel = set(sessions[:k])
        mask = np.array([s in sel for s in p["sessions"]])
        n_sel = int(mask.sum())
        chunk = seq[:n_sel * N_GROUPS]
        Xs.append(np.bincount(chunk, minlength=vocab).astype(np.float32))
        ys.append(p["label"])
    return np.array(Xs, np.float32), np.array(ys, np.int64)


def fm_tok_infant_age(encoder, pools, age_max):
    tokens_dict, centroids = get_tokens(encoder)
    vocab = len(centroids)
    Xs, ys = [], []
    for p in pools:
        inf = p["infant"]
        if inf not in tokens_dict:
            continue
        seq = tokens_dict[inf].astype(int)
        sessions = sorted(np.unique(p["sessions"]))
        ages = [session_age(inf, s) for s in sessions]
        sel = set(s for s, a in zip(sessions, ages) if a <= age_max) or {sessions[0]}
        mask = np.array([s in sel for s in p["sessions"]])
        n_sel = int(mask.sum())
        chunk = seq[:n_sel * N_GROUPS]
        Xs.append(np.bincount(chunk, minlength=vocab).astype(np.float32))
        ys.append(p["label"])
    return np.array(Xs, np.float32), np.array(ys, np.int64)


# ── Embedding biomarkers (exp3) ──────────────────────────────────────────────

def infant_emb_biomarkers(emb: np.ndarray) -> dict:
    """Continuous embedding biomarkers for one infant's window embeddings."""
    N, D = emb.shape
    res = dict(emb_n_windows=N)
    res["emb_mean_norm"] = float(np.linalg.norm(emb.mean(0)))
    res["emb_std_mean"] = float(emb.std(0).mean())
    norms = np.linalg.norm(emb, axis=1)
    res["emb_cv_norm"] = float(norms.std() / (norms.mean() + 1e-8))
    if N > 1:
        diffs = np.diff(emb, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)
        res["emb_speed_l2"] = float(speeds.mean())
        res["emb_smoothness"] = float(speeds.std() / (speeds.mean() + 1e-8)) if speeds.mean() > 0 else np.nan
        norm_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        cos_sim = (norm_emb[:-1] * norm_emb[1:]).sum(axis=1)
        res["emb_cos_sim_mean"] = float(cos_sim.mean())
        res["emb_cos_sim_std"] = float(cos_sim.std())
    else:
        res.update(emb_speed_l2=np.nan, emb_smoothness=np.nan,
                   emb_cos_sim_mean=np.nan, emb_cos_sim_std=np.nan)
    res["emb_id"] = twonn_id(emb) if N >= 5 else np.nan
    if N >= 8:
        from sklearn.cluster import MiniBatchKMeans
        k = min(32, N // 2)
        labels = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(emb)
        c = np.bincount(labels, minlength=k)
        p = c / c.sum(); p_nz = p[p > 0]
        res["emb_disc_entropy"] = float(scipy_entropy(p_nz))
        res["emb_disc_eff_vocab"] = float(np.exp(scipy_entropy(p_nz)))
    else:
        res["emb_disc_entropy"] = np.nan
        res["emb_disc_eff_vocab"] = np.nan
    return res


# ── Exp3 runners ─────────────────────────────────────────────────────────────

def run_exp3_encoder(encoder: str, pools: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp3 for continuous FM embeddings: session biomarkers → median → stats."""
    embs = get_embeddings(encoder, pools)
    rows = []
    for p in pools:
        e = embs[p["infant"]]
        sess_bms = []
        for sess in np.unique(p["sessions"]):
            mask = p["sessions"] == sess
            if mask.sum() == 0:
                continue
            sess_bms.append(infant_emb_biomarkers(e[mask]))
        bm = {k: float(np.median([s[k] for s in sess_bms])) for k in sess_bms[0]}
        bm.update(infant=p["infant"], outcome=p["label"],
                  n_sess=len(np.unique(p["sessions"])))
        rows.append(bm)
    df_bm = pd.DataFrame(rows)

    T_df = df_bm[df_bm.outcome == 0]
    A_df = df_bm[df_bm.outcome == 1]
    bm_cols = [c for c in df_bm.columns if c not in ("infant", "outcome", "n_sess")]
    stat_rows = []
    for m in bm_cols:
        tv = T_df[m].dropna().values; av = A_df[m].dropna().values
        if len(tv) < 3 or len(av) < 3: continue
        u, p_val = mannwhitneyu(tv, av, alternative="two-sided")
        cd = cohen_d(tv, av)
        stat_rows.append(dict(candidate=encoder, metric=m, group="embedding",
                              T_median=np.median(tv), A_median=np.median(av),
                              T_mean=tv.mean(), A_mean=av.mean(),
                              cohen_d=cd, p_mwu=p_val,
                              n_T=len(tv), n_A=len(av)))

    df_stats = pd.DataFrame(stat_rows)
    if len(df_stats):
        _, df_stats["p_fdr"], _, _ = multipletests(df_stats["p_mwu"], method="fdr_bh")
        df_stats["sig"] = df_stats["p_fdr"].apply(
            lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "")))
        df_stats = df_stats.sort_values("p_mwu")
    df_bm["candidate"] = encoder
    return df_bm, df_stats


def run_exp3_encoder_tokens(encoder: str, pools: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp3 for FM tokens: session-level token biomarkers → median → stats."""
    tokens_dict, centroids = get_tokens(encoder)
    n_clusters = len(centroids) // N_GROUPS

    rows = []
    for p in pools:
        inf = p["infant"]
        if inf not in tokens_dict:
            continue
        seq = tokens_dict[inf]
        unique_sess = np.unique(p["sessions"])
        n_win_per_sess = [int((p["sessions"] == s).sum()) for s in unique_sess]
        toks_per_win = len(seq) // sum(n_win_per_sess) if sum(n_win_per_sess) > 0 else len(seq)
        sess_bms = []
        offset = 0
        for nw in n_win_per_sess:
            n_tok = nw * toks_per_win
            s = seq[offset:offset + n_tok]
            offset += n_tok
            if len(s) > 0:
                sess_bms.append(infant_token_biomarkers(s, centroids))
        if not sess_bms:
            continue
        bm = {k: float(np.median([s[k] for s in sess_bms])) for k in sess_bms[0]}
        bm.update(infant=inf, outcome=p["label"],
                  n_sess=len(unique_sess))
        rows.append(bm)
    df_bm = pd.DataFrame(rows)

    T_df = df_bm[df_bm.outcome == 0]
    A_df = df_bm[df_bm.outcome == 1]
    bm_cols = [c for c in df_bm.columns if c not in ("infant", "outcome", "n_sess")]
    stat_rows = []
    for m in bm_cols:
        tv = T_df[m].dropna().values; av = A_df[m].dropna().values
        if len(tv) < 3 or len(av) < 3: continue
        u, p_val = mannwhitneyu(tv, av, alternative="two-sided")
        cd = cohen_d(tv, av)
        grp = next((g for g in GROUP_NAMES if m.startswith(g + "_")), "flat")
        stat_rows.append(dict(candidate=f"{encoder}_tok", metric=m, group=grp,
                              T_median=np.median(tv), A_median=np.median(av),
                              T_mean=tv.mean(), A_mean=av.mean(),
                              cohen_d=cd, p_mwu=p_val,
                              n_T=len(tv), n_A=len(av)))

    df_stats = pd.DataFrame(stat_rows)
    if len(df_stats):
        _, df_stats["p_fdr"], _, _ = multipletests(df_stats["p_mwu"], method="fdr_bh")
        df_stats["sig"] = df_stats["p_fdr"].apply(
            lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "")))
        df_stats = df_stats.sort_values("p_mwu")
    df_bm["candidate"] = f"{encoder}_tok"
    return df_bm, df_stats
