"""VQ-VAE tokenised representation: TF-IDF builders and token biomarkers."""

import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, entropy as scipy_entropy
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

from src.exp_config import (
    CANDIDATES, VOCAB_SIZE, N_GROUPS, GROUP_VOCAB,
    GROUP_NAMES, GROUP_OFFSETS, TOKENS_PER_SEC,
    WIN_SEC, OUT_DIR, session_age, cohen_d,
)
from src.path import TOKENIZED_DIR


# ── VQ-VAE codebook loading ─────────────────────────────────────────────────

_CODEBOOK_CACHE: dict[str, np.ndarray] = {}


def get_codebook(cand: str) -> np.ndarray:
    if cand in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[cand]
    cfg = CANDIDATES[cand]
    ckpt = cfg.get("ckpt")
    if not ckpt or not os.path.exists(ckpt):
        # no checkpoint configured or file missing — random codebook for dry-run testing
        rng = np.random.default_rng(42)
        cb = rng.standard_normal((VOCAB_SIZE, 64)).astype(np.float32)
        _CODEBOOK_CACHE[cand] = cb
        return cb
    import torch
    from src.vqvae import get_codebook as vqvae_get_codebook
    # Load checkpoint: instantiate the right model class, load weights, extract codebook.
    # Implementation depends on the training harness used to save the checkpoint.
    raise NotImplementedError(
        f"Checkpoint loading for '{cand}' not yet implemented. "
        f"Set VQVAE_{cand.upper()}_CKPT to the .pt file path."
    )


def load_manifest(cand: str) -> tuple[pd.DataFrame, str]:
    d = os.path.join(TOKENIZED_DIR, CANDIDATES[cand]["dir"])
    mf = pd.read_csv(os.path.join(d, "manifest.csv"),
                     dtype={"infant": str, "session": str})
    return mf[mf["outcome"].isin([0, 1])].copy(), d


def load_session_tokens(row: pd.Series, d: str) -> np.ndarray:
    return np.load(os.path.join(d, row["file"]))


# ── TF-IDF feature builders ─────────────────────────────────────────────────

def build_tfidf(scale: str, cand: str):
    """Build token-histogram feature matrix from VQ-VAE tokens.

    Window:  bincount of tokens in sliding windows.
    Session: bincount of all tokens in one session.
    Infant:  pooled bincount across all sessions.
    Returns (X, y, groups).
    """
    mf, d = load_manifest(cand)
    if scale == "window":
        win_toks = (int(round(WIN_SEC * TOKENS_PER_SEC)) // N_GROUPS) * N_GROUPS
        Xs, ys, gs = [], [], []
        for _, row in mf.iterrows():
            toks = load_session_tokens(row, d).astype(int)
            for s in range(0, len(toks) - win_toks + 1, win_toks):
                Xs.append(np.bincount(toks[s:s+win_toks], minlength=VOCAB_SIZE))
                ys.append(int(row["outcome"])); gs.append(str(row["infant"]))
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "session":
        Xs, ys, gs = [], [], []
        for _, row in mf.iterrows():
            toks = load_session_tokens(row, d).astype(int)
            Xs.append(np.bincount(toks, minlength=VOCAB_SIZE).astype(np.float32))
            ys.append(int(row["outcome"])); gs.append(str(row["infant"]))
        return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)

    if scale == "infant":
        by: dict[str, dict] = {}
        for _, row in mf.iterrows():
            inf = str(row["infant"])
            toks = load_session_tokens(row, d).astype(int)
            if inf not in by:
                by[inf] = {"c": np.zeros(VOCAB_SIZE, np.float32), "label": int(row["outcome"])}
            by[inf]["c"] += np.bincount(toks, minlength=VOCAB_SIZE)
        infants = sorted(by)
        X = np.stack([by[i]["c"] for i in infants])
        y = np.array([by[i]["label"] for i in infants], np.int64)
        g = np.array(infants)
        return X, y, g

    raise ValueError(scale)


def tfidf_window_dur(cand, max_minutes):
    """Exp2: window-level token histograms limited to first max_minutes per infant."""
    mf, d = load_manifest(cand)
    win_toks = (int(round(WIN_SEC * TOKENS_PER_SEC)) // N_GROUPS) * N_GROUPS
    max_windows = int(max_minutes * 60 / WIN_SEC)
    by_infant: dict[str, list] = {}
    for _, row in mf.iterrows():
        inf = str(row["infant"])
        if inf not in by_infant:
            by_infant[inf] = []
        by_infant[inf].append(row)
    Xs, ys, gs = [], [], []
    for inf in sorted(by_infant):
        rows = by_infant[inf]
        n_win = 0
        for row in rows:
            if n_win >= max_windows:
                break
            toks = load_session_tokens(row, d).astype(int)
            for s in range(0, len(toks) - win_toks + 1, win_toks):
                if n_win >= max_windows:
                    break
                Xs.append(np.bincount(toks[s:s+win_toks], minlength=VOCAB_SIZE))
                ys.append(int(row["outcome"])); gs.append(inf)
                n_win += 1
    return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(gs)


def tfidf_infant_frac(cand, frac):
    """Exp2: pooled token counts from first frac fraction of sessions."""
    mf, d = load_manifest(cand)
    by: dict[str, dict] = {}
    for inf, sub in mf.groupby("infant"):
        sub = sub.sort_values("session")
        k = max(1, int(np.ceil(frac * len(sub))))
        c = np.zeros(VOCAB_SIZE, np.float32)
        for _, row in sub.iloc[:k].iterrows():
            c += np.bincount(load_session_tokens(row, d).astype(int), minlength=VOCAB_SIZE)
        by[str(inf)] = {"c": c, "label": int(sub["outcome"].iloc[0])}
    infs = sorted(by)
    return (np.stack([by[i]["c"] for i in infs]),
            np.array([by[i]["label"] for i in infs], np.int64))


def tfidf_infant_age(cand, age_max):
    """Exp2: pooled token counts from sessions before age_max months."""
    mf, d = load_manifest(cand)
    by: dict[str, dict] = {}
    for inf, sub in mf.groupby("infant"):
        sub = sub.copy()
        sub["age"] = [session_age(str(inf), s) for s in sub["session"]]
        sel = sub[sub["age"] <= age_max]
        if sel.empty: sel = sub.sort_values("age").iloc[:1]
        c = np.zeros(VOCAB_SIZE, np.float32)
        for _, row in sel.iterrows():
            c += np.bincount(load_session_tokens(row, d).astype(int), minlength=VOCAB_SIZE)
        by[str(inf)] = {"c": c, "label": int(sub["outcome"].iloc[0])}
    infs = sorted(by)
    return (np.stack([by[i]["c"] for i in infs]),
            np.array([by[i]["label"] for i in infs], np.int64))


# ── Token biomarkers ─────────────────────────────────────────────────────────

def group_biomarkers(s_rel, cb_g, name):
    """Per-group token biomarkers: diversity, dynamics, repetition."""
    s_rel = np.asarray(s_rel, int)
    c = np.bincount(s_rel, minlength=GROUP_VOCAB)
    p = c / (c.sum() + 1e-8); p_nz = p[p > 0]
    h = float(scipy_entropy(p_nz))
    res = {f"{name}_n_unique": int((c > 0).sum()),
           f"{name}_entropy":  h,
           f"{name}_eff_vocab": float(np.exp(h)),
           f"{name}_ttr":      int((c > 0).sum()) / max(len(s_rel), 1)}
    traj = cb_g[s_rel]
    diff = np.diff(traj, axis=0)
    spd  = np.linalg.norm(diff, axis=1)
    spd_l2 = float(spd.mean()) if len(spd) else np.nan
    norm = traj / (np.linalg.norm(traj, axis=1, keepdims=True) + 1e-8)
    cos  = (norm[:-1] * norm[1:]).sum(axis=1)
    res.update({f"{name}_speed_l2":   spd_l2,
                f"{name}_speed_cos":  float(cos.mean()) if len(cos) else np.nan,
                f"{name}_smoothness": float(spd.std() / (spd_l2 + 1e-8)) if len(spd) > 1 else np.nan})
    res[f"{name}_self_trans"] = float((s_rel[:-1] == s_rel[1:]).mean()) if len(s_rel) > 1 else np.nan
    runs, cur = [], 1
    for i in range(1, len(s_rel)):
        if s_rel[i] == s_rel[i-1]: cur += 1
        else: runs.append(cur); cur = 1
    runs.append(cur)
    res[f"{name}_mean_run"] = float(np.mean(runs))
    res[f"{name}_max_run"]  = float(np.max(runs))
    td: dict[int, Counter] = defaultdict(Counter)
    for a, b in zip(s_rel[:-1], s_rel[1:]): td[a][b] += 1
    ents = [scipy_entropy(np.array(list(r.values()), float) / sum(r.values()))
            for r in td.values()]
    res[f"{name}_trans_entropy"] = float(np.mean(ents)) if ents else np.nan
    return res


def twonn_id(X):
    """TWO-NN intrinsic dimensionality estimate."""
    if len(X) < 5: return np.nan
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / (r1 + 1e-8)
    mu_s = np.sort(mu); F = np.arange(1, len(mu_s)+1) / len(mu_s)
    mask = (mu_s > 1) & (F < 0.9)
    if mask.sum() < 5: return np.nan
    slope, _ = np.polyfit(np.log(mu_s[mask]), np.log(1 - F[mask] + 1e-8), 1)
    return float(-slope)


def infant_token_biomarkers(seq, codebook):
    """Full biomarker set for one token sequence: flat diversity + per-group + cross-sync."""
    seq = np.asarray(seq, int)
    c = np.bincount(seq, minlength=VOCAB_SIZE)
    p = c / (c.sum() + 1e-8); p_nz = p[p > 0]
    h = float(scipy_entropy(p_nz))
    res = dict(n_tokens=len(seq), n_unique=int((c > 0).sum()),
               entropy=h, eff_vocab=float(np.exp(h)),
               ttr=int((c > 0).sum()) / max(len(seq), 1))
    cbs = [codebook[g*GROUP_VOCAB:(g+1)*GROUP_VOCAB] for g in range(N_GROUPS)]
    for g, (name, cb) in enumerate(zip(GROUP_NAMES, cbs)):
        res.update(group_biomarkers(seq[g::N_GROUPS] - GROUP_OFFSETS[g], cb, name))
    n_t = len(seq) // N_GROUPS
    if n_t > 1:
        changed = np.zeros((n_t-1, N_GROUPS), bool)
        for g in range(N_GROUPS):
            s = seq[g::N_GROUPS][:n_t] - GROUP_OFFSETS[g]
            changed[:, g] = s[1:] != s[:-1]
        res["cross_sync"] = float(changed.all(axis=1).mean())
    else:
        res["cross_sync"] = np.nan
    grp_vecs = []
    for g, (name, cb) in enumerate(zip(GROUP_NAMES, cbs)):
        s_g = seq[g::N_GROUPS][:n_t] - GROUP_OFFSETS[g]
        unique_pts = np.unique(cb[s_g], axis=0)
        res[f"{name}_id_cb"] = twonn_id(unique_pts)
        grp_vecs.append(cb[s_g])
    flat_pts = np.hstack(grp_vecs)
    unique_flat = np.unique(flat_pts, axis=0)
    res["id_cb_flat"] = twonn_id(unique_flat)
    return res


# ── Exp3 runner ──────────────────────────────────────────────────────────────

def run_exp3_candidate(cand: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp3 for one VQ-VAE candidate: session-level biomarkers → median → stats."""
    mf, d = load_manifest(cand)
    cb    = get_codebook(cand)

    rows = []
    for inf, sub in mf.groupby("infant"):
        sess_bms = [infant_token_biomarkers(load_session_tokens(row, d), cb)
                    for _, row in sub.iterrows()]
        bm = {k: float(np.median([s[k] for s in sess_bms])) for k in sess_bms[0]}
        bm.update(infant=str(inf), outcome=int(sub["outcome"].iloc[0]),
                  n_sess=len(sub))
        rows.append(bm)
    df_bm = pd.DataFrame(rows)

    T_df = df_bm[df_bm.outcome == 0]
    A_df = df_bm[df_bm.outcome == 1]
    bm_cols = [c for c in df_bm.columns if c not in ("infant", "outcome", "n_sess")]
    stat_rows = []
    for m in bm_cols:
        tv = T_df[m].dropna().values; av = A_df[m].dropna().values
        if len(tv) < 3 or len(av) < 3: continue
        u, p = mannwhitneyu(tv, av, alternative="two-sided")
        cd = cohen_d(tv, av)
        grp = next((g for g in GROUP_NAMES if m.startswith(g + "_")), "flat")
        stat_rows.append(dict(candidate=cand, metric=m, group=grp,
                              T_median=np.median(tv), A_median=np.median(av),
                              T_mean=tv.mean(), A_mean=av.mean(),
                              cohen_d=cd, p_mwu=p,
                              n_T=len(tv), n_A=len(av)))

    df_stats = pd.DataFrame(stat_rows)
    _, df_stats["p_fdr"], _, _ = multipletests(df_stats["p_mwu"], method="fdr_bh")
    df_stats["sig"] = df_stats["p_fdr"].apply(
        lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "")))
    df_stats = df_stats.sort_values("p_mwu")
    df_bm["candidate"] = cand
    return df_bm, df_stats
