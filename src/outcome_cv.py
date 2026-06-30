"""
Outcome prediction library: feature extraction, per-model fit functions, nested GroupKFold CV.

  extract_signal / extract_tfidf  ->  {win_sec: {"X", "y", "groups"}}
  fit_logreg / fit_rf / fit_mlp / fit_xgb  (preprocessed X in, probabilities out)
  run_nested_cv   nested infant-level CV with HPO over (window_size, model params)
  run_session_cv  session-level GroupKFold, soft-vote infant aggregation (no HPO)

See run_experiment.py for the CLI / result-saving driver.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.signal_outcome import feature_extraction
from src.dataset import OutcomeDataset
from src.path import TOKENIZED_DIR
from src.tfidf_outcome import LognormSmoothTfidf

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_GRID = (10.0, 30.0, 60.0)
CACHE_DIR = "results/cache"
THRESHOLD = 0.5

MAIN_CANDIDATE = "grouped"
ABLATION_CANDIDATES = ("cm", "cross_group")
ALL_CANDIDATES = (MAIN_CANDIDATE,) + ABLATION_CANDIDATES

STOCHASTIC_MODELS = {"rf", "mlp", "xgb"}

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_signal(window_grid: tuple = WINDOW_GRID,
                   cache_dir: str = CACHE_DIR) -> dict[float, dict]:
    """Signal features (140-dim hand-crafted, per window).
    Returns {win_sec: {"X": (n,140) float32, "y": (n,) int64, "groups": (n,) str}}."""
    os.makedirs(cache_dir, exist_ok=True)
    out: dict[float, dict] = {}
    for win in window_grid:
        path = os.path.join(cache_dir, f"signal_win{win:g}.npz")
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            out[win] = dict(X=d["X"], y=d["y"], groups=d["groups"])
        else:
            ds = OutcomeDataset(win=win, hop=win / 2, deterministic=True)
            feats, labels, groups = [], [], []
            for pool in ds._pools:
                F = feature_extraction(pool["chunks"])
                feats.append(F)
                labels.append(np.full(len(F), pool["label"], dtype=np.int64))
                groups.append(np.full(len(F), pool["infant"]))
            data = dict(
                X=np.concatenate(feats).astype(np.float32),
                y=np.concatenate(labels),
                groups=np.concatenate(groups),
            )
            np.savez(path, **data)
            out[win] = data
    return out


def extract_tfidf(candidate: str, window_grid: tuple = WINDOW_GRID,
                   cache_dir: str = CACHE_DIR) -> dict[float, dict]:
    """Bag-of-codes features over VQ-VAE tokens (per window).
    Returns {win_sec: {"X": (n, vocab) float32, "y": (n,) int64, "groups": (n,) str}}."""
    os.makedirs(cache_dir, exist_ok=True)
    dir_path = os.path.join(TOKENIZED_DIR, candidate)
    manifest = pd.read_csv(os.path.join(dir_path, "manifest.csv"),
                            dtype={"infant": str, "session": str})
    with open(os.path.join(dir_path, "config.json")) as f:
        cfg = json.load(f)
    n_codes = cfg["vocab_size"]
    toks_per_sec = cfg["tokens_per_sec_per_group"] * cfg["n_groups"]
    labeled = manifest[manifest["outcome"].isin([0, 1])]

    out: dict[float, dict] = {}
    for win in window_grid:
        path = os.path.join(cache_dir, f"symbolic_{candidate}_win{win:g}.npz")
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            out[win] = dict(X=d["X"], y=d["y"], groups=d["groups"])
        else:
            win_toks = max(1, int(round(win * toks_per_sec)))
            feats, labels, groups = [], [], []
            for _, row in labeled.iterrows():
                tokens = np.load(os.path.join(dir_path, row["file"]))
                for start in range(0, len(tokens) - win_toks + 1, win_toks):
                    feats.append(np.bincount(tokens[start:start + win_toks], minlength=n_codes))
                    labels.append(int(row["outcome"]))
                    groups.append(row["infant"])
            data = dict(
                X=np.array(feats, dtype=np.float32),
                y=np.array(labels, dtype=np.int64),
                groups=np.array(groups),
            )
            np.savez(path, **data)
            out[win] = data
    return out


# ── Preprocessing (fold-safe) ─────────────────────────────────────────────────

def _preprocess(X_tr: np.ndarray, X_te: np.ndarray,
                kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit on X_tr, apply to both. kind: 'scale' (signal) | 'tfidf' (symbolic)."""
    pre = StandardScaler() if kind == "scale" else LognormSmoothTfidf()
    Xtr = pre.fit_transform(X_tr)
    Xte = pre.transform(X_te)
    if hasattr(Xtr, "toarray"):
        Xtr, Xte = Xtr.toarray(), Xte.toarray()
    return Xtr, Xte


# ── Model fit functions ───────────────────────────────────────────────────────
# Each accepts preprocessed (X_tr, y_tr, X_te) and returns predicted probabilities.

def fit_logreg(X_tr, y_tr, X_te, *, l1_ratio=0.0, C=1.0, seed=0) -> np.ndarray:
    # l1_ratio=0 -> L2, l1_ratio=1 -> L1 (saga supports both)
    clf = LogisticRegression(l1_ratio=l1_ratio, C=C, solver="saga",
                              class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def fit_rf(X_tr, y_tr, X_te, *, n_estimators=200, max_depth=None,
            min_samples_leaf=1, seed=0) -> np.ndarray:
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  class_weight="balanced", random_state=seed, n_jobs=4)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def fit_mlp(X_tr, y_tr, X_te, *, hidden_layer_sizes=(64,), alpha=1e-4,
             seed=0) -> np.ndarray:
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation="relu",
                         alpha=alpha, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, n_iter_no_change=10, random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def fit_xgb(X_tr, y_tr, X_te, *, n_estimators=200, max_depth=5,
             learning_rate=0.1, subsample=0.85, seed=0) -> np.ndarray:
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, subsample=subsample,
                         scale_pos_weight=n_neg / max(n_pos, 1),
                         eval_metric="logloss", random_state=seed, n_jobs=4)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


FIT_FNS: dict[str, callable] = {
    "logreg": fit_logreg,
    "rf": fit_rf,
    "mlp": fit_mlp,
    "xgb": fit_xgb,
}

# ── Aggregation ───────────────────────────────────────────────────────────────

def _group_scores(window_scores: np.ndarray, window_groups: np.ndarray,
                   infants: np.ndarray, reduce) -> np.ndarray:
    out = np.empty(len(infants), dtype=np.float64)
    for i, inf in enumerate(infants):
        out[i] = reduce(window_scores[window_groups == inf])
    return out


def aggregate_mean_prob(window_scores: np.ndarray, window_groups: np.ndarray,
                         infants: np.ndarray,
                         threshold: float = THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Mean window probability -> (infant_score, infant_pred)."""
    score = _group_scores(window_scores, window_groups, infants, np.mean)
    return score, (score >= threshold).astype(int)


def aggregate_majority_vote(window_scores: np.ndarray, window_groups: np.ndarray,
                              infants: np.ndarray,
                              threshold: float = THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of windows voting positive -> (infant_score, infant_pred)."""
    votes = (window_scores >= threshold).astype(np.float64)
    score = _group_scores(votes, window_groups, infants, np.mean)
    return score, (score >= 0.5).astype(int)


AGGREGATORS: dict[str, callable] = {
    "mean_prob": aggregate_mean_prob,
    "majority_vote": aggregate_majority_vote,
}

# ── Metrics ───────────────────────────────────────────────────────────────────

def infant_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    """AUC + clinical metrics at the fixed decision threshold."""
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return dict(auc=auc, sensitivity=sens, specificity=spec,
                 ppv=ppv, npv=npv, bacc=balanced_accuracy_score(y_true, y_pred))

# ── HPO helpers ───────────────────────────────────────────────────────────────

def _sample_params(model: str, rng: np.random.Generator) -> dict:
    def pick(opts):
        return opts[rng.integers(len(opts))]
    if model == "logreg":
        return dict(l1_ratio=pick([0.0, 1.0]),
                    C=float(10 ** rng.uniform(-3, 2)))
    if model == "rf":
        return dict(n_estimators=int(pick([100, 200, 300])),
                    max_depth=pick([None, 5, 10, 20]),
                    min_samples_leaf=int(pick([1, 2, 4])))
    if model == "mlp":
        return dict(hidden_layer_sizes=pick([(64,), (128,), (64, 32)]),
                    alpha=float(10 ** rng.uniform(-5, -1)))
    if model == "xgb":
        return dict(n_estimators=int(pick([100, 200, 300])),
                    max_depth=int(pick([3, 5, 7])),
                    learning_rate=float(pick([0.01, 0.05, 0.1, 0.2])),
                    subsample=float(pick([0.7, 0.85, 1.0])))
    raise ValueError(f"unknown model {model!r}")


# ── Nested GroupKFold CV ──────────────────────────────────────────────────────

def run_nested_cv(
    precomputed: dict,
    representation: str,
    model: str,
    *,
    n_outer_folds: int = 5,
    n_repeats: int = 3,
    n_inner_folds: int = 3,
    n_hpo_iters: int = 8,
    n_seeds: int = 3,
    window_grid: tuple = WINDOW_GRID,
    threshold: float = THRESHOLD,
    base_seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Nested infant-level CV with random-search HPO over (window_size, model params).

    Outer: StratifiedGroupKFold x n_repeats, split on infants.
    Inner: StratifiedGroupKFold HPO, selection = infant mean-prob AUC.
    Refit best config on full outer-train (multi-seed for stochastic models).
    Aggregate predictions both ways (mean_prob, majority_vote).

    Returns (fold_records, pred_records).
    """
    kind = "scale" if representation == "signal" else "tfidf"
    fit_fn = FIT_FNS[model]
    n_seeds_use = n_seeds if model in STOCHASTIC_MODELS else 1

    any_win = next(iter(precomputed.values()))
    g_all, y_all = any_win["groups"], any_win["y"]
    infants = np.array(sorted(np.unique(g_all)))
    y_infant = np.array([y_all[g_all == inf][0] for inf in infants])
    inf_label = dict(zip(infants, y_infant))

    fold_records: list[dict] = []
    pred_records: list[dict] = []

    for repeat in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True,
                                      random_state=base_seed * 1000 + repeat)
        for fold, (tr_i, te_i) in enumerate(
                outer.split(np.zeros(len(infants)), y_infant, groups=infants)):
            train_inf = infants[tr_i]
            test_inf  = infants[te_i]
            y_tr_inf  = y_infant[tr_i]

            rng = np.random.default_rng(base_seed * 10_000 + repeat * 100 + fold)
            inner = StratifiedGroupKFold(
                n_splits=n_inner_folds, shuffle=True,
                random_state=base_seed * 100_000 + repeat * 1000 + fold)
            inner_splits = list(inner.split(
                np.zeros(len(train_inf)), y_tr_inf, groups=train_inf))

            best = None  # (win, params, mean_inner_auc)
            for _ in range(n_hpo_iters):
                win    = window_grid[rng.integers(len(window_grid))]
                params = _sample_params(model, rng)
                X, y, g = (precomputed[win]["X"],
                            precomputed[win]["y"],
                            precomputed[win]["groups"])

                inner_aucs = []
                for in_tr_i, in_val_i in inner_splits:
                    in_tr_inf  = train_inf[in_tr_i]
                    in_val_inf = train_inf[in_val_i]
                    tr_m  = np.isin(g, in_tr_inf)
                    val_m = np.isin(g, in_val_inf)
                    if len(np.unique(y[tr_m])) < 2 or not val_m.any():
                        continue
                    if len({inf_label[i] for i in in_val_inf}) < 2:
                        continue
                    Xtr, Xte = _preprocess(X[tr_m], X[val_m], kind)
                    scores = fit_fn(Xtr, y[tr_m], Xte, **params)
                    inf_score, _ = aggregate_mean_prob(scores, g[val_m], in_val_inf, threshold)
                    inner_aucs.append(roc_auc_score(
                        [inf_label[i] for i in in_val_inf], inf_score))

                mean_auc = float(np.mean(inner_aucs)) if inner_aucs else 0.5
                if best is None or mean_auc > best[2]:
                    best = (win, params, mean_auc)

            best_win, best_params, _ = best
            X, y, g = (precomputed[best_win]["X"],
                        precomputed[best_win]["y"],
                        precomputed[best_win]["groups"])
            tr_m = np.isin(g, train_inf)
            te_m = np.isin(g, test_inf)
            Xtr, Xte = _preprocess(X[tr_m], X[te_m], kind)

            seed_scores = [fit_fn(Xtr, y[tr_m], Xte, **best_params, seed=s)
                           for s in range(n_seeds_use)]
            mean_scores = np.mean(seed_scores, axis=0)
            y_te_inf = np.array([inf_label[i] for i in test_inf])

            for agg_name, agg_fn in AGGREGATORS.items():
                inf_score, inf_pred = agg_fn(mean_scores, g[te_m], test_inf, threshold)
                m = infant_metrics(y_te_inf, inf_score, inf_pred)
                fold_records.append(dict(
                    model=model, repeat=repeat, outer_fold=fold,
                    window_size=best_win, n_test_infants=len(test_inf),
                    aggregation=agg_name, **m,
                ))
                if agg_name == "mean_prob":
                    for inf, yt, ys in zip(test_inf, y_te_inf, inf_score):
                        pred_records.append(dict(
                            model=model, repeat=repeat, outer_fold=fold,
                            infant_id=str(inf), y_true=int(yt), y_score=float(ys),
                        ))

    return fold_records, pred_records


# ── Session-level GroupKFold CV (no HPO) ─────────────────────────────────────

def run_session_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str = "logreg",
    kind: str = "scale",
    *,
    n_folds: int = 5,
    seed: int = 42,
    threshold: float = THRESHOLD,
) -> dict:
    """GroupKFold(infant) on session-level features. Out-of-fold session
    probabilities are averaged per infant (soft vote) for infant-level metrics.

    kind: 'scale' (signal features) | 'tfidf' (token count features).
    Returns session + infant-level metrics dict.
    """
    fit_fn = FIT_FNS[model]
    session_prob = np.zeros(len(y))

    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups=groups):
        Xtr, Xte = _preprocess(X[tr], X[te], kind)
        session_prob[te] = fit_fn(Xtr, y[tr], Xte, seed=seed)

    session_pred = (session_prob >= threshold).astype(int)
    infants = np.array(sorted(np.unique(groups)))
    y_infant = np.array([y[groups == inf][0] for inf in infants])
    inf_score, inf_pred = aggregate_mean_prob(session_prob, groups, infants, threshold)

    return dict(
        n_sessions=len(y),
        n_infants=len(infants),
        session_auc=roc_auc_score(y, session_prob) if len(np.unique(y)) > 1 else float("nan"),
        session_bacc=balanced_accuracy_score(y, session_pred),
        **{f"infant_{k}": v
           for k, v in infant_metrics(y_infant, inf_score, inf_pred).items()},
    )


