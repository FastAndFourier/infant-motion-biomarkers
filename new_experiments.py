"""
new_experiments.py  —  Experiments 1, 2, 3

Exp 1  Nested GroupKFold (outer eval / inner HPO) at three scales:
         window + aggregate  |  session  |  infant
       Representations: signal (stats, 140-dim) | symbolic (TF-IDF unigrams)
                       | embedding (FM continuous) | fm_tokens (FM 5s action)
       Models: LR, SVC, RF, XGBoost
       Window aggregation: soft vote | hard vote | MIL vote (max prob)
       Output: paired t-test comparison across configs

Exp 2  Early prediction — infant scale only, growing signal
       Axes: first k% sessions | sessions up to age T months
       Config: default (all models) or best (from exp1 HPO)

Exp 3  Token biomarkers per infant (whole signal pooled), T vs A test
       Metrics from token_analysis notebook + cross-modality sync
       Mann-Whitney U + Cohen's d + FDR

Usage:
  python new_experiments.py --exp 1
  python new_experiments.py --exp 2
  python new_experiments.py --exp 2 --exp2-config best
  python new_experiments.py --exp 3
  python new_experiments.py --exp all
"""

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

from src.tfidf_outcome import LognormSmoothTfidf
from src.dataset import OutcomeDataset

from src.exp_config import (
    CANDIDATES, ENCODERS, ENCODERS_5S, TOKENISE_CHOICES,
    WIN_SEC, HOP_SEC, VOCAB_SIZE, N_GROUPS, GROUP_VOCAB,
    GROUP_NAMES, GROUP_OFFSETS, TOKENS_PER_SEC,
    THRESHOLD, N_OUTER, N_INNER, N_SEEDS, SEED, SEEDS,
    MODELS, MODELS_WINDOW, OUT_DIR,
    session_age, cohen_d,
)

from src.repr_signal import build_signal, signal_infant_frac, signal_infant_age, signal_window_dur
from src.repr_symbolic import build_tfidf, tfidf_infant_frac, tfidf_infant_age, tfidf_window_dur, run_exp3_candidate
from src.repr_fm import (
    build_embedding, build_tfidf_fm,
    emb_infant_frac, emb_infant_age,
    emb_window_dur, fm_tok_window_dur,
    fm_tok_infant_frac, fm_tok_infant_age,
    run_exp3_encoder, run_exp3_encoder_tokens,
    precompute_embeddings, precompute_tokens, token_cache_path,
)

warnings.filterwarnings("ignore")


# ── Data loading ─────────────────────────────────────────────────────────────

def pools():
    return OutcomeDataset(win=WIN_SEC, hop=HOP_SEC, deterministic=True)._pools


def pools_5s():
    return OutcomeDataset(win=5.0, hop=5.0, deterministic=True)._pools


# ── Preprocessing (fold-safe) ───────────────────────────────────────────────

def preprocess(Xtr, Xte, kind):
    if kind != "scale":
        from scipy.sparse import csr_matrix, issparse
        if not issparse(Xtr):
            Xtr, Xte = csr_matrix(Xtr), csr_matrix(Xte)
    pre = StandardScaler() if kind == "scale" else LognormSmoothTfidf()
    Xtr = pre.fit_transform(Xtr)
    Xte = pre.transform(Xte)
    return Xtr, Xte


# ── Models + HPO ─────────────────────────────────────────────────────────────

_USE_GPU_XGB = False


def set_xgb_gpu(enabled: bool = True):
    global _USE_GPU_XGB
    _USE_GPU_XGB = enabled


def make_model(name: str, params: dict = {}, seed: int = SEED):
    defaults = {
        "lr":  dict(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=500, tol=1e-3),
        "svc": dict(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                    probability=True),
        "rf":  dict(n_estimators=200, class_weight="balanced", n_jobs=4),
        "xgb": dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.85,
                    eval_metric="logloss", n_jobs=4),
    }
    kw = {**defaults[name], **params, "random_state": seed}
    if name == "lr":  return LogisticRegression(**kw)
    if name == "svc": return SVC(**kw)
    if name == "rf":  return RandomForestClassifier(**kw)
    if name == "xgb":
        kw.pop("class_weight", None)
        if _USE_GPU_XGB:
            kw["device"] = "cuda"
        return XGBClassifier(**kw)
    raise ValueError(name)


def param_grid(name: str) -> list[dict]:
    """Exhaustive grid per model — small, realistic, deterministic."""
    from itertools import product
    if name == "lr":
        return ([dict(C=c, penalty="l2") for c in [0.01, 0.1, 1.0, 10.0]]
                + [dict(C=1.0, penalty=None)])
    if name == "svc":
        return [dict(C=c, kernel=k)
                for c, k in product([0.1, 1.0, 10.0], ["rbf", "linear"])]
    if name == "rf":
        return [dict(n_estimators=200, max_depth=d, min_samples_leaf=l)
                for d, l in product([None, 5, 10], [1, 3])]
    if name == "xgb":
        return [dict(n_estimators=200, max_depth=d, learning_rate=lr, subsample=s)
                for d, lr, s in product([3, 5], [0.05, 0.1], [0.8, 1.0])]
    raise ValueError(name)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_score, y_pred) -> dict:
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    return dict(auc=auc, bacc=balanced_accuracy_score(y_true, y_pred),
                sens=sens, spec=spec, ppv=ppv, npv=npv, n=len(y_true))


# ── Aggregation (window → infant) ───────────────────────────────────────────

def aggregate(window_scores, window_groups, all_inf_labels, threshold=THRESHOLD):
    infants = sorted(set(window_groups))
    y_true  = np.array([all_inf_labels[i] for i in infants])
    results = {}
    for agg in ("soft", "hard", "mil"):
        scores = np.zeros(len(infants))
        for j, inf in enumerate(infants):
            w = window_scores[window_groups == inf]
            if agg == "soft": scores[j] = w.mean()
            elif agg == "hard": scores[j] = (w >= threshold).mean()
            elif agg == "mil":  scores[j] = w.max()
        preds = (scores >= threshold).astype(int)
        results[agg] = (y_true, scores, preds)
    return results


def set_xgb_weight(clf, y_tr):
    if isinstance(clf, XGBClassifier):
        n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
        clf.set_params(scale_pos_weight=n_neg / max(n_pos, 1))


# ── Nested GroupKFold CV ─────────────────────────────────────────────────────

def _scores_to_infant(scores, groups, inf_label, scale):
    """Aggregate sample-level scores to infant-level (y_true, y_score, y_pred)."""
    if scale == "window":
        return aggregate(scores, groups, inf_label)
    inf_s: dict[str, list] = defaultdict(list)
    for inf, s in zip(groups, scores):
        inf_s[inf].append(s)
    val_infs = sorted(inf_s)
    yt = np.array([inf_label[i] for i in val_infs])
    ys = np.array([np.mean(inf_s[i]) for i in val_infs])
    yp = (ys >= THRESHOLD).astype(int)
    return {"direct": (yt, ys, yp)}


def nested_cv(X, y, groups, model_name, kind,
              n_outer=N_OUTER, n_inner=N_INNER,
              scale="session", seeds=None, fixed_params=None):
    """
    5-fold StratifiedGroupKFold outer × 5-fold inner HPO, all scales.
    If fixed_params is given, skip inner HPO and use those params directly.
    Returns list of per-fold result dicts, each including 'best_params'.
    """
    if seeds is None:
        seeds = [SEED]

    inf_label = {}
    for inf, label in zip(groups, y):
        inf_label[inf] = int(label)

    all_records = []
    total_iters = len(seeds) * n_outer
    pbar = tqdm(total=total_iters, desc=f"  {model_name}", unit="fold", leave=False)

    for seed in seeds:
        infants = np.array(sorted(np.unique(groups)))
        y_inf   = np.array([inf_label[i] for i in infants])
        outer   = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

        for fold, (tr_i, te_i) in enumerate(outer.split(np.zeros(len(infants)), y_inf, infants)):
            train_inf = infants[tr_i]
            test_inf  = infants[te_i]
            y_tr_inf  = y_inf[tr_i]

            if fixed_params is not None:
                best_params = fixed_params
                best_auc = np.nan
            else:
                inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                             random_state=seed * 100 + fold)
                inner_splits = list(inner.split(np.zeros(len(train_inf)), y_tr_inf, train_inf))

                best_params, best_auc = {}, -1.0
                for params in param_grid(model_name):
                    inner_aucs = []
                    for in_tr_i, in_val_i in inner_splits:
                        in_tr_inf  = train_inf[in_tr_i]
                        in_val_inf = train_inf[in_val_i]
                        if len(np.unique(y_tr_inf[in_tr_i])) < 2: continue
                        if len(np.unique([inf_label[i] for i in in_val_inf])) < 2: continue
                        tr_m  = np.isin(groups, in_tr_inf)
                        val_m = np.isin(groups, in_val_inf)
                        Xtr, Xval = preprocess(X[tr_m], X[val_m], kind)
                        clf = make_model(model_name, params, seed=seed)
                        if isinstance(clf, SVC):
                            clf.set_params(probability=False)
                        set_xgb_weight(clf, y[tr_m])
                        clf.fit(Xtr, y[tr_m])
                        val_scores = clf.decision_function(Xval) if isinstance(clf, SVC) else clf.predict_proba(Xval)[:, 1]

                        agg = _scores_to_infant(val_scores, groups[val_m], inf_label, scale)
                        agg_key = next(iter(agg))
                        yt, ys, _ = agg[agg_key]
                        if len(np.unique(yt)) < 2: continue
                        inner_aucs.append(roc_auc_score(yt, ys))

                    mean_auc = float(np.mean(inner_aucs)) if inner_aucs else 0.5
                    if mean_auc > best_auc:
                        best_auc, best_params = mean_auc, params

            tr_m = np.isin(groups, train_inf)
            te_m = np.isin(groups, test_inf)
            Xtr, Xte = preprocess(X[tr_m], X[te_m], kind)
            clf = make_model(model_name, best_params, seed=seed)
            set_xgb_weight(clf, y[tr_m])
            clf.fit(Xtr, y[tr_m])
            test_scores = clf.predict_proba(Xte)[:, 1]

            agg = _scores_to_infant(test_scores, groups[te_m], inf_label, scale)
            for agg_name, (yt, ys, yp) in agg.items():
                all_records.append(dict(fold=fold, aggregation=agg_name, seed=seed,
                                        best_params=best_params, inner_auc=best_auc,
                                        **compute_metrics(yt, ys, yp)))
            pbar.update(1)

    pbar.close()
    return all_records


# ── Experiment 1 ─────────────────────────────────────────────────────────────

def run_exp1(scales=None, candidate=None, model_filter=None):
    if scales is None: scales = ["window", "session", "infant"]
    all_special = set(ENCODERS) | set(ENCODERS_5S) | {"signal"}
    cands_sym = [candidate] if candidate and candidate in CANDIDATES else (
                 [] if candidate in all_special else list(CANDIDATES))
    run_signal = candidate in (None, "signal")
    run_encoders = ([candidate] if candidate in ENCODERS
                    else ([] if candidate and candidate not in ENCODERS else ENCODERS))

    print("\n" + "="*70)
    print(f"EXPERIMENT 1  scales={scales}  candidate={candidate or 'all'}")
    print("="*70)

    ps = pools()
    records = []

    for scale in scales:
        t_scale = time.time()
        model_list = MODELS_WINDOW if scale == "window" else MODELS
        if model_filter:
            model_list = [m for m in model_list if m in model_filter]

        if run_signal:
            Xs, ys, gs = build_signal(scale, ps)
            print(f"\n  {scale}/signal {Xs.shape}")
            for model_name in model_list:
                folds = nested_cv(Xs, ys, gs, model_name, "scale",
                                  scale=scale, seeds=SEEDS)
                for fr in folds:
                    records.append(dict(scale=scale, repr="signal",
                                        candidate="—", model=model_name, **fr))
                print(f"    {model_name}: AUC={np.mean([f['auc'] for f in folds]):.3f}")

        for cand in cands_sym:
            Xt, yt, gt = build_tfidf(scale, cand)
            print(f"\n  {scale}/symbolic/{cand} {Xt.shape}")
            for model_name in model_list:
                folds = nested_cv(Xt, yt, gt, model_name, "tfidf",
                                  scale=scale, seeds=SEEDS)
                for fr in folds:
                    records.append(dict(scale=scale, repr="symbolic",
                                        candidate=cand, model=model_name, **fr))
                print(f"    {model_name}: AUC={np.mean([f['auc'] for f in folds]):.3f}")

        for enc in run_encoders:
            Xe, ye, ge = build_embedding(scale, enc, ps)
            print(f"\n  {scale}/embedding/{enc} {Xe.shape}")
            for model_name in model_list:
                folds = nested_cv(Xe, ye, ge, model_name, "scale",
                                  scale=scale, seeds=SEEDS)
                for fr in folds:
                    records.append(dict(scale=scale, repr="embedding",
                                        candidate=enc, model=model_name, **fr))
                print(f"    {model_name}: AUC={np.mean([f['auc'] for f in folds]):.3f}")

        run_fm5s = ([candidate] if candidate in ENCODERS_5S
                    else ([] if candidate and candidate not in ENCODERS_5S else ENCODERS_5S))
        if run_fm5s:
            ps5 = pools_5s()
            for enc in run_fm5s:
                tok_path = token_cache_path(enc)
                if not os.path.exists(tok_path):
                    print(f"\n  Skipping {enc} (no tokens at {tok_path})")
                    continue
                Xf, yf, gf = build_tfidf_fm(scale, enc, ps5)
                print(f"\n  {scale}/fm_tokens/{enc} {Xf.shape}")
                for model_name in model_list:
                    folds = nested_cv(Xf, yf, gf, model_name, "tfidf",
                                      scale=scale, seeds=SEEDS)
                    for fr in folds:
                        records.append(dict(scale=scale, repr="fm_tokens",
                                            candidate=enc, model=model_name, **fr))
                    print(f"    {model_name}: AUC={np.mean([f['auc'] for f in folds]):.3f}")

        print(f"  [{scale} done in {(time.time()-t_scale)/60:.1f} min]")

    df = pd.DataFrame(records)
    model_tag = f"_{'_'.join(sorted(model_filter))}" if model_filter else ""
    slice_tag = f"{'_'.join(scales)}_{candidate or 'all'}{model_tag}"
    path = os.path.join(OUT_DIR, f"exp1_{slice_tag}.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    if not model_filter:
        save_exp1_configs(df)
    return df


def save_exp1_configs(df: pd.DataFrame):
    """Extract best (model, params) per repr/candidate/scale from exp1 results."""
    configs = {}
    for (scale, repr_name, cand), grp in df.groupby(["scale", "repr", "candidate"]):
        best_model = grp.groupby("model")["auc"].mean().idxmax()
        model_rows = grp[grp["model"] == best_model]
        all_params = model_rows["best_params"].tolist()
        parsed = []
        for p in all_params:
            if isinstance(p, dict) and p:
                parsed.append(p)
            elif isinstance(p, str):
                import ast
                try:
                    d = ast.literal_eval(p)
                    if isinstance(d, dict) and d:
                        parsed.append(d)
                except (ValueError, SyntaxError):
                    pass
        best = parsed[0] if parsed else {}
        serialisable = {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in best.items()}
        key = f"{scale}/{repr_name}/{cand}"
        configs[key] = dict(model=best_model, params=serialisable,
                            mean_auc=float(grp[grp["model"] == best_model]["auc"].mean()))
    path = os.path.join(OUT_DIR, "exp1_best_configs.json")
    with open(path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"  Best configs → {path}")


def load_exp1_configs() -> dict:
    path = os.path.join(OUT_DIR, "exp1_best_configs.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No exp1 configs at {path}. Run exp1 first.")
    with open(path) as f:
        return json.load(f)


def run_exp1_collect():
    import glob
    files = glob.glob(os.path.join(OUT_DIR, "exp1_*.csv"))
    files = [f for f in files if not any(x in os.path.basename(f) for x in ["results", "ttests"])]
    if not files:
        print("No exp1 slice files found."); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates()
    path = os.path.join(OUT_DIR, "exp1_results.csv")
    df.to_csv(path, index=False)
    print(f"Merged {len(files)} slices → {path}")
    print("\n── Exp 1 Summary (mean AUC across seeds) ──")
    print(df.groupby(["scale","repr","candidate","model","aggregation"])["auc"]
            .mean().round(3).to_string())
    paired_ttests(df)
    plot_exp1(df)
    save_exp1_configs(df)


def paired_ttests(df: pd.DataFrame):
    print("\n── Paired t-tests (signal/lr vs others, per scale) ──")
    records = []
    for scale in df["scale"].unique():
        sub = df[df["scale"] == scale]
        configs = sub.groupby(["repr", "model", "aggregation"])
        auc_per_config: dict[str, list] = {}
        for cfg, grp in configs:
            key = "/".join(cfg)
            auc_per_config[key] = grp["auc"].tolist()

        baselines = [k for k in auc_per_config if k.startswith("signal/lr")]
        for baseline in baselines:
            b_aucs = np.array(auc_per_config[baseline])
            for cfg, aucs in auc_per_config.items():
                if cfg == baseline: continue
                a_aucs = np.array(aucs)
                n = min(len(b_aucs), len(a_aucs))
                if n < 2: continue
                t, p = ttest_rel(b_aucs[:n], a_aucs[:n])
                records.append(dict(scale=scale, baseline=baseline, config=cfg,
                                    delta_auc=float(a_aucs[:n].mean() - b_aucs[:n].mean()),
                                    t=t, p=p))

    df_t = pd.DataFrame(records)
    if len(df_t):
        _, df_t["p_holm"], _, _ = multipletests(df_t["p"], method="holm")
        df_t["sig"] = df_t["p_holm"].apply(
            lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "")))
        path = os.path.join(OUT_DIR, "exp1_ttests.csv")
        df_t.to_csv(path, index=False)
        sig = df_t[df_t["sig"] != ""]
        if len(sig):
            print(sig[["scale", "baseline", "config", "delta_auc", "p_holm", "sig"]].to_string(index=False))
        else:
            print("  No significant differences after Holm correction.")
        print(f"  Full table: {path}")
    return df_t


def plot_exp1(df: pd.DataFrame):
    scales = ["window", "session", "infant"]
    repr_names = [r for r in ["signal", "symbolic", "embedding", "fm_tokens"] if r in df["repr"].values]
    n_repr = len(repr_names)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"lr": "#4C9BE8", "svc": "#E8724C", "rf": "#5CBF5C", "xgb": "#B07FCC"}
    hatches = {"signal": "", "symbolic": "///", "embedding": "xxx", "fm_tokens": "..."}

    for ax, scale in zip(axes, scales):
        sub = df[df["scale"] == scale]
        agg_summary = (sub.groupby(["repr", "model", "aggregation"])["auc"]
                       .mean().reset_index())
        default_agg = "soft" if scale in ("window", "session") else "direct"
        agg_summary = agg_summary[agg_summary["aggregation"] == default_agg]

        x = np.arange(len(MODELS))
        w = 0.8 / max(n_repr, 1)
        for ri, repr_name in enumerate(repr_names):
            vals = []
            for m in MODELS:
                row = agg_summary[(agg_summary["repr"] == repr_name) &
                                  (agg_summary["model"] == m)]["auc"].values
                vals.append(row[0] if len(row) else np.nan)
            ax.bar(x + ri * w, vals, w, label=repr_name,
                   color=[colors[m] for m in MODELS],
                   hatch=hatches.get(repr_name, ""), alpha=0.8,
                   edgecolor="k", linewidth=0.5)

        ax.set_xticks(x + w * (n_repr - 1) / 2)
        ax.set_xticklabels([m.upper() for m in MODELS])
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC (soft vote / direct)" if scale == "window" else "")
        ax.set_title(f"Scale: {scale} ({default_agg})")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    plt.suptitle("Experiment 1 — Nested GroupKFold AUC", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figs", "exp1_auc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path}")


# ── Experiment 2 ─────────────────────────────────────────────────────────────

def run_exp2(candidate=None, config_mode="default", model_filter=None):
    """
    config_mode: "default" = all 4 models with default params
                 "best"    = best model+params from exp1 (infant + window scale)
    """
    all_special = set(ENCODERS) | set(ENCODERS_5S) | {"signal"}
    cands_sym = [candidate] if candidate and candidate in CANDIDATES else (
                 [] if candidate in all_special else list(CANDIDATES))
    run_signal = candidate in (None, "signal")
    run_encoders = ([candidate] if candidate in ENCODERS
                    else ([] if candidate and candidate not in ENCODERS else ENCODERS))

    if config_mode == "best":
        exp1_configs = load_exp1_configs()
        print("Using best configs from exp1")
    else:
        exp1_configs = None

    use_fixed = config_mode == "best"

    def get_model_list(repr_name, cand_name, scale="infant"):
        if exp1_configs is None:
            base = MODELS_WINDOW if scale == "window" else MODELS
            if model_filter:
                base = [m for m in base if m in model_filter]
            return [(m, {}) for m in base]
        key = f"{scale}/{repr_name}/{cand_name}"
        if key in exp1_configs:
            cfg = exp1_configs[key]
            return [(cfg["model"], cfg["params"])]
        return [(m, {}) for m in (MODELS_WINDOW if scale == "window" else MODELS)]

    print("\n" + "="*70)
    print(f"EXPERIMENT 2  candidate={candidate or 'all'}  config={config_mode}")
    print("="*70)

    ps = pools()
    FRACS    = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    AGE_CUTS = [3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 15.0]
    records  = []

    print("\n── Axis 1: fraction of sessions ──")
    t_axis1 = time.time()
    for frac in FRACS:
        yi = None
        if run_signal:
            Xi_sig, yi = signal_infant_frac(ps, frac)
            g = np.arange(len(yi)).astype(str)
            for model_name, params in get_model_list("signal", "—"):
                folds = nested_cv(Xi_sig, yi, g, model_name, "scale", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="frac", value=frac, repr="signal",
                                    candidate="—", model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  frac={frac:.2f}  signal  AUC={records[-1]['auc']:.3f}")
        if yi is None:
            yi = np.array([p["label"] for p in ps], np.int64)
        g = np.arange(len(yi)).astype(str)
        for cand in cands_sym:
            Xi_tok, _ = tfidf_infant_frac(cand, frac)
            for model_name, params in get_model_list("symbolic", cand):
                folds = nested_cv(Xi_tok, yi, g, model_name, "tfidf", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="frac", value=frac, repr="symbolic",
                                    candidate=cand, model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  frac={frac:.2f}  {cand}  AUC={records[-1]['auc']:.3f}")
        for enc in run_encoders:
            Xi_emb, _ = emb_infant_frac(enc, ps, frac)
            for model_name, params in get_model_list("embedding", enc):
                folds = nested_cv(Xi_emb, yi, g, model_name, "scale", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="frac", value=frac, repr="embedding",
                                    candidate=enc, model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  frac={frac:.2f}  {enc}  AUC={records[-1]['auc']:.3f}")
        run_fm5s = ([candidate] if candidate in ENCODERS_5S
                    else ([] if candidate and candidate not in ENCODERS_5S else ENCODERS_5S))
        if run_fm5s:
            ps5 = pools_5s()
        for enc in run_fm5s:
            if not os.path.exists(token_cache_path(enc)):
                continue
            Xi_ft, _ = fm_tok_infant_frac(enc, ps5, frac)
            for model_name, params in get_model_list("fm_tokens", enc):
                folds = nested_cv(Xi_ft, yi, g, model_name, "tfidf", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="frac", value=frac, repr="fm_tokens",
                                    candidate=enc, model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  frac={frac:.2f}  {enc}  AUC={records[-1]['auc']:.3f}")

    print(f"  [axis1 done in {(time.time()-t_axis1)/60:.1f} min]")
    print("\n── Axis 2: age cutoff (months) ──")
    t_axis2 = time.time()
    for age_max in AGE_CUTS:
        yi = None
        if run_signal:
            Xi_sig, yi = signal_infant_age(ps, age_max)
            g = np.arange(len(yi)).astype(str)
            for model_name, params in get_model_list("signal", "—"):
                folds = nested_cv(Xi_sig, yi, g, model_name, "scale", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="age", value=age_max, repr="signal",
                                    candidate="—", model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  age≤{age_max:.1f}mo  signal  AUC={records[-1]['auc']:.3f}")
        if yi is None:
            yi = np.array([p["label"] for p in ps], np.int64)
        g = np.arange(len(yi)).astype(str)
        for cand in cands_sym:
            Xi_tok, _ = tfidf_infant_age(cand, age_max)
            for model_name, params in get_model_list("symbolic", cand):
                folds = nested_cv(Xi_tok, yi, g, model_name, "tfidf", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="age", value=age_max, repr="symbolic",
                                    candidate=cand, model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  age≤{age_max:.1f}mo  {cand}  AUC={records[-1]['auc']:.3f}")
        for enc in run_encoders:
            Xi_emb, _ = emb_infant_age(enc, ps, age_max)
            for model_name, params in get_model_list("embedding", enc):
                folds = nested_cv(Xi_emb, yi, g, model_name, "scale", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="age", value=age_max, repr="embedding",
                                    candidate=enc, model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  age≤{age_max:.1f}mo  {enc}  AUC={records[-1]['auc']:.3f}")
        run_fm5s = ([candidate] if candidate in ENCODERS_5S
                    else ([] if candidate and candidate not in ENCODERS_5S else ENCODERS_5S))
        if run_fm5s:
            ps5 = pools_5s()
        for enc in run_fm5s:
            if not os.path.exists(token_cache_path(enc)):
                continue
            Xi_ft, _ = fm_tok_infant_age(enc, ps5, age_max)
            for model_name, params in get_model_list("fm_tokens", enc):
                folds = nested_cv(Xi_ft, yi, g, model_name, "tfidf", scale="infant",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                records.append(dict(axis="age", value=age_max, repr="fm_tokens",
                                    candidate=enc, model=model_name,
                                    auc=np.mean([f["auc"] for f in folds]),
                                    bacc=np.mean([f["bacc"] for f in folds]),
                                    sens=np.mean([f["sens"] for f in folds]),
                                    spec=np.mean([f["spec"] for f in folds])))
            print(f"  age≤{age_max:.1f}mo  {enc}  AUC={records[-1]['auc']:.3f}")

    print(f"  [axis2 done in {(time.time()-t_axis2)/60:.1f} min]")

    DUR_CUTS = [5, 10, 15, 30, 45, 60]
    print("\n── Axis 3: duration cutoff (minutes), window scale + soft vote ──")
    t_axis3 = time.time()
    for dur in DUR_CUTS:
        if run_signal:
            X_w, y_w, g_w = signal_window_dur(ps, dur)
            for model_name, params in get_model_list("signal", "—", scale="window"):
                folds = nested_cv(X_w, y_w, g_w, model_name, "scale", scale="window",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                soft = [f for f in folds if f["aggregation"] == "soft"]
                records.append(dict(axis="duration", value=dur, repr="signal",
                                    candidate="—", model=model_name,
                                    auc=np.mean([f["auc"] for f in soft]),
                                    bacc=np.mean([f["bacc"] for f in soft]),
                                    sens=np.mean([f["sens"] for f in soft]),
                                    spec=np.mean([f["spec"] for f in soft])))
            print(f"  dur≤{dur}min  signal  AUC={records[-1]['auc']:.3f}")
        for cand in cands_sym:
            X_w, y_w, g_w = tfidf_window_dur(cand, dur)
            for model_name, params in get_model_list("symbolic", cand, scale="window"):
                folds = nested_cv(X_w, y_w, g_w, model_name, "tfidf", scale="window",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                soft = [f for f in folds if f["aggregation"] == "soft"]
                records.append(dict(axis="duration", value=dur, repr="symbolic",
                                    candidate=cand, model=model_name,
                                    auc=np.mean([f["auc"] for f in soft]),
                                    bacc=np.mean([f["bacc"] for f in soft]),
                                    sens=np.mean([f["sens"] for f in soft]),
                                    spec=np.mean([f["spec"] for f in soft])))
            print(f"  dur≤{dur}min  {cand}  AUC={records[-1]['auc']:.3f}")
        for enc in run_encoders:
            X_w, y_w, g_w = emb_window_dur(enc, ps, dur)
            for model_name, params in get_model_list("embedding", enc, scale="window"):
                folds = nested_cv(X_w, y_w, g_w, model_name, "scale", scale="window",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                soft = [f for f in folds if f["aggregation"] == "soft"]
                records.append(dict(axis="duration", value=dur, repr="embedding",
                                    candidate=enc, model=model_name,
                                    auc=np.mean([f["auc"] for f in soft]),
                                    bacc=np.mean([f["bacc"] for f in soft]),
                                    sens=np.mean([f["sens"] for f in soft]),
                                    spec=np.mean([f["spec"] for f in soft])))
            print(f"  dur≤{dur}min  {enc}  AUC={records[-1]['auc']:.3f}")
        run_fm5s = ([candidate] if candidate in ENCODERS_5S
                    else ([] if candidate and candidate not in ENCODERS_5S else ENCODERS_5S))
        if run_fm5s:
            ps5 = pools_5s()
        for enc in run_fm5s:
            if not os.path.exists(token_cache_path(enc)):
                continue
            X_w, y_w, g_w = fm_tok_window_dur(enc, ps5, dur)
            for model_name, params in get_model_list("fm_tokens", enc, scale="window"):
                folds = nested_cv(X_w, y_w, g_w, model_name, "tfidf", scale="window",
                                  seeds=SEEDS, fixed_params=params if use_fixed else None)
                soft = [f for f in folds if f["aggregation"] == "soft"]
                records.append(dict(axis="duration", value=dur, repr="fm_tokens",
                                    candidate=enc, model=model_name,
                                    auc=np.mean([f["auc"] for f in soft]),
                                    bacc=np.mean([f["bacc"] for f in soft]),
                                    sens=np.mean([f["sens"] for f in soft]),
                                    spec=np.mean([f["spec"] for f in soft])))
            print(f"  dur≤{dur}min  {enc}  AUC={records[-1]['auc']:.3f}")
    print(f"  [axis3 done in {(time.time()-t_axis3)/60:.1f} min]")

    df = pd.DataFrame(records)
    slice_tag = candidate or "all"
    path = os.path.join(OUT_DIR, f"exp2_{slice_tag}.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


def run_exp2_collect():
    import glob
    files = glob.glob(os.path.join(OUT_DIR, "exp2_*.csv"))
    files = [f for f in files if "collect" not in f]
    if not files:
        print("No exp2 slice files found."); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
    path = os.path.join(OUT_DIR, "exp2_results.csv")
    df.to_csv(path, index=False)
    print(f"Merged {len(files)} slices → {path}")
    plot_exp2(df)


def plot_exp2(df):
    colors = {"signal": "#4C9BE8", "symbolic": "#E8724C", "embedding": "#2ECC71", "fm_tokens": "#B07FCC"}
    styles = {"lr": "-", "svc": "--", "rf": "-.", "xgb": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    repr_names = [r for r in ["signal", "symbolic", "embedding", "fm_tokens"] if r in df["repr"].values]
    for ax, axis in zip(axes, ["frac", "age"]):
        sub = df[df["axis"] == axis].sort_values("value")
        xlabel = "Fraction of sessions" if axis == "frac" else "Corrected age cutoff (months)"
        for repr_name in repr_names:
            for m in MODELS:
                s = sub[(sub["repr"] == repr_name) & (sub["model"] == m)]
                if s.empty: continue
                ax.plot(s["value"], s["auc"], color=colors.get(repr_name, "#999"),
                        linestyle=styles[m], marker="o", markersize=4,
                        linewidth=1.8, label=f"{repr_name}/{m}")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("AUC")
        ax.set_title(f"Exp 2 — {axis} cutoff (infant level)")
        ax.set_ylim(0.3, 1.0)
        if axis == "frac":
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figs", "exp2_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path}")

    dur_df = df[df["axis"] == "duration"]
    if dur_df.empty:
        return
    plot_exp2_duration(dur_df)


def plot_exp2_duration(df):
    colors = {"signal": "#4C9BE8", "symbolic": "#E8724C", "embedding": "#2ECC71", "fm_tokens": "#B07FCC"}
    styles = {"lr": "-", "svc": "--", "rf": "-.", "xgb": ":"}
    metrics = [("auc", "AUC"), ("sens", "Sensitivity"), ("spec", "Specificity")]
    repr_names = [r for r in ["signal", "symbolic", "embedding", "fm_tokens"] if r in df["repr"].values]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sub = df.sort_values("value")
    for ax, (metric, label) in zip(axes, metrics):
        for repr_name in repr_names:
            for m in set(sub["model"]):
                s = sub[(sub["repr"] == repr_name) & (sub["model"] == m)]
                if s.empty:
                    continue
                cand_str = s["candidate"].iloc[0]
                tag = f"{repr_name}/{m}" if cand_str == "—" else f"{repr_name}({cand_str})/{m}"
                ax.plot(s["value"], s[metric], color=colors.get(repr_name, "#999"),
                        linestyle=styles.get(m, "-"), marker="o", markersize=4,
                        linewidth=1.8, label=tag)
        if metric == "auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Available data per infant (minutes)")
        ax.set_ylabel(label)
        ax.set_title(f"Exp 2 — {label} (window + soft vote)")
        ax.set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figs", "exp2_duration_curves.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {path}")


# ── Experiment 3 ─────────────────────────────────────────────────────────────

def run_exp3(candidate=None):
    svd_variants = [v for v in TOKENISE_CHOICES if v not in ENCODERS]
    is_svd = candidate in svd_variants
    cands = ([candidate] if candidate and candidate in CANDIDATES
             else ([] if candidate in ENCODERS or is_svd else list(CANDIDATES)))
    run_encoders = ([candidate] if candidate in ENCODERS
                    else ([] if candidate and candidate not in ENCODERS else ENCODERS))

    print("\n" + "="*70)
    print(f"EXPERIMENT 3  candidate={candidate or 'all'}")
    print("="*70)

    for cand in cands:
        print(f"\n── Candidate: {cand} ──")
        t_cand = time.time()
        df_bm, df_stats = run_exp3_candidate(cand)

        sig = df_stats[df_stats["p_fdr"] < 0.05]
        print(f"  N: T={len(df_bm[df_bm.outcome==0])}  A={len(df_bm[df_bm.outcome==1])}")
        print(f"  Significant (FDR<0.05): {len(sig)}")
        if len(sig):
            pd.set_option("display.float_format", "{:.3f}".format)
            print(sig[["metric","group","cohen_d","p_fdr","sig"]].to_string(index=False))

        df_bm.to_csv(os.path.join(OUT_DIR, f"exp3_bm_{cand}.csv"), index=False)
        df_stats.to_csv(os.path.join(OUT_DIR, f"exp3_stats_{cand}.csv"), index=False)
        print(f"  Saved: exp3_bm_{cand}.csv  exp3_stats_{cand}.csv")
        print(f"  [{cand} done in {(time.time()-t_cand)/60:.1f} min]")
        plot_exp3(df_bm, df_stats)

    if run_encoders:
        ps = pools()
        for enc in run_encoders:
            print(f"\n── Encoder: {enc} ──")
            t_enc = time.time()
            df_bm, df_stats = run_exp3_encoder(enc, ps)

            if len(df_stats):
                sig = df_stats[df_stats["p_fdr"] < 0.05]
                print(f"  N: T={len(df_bm[df_bm.outcome==0])}  A={len(df_bm[df_bm.outcome==1])}")
                print(f"  Significant (FDR<0.05): {len(sig)}")
                if len(sig):
                    pd.set_option("display.float_format", "{:.3f}".format)
                    print(sig[["metric","group","cohen_d","p_fdr","sig"]].to_string(index=False))

            df_bm.to_csv(os.path.join(OUT_DIR, f"exp3_bm_{enc}.csv"), index=False)
            df_stats.to_csv(os.path.join(OUT_DIR, f"exp3_stats_{enc}.csv"), index=False)
            print(f"  Saved: exp3_bm_{enc}.csv  exp3_stats_{enc}.csv")
            print(f"  [{enc} done in {(time.time()-t_enc)/60:.1f} min]")
            plot_exp3(df_bm, df_stats)

            tok_path = token_cache_path(enc)
            if os.path.exists(tok_path):
                print(f"\n── Encoder tokens: {enc}_tok ──")
                t_tok = time.time()
                df_bm_t, df_stats_t = run_exp3_encoder_tokens(enc, ps)

                if len(df_stats_t):
                    sig = df_stats_t[df_stats_t["p_fdr"] < 0.05]
                    print(f"  N: T={len(df_bm_t[df_bm_t.outcome==0])}  A={len(df_bm_t[df_bm_t.outcome==1])}")
                    print(f"  Significant (FDR<0.05): {len(sig)}")
                    if len(sig):
                        pd.set_option("display.float_format", "{:.3f}".format)
                        print(sig[["metric","group","cohen_d","p_fdr","sig"]].to_string(index=False))

                df_bm_t.to_csv(os.path.join(OUT_DIR, f"exp3_bm_{enc}_tok.csv"), index=False)
                df_stats_t.to_csv(os.path.join(OUT_DIR, f"exp3_stats_{enc}_tok.csv"), index=False)
                print(f"  Saved: exp3_bm_{enc}_tok.csv  exp3_stats_{enc}_tok.csv")
                print(f"  [{enc}_tok done in {(time.time()-t_tok)/60:.1f} min]")
                plot_exp3(df_bm_t, df_stats_t)
            else:
                print(f"  Skipping {enc}_tok (no tokens at {tok_path}, "
                      f"run --precompute-tokens {enc})")

    if not candidate or is_svd:
        to_run = [candidate] if is_svd else svd_variants
        ps = pools()
        for enc in to_run:
            tok_path = token_cache_path(enc)
            if not os.path.exists(tok_path):
                print(f"  Skipping {enc}_tok (no tokens at {tok_path})")
                continue
            print(f"\n── SVD tokens: {enc}_tok ──")
            t_tok = time.time()
            df_bm_t, df_stats_t = run_exp3_encoder_tokens(enc, ps)

            if len(df_stats_t):
                sig = df_stats_t[df_stats_t["p_fdr"] < 0.05]
                print(f"  N: T={len(df_bm_t[df_bm_t.outcome==0])}  A={len(df_bm_t[df_bm_t.outcome==1])}")
                print(f"  Significant (FDR<0.05): {len(sig)}")
                if len(sig):
                    pd.set_option("display.float_format", "{:.3f}".format)
                    print(sig[["metric","group","cohen_d","p_fdr","sig"]].to_string(index=False))

            df_bm_t.to_csv(os.path.join(OUT_DIR, f"exp3_bm_{enc}_tok.csv"), index=False)
            df_stats_t.to_csv(os.path.join(OUT_DIR, f"exp3_stats_{enc}_tok.csv"), index=False)
            print(f"  Saved: exp3_bm_{enc}_tok.csv  exp3_stats_{enc}_tok.csv")
            print(f"  [{enc}_tok done in {(time.time()-t_tok)/60:.1f} min]")
            plot_exp3(df_bm_t, df_stats_t)

    if not candidate or candidate in ENCODERS_5S:
        to_run_5s = [candidate] if candidate in ENCODERS_5S else ENCODERS_5S
        ps_5s = pools_5s()
        for enc in to_run_5s:
            tok_path = token_cache_path(enc)
            if not os.path.exists(tok_path):
                print(f"  Skipping {enc}_tok (no tokens at {tok_path})")
                continue
            print(f"\n── 5s action tokens: {enc}_tok ──")
            t_tok = time.time()
            df_bm_t, df_stats_t = run_exp3_encoder_tokens(enc, ps_5s)

            if len(df_stats_t):
                sig = df_stats_t[df_stats_t["p_fdr"] < 0.05]
                print(f"  N: T={len(df_bm_t[df_bm_t.outcome==0])}  A={len(df_bm_t[df_bm_t.outcome==1])}")
                print(f"  Significant (FDR<0.05): {len(sig)}")
                if len(sig):
                    pd.set_option("display.float_format", "{:.3f}".format)
                    print(sig[["metric","group","cohen_d","p_fdr","sig"]].to_string(index=False))

            df_bm_t.to_csv(os.path.join(OUT_DIR, f"exp3_bm_{enc}_tok.csv"), index=False)
            df_stats_t.to_csv(os.path.join(OUT_DIR, f"exp3_stats_{enc}_tok.csv"), index=False)
            print(f"  Saved: exp3_bm_{enc}_tok.csv  exp3_stats_{enc}_tok.csv")
            print(f"  [{enc}_tok done in {(time.time()-t_tok)/60:.1f} min]")
            plot_exp3(df_bm_t, df_stats_t)


def run_exp3_collect():
    import glob
    bm_files    = glob.glob(os.path.join(OUT_DIR, "exp3_bm_*.csv"))
    stats_files = glob.glob(os.path.join(OUT_DIR, "exp3_stats_*.csv"))
    if not bm_files:
        print("No exp3 slice files found."); return
    df_bm    = pd.concat([pd.read_csv(f) for f in bm_files],    ignore_index=True)
    df_stats = pd.concat([pd.read_csv(f) for f in stats_files], ignore_index=True)
    df_bm.to_csv(os.path.join(OUT_DIR, "exp3_biomarkers.csv"), index=False)
    df_stats.to_csv(os.path.join(OUT_DIR, "exp3_stats.csv"),   index=False)
    print(f"Merged {len(bm_files)} candidates.")
    print("\n── Top 20 overall (by p_mwu) ──")
    print(df_stats.sort_values("p_mwu").head(20)
          [["candidate","metric","group","cohen_d","p_mwu","p_fdr","sig"]].to_string(index=False))


def plot_exp3(df_bm, df_stats):
    for cand in df_bm["candidate"].unique():
        sub_bm    = df_bm[df_bm["candidate"] == cand]
        sub_stats = df_stats[df_stats["candidate"] == cand]
        sig_m = sub_stats[sub_stats["p_fdr"] < 0.05]["metric"].tolist()
        metrics = sig_m if sig_m else sub_stats.head(12)["metric"].tolist()
        if not metrics: continue
        COLORS = {0: "#4C9BE8", 1: "#E8724C"}
        ncols = min(4, len(metrics))
        nrows = int(np.ceil(len(metrics) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.5))
        axes_flat = np.array(axes).flatten() if len(metrics) > 1 else [axes]
        for ax, m in zip(axes_flat, metrics):
            for outcome in [0, 1]:
                vals = sub_bm[sub_bm.outcome == outcome][m].dropna().values
                if not len(vals): continue
                parts = ax.violinplot([vals], positions=[outcome], showmedians=True, widths=0.6)
                for pc in parts["bodies"]: pc.set_facecolor(COLORS[outcome]); pc.set_alpha(0.6)
                for k in ("cmedians","cbars","cmins","cmaxes"):
                    if k in parts: parts[k].set_color(COLORS[outcome])
                ax.scatter([outcome]*len(vals), vals, c=COLORS[outcome], s=18, alpha=0.6, zorder=3)
            row = sub_stats[sub_stats.metric == m]
            if len(row):
                r = row.iloc[0]
                ax.set_title(f"{m}\np_fdr={r.p_fdr:.3f}{' '+r.sig}  d={r.cohen_d:.2f}", fontsize=7)
            ax.set_xticks([0,1]); ax.set_xticklabels(["T","A"], fontsize=9)
        for ax in axes_flat[len(metrics):]: ax.set_visible(False)
        patches = [mpatches.Patch(color=COLORS[o], label=["Typical","Atypical"][o]) for o in [0,1]]
        fig.legend(handles=patches, loc="lower right", fontsize=9)
        fig.suptitle(f"Exp 3 [{cand}] — Token biomarkers T vs A (FDR<0.05)", fontsize=9, y=1.01)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "figs", f"exp3_biomarkers_{cand}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure: {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",       choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--scale",     choices=["window", "session", "infant"], default=None)
    parser.add_argument("--candidate", choices=list(CANDIDATES) + ENCODERS + TOKENISE_CHOICES + ENCODERS_5S + ["signal"],
                        default=None)

    parser.add_argument("--precompute", nargs="+", choices=ENCODERS, default=None)
    parser.add_argument("--precompute-tokens", nargs="+", choices=TOKENISE_CHOICES, default=None)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=512)
    parser.add_argument("--mantis-ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model",     nargs="+", choices=MODELS, default=None,
                        help="Run only these models (default: all)")
    parser.add_argument("--collect",   action="store_true")
    parser.add_argument("--gpu",       action="store_true",
                        help="Use GPU for XGBoost (tree_method=gpu_hist)")
    parser.add_argument("--exp2-config", choices=["default", "best"], default="default",
                        help="Exp2 model config: 'default' (all models) or 'best' (from exp1)")

    args = parser.parse_args()

    if args.gpu:
        set_xgb_gpu(True)
        print("XGBoost: using GPU (tree_method=gpu_hist)")

    scales    = [args.scale] if args.scale else ["window", "session", "infant"]
    candidate = args.candidate

    t0 = time.time()

    if args.precompute:
        precompute_embeddings(args.precompute, mantis_ckpt=args.mantis_ckpt,
                              device=args.device)
    if args.precompute_tokens:
        precompute_tokens(args.precompute_tokens, mantis_ckpt=args.mantis_ckpt,
                          device=args.device,
                          n_components=args.n_components, n_clusters=args.n_clusters)
    if (args.precompute or args.precompute_tokens) and not args.collect:
        print(f"\nDone after {(time.time() - t0) / 60:.1f} minutes")
        raise SystemExit(0)
    if args.collect:
        if args.exp in ("1", "all"): run_exp1_collect()
        if args.exp in ("2", "all"): run_exp2_collect()
        if args.exp in ("3", "all"): run_exp3_collect()
    else:
        if args.exp in ("1", "all"): run_exp1(scales=scales, candidate=candidate, model_filter=args.model)
        if args.exp in ("2", "all"): run_exp2(candidate=candidate, config_mode=args.exp2_config, model_filter=args.model)
        if args.exp in ("3", "all"): run_exp3(candidate=candidate)

    print(f"\nDone after {(time.time() - t0) / 60:.1f} minutes")
