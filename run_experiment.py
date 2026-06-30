import argparse
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.outcome_cv import (
    ALL_CANDIDATES,
    MAIN_CANDIDATE,
    WINDOW_GRID,
    FIT_FNS,
    extract_signal,
    extract_tfidf,
    run_nested_cv,
)

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR  = "results"
PARTIAL_DIR  = f"{RESULTS_DIR}/partial"

MAIN_RESULTS_CSV   = f"{RESULTS_DIR}/main_results.csv"
ABLATION_CSV       = f"{RESULTS_DIR}/ablation_results.csv"
PREDICTIONS_CSV    = f"{RESULTS_DIR}/main_results_predictions.csv"

MODELS = list(FIT_FNS)

# (representation, model, candidate|None, role)
ALL_CONFIGS: list[tuple[str, str, str | None, str]] = [
    *[("signal",   m, None,          "main")      for m in MODELS],
    *[("symbolic", m, MAIN_CANDIDATE, "main")      for m in MODELS],
    *[("symbolic", m, c,              "ablation")
      for c in ALL_CANDIDATES[1:] for m in MODELS],
]

CV_FULL  = dict(n_outer_folds=5, n_repeats=3, n_inner_folds=3, n_hpo_iters=8, n_seeds=3)
CV_QUICK = dict(n_outer_folds=3, n_repeats=1, n_inner_folds=2, n_hpo_iters=2, n_seeds=1)


def _config_tag(representation: str, model: str, candidate: str | None) -> str:
    if representation == "signal":
        return f"signal_{model}"
    return f"symbolic_{candidate}_{model}"


# ── Per-config runner ─────────────────────────────────────────────────────────

def run_one(representation: str, model: str, candidate: str | None,
             cv_kw: dict | None = None) -> None:
    cv_kw = cv_kw or CV_FULL
    tag   = _config_tag(representation, model, candidate)
    t0    = time.time()

    print(f"[{tag}] extracting features...")
    if representation == "signal":
        precomputed = extract_signal()
    else:
        precomputed = extract_tfidf(candidate)

    print(f"[{tag}] nested CV  {cv_kw}")
    fold_records, pred_records = run_nested_cv(
        precomputed, representation, model, **cv_kw)

    _save_partial(tag, representation, candidate, fold_records, pred_records)

    mean_prob = [r for r in fold_records if r["aggregation"] == "mean_prob"]
    print(f"[{tag}] AUC(mean_prob)={np.mean([r['auc'] for r in mean_prob]):.3f}  "
          f"({time.time() - t0:.1f}s)")


# ── Partial CSV persistence ───────────────────────────────────────────────────

def _save_partial(tag: str, representation: str, candidate: str | None,
                   fold_records: list[dict], pred_records: list[dict]) -> None:
    os.makedirs(PARTIAL_DIR, exist_ok=True)

    df = pd.DataFrame(fold_records)
    df.insert(0, "config", tag)
    df.insert(1, "representation", representation)
    df.insert(2, "candidate", candidate or "")
    df.to_csv(f"{PARTIAL_DIR}/{tag}.csv", index=False)

    pf = pd.DataFrame(pred_records)
    pf.insert(0, "config", tag)
    pf.insert(1, "representation", representation)
    pf.insert(2, "candidate", candidate or "")
    pf.to_csv(f"{PARTIAL_DIR}/{tag}_predictions.csv", index=False)


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate() -> None:
    """Merge results/partial/*.csv -> main_results.csv, ablation_results.csv,
    main_results_predictions.csv."""
    main_folds, ablation_folds, main_preds = [], [], []

    for representation, model, candidate, role in ALL_CONFIGS:
        tag  = _config_tag(representation, model, candidate)
        path = f"{PARTIAL_DIR}/{tag}.csv"
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        df   = pd.read_csv(path)
        pf   = pd.read_csv(f"{PARTIAL_DIR}/{tag}_predictions.csv")

        if role == "main":
            main_folds.append(df[df["aggregation"] == "mean_prob"])
            main_preds.append(pf)
            ablation_folds.append(df[df["aggregation"] == "majority_vote"])
        else:
            ablation_folds.append(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    main = pd.concat(main_folds, ignore_index=True)
    main.to_csv(MAIN_RESULTS_CSV, index=False)
    print(f"wrote {MAIN_RESULTS_CSV}  ({len(main)} rows)")

    abl = pd.concat(ablation_folds, ignore_index=True)
    abl.to_csv(ABLATION_CSV, index=False)
    print(f"wrote {ABLATION_CSV}  ({len(abl)} rows)")

    preds = pd.concat(main_preds, ignore_index=True)
    preds.to_csv(PREDICTIONS_CSV, index=False)
    print(f"wrote {PREDICTIONS_CSV}  ({len(preds)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representation", choices=["signal", "symbolic"])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--candidate", choices=ALL_CANDIDATES,
                   help="Required when --representation symbolic")
    p.add_argument("--all", action="store_true",
                   help="Run all 16 configs sequentially then aggregate")
    p.add_argument("--aggregate", action="store_true",
                   help="Merge results/partial/ into final CSVs (no CV run)")
    p.add_argument("--quick", action="store_true",
                   help="Scaled-down CV for smoke testing")
    return p.parse_args()


def main():
    args = _parse_args()
    cv_kw = CV_QUICK if args.quick else CV_FULL

    if args.aggregate:
        aggregate()
        return

    if args.all:
        for representation, model, candidate, _ in ALL_CONFIGS:
            run_one(representation, model, candidate, cv_kw)
        print("\naggregating...")
        aggregate()
        return

    if args.representation is None or args.model is None:
        raise SystemExit("provide --representation + --model, or --all, or --aggregate")
    if args.representation == "symbolic" and args.candidate is None:
        raise SystemExit("--candidate required with --representation symbolic")
    if args.representation == "signal" and args.candidate is not None:
        raise SystemExit("--candidate is only used with --representation symbolic")

    run_one(args.representation, args.model, args.candidate, cv_kw)


if __name__ == "__main__":
    main()
