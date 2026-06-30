
import json
import argparse
import numpy as np
import pandas as pd
import os
import warnings
import torch.nn as nn
from src.path import GRASP_ANNOT_DIR, DATASET_PATH, VIDEO_TS_DIR, OUTCOME_PATH
from typing import Tuple
import re
from datetime import datetime



def print_param_count(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total  |  {trainable:,} trainable")


def save_config(args: argparse.Namespace, ckpt_path: str) -> None:
    """Save CLI args as JSON next to the checkpoint."""
    config_path = os.path.splitext(ckpt_path)[0] + "_config.json"
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved → {config_path}")


def get_splits(
    dataset,
    split:          str   = "random",
    n_folds:        int   = 5,
    val_frac:       float = 0.15,
    train_frac:     float = 0.70,
    seed:           int   = 42,
    val_from_train: bool  = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """
    Return a list of (train_idx, val_idx, test_idx, fold_name) for each fold.

    split="random"     → 1 fold, window-level random split (existing behaviour)
    split="loio"       → N folds, one per unique infant (Leave-One-Infant-Out)
    split="groupkfold" → n_folds folds, grouped by infant
    """
    infants  = np.array(dataset.infant)
    all_idx  = np.arange(len(dataset))
    rng      = np.random.default_rng(seed)

    if split == "random":
        perm = rng.permutation(len(dataset))
        if n_folds == 1:
            n_val   = int(len(dataset) * val_frac)
            n_test  = len(dataset) - int(len(dataset) * train_frac) - n_val
            n_train = len(dataset) - n_val - n_test
            return [(perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:], "")]
        # n_folds > 1: window-level KFold
        # test = fold f; val = fold (f+1)%K when K>=3, else val_frac of non-test
        fold_sizes  = np.full(n_folds, len(dataset) // n_folds, dtype=int)
        fold_sizes[: len(dataset) % n_folds] += 1
        fold_ends   = np.cumsum(fold_sizes)
        fold_starts = np.concatenate([[0], fold_ends[:-1]])
        folds = []
        for f in range(n_folds):
            test_idx = perm[fold_starts[f] : fold_ends[f]]
            rest     = np.concatenate([
                perm[fold_starts[i] : fold_ends[i]]
                for i in range(n_folds) if i != f
            ])
            if n_folds >= 3:
                val_f     = (f + 1) % n_folds
                val_idx   = perm[fold_starts[val_f] : fold_ends[val_f]]
                train_idx = np.concatenate([
                    perm[fold_starts[i] : fold_ends[i]]
                    for i in range(n_folds) if i != f and i != val_f
                ])
            else:
                # n_folds == 2: take val_frac from the non-test half
                rest = rng.permutation(rest)
                n_val     = max(1, int(len(rest) * val_frac))
                val_idx   = rest[:n_val]
                train_idx = rest[n_val:]
            folds.append((train_idx, val_idx, test_idx, str(f)))
        return folds

    unique = sorted(set(infants))

    def _split_train_val(pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pool = rng.permutation(pool)
        n_val = int(len(pool) * val_frac)   # 0 when val_frac=0
        return pool[n_val:], pool[:n_val]

    if split == "loio":
        folds = []
        for infant in unique:
            test_idx           = all_idx[infants == infant]
            train_idx, val_idx = _split_train_val(all_idx[infants != infant])
            folds.append((train_idx, val_idx, test_idx, str(infant)))
        return folds

    # groupkfold
    perm_infants = list(rng.permutation(unique))
    fold_map     = {inf: i % n_folds for i, inf in enumerate(perm_infants)}
    folds = []
    for f in range(n_folds):
        test_mask  = np.array([fold_map[inf] == f for inf in infants])
        train_pool = all_idx[~test_mask]

        if val_from_train or n_folds < 3:
            # val drawn from training infants — stable, same distribution as train
            train_pool = rng.permutation(train_pool)
            n_val      = max(1, int(len(train_pool) * val_frac))
            val_idx    = train_pool[:n_val]
            train_idx  = train_pool[n_val:]
        else:
            # val is the adjacent infant fold — cross-infant (original behaviour)
            val_f      = (f + 1) % n_folds
            val_mask   = np.array([fold_map[inf] == val_f for inf in infants])
            train_mask = ~test_mask & ~val_mask
            val_idx    = all_idx[val_mask]
            train_idx  = all_idx[train_mask]

        folds.append((train_idx, val_idx, all_idx[test_mask], str(f)))
    return folds


def print_aggregate(all_metrics: list[dict]) -> None:
    """Print mean ± std across folds for each metric."""
    keys = ["frame_acc", "edit", "f1@10", "f1@25", "f1@50"]
    print("\n── Aggregate across folds ───────────────────────")
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"  {k:<12}  {np.mean(vals):.3f} ± {np.std(vals):.3f}")


def is_session_valid(
    accel: np.ndarray,        # (T, 3)
    fs: float,
    stuck_std_thresh: float = 1e-3,
    max_stuck_ratio: float = 0.5,
    min_activity_ratio: float = 0.001,
    activity_thresh: float = 0.1,
) -> tuple[bool, dict]:
    """
    Session-level QC. Returns (is_valid, stats_dict).
    """
    stats = {}

    # Stuck sensor
    stuck = np.any(
        pd.DataFrame(accel).rolling(int(fs)).std().fillna(1) < stuck_std_thresh,
        axis=1
    ).values
    stats['stuck_ratio'] = stuck.mean()

    # Low activity
    mag = np.linalg.norm(accel, axis=-1)
    active = np.abs(mag - 1) > activity_thresh

    stats['activity_ratio'] = active.mean()

    is_valid = (
        stats['stuck_ratio']    < max_stuck_ratio and
        stats['activity_ratio'] > min_activity_ratio
    )
    return is_valid, stats



def remove_overlaps_and_split(t_start, t_end, labels):
    import numpy as np

    events = []
    for s, e, l in zip(t_start, t_end, labels):
        events.append((s, 'start', l))
        events.append((e, 'end', l))

    # Sort events: by time, then end before start to close current before opening next
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))

    cleaned_t_start = []
    cleaned_t_end = []
    cleaned_label = []

    active_labels = []
    last_time = None

    for time, typ, label in events:
        if last_time is not None and time > last_time and active_labels:
            # Add a segment from last_time to current with top-most label (last added)
            cleaned_t_start.append(last_time)
            cleaned_t_end.append(time)
            cleaned_label.append(active_labels[-1])

        if typ == 'start':
            active_labels.append(label)
        elif typ == 'end':
            # Remove only one instance (in case of duplicates)
            for i in range(len(active_labels)-1, -1, -1):
                if active_labels[i] == label:
                    del active_labels[i]
                    break

        last_time = time

    return cleaned_t_start, cleaned_t_end, cleaned_label



def minsec_to_timestamp(t: str) -> int:
    return int(t*1000)



def get_abs_timestamps_labels(id_: str, session : str):
    #-> Tuple[Tuple[list], list, Tuple[int]]

    """
        Returns UNIX timestamp of annotations
        based on video's timestamps    

        id_ : Infant ID
        session : Session ID in format 'YYYYMMDD-HHMM'

    """



    #Check if infant+session has annotations
    if not os.path.isfile(os.path.join(VIDEO_TS_DIR,f'{id_}_sync-timestamp_{session}.ts')):
        warnings.warn(f"No video timestamps available for infant {id_} - session {session}")
        return None

    video_ts = open(os.path.join(VIDEO_TS_DIR,f'{id_}_sync-timestamp_{session}.ts'),'r')
    timestamp = video_ts.readlines()
    t0, t_last = float(timestamp[0]), float(timestamp[-1])
    video_ts.close()

    annotation = pd.read_csv(os.path.join(GRASP_ANNOT_DIR,f"{id_}_graspanalysis_{session}.csv"))
    

    t_start = []
    t_end = []
    labels = []

    for idx, row in annotation.iterrows():

        if row['Behavior type'] == "POINT":
            # Point event are transformed into State event by taking a window around the onset [onset - 500ms, onset + 500ms]
            t_start.append(minsec_to_timestamp(row['Time']-0.5)+t0)
            t_end.append(minsec_to_timestamp(row['Time']+0.5)+t0) 
            labels.append(row['Behavior'])
        
        elif row['Behavior type'] == "START":
            t_start.append(minsec_to_timestamp(annotation['Time'].iloc[idx])+t0)
            # Find the next row with same behavior and STOP
            stop_mask = (annotation['Behavior type'] == 'STOP') & (annotation['Behavior'] == row['Behavior']) & (annotation.index > idx)
            stop_indices = np.where(stop_mask)[0][0]
            t_end.append(minsec_to_timestamp(annotation['Time'].iloc[stop_indices])+t0) 
            labels.append(row['Behavior'])
        else:
            continue

    t_start, t_end, labels = remove_overlaps_and_split(t_start, t_end, labels)


    return (t_start, t_end), labels, (t0, t_last)


def get_infant_info(infant:str, session:str):
    full_outcome = pd.read_csv(OUTCOME_PATH)
    columns = [
        'ID', 'gender',
        'Gestational Age (weeks); no prematurity if ≥37',
        'date of assessment (T0) ', 'correct age at T0 (months)'
    ]

    infant_info = full_outcome.loc[full_outcome['ID']==infant]
    session = re.sub(r'(\d{2})(\d{2})(\d{2})(\d{2})-\d{4}', r'\4/\3/\2', session)

    age = abs(datetime.strptime(session, "%d/%m/%y") - datetime.strptime(infant_info['date of assessment (T0) '].values[0], "%d/%m/%y")).days/30.44 
    age += float(infant_info['correct age at T0 (months)'].values[0].replace(',', '.'))
    
    gestational_age = int(infant_info["Gestational Age (weeks); no prematurity if ≥37"].values[0].split("+")[0])

    info = [
        int(infant_info['gender'].values[0]=="f"),
        age, gestational_age
    ]

    return info







