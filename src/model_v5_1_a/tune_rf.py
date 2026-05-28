"""
Random Forest hyperparameter grid search.

Evaluates every combination of max_depth, min_samples_leaf, max_features,
and min_samples_split over the last TUNE_LAST_N_FOLDS walk-forward test years.

RF trains in seconds so we can afford more folds (7) and a larger grid than
the FTT search. n_estimators is fixed at 500 — enough trees that variance is
negligible and adding more has diminishing returns.

Run:
    python -m src.model_v5_1_a.tune_rf
"""

import itertools
import os
import time

# RF tuning only — no PyTorch ops, so allow full CPU parallelism
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.model_v5_1_a.train_n_evaluate import (
    CAT_COLS,
    CFB_STAT_COLS,
    EXCLUDE_COLLEGE_STATS,
    LOG_PRED_CLIP,
    MODEL_NUM_COLS,
    RANDOM_STATE,
    WALK_FORWARD_START_YEAR,
    add_engineered_features,
    build_model_frame,
    build_tree_preprocessor,
    eval_metrics,
    set_seeds,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v5_1_a")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tuning config ─────────────────────────────────────────────────────────────

# More folds than FTT since RF is fast — covers recent years with larger
# training sets where RF performance is most representative.
TUNE_LAST_N_FOLDS = 7   # 2018–2024

RF_GRID = {
    "max_depth":        [None, 10, 15, 20],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     [0.3, 0.5, 0.7],
}

# 200 trees for tuning — enough to rank configs reliably, faster than 500
N_ESTIMATORS = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def all_configs(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def rf_predict(model, X_test) -> np.ndarray:
    raw = model.predict(X_test)
    return np.expm1(np.clip(raw, *LOG_PRED_CLIP))


def run_fold(cfg: dict, X_train, X_test, y_train) -> np.ndarray:
    pre, _, _ = build_tree_preprocessor(X_train)
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        **cfg,
    )
    model = Pipeline([("pre", pre), ("rf", rf)])
    model.fit(X_train, y_train)
    return rf_predict(model, X_test)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seeds(RANDOM_STATE)

    print("Loading data...", flush=True)
    df = build_model_frame()
    years = df["draft_season"].to_numpy(dtype=int)
    unique_years = sorted(np.unique(years))
    test_years = [yr for yr in unique_years if yr > WALK_FORWARD_START_YEAR]
    tune_years = test_years[-TUNE_LAST_N_FOLDS:]
    print(f"Tuning on folds: {tune_years}")

    print("Pre-computing fold data...", flush=True)
    folds = []
    for test_year in tune_years:
        train_mask = years < test_year
        test_mask  = years == test_year
        train_df = df.loc[train_mask].copy()
        test_df  = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            continue

        train_aug, test_aug, _ = add_engineered_features(train_df, test_df)

        y_train_orig = np.clip(train_aug["av_2yr"].to_numpy(dtype=float), 0, None)
        y_train = np.log1p(y_train_orig)
        y_test  = test_aug["av_2yr"].to_numpy(dtype=float)

        active_num_cols = MODEL_NUM_COLS
        if EXCLUDE_COLLEGE_STATS:
            excl = set(CFB_STAT_COLS) | {"college_perf_score", "cfb_matched"}
            active_num_cols = [c for c in MODEL_NUM_COLS if c not in excl]

        feat_cols = [c for c in active_num_cols + CAT_COLS if c in train_aug.columns]
        X_train = train_aug[feat_cols].copy()
        X_test  = test_aug[feat_cols].copy()

        folds.append({
            "year":    test_year,
            "X_train": X_train,
            "X_test":  X_test,
            "y_train": y_train,
            "y_test":  y_test,
        })

    print(f"Prepared {len(folds)} folds.", flush=True)

    configs = all_configs(RF_GRID)
    print(f"\nGrid: {len(configs)} configs × {len(folds)} folds = "
          f"{len(configs) * len(folds)} fits\n")

    records = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        fold_metrics = []
        for fold in folds:
            set_seeds(RANDOM_STATE)
            preds = run_fold(cfg, fold["X_train"], fold["X_test"], fold["y_train"])
            fold_metrics.append(eval_metrics(fold["y_test"], preds))

        avg = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]}

        # Also track most-recent-fold (2024) separately
        last = fold_metrics[-1]

        elapsed = time.time() - t0
        record = {
            **cfg,
            **{f"avg_{k}": round(v, 4) for k, v in avg.items()},
            **{f"last_{k}": round(v, 4) for k, v in last.items()},
            "elapsed_s": round(elapsed, 1),
        }
        records.append(record)

        depth_str = str(cfg["max_depth"]) if cfg["max_depth"] else "None"
        print(
            f"[{i+1:2d}/{len(configs)}] "
            f"depth={depth_str:4s} leaf={cfg['min_samples_leaf']:2d} feat={cfg['max_features']}  |  "
            f"avg MAE={avg['mae']:.4f}  last MAE={last['mae']:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

    results_df = pd.DataFrame(records).sort_values("avg_mae").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, "rf_grid_search_results.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nTop 10 configs by avg MAE (last {TUNE_LAST_N_FOLDS} folds):")
    print(results_df.head(10).to_string(index=False))
    print(f"\nTop 10 configs by last-fold MAE (2024):")
    print(results_df.sort_values("last_mae").head(10).to_string(index=False))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
