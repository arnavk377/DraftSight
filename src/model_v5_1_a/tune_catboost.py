"""
CatBoost hyperparameter grid search.

Tunes depth, learning_rate, and l2_leaf_reg over the last TUNE_LAST_N_FOLDS
walk-forward test years.

CatBoost uses Tweedie loss on original AV scale (same as production) with a 15%
held-out validation split for early stopping. iterations is set high (800) and
early stopping selects the optimal count — so lr and iterations are jointly
tuned without an explicit grid over iterations.

subsample and rsm are kept at CatBoost defaults (no explicit setting).
tweedie_variance_power is fixed at 1.5.

Run:
    python -m src.model_v5_1_a.tune_catboost
"""

import itertools
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.model_v5_1_a.train_n_evaluate import (
    AV_MAX,
    CAT_COLS,
    CFB_STAT_COLS,
    EXCLUDE_COLLEGE_STATS,
    MODEL_NUM_COLS,
    RANDOM_STATE,
    WALK_FORWARD_START_YEAR,
    add_engineered_features,
    build_catboost_data,
    build_model_frame,
    eval_metrics,
    set_seeds,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v5_1_a")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tuning config ─────────────────────────────────────────────────────────────

TUNE_LAST_N_FOLDS = 7   # 2018–2024: consistent with XGB and RF grids

CB_GRID = {
    "depth":         [4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "l2_leaf_reg":   [1.0, 3.0, 5.0, 10.0],
}

# High cap — early stopping selects the actual best iteration per fold
ITERATIONS       = 800
VAL_FRACTION     = 0.15
EARLY_STOP_ROUNDS = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def all_configs(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def run_fold(
    cfg: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_orig: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Train CatBoost on one fold and return (predictions, best_iteration)."""
    X_tr_cb, X_te_cb, cat_idx, _ = build_catboost_data(X_train, X_test)

    n_val = max(1, int(len(X_tr_cb) * VAL_FRACTION))
    X_t, X_v = X_tr_cb.iloc[:-n_val], X_tr_cb.iloc[-n_val:]
    y_t, y_v = y_train_orig[:-n_val], y_train_orig[-n_val:]

    model = CatBoostRegressor(
        iterations=ITERATIONS,
        loss_function="Tweedie:variance_power=1.5",
        random_seed=RANDOM_STATE,
        cat_features=cat_idx,
        thread_count=1,
        verbose=0,
        **cfg,
    )
    model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=EARLY_STOP_ROUNDS)

    preds = np.clip(model.predict(X_te_cb), 0.0, AV_MAX)
    return preds, int(model.best_iteration_)


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

        # CatBoost trains on original AV scale (Tweedie loss — no log transform)
        y_train_orig = np.clip(train_aug["av_2yr"].to_numpy(dtype=float), 0, None)
        y_test       = test_aug["av_2yr"].to_numpy(dtype=float)

        active_num_cols = MODEL_NUM_COLS
        if EXCLUDE_COLLEGE_STATS:
            excl = set(CFB_STAT_COLS) | {"college_perf_score", "cfb_matched"}
            active_num_cols = [c for c in MODEL_NUM_COLS if c not in excl]

        feat_cols = [c for c in active_num_cols + CAT_COLS if c in train_aug.columns]
        X_train = train_aug[feat_cols].copy()
        X_test  = test_aug[feat_cols].copy()

        folds.append({
            "year":         test_year,
            "X_train":      X_train,
            "X_test":       X_test,
            "y_train_orig": y_train_orig,
            "y_test":       y_test,
        })

    print(f"Prepared {len(folds)} folds.", flush=True)

    configs = all_configs(CB_GRID)
    print(f"\nGrid: {len(configs)} configs × {len(folds)} folds = "
          f"{len(configs) * len(folds)} fits\n")

    records = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        fold_metrics = []
        best_iters   = []

        for fold in folds:
            set_seeds(RANDOM_STATE)
            preds, best_iter = run_fold(
                cfg, fold["X_train"], fold["X_test"], fold["y_train_orig"]
            )
            fold_metrics.append(eval_metrics(fold["y_test"], preds))
            best_iters.append(best_iter)

        avg     = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]}
        last    = fold_metrics[-1]
        elapsed = time.time() - t0

        record = {
            **cfg,
            **{f"avg_{k}": round(v, 4) for k, v in avg.items()},
            **{f"last_{k}": round(v, 4) for k, v in last.items()},
            "avg_best_iter": round(float(np.mean(best_iters)), 0),
            "elapsed_s":     round(elapsed, 1),
        }
        records.append(record)

        print(
            f"[{i+1:3d}/{len(configs)}] "
            f"depth={cfg['depth']} lr={cfg['learning_rate']:.2f} "
            f"l2={cfg['l2_leaf_reg']:.1f}  |  "
            f"avg MAE={avg['mae']:.4f}  last MAE={last['mae']:.4f}  "
            f"avg_iter={np.mean(best_iters):.0f}  ({elapsed:.1f}s)",
            flush=True,
        )

    results_df = pd.DataFrame(records).sort_values("avg_mae").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, "catboost_grid_search_results.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nTop 10 configs by avg MAE (last {TUNE_LAST_N_FOLDS} folds):")
    print(results_df.head(10).to_string(index=False))
    print(f"\nTop 10 configs by last-fold MAE (2024):")
    print(results_df.sort_values("last_mae").head(10).to_string(index=False))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
