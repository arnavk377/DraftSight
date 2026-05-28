"""
XGBoost hyperparameter grid search.

Tunes max_depth, learning_rate, min_child_weight, and reg_lambda over the
last TUNE_LAST_N_FOLDS walk-forward test years.

XGB uses Tweedie loss on original AV scale (same as production) with a 15%
held-out validation split for early stopping. n_estimators is set high (800)
and early stopping selects the optimal iteration — so lr and n_estimators are
jointly tuned without an explicit grid over n_estimators.

subsample and colsample_bytree are fixed at 0.85 (production default).
tweedie_variance_power is fixed at 1.5.

Run:
    python -m src.model_v6.tune_xgb
"""

import itertools
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.model_v6.train_n_evaluate import (
    AV_MAX,
    CAT_COLS,
    CFB_STAT_COLS,
    EXCLUDE_COLLEGE_STATS,
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
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v6")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tuning config ─────────────────────────────────────────────────────────────

TUNE_LAST_N_FOLDS = 7   # 2018–2024

XGB_GRID = {
    "max_depth":        [3, 4, 6],
    "learning_rate":    [0.02, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "reg_lambda":       [3.0, 5.0, 10.0],
}

N_ESTIMATORS = 800
VAL_FRACTION = 0.15
EARLY_STOPPING_ROUNDS = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def all_configs(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def run_fold(cfg: dict, X_train, X_test, y_train_orig: np.ndarray) -> tuple[np.ndarray, int]:
    pre, _, _ = build_tree_preprocessor(X_train)

    n_val = max(1, int(len(X_train) * VAL_FRACTION))
    X_tr, X_vl = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_vl = y_train_orig[:-n_val], y_train_orig[-n_val:]

    pre_fit = pre.fit(X_tr)

    xgb = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        subsample=0.85,
        colsample_bytree=0.85,
        nthread=1,
        random_state=RANDOM_STATE,
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
        **cfg,
    )
    xgb.fit(
        pre_fit.transform(X_tr), y_tr,
        eval_set=[(pre_fit.transform(X_vl), y_vl)],
        verbose=False,
    )
    preds = np.clip(xgb.predict(pre_fit.transform(X_test)), 0.0, AV_MAX)
    return preds, int(xgb.best_iteration)


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

    configs = all_configs(XGB_GRID)
    print(f"\nGrid: {len(configs)} configs × {len(folds)} folds = "
          f"{len(configs) * len(folds)} fits\n")

    records = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        fold_metrics = []
        best_iters   = []

        for fold in folds:
            set_seeds(RANDOM_STATE)
            preds, best_iter = run_fold(cfg, fold["X_train"], fold["X_test"], fold["y_train_orig"])
            fold_metrics.append(eval_metrics(fold["y_test"], preds))
            best_iters.append(best_iter)

        avg  = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]}
        last = fold_metrics[-1]
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
            f"depth={cfg['max_depth']} lr={cfg['learning_rate']:.2f} "
            f"mcw={cfg['min_child_weight']} λ={cfg['reg_lambda']:.1f}  |  "
            f"avg MAE={avg['mae']:.4f}  last MAE={last['mae']:.4f}  "
            f"avg_iter={np.mean(best_iters):.0f}  ({elapsed:.1f}s)",
            flush=True,
        )

    results_df = pd.DataFrame(records).sort_values("avg_mae").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, "xgb_grid_search_results.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nTop 10 configs by avg MAE (last {TUNE_LAST_N_FOLDS} folds):")
    print(results_df.head(10).to_string(index=False))
    print(f"\nTop 10 configs by last-fold MAE (2024):")
    print(results_df.sort_values("last_mae").head(10).to_string(index=False))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
