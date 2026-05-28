"""
FT-Transformer hyperparameter grid search.

Evaluates every (d_token, n_heads, n_layers, dropout, lr) combination over the
last TUNE_LAST_N_FOLDS walk-forward test years and reports average MAE,
RMSE, and Spearman across those folds.

Only the FTT is retrained for each config — data loading and feature
engineering run once up front.

Run:
    python -m src.model_v5_1_a.tune_ftt
"""

import itertools
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from src.model_v5_1_a.ft_transformer import FTTransformer
from src.model_v5_1_a.train_n_evaluate import (
    DEVICE,
    EMBED_TRAIN_CFG,
    EXCLUDE_COLLEGE_STATS,
    LOG_PRED_CLIP,
    RANDOM_STATE,
    WALK_FORWARD_START_YEAR,
    CFB_STAT_COLS,
    MODEL_NUM_COLS,
    CAT_COLS,
    add_engineered_features,
    build_embed_data,
    build_model_frame,
    eval_metrics,
    set_seeds,
    to_tensor,
    train_embed_model,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v5_1_a")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Tuning config ─────────────────────────────────────────────────────────────

# Only the most recent N folds are evaluated — recent folds are most
# representative of 2024 performance and cutting from 5→3 saves 40% runtime.
TUNE_LAST_N_FOLDS = 3

# Focused grid: n_heads fixed at 4 (less impactful than depth/width),
# dropout and lr at two values each, d_token and n_layers cover the key
# capacity dimensions. 16 configs × 3 folds = 48 training runs.
FTT_GRID = {
    "d_token":  [32, 64],   # 16 is the current default — try larger
    "n_heads":  [4],         # fixed; both 32 and 64 divide evenly by 4
    "n_layers": [2, 3],
    "dropout":  [0.1, 0.2],
    "lr":       [1e-4, 3e-4],
}

# Shorter epochs for tuning speed — early stopping still applies.
_BASE_TRAIN_CFG = dict(
    batch_size=EMBED_TRAIN_CFG["batch_size"],
    max_epochs=150,   # down from 200
    patience=20,      # down from 35
    val_fraction=EMBED_TRAIN_CFG["val_fraction"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def valid_configs(grid: dict) -> list[dict]:
    """Return all grid combinations where n_heads divides d_token."""
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        if cfg["d_token"] % cfg["n_heads"] == 0:
            combos.append(cfg)
    return combos


def ftt_predict(model, X_num_te: np.ndarray, X_cat_te: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(
            to_tensor(X_num_te).to(DEVICE),
            torch.tensor(X_cat_te, dtype=torch.long).to(DEVICE),
        ).cpu().numpy()
    return np.expm1(np.clip(preds, *LOG_PRED_CLIP))


def run_fold(cfg: dict, X_num_tr, X_cat_tr, y_train, X_num_te, X_cat_te, cardinalities):
    """Train one FTT config on a single fold and return predictions."""
    model = FTTransformer(
        n_numeric=X_num_tr.shape[1],
        cat_cardinalities=cardinalities,
        d_token=cfg["d_token"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    train_cfg = {**_BASE_TRAIN_CFG, "lr": cfg["lr"]}
    model = train_embed_model(
        model, X_num_tr, X_cat_tr, y_train,
        desc="FTT-tune", use_huber=True, **train_cfg,
    )
    preds = ftt_predict(model, X_num_te, X_cat_te)

    # Release MPS/CUDA memory immediately after each training run
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seeds(RANDOM_STATE)
    print(f"Device: {DEVICE}", flush=True)

    print("Loading data...", flush=True)
    df = build_model_frame()
    years = df["draft_season"].to_numpy(dtype=int)
    unique_years = sorted(np.unique(years))
    test_years = [yr for yr in unique_years if yr > WALK_FORWARD_START_YEAR]
    tune_years = test_years[-TUNE_LAST_N_FOLDS:]
    print(f"Tuning on folds: {tune_years}")

    # Pre-compute fold data (feature engineering) once for all configs
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
        active_cat_cols = CAT_COLS
        if EXCLUDE_COLLEGE_STATS:
            excl = set(CFB_STAT_COLS) | {"college_perf_score", "cfb_matched"}
            active_num_cols = [c for c in MODEL_NUM_COLS if c not in excl]

        X_train = train_aug[[c for c in active_num_cols + active_cat_cols if c in train_aug.columns]]
        X_test  = test_aug [[c for c in active_num_cols + active_cat_cols if c in test_aug.columns]]

        X_num_tr, X_num_te, X_cat_tr, X_cat_te, cardinalities, _, _ = build_embed_data(X_train, X_test)

        folds.append({
            "year":          test_year,
            "X_num_tr":      X_num_tr,
            "X_cat_tr":      X_cat_tr,
            "y_train":       y_train,
            "X_num_te":      X_num_te,
            "X_cat_te":      X_cat_te,
            "cardinalities": cardinalities,
            "y_test":        y_test,
        })
    print(f"Prepared {len(folds)} folds.", flush=True)

    configs = valid_configs(FTT_GRID)
    print(f"\nGrid: {len(configs)} valid configs × {len(folds)} folds = "
          f"{len(configs) * len(folds)} training runs\n")

    records = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        fold_metrics = []
        for fold in folds:
            set_seeds(RANDOM_STATE)
            preds = run_fold(
                cfg,
                fold["X_num_tr"], fold["X_cat_tr"], fold["y_train"],
                fold["X_num_te"], fold["X_cat_te"], fold["cardinalities"],
            )
            fold_metrics.append(eval_metrics(fold["y_test"], preds))

        avg = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0]}
        elapsed = time.time() - t0
        record = {**cfg, **{f"avg_{k}": v for k, v in avg.items()}, "elapsed_s": round(elapsed, 1)}
        records.append(record)

        print(
            f"[{i+1:3d}/{len(configs)}] "
            f"d={cfg['d_token']:2d} h={cfg['n_heads']} l={cfg['n_layers']} "
            f"do={cfg['dropout']} lr={cfg['lr']:.0e}  |  "
            f"MAE={avg['mae']:.4f}  RMSE={avg['rmse']:.4f}  "
            f"Sp={avg['spearman']:.4f}  ({elapsed:.0f}s)",
            flush=True,
        )

    results_df = pd.DataFrame(records).sort_values("avg_mae").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, "ftt_grid_search_results.csv")
    results_df.to_csv(out_path, index=False)

    print(f"\nTop 10 configs by avg MAE (last {TUNE_LAST_N_FOLDS} folds):")
    print(results_df.head(10).to_string(index=False))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
