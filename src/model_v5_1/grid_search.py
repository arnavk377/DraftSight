"""
Hyperparameter grid search for model_v5_1.

Evaluates each model over GRID_SEARCH_YEARS walk-forward folds and reports
the best parameters per model. Neural net epochs are reduced for speed.

Run: python -m src.model_v5_1.grid_search
Outputs: poc_outputs_v5_1/grid_search/
"""

import itertools
import os
import random
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xgboost import XGBRegressor

from src.model_v5_1.ft_transformer import FTTransformer
from src.model_v5_1.mlp_embeddings import MLPWithEmbeddings
from src.model_v5_1.tabnet import TabNetRegressor
from src.model_v5_1.train_n_evaluate import (
    DEVICE, MODEL_CAT_COLS, MODEL_NUM_COLS, RANDOM_STATE,
    add_engineered_features, build_catboost_data, build_embed_data,
    build_model_frame, build_tree_preprocessor, eval_metrics,
    get_feature_names, to_tensor,
)

# ── Config ────────────────────────────────────────────────────────────────────

GRID_SEARCH_YEARS = [2009, 2012, 2015, 2018, 2021, 2024]

# Reduced epochs for grid search — use full train_n_evaluate settings for final runs
GS_MAX_EPOCHS   = 50
GS_PATIENCE     = 15
GS_VAL_FRAC     = 0.15
GS_BATCH_SIZE   = 128
GS_LR           = 1e-3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v5_1", "grid_search")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Parameter grids ───────────────────────────────────────────────────────────

GRIDS: dict[str, dict] = {
    "spline": {
        "alpha":        [1.0, 4.0, 10.0, 25.0],
        "n_knots_pick": [4, 6, 8],
    },
    "xgb": {
        "max_depth":        [3, 4, 5],
        "min_child_weight": [2, 5],
        "reg_lambda":       [1.0, 3.0, 5.0],
    },
    "catboost": {
        "depth":         [4, 5, 6],
        "l2_leaf_reg":   [1.0, 3.0, 6.0],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "rf": {
        "max_depth":       [None, 10, 20],
        "min_samples_leaf": [3, 5, 10],
        "max_features":    [0.3, 0.5, 0.7],
    },
    "mlpe": {
        "hidden_dims": [(64, 32), (128, 64), (256, 128, 64)],
        "dropout":     [0.3, 0.4, 0.5],
    },
    "ftt": {
        "d_token":  [16, 32],
        "n_layers": [2, 3],
        "dropout":  [0.1, 0.2, 0.3],
    },
    "tabnet": {
        "n_d":    [8, 16, 32],
        "n_steps": [3, 4],
        "gamma":  [1.3, 1.5, 1.8],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _set_seeds():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def _build_spline_preprocessor(df: pd.DataFrame, n_knots_pick: int):
    """Spline preprocessor with variable n_knots on pick."""
    from src.model_v5_1.train_n_evaluate import DRAFT_NUM_COLS
    spline_pool = DRAFT_NUM_COLS + ["college_perf_score", "cfb_matched"]
    num_cols = [c for c in spline_pool if c in df.columns and df[c].notna().any()]
    cat_cols = [c for c in MODEL_CAT_COLS if c in df.columns]
    spline_cols = [c for c in ["pick", "college_perf_score"] if c in num_cols]
    rest_num = [c for c in num_cols if c not in spline_cols]
    transformers = []
    n_knots_map = {"pick": n_knots_pick, "college_perf_score": 5}
    for col in spline_cols:
        transformers.append((f"{col}_spline", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("spl", SplineTransformer(n_knots=n_knots_map.get(col, 5), degree=3, include_bias=False)),
        ]), [col]))
    if rest_num:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), rest_num))
    if cat_cols:
        transformers.append(("cat", _ohe(), cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def _train_embed_gs(model, X_num_tr, X_cat_tr, y_tr):
    """Compact embed training loop using grid-search epoch settings."""
    n_val = max(1, int(len(y_tr) * GS_VAL_FRAC))
    Xn_t, Xn_v = X_num_tr[:-n_val], X_num_tr[-n_val:]
    Xc_t, Xc_v = X_cat_tr[:-n_val], X_cat_tr[-n_val:]
    yt, yv = y_tr[:-n_val], y_tr[-n_val:]
    loader = DataLoader(
        TensorDataset(to_tensor(Xn_t), torch.tensor(Xc_t, dtype=torch.long), to_tensor(yt)),
        batch_size=GS_BATCH_SIZE, shuffle=True,
    )
    Xn_v_t = to_tensor(Xn_v).to(DEVICE)
    Xc_v_t = torch.tensor(Xc_v, dtype=torch.long).to(DEVICE)
    yv_t   = to_tensor(yv).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=GS_LR, weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=GS_MAX_EPOCHS)
    best_val, best_state, pat = float("inf"), None, 0
    for _ in range(GS_MAX_EPOCHS):
        model.train()
        for Xb_n, Xb_c, yb in loader:
            Xb_n, Xb_c, yb = Xb_n.to(DEVICE), Xb_c.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.mse_loss(model(Xb_n, Xb_c), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xn_v_t, Xc_v_t), yv_t).item()
        if vl < best_val:
            best_val, pat = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= GS_PATIENCE:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _train_tabnet_gs(X_tr: np.ndarray, y_tr: np.ndarray, n_features: int, cfg: dict):
    """Compact TabNet training loop using grid-search epoch settings."""
    model = TabNetRegressor(n_features=n_features, **cfg).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = ExponentialLR(opt, gamma=0.96)
    n_val = max(1, int(len(X_tr) * GS_VAL_FRAC))
    Xt, Xv = X_tr[:-n_val], X_tr[-n_val:]
    yt, yv = y_tr[:-n_val], y_tr[-n_val:]
    batch_size = min(256, max(1, len(Xt)))
    loader = DataLoader(
        TensorDataset(to_tensor(Xt), to_tensor(yt)),
        batch_size=batch_size, shuffle=True,
    )
    Xv_t = to_tensor(Xv).to(DEVICE)
    yv_t = to_tensor(yv).to(DEVICE)
    best_val, best_state, pat, step = float("inf"), None, 0, 0
    for _ in range(GS_MAX_EPOCHS):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred, sp, _ = model(Xb)
            loss = model.loss(pred, yb, sp)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                sched.step()
        model.eval()
        with torch.no_grad():
            vp, vsp, _ = model(Xv_t)
            vl = float(model.loss(vp, yv_t, vsp).item())
        if vl < best_val:
            best_val, pat = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= GS_PATIENCE:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Per-model predict functions ───────────────────────────────────────────────

def _predict_spline(params, X_train, X_test, y_train):
    pre, _, _ = _build_spline_preprocessor(X_train, params["n_knots_pick"])
    model = Pipeline([("pre", pre), ("reg", Ridge(alpha=params["alpha"]))])
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _predict_xgb(params, X_train, X_test, y_train):
    pre, _, _ = build_tree_preprocessor(X_train)
    model = Pipeline([("pre", pre), ("xgb", XGBRegressor(
        n_estimators=400, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, nthread=1, random_state=RANDOM_STATE,
        **params,
    ))])
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _predict_catboost(params, X_train, X_test, y_train):
    X_tr_cb, X_te_cb, cat_idx, _ = build_catboost_data(X_train, X_test)
    model = CatBoostRegressor(
        iterations=500, random_seed=RANDOM_STATE, verbose=0,
        thread_count=1, cat_features=cat_idx, **params,
    )
    model.fit(X_tr_cb, y_train)
    return model.predict(X_te_cb)


def _predict_rf(params, X_train, X_test, y_train):
    pre, _, _ = build_tree_preprocessor(X_train)
    model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(
        n_estimators=300, n_jobs=1, random_state=RANDOM_STATE, **params,
    ))])
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _predict_mlpe(params, X_train, X_test, y_train):
    X_num_tr, X_num_te, X_cat_tr, X_cat_te, cards, _, _ = build_embed_data(X_train, X_test)
    model = MLPWithEmbeddings(
        X_num_tr.shape[1], cards,
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
    ).to(DEVICE)
    model = _train_embed_gs(model, X_num_tr, X_cat_tr, y_train)
    model.eval()
    with torch.no_grad():
        return model(
            to_tensor(X_num_te).to(DEVICE),
            torch.tensor(X_cat_te, dtype=torch.long).to(DEVICE),
        ).cpu().numpy()


def _predict_ftt(params, X_train, X_test, y_train):
    X_num_tr, X_num_te, X_cat_tr, X_cat_te, cards, _, _ = build_embed_data(X_train, X_test)
    model = FTTransformer(
        X_num_tr.shape[1], cards,
        d_token=params["d_token"],
        n_heads=4,
        n_layers=params["n_layers"],
        dropout=params["dropout"],
    ).to(DEVICE)
    model = _train_embed_gs(model, X_num_tr, X_cat_tr, y_train)
    model.eval()
    with torch.no_grad():
        return model(
            to_tensor(X_num_te).to(DEVICE),
            torch.tensor(X_cat_te, dtype=torch.long).to(DEVICE),
        ).cpu().numpy()


def _predict_tabnet(params, X_train, X_test, y_train):
    pre, t_num, t_cat = build_tree_preprocessor(X_train)
    X_tr_t = pre.fit_transform(X_train).astype(np.float32)
    X_te_t = pre.transform(X_test).astype(np.float32)
    np.nan_to_num(X_tr_t, copy=False)
    np.nan_to_num(X_te_t, copy=False)
    n_d = params["n_d"]
    cfg = dict(
        n_d=n_d, n_a=n_d, n_steps=params["n_steps"], gamma=params["gamma"],
        n_shared=2, n_step_dep=2, vbs=64, momentum=0.02, lambda_sparse=1e-3,
    )
    model = _train_tabnet_gs(X_tr_t, y_train, X_tr_t.shape[1], cfg)
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(to_tensor(X_te_t).to(DEVICE))
    return pred.cpu().numpy()


PREDICT_FNS = {
    "spline":  _predict_spline,
    "xgb":     _predict_xgb,
    "catboost": _predict_catboost,
    "rf":      _predict_rf,
    "mlpe":    _predict_mlpe,
    "ftt":     _predict_ftt,
    "tabnet":  _predict_tabnet,
}


# ── Grid search core ──────────────────────────────────────────────────────────

def grid_search_model(
    model_name: str,
    grid: dict,
    df: pd.DataFrame,
) -> pd.DataFrame:
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    years  = df["draft_season"].to_numpy(int)
    predict_fn = PREDICT_FNS[model_name]
    rows = []

    for combo in tqdm(combos, desc=f"  {model_name}", unit="combo", leave=True):
        params = dict(zip(keys, combo))
        year_metrics: list[dict] = []

        for year in GRID_SEARCH_YEARS:
            train_df = df[years < year].copy()
            test_df  = df[years == year].copy()
            if train_df.empty or test_df.empty:
                continue

            _set_seeds()
            train_aug, test_aug, _ = add_engineered_features(train_df, test_df)

            feat_cols = [c for c in MODEL_NUM_COLS + MODEL_CAT_COLS if c in train_aug.columns]
            X_train = train_aug[feat_cols].copy()
            X_test  = test_aug[feat_cols].copy()
            y_train = np.log1p(np.clip(train_aug["av_2yr"].to_numpy(float), 0, None))
            y_test  = test_aug["av_2yr"].to_numpy(float)  # original scale for eval

            try:
                AV_MAX = 60.0
                pred = np.expm1(np.clip(predict_fn(params, X_train, X_test, y_train), 0.0, np.log1p(AV_MAX)))
                year_metrics.append({**eval_metrics(y_test, pred), "year": year})
            except Exception as exc:
                tqdm.write(f"    [{model_name}] {params} year={year} failed: {exc}")
                continue

        if not year_metrics:
            continue

        row: dict = {k: str(v) for k, v in params.items()}
        for metric in ("mae", "rmse", "spearman", "r2"):
            vals = [m[metric] for m in year_metrics]
            row[f"mean_{metric}"] = round(float(np.mean(vals)), 4)
            row[f"std_{metric}"]  = round(float(np.std(vals)),  4)
        for m in year_metrics:
            row[f"mae_{m['year']}"] = round(m["mae"], 4)
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _param_cols(df: pd.DataFrame, grid: dict) -> list[str]:
    return [k for k in grid.keys() if k in df.columns]


def print_model_report(model_name: str, results: pd.DataFrame, grid: dict, top_k: int = 3):
    param_cols  = _param_cols(results, grid)
    metric_cols = ["mean_mae", "std_mae", "mean_rmse", "mean_spearman", "mean_r2"]
    display_cols = param_cols + [c for c in metric_cols if c in results.columns]

    width = 80
    print()
    print("═" * width)
    print(f"  {model_name.upper()}  —  Grid Search Results  (top {top_k} by MAE)")
    print("═" * width)

    top = results.head(top_k)[display_cols].copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    print(top.to_string(index=False))

    best = results.iloc[0]
    best_str = "  Best params:  " + "   ".join(
        f"{k}={best[k]}" for k in param_cols
    )
    print()
    print(best_str)
    mae_yr = [c for c in results.columns if c.startswith("mae_")]
    if mae_yr:
        yr_str = "  Per-year MAE: " + "   ".join(
            f"{c.replace('mae_', '')}={best[c]:.3f}" for c in mae_yr
        )
        print(yr_str)
    print("═" * width)


def build_best_params_summary(all_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, results in all_results.items():
        if results.empty:
            continue
        best = results.iloc[0].to_dict()
        row = {"model": model_name}
        # include all non-metric columns as param columns
        metric_prefixes = ("mean_", "std_", "mae_", "rmse_", "spearman_", "r2_")
        for k, v in best.items():
            if not any(k.startswith(p) for p in metric_prefixes):
                row[f"param_{k}"] = v
        for m in ("mean_mae", "mean_rmse", "mean_spearman", "mean_r2"):
            row[m] = best.get(m)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building model frame…")
    df = build_model_frame()
    df = df[df["draft_season"].isin(
        [y for y in df["draft_season"].unique() if y <= max(GRID_SEARCH_YEARS)]
    )].copy()

    model_names = list(GRIDS.keys())
    n_combos = {m: len(list(itertools.product(*GRIDS[m].values()))) for m in model_names}
    print(f"\nGrid search over {GRID_SEARCH_YEARS} | models: {model_names}")
    for m, n in n_combos.items():
        print(f"  {m:10s}: {n:3d} combos × {len(GRID_SEARCH_YEARS)} years = {n * len(GRID_SEARCH_YEARS)} fits")
    print()

    all_results: dict[str, pd.DataFrame] = {}

    for model_name in tqdm(model_names, desc="Models", unit="model"):
        results = grid_search_model(model_name, GRIDS[model_name], df)
        all_results[model_name] = results

        # Save per-model CSV
        csv_path = os.path.join(OUT_DIR, f"gs_{model_name}.csv")
        results.to_csv(csv_path, index=False)

        # Print report
        print_model_report(model_name, results, GRIDS[model_name])

    # Save best-params summary
    summary = build_best_params_summary(all_results)
    summary_path = os.path.join(OUT_DIR, "best_params_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Final summary table
    print()
    print("═" * 80)
    print("  BEST PARAMETERS SUMMARY")
    print("═" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print("═" * 80)
    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
