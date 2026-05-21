"""
NFL Draft Value Grid Search
===========================
Models: XGBoost, CatBoost, Spline Regression, MLP Neural Network
Target: 2-year Approximate Value (AV) from av.csv
Frame:  drafts.csv + college_stats.csv + draft_pick_context_features.csv

Run:
    pip install xgboost catboost scikit-learn pandas numpy scipy joblib matplotlib seaborn
    python modeling/grid_search.py
"""

import argparse
import ast
import json
import re
import warnings
from pathlib import Path
from sklearn.base import clone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import os
os.environ["PYTHONWARNINGS"] = "ignore"  # propagates to sklearn parallel workers

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "supabase_exports"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────

N_ITER = 60          # RandomizedSearchCV iterations per model
CV_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = -1          # use all cores

# post-draft outcome columns that must never be features
LEAKAGE_COLS = {
    "car_av", "w_av", "dr_av", "games", "allpro", "probowls",
    "seasons_started", "hof", "last_nfl_season",
    "pass_completions", "pass_attempts", "pass_yards", "pass_tds", "pass_ints",
    "rush_atts", "rush_yards", "rush_tds",
    "receptions", "rec_yards", "rec_tds",
    "def_solo_tackles", "def_ints", "def_sacks",
    "pick_approx_value_points",   # conservative: unclear if pre-draft
}

# identifier / free-text columns to drop entirely
DROP_IDS = {
    "draft_pick_id", "gsis_id", "pfr_player_id", "sports_ref_cfb_player_id",
    "pfr_player_name", "team", "team_raw", "franchise_id",
    "college_stats_id", "cfb_player_id",
    "pick_first_trade_date", "pick_last_trade_date",
    "pick_first_gave_team", "pick_first_received_team",
    "pick_last_gave_team", "pick_last_received_team",
    "pick_trade_team_chain",
    "roster_context_available", "roster_context_season",
    # draft_season: kept in frame for joins/walk-forward splits but is not a valid
    # feature — it's a proxy for AV era trends (AV accumulation has shifted over decades)
    "draft_season",
    # pick_post_year_trade_count: all zeros, zero variance, no signal
    "pick_post_year_trade_count",
}

# low-cardinality categoricals safe to one-hot encode
OHE_COLS = ["position", "position_group", "category", "side"]

# high-cardinality: college name — used by CatBoost natively, dropped for others
HIGH_CARD_CATS = {"college"}


# ── Data loading & frame construction ────────────────────────────────────────

def load_data():
    drafts  = pd.read_csv(DATA_DIR / "drafts.csv")
    av      = pd.read_csv(DATA_DIR / "av.csv")
    college = pd.read_csv(DATA_DIR / "college_stats.csv")
    context = pd.read_csv(DATA_DIR / "draft_pick_context_features.csv")
    return drafts, av, college, context


def build_target(drafts: pd.DataFrame, av: pd.DataFrame) -> pd.DataFrame:
    """Sum AV in draft_season and draft_season+1 per player → av_2yr."""
    base = drafts[["pfr_player_id", "draft_season"]].dropna(subset=["pfr_player_id"])
    merged = av[av["pfr_player_id"].notna()].merge(base, on="pfr_player_id", how="inner")
    two_yr = merged[
        (merged["season"] == merged["draft_season"]) |
        (merged["season"] == merged["draft_season"] + 1)
    ]
    return (
        two_yr.groupby("pfr_player_id", as_index=False)["av"]
        .sum()
        .rename(columns={"av": "av_2yr"})
    )


def build_frame(drafts, av, college, context) -> pd.DataFrame:
    # Remove post-draft leakage from drafts
    draft_cols = [c for c in drafts.columns if c not in LEAKAGE_COLS]
    frame = drafts[draft_cols].copy()

    # Attach 2-year AV target
    av_2yr = build_target(drafts, av)
    frame = frame.merge(av_2yr, on="pfr_player_id", how="left")
    frame["av_2yr"] = frame["av_2yr"].fillna(0)

    # College aggregate features (one row per player already)
    college_drop = {
        "college_stats_id", "cfb_player_id", "player", "college_team",
        "conference", "position", "position_group", "draft_round",
        "draft_pick_in_round",
    }
    college_keep = [c for c in college.columns if c not in college_drop]
    college_join = college[college_keep].rename(columns={"draft_overall": "pick"})
    frame = frame.merge(college_join, on=["draft_season", "pick"], how="left")

    # Draft-pick context features
    ctx_drop = {"team", "team_draft_code", "franchise_id", "position", "draft_position_group"}
    ctx_cols = [c for c in context.columns if c not in ctx_drop]
    frame = frame.merge(context[ctx_cols], on=["draft_season", "pick"], how="left")

    return frame


def prepare_features(frame: pd.DataFrame, use_college_name: bool = False):
    """Return X, y, numeric cols, categorical cols."""
    drop = DROP_IDS | LEAKAGE_COLS
    if not use_college_name:
        drop |= HIGH_CARD_CATS

    X = frame.drop(columns=["av_2yr"] + [c for c in drop if c in frame.columns])
    y = frame["av_2yr"].values

    # Coerce booleans to int
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    X[bool_cols] = X[bool_cols].astype(int)

    # Drop any remaining object columns not in OHE_COLS
    extra_obj = [
        c for c in X.select_dtypes(include="object").columns
        if c not in OHE_COLS and c not in HIGH_CARD_CATS
    ]
    X = X.drop(columns=extra_obj)

    cat_cols = [c for c in OHE_COLS if c in X.columns]
    if use_college_name:
        cat_cols += [c for c in HIGH_CARD_CATS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return X, y, num_cols, cat_cols


# ── Preprocessor factory ──────────────────────────────────────────────────────

def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


# ── Model definitions ─────────────────────────────────────────────────────────

def xgboost_search(X, y, cv):
    print("\n[XGBoost] Running RandomizedSearchCV ...")
    pipe = Pipeline([
        ("pre", make_preprocessor(
            [c for c in X.columns if c not in OHE_COLS],
            [c for c in OHE_COLS if c in X.columns],
        )),
        ("model", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
        )),
    ])
    param_dist = {
        "model__n_estimators":       randint(200, 1200),
        "model__max_depth":          randint(3, 10),
        "model__learning_rate":      loguniform(1e-3, 0.3),
        "model__subsample":          uniform(0.5, 0.5),
        "model__colsample_bytree":   uniform(0.4, 0.6),
        "model__min_child_weight":   randint(1, 10),
        "model__gamma":              uniform(0, 1),
        "model__reg_alpha":          loguniform(1e-4, 10),
        "model__reg_lambda":         loguniform(1e-4, 10),
    }
    return RandomizedSearchCV(
        pipe, param_dist, n_iter=N_ITER, cv=cv,
        scoring="neg_mean_squared_error", n_jobs=N_JOBS,
        random_state=RANDOM_STATE, verbose=1, refit=True,
    )


def catboost_search(X, y, cv, cat_cols):
    """CatBoost handles categoricals natively — no OHE needed."""
    print("\n[CatBoost] Running RandomizedSearchCV ...")
    cat_indices = [list(X.columns).index(c) for c in cat_cols if c in X.columns]

    param_dist = {
        "iterations":        randint(200, 1200),
        "depth":             randint(3, 10),
        "learning_rate":     loguniform(1e-3, 0.3),
        "l2_leaf_reg":       loguniform(1e-2, 20),
        "bagging_temperature": uniform(0, 2),
        "random_strength":   uniform(0, 3),
        "border_count":      randint(32, 256),
    }

    best_score = np.inf
    best_params = {}
    best_model = None

    rng = np.random.default_rng(RANDOM_STATE)
    samples = [
        {k: (v.rvs(random_state=int(rng.integers(1e6))) if hasattr(v, "rvs") else v)
         for k, v in param_dist.items()}
        for _ in range(N_ITER)
    ]

    # Manual CV because CatBoostRegressor isn't a sklearn estimator with set_params
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for i, params in enumerate(samples):
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            X_tr = X_tr.copy()
            X_val = X_val.copy()
            for c in cat_cols:
                if c in X_tr.columns:
                    X_tr[c] = X_tr[c].fillna("_missing_").astype(str)
                    X_val[c] = X_val[c].fillna("_missing_").astype(str)

            model = CatBoostRegressor(
                **params,
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
                cat_features=cat_indices,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))

        mean_mse = np.mean(scores)
        if mean_mse < best_score:
            best_score = mean_mse
            best_params = params
        if (i + 1) % 10 == 0:
            print(f"  iter {i+1}/{N_ITER}  best_rmse={np.sqrt(best_score):.4f}")

    # Refit on full data with best params
    X_full = X.copy()
    for c in cat_cols:
        if c in X_full.columns:
            X_full[c] = X_full[c].fillna("_missing_").astype(str)

    best_model = CatBoostRegressor(
        **best_params,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        cat_features=cat_indices,
    )
    best_model.fit(X_full, y)
    return best_model, best_params, best_score


def spline_search(X, y, cv, num_cols, cat_cols):
    print("\n[Spline Regression] Running RandomizedSearchCV ...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("spline", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer()),
                ("post_scaler", StandardScaler()),  # prevent overflow from high-degree basis
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", Ridge()),
    ])
    param_dist = {
        "pre__spline__spline__n_knots":  randint(3, 12),
        "pre__spline__spline__degree":   [2, 3, 4],
        "model__alpha":                  loguniform(1e-3, 1e4),
    }
    return RandomizedSearchCV(
        pipe, param_dist, n_iter=N_ITER, cv=cv,
        scoring="neg_mean_squared_error", n_jobs=N_JOBS,
        random_state=RANDOM_STATE, verbose=1, refit=True,
    )


def mlp_search(X, y, cv, num_cols, cat_cols):
    print("\n[MLP] Running RandomizedSearchCV ...")
    pipe = Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("model", MLPRegressor(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ])
    param_dist = {
        "model__hidden_layer_sizes": [
            (128,), (256,), (512,),
            (128, 64), (256, 128), (512, 256),
            (256, 128, 64), (512, 256, 128),
        ],
        "model__activation":        ["relu", "tanh"],
        "model__alpha":             loguniform(1e-5, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 1e-2),
        "model__batch_size":        [64, 128, 256, 512],
    }
    return RandomizedSearchCV(
        pipe, param_dist, n_iter=N_ITER, cv=cv,
        scoring="neg_mean_squared_error", n_jobs=N_JOBS,
        random_state=RANDOM_STATE, verbose=1, refit=True,
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_best(search, X, y, name):
    preds = search.best_estimator_.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)
    cv_rmse = np.sqrt(-search.best_score_)
    print(f"\n{name} results:")
    print(f"  CV RMSE (best):  {cv_rmse:.4f}")
    print(f"  Train RMSE:      {rmse:.4f}")
    print(f"  Train R²:        {r2:.4f}")
    print(f"  Best params:     {search.best_params_}")
    return {"cv_rmse": cv_rmse, "train_rmse": rmse, "train_r2": r2,
            "best_params": str(search.best_params_), "preds": preds}


def eval_catboost(model, best_params, best_mse, X, y, name="CatBoost"):
    X = X.copy()
    for i in model.get_cat_feature_indices():
        col = X.columns[i]
        if X[col].isna().any():
            X[col] = X[col].fillna("_missing_").astype(str)
    preds = model.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    r2    = r2_score(y, preds)
    cv_rmse = np.sqrt(best_mse)
    print(f"\n{name} results:")
    print(f"  CV RMSE (best):  {cv_rmse:.4f}")
    print(f"  Train RMSE:      {rmse:.4f}")
    print(f"  Train R²:        {r2:.4f}")
    print(f"  Best params:     {best_params}")
    return {"cv_rmse": cv_rmse, "train_rmse": rmse, "train_r2": r2,
            "best_params": str(best_params), "preds": preds}


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def _prep_sk_walkforward(train_frame, test_frame):
    """Aligned train/test matrices for sklearn models (no college name)."""
    X_tr, y_tr, num_tr, cat_tr = prepare_features(train_frame, use_college_name=False)
    X_te, y_te, _,      _      = prepare_features(test_frame,  use_college_name=False)

    X_te = X_te.reindex(columns=X_tr.columns)
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)

    for c in num_tr:
        med = X_tr[c].median()
        if pd.isna(med):
            med = 0.0
        X_tr[c] = X_tr[c].fillna(med)
        X_te[c] = X_te[c].fillna(med)

    for c in cat_tr:
        X_tr[c] = X_tr[c].fillna("_missing_").astype(str)
        X_te[c] = X_te[c].fillna("_missing_").astype(str)

    return X_tr, y_tr, X_te, y_te


def _prep_cb_walkforward(train_frame, test_frame):
    """Aligned train/test matrices for CatBoost (includes college name)."""
    X_tr, y_tr, num_tr, cat_tr = prepare_features(train_frame, use_college_name=True)
    X_te, y_te, _,      _      = prepare_features(test_frame,  use_college_name=True)

    X_te = X_te.reindex(columns=X_tr.columns)
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)

    for c in num_tr:
        med = X_tr[c].median()
        if pd.isna(med):
            med = 0.0
        X_tr[c] = X_tr[c].fillna(med)
        X_te[c] = X_te[c].fillna(med)

    for c in cat_tr:
        X_tr[c] = X_tr[c].fillna("_missing_").astype(str)
        X_te[c] = X_te[c].fillna("_missing_").astype(str)

    return X_tr, y_tr, X_te, y_te, cat_tr


def walk_forward_backtest(frame, sk_estimators, cb_params=None, min_train_years=2):
    """
    Expanding-window walk-forward backtest for all available models.

    sk_estimators: dict {name: fitted_sklearn_estimator} — cloned fresh each fold.
    cb_params:     CatBoost best param dict, or None to skip CatBoost.

    For each draft year T:
      - Train on all classes < T, predict class T
      - Last 2 years flagged as incomplete (av_2yr needs 2 seasons to materialise)
    """
    years    = sorted(frame["draft_season"].dropna().unique().astype(int))
    max_year = max(years)

    model_names = list(sk_estimators.keys()) + (["CatBoost"] if cb_params else [])
    all_rows    = {name: [] for name in model_names}

    print("\n[Walk-Forward Backtest] Expanding window, all models ...")
    print(f"  Models included: {', '.join(model_names)}")
    print(f"  Years: {min(years)} → {max(years)}  |  min_train_years={min_train_years}")
    print(f"  Last 2 years ({max(years)-1}, {max(years)}) flagged as incomplete outcomes")
    print()
    for t in years:
        train_years = [y for y in years if y < t]
        if len(train_years) < min_train_years:
            continue

        train_frame = frame[frame["draft_season"].isin(train_years)]
        test_frame  = frame[frame["draft_season"] == t]
        incomplete  = t > max_year - 2

        # ── sklearn models (XGBoost, Spline, MLP) ────────────────────────────
        X_tr_sk, y_tr_sk, X_te_sk, y_te_sk = _prep_sk_walkforward(train_frame, test_frame)

        if y_tr_sk.std() == 0:
            print(f"  {t}  skipped — training targets all equal")
            continue

        for name, est in sk_estimators.items():
            print(f"    → {name}: training on {len(y_tr_sk)} samples, predicting {len(y_te_sk)} ...")
            model = clone(est)
            model.fit(X_tr_sk, y_tr_sk)
            preds = model.predict(X_te_sk)
            all_rows[name].append({
                "draft_year":         t,
                "n_train":            len(y_tr_sk),
                "n_test":             len(y_te_sk),
                "rmse":               np.sqrt(mean_squared_error(y_te_sk, preds)),
                "r2":                 r2_score(y_te_sk, preds),
                "incomplete_outcome": incomplete,
                "preds":              preds,
                "actuals":            y_te_sk,
            })

        # ── CatBoost ──────────────────────────────────────────────────────────
        if cb_params:
            print(f"    → CatBoost: training on {len(y_tr_sk)} samples, predicting {len(y_te_sk)} ...")
            X_tr_cb, y_tr_cb, X_te_cb, y_te_cb, cat_cols = _prep_cb_walkforward(
                train_frame, test_frame
            )
            cat_indices = [list(X_tr_cb.columns).index(c) for c in cat_cols
                           if c in X_tr_cb.columns]
            cb_model = CatBoostRegressor(
                **cb_params, loss_function="RMSE",
                random_seed=RANDOM_STATE, verbose=False, cat_features=cat_indices,
            )
            cb_model.fit(X_tr_cb, y_tr_cb)
            for i in cb_model.get_cat_feature_indices():
                col = X_te_cb.columns[i]
                if X_te_cb[col].isna().any():
                    X_te_cb[col] = X_te_cb[col].fillna("_missing_").astype(str)
            preds = cb_model.predict(X_te_cb)
            all_rows["CatBoost"].append({
                "draft_year":         t,
                "n_train":            len(y_tr_cb),
                "n_test":             len(y_te_cb),
                "rmse":               np.sqrt(mean_squared_error(y_te_cb, preds)),
                "r2":                 r2_score(y_te_cb, preds),
                "incomplete_outcome": incomplete,
                "preds":              preds,
                "actuals":            y_te_cb,
            })

        flag = "  *** incomplete 2yr outcome ***" if incomplete else ""
        print(f"  [{t}] train={len(y_tr_sk):4d}  test={len(y_te_sk):3d}{flag}")
        for name in model_names:
            if all_rows[name] and all_rows[name][-1]["draft_year"] == t:
                r = all_rows[name][-1]
                print(f"    {name:<22} RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}")
        print()

    return all_rows


def generate_backtest_plots(all_rows):
    """
    Figures 7a–7c: walk-forward results for all models.
      7a — RMSE by draft year (all models overlaid)
      7b — R² by draft year (all models overlaid)
      7c_<model> — predicted vs actual scatter per model
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

    # ── Fig 7a: RMSE by year, all models ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, rows in all_rows.items():
        color  = COLORS.get(name, "#607D8B")
        valid  = [r for r in rows if not r["incomplete_outcome"]]
        inc    = [r for r in rows if r["incomplete_outcome"]]
        if not valid:
            continue
        ax.plot([r["draft_year"] for r in valid],
                [r["rmse"]       for r in valid],
                marker="o", linewidth=2, color=color, label=name)
        if inc:
            ax.plot([r["draft_year"] for r in inc],
                    [r["rmse"]       for r in inc],
                    marker="o", linewidth=2, linestyle="--", color=color, alpha=0.5)
    ax.set_xlabel("Draft Year")
    ax.set_ylabel("RMSE (AV units)")
    ax.set_title("Figure 7a: Walk-Forward Backtest — RMSE by Draft Year\n"
                 "(dashed = incomplete 2yr outcome)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7a_walkforward_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig7a_walkforward_rmse.png")

    # ── Fig 7b: R² by year, all models ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, rows in all_rows.items():
        color = COLORS.get(name, "#607D8B")
        valid = [r for r in rows if not r["incomplete_outcome"]]
        inc   = [r for r in rows if r["incomplete_outcome"]]
        if not valid:
            continue
        ax.plot([r["draft_year"] for r in valid],
                [r["r2"]         for r in valid],
                marker="o", linewidth=2, color=color, label=name)
        if inc:
            ax.plot([r["draft_year"] for r in inc],
                    [r["r2"]         for r in inc],
                    marker="o", linewidth=2, linestyle="--", color=color, alpha=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xlabel("Draft Year")
    ax.set_ylabel("R²")
    ax.set_title("Figure 7b: Walk-Forward Backtest — R² by Draft Year",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7b_walkforward_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig7b_walkforward_r2.png")

    # ── Fig 7c: per-model predicted vs actual scatter ─────────────────────────
    summary = {}
    for name, rows in all_rows.items():
        valid = [r for r in rows if not r["incomplete_outcome"]]
        if not valid:
            continue
        all_preds   = np.concatenate([r["preds"]   for r in valid])
        all_actuals = np.concatenate([r["actuals"] for r in valid])
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        overall_r2   = r2_score(all_actuals, all_preds)
        summary[name] = (overall_rmse, overall_r2)

        color = COLORS.get(name, "#607D8B")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(all_actuals, all_preds, alpha=0.25, s=12,
                   color=color, edgecolors="none")
        lim = max(float(all_actuals.max()), float(all_preds.max())) * 1.08
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="y = x  (perfect)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Actual 2-Year AV")
        ax.set_ylabel("Predicted 2-Year AV")
        ax.set_title(
            f"Figure 7c: {name} — Walk-Forward Predicted vs. Actual\n"
            f"(All validated draft classes pooled)\n"
            f"RMSE = {overall_rmse:.3f}   R² = {overall_r2:.3f}",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        slug  = name.lower().replace(" ", "_")
        fname = f"fig7c_walkforward_scatter_{slug}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "XGBoost":          "#2196F3",
    "CatBoost":         "#4CAF50",
    "SplineRegression": "#FF9800",
    "MLP":              "#9C27B0",
}


def generate_plots(results, y, xgb_est=None, cb_model=None):
    """Save all evaluation figures to OUT_DIR."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

    # ── Fig 1: Predicted vs Actual (one image per model) ─────────────────────
    for i, (name, r) in enumerate(results.items(), start=1):
        preds = r["preds"]
        color = COLORS.get(name, "#607D8B")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(y, preds, alpha=0.35, s=18, color=color, edgecolors="none")
        lim = max(float(y.max()), float(preds.max())) * 1.08
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="y = x  (perfect)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Actual 2-Year AV")
        ax.set_ylabel("Predicted 2-Year AV")
        ax.set_title(
            f"Figure 1{chr(96 + i)}: {name} — Predicted vs. Actual 2-Year AV\n"
            f"CV RMSE = {r['cv_rmse']:.3f}   Train R² = {r['train_r2']:.3f}\n"
            "(Closer to y = x line indicates better predictive accuracy)",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        slug = name.lower().replace(" ", "_")
        fname = f"fig1{chr(96 + i)}_predicted_vs_actual_{slug}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ── Fig 2: Model Comparison bar chart ────────────────────────────────────
    names    = list(results.keys())
    cv_rmses = [results[n]["cv_rmse"]  for n in names]
    r2s      = [results[n]["train_r2"] for n in names]
    colors   = [COLORS.get(n, "#607D8B") for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 2: Model Performance Comparison", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(names, cv_rmses, color=colors, edgecolor="white", width=0.55)
    ax1.bar_label(bars1, fmt="%.3f", padding=4)
    ax1.set_title("Cross-Validated RMSE  (lower is better)")
    ax1.set_ylabel("CV RMSE")
    ax1.set_ylim(0, max(cv_rmses) * 1.2)

    bars2 = ax2.bar(names, r2s, color=colors, edgecolor="white", width=0.55)
    ax2.bar_label(bars2, fmt="%.3f", padding=4)
    ax2.set_title("Train R²  (higher is better)")
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, min(max(r2s) * 1.2, 1.15))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig2_model_comparison.png")

    # ── Fig 3: Residual Distributions (2×2) ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Figure 3: Residual Distributions  (Residual = Actual − Predicted)\n"
        "(Ideal: symmetric around 0)",
        fontsize=13, fontweight="bold",
    )
    for ax, (name, r) in zip(axes.flat, results.items()):
        resid = y - r["preds"]
        color = COLORS.get(name, "#607D8B")
        sns.histplot(resid, kde=True, ax=ax, color=color, bins=40, alpha=0.7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name}  (mean={resid.mean():.2f}, std={resid.std():.2f})")
        ax.set_xlabel("Residual (AV units)")
        ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_residual_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig3_residual_distributions.png")

    # ── Fig 4: XGBoost Feature Importance ────────────────────────────────────
    if xgb_est is not None:
        try:
            pre         = xgb_est.named_steps["pre"]
            feat_names  = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
            importances = xgb_est.named_steps["model"].feature_importances_
            top_n = min(20, len(feat_names))
            idx   = np.argsort(importances)[-top_n:]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh([feat_names[i] for i in idx], importances[idx], color=COLORS["XGBoost"])
            ax.set_title(f"Figure 4: XGBoost — Top {top_n} Feature Importances",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance (F-score gain)")
            plt.tight_layout()
            fig.savefig(OUT_DIR / "fig4_xgboost_importance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved: fig4_xgboost_importance.png")
        except Exception as e:
            print(f"  [warn] XGBoost importance plot skipped: {e}")

    # ── Fig 5: CatBoost Feature Importance ───────────────────────────────────
    if cb_model is not None:
        try:
            feat_names  = list(cb_model.feature_names_)
            importances = cb_model.get_feature_importance()
            top_n = min(20, len(feat_names))
            idx   = np.argsort(importances)[-top_n:]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh([feat_names[i] for i in idx], importances[idx], color=COLORS["CatBoost"])
            ax.set_title(f"Figure 5: CatBoost — Top {top_n} Feature Importances",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fig.savefig(OUT_DIR / "fig5_catboost_importance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved: fig5_catboost_importance.png")
        except Exception as e:
            print(f"  [warn] CatBoost importance plot skipped: {e}")

    # ── Fig 6: Target Distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(y, kde=True, ax=ax, color="#607D8B", bins=50)
    ax.set_title("Figure 6: Distribution of 2-Year AV (Target Variable)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("2-Year Approximate Value (AV)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig6_target_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NFL Draft Value Grid Search")
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip XGBoost and CatBoost — run only Spline Regression and MLP",
    )
    parser.add_argument(
        "--refit", action="store_true",
        help="Skip grid search: load saved best params, re-fit all models once, run walk-forward",
    )
    args = parser.parse_args()

    print("Loading data ...")
    drafts, av, college, context = load_data()
    print(f"  drafts:  {drafts.shape}  |  av: {av.shape}  |  "
          f"college: {college.shape}  |  context: {context.shape}")

    print("\nBuilding model frame ...")
    frame = build_frame(drafts, av, college, context)
    print(f"  frame shape: {frame.shape}  |  target mean av_2yr: {frame['av_2yr'].mean():.2f}")

    # ── Standard frame (no high-cardinality college name) ──
    X, y, num_cols, cat_cols = prepare_features(frame, use_college_name=False)
    print(f"\nFeature matrix: {X.shape[1]} features, {X.shape[0]} rows")
    print(f"  Numeric: {len(num_cols)}  |  Categorical: {len(cat_cols)} → {cat_cols}")

    # Fill NaNs before passing to sklearn models (pipeline also imputes numerics)
    X_sk = X.copy()
    X_sk = X_sk.replace([np.inf, -np.inf], np.nan)
    for c in num_cols:
        if X_sk[c].isna().any():
            med = X_sk[c].median()
            X_sk[c] = X_sk[c].fillna(med if not pd.isna(med) else 0.0)
    for c in cat_cols:
        X_sk[c] = X_sk[c].fillna("_missing_").astype(str)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results  = {}
    xgb_est  = None
    spl_est  = None
    mlp_est  = None
    cb_model = None
    cb_params = None

    def _parse_params(raw):
        return ast.literal_eval(re.sub(r"np\.\w+\(([^)]+)\)", r"\1", raw))

    if args.refit:
        # ── Re-fit all models with saved hyperparams (no CV) ─────────────────
        saved_json_path = OUT_DIR / "grid_search_results.json"
        if not saved_json_path.exists():
            raise FileNotFoundError(
                "No saved results found — run without --refit first to run the grid search."
            )
        with open(saved_json_path) as f:
            saved = json.load(f)
        print("\n[--refit] Loading saved hyperparams and re-fitting with updated features ...")

        if "XGBoost" in saved:
            print("  [XGBoost] re-fitting ...")
            xgb_params = _parse_params(saved["XGBoost"]["best_params"])
            xgb_est = Pipeline([
                ("pre", make_preprocessor(
                    [c for c in X_sk.columns if c not in OHE_COLS],
                    [c for c in OHE_COLS if c in X_sk.columns],
                )),
                ("model", xgb.XGBRegressor(
                    objective="reg:squarederror", random_state=RANDOM_STATE,
                    verbosity=0, n_jobs=1,
                )),
            ])
            xgb_est.set_params(**xgb_params)
            xgb_est.fit(X_sk, y)
            preds = xgb_est.predict(X_sk)
            results["XGBoost"] = {
                "cv_rmse":    saved["XGBoost"]["cv_rmse"],
                "train_rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "train_r2":   float(r2_score(y, preds)),
                "best_params": saved["XGBoost"]["best_params"],
                "preds": preds,
            }
            joblib.dump(xgb_est, OUT_DIR / "xgboost_best.pkl")

        if "CatBoost" in saved:
            print("  [CatBoost] re-fitting ...")
            cb_params = _parse_params(saved["CatBoost"]["best_params"])
            X_cb, y_cb, num_cols_cb, cat_cols_cb = prepare_features(frame, use_college_name=True)
            X_cb = X_cb.replace([np.inf, -np.inf], np.nan)
            for c in num_cols_cb:
                med = X_cb[c].median()
                X_cb[c] = X_cb[c].fillna(med if not pd.isna(med) else 0.0)
            for c in cat_cols_cb:
                X_cb[c] = X_cb[c].fillna("_missing_").astype(str)
            cat_indices = [list(X_cb.columns).index(c) for c in cat_cols_cb if c in X_cb.columns]
            cb_model = CatBoostRegressor(
                **cb_params, loss_function="RMSE", random_seed=RANDOM_STATE,
                verbose=False, cat_features=cat_indices,
            )
            cb_model.fit(X_cb, y_cb)
            cb_mse = saved["CatBoost"]["cv_rmse"] ** 2
            results["CatBoost"] = eval_catboost(cb_model, cb_params, cb_mse, X_cb, y_cb)
            cb_model.save_model(str(OUT_DIR / "catboost_best.cbm"))

        if "SplineRegression" in saved:
            print("  [SplineRegression] re-fitting ...")
            spl_params = _parse_params(saved["SplineRegression"]["best_params"])
            spl_pre = ColumnTransformer(
                transformers=[
                    ("spline", Pipeline([
                        ("imputer",     SimpleImputer(strategy="median")),
                        ("scaler",      StandardScaler()),
                        ("spline",      SplineTransformer()),
                        ("post_scaler", StandardScaler()),
                    ]), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ],
                remainder="drop",
            )
            spl_est = Pipeline([("pre", spl_pre), ("model", Ridge())])
            spl_est.set_params(**spl_params)
            spl_est.fit(X_sk, y)
            preds = spl_est.predict(X_sk)
            results["SplineRegression"] = {
                "cv_rmse":    saved["SplineRegression"]["cv_rmse"],
                "train_rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "train_r2":   float(r2_score(y, preds)),
                "best_params": saved["SplineRegression"]["best_params"],
                "preds": preds,
            }
            joblib.dump(spl_est, OUT_DIR / "spline_best.pkl")

        if "MLP" in saved:
            print("  [MLP] re-fitting ...")
            mlp_params = _parse_params(saved["MLP"]["best_params"])
            mlp_est = Pipeline([
                ("pre", make_preprocessor(num_cols, cat_cols)),
                ("model", MLPRegressor(
                    max_iter=500, early_stopping=True,
                    validation_fraction=0.1, random_state=RANDOM_STATE,
                )),
            ])
            mlp_est.set_params(**mlp_params)
            mlp_est.fit(X_sk, y)
            preds = mlp_est.predict(X_sk)
            results["MLP"] = {
                "cv_rmse":    saved["MLP"]["cv_rmse"],
                "train_rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "train_r2":   float(r2_score(y, preds)),
                "best_params": saved["MLP"]["best_params"],
                "preds": preds,
            }
            joblib.dump(mlp_est, OUT_DIR / "mlp_best.pkl")

    elif args.fast:
        print("\n[--fast] Skipping XGBoost and CatBoost.")
        saved_json_path = OUT_DIR / "grid_search_results.json"
        if saved_json_path.exists():
            with open(saved_json_path) as f:
                saved = json.load(f)
            if "CatBoost" in saved:
                cb_params = _parse_params(saved["CatBoost"]["best_params"])
                print("  Loaded saved CatBoost params for walk-forward backtest.")

        spl_search = spline_search(X_sk, y, cv, num_cols, cat_cols)
        spl_search.fit(X_sk, y)
        results["SplineRegression"] = eval_best(spl_search, X_sk, y, "Spline Regression")
        spl_est = spl_search.best_estimator_
        joblib.dump(spl_est, OUT_DIR / "spline_best.pkl")

        mlp_srch = mlp_search(X_sk, y, cv, num_cols, cat_cols)
        mlp_srch.fit(X_sk, y)
        results["MLP"] = eval_best(mlp_srch, X_sk, y, "MLP")
        mlp_est = mlp_srch.best_estimator_
        joblib.dump(mlp_est, OUT_DIR / "mlp_best.pkl")

    else:
        # ── Full grid search ──────────────────────────────────────────────────
        xgb_search = xgboost_search(X_sk, y, cv)
        xgb_search.fit(X_sk, y)
        results["XGBoost"] = eval_best(xgb_search, X_sk, y, "XGBoost")
        xgb_est = xgb_search.best_estimator_
        joblib.dump(xgb_est, OUT_DIR / "xgboost_best.pkl")

        X_cb, y_cb, num_cols_cb, cat_cols_cb = prepare_features(frame, use_college_name=True)
        X_cb = X_cb.replace([np.inf, -np.inf], np.nan)
        for c in num_cols_cb:
            if X_cb[c].isna().any():
                med = X_cb[c].median()
                X_cb[c] = X_cb[c].fillna(med if not pd.isna(med) else 0.0)
        for c in cat_cols_cb:
            X_cb[c] = X_cb[c].fillna("_missing_").astype(str)
        cb_model, cb_params, cb_mse = catboost_search(X_cb, y_cb, cv, cat_cols_cb)
        results["CatBoost"] = eval_catboost(cb_model, cb_params, cb_mse, X_cb, y_cb)
        cb_model.save_model(str(OUT_DIR / "catboost_best.cbm"))

        spl_search = spline_search(X_sk, y, cv, num_cols, cat_cols)
        spl_search.fit(X_sk, y)
        results["SplineRegression"] = eval_best(spl_search, X_sk, y, "Spline Regression")
        spl_est = spl_search.best_estimator_
        joblib.dump(spl_est, OUT_DIR / "spline_best.pkl")

        mlp_srch = mlp_search(X_sk, y, cv, num_cols, cat_cols)
        mlp_srch.fit(X_sk, y)
        results["MLP"] = eval_best(mlp_srch, X_sk, y, "MLP")
        mlp_est = mlp_srch.best_estimator_
        joblib.dump(mlp_est, OUT_DIR / "mlp_best.pkl")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY  (sorted by CV RMSE)")
    print("=" * 65)
    print(f"  {'Model':<22} {'CV RMSE':>10} {'Train RMSE':>12} {'Train R²':>10}")
    print("  " + "-" * 58)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["cv_rmse"]):
        print(f"  {name:<22} {r['cv_rmse']:>10.4f} {r['train_rmse']:>12.4f} {r['train_r2']:>10.4f}")
    print("=" * 65)

    json_results = {
        k: {kk: vv for kk, vv in v.items() if kk != "preds"}
        for k, v in results.items()
    }
    with open(OUT_DIR / "grid_search_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    for old_fig in OUT_DIR.glob("*.png"):
        old_fig.unlink()

    print(f"\nGenerating plots ...")
    generate_plots(results, y, xgb_est, cb_model)

    # ── Walk-forward backtest (all models, expanding window) ─────────────────
    sk_estimators = {}
    if xgb_est is not None:
        sk_estimators["XGBoost"] = xgb_est
    if spl_est is not None:
        sk_estimators["SplineRegression"] = spl_est
    if mlp_est is not None:
        sk_estimators["MLP"] = mlp_est

    all_rows = walk_forward_backtest(frame, sk_estimators, cb_params, min_train_years=2)

    # Print per-model summary tables
    for name, rows in all_rows.items():
        valid = [r for r in rows if not r["incomplete_outcome"]]
        print(f"\n[Walk-Forward: {name}]")
        print(f"  {'Year':<6} {'N Train':>8} {'N Test':>7} {'RMSE':>8} {'R²':>8}  Note")
        print("  " + "-" * 55)
        for r in rows:
            note = "* incomplete" if r["incomplete_outcome"] else ""
            print(f"  {r['draft_year']:<6} {r['n_train']:>8} {r['n_test']:>7} "
                  f"{r['rmse']:>8.4f} {r['r2']:>8.4f}  {note}")
        if valid:
            overall_rmse = np.sqrt(mean_squared_error(
                np.concatenate([r["actuals"] for r in valid]),
                np.concatenate([r["preds"]   for r in valid]),
            ))
            print(f"  → Overall RMSE = {overall_rmse:.4f}")

    wf_summary = generate_backtest_plots(all_rows)
    print("\n[Walk-Forward Overall Summary]")
    print(f"  {'Model':<22} {'WF RMSE':>10} {'WF R²':>8}")
    print("  " + "-" * 44)
    for name, (rmse, r2) in sorted(wf_summary.items(), key=lambda x: x[1][0]):
        print(f"  {name:<22} {rmse:>10.4f} {r2:>8.4f}")

    # Save per-year results for all models to CSV
    csv_rows = []
    for name, rows in all_rows.items():
        for r in rows:
            csv_rows.append({
                "model":            name,
                "draft_year":       r["draft_year"],
                "n_train":          r["n_train"],
                "n_test":           r["n_test"],
                "rmse":             r["rmse"],
                "r2":               r["r2"],
                "incomplete_outcome": r["incomplete_outcome"],
            })
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "walkforward_results.csv", index=False)
    print("  Saved: walkforward_results.csv")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
