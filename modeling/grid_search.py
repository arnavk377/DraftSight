"""
NFL Draft Value Grid Search
===========================
Models: XGBoost, CatBoost, Spline Regression, MLP Neural Network
Target: 2-year Approximate Value (AV) from av.csv
Frame:  drafts.csv + college_stats.csv + draft_pick_context_features.csv

Run:
    pip install xgboost catboost scikit-learn pandas numpy scipy joblib
    python modeling/grid_search.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

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
            ("num", StandardScaler(), num_cols),
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
        {k: (int(v.rvs(random_state=rng.integers(1e6)))
             if hasattr(v, "rvs") else v)
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
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer()),
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
            "best_params": str(search.best_params_)}


def eval_catboost(model, best_params, best_mse, X, y, name="CatBoost"):
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
            "best_params": str(best_params)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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

    # Fill numeric NaNs with column median for sklearn models
    X_sk = X.copy()
    for c in num_cols:
        if X_sk[c].isna().any():
            X_sk[c] = X_sk[c].fillna(X_sk[c].median())

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_search = xgboost_search(X_sk, y, cv)
    xgb_search.fit(X_sk, y)
    results["XGBoost"] = eval_best(xgb_search, X_sk, y, "XGBoost")
    joblib.dump(xgb_search.best_estimator_, OUT_DIR / "xgboost_best.pkl")

    # ── CatBoost (uses college name as cat feature) ───────────────────────────
    X_cb, y_cb, _, cat_cols_cb = prepare_features(frame, use_college_name=True)
    cb_model, cb_params, cb_mse = catboost_search(X_cb, y_cb, cv, cat_cols_cb)
    results["CatBoost"] = eval_catboost(cb_model, cb_params, cb_mse, X_cb, y_cb)
    cb_model.save_model(str(OUT_DIR / "catboost_best.cbm"))

    # ── Spline Regression ─────────────────────────────────────────────────────
    spl_search = spline_search(X_sk, y, cv, num_cols, cat_cols)
    spl_search.fit(X_sk, y)
    results["SplineRegression"] = eval_best(spl_search, X_sk, y, "Spline Regression")
    joblib.dump(spl_search.best_estimator_, OUT_DIR / "spline_best.pkl")

    # ── MLP ───────────────────────────────────────────────────────────────────
    mlp_srch = mlp_search(X_sk, y, cv, num_cols, cat_cols)
    mlp_srch.fit(X_sk, y)
    results["MLP"] = eval_best(mlp_srch, X_sk, y, "MLP")
    joblib.dump(mlp_srch.best_estimator_, OUT_DIR / "mlp_best.pkl")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY  (sorted by CV RMSE)")
    print("=" * 60)
    summary = sorted(results.items(), key=lambda kv: kv[1]["cv_rmse"])
    for name, r in summary:
        print(f"  {name:<20} CV RMSE={r['cv_rmse']:.4f}  R²={r['train_r2']:.4f}")

    with open(OUT_DIR / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
