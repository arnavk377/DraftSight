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
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

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
    X_sk = X_sk.replace([np.inf, -np.inf], np.nan)
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
    X_cb, y_cb, num_cols_cb, cat_cols_cb = prepare_features(frame, use_college_name=True)
    X_cb = X_cb.replace([np.inf, -np.inf], np.nan)
    for c in num_cols_cb:
        if X_cb[c].isna().any():
            X_cb[c] = X_cb[c].fillna(X_cb[c].median())
    for c in cat_cols_cb:
        X_cb[c] = X_cb[c].fillna("Unknown").astype(str)
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
    print("\n" + "=" * 65)
    print("SUMMARY  (sorted by CV RMSE)")
    print("=" * 65)
    print(f"  {'Model':<22} {'CV RMSE':>10} {'Train RMSE':>12} {'Train R²':>10}")
    print("  " + "-" * 58)
    summary = sorted(results.items(), key=lambda kv: kv[1]["cv_rmse"])
    for name, r in summary:
        print(f"  {name:<22} {r['cv_rmse']:>10.4f} {r['train_rmse']:>12.4f} {r['train_r2']:>10.4f}")
    print("=" * 65)

    json_results = {
        k: {kk: vv for kk, vv in v.items() if kk != "preds"}
        for k, v in results.items()
    }
    with open(OUT_DIR / "grid_search_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nGenerating plots ...")
    generate_plots(results, y, xgb_search.best_estimator_, cb_model)
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
