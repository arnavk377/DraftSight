"""
export_pick_values.py
=====================
Generate predicted 2-year AV for NFL draft picks 1–262 and export
as pick_values.json for the DraftSight React frontend.

Approach:
  - Loads saved XGBoost hyperparams from modeling/results/grid_search_results.json
  - Refits XGBoost pipeline on full training data (sklearn-version safe)
  - Builds synthetic rows for picks 1–262 (all non-pick features = training median)
  - Applies isotonic regression to enforce monotonic decrease
  - Writes pick_values.json to the repo root

Run:
    python export_pick_values.py
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data" / "supabase_exports"
RESULTS_DIR   = ROOT / "modeling" / "results"
PARAMS_PATH   = RESULTS_DIR / "grid_search_results.json"
OUT_PATH      = ROOT / "pick_values.json"

RANDOM_STATE = 42

# ── Mirror constants from grid_search.py ──────────────────────────────────────
LEAKAGE_COLS = {
    "car_av", "w_av", "dr_av", "games", "allpro", "probowls",
    "seasons_started", "hof", "last_nfl_season",
    "pass_completions", "pass_attempts", "pass_yards", "pass_tds", "pass_ints",
    "rush_atts", "rush_yards", "rush_tds",
    "receptions", "rec_yards", "rec_tds",
    "def_solo_tackles", "def_ints", "def_sacks",
    "pick_approx_value_points",
}
DROP_IDS = {
    "draft_pick_id", "gsis_id", "pfr_player_id", "sports_ref_cfb_player_id",
    "pfr_player_name", "team", "team_raw", "franchise_id",
    "college_stats_id", "cfb_player_id",
    "pick_first_trade_date", "pick_last_trade_date",
    "pick_first_gave_team", "pick_first_received_team",
    "pick_last_gave_team", "pick_last_received_team",
    "pick_trade_team_chain",
    "roster_context_available", "roster_context_season",
    "draft_season",
    "pick_post_year_trade_count",
}
OHE_COLS      = ["position", "position_group", "category", "side"]
HIGH_CARD_CATS = {"college"}


# ── Data loading (mirrors grid_search.py) ─────────────────────────────────────
def load_data():
    drafts  = pd.read_csv(DATA_DIR / "drafts.csv")
    av      = pd.read_csv(DATA_DIR / "av.csv")
    college = pd.read_csv(DATA_DIR / "college_stats.csv")
    context = pd.read_csv(DATA_DIR / "draft_pick_context_features.csv")
    return drafts, av, college, context


def build_target(drafts, av):
    base   = drafts[["pfr_player_id", "draft_season"]].dropna(subset=["pfr_player_id"])
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


def build_frame(drafts, av, college, context):
    draft_cols = [c for c in drafts.columns if c not in LEAKAGE_COLS]
    frame = drafts[draft_cols].copy()

    av_2yr = build_target(drafts, av)
    frame  = frame.merge(av_2yr, on="pfr_player_id", how="left")
    frame["av_2yr"] = frame["av_2yr"].fillna(0)

    college_drop = {
        "college_stats_id", "cfb_player_id", "player", "college_team",
        "conference", "position", "position_group", "draft_round",
        "draft_pick_in_round",
    }
    college_keep = [c for c in college.columns if c not in college_drop]
    college_join = college[college_keep].rename(columns={"draft_overall": "pick"})
    frame = frame.merge(college_join, on=["draft_season", "pick"], how="left")

    ctx_drop = {"team", "team_draft_code", "franchise_id", "position", "draft_position_group"}
    ctx_cols = [c for c in context.columns if c not in ctx_drop]
    frame = frame.merge(context[ctx_cols], on=["draft_season", "pick"], how="left")

    return frame


def prepare_features(frame, use_college_name=False):
    drop = DROP_IDS | LEAKAGE_COLS
    if not use_college_name:
        drop |= HIGH_CARD_CATS

    X = frame.drop(columns=["av_2yr"] + [c for c in drop if c in frame.columns])
    y = frame["av_2yr"].values

    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    X[bool_cols] = X[bool_cols].astype(int)

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


# ── Derive round from pick number ─────────────────────────────────────────────
def pick_to_round(pick: int) -> int:
    """
    Standard NFL round boundaries (approximate; compensatory picks are counted
    in their respective rounds). Covers picks 1–262.
    """
    boundaries = [32, 64, 100, 135, 176, 215, 262]
    for rnd, upper in enumerate(boundaries, start=1):
        if pick <= upper:
            return rnd
    return 7


# ── Preprocessor factory (mirrors grid_search.py) ─────────────────────────────
def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def parse_params(raw: str) -> dict:
    """Parse best_params string that may contain np.float64(...) wrappers."""
    cleaned = re.sub(r"np\.\w+\(([^)]+)\)", r"\1", raw)
    return ast.literal_eval(cleaned)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load saved hyperparameters
    if not PARAMS_PATH.exists():
        sys.exit(f"Saved params not found: {PARAMS_PATH}\n"
                 "Run: python modeling/grid_search.py")
    with open(PARAMS_PATH) as f:
        saved = json.load(f)
    xgb_params = parse_params(saved["XGBoost"]["best_params"])
    print(f"Loaded XGBoost hyperparams (CV RMSE={saved['XGBoost']['cv_rmse']:.4f})")

    # 2. Build training frame
    print("Loading data and building training frame ...")
    drafts, av, college, context = load_data()
    frame = build_frame(drafts, av, college, context)
    X_train, y_train, num_cols, cat_cols = prepare_features(frame, use_college_name=False)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    bool_c  = X_train.select_dtypes(include="bool").columns.tolist()
    X_train[bool_c] = X_train[bool_c].astype(int)
    for c in num_cols:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med if not np.isnan(med) else 0.0)
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("_missing_").astype(str)

    print(f"  Training frame: {X_train.shape[1]} features, {X_train.shape[0]} rows")

    # 3. Refit XGBoost pipeline on full training data with saved hyperparams
    print("Refitting XGBoost pipeline on full training data ...")
    pipe = Pipeline([
        ("pre", make_preprocessor(
            [c for c in X_train.columns if c not in OHE_COLS],
            [c for c in OHE_COLS if c in X_train.columns],
        )),
        ("model", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )),
    ])
    pipe.set_params(**xgb_params)
    pipe.fit(X_train, y_train)
    train_preds = pipe.predict(X_train)
    from sklearn.metrics import mean_squared_error, r2_score
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.4f}"
          f"  Train R²: {r2_score(y_train, train_preds):.4f}")

    # 4. Predict on ALL training rows, then aggregate by pick slot
    #    This uses actual player college stats for each pick slot, giving realistic
    #    scale, rather than using all-median synthetic rows which suppress the signal.
    print("Predicting on full training set and aggregating by pick slot ...")
    all_preds = pipe.predict(X_train)

    # Build a DataFrame with pick slot + model prediction + actual AV
    agg_df = pd.DataFrame({
        "pick":   X_train["pick"].values,
        "pred":   all_preds,
        "actual": y_train,
    })
    agg_df["pick"] = agg_df["pick"].astype(float)

    # Average model-predicted AV per pick slot (covers picks actually in training data)
    pick_avg = (
        agg_df.groupby("pick")["pred"]
        .mean()
        .reset_index()
        .rename(columns={"pred": "avg_pred"})
    )
    pick_avg["pick"] = pick_avg["pick"].astype(int)

    # 5. For each pick 1–262, use pick_avg if available, else interpolate
    picks = list(range(1, 263))
    known_picks = dict(zip(pick_avg["pick"], pick_avg["avg_pred"]))

    # Fit a smooth curve using all known pick-value pairs for interpolation
    known_x = np.array(sorted(known_picks.keys()), dtype=float)
    known_y = np.array([known_picks[p] for p in sorted(known_picks.keys())], dtype=float)
    raw_preds = np.interp(np.array(picks, dtype=float), known_x, known_y)
    print(f"  Raw pred range: [{raw_preds.min():.2f}, {raw_preds.max():.2f}]")

    # 7. Isotonic regression to enforce strict non-increase (pick 1 → pick 262)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso_preds = iso.fit_transform(np.array(picks, dtype=float), raw_preds)

    # 8. Clip negatives, round to 2 decimal places
    iso_preds = np.clip(iso_preds, 0.0, None)
    iso_preds = np.round(iso_preds, 2)

    print(f"  After isotonic:  [{iso_preds.min():.2f}, {iso_preds.max():.2f}]")
    for label, idx in [("Pick 1", 0), ("Pick 32", 31), ("Pick 64", 63),
                        ("Pick 100", 99), ("Pick 200", 199), ("Pick 262", 261)]:
        print(f"  {label:<10} -> {iso_preds[idx]:.2f}")

    # 9. Export JSON
    output = [
        {"pick": int(p), "value": float(v)}
        for p, v in zip(picks, iso_preds)
    ]
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported {len(output)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
