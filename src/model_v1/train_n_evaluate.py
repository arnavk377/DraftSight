import os
import numpy as np
import pandas as pd
import glob

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

# xgboost
from xgboost import XGBRegressor


# ---------------------------
# CONFIG (edit these 3 paths)
# ---------------------------

# repo root = two levels up from src/model_v1/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DRAFT_CSV = os.path.join(REPO_ROOT, "src", "data", "raw", "draft_picks.csv")

# Output directory
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------
# Helpers
# ---------------------------

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sp = spearmanr(y_true, y_pred).correlation
    return mae, rmse, sp


def _nice_axes(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def save_pred_vs_actual_pretty(y_true, y_pred, out_path, draft_year, model_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae, rmse, sp = eval_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
    ax.scatter(y_true, y_pred, s=22, alpha=0.65, edgecolor="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1.2, alpha=0.85)

    title = f"{model_name}: {draft_year} Draft Class"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(f"Actual 2-Year AV = AV({draft_year}) + AV({draft_year+1})", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)

    txt = f"MAE={mae:.2f}  RMSE={rmse:.2f}  Spearman={sp:.2f}"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, edgecolor="0.85")
    )

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_spline_value_curve_pretty(model, X_train, out_path, train_end_year):
    grid = pd.DataFrame({"pick": np.arange(1, 261)})
    for col in X_train.columns:
        if col == "pick":
            continue
        if X_train[col].dtype == "O":
            grid[col] = X_train[col].mode().iloc[0]
        else:
            grid[col] = float(X_train[col].median())

    preds = model.predict(grid)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)
    ax.plot(grid["pick"], preds, linewidth=2.0)

    ax.set_title("Estimated Pick Value Curve (Predicted 2-year AV)", fontsize=13, pad=10)
    ax.set_xlabel("Pick number", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)


    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_draft(draft_csv: str) -> pd.DataFrame:
    df = pd.read_csv(draft_csv)
    df.columns = [c.strip() for c in df.columns]

    # Normalize common names depending on whether draft_picks.csv or draft_features.csv
    # Expected key columns: season, pick, round, team, position/category/side, age/college, pfr_player_id
    rename_map = {}
    if "draft_season" in df.columns and "season" not in df.columns:
        rename_map["draft_season"] = "season"
    if rename_map:
        df = df.rename(columns=rename_map)

    needed = ["season", "pick", "round", "team", "position", "category", "side", "age", "college", "pfr_player_id", "pfr_player_name"]
    # Keep what exists (don’t crash if some are missing)
    keep = [c for c in needed if c in df.columns]
    df = df[keep].copy()

    if "season" not in df.columns or "pick" not in df.columns:
        raise ValueError(f"Draft CSV missing required columns. Found: {df.columns.tolist()}")

    df["season"] = df["season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)

    # round/age may be missing
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Make categoricals clean
    for c in ["team", "position", "category", "side", "college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

AV_DIR = os.path.join(REPO_ROOT, "scraping_av", "data")

def load_av_from_year_files(av_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(av_dir, "*_av.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_av.csv files found in {av_dir}")

    dfs = []
    for fp in paths:
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]

        # Normalize expected columns
        # Your files have: Year, Team, PlayerID, Player, Position, Experience, AV
        df = df.rename(columns={
            "Year": "season",
            "PlayerID": "pfr_player_id",
            "Player": "player_name",
            "AV": "av",
        })

        needed = {"season", "pfr_player_id", "player_name", "av"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{fp} missing columns {needed}. Found: {df.columns.tolist()}")

        df = df[list(needed)].copy()
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["av"] = pd.to_numeric(df["av"], errors="coerce").fillna(0.0)

        dfs.append(df)

    av_long = pd.concat(dfs, ignore_index=True)

    # collapse duplicates within season/player_id
    av_long = av_long.groupby(["season", "pfr_player_id"], as_index=False).agg(
        player_name=("player_name", "first"),
        av=("av", "sum")
    )

    return av_long


def build_two_year_labels(av_long: pd.DataFrame) -> pd.DataFrame:
    # label for draft class Y = AV(Y) + AV(Y+1)
    a = av_long.rename(columns={"season": "draft_season", "av": "av_y"})
    b = av_long.rename(columns={"season": "next_season", "av": "av_y1"})[["next_season", "pfr_player_id", "av_y1"]]

    merged = a.merge(b, on="pfr_player_id", how="left")
    merged = merged[merged["next_season"] == merged["draft_season"] + 1].copy()
    merged["av_2yr"] = merged["av_y"] + merged["av_y1"].fillna(0.0)

    return merged[["draft_season", "pfr_player_id", "av_2yr"]]


def main():
    print("Loading draft:", DRAFT_CSV)
    draft = load_draft(DRAFT_CSV)

    print("Loading AVs:")
    av_long = load_av_from_year_files(AV_DIR)
    labels = build_two_year_labels(av_long)

    # Join draft to labels via pfr_player_id
    if "pfr_player_id" not in draft.columns:
        raise ValueError("Draft data missing pfr_player_id. Use draft_picks.csv or draft_features.csv that contains it.")

    df = draft.rename(columns={"season": "draft_season"}).merge(
        labels, on=["draft_season", "pfr_player_id"], how="left"
    )

    # Join coverage
    total = len(df)
    labeled = df["av_2yr"].notna().sum()
    print(f"Join coverage: labeled {labeled}/{total} = {labeled/total:.1%}")

    # Keep labeled only, restrict to 2000+ to avoid sparse/incomplete early data
    df = df.dropna(subset=["av_2yr"]).copy()
    df = df[df["draft_season"] >= 2000].copy()
    df["av_2yr"] = df["av_2yr"].astype(float)

    # Feature columns (draft-night only)
    num_cols = [c for c in ["pick", "round", "age"] if c in df.columns]
    cat_cols = [c for c in ["position", "category", "team", "college", "side"] if c in df.columns]

    X = df[num_cols + cat_cols].copy()
    y = df["av_2yr"].values
    years = df["draft_season"].values

    # Preprocessing: spline on pick, one-hot categoricals
    transformers = []
    if "pick" in num_cols:
        transformers.append(("pick_spline", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["pick"]))
        other_nums = [c for c in num_cols if c != "pick"]
        if other_nums:
            transformers.append(("num_passthrough", "passthrough", other_nums))
    else:
        transformers.append(("num_passthrough", "passthrough", num_cols))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    pre = ColumnTransformer(transformers)

    spline_model = Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0))
    ])

    xgb_model = Pipeline([
        ("pre", pre),
        ("xgb", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42
        ))
    ])

    unique_years = sorted(np.unique(years))
    if len(unique_years) < 2:
        raise ValueError(f"Not enough labeled years to backtest. Years found: {unique_years}")

    results = []
    latest_year = max(unique_years)

    for test_year in unique_years[1:]:
        train_mask = years < test_year
        test_mask = years == test_year

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Fit + predict spline
        spline_model.fit(X_train, y_train)
        pred_s = spline_model.predict(X_test)

        # Fit + predict xgb
        xgb_model.fit(X_train, y_train)
        pred_x = xgb_model.predict(X_test)

        mae_s, rmse_s, sp_s = eval_metrics(y_test, pred_s)
        mae_x, rmse_x, sp_x = eval_metrics(y_test, pred_x)

        results.append({
            "test_year": int(test_year),
            "n_test": int(test_mask.sum()),
            "spline_mae": float(mae_s),
            "spline_rmse": float(rmse_s),
            "spline_spearman": float(sp_s),
            "xgb_mae": float(mae_x),
            "xgb_rmse": float(rmse_x),
            "xgb_spearman": float(sp_x),
        })

        # Save plots for the latest year only (best for slides)
        if test_year == latest_year:
            save_pred_vs_actual_pretty(
                y_test, pred_s,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_spline_{test_year}.png"),
                draft_year=test_year,
                model_name="Spline Ridge"
            )
            save_pred_vs_actual_pretty(
                y_test, pred_x,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_xgb_{test_year}.png"),
                draft_year=test_year,
                model_name="XGBoost"
            )
            save_spline_value_curve_pretty(
                spline_model, X_train,
                os.path.join(OUT_DIR, "plot_value_curve_spline.png"),
                train_end_year=test_year - 1
            )

    res_df = pd.DataFrame(results)
    print("\nWalk-forward results:")
    print(res_df)

    out_csv = os.path.join(OUT_DIR, "poc_walkforward_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results CSV: {out_csv}")
    print(f"Saved plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()