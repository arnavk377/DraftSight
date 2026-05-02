"""
Unified walk-forward training and evaluation for NFL draft 2-year AV prediction.

Models compared:
  - Pick-bin baseline: mean 2-year AV for historical picks in the same draft-slot bin
  - Spline + Ridge
  - XGBoost
  - TabNet

Outputs:
  - poc_outputs/model_v3_walkforward_results.csv
  - poc_outputs/model_v3_overall_summary.csv
  - poc_outputs/model_v3_latest_year_predictions_<year>.csv
  - Latest-year scatter plots plus spline and pick-bin value curves
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.model_v2.tabnet import TabNetRegressor


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRAFT_CSV = os.path.join(REPO_ROOT, "src", "data", "raw", "draft_picks.csv")
AV_DIR = os.path.join(REPO_ROOT, "scraping_av", "data")
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WALK_FORWARD_START_YEAR = 1999
PICK_BIN_SIZE = 16
SELECTED_SUMMARY_YEARS = [2000, 2010, 2020, 2024]

TABNET_CFG = dict(
    n_d=16,
    n_a=16,
    n_steps=4,
    gamma=1.5,
    n_shared=2,
    n_step_dep=2,
    vbs=64,
    momentum=0.02,
    lambda_sparse=1e-3,
)

TRAIN_CFG = dict(
    lr=0.02,
    lr_decay=0.95,
    lr_decay_steps=200,
    batch_size=256,
    max_epochs=500,
    patience=50,
    val_fraction=0.2,
)

NUM_COLS = ["pick", "round", "age"]
CAT_COLS = ["position", "category", "team", "college", "side"]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_draft(draft_csv: str) -> pd.DataFrame:
    df = pd.read_csv(draft_csv)
    df.columns = [c.strip() for c in df.columns]

    if "draft_season" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"draft_season": "season"})

    needed = [
        "season",
        "pick",
        "round",
        "team",
        "position",
        "category",
        "side",
        "age",
        "college",
        "pfr_player_id",
        "pfr_player_name",
    ]
    keep = [c for c in needed if c in df.columns]
    df = df[keep].copy()

    if "season" not in df.columns or "pick" not in df.columns:
        raise ValueError(f"Draft CSV missing required columns. Found: {df.columns.tolist()}")

    df["season"] = df["season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)

    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def load_av_from_year_files(av_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(av_dir, "*_av.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_av.csv files found in {av_dir}")

    dfs = []
    for fp in paths:
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(
            columns={
                "Year": "season",
                "PlayerID": "pfr_player_id",
                "Player": "player_name",
                "AV": "av",
            }
        )
        needed = {"season", "pfr_player_id", "player_name", "av"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{fp} missing columns {needed}. Found: {df.columns.tolist()}")

        df = df[list(needed)].copy()
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["av"] = pd.to_numeric(df["av"], errors="coerce").fillna(0.0)
        dfs.append(df)

    av_long = pd.concat(dfs, ignore_index=True)
    av_long = av_long.groupby(["season", "pfr_player_id"], as_index=False).agg(
        player_name=("player_name", "first"),
        av=("av", "sum"),
    )
    return av_long


def build_two_year_labels(av_long: pd.DataFrame) -> pd.DataFrame:
    cur = av_long.rename(columns={"season": "draft_season", "av": "av_y"})
    nxt = av_long.rename(columns={"season": "next_season", "av": "av_y1"})[
        ["next_season", "pfr_player_id", "av_y1"]
    ]

    merged = cur.merge(nxt, on="pfr_player_id", how="left")
    merged = merged[merged["next_season"] == merged["draft_season"] + 1].copy()
    merged["av_2yr"] = merged["av_y"] + merged["av_y1"].fillna(0.0)
    return merged[["draft_season", "pfr_player_id", "av_2yr"]]


def build_model_frame() -> pd.DataFrame:
    draft = load_draft(DRAFT_CSV)
    av_long = load_av_from_year_files(AV_DIR)
    labels = build_two_year_labels(av_long)

    if "pfr_player_id" not in draft.columns:
        raise ValueError("Draft data missing pfr_player_id.")

    df = draft.rename(columns={"season": "draft_season"}).merge(
        labels,
        on=["draft_season", "pfr_player_id"],
        how="left",
    )

    labeled = df["av_2yr"].notna().sum()
    print(f"Join coverage: labeled {labeled}/{len(df)} = {labeled / len(df):.1%}")

    df = df.dropna(subset=["av_2yr"]).copy()
    df["av_2yr"] = df["av_2yr"].astype(float)
    return df


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    sp = spearmanr(y_true, y_pred).correlation
    if pd.isna(sp):
        sp = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(sp),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _nice_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_pred_vs_actual(y_true, y_pred, out_path, draft_year, model_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = eval_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
    ax.scatter(y_true, y_pred, s=22, alpha=0.65, edgecolor="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1.2, alpha=0.85)

    ax.set_title(f"{model_name}: {draft_year} Draft Class", fontsize=13, pad=10)
    ax.set_xlabel(f"Actual 2-Year AV = AV({draft_year}) + AV({draft_year + 1})", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)
    ax.text(
        0.02,
        0.98,
        (
            f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
            f"Spearman={metrics['spearman']:.2f}  R2={metrics['r2']:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, edgecolor="0.85"),
    )
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_pick_curve(picks, preds, out_path, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)
    ax.plot(picks, preds, linewidth=2.0)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Pick number", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_importance(importance: np.ndarray, feature_names: list[str], out_path: str):
    idx = np.argsort(importance)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        importance[list(reversed(idx))],
        color="#4C72B0",
    )
    ax.set_xlabel("Aggregate TabNet Feature Importance", fontsize=11)
    ax.set_title("TabNet: Top-20 Feature Importances", fontsize=13, pad=10)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_loss_curves(train_losses, val_losses, out_path, test_year):
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=180)
    ax.plot(train_losses, label="Train loss", linewidth=1.4)
    ax.plot(val_losses, label="Val loss", linewidth=1.4)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (MSE + sparsity)", fontsize=11)
    ax.set_title(f"TabNet Training Curves — Test Year {test_year}", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_tabular_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    transformers = []
    if "pick" in num_cols:
        transformers.append(
            ("pick_spline", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["pick"])
        )
        other_nums = [c for c in num_cols if c != "pick"]
        if other_nums:
            transformers.append(("num_passthrough", "passthrough", other_nums))
    elif num_cols:
        transformers.append(("num_passthrough", "passthrough", num_cols))

    if cat_cols:
        transformers.append(("cat", make_one_hot_encoder(), cat_cols))

    return ColumnTransformer(transformers), num_cols, cat_cols


def build_tabnet_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols)
        )
    if cat_cols:
        transformers.append(("cat", make_one_hot_encoder(), cat_cols))

    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def get_tabnet_feature_names(preprocessor, num_cols, cat_cols) -> list[str]:
    names = list(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]
        names += list(ohe.get_feature_names_out(cat_cols))
    return names


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    tabnet_cfg: dict,
    train_cfg: dict,
    device: torch.device,
    verbose: bool = True,
) -> tuple[TabNetRegressor, list[float], list[float]]:
    model = TabNetRegressor(n_features=n_features, **tabnet_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg["lr_decay"])

    n_val = max(1, int(len(X_train) * train_cfg["val_fraction"]))
    X_t, X_v = X_train[:-n_val], X_train[-n_val:]
    y_t, y_v = y_train[:-n_val], y_train[-n_val:]

    ds_train = TensorDataset(to_tensor(X_t), to_tensor(y_t))
    loader = DataLoader(ds_train, batch_size=train_cfg["batch_size"], shuffle=True)

    X_val_t = to_tensor(X_v).to(device)
    y_val_t = to_tensor(y_v).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    step = 0

    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred, sp_loss, _ = model(X_batch)
            loss = model.loss(pred, y_batch, sp_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            if step % train_cfg["lr_decay_steps"] == 0:
                scheduler.step()

        train_losses.append(epoch_loss / max(1, len(loader)))

        model.eval()
        with torch.no_grad():
            v_pred, v_sp, _ = model(X_val_t)
            val_loss = model.loss(v_pred, y_val_t, v_sp).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"  epoch {epoch + 1:4d}  train={train_losses[-1]:.3f}  "
                f"val={val_loss:.3f}  best_val={best_val_loss:.3f}  "
                f"patience={patience_counter}/{train_cfg['patience']}"
            )

        if patience_counter >= train_cfg["patience"]:
            if verbose:
                print(f"  Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


class PickBinBaseline:
    def __init__(self, bin_size: int = 16):
        self.bin_size = bin_size
        self.global_mean_: float | None = None
        self.bin_means_: dict[int, float] = {}

    def _bin_id(self, pick_values) -> np.ndarray:
        pick_values = np.asarray(pick_values, dtype=int)
        return (pick_values - 1) // self.bin_size

    def fit(self, picks, y):
        df = pd.DataFrame({"pick": np.asarray(picks, dtype=int), "y": np.asarray(y, dtype=float)})
        df["pick_bin"] = self._bin_id(df["pick"].values)
        self.global_mean_ = float(df["y"].mean())
        self.bin_means_ = df.groupby("pick_bin")["y"].mean().to_dict()
        return self

    def predict(self, picks) -> np.ndarray:
        if self.global_mean_ is None:
            raise ValueError("PickBinBaseline must be fit before predict.")
        bins = self._bin_id(picks)
        return np.array([self.bin_means_.get(int(bin_id), self.global_mean_) for bin_id in bins], dtype=float)

    def predict_curve(self, max_pick: int = 260) -> tuple[np.ndarray, np.ndarray]:
        picks = np.arange(1, max_pick + 1)
        return picks, self.predict(picks)


def main():
    print(f"Loading draft: {DRAFT_CSV}")
    print(f"Loading AVs from: {AV_DIR}")
    print(f"Using AV label coverage through draft year 2024 because 2025 AV is now present.")
    print(f"Pick-bin baseline width: {PICK_BIN_SIZE} picks")

    df = build_model_frame()

    feat_cols = [c for c in NUM_COLS + CAT_COLS if c in df.columns]
    years = df["draft_season"].values
    unique_years = sorted(np.unique(years))
    test_years = [year for year in unique_years if year > WALK_FORWARD_START_YEAR]
    if not test_years:
        raise ValueError(f"No test years found after {WALK_FORWARD_START_YEAR}.")

    latest_year = max(test_years)
    results = []
    overall_rows = []

    for test_year in test_years:
        train_mask = (years >= WALK_FORWARD_START_YEAR) & (years < test_year)
        test_mask = years == test_year

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        X_train = train_df[feat_cols].copy()
        X_test = test_df[feat_cols].copy()
        y_train = train_df["av_2yr"].to_numpy(dtype=float)
        y_test = test_df["av_2yr"].to_numpy(dtype=float)

        print(
            f"\n--- Test year {test_year}  train={train_mask.sum()}  test={test_mask.sum()} ---"
        )

        pick_bin_model = PickBinBaseline(bin_size=PICK_BIN_SIZE).fit(train_df["pick"].values, y_train)
        pred_pick_bin = pick_bin_model.predict(test_df["pick"].values)

        spline_pre, _, _ = build_tabular_preprocessor(X_train)
        spline_model = Pipeline([("pre", spline_pre), ("reg", Ridge(alpha=1.0))])
        spline_model.fit(X_train, y_train)
        pred_spline = spline_model.predict(X_test)

        xgb_pre, _, _ = build_tabular_preprocessor(X_train)
        xgb_model = Pipeline(
            [
                ("pre", xgb_pre),
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )
        xgb_model.fit(X_train, y_train)
        pred_xgb = xgb_model.predict(X_test)

        tabnet_pre, tabnet_num_cols, tabnet_cat_cols = build_tabnet_preprocessor(X_train)
        X_train_tabnet = tabnet_pre.fit_transform(X_train).astype(np.float32)
        X_test_tabnet = tabnet_pre.transform(X_test).astype(np.float32)
        tabnet_feature_names = get_tabnet_feature_names(
            tabnet_pre, tabnet_num_cols, tabnet_cat_cols
        )

        tabnet_model, train_losses, val_losses = train_tabnet(
            X_train_tabnet,
            y_train,
            n_features=X_train_tabnet.shape[1],
            tabnet_cfg=TABNET_CFG,
            train_cfg=TRAIN_CFG,
            device=DEVICE,
            verbose=True,
        )
        tabnet_model.eval()
        with torch.no_grad():
            pred_tabnet_raw, _, masks_te = tabnet_model(to_tensor(X_test_tabnet).to(DEVICE))
            pred_tabnet = pred_tabnet_raw.cpu().numpy()
            tabnet_importance = tabnet_model.encoder.aggregate_importance(masks_te).cpu().numpy()

        year_result = {
            "test_year": int(test_year),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        for model_name, preds in [
            ("pick_bin", pred_pick_bin),
            ("spline", pred_spline),
            ("xgb", pred_xgb),
            ("tabnet", pred_tabnet),
        ]:
            metrics = eval_metrics(y_test, preds)
            print(
                f"  {model_name:8s}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
                f"Spearman={metrics['spearman']:.3f}  R2={metrics['r2']:.3f}"
            )
            for metric_name, metric_value in metrics.items():
                year_result[f"{model_name}_{metric_name}"] = metric_value

            pred_frame = test_df[
                ["draft_season", "pick", "round", "team", "position", "college", "pfr_player_name", "av_2yr"]
            ].copy()
            pred_frame["model"] = model_name
            pred_frame["prediction"] = preds
            overall_rows.append(pred_frame)

        results.append(year_result)

        if test_year == latest_year:
            latest_pred_df = test_df[
                ["draft_season", "pick", "round", "team", "position", "college", "pfr_player_name", "av_2yr"]
            ].copy()
            latest_pred_df["pred_pick_bin"] = pred_pick_bin
            latest_pred_df["pred_spline"] = pred_spline
            latest_pred_df["pred_xgb"] = pred_xgb
            latest_pred_df["pred_tabnet"] = pred_tabnet
            latest_pred_df.to_csv(
                os.path.join(OUT_DIR, f"model_v3_latest_year_predictions_{test_year}.csv"),
                index=False,
            )

            save_pred_vs_actual(
                y_test,
                pred_pick_bin,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_pick_bin_{test_year}.png"),
                draft_year=test_year,
                model_name="Pick-Bin Baseline",
            )
            save_pred_vs_actual(
                y_test,
                pred_spline,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_spline_{test_year}.png"),
                draft_year=test_year,
                model_name="Spline Ridge",
            )
            save_pred_vs_actual(
                y_test,
                pred_xgb,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_xgb_{test_year}.png"),
                draft_year=test_year,
                model_name="XGBoost",
            )
            save_pred_vs_actual(
                y_test,
                pred_tabnet,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_tabnet_{test_year}.png"),
                draft_year=test_year,
                model_name="TabNet",
            )

            curve_picks, curve_preds = pick_bin_model.predict_curve()
            save_pick_curve(
                curve_picks,
                curve_preds,
                os.path.join(OUT_DIR, "plot_value_curve_pick_bin.png"),
                title=f"Pick-Bin Baseline Value Curve ({PICK_BIN_SIZE}-pick bins)",
            )

            spline_grid = pd.DataFrame({"pick": np.arange(1, 261)})
            for col in X_train.columns:
                if col == "pick":
                    continue
                if X_train[col].dtype == "O":
                    spline_grid[col] = X_train[col].mode().iloc[0]
                else:
                    spline_grid[col] = float(X_train[col].median())
            spline_curve = spline_model.predict(spline_grid)
            save_pick_curve(
                spline_grid["pick"].to_numpy(),
                spline_curve,
                os.path.join(OUT_DIR, "plot_value_curve_spline.png"),
                title="Spline Estimated Pick Value Curve",
            )

            save_feature_importance(
                tabnet_importance,
                tabnet_feature_names,
                os.path.join(OUT_DIR, "plot_feature_importance_tabnet.png"),
            )
            save_loss_curves(
                train_losses,
                val_losses,
                os.path.join(OUT_DIR, f"plot_tabnet_loss_curves_{test_year}.png"),
                test_year=test_year,
            )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUT_DIR, "model_v3_walkforward_results.csv")
    results_df.to_csv(results_path, index=False)

    selected_rows = []
    for year in SELECTED_SUMMARY_YEARS:
        year_slice = results_df[results_df["test_year"] == year]
        if year_slice.empty:
            continue
        row = year_slice.iloc[0]
        for model_name in ["pick_bin", "spline", "xgb", "tabnet"]:
            selected_rows.append(
                {
                    "year": int(row["test_year"]),
                    "model_type": model_name,
                    "n_train": int(row["n_train"]),
                    "n_test": int(row["n_test"]),
                    "mae": float(row[f"{model_name}_mae"]),
                    "rmse": float(row[f"{model_name}_rmse"]),
                    "spearman": float(row[f"{model_name}_spearman"]),
                    "r2": float(row[f"{model_name}_r2"]),
                }
            )
    selected_summary_df = pd.DataFrame(selected_rows)[
        ["year", "model_type", "n_train", "n_test", "mae", "rmse", "spearman", "r2"]
    ]
    selected_summary_path = os.path.join(OUT_DIR, "model_v3_selected_years_summary.csv")
    selected_summary_df.to_csv(selected_summary_path, index=False)

    overall_df = pd.concat(overall_rows, ignore_index=True)
    summary_rows = []
    for model_name in ["pick_bin", "spline", "xgb", "tabnet"]:
        model_df = overall_df[overall_df["model"] == model_name]
        metrics = eval_metrics(model_df["av_2yr"].values, model_df["prediction"].values)
        metrics["model"] = model_name
        metrics["n_predictions"] = int(len(model_df))
        summary_rows.append(metrics)
    summary_df = pd.DataFrame(summary_rows)[
        ["model", "n_predictions", "mae", "rmse", "spearman", "r2"]
    ]
    summary_path = os.path.join(OUT_DIR, "model_v3_overall_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nWalk-forward results:")
    print(results_df.to_string(index=False))
    print("\nOverall out-of-sample summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved per-year metrics: {results_path}")
    print(f"Saved selected-years summary: {selected_summary_path}")
    print(f"Saved overall summary: {summary_path}")
    print(f"Saved plots and latest-year predictions in: {OUT_DIR}")


if __name__ == "__main__":
    main()
