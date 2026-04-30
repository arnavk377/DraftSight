"""
TabNet walk-forward training and evaluation for NFL draft 2-year AV prediction.

Input/output contract mirrors src/model_v1/train_n_evaluate.py:
  - Reads draft picks from src/data/raw/draft_picks.csv
  - Reads per-year AV from scraping_av/data/*_av.csv
  - Target: av_2yr = AV(draft_year) + AV(draft_year + 1)
  - Walk-forward backtesting: train on years < test_year, evaluate on test_year
  - Outputs: CSV of per-year metrics + scatter plots + feature importance plot
"""

import os
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

from src.model_v2.tabnet import TabNetRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRAFT_CSV = os.path.join(REPO_ROOT, "src", "data", "raw", "draft_picks.csv")
AV_DIR = os.path.join(REPO_ROOT, "scraping_av", "data")
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TabNet hyperparameters
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

# Training hyperparameters
TRAIN_CFG = dict(
    lr=0.02,
    lr_decay=0.95,
    lr_decay_steps=200,
    batch_size=256,
    max_epochs=500,
    patience=50,         # early stopping patience (epochs)
    val_fraction=0.2,    # fraction of training set held out for validation
)

# ---------------------------------------------------------------------------
# Data loading  (mirrors model_v1 helpers)
# ---------------------------------------------------------------------------

def load_draft(draft_csv: str) -> pd.DataFrame:
    df = pd.read_csv(draft_csv)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    if "draft_season" in df.columns and "season" not in df.columns:
        rename_map["draft_season"] = "season"
    if rename_map:
        df = df.rename(columns=rename_map)

    needed = ["season", "pick", "round", "team", "position", "category",
              "side", "age", "college", "pfr_player_id", "pfr_player_name"]
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

    for c in ["team", "position", "category", "side", "college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


def load_av_from_year_files(av_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(av_dir, "*_av.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_av.csv files found in {av_dir}")

    dfs = []
    for fp in paths:
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Year": "season", "PlayerID": "pfr_player_id",
                                 "Player": "player_name", "AV": "av"})
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
    a = av_long.rename(columns={"season": "draft_season", "av": "av_y"})
    b = av_long.rename(columns={"season": "next_season", "av": "av_y1"})[
        ["next_season", "pfr_player_id", "av_y1"]
    ]
    merged = a.merge(b, on="pfr_player_id", how="left")
    merged = merged[merged["next_season"] == merged["draft_season"] + 1].copy()
    merged["av_2yr"] = merged["av_y"] + merged["av_y1"].fillna(0.0)
    return merged[["draft_season", "pfr_player_id", "av_2yr"]]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

NUM_COLS = ["pick", "round", "age"]
CAT_COLS = ["position", "category", "team", "college", "side"]


def build_preprocessor(df: pd.DataFrame):
    """Build sklearn ColumnTransformer: passthrough nums, OHE cats."""
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median"))]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def get_feature_names(preprocessor, num_cols, cat_cols) -> list[str]:
    names = list(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]
        names += list(ohe.get_feature_names_out(cat_cols))
    return names


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    tabnet_cfg: dict,
    train_cfg: dict,
    device: torch.device,
    verbose: bool = True,
) -> tuple[TabNetRegressor, list, list]:
    """Train a TabNetRegressor with early stopping.

    Returns the trained model plus train/val loss histories.
    """
    model = TabNetRegressor(n_features=n_features, **tabnet_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg["lr_decay"])

    # Split into train / val
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
    train_losses, val_losses = [], []
    step = 0

    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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

        # Validation
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
            print(f"  epoch {epoch+1:4d}  train={train_losses[-1]:.3f}  val={val_loss:.3f}  "
                  f"best_val={best_val_loss:.3f}  patience={patience_counter}/{train_cfg['patience']}")

        if patience_counter >= train_cfg["patience"]:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sp = spearmanr(y_true, y_pred).correlation
    return mae, rmse, sp


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _nice_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_pred_vs_actual(y_true, y_pred, out_path, draft_year, model_name="TabNet"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae, rmse, sp = eval_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
    ax.scatter(y_true, y_pred, s=22, alpha=0.65, edgecolor="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1.2, alpha=0.85)
    ax.set_title(f"{model_name}: {draft_year} Draft Class", fontsize=13, pad=10)
    ax.set_xlabel(f"Actual 2-Year AV  (AV{draft_year} + AV{draft_year+1})", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)
    ax.text(
        0.02, 0.98,
        f"MAE={mae:.2f}  RMSE={rmse:.2f}  Spearman={sp:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, edgecolor="0.85"),
    )
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_importance(importance: np.ndarray, feature_names: list[str], out_path: str):
    """Bar chart of top-20 TabNet feature importances."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading draft:", DRAFT_CSV)
    draft = load_draft(DRAFT_CSV)

    print("Loading AVs from:", AV_DIR)
    av_long = load_av_from_year_files(AV_DIR)
    labels = build_two_year_labels(av_long)

    if "pfr_player_id" not in draft.columns:
        raise ValueError("Draft CSV missing pfr_player_id.")

    df = draft.rename(columns={"season": "draft_season"}).merge(
        labels, on=["draft_season", "pfr_player_id"], how="left"
    )

    total, labeled = len(df), df["av_2yr"].notna().sum()
    print(f"Join coverage: {labeled}/{total} = {labeled/total:.1%}")

    df = df.dropna(subset=["av_2yr"]).copy()
    df["av_2yr"] = df["av_2yr"].astype(float)

    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    feat_cols = num_cols + cat_cols

    X_raw = df[feat_cols].copy()
    y = df["av_2yr"].values
    years = df["draft_season"].values

    unique_years = sorted(np.unique(years))
    if len(unique_years) < 2:
        raise ValueError(f"Not enough labeled years. Found: {unique_years}")

    results = []
    latest_year = max(unique_years)

    for test_year in unique_years[1:]:
        train_mask = years < test_year
        test_mask = years == test_year

        X_tr_raw = X_raw[train_mask]
        X_te_raw = X_raw[test_mask]
        y_tr = y[train_mask]
        y_te = y[test_mask]

        # Build and fit preprocessor on training data only
        preprocessor, _num, _cat = build_preprocessor(X_tr_raw)
        X_tr = preprocessor.fit_transform(X_tr_raw).astype(np.float32)
        X_te = preprocessor.transform(X_te_raw).astype(np.float32)
        feature_names = get_feature_names(preprocessor, _num, _cat)
        n_features = X_tr.shape[1]

        print(f"\n--- Test year {test_year}  train={train_mask.sum()}  "
              f"test={test_mask.sum()}  features={n_features} ---")

        model, train_losses, val_losses = train_tabnet(
            X_tr, y_tr, n_features,
            tabnet_cfg=TABNET_CFG,
            train_cfg=TRAIN_CFG,
            device=DEVICE,
            verbose=True,
        )

        # Predict
        model.eval()
        with torch.no_grad():
            X_te_t = to_tensor(X_te).to(DEVICE)
            pred_raw, _, masks_te = model(X_te_t)
            pred = pred_raw.cpu().numpy()
            importance = model.encoder.aggregate_importance(masks_te).cpu().numpy()

        mae, rmse, sp = eval_metrics(y_te, pred)
        print(f"  TabNet  MAE={mae:.3f}  RMSE={rmse:.3f}  Spearman={sp:.3f}")

        results.append({
            "test_year": int(test_year),
            "n_test": int(test_mask.sum()),
            "tabnet_mae": float(mae),
            "tabnet_rmse": float(rmse),
            "tabnet_spearman": float(sp),
        })

        # Plots for the latest test year
        if test_year == latest_year:
            save_pred_vs_actual(
                y_te, pred,
                os.path.join(OUT_DIR, f"plot_pred_vs_actual_tabnet_{test_year}.png"),
                draft_year=test_year,
            )
            save_feature_importance(
                importance, feature_names,
                os.path.join(OUT_DIR, "plot_feature_importance_tabnet.png"),
            )
            save_loss_curves(
                train_losses, val_losses,
                os.path.join(OUT_DIR, f"plot_tabnet_loss_curves_{test_year}.png"),
                test_year=test_year,
            )

    res_df = pd.DataFrame(results)
    print("\nWalk-forward results:")
    print(res_df.to_string(index=False))

    out_csv = os.path.join(OUT_DIR, "tabnet_walkforward_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results CSV: {out_csv}")
    print(f"Saved plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
