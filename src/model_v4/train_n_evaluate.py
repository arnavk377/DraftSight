"""
Model v4: presentation-ready walk-forward evaluation with college features.

Builds on main branch model_v3 and adds:
  - pick + college-performance bin baseline
  - 2x2 latest-year predicted-vs-actual plot
  - overall summary + selected-years summary CSVs
  - walk-forward metric grids, residual diagnostics, loss curves, and value curves

Inputs:
  - src/data/raw/draft_picks.csv
  - data/clean_cfb/05_04_all_players_2004_2024.csv
  - scraping_av/data/*_av.csv

Outputs:
  - poc_outputs_v4/
"""

import os
from collections.abc import Iterable

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
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.model_v3.data_loader import (
    CFB_STAT_COLS,
    DRAFT_CAT_COLS,
    DRAFT_NUM_COLS,
    build_two_year_labels,
    join_cfb_to_draft,
    load_av_from_year_files,
    load_cfb,
    load_draft,
)
from src.model_v3.tabnet import TabNetRegressor


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRAFT_CSV = os.path.join(REPO_ROOT, "src", "data", "raw", "draft_picks.csv")
CFB_CSV = os.path.join(REPO_ROOT, "data", "clean_cfb", "05_04_all_players_2004_2024.csv")
AV_DIR = os.path.join(REPO_ROOT, "scraping_av", "data")
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v4")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

JOIN_ON_COLLEGE = False
USE_FUZZY = False
FUZZY_THRESHOLD = 85

# The joined CFB feature coverage becomes meaningfully usable starting in 2010.
# Before that, exact name matches are close to zero, so those folds behave like
# draft-only models and muddy the college-feature story.
WALK_FORWARD_START_YEAR = 2009
PICK_BIN_SIZE = 16
PERF_BIN_COUNT = 4
SELECTED_SUMMARY_YEARS = [2010, 2015, 2020, 2024]

MODEL_ORDER = ["pick_cfb_bin", "spline", "xgb", "tabnet"]
MODEL_LABELS = {
    "pick_cfb_bin": "Pick + College Bin",
    "spline": "Spline Ridge",
    "xgb": "XGBoost",
    "tabnet": "TabNet",
}
MODEL_COLORS = {
    "pick_cfb_bin": "#3B6FB6",
    "spline": "#D77A61",
    "xgb": "#2A9D8F",
    "tabnet": "#7A4EAB",
}

MODEL_NUM_COLS = DRAFT_NUM_COLS + ["college_perf_score", "cfb_matched"] + CFB_STAT_COLS
MODEL_CAT_COLS = DRAFT_CAT_COLS

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
    lr=0.01,
    lr_decay=0.96,
    lr_decay_steps=200,
    batch_size=256,
    max_epochs=300,
    patience=40,
    val_fraction=0.15,
)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_requested_test_years(all_years: list[int]) -> list[int]:
    requested = os.getenv("MODEL_V4_TEST_YEAR_LIST", "").strip()
    if requested:
        keep = {int(part.strip()) for part in requested.split(",") if part.strip()}
        return [year for year in all_years if year in keep]

    year_min = os.getenv("MODEL_V4_TEST_YEAR_MIN", "").strip()
    year_max = os.getenv("MODEL_V4_TEST_YEAR_MAX", "").strip()
    lo = int(year_min) if year_min else min(all_years)
    hi = int(year_max) if year_max else max(all_years)
    return [year for year in all_years if lo <= year <= hi]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    spearman = spearmanr(y_true, y_pred).correlation
    if pd.isna(spearman):
        spearman = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearman),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _nice_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return series.dropna().iloc[0] if series.notna().any() else "Unknown"


class CollegePerformanceScorer:
    """Create a simple position-aware college performance score from CFB stats.

    For each position, compute percentile ranks across "active" college stat columns
    and average the available percentiles for a 0-to-1 score.
    """

    def __init__(
        self,
        feature_cols: Iterable[str],
        group_col: str = "position",
        min_group_size: int = 25,
        min_nonmissing: int = 15,
        min_nonzero_rate: float = 0.05,
    ):
        self.feature_cols = list(feature_cols)
        self.group_col = group_col
        self.min_group_size = min_group_size
        self.min_nonmissing = min_nonmissing
        self.min_nonzero_rate = min_nonzero_rate
        self.group_schemas_: dict[str, dict] = {}
        self.global_schema_: dict | None = None
        self.feature_cols_: list[str] = []

    def fit(self, df: pd.DataFrame):
        self.feature_cols_ = [col for col in self.feature_cols if col in df.columns]
        self.global_schema_ = self._fit_schema(df)
        self.group_schemas_ = {}
        if self.group_col in df.columns:
            for group_value, group_df in df.groupby(self.group_col):
                if len(group_df) < self.min_group_size:
                    continue
                schema = self._fit_schema(group_df)
                if schema["active_cols"]:
                    self.group_schemas_[str(group_value)] = schema
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.global_schema_ is None:
            raise ValueError("CollegePerformanceScorer must be fit before transform.")

        scores = np.full(len(df), 0.5, dtype=float)
        if not self.feature_cols_:
            return scores

        for pos, (_, row) in enumerate(df.iterrows()):
            group_schema = None
            if self.group_col in df.columns:
                group_schema = self.group_schemas_.get(str(row.get(self.group_col)))

            score = self._row_score(row, group_schema) if group_schema else np.nan
            if pd.isna(score):
                score = self._row_score(row, self.global_schema_)
            scores[pos] = 0.5 if pd.isna(score) else score
        return scores

    def _fit_schema(self, df: pd.DataFrame) -> dict:
        active_cols = []
        sorted_values = {}
        for col in self.feature_cols_:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) < self.min_nonmissing or values.nunique() < 2:
                continue

            nonzero_rate = float((values != 0).mean())
            if col != "career_years" and nonzero_rate < self.min_nonzero_rate and values.nunique() < 8:
                continue

            active_cols.append(col)
            sorted_values[col] = np.sort(values.to_numpy(dtype=float))

        return {
            "active_cols": active_cols,
            "sorted_values": sorted_values,
        }

    def _row_score(self, row: pd.Series, schema: dict | None) -> float:
        if not schema or not schema["active_cols"]:
            return np.nan

        percentiles = []
        for col in schema["active_cols"]:
            value = row.get(col)
            if pd.isna(value):
                continue
            arr = schema["sorted_values"][col]
            pct = float(np.searchsorted(arr, float(value), side="right") / len(arr))
            percentiles.append(pct)

        if not percentiles:
            return np.nan
        return float(np.mean(percentiles))


def make_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for idx in range(1, len(edges) - 1):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-6
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges[1:-1], right=False).astype(int)


class PickCollegeBinBaseline:
    """Simple hierarchical baseline using pick bins and college performance bins."""

    def __init__(self, pick_bin_size: int = 16, perf_bin_count: int = 4):
        self.pick_bin_size = pick_bin_size
        self.perf_bin_count = perf_bin_count
        self.global_mean_: float | None = None
        self.scorer_: CollegePerformanceScorer | None = None
        self.perf_bin_edges_: np.ndarray | None = None
        self.reference_scores_: np.ndarray | None = None
        self.position_means_: dict[tuple[int, str, int], float] = {}
        self.category_means_: dict[tuple[int, str, int], float] = {}
        self.pick_perf_means_: dict[tuple[int, int], float] = {}
        self.pick_means_: dict[int, float] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.scorer_ = CollegePerformanceScorer(CFB_STAT_COLS).fit(X)
        scores = self.scorer_.transform(X)
        self.perf_bin_edges_ = make_quantile_edges(scores, self.perf_bin_count)
        self.reference_scores_ = np.quantile(scores, [0.2, 0.5, 0.8])
        perf_bins = assign_bins(scores, self.perf_bin_edges_)

        baseline_df = pd.DataFrame(
            {
                "pick_bin": self._pick_bin_id(X["pick"].to_numpy(dtype=int)),
                "perf_bin": perf_bins,
                "target": np.asarray(y, dtype=float),
                "position": X["position"].astype(str).values if "position" in X.columns else "UNK",
                "category": X["category"].astype(str).values if "category" in X.columns else "UNK",
            }
        )

        self.global_mean_ = float(baseline_df["target"].mean())
        self.position_means_ = baseline_df.groupby(["pick_bin", "position", "perf_bin"])["target"].mean().to_dict()
        self.category_means_ = baseline_df.groupby(["pick_bin", "category", "perf_bin"])["target"].mean().to_dict()
        self.pick_perf_means_ = baseline_df.groupby(["pick_bin", "perf_bin"])["target"].mean().to_dict()
        self.pick_means_ = baseline_df.groupby("pick_bin")["target"].mean().to_dict()
        return self

    def score_features(self, X: pd.DataFrame) -> np.ndarray:
        if self.scorer_ is None:
            raise ValueError("Baseline must be fit before calling score_features.")
        return self.scorer_.transform(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None or self.perf_bin_edges_ is None:
            raise ValueError("Baseline must be fit before predict.")

        scores = self.score_features(X)
        perf_bins = assign_bins(scores, self.perf_bin_edges_)
        pick_bins = self._pick_bin_id(X["pick"].to_numpy(dtype=int))
        positions = X["position"].astype(str).to_numpy() if "position" in X.columns else np.repeat("UNK", len(X))
        categories = X["category"].astype(str).to_numpy() if "category" in X.columns else np.repeat("UNK", len(X))

        preds = []
        for pick_bin, position, category, perf_bin in zip(pick_bins, positions, categories, perf_bins):
            pred = self.position_means_.get((pick_bin, position, perf_bin))
            if pred is None:
                pred = self.category_means_.get((pick_bin, category, perf_bin))
            if pred is None:
                pred = self.pick_perf_means_.get((pick_bin, perf_bin))
            if pred is None:
                pred = self.pick_means_.get(pick_bin)
            if pred is None:
                pred = self.global_mean_
            preds.append(float(pred))

        return np.asarray(preds, dtype=float)

    def _pick_bin_id(self, picks: np.ndarray) -> np.ndarray:
        return ((np.asarray(picks, dtype=int) - 1) // self.pick_bin_size).astype(int)


def add_engineered_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, PickCollegeBinBaseline]:
    baseline = PickCollegeBinBaseline(pick_bin_size=PICK_BIN_SIZE, perf_bin_count=PERF_BIN_COUNT)
    baseline.fit(train_df, train_df["av_2yr"].to_numpy(dtype=float))

    train_aug = train_df.copy()
    test_aug = test_df.copy()
    train_aug["college_perf_score"] = baseline.score_features(train_df)
    test_aug["college_perf_score"] = baseline.score_features(test_df)

    cfb_cols = [col for col in CFB_STAT_COLS if col in train_df.columns]
    if cfb_cols:
        train_aug["cfb_matched"] = train_df[cfb_cols].notna().any(axis=1).astype(float)
        test_aug["cfb_matched"] = test_df[cfb_cols].notna().any(axis=1).astype(float)
    else:
        train_aug["cfb_matched"] = 0.0
        test_aug["cfb_matched"] = 0.0

    return train_aug, test_aug, baseline


def build_spline_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    spline_num_pool = DRAFT_NUM_COLS + ["college_perf_score", "cfb_matched"]
    num_cols = [col for col in spline_num_pool if col in df.columns and df[col].notna().any()]
    cat_cols = [col for col in MODEL_CAT_COLS if col in df.columns]

    spline_cols = [col for col in ["pick", "college_perf_score"] if col in num_cols]
    rest_num_cols = [col for col in num_cols if col not in spline_cols]

    transformers = []
    for col in spline_cols:
        transformers.append(
            (
                f"{col}_spline",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("spl", SplineTransformer(n_knots=6 if col == "pick" else 5, degree=3, include_bias=False)),
                    ]
                ),
                [col],
            )
        )
    if rest_num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                rest_num_cols,
            )
        )
    if cat_cols:
        transformers.append(("cat", make_one_hot_encoder(), cat_cols))

    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_xgb_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [col for col in MODEL_NUM_COLS if col in df.columns and df[col].notna().any()]
    cat_cols = [col for col in MODEL_CAT_COLS if col in df.columns]

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", make_one_hot_encoder(), cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_tabnet_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [col for col in MODEL_NUM_COLS if col in df.columns and df[col].notna().any()]
    cat_cols = [col for col in MODEL_CAT_COLS if col in df.columns]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(("cat", make_one_hot_encoder(), cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def get_feature_names(preprocessor: ColumnTransformer, num_cols: list[str], cat_cols: list[str]) -> list[str]:
    names = list(num_cols)
    if cat_cols and "cat" in preprocessor.named_transformers_:
        names += list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    return names


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    verbose: bool = True,
) -> tuple[TabNetRegressor, list[float], list[float]]:
    model = TabNetRegressor(n_features=n_features, **TABNET_CFG).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])
    scheduler = ExponentialLR(optimizer, gamma=TRAIN_CFG["lr_decay"])

    n_val = max(1, int(len(X_train) * TRAIN_CFG["val_fraction"]))
    X_t, X_v = X_train[:-n_val], X_train[-n_val:]
    y_t, y_v = y_train[:-n_val], y_train[-n_val:]

    batch_size = min(TRAIN_CFG["batch_size"], max(1, len(X_t)))
    loader = DataLoader(
        TensorDataset(to_tensor(X_t), to_tensor(y_t)),
        batch_size=batch_size,
        shuffle=True,
    )

    X_val_t = to_tensor(X_v).to(DEVICE)
    y_val_t = to_tensor(y_v).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    step = 0

    for epoch in range(TRAIN_CFG["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred, sp_loss, _ = model(X_batch)
            loss = model.loss(pred, y_batch, sp_loss)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            valid_batches += 1
            step += 1
            if step % TRAIN_CFG["lr_decay_steps"] == 0:
                scheduler.step()

        if valid_batches == 0:
            train_losses.append(np.nan)
            val_losses.append(np.nan)
            patience_counter += 1
            if patience_counter >= TRAIN_CFG["patience"]:
                break
            continue

        train_losses.append(epoch_loss / valid_batches)

        model.eval()
        with torch.no_grad():
            v_pred, v_sp, _ = model(X_val_t)
            val_loss = float(model.loss(v_pred, y_val_t, v_sp).item())
        if not np.isfinite(val_loss):
            val_loss = float("inf")
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: param.cpu().clone() for name, param in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"  epoch {epoch + 1:4d}  train={train_losses[-1]:.3f}  "
                f"val={val_loss:.3f}  best_val={best_val_loss:.3f}  "
                f"patience={patience_counter}/{TRAIN_CFG['patience']}"
            )

        if patience_counter >= TRAIN_CFG["patience"]:
            if verbose:
                print(f"  Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def save_pred_vs_actual_grid(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_path: str,
    draft_year: int,
):
    y_true = np.asarray(y_true, dtype=float)
    lo = float(min(np.min(y_true), min(np.min(pred) for pred in predictions.values())))
    hi = float(max(np.max(y_true), max(np.max(pred) for pred in predictions.values())))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=220, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, model_name in zip(axes, MODEL_ORDER):
        pred = np.asarray(predictions[model_name], dtype=float)
        metrics = eval_metrics(y_true, pred)
        ax.scatter(
            y_true,
            pred,
            s=24,
            alpha=0.68,
            edgecolor="none",
            color=MODEL_COLORS[model_name],
        )
        ax.plot([lo, hi], [lo, hi], color="#2F3E46", linewidth=1.2, alpha=0.85)
        ax.set_title(MODEL_LABELS[model_name], fontsize=13, pad=10)
        ax.text(
            0.03,
            0.97,
            (
                f"MAE={metrics['mae']:.2f}\n"
                f"RMSE={metrics['rmse']:.2f}\n"
                f"R2={metrics['r2']:.2f}\n"
                f"Spearman={metrics['spearman']:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="0.85"),
        )
        _nice_axes(ax)

    fig.suptitle(f"{draft_year} Draft Class: Predicted vs Actual 2-Year AV", fontsize=16, y=0.98)
    fig.supxlabel(f"Actual 2-Year AV = AV({draft_year}) + AV({draft_year + 1})", fontsize=12)
    fig.supylabel("Predicted 2-Year AV", fontsize=12)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def save_walkforward_metric_grid(results_df: pd.DataFrame, out_path: str):
    metric_labels = {
        "mae": "MAE",
        "rmse": "RMSE",
        "spearman": "Spearman",
        "r2": "R2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220, sharex=True)
    axes = axes.flatten()

    for ax, metric_name in zip(axes, metric_labels):
        for model_name in MODEL_ORDER:
            ax.plot(
                results_df["test_year"],
                results_df[f"{model_name}_{metric_name}"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
                alpha=0.95,
            )
        ax.set_title(metric_labels[metric_name], fontsize=13, pad=8)
        _nice_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=9, frameon=False, ncol=2)
    fig.suptitle("Walk-Forward Metrics by Draft Year", fontsize=16, y=0.98)
    fig.supxlabel("Test draft year", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def save_overall_summary_grid(summary_df: pd.DataFrame, out_path: str):
    metric_labels = {
        "mae": "Lower is better",
        "rmse": "Lower is better",
        "spearman": "Higher is better",
        "r2": "Higher is better",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220)
    axes = axes.flatten()
    x = np.arange(len(summary_df))
    display_names = [MODEL_LABELS[name] for name in summary_df["model_type"]]
    colors = [MODEL_COLORS[name] for name in summary_df["model_type"]]

    for ax, metric_name in zip(axes, metric_labels):
        vals = summary_df[metric_name].to_numpy(dtype=float)
        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=18, ha="right")
        ax.set_title(f"{metric_name.upper()} ({metric_labels[metric_name]})", fontsize=13, pad=8)
        for idx, val in enumerate(vals):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        _nice_axes(ax)

    fig.suptitle("Overall Out-of-Sample Summary", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_importance(importance: np.ndarray, feature_names: list[str], out_path: str, title: str):
    top_k = min(20, len(feature_names))
    idx = np.argsort(importance)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        importance[list(reversed(idx))],
        color="#4C72B0",
    )
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_loss_curves(train_losses: list[float], val_losses: list[float], out_path: str, draft_year: int):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    ax.plot(train_losses, label="Train", linewidth=1.5, color="#2A9D8F")
    ax.plot(val_losses, label="Validation", linewidth=1.5, color="#D77A61")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"TabNet Training Curves — Latest Test Year {draft_year}", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_latest_year_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    picks: np.ndarray,
    out_path: str,
    draft_year: int,
    model_name: str,
):
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    picks = np.asarray(picks, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220)
    ax0, ax1, ax2, ax3 = axes.flatten()

    ax0.scatter(y_pred, residual, s=22, alpha=0.65, color=MODEL_COLORS[model_name], edgecolor="none")
    ax0.axhline(0.0, color="#2F3E46", linewidth=1.0)
    ax0.set_title("Residual vs Predicted", fontsize=12)
    ax0.set_xlabel("Predicted AV")
    ax0.set_ylabel("Actual - Predicted")
    _nice_axes(ax0)

    ax1.hist(residual, bins=18, color=MODEL_COLORS[model_name], alpha=0.85)
    ax1.axvline(0.0, color="#2F3E46", linewidth=1.0)
    ax1.set_title("Residual Distribution", fontsize=12)
    ax1.set_xlabel("Actual - Predicted")
    _nice_axes(ax1)

    calib_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).sort_values("y_pred")
    calib_df["decile"] = pd.qcut(calib_df["y_pred"], q=min(10, len(calib_df)), labels=False, duplicates="drop")
    calib = calib_df.groupby("decile", as_index=False).agg(
        actual=("y_true", "mean"),
        predicted=("y_pred", "mean"),
    )
    ax2.plot(calib["decile"], calib["actual"], marker="o", linewidth=2.0, label="Actual", color="#2A9D8F")
    ax2.plot(
        calib["decile"],
        calib["predicted"],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        label="Predicted",
        color="#D77A61",
    )
    ax2.set_title("Calibration by Prediction Decile", fontsize=12)
    ax2.set_xlabel("Prediction decile")
    ax2.set_ylabel("Mean 2-Year AV")
    ax2.legend(frameon=False, fontsize=9)
    _nice_axes(ax2)

    error_df = pd.DataFrame(
        {
            "pick_bin": ((picks - 1) // PICK_BIN_SIZE).astype(int) + 1,
            "abs_error": np.abs(residual),
        }
    )
    by_bin = error_df.groupby("pick_bin", as_index=False)["abs_error"].mean()
    ax3.plot(by_bin["pick_bin"], by_bin["abs_error"], marker="o", linewidth=2.0, color="#3B6FB6")
    ax3.set_title(f"Mean Absolute Error by {PICK_BIN_SIZE}-Pick Bin", fontsize=12)
    ax3.set_xlabel("Pick bin")
    ax3.set_ylabel("Mean absolute error")
    _nice_axes(ax3)

    fig.suptitle(
        f"{draft_year} Latest-Year Diagnostics — {MODEL_LABELS[model_name]}",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def make_reference_player_grid(
    train_df: pd.DataFrame,
    picks: np.ndarray,
    perf_score: float,
) -> pd.DataFrame:
    grid = pd.DataFrame({"pick": picks})
    for col in train_df.columns:
        if col == "pick":
            continue
        if col in MODEL_CAT_COLS:
            grid[col] = _safe_mode(train_df[col])
        else:
            vals = pd.to_numeric(train_df[col], errors="coerce")
            grid[col] = float(vals.median()) if vals.notna().any() else 0.0

    if "college_perf_score" in grid.columns:
        grid["college_perf_score"] = perf_score
    if "cfb_matched" in grid.columns:
        grid["cfb_matched"] = 1.0
    return grid


def get_curve_reference_scores(train_df: pd.DataFrame) -> np.ndarray:
    score_series = train_df["college_perf_score"].dropna()
    if "cfb_matched" in train_df.columns:
        matched_scores = train_df.loc[train_df["cfb_matched"] > 0.5, "college_perf_score"].dropna()
        if len(matched_scores) >= 20:
            score_series = matched_scores

    if score_series.empty:
        return np.array([0.25, 0.5, 0.75], dtype=float)

    candidates = [
        np.quantile(score_series, [0.2, 0.5, 0.8]),
        np.quantile(score_series, [0.1, 0.5, 0.9]),
        np.array([score_series.min(), score_series.median(), score_series.max()], dtype=float),
    ]
    for arr in candidates:
        if len(np.unique(np.round(arr, 6))) == 3:
            return np.asarray(arr, dtype=float)
    return np.asarray(candidates[-1], dtype=float)


def build_curve_tiers(curve_reference_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    curve_df = curve_reference_df.copy()
    if curve_df.empty:
        raise ValueError("Need at least one matched row to build curve tiers.")

    curve_df["curve_tier"] = pd.qcut(
        curve_df["college_perf_score"],
        q=3,
        labels=False,
        duplicates="drop",
    )
    if curve_df["curve_tier"].nunique() < 3:
        ranked = curve_df["college_perf_score"].rank(method="first")
        curve_df["curve_tier"] = pd.qcut(
            ranked,
            q=3,
            labels=False,
            duplicates="drop",
        )
    tier_scores = (
        curve_df.groupby("curve_tier", as_index=True)["college_perf_score"]
        .median()
        .sort_index()
        .to_numpy(dtype=float)
    )
    return curve_df, tier_scores


def save_pick_value_curve_grid(
    baseline_model: PickCollegeBinBaseline,
    spline_model: Pipeline,
    train_df: pd.DataFrame,
    out_path: str,
    draft_year: int,
):
    max_pick = 256
    picks = np.arange(1, max_pick + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=220, sharey=True)
    labels = ["Lower college score", "Median college score", "Higher college score"]
    line_colors = ["#457B9D", "#E9C46A", "#E76F51"]

    bin_ids = np.arange(0, int(np.ceil(max_pick / PICK_BIN_SIZE)))
    bin_centers = bin_ids * PICK_BIN_SIZE + (PICK_BIN_SIZE + 1) / 2
    bin_starts = bin_ids * PICK_BIN_SIZE + 1
    curve_reference_df = train_df.loc[train_df["cfb_matched"] > 0.5].copy() if "cfb_matched" in train_df.columns else train_df
    if curve_reference_df.empty:
        curve_reference_df = train_df
    curve_reference_df, ref_scores = build_curve_tiers(curve_reference_df)
    curve_reference_df["pick_bin"] = ((curve_reference_df["pick"].to_numpy(dtype=int) - 1) // PICK_BIN_SIZE).astype(int)
    pick_bin_mean = (
        train_df.assign(pick_bin=((train_df["pick"].to_numpy(dtype=int) - 1) // PICK_BIN_SIZE).astype(int))
        .groupby("pick_bin", as_index=True)["av_2yr"]
        .mean()
        .to_dict()
    )
    tier_stats = (
        curve_reference_df.groupby(["pick_bin", "curve_tier"], as_index=False)
        .agg(mean_av=("av_2yr", "mean"), n_players=("av_2yr", "size"))
    )
    tier_map = {
        (int(row["pick_bin"]), int(row["curve_tier"])): (float(row["mean_av"]), int(row["n_players"]))
        for _, row in tier_stats.iterrows()
    }

    for tier_id, perf_score, label, color in zip(sorted(curve_reference_df["curve_tier"].unique()), ref_scores, labels, line_colors):
        base_curve = []
        for pick_bin in bin_ids:
            tier_val = tier_map.get((int(pick_bin), int(tier_id)))
            if tier_val is not None and tier_val[1] >= 5:
                pred = tier_val[0]
            else:
                pred = pick_bin_mean.get(int(pick_bin), baseline_model.global_mean_)
            base_curve.append(float(pred))

        axes[0].step(bin_starts, base_curve, where="post", label=label, linewidth=2.0, color=color, alpha=0.95)
        axes[0].scatter(bin_centers, base_curve, s=18, color=color, alpha=0.95)

        grid = make_reference_player_grid(curve_reference_df, picks, float(perf_score))
        pred_spline = spline_model.predict(grid)
        axes[1].plot(picks, pred_spline, label=label, linewidth=2.0, color=color)

    axes[0].set_title("Empirical Pick + College Baseline", fontsize=13, pad=10)
    axes[1].set_title("Spline Value Curve", fontsize=13, pad=10)
    for ax in axes:
        ax.set_xlabel("Pick number", fontsize=11)
        _nice_axes(ax)
    axes[0].set_ylabel("Predicted 2-Year AV", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].text(
        0.03,
        0.04,
        f"Baseline shown in {PICK_BIN_SIZE}-pick steps",
        transform=axes[0].transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.suptitle(f"Latest Training Window Through {draft_year - 1}", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def build_model_frame() -> pd.DataFrame:
    print("Loading draft:", DRAFT_CSV)
    draft = load_draft(DRAFT_CSV)

    print("Loading CFB:", CFB_CSV)
    cfb = load_cfb(CFB_CSV)

    print("Loading AVs:", AV_DIR)
    av_long = load_av_from_year_files(AV_DIR)
    labels = build_two_year_labels(av_long)

    merged, join_stats = join_cfb_to_draft(
        draft,
        cfb,
        join_on_college=JOIN_ON_COLLEGE,
        use_fuzzy=USE_FUZZY,
        fuzzy_threshold=FUZZY_THRESHOLD,
    )
    print(f"Join stats: {join_stats}")

    df = merged.rename(columns={"season": "draft_season"}).merge(
        labels,
        on=["draft_season", "pfr_player_id"],
        how="left",
    )
    total_rows = len(df)
    labeled_rows = int(df["av_2yr"].notna().sum())
    print(f"AV label coverage: {labeled_rows}/{total_rows} = {labeled_rows / total_rows:.1%}")

    df = df.dropna(subset=["av_2yr"]).copy()
    df["av_2yr"] = df["av_2yr"].astype(float)
    return df


def create_selected_years_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in SELECTED_SUMMARY_YEARS:
        year_df = results_df[results_df["test_year"] == year]
        if year_df.empty:
            continue
        row = year_df.iloc[0]
        for model_name in MODEL_ORDER:
            rows.append(
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
    return pd.DataFrame(rows)


def create_overall_summary(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        model_df = overall_df[overall_df["model_type"] == model_name]
        metrics = eval_metrics(model_df["av_2yr"].values, model_df["prediction"].values)
        rows.append(
            {
                "model_type": model_name,
                "n_predictions": int(len(model_df)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main():
    df = build_model_frame()
    years = df["draft_season"].to_numpy(dtype=int)
    unique_years = sorted(np.unique(years))
    candidate_test_years = [year for year in unique_years if year > WALK_FORWARD_START_YEAR]
    test_years = parse_requested_test_years(candidate_test_years)
    if not test_years:
        raise ValueError("No test years selected for model_v4.")

    print(f"Using test years: {test_years[0]}-{test_years[-1]} ({len(test_years)} years)")

    results = []
    overall_rows = []
    latest_year = max(test_years)
    latest_artifacts = {}

    for test_year in test_years:
        # Use all prior labeled years for training so early folds have enough
        # history, while keeping the evaluation window itself in the college-data era.
        train_mask = years < test_year
        test_mask = years == test_year

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            continue

        train_aug, test_aug, baseline_model = add_engineered_features(train_df, test_df)
        y_train = train_aug["av_2yr"].to_numpy(dtype=float)
        y_test = test_aug["av_2yr"].to_numpy(dtype=float)

        print(
            f"\n=== Test year {test_year}  train={len(train_aug)}  test={len(test_aug)}  "
            f"cfb_match_train={train_aug['cfb_matched'].mean():.1%}  cfb_match_test={test_aug['cfb_matched'].mean():.1%} ==="
        )

        feat_cols = [col for col in MODEL_NUM_COLS + MODEL_CAT_COLS if col in train_aug.columns]
        X_train = train_aug[feat_cols].copy()
        X_test = test_aug[feat_cols].copy()

        # Pick + college bin baseline
        pred_pick = baseline_model.predict(X_test)

        # Spline model
        spline_pre, _, _ = build_spline_preprocessor(X_train)
        spline_model = Pipeline([("pre", spline_pre), ("reg", Ridge(alpha=4.0))])
        spline_model.fit(X_train, y_train)
        pred_spline = spline_model.predict(X_test)

        # XGBoost
        xgb_pre, xgb_num_cols, xgb_cat_cols = build_xgb_preprocessor(X_train)
        xgb_model = Pipeline(
            [
                ("pre", xgb_pre),
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_weight=2,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        xgb_model.fit(X_train, y_train)
        pred_xgb = xgb_model.predict(X_test)

        # TabNet
        tabnet_pre, tabnet_num_cols, tabnet_cat_cols = build_tabnet_preprocessor(X_train)
        X_train_tabnet = tabnet_pre.fit_transform(X_train).astype(np.float32)
        X_test_tabnet = tabnet_pre.transform(X_test).astype(np.float32)
        np.nan_to_num(X_train_tabnet, copy=False)
        np.nan_to_num(X_test_tabnet, copy=False)
        tabnet_feature_names = get_feature_names(tabnet_pre, tabnet_num_cols, tabnet_cat_cols)

        tabnet_model, train_losses, val_losses = train_tabnet(
            X_train_tabnet,
            y_train,
            n_features=X_train_tabnet.shape[1],
            verbose=True,
        )
        tabnet_model.eval()
        with torch.no_grad():
            pred_raw, _, masks = tabnet_model(to_tensor(X_test_tabnet).to(DEVICE))
            pred_tabnet = pred_raw.cpu().numpy()
            tabnet_importance = tabnet_model.encoder.aggregate_importance(masks).cpu().numpy()

        year_result = {
            "test_year": int(test_year),
            "n_train": int(len(train_aug)),
            "n_test": int(len(test_aug)),
            "cfb_match_train_rate": float(train_aug["cfb_matched"].mean()),
            "cfb_match_test_rate": float(test_aug["cfb_matched"].mean()),
        }

        prediction_map = {
            "pick_cfb_bin": pred_pick,
            "spline": pred_spline,
            "xgb": pred_xgb,
            "tabnet": pred_tabnet,
        }

        for model_name in MODEL_ORDER:
            metrics = eval_metrics(y_test, prediction_map[model_name])
            print(
                f"  {MODEL_LABELS[model_name]:18s} "
                f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
                f"R2={metrics['r2']:.3f}  Spearman={metrics['spearman']:.3f}"
            )
            for metric_name, metric_value in metrics.items():
                year_result[f"{model_name}_{metric_name}"] = metric_value

            pred_frame = test_aug[
                [
                    "draft_season",
                    "pick",
                    "round",
                    "team",
                    "position",
                    "category",
                    "college",
                    "pfr_player_name",
                    "college_perf_score",
                    "cfb_matched",
                    "av_2yr",
                ]
            ].copy()
            pred_frame["model_type"] = model_name
            pred_frame["prediction"] = prediction_map[model_name]
            pred_frame["residual"] = pred_frame["av_2yr"] - pred_frame["prediction"]
            overall_rows.append(pred_frame)

        results.append(year_result)

        if test_year == latest_year:
            xgb_feature_names = get_feature_names(
                xgb_model.named_steps["pre"],
                xgb_num_cols,
                xgb_cat_cols,
            )
            latest_artifacts = {
                "test_year": test_year,
                "y_test": y_test,
                "test_aug": test_aug.copy(),
                "prediction_map": prediction_map,
                "baseline_model": baseline_model,
                "spline_model": spline_model,
                "train_aug": train_aug.copy(),
                "xgb_importance": xgb_model.named_steps["xgb"].feature_importances_,
                "xgb_feature_names": xgb_feature_names,
                "tabnet_importance": tabnet_importance,
                "tabnet_feature_names": tabnet_feature_names,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }

    results_df = pd.DataFrame(results)
    overall_df = pd.concat(overall_rows, ignore_index=True)
    selected_df = create_selected_years_summary(results_df)
    summary_df = create_overall_summary(overall_df)

    results_path = os.path.join(OUT_DIR, "model_v4_walkforward_results.csv")
    overall_path = os.path.join(OUT_DIR, "model_v4_overall_summary.csv")
    selected_path = os.path.join(OUT_DIR, "model_v4_selected_years_summary.csv")
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(overall_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    if latest_artifacts:
        year = latest_artifacts["test_year"]
        y_test = latest_artifacts["y_test"]
        prediction_map = latest_artifacts["prediction_map"]
        latest_pred_df = latest_artifacts["test_aug"][
            [
                "draft_season",
                "pick",
                "round",
                "team",
                "position",
                "category",
                "college",
                "pfr_player_name",
                "college_perf_score",
                "cfb_matched",
                "av_2yr",
            ]
        ].copy()
        for model_name in MODEL_ORDER:
            latest_pred_df[f"pred_{model_name}"] = prediction_map[model_name]
        latest_pred_path = os.path.join(OUT_DIR, f"model_v4_latest_year_predictions_{year}.csv")
        latest_pred_df.to_csv(latest_pred_path, index=False)

        save_pred_vs_actual_grid(
            y_test,
            prediction_map,
            os.path.join(OUT_DIR, f"plot_pred_vs_actual_grid_{year}.png"),
            year,
        )
        save_walkforward_metric_grid(
            results_df,
            os.path.join(OUT_DIR, "plot_walkforward_metric_grid_v4.png"),
        )
        save_overall_summary_grid(
            summary_df,
            os.path.join(OUT_DIR, "plot_overall_summary_grid_v4.png"),
        )
        save_feature_importance(
            latest_artifacts["xgb_importance"],
            latest_artifacts["xgb_feature_names"],
            os.path.join(OUT_DIR, "plot_xgb_feature_importance_v4.png"),
            title=f"XGBoost Feature Importance — trained through {year - 1}",
        )
        save_feature_importance(
            latest_artifacts["tabnet_importance"],
            latest_artifacts["tabnet_feature_names"],
            os.path.join(OUT_DIR, "plot_tabnet_feature_importance_v4.png"),
            title=f"TabNet Feature Importance — trained through {year - 1}",
        )
        save_loss_curves(
            latest_artifacts["train_losses"],
            latest_artifacts["val_losses"],
            os.path.join(OUT_DIR, f"plot_tabnet_loss_curves_{year}.png"),
            year,
        )
        save_pick_value_curve_grid(
            latest_artifacts["baseline_model"],
            latest_artifacts["spline_model"],
            latest_artifacts["train_aug"],
            os.path.join(OUT_DIR, "plot_pick_value_curve_grid_v4.png"),
            year,
        )

        latest_metric_rows = [
            {
                "model_name": model_name,
                **eval_metrics(y_test, prediction_map[model_name]),
            }
            for model_name in MODEL_ORDER
        ]
        best_model = min(latest_metric_rows, key=lambda row: (row["rmse"], row["mae"]))["model_name"]
        save_latest_year_residual_diagnostics(
            y_test,
            prediction_map[best_model],
            latest_artifacts["test_aug"]["pick"].to_numpy(dtype=int),
            os.path.join(OUT_DIR, f"plot_latest_year_residual_diagnostics_{year}.png"),
            year,
            best_model,
        )

    print("\nWalk-forward results:")
    print(results_df.to_string(index=False))
    print("\nOverall summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved walk-forward metrics: {results_path}")
    print(f"Saved overall summary:     {overall_path}")
    print(f"Saved selected years:      {selected_path}")
    print(f"Saved plots in:            {OUT_DIR}")


if __name__ == "__main__":
    main()
