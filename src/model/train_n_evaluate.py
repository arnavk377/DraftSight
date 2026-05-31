"""
Model v6: Spline, XGBoost, CatBoost, Random Forest, FT-Transformer, Stacked Ensemble.

Walk-forward backtesting with OOF stacking.  MLPE, TabNet, and naive baselines
removed vs v5.1a.  Base models stacked: CatBoost, RF, FTT.
Meta-learner: NNLS (scipy) with recency weighting.

Outputs: poc_outputs_v6/
"""

import math
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from scipy.optimize import nnls

from src.model.ft_transformer import FTTransformer
from src.model.data_loader import (
    CFB_STAT_COLS,
    CONTEXT_ROSTER_COLS,
    DRAFT_CAT_COLS,
    DRAFT_NUM_COLS,
    DRAFT_POS_CONTEXT_COLS,
    NUM_COLS,
    CAT_COLS,
    PICK_TRADE_COLS,
    VET_PERF_COLS,
    build_two_year_labels,
    join_college_stats,
    join_veteran_performance,
    load_av,
    load_college_stats,
    load_draft,
    load_draft_pos_context,
    load_pick_trade_flags,
    load_roster_context,
    load_veteran_performance,
)


# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUPABASE_DIR = os.path.join(REPO_ROOT, "data", "supabase_exports")
DRAFTS_CSV = os.path.join(SUPABASE_DIR, "drafts.csv")
COLLEGE_STATS_CSV = os.path.join(SUPABASE_DIR, "college_stats.csv")
AV_CSV = os.path.join(SUPABASE_DIR, "av.csv")
ROSTER_CONTEXT_CSV = os.path.join(SUPABASE_DIR, "draft_pick_context_features.csv")
ROSTER_DIR = os.path.join(REPO_ROOT, "data", "roster")
VET_PERF_CSV = os.path.join(ROSTER_DIR, "veteran_performance_features.csv")
DRAFT_POS_CONTEXT_CSV = os.path.join(ROSTER_DIR, "draft_positional_context_features.csv")
PICK_TRADE_FLAGS_CSV = os.path.join(ROSTER_DIR, "pick_trade_flags.csv")
OUT_DIR = os.path.join(REPO_ROOT, "poc_outputs_v6")
os.makedirs(OUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
RANDOM_STATE = 42

AV_MAX = 60.0
LOG_PRED_CLIP = (0.0, np.log1p(AV_MAX))


def set_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Walk-forward config ───────────────────────────────────────────────────────

WALK_FORWARD_START_YEAR = 2006
PICK_BIN_SIZE = 16
PERF_BIN_COUNT = 4
SELECTED_SUMMARY_YEARS = [2010, 2015, 2020, 2024]

USE_POSITION_MEDIAN_IMPUTE: bool = False
EXCLUDE_COLLEGE_STATS: bool = True


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_ORDER = ["spline", "xgb", "catboost", "rf", "ftt", "stack", "pick_bin"]

MODEL_LABELS = {
    "spline":   "Spline Ridge",
    "xgb":      "XGBoost",
    "catboost": "CatBoost",
    "rf":       "Random Forest",
    "ftt":      "FT-Transformer",
    "stack":    "Stacked Ensemble",
    "pick_bin": "Pick Bin Mean",
}

MODEL_COLORS = {
    "spline":   "#3B6FB6",
    "xgb":      "#2A9D8F",
    "catboost": "#E9C46A",
    "rf":       "#F4A261",
    "ftt":      "#264653",
    "stack":    "#E63946",
    "pick_bin": "#888888",
}


# ── Hyperparameters ───────────────────────────────────────────────────────────

SPLINE_CFG = dict(alpha=25.0, n_knots_pick=8)

CATBOOST_CFG = dict(
    iterations=500, learning_rate=0.03, depth=5,
    l2_leaf_reg=3.0, random_seed=RANDOM_STATE, verbose=0,
    loss_function="Tweedie:variance_power=1.5",
)

RF_CFG = dict(
    n_estimators=500, max_depth=10, min_samples_leaf=20,
    max_features=0.5, n_jobs=1, random_state=RANDOM_STATE,
)

FTT_CFG = dict(d_token=64, n_heads=4, n_layers=2, dropout=0.2)
EMBED_TRAIN_CFG = dict(
    lr=3e-4, batch_size=128, max_epochs=200, patience=35, val_fraction=0.15,
)
HUBER_DELTA = 1.0

ORIG_SCALE_MODELS = {"xgb", "catboost", "pick_bin"}

STACK_BASE_MODELS = ["catboost", "rf", "ftt"]
META_MIN_TRAIN_YEARS = 3
STACK_DECAY = 0.75


# ── Metrics ───────────────────────────────────────────────────────────────────

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    spearman = spearmanr(y_true, y_pred).correlation
    return {
        "mae":      float(mean_absolute_error(y_true, y_pred)),
        "rmse":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearman) if not pd.isna(spearman) else 0.0,
        "r2":       float(r2_score(y_true, y_pred)),
    }


# ── College performance scorer ────────────────────────────────────────────────

class CollegePerformanceScorer:
    """Position-aware percentile score across CFB stat columns → single 0-to-1 value."""

    def __init__(self, feature_cols, group_col="position", min_group_size=25,
                 min_nonmissing=15, min_nonzero_rate=0.05):
        self.feature_cols = list(feature_cols)
        self.group_col = group_col
        self.min_group_size = min_group_size
        self.min_nonmissing = min_nonmissing
        self.min_nonzero_rate = min_nonzero_rate
        self.group_schemas_: dict = {}
        self.global_schema_: dict | None = None
        self.feature_cols_: list = []

    def fit(self, df: pd.DataFrame):
        self.feature_cols_ = [c for c in self.feature_cols if c in df.columns]
        self.global_schema_ = self._fit_schema(df)
        self.group_schemas_ = {}
        if self.group_col in df.columns:
            for gval, gdf in df.groupby(self.group_col):
                if len(gdf) < self.min_group_size:
                    continue
                schema = self._fit_schema(gdf)
                if schema["active_cols"]:
                    self.group_schemas_[str(gval)] = schema
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(df), 0.5, dtype=float)
        for pos, (_, row) in enumerate(df.iterrows()):
            schema = self.group_schemas_.get(str(row.get(self.group_col))) if self.group_col in df.columns else None
            score = self._row_score(row, schema) if schema else np.nan
            if pd.isna(score):
                score = self._row_score(row, self.global_schema_)
            scores[pos] = 0.5 if pd.isna(score) else score
        return scores

    def _fit_schema(self, df: pd.DataFrame) -> dict:
        active_cols, sorted_values = [], {}
        for col in self.feature_cols_:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) < self.min_nonmissing or vals.nunique() < 2:
                continue
            if col != "career_years" and float((vals != 0).mean()) < self.min_nonzero_rate and vals.nunique() < 8:
                continue
            active_cols.append(col)
            sorted_values[col] = np.sort(vals.to_numpy(dtype=float))
        return {"active_cols": active_cols, "sorted_values": sorted_values}

    def _row_score(self, row: pd.Series, schema: dict | None) -> float:
        if not schema or not schema["active_cols"]:
            return np.nan
        percentiles = []
        for col in schema["active_cols"]:
            val = row.get(col)
            if pd.isna(val):
                continue
            arr = schema["sorted_values"][col]
            percentiles.append(float(np.searchsorted(arr, float(val), side="right") / len(arr)))
        return float(np.mean(percentiles)) if percentiles else np.nan


def add_engineered_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, CollegePerformanceScorer]:
    scorer = CollegePerformanceScorer(CFB_STAT_COLS).fit(train_df)
    train_aug, test_aug = train_df.copy(), test_df.copy()
    train_aug["college_perf_score"] = scorer.transform(train_df)
    test_aug["college_perf_score"] = scorer.transform(test_df)
    cfb_cols = [c for c in CFB_STAT_COLS if c in train_df.columns]
    if cfb_cols:
        train_aug["cfb_matched"] = train_df[cfb_cols].notna().any(axis=1).astype(float)
        test_aug["cfb_matched"] = test_df[cfb_cols].notna().any(axis=1).astype(float)
    else:
        train_aug["cfb_matched"] = 0.0
        test_aug["cfb_matched"] = 0.0
    if USE_POSITION_MEDIAN_IMPUTE:
        cfb_cols_present = [c for c in CFB_STAT_COLS if c in train_aug.columns]
        if cfb_cols_present and "position" in train_aug.columns:
            has_data = train_aug["cfb_matched"] > 0.5
            pos_medians = (
                train_aug[has_data]
                .groupby("position")[cfb_cols_present]
                .median()
                .reset_index()
            )
            global_med = train_aug[has_data][cfb_cols_present].median()
            for aug in (train_aug, test_aug):
                no_data = aug["cfb_matched"] < 0.5
                if not no_data.any():
                    continue
                fill = (
                    aug.loc[no_data, ["position"]]
                    .merge(pos_medians, on="position", how="left")
                    .set_index(aug.index[no_data])
                )
                for col in cfb_cols_present:
                    aug.loc[no_data, col] = fill[col].fillna(global_med[col]).values

    # pos × round target encoding (Bayesian smoothing, k=10)
    SMOOTH_K = 10
    global_av_mean = float(train_aug["av_2yr"].mean())
    pos_round_stats = (
        train_aug.groupby(["position_group", "round"])["av_2yr"]
        .agg(["mean", "count"])
        .reset_index()
    )
    pos_round_stats["smoothed"] = (
        (pos_round_stats["count"] * pos_round_stats["mean"] + SMOOTH_K * global_av_mean)
        / (pos_round_stats["count"] + SMOOTH_K)
    )
    _pr_map = pos_round_stats.set_index(["position_group", "round"])["smoothed"].to_dict()

    for aug in (train_aug, test_aug):
        aug["pos_x_round_mean_av"] = [
            _pr_map.get((pg, rd), global_av_mean)
            for pg, rd in zip(
                aug["position_group"].fillna("UNKNOWN"),
                aug["round"].fillna(7).astype(int),
            )
        ]

    for aug in (train_aug, test_aug):
        log_pick = np.log1p(aug["pick"].to_numpy(dtype=float))
        pos_grp  = aug["position_group"].fillna("UNKNOWN")

        aug["age_x_log_pick"] = aug["age"] * log_pick
        aug["pick_within_round"] = aug["pick"] - (aug["round"] - 1) * 32
        aug["pos_pick_rank"] = (
            aug.groupby(["draft_season", "position_group"])["pick"]
            .rank(method="min")
        )
        aug["pos_group_drafted_before"] = (
            aug.groupby(["draft_season", "position_group"])["pick"]
            .rank(method="min") - 1
        ).astype(int)
        aug["need_score"] = (
            aug["vet_av"].fillna(0) * aug["prev_same_position_group_share"].fillna(0)
        )
        for pos in MAJOR_POS_GROUPS:
            aug[f"log_pick_x_{pos.lower()}"] = log_pick * (pos_grp == pos).astype(float)

    for aug in (train_aug, test_aug):
        num_present = [c for c in MODEL_NUM_COLS if c in aug.columns]
        aug[num_present] = aug[num_present].replace([np.inf, -np.inf], np.nan)
    return train_aug, test_aug, scorer


# ── Preprocessing builders ────────────────────────────────────────────────────

MAJOR_POS_GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]

INTERACTION_COLS = (
    ["age_x_log_pick", "pick_within_round", "pos_pick_rank", "need_score",
     "pos_x_round_mean_av", "pos_group_drafted_before"]
    + [f"log_pick_x_{pos.lower()}" for pos in MAJOR_POS_GROUPS]
)

MODEL_NUM_COLS = (
    DRAFT_NUM_COLS
    + ["college_perf_score", "cfb_matched"]
    + CFB_STAT_COLS
    + CONTEXT_ROSTER_COLS
    + PICK_TRADE_COLS
    + DRAFT_POS_CONTEXT_COLS
    + VET_PERF_COLS
    + INTERACTION_COLS
)
MODEL_CAT_COLS = DRAFT_CAT_COLS


def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_spline_preprocessor(df: pd.DataFrame):
    spline_pool = DRAFT_NUM_COLS + ["college_perf_score", "cfb_matched"]
    num_cols = [c for c in spline_pool if c in df.columns and df[c].notna().any()]
    cat_cols = [c for c in MODEL_CAT_COLS if c in df.columns]
    spline_cols = [c for c in ["pick", "college_perf_score"] if c in num_cols]
    rest_num = [c for c in num_cols if c not in spline_cols]

    transformers = []
    for col in spline_cols:
        transformers.append((f"{col}_spline", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=0)),
            ("spl", SplineTransformer(n_knots=SPLINE_CFG["n_knots_pick"] if col == "pick" else 5, degree=3, include_bias=False)),
        ]), [col]))
    if rest_num:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
        ]), rest_num))
    if cat_cols:
        transformers.append(("cat", _ohe(), cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_tree_preprocessor(df: pd.DataFrame):
    num_cols = [c for c in MODEL_NUM_COLS if c in df.columns and df[c].notna().any()]
    cat_cols = [c for c in MODEL_CAT_COLS if c in df.columns]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="constant", fill_value=0), num_cols))
    if cat_cols:
        transformers.append(("cat", _ohe(), cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def build_catboost_data(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    num_cols = [c for c in MODEL_NUM_COLS if c in df_tr.columns and df_tr[c].notna().any()]
    cat_cols = [c for c in MODEL_CAT_COLS if c in df_tr.columns]
    all_cols = num_cols + cat_cols
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    X_tr = df_tr[all_cols].copy()
    X_te = df_te[all_cols].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].fillna("nan").astype(str)
        X_te[c] = X_te[c].fillna("nan").astype(str)
    return X_tr, X_te, cat_indices, all_cols


def build_embed_data(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    num_cols = [c for c in MODEL_NUM_COLS if c in df_tr.columns and df_tr[c].notna().any()]
    cat_cols = [c for c in MODEL_CAT_COLS if c in df_tr.columns]
    num_imp = SimpleImputer(strategy="constant", fill_value=0)
    X_num_tr = num_imp.fit_transform(df_tr[num_cols]).astype(np.float32)
    X_num_te = num_imp.transform(df_te[num_cols]).astype(np.float32)
    cat_tr = df_tr[cat_cols].fillna("__nan__").astype(str)
    cat_te = df_te[cat_cols].fillna("__nan__").astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    X_cat_tr = enc.fit_transform(cat_tr)
    X_cat_te = enc.transform(cat_te)
    cardinalities = [len(cats) + 1 for cats in enc.categories_]
    for j in range(X_cat_tr.shape[1]):
        X_cat_tr[X_cat_tr[:, j] == -1, j] = cardinalities[j] - 1
        X_cat_te[X_cat_te[:, j] == -1, j] = cardinalities[j] - 1
    return X_num_tr, X_num_te, X_cat_tr, X_cat_te, cardinalities, num_cols, cat_cols


def get_feature_names(preprocessor: ColumnTransformer, num_cols: list, cat_cols: list) -> list:
    names = list(num_cols)
    if cat_cols and "cat" in preprocessor.named_transformers_:
        names += list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    return names


# ── Training helpers ──────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


def train_embed_model(model, X_num_tr, X_cat_tr, y_tr, desc="Embed", verbose=False, use_huber=False, **cfg_overrides):
    cfg = {**EMBED_TRAIN_CFG, **cfg_overrides}
    loss_fn = (lambda p, t: F.huber_loss(p, t, delta=HUBER_DELTA)) if use_huber else F.mse_loss
    n_val = max(1, int(len(y_tr) * cfg["val_fraction"]))
    Xn_t, Xn_v = X_num_tr[:-n_val], X_num_tr[-n_val:]
    Xc_t, Xc_v = X_cat_tr[:-n_val], X_cat_tr[-n_val:]
    yt, yv = y_tr[:-n_val], y_tr[-n_val:]
    loader = DataLoader(
        TensorDataset(to_tensor(Xn_t), torch.tensor(Xc_t, dtype=torch.long), to_tensor(yt)),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
    )
    Xn_v_t = to_tensor(Xn_v).to(DEVICE)
    Xc_v_t = torch.tensor(Xc_v, dtype=torch.long).to(DEVICE)
    yv_t = to_tensor(yv).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])
    best_val, best_state, patience_ctr = float("inf"), None, 0

    pbar = tqdm(range(cfg["max_epochs"]), desc=desc, unit="ep", leave=False)
    for epoch in pbar:
        model.train()
        for Xb_n, Xb_c, yb in loader:
            Xb_n, Xb_c, yb = Xb_n.to(DEVICE), Xb_c.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss_fn(model(Xb_n, Xb_c), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xn_v_t, Xc_v_t), yv_t).item()
        if vl < best_val:
            best_val, patience_ctr = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        pbar.set_postfix(val=f"{vl:.3f}", best=f"{best_val:.3f}",
                         pat=f"{patience_ctr}/{cfg['patience']}")
        if patience_ctr >= cfg["patience"]:
            pbar.set_postfix_str(f"early stop @ {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _nice_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_pred_vs_actual_grid(y_true, predictions, out_path, draft_year):
    y_true = np.asarray(y_true, dtype=float)
    n = len(MODEL_ORDER)
    ncols = 3
    nrows = math.ceil(n / ncols)
    lo = float(min(np.min(y_true), min(np.min(p) for p in predictions.values())))
    hi = float(max(np.max(y_true), max(np.max(p) for p in predictions.values())))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), dpi=200, sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for ax, model_name in zip(axes[:n], MODEL_ORDER):
        pred = np.asarray(predictions[model_name], dtype=float)
        m = eval_metrics(y_true, pred)
        ax.scatter(y_true, pred, s=22, alpha=0.65, edgecolor="none", color=MODEL_COLORS[model_name])
        ax.plot([lo, hi], [lo, hi], color="#2F3E46", linewidth=1.2, alpha=0.85)
        ax.set_title(MODEL_LABELS[model_name], fontsize=12, pad=8)
        ax.text(0.03, 0.97,
                f"MAE={m['mae']:.2f}\nRMSE={m['rmse']:.2f}\nR2={m['r2']:.2f}\nSpearman={m['spearman']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.85"))
        _nice_axes(ax)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{draft_year} Draft Class: Predicted vs Actual 2-Year AV", fontsize=15, y=0.99)
    fig.supxlabel(f"Actual 2-Year AV = AV({draft_year}) + AV({draft_year + 1})", fontsize=11)
    fig.supylabel("Predicted 2-Year AV", fontsize=11)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def save_walkforward_metric_grid(results_df, out_path):
    metrics = {"mae": "MAE", "rmse": "RMSE", "spearman": "Spearman", "r2": "R²"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200, sharex=True)
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, metrics.items()):
        for model_name in MODEL_ORDER:
            col = f"{model_name}_{metric_key}"
            if col not in results_df.columns:
                continue
            ax.plot(results_df["test_year"], results_df[col],
                    marker="o", linewidth=1.8, markersize=4,
                    color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name], alpha=0.9)
        ax.set_title(metric_label, fontsize=12, pad=8)
        _nice_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Walk-Forward Metrics by Draft Year", fontsize=15, y=0.99)
    fig.supxlabel("Test draft year", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def save_overall_summary_grid(summary_df, out_path):
    metric_info = {
        "mae": "Lower is better", "rmse": "Lower is better",
        "spearman": "Higher is better", "r2": "Higher is better",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    axes = axes.flatten()
    x = np.arange(len(summary_df))
    display_names = [MODEL_LABELS[n] for n in summary_df["model_type"]]
    colors = [MODEL_COLORS[n] for n in summary_df["model_type"]]

    for ax, (metric_key, subtitle) in zip(axes, metric_info.items()):
        vals = summary_df[metric_key].to_numpy(dtype=float)
        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{metric_key.upper()} ({subtitle})", fontsize=12, pad=8)
        for idx, val in enumerate(vals):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        _nice_axes(ax)

    fig.suptitle("Overall Out-of-Sample Summary", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def save_feature_importance(importance, feature_names, out_path, title):
    top_k = min(20, len(feature_names))
    idx = np.argsort(importance)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh([feature_names[i] for i in reversed(idx)], importance[list(reversed(idx))], color="#4C72B0")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_residual_diagnostics(y_true, y_pred, picks, out_path, draft_year, model_name):
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    picks = np.asarray(picks, dtype=int)
    color = MODEL_COLORS[model_name]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    ax0, ax1, ax2, ax3 = axes.flatten()

    ax0.scatter(y_pred, residual, s=22, alpha=0.65, color=color, edgecolor="none")
    ax0.axhline(0.0, color="#2F3E46", linewidth=1.0)
    ax0.set_title("Residual vs Predicted", fontsize=12)
    ax0.set_xlabel("Predicted AV")
    ax0.set_ylabel("Actual − Predicted")
    _nice_axes(ax0)

    ax1.hist(residual, bins=18, color=color, alpha=0.85)
    ax1.axvline(0.0, color="#2F3E46", linewidth=1.0)
    ax1.set_title("Residual Distribution", fontsize=12)
    ax1.set_xlabel("Actual − Predicted")
    _nice_axes(ax1)

    calib_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).sort_values("y_pred")
    calib_df["decile"] = pd.qcut(calib_df["y_pred"], q=min(10, len(calib_df)), labels=False, duplicates="drop")
    calib = calib_df.groupby("decile", as_index=False).agg(
        actual=("y_true", "mean"), predicted=("y_pred", "mean")
    )
    ax2.plot(calib["decile"], calib["actual"], marker="o", linewidth=2.0, label="Actual", color="#2A9D8F")
    ax2.plot(calib["decile"], calib["predicted"], marker="o", linewidth=2.0, linestyle="--",
             label="Predicted", color="#D77A61")
    ax2.set_title("Calibration by Prediction Decile", fontsize=12)
    ax2.set_xlabel("Prediction decile")
    ax2.set_ylabel("Mean 2-Year AV")
    ax2.legend(frameon=False, fontsize=9)
    _nice_axes(ax2)

    error_df = pd.DataFrame({
        "pick_bin": ((picks - 1) // PICK_BIN_SIZE).astype(int) + 1,
        "abs_error": np.abs(residual),
    })
    by_bin = error_df.groupby("pick_bin", as_index=False)["abs_error"].mean()
    ax3.plot(by_bin["pick_bin"], by_bin["abs_error"], marker="o", linewidth=2.0, color="#3B6FB6")
    ax3.set_title(f"Mean Absolute Error by {PICK_BIN_SIZE}-Pick Bin", fontsize=12)
    ax3.set_xlabel("Pick bin")
    ax3.set_ylabel("Mean absolute error")
    _nice_axes(ax3)

    fig.suptitle(f"{draft_year} Residual Diagnostics — {MODEL_LABELS[model_name]}", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def save_pick_value_curve(spline_model, train_df, scorer, out_path, draft_year):
    """Spline model pick value curve for low / median / high college perf score tiers."""
    max_pick = 256
    picks = np.arange(1, max_pick + 1)
    labels = ["Lower college score", "Median college score", "Higher college score"]
    colors = ["#457B9D", "#E9C46A", "#E76F51"]

    matched = train_df.loc[train_df["cfb_matched"] > 0.5] if "cfb_matched" in train_df.columns else train_df
    if matched.empty:
        matched = train_df
    scores = matched["college_perf_score"].dropna()
    if scores.empty or scores.nunique() < 3:
        return
    ref_scores = np.quantile(scores, [0.2, 0.5, 0.8])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    for perf_score, label, color in zip(ref_scores, labels, colors):
        grid = pd.DataFrame({"pick": picks})
        for col in train_df.columns:
            if col == "pick":
                continue
            if col in MODEL_CAT_COLS:
                mode = train_df[col].mode(dropna=True)
                grid[col] = mode.iloc[0] if not mode.empty else "UNK"
            else:
                vals = pd.to_numeric(train_df[col], errors="coerce")
                grid[col] = float(vals.median()) if vals.notna().any() else 0.0
        grid["college_perf_score"] = perf_score
        grid["cfb_matched"] = 1.0
        ax.plot(picks, np.expm1(spline_model.predict(grid)), label=label, linewidth=2.0, color=color)

    ax.set_xlabel("Pick number", fontsize=11)
    ax.set_ylabel("Predicted 2-Year AV", fontsize=11)
    ax.set_title(f"Spline Pick Value Curve — Trained Through {draft_year - 1}", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=9)
    _nice_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ── Data assembly ─────────────────────────────────────────────────────────────

def build_model_frame() -> pd.DataFrame:
    print("Loading draft:", DRAFTS_CSV)
    draft = load_draft(DRAFTS_CSV)

    print("Loading college stats:", COLLEGE_STATS_CSV)
    college_stats = load_college_stats(COLLEGE_STATS_CSV)

    print("Loading AV:", AV_CSV)
    av_long = load_av(AV_CSV)
    labels = build_two_year_labels(av_long)

    print("Loading roster context:", ROSTER_CONTEXT_CSV)
    roster_ctx = load_roster_context(ROSTER_CONTEXT_CSV)

    print("Loading pick trade flags:", PICK_TRADE_FLAGS_CSV)
    trade_flags = load_pick_trade_flags(PICK_TRADE_FLAGS_CSV)

    print("Loading draft positional context:", DRAFT_POS_CONTEXT_CSV)
    pos_ctx = load_draft_pos_context(DRAFT_POS_CONTEXT_CSV)

    print("Loading veteran performance:", VET_PERF_CSV)
    vet = load_veteran_performance(VET_PERF_CSV)

    merged, join_stats = join_college_stats(draft, college_stats)
    print(f"College stats join: {join_stats}")

    df = merged.rename(columns={"season": "draft_season"}).merge(
        labels, on=["draft_season", "pfr_player_id"], how="left"
    )
    df = df.merge(roster_ctx, on=["draft_season", "pick"], how="left")
    df = df.merge(trade_flags, on=["draft_season", "pick"], how="left")
    print(f"Trade flags: {df['was_traded'].notna().mean():.1%} picks covered")
    df = df.merge(pos_ctx, on=["draft_season", "pick"], how="left")
    df = join_veteran_performance(df, vet)
    print(f"Veteran performance: {df['vet_av'].notna().mean():.1%} picks covered")

    labeled = int(df["av_2yr"].notna().sum())
    print(f"AV label coverage before fill: {labeled}/{len(df)} = {labeled/len(df):.1%}")

    max_av_season = int(av_long["season"].dropna().max())
    fully_covered = df["draft_season"] + 1 <= max_av_season
    df.loc[fully_covered, "av_2yr"] = df.loc[fully_covered, "av_2yr"].fillna(0.0)
    df = df.dropna(subset=["av_2yr"]).copy()
    df["av_2yr"] = df["av_2yr"].astype(float)

    labeled_final = int((df["av_2yr"] > 0).sum())
    print(f"AV label coverage after fill: {len(df)} picks  "
          f"({labeled_final} with AV>0, {len(df)-labeled_final} filled as 0)")

    num_cols_present = [c for c in NUM_COLS if c in df.columns]
    df[num_cols_present] = df[num_cols_present].replace([np.inf, -np.inf], np.nan)
    return df


# ── Summary helpers ───────────────────────────────────────────────────────────

def create_overall_summary(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        sub = overall_df[overall_df["model_type"] == model_name]
        if sub.empty:
            continue
        m = eval_metrics(sub["av_2yr"].values, sub["prediction"].values)
        rows.append({"model_type": model_name, "n_predictions": len(sub), **m})
    return pd.DataFrame(rows)


def create_selected_years_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in SELECTED_SUMMARY_YEARS:
        year_df = results_df[results_df["test_year"] == year]
        if year_df.empty:
            continue
        row = year_df.iloc[0]
        for model_name in MODEL_ORDER:
            col_mae = f"{model_name}_mae"
            if col_mae not in row:
                continue
            rows.append({
                "year": int(row["test_year"]),
                "model_type": model_name,
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
                "mae": float(row[f"{model_name}_mae"]),
                "rmse": float(row[f"{model_name}_rmse"]),
                "spearman": float(row[f"{model_name}_spearman"]),
                "r2": float(row[f"{model_name}_r2"]),
            })
    return pd.DataFrame(rows)


# ── Main walk-forward loop ────────────────────────────────────────────────────

def main():
    set_seeds(RANDOM_STATE)
    df = build_model_frame()
    years = df["draft_season"].to_numpy(dtype=int)
    unique_years = sorted(np.unique(years))
    test_years = [yr for yr in unique_years if yr > WALK_FORWARD_START_YEAR]
    if not test_years:
        raise ValueError(f"No test years after {WALK_FORWARD_START_YEAR}.")
    print(f"Test years: {test_years[0]}–{test_years[-1]}  ({len(test_years)} folds)")

    results = []
    overall_rows = []
    latest_year = max(test_years)
    latest_artifacts: dict = {}
    oof_stack: list = []

    for test_year in tqdm(test_years, desc="Walk-forward folds", unit="yr"):
        train_mask = years < test_year
        test_mask = years == test_year
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            continue

        train_aug, test_aug, scorer = add_engineered_features(train_df, test_df)
        y_train_orig = np.clip(train_aug["av_2yr"].to_numpy(dtype=float), 0, None)
        y_train = np.log1p(y_train_orig)
        y_test = test_aug["av_2yr"].to_numpy(dtype=float)

        active_num_cols = MODEL_NUM_COLS
        if EXCLUDE_COLLEGE_STATS:
            _cfb_excl = set(CFB_STAT_COLS) | {"college_perf_score", "cfb_matched"}
            active_num_cols = [c for c in MODEL_NUM_COLS if c not in _cfb_excl]
        feat_cols = [c for c in active_num_cols + MODEL_CAT_COLS if c in train_aug.columns]
        X_train = train_aug[feat_cols].copy()
        X_test = test_aug[feat_cols].copy()

        print(f"\n=== Test {test_year}  train={len(train_aug)}  test={len(test_aug)}  "
              f"cfb_match={train_aug['cfb_matched'].mean():.1%} ===")

        prediction_map: dict[str, np.ndarray] = {}

        # Spline Ridge
        pre_s, s_num, s_cat = build_spline_preprocessor(X_train)
        spline_model = Pipeline([("pre", pre_s), ("reg", Ridge(alpha=SPLINE_CFG["alpha"]))])
        print("  [spline] fitting...", flush=True)
        spline_model.fit(X_train, y_train)
        prediction_map["spline"] = spline_model.predict(X_test)
        print("  [spline] done", flush=True)

        # XGBoost — Tweedie on original scale + 15% held-out eval for early stopping
        pre_x, x_num, x_cat = build_tree_preprocessor(X_train)
        n_xgb_val = max(1, int(len(X_train) * 0.15))
        X_xgb_t, X_xgb_v = X_train.iloc[:-n_xgb_val], X_train.iloc[-n_xgb_val:]
        y_xgb_t, y_xgb_v = y_train_orig[:-n_xgb_val], y_train_orig[-n_xgb_val:]
        xgb_pre = pre_x.fit(X_xgb_t)
        xgb_inner = XGBRegressor(
            n_estimators=600, learning_rate=0.02, max_depth=6,
            min_child_weight=1, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, nthread=1, random_state=RANDOM_STATE,
            objective="reg:tweedie", tweedie_variance_power=1.5,
            early_stopping_rounds=50, eval_metric="rmse",
        )
        print("  [xgb] fitting...", flush=True)
        xgb_inner.fit(
            xgb_pre.transform(X_xgb_t), y_xgb_t,
            eval_set=[(xgb_pre.transform(X_xgb_v), y_xgb_v)],
            verbose=False,
        )
        xgb_model = Pipeline([("pre", xgb_pre), ("xgb", xgb_inner)])
        prediction_map["xgb"] = xgb_model.predict(X_test)
        print("  [xgb] done", flush=True)

        # CatBoost — Tweedie on original scale + 15% held-out eval for early stopping
        X_tr_cb, X_te_cb, cat_idx_cb, cb_feat_names = build_catboost_data(X_train, X_test)
        n_cb_val = max(1, int(len(X_tr_cb) * 0.15))
        X_cb_t, X_cb_v = X_tr_cb.iloc[:-n_cb_val], X_tr_cb.iloc[-n_cb_val:]
        y_cb_t, y_cb_v = y_train_orig[:-n_cb_val], y_train_orig[-n_cb_val:]
        cb_model = CatBoostRegressor(**CATBOOST_CFG, cat_features=cat_idx_cb, thread_count=1)
        print("  [catboost] fitting...", flush=True)
        cb_model.fit(X_cb_t, y_cb_t, eval_set=(X_cb_v, y_cb_v), early_stopping_rounds=50)
        prediction_map["catboost"] = cb_model.predict(X_te_cb)
        print("  [catboost] done", flush=True)

        # Random Forest
        pre_rf, rf_num, rf_cat = build_tree_preprocessor(X_train)
        rf_model = Pipeline([("pre", pre_rf), ("rf", RandomForestRegressor(**RF_CFG))])
        print("  [rf] fitting...", flush=True)
        rf_model.fit(X_train, y_train)
        prediction_map["rf"] = rf_model.predict(X_test)
        print("  [rf] done", flush=True)

        # FT-Transformer
        X_num_tr, X_num_te, X_cat_tr, X_cat_te, cardinalities, embed_num, embed_cat = \
            build_embed_data(X_train, X_test)
        print("  [ftt] fitting...", flush=True)
        ftt = FTTransformer(X_num_tr.shape[1], cardinalities, **FTT_CFG).to(DEVICE)
        ftt = train_embed_model(ftt, X_num_tr, X_cat_tr, y_train, desc="FTT", use_huber=False, lr=1e-4, patience=50)
        ftt.eval()
        with torch.no_grad():
            prediction_map["ftt"] = ftt(
                to_tensor(X_num_te).to(DEVICE),
                torch.tensor(X_cat_te, dtype=torch.long).to(DEVICE),
            ).cpu().numpy()
        print("  [ftt] done", flush=True)

        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Pick-bin baseline — mean training AV per 16-pick bin, original AV scale
        _global_mean = float(y_train_orig.mean())
        _train_bins = ((np.clip(train_aug["pick"].to_numpy(dtype=float), 1, None) - 1) // PICK_BIN_SIZE).astype(int)
        _test_bins  = ((np.clip(test_aug["pick"].to_numpy(dtype=float),  1, None) - 1) // PICK_BIN_SIZE).astype(int)
        _bin_means  = {b: float(y_train_orig[_train_bins == b].mean()) for b in np.unique(_train_bins)}
        prediction_map["pick_bin"] = np.array([_bin_means.get(b, _global_mean) for b in _test_bins])

        # Back-transform to original AV scale
        for _name in MODEL_ORDER:
            if _name == "stack":
                continue
            if _name in ORIG_SCALE_MODELS:
                prediction_map[_name] = np.clip(prediction_map[_name], 0.0, AV_MAX)
            else:
                prediction_map[_name] = np.expm1(np.clip(prediction_map[_name], *LOG_PRED_CLIP))

        # Stacking meta-learner (recency-weighted NNLS)
        oof_entry = {m: prediction_map[m].copy() for m in STACK_BASE_MODELS}
        oof_entry["y"] = y_test.copy()

        if len(oof_stack) >= META_MIN_TRAIN_YEARS:
            n_folds = len(oof_stack)
            fold_weights = np.array([STACK_DECAY ** (n_folds - 1 - i) for i in range(n_folds)])

            X_parts, y_parts, w_parts = [], [], []
            for i, entry in enumerate(oof_stack):
                X_f = np.column_stack([entry[m] for m in STACK_BASE_MODELS])
                X_parts.append(X_f)
                y_parts.append(entry["y"])
                w_parts.append(np.full(len(entry["y"]), fold_weights[i]))

            meta_X_tr = np.vstack(X_parts)
            meta_y_tr = np.concatenate(y_parts)
            sqrt_w = np.sqrt(np.concatenate(w_parts))

            coef, _ = nnls(meta_X_tr * sqrt_w[:, None], meta_y_tr * sqrt_w)
            coef = coef / (coef.sum() + 1e-10)
            meta_X_te = np.column_stack([prediction_map[m] for m in STACK_BASE_MODELS])
            prediction_map["stack"] = np.clip(meta_X_te @ coef, 0.0, AV_MAX)
            print(f"  [stack] weights: { {m: round(w, 3) for m, w in zip(STACK_BASE_MODELS, coef)} }")
        else:
            prediction_map["stack"] = prediction_map["catboost"].copy()
            remaining = META_MIN_TRAIN_YEARS - len(oof_stack)
            print(f"  [stack] fallback to catboost ({remaining} more OOF fold(s) needed)")

        oof_stack.append(oof_entry)

        year_result = {
            "test_year": int(test_year),
            "n_train": int(len(train_aug)),
            "n_test": int(len(test_aug)),
            "cfb_match_rate": float(train_aug["cfb_matched"].mean()),
        }
        for model_name in MODEL_ORDER:
            m = eval_metrics(y_test, prediction_map[model_name])
            print(f"  {MODEL_LABELS[model_name]:18s}  "
                  f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
                  f"R2={m['r2']:.3f}  Spearman={m['spearman']:.3f}")
            for metric_key, metric_val in m.items():
                year_result[f"{model_name}_{metric_key}"] = metric_val

            pred_frame = test_aug[[
                "draft_season", "pick", "round", "team", "position",
                "position_group", "category", "college", "pfr_player_name",
                "college_perf_score", "cfb_matched", "av_2yr",
            ]].copy()
            pred_frame["model_type"] = model_name
            pred_frame["prediction"] = prediction_map[model_name]
            pred_frame["residual"] = pred_frame["av_2yr"] - pred_frame["prediction"]
            overall_rows.append(pred_frame)

        results.append(year_result)

        if test_year == latest_year:
            xgb_feat_names = get_feature_names(xgb_model.named_steps["pre"], x_num, x_cat)
            rf_feat_names = get_feature_names(rf_model.named_steps["pre"], rf_num, rf_cat)
            latest_artifacts = {
                "test_year": test_year,
                "y_test": y_test,
                "test_aug": test_aug.copy(),
                "train_aug": train_aug.copy(),
                "prediction_map": prediction_map,
                "scorer": scorer,
                "spline_model": spline_model,
                "xgb_importance": xgb_model.named_steps["xgb"].feature_importances_,
                "xgb_feat_names": xgb_feat_names,
                "cb_importance": cb_model.get_feature_importance(),
                "cb_feat_names": cb_feat_names,
                "rf_importance": rf_model.named_steps["rf"].feature_importances_,
                "rf_feat_names": rf_feat_names,
            }

    results_df = pd.DataFrame(results)
    overall_df = pd.concat(overall_rows, ignore_index=True)
    summary_df = create_overall_summary(overall_df)
    selected_df = create_selected_years_summary(results_df)

    results_df.to_csv(os.path.join(OUT_DIR, "model_v6_walkforward_results.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "model_v6_overall_summary.csv"), index=False)
    selected_df.to_csv(os.path.join(OUT_DIR, "model_v6_selected_years_summary.csv"), index=False)

    id_cols = [
        "draft_season", "pick", "round", "team", "position", "position_group",
        "category", "college", "pfr_player_name", "college_perf_score", "cfb_matched", "av_2yr",
    ]
    all_pred_wide = (
        overall_df[id_cols + ["model_type", "prediction"]]
        .pivot_table(index=id_cols, columns="model_type", values="prediction", aggfunc="first")
        .reset_index()
    )
    all_pred_wide.columns.name = None
    all_pred_wide = all_pred_wide.rename(
        columns={m: f"pred_{m}" for m in MODEL_ORDER if m in all_pred_wide.columns}
    )
    all_pred_wide.to_csv(os.path.join(OUT_DIR, "model_v6_all_predictions.csv"), index=False)

    if latest_artifacts:
        year = latest_artifacts["test_year"]
        y_test = latest_artifacts["y_test"]
        prediction_map = latest_artifacts["prediction_map"]

        save_pred_vs_actual_grid(
            y_test, prediction_map,
            os.path.join(OUT_DIR, f"plot_pred_vs_actual_grid_{year}.png"), year)

        save_walkforward_metric_grid(
            results_df, os.path.join(OUT_DIR, "plot_walkforward_metric_grid_v6.png"))

        save_overall_summary_grid(
            summary_df, os.path.join(OUT_DIR, "plot_overall_summary_grid_v6.png"))

        for model_name, imp, feat_names in [
            ("xgb",      latest_artifacts["xgb_importance"], latest_artifacts["xgb_feat_names"]),
            ("catboost", latest_artifacts["cb_importance"],  latest_artifacts["cb_feat_names"]),
            ("rf",       latest_artifacts["rf_importance"],  latest_artifacts["rf_feat_names"]),
        ]:
            save_feature_importance(
                imp, feat_names,
                os.path.join(OUT_DIR, f"plot_{model_name}_feature_importance_v6.png"),
                title=f"{MODEL_LABELS[model_name]} Feature Importance — trained through {year - 1}",
            )

        best_model = min(MODEL_ORDER, key=lambda n: eval_metrics(y_test, prediction_map[n])["rmse"])
        save_residual_diagnostics(
            y_test, prediction_map[best_model],
            latest_artifacts["test_aug"]["pick"].to_numpy(dtype=int),
            os.path.join(OUT_DIR, f"plot_residual_diagnostics_{year}.png"),
            year, best_model)

        save_pick_value_curve(
            latest_artifacts["spline_model"],
            latest_artifacts["train_aug"],
            latest_artifacts["scorer"],
            os.path.join(OUT_DIR, "plot_pick_value_curve_v6.png"),
            year)

        latest_pred_df = latest_artifacts["test_aug"][[
            "draft_season", "pick", "round", "team", "position", "position_group",
            "category", "college", "pfr_player_name", "college_perf_score",
            "cfb_matched", "av_2yr",
        ]].copy()
        for model_name in MODEL_ORDER:
            latest_pred_df[f"pred_{model_name}"] = prediction_map[model_name]
        latest_pred_df.to_csv(
            os.path.join(OUT_DIR, f"model_v6_latest_year_predictions_{year}.csv"), index=False)

    print("\nWalk-forward results:")
    print(results_df.to_string(index=False))
    print("\nOverall summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
