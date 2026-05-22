"""Final poster modeling pipeline for DraftSight.

This script builds a leakage-aware model frame from the Supabase exports, tunes
four model families, runs expanding-window walk-forward evaluation, and saves
poster-ready figures and CSVs.

Models:
  - XGBoost
  - CatBoost
  - MLPRegressor
  - Spline Ridge regression

Default output:
  reports/final_poster_modeling/latest/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("/private/tmp") / "draftsight_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, randint, spearmanr, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - handled at runtime for local environments.
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled at runtime for local environments.
    XGBRegressor = None


warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "exports" / "supabase"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "final_poster_modeling" / "latest"

ALL_MODELS = ("xgboost", "catboost", "mlp", "spline")
PICK_BIN_BASELINE_KEY = "pick_bin"
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "mlp": "MLP Neural Net",
    "spline": "Spline Ridge",
}
RESULT_LABELS = {
    **MODEL_LABELS,
    PICK_BIN_BASELINE_KEY: "Pick-Bin Baseline",
}
MODEL_COLORS = {
    "xgboost": "#2864a6",
    "catboost": "#2a9d8f",
    "mlp": "#7b4ab3",
    "spline": "#e6863a",
    PICK_BIN_BASELINE_KEY: "#555555",
}

LEAKAGE_COLS = {
    "car_av",
    "w_av",
    "dr_av",
    "games",
    "allpro",
    "probowls",
    "seasons_started",
    "hof",
    "last_nfl_season",
    "pass_completions",
    "pass_attempts",
    "pass_yards",
    "pass_tds",
    "pass_ints",
    "rush_atts",
    "rush_yards",
    "rush_tds",
    "receptions",
    "rec_yards",
    "rec_tds",
    "def_solo_tackles",
    "def_ints",
    "def_sacks",
}

DROP_ALWAYS = {
    "draft_pick_id",
    "gsis_id",
    "pfr_player_id",
    "sports_ref_cfb_player_id",
    "pfr_player_name",
    "team_raw",
    "franchise_id",
    "draft_season",
    "av_2yr",
    "target_complete",
    "label_year_max",
    "college_stats_id",
    "cfb_player_id",
    "cfb_player",
    "cfb_college_team",
    "pick_first_trade_date",
    "pick_last_trade_date",
    "pick_first_gave_team",
    "pick_first_received_team",
    "pick_last_gave_team",
    "pick_last_received_team",
    "pick_trade_team_chain",
    "roster_context_available",
    "roster_context_season",
}

LOW_CARD_CATEGORICALS = [
    "team",
    "position",
    "position_group",
    "category",
    "side",
    "college",
    "cfb_conference",
    "context_draft_position_group",
]


@dataclass
class SearchResult:
    model_key: str
    estimator: Any
    best_params: dict[str, Any]
    cv_rmse: float
    train_mae: float
    train_rmse: float
    train_r2: float
    train_spearman: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument(
        "--allow-partial-model-run",
        action="store_true",
        help=(
            "Opt-in escape hatch for local debugging. Final poster runs should "
            "include all four tuned model families."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build and validate the cleaned modeling frame; do not train models.",
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=50, help="Random search iterations per model.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for sklearn searches.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--first-test-year",
        type=int,
        default=2010,
        help=(
            "First draft class used as a walk-forward test year. Defaults to "
            "2010 so hyperparameters can be tuned on a meaningful pre-2010 "
            "history without peeking at poster evaluation years."
        ),
    )
    parser.add_argument("--max-test-year", type=int, default=None)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--curve-years", nargs="*", type=int, default=None)
    parser.add_argument("--pick-bin-size", type=int, default=16)
    parser.add_argument(
        "--prediction-upper-quantile",
        type=float,
        default=1.0,
        help=(
            "Upper bound for predictions based on the training target quantile. "
            "Default 1.0 clips predictions to the historical max 2-year AV "
            "available in that training window."
        ),
    )
    parser.add_argument(
        "--no-prediction-clipping",
        action="store_true",
        help="Disable AV-range clipping. Usually not recommended for poster metrics.",
    )
    parser.add_argument("--ohe-min-frequency", type=int, default=10)
    parser.add_argument("--top-n-features", type=int, default=24)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    selected = tuple(dict.fromkeys(args.models))
    args.models = list(selected)
    missing_models = sorted(set(ALL_MODELS).difference(selected))
    if missing_models and not args.allow_partial_model_run and not args.prepare_only:
        missing = ", ".join(MODEL_LABELS[key] for key in missing_models)
        raise SystemExit(
            "Refusing to run a final-looking partial model comparison. "
            f"Missing tuned model(s): {missing}. "
            "Install dependencies and run all four models, or pass "
            "`--allow-partial-model-run` only for local debugging."
        )

    if args.output_dir.name.lower() in {"smoke", "smoke_plots", "test", "tmp"}:
        raise SystemExit(
            "Refusing to write final poster artifacts to a throwaway-looking output "
            f"folder: {args.output_dir}. Use a descriptive folder under "
            "`reports/final_poster_modeling/`."
        )

    if not 0.5 <= args.prediction_upper_quantile <= 1.0:
        raise SystemExit("`--prediction-upper-quantile` must be between 0.5 and 1.0.")

    missing_deps: list[str] = []
    if "catboost" in selected and CatBoostRegressor is None and not args.prepare_only:
        missing_deps.append("catboost")
    if "xgboost" in selected and XGBRegressor is None and not args.prepare_only:
        missing_deps.append("xgboost")
    if missing_deps:
        raise SystemExit(
            "Missing required model dependency/dependencies for the selected final models: "
            f"{', '.join(missing_deps)}. Run `pip install -r requirements.txt` first."
        )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def av_log_transform(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(y, 0.0))


def av_inverse_transform(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).correlation)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def prepare_output_dir(output_dir: Path) -> None:
    """Clear generated artifact folders so stale plots/tables do not linger."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["tables", "figures", "models", "best_model"]:
        path = output_dir / subdir
        if path.exists():
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        path.mkdir(parents=True, exist_ok=True)


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].where(out[col].notna(), "")
    return out


def count_nonfinite_numeric(df: pd.DataFrame) -> int:
    count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        count += int(np.isinf(values).sum())
    return count


def source_table_quality(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_cols = {
        "drafts": ["draft_season", "pick"],
        "college_stats": ["draft_season", "draft_overall"],
        "draft_board": ["draft_season", "draft_overall"],
        "draft_pick_context_features": ["draft_season", "pick"],
        "av": ["season", "pfr_player_id"],
    }
    for name, df in tables.items():
        key = key_cols.get(name)
        duplicate_key_rows = int(df.duplicated(key).sum()) if key and set(key).issubset(df.columns) else 0
        rows.append(
            {
                "table": name,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_model_key_rows": duplicate_key_rows,
                "missing_cells": int(df.isna().sum().sum()),
                "nonfinite_numeric_values": count_nonfinite_numeric(df),
            }
        )
    return pd.DataFrame(rows).sort_values("table")


def aggregate_av(av: pd.DataFrame) -> pd.DataFrame:
    """Collapse player-team-season AV rows to player-season rows."""
    required = {"season", "pfr_player_id", "av"}
    missing = required.difference(av.columns)
    if missing:
        raise ValueError(f"av.csv missing columns: {sorted(missing)}")

    av = clean_values(av)
    av["season"] = pd.to_numeric(av["season"], errors="coerce")
    av["av"] = pd.to_numeric(av["av"], errors="coerce").fillna(0.0)
    av = av.dropna(subset=["season", "pfr_player_id"]).copy()
    av["season"] = av["season"].astype(int)
    return (
        av.groupby(["season", "pfr_player_id"], as_index=False)
        .agg(
            av=("av", "sum"),
            player_name=("player_name", "first") if "player_name" in av.columns else ("pfr_player_id", "first"),
        )
        .sort_values(["season", "pfr_player_id"])
    )


def load_exports(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in [
        "drafts",
        "av",
        "college_stats",
        "college_player_seasons",
        "draft_board",
        "draft_pick_context_features",
        "rosters",
        "trades",
    ]:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required export: {path}")
        tables[name] = clean_values(pd.read_csv(path, low_memory=False))
    tables["av"] = aggregate_av(tables["av"])
    return tables


def build_target(drafts: pd.DataFrame, av: pd.DataFrame, horizon: int) -> pd.DataFrame:
    base = drafts[["draft_pick_id", "draft_season", "pfr_player_id"]].copy()
    base["draft_season"] = pd.to_numeric(base["draft_season"], errors="coerce")
    labeled = base.dropna(subset=["draft_season", "pfr_player_id"]).copy()
    labeled["draft_season"] = labeled["draft_season"].astype(int)

    merged = labeled.merge(av[["season", "pfr_player_id", "av"]], on="pfr_player_id", how="left")
    in_horizon = (
        merged["season"].notna()
        & (merged["season"] >= merged["draft_season"])
        & (merged["season"] <= merged["draft_season"] + horizon - 1)
    )
    target = (
        merged[in_horizon]
        .groupby("draft_pick_id", as_index=False)["av"]
        .sum()
        .rename(columns={"av": "av_2yr"})
    )
    target = base[["draft_pick_id", "draft_season"]].merge(target, on="draft_pick_id", how="left")

    min_av_season = int(av["season"].min())
    max_av_season = int(av["season"].max())
    target["label_year_max"] = target["draft_season"] + horizon - 1
    target["target_complete"] = (
        (target["draft_season"] >= min_av_season)
        & (target["label_year_max"] <= max_av_season)
    )
    target.loc[target["target_complete"], "av_2yr"] = target.loc[
        target["target_complete"], "av_2yr"
    ].fillna(0.0)
    target.loc[~target["target_complete"], "av_2yr"] = np.nan
    return target[["draft_pick_id", "av_2yr", "target_complete", "label_year_max"]]


def prefixed_join_columns(
    df: pd.DataFrame,
    prefix: str,
    join_cols: list[str],
    drop_cols: set[str],
) -> pd.DataFrame:
    keep = [col for col in df.columns if col not in drop_cols]
    out = df[keep].copy()
    rename = {
        col: f"{prefix}{col}"
        for col in out.columns
        if col not in join_cols
    }
    return out.rename(columns=rename)


def build_model_frame(tables: dict[str, pd.DataFrame], horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    drafts = tables["drafts"].copy()
    college = tables["college_stats"].copy()
    draft_board = tables["draft_board"].copy()
    context = tables["draft_pick_context_features"].copy()

    drafts["draft_season"] = pd.to_numeric(drafts["draft_season"], errors="coerce").astype("Int64")
    drafts["pick"] = pd.to_numeric(drafts["pick"], errors="coerce").astype("Int64")

    target = build_target(drafts, tables["av"], horizon)
    frame = drafts.merge(target, on="draft_pick_id", how="left")

    college_drop = {
        "college_stats_id",
        "cfb_player_id",
        "player",
        "college_team",
        "position",
        "position_group",
        "draft_round",
        "draft_pick_in_round",
    }
    college = college.rename(columns={"draft_overall": "pick"})
    college["draft_season"] = pd.to_numeric(college["draft_season"], errors="coerce").astype("Int64")
    college["pick"] = pd.to_numeric(college["pick"], errors="coerce").astype("Int64")
    college_join = prefixed_join_columns(
        college,
        prefix="cfb_",
        join_cols=["draft_season", "pick"],
        drop_cols=college_drop,
    )
    frame = frame.merge(college_join, on=["draft_season", "pick"], how="left")

    board_drop = {
        "draft_board_id",
        "draft_pick_in_round",
        "team",
        "nfl_team",
        "nfl_team_id",
        "player",
        "position",
        "position_group",
        "nfl_athlete_id",
        "cfb_player_id",
        "college_id",
        "college_team",
        "hometown_info_json",
    }
    draft_board = draft_board.rename(columns={"draft_overall": "pick"})
    draft_board["draft_season"] = pd.to_numeric(draft_board["draft_season"], errors="coerce").astype("Int64")
    draft_board["pick"] = pd.to_numeric(draft_board["pick"], errors="coerce").astype("Int64")
    board_join = prefixed_join_columns(
        draft_board,
        prefix="board_",
        join_cols=["draft_season", "pick"],
        drop_cols=board_drop,
    )
    frame = frame.merge(board_join, on=["draft_season", "pick"], how="left")

    context_drop = {"team", "team_draft_code", "franchise_id", "position"}
    context["draft_season"] = pd.to_numeric(context["draft_season"], errors="coerce").astype("Int64")
    context["pick"] = pd.to_numeric(context["pick"], errors="coerce").astype("Int64")
    context_join = prefixed_join_columns(
        context,
        prefix="context_",
        join_cols=["draft_season", "pick"],
        drop_cols=context_drop,
    )
    frame = frame.merge(context_join, on=["draft_season", "pick"], how="left")

    frame = clean_values(frame)
    frame["draft_season"] = pd.to_numeric(frame["draft_season"], errors="coerce").astype(int)
    frame["pick"] = pd.to_numeric(frame["pick"], errors="coerce").astype(int)

    quality = {
        "rows": int(len(frame)),
        "complete_target_rows": int(frame["target_complete"].sum()),
        "target_year_min": int(frame.loc[frame["target_complete"], "draft_season"].min()),
        "target_year_max": int(frame.loc[frame["target_complete"], "draft_season"].max()),
        "av_duplicate_player_seasons_after_aggregation": int(
            tables["av"].duplicated(["season", "pfr_player_id"]).sum()
        ),
        "nonfinite_numeric_values": count_nonfinite_numeric(frame),
        "college_match_rate_complete": float(
            frame.loc[frame["target_complete"], "cfb_num_seasons"].notna().mean()
            if "cfb_num_seasons" in frame.columns
            else 0.0
        ),
        "context_match_rate_complete": float(
            frame.loc[frame["target_complete"], "context_prev_roster_players"].notna().mean()
            if "context_prev_roster_players" in frame.columns
            else 0.0
        ),
        "draft_board_match_rate_complete": float(
            frame.loc[frame["target_complete"], "board_pre_draft_ranking"].notna().mean()
            if "board_pre_draft_ranking" in frame.columns
            else 0.0
        ),
        "source_table_rows": {name: int(len(df)) for name, df in tables.items()},
    }
    return frame, quality


def select_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    drop = DROP_ALWAYS | LEAKAGE_COLS
    feature_cols = [col for col in frame.columns if col not in drop]
    feature_cols = [col for col in feature_cols if not col.endswith("_id")]
    feature_cols = [col for col in feature_cols if not col.startswith("pfr_")]
    feature_cols = [col for col in feature_cols if col not in {"label_year_max"}]

    categorical_cols = [
        col
        for col in LOW_CARD_CATEGORICALS
        if col in feature_cols
    ]
    # Allow a few generated categorical fields if they are intentionally low-cardinality.
    for col in feature_cols:
        if frame[col].dtype == "object" and col not in categorical_cols:
            nunique = frame[col].nunique(dropna=True)
            if 1 < nunique <= 80:
                categorical_cols.append(col)

    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    numeric_cols = [
        col for col in numeric_cols
        if pd.api.types.is_numeric_dtype(frame[col]) or frame[col].dtype != "object"
    ]
    feature_cols = numeric_cols + categorical_cols
    return feature_cols, numeric_cols, categorical_cols


def make_ohe(min_frequency: int) -> OneHotEncoder:
    kwargs = {
        "handle_unknown": "infrequent_if_exist",
        "min_frequency": min_frequency,
        "sparse_output": False,
    }
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs.pop("sparse_output")
        kwargs["sparse"] = False
        return OneHotEncoder(**kwargs)


def make_preprocessor(num_cols: list[str], cat_cols: list[str], min_frequency: int) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            ("cat", make_ohe(min_frequency), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_spline_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    min_frequency: int,
) -> ColumnTransformer:
    non_pick_num = [col for col in num_cols if col != "pick"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if "pick" in num_cols:
        transformers.append(
            (
                "pick_spline",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("spline", SplineTransformer(include_bias=False)),
                        ("post_scaler", StandardScaler()),
                    ]
                ),
                ["pick"],
            )
        )
    if non_pick_num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                non_pick_num,
            )
        )
    if cat_cols:
        transformers.append(("cat", make_ohe(min_frequency), cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def split_xy(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    X = frame[feature_cols].copy()
    y = frame["av_2yr"].astype(float).to_numpy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in cat_cols:
        X[col] = X[col].fillna("_missing_").astype(str)
    return X, y


def model_search_space(
    model_key: str,
    num_cols: list[str],
    cat_cols: list[str],
    min_frequency: int,
    random_state: int,
) -> tuple[BaseEstimator, dict[str, Any]]:
    if model_key == "xgboost":
        if XGBRegressor is None:
            raise ImportError("XGBoost is requested but not installed. Run `pip install xgboost`.")
        pipe = Pipeline(
            [
                ("pre", make_preprocessor(num_cols, cat_cols, min_frequency)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        )
        params = {
            "model__n_estimators": randint(250, 1300),
            "model__max_depth": randint(2, 9),
            "model__learning_rate": loguniform(0.003, 0.12),
            "model__subsample": uniform(0.55, 0.4),
            "model__colsample_bytree": uniform(0.55, 0.4),
            "model__min_child_weight": randint(1, 12),
            "model__gamma": loguniform(1e-5, 2.0),
            "model__reg_alpha": loguniform(1e-5, 5.0),
            "model__reg_lambda": loguniform(1e-3, 20.0),
        }
        return pipe, params

    if model_key == "mlp":
        base_pipe = Pipeline(
            [
                ("pre", make_preprocessor(num_cols, cat_cols, min_frequency)),
                (
                    "model",
                    MLPRegressor(
                        max_iter=900,
                        early_stopping=True,
                        validation_fraction=0.12,
                        learning_rate="adaptive",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe = TransformedTargetRegressor(
            regressor=base_pipe,
            func=av_log_transform,
            inverse_func=av_inverse_transform,
            check_inverse=False,
        )
        params = {
            "regressor__model__hidden_layer_sizes": [
                (64,),
                (128,),
                (128, 64),
                (256, 128),
            ],
            "regressor__model__activation": ["relu", "tanh"],
            "regressor__model__alpha": loguniform(1e-4, 2e-1),
            "regressor__model__learning_rate_init": loguniform(5e-5, 2e-3),
            "regressor__model__batch_size": [128, 256, 512],
        }
        return pipe, params

    if model_key == "spline":
        base_pipe = Pipeline(
            [
                ("pre", make_spline_preprocessor(num_cols, cat_cols, min_frequency)),
                ("model", Ridge()),
            ]
        )
        pipe = TransformedTargetRegressor(
            regressor=base_pipe,
            func=av_log_transform,
            inverse_func=av_inverse_transform,
            check_inverse=False,
        )
        params = {
            "regressor__pre__pick_spline__spline__n_knots": randint(3, 14),
            "regressor__pre__pick_spline__spline__degree": [2, 3, 4],
            "regressor__model__alpha": loguniform(1e-2, 1e5),
        }
        return pipe, params

    raise ValueError(f"Unsupported sklearn model: {model_key}")


def fit_sklearn_search(
    model_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    args: argparse.Namespace,
) -> SearchResult:
    estimator, params = model_search_space(
        model_key,
        num_cols,
        cat_cols,
        args.ohe_min_frequency,
        args.random_state,
    )
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=args.n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1,
        refit=True,
    )
    print(f"\n[{MODEL_LABELS[model_key]}] Grid search on {len(y):,} rows ...")
    search.fit(X, y)
    preds = search.best_estimator_.predict(X)
    return SearchResult(
        model_key=model_key,
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        cv_rmse=float(-search.best_score_),
        train_mae=float(mean_absolute_error(y, preds)),
        train_rmse=rmse(y, preds),
        train_r2=float(r2_score(y, preds)),
        train_spearman=safe_spearman(y, preds),
    )


def sample_catboost_params(n_iter: int, random_state: int) -> list[dict[str, Any]]:
    space = {
        "iterations": randint(250, 1200),
        "depth": randint(3, 9),
        "learning_rate": loguniform(0.004, 0.12),
        "l2_leaf_reg": loguniform(0.05, 25.0),
        "bagging_temperature": uniform(0.0, 1.5),
        "random_strength": uniform(0.0, 2.0),
        "border_count": randint(64, 255),
    }
    return list(ParameterSampler(space, n_iter=n_iter, random_state=random_state))


def prepare_catboost_X(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy().replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if col in cat_cols:
            out[col] = out[col].fillna("_missing_").astype(str)
        else:
            med = pd.to_numeric(out[col], errors="coerce").median()
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0 if pd.isna(med) else med)
    return out


def fit_catboost_search(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    args: argparse.Namespace,
) -> SearchResult:
    if CatBoostRegressor is None:
        raise ImportError(
            "CatBoost is requested but not installed. Install it with `pip install catboost`."
        )

    X_cb = prepare_catboost_X(X, cat_cols)
    cat_indices = [list(X_cb.columns).index(col) for col in cat_cols if col in X_cb.columns]
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    best_params: dict[str, Any] | None = None
    best_rmse = float("inf")

    print(f"\n[CatBoost] Manual grid search on {len(y):,} rows ...")
    for i, params in enumerate(sample_catboost_params(args.n_iter, args.random_state), start=1):
        fold_scores = []
        for train_idx, val_idx in cv.split(X_cb):
            model = CatBoostRegressor(
                **params,
                loss_function="RMSE",
                random_seed=args.random_state,
                verbose=False,
            )
            fit_kwargs = {"cat_features": cat_indices} if cat_indices else {}
            model.fit(X_cb.iloc[train_idx], y[train_idx], **fit_kwargs)
            pred = model.predict(X_cb.iloc[val_idx])
            fold_scores.append(rmse(y[val_idx], pred))
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_rmse:
            best_rmse = mean_score
            best_params = params
        if i == 1 or i % 10 == 0 or i == args.n_iter:
            print(f"  iter {i:>3}/{args.n_iter}: best_cv_rmse={best_rmse:.4f}")

    if best_params is None:
        raise RuntimeError("CatBoost search did not produce parameters.")
    model = CatBoostRegressor(
        **best_params,
        loss_function="RMSE",
        random_seed=args.random_state,
        verbose=False,
    )
    fit_kwargs = {"cat_features": cat_indices} if cat_indices else {}
    model.fit(X_cb, y, **fit_kwargs)
    preds = model.predict(X_cb)
    return SearchResult(
        model_key="catboost",
        estimator=model,
        best_params=best_params,
        cv_rmse=best_rmse,
        train_mae=float(mean_absolute_error(y, preds)),
        train_rmse=rmse(y, preds),
        train_r2=float(r2_score(y, preds)),
        train_spearman=safe_spearman(y, preds),
    )


def fit_searches(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, SearchResult]:
    X, y = split_xy(train_frame, feature_cols, cat_cols)
    results: dict[str, SearchResult] = {}
    for model_key in args.models:
        if model_key == "catboost":
            results[model_key] = fit_catboost_search(X, y, cat_cols, args)
        else:
            results[model_key] = fit_sklearn_search(model_key, X, y, num_cols, cat_cols, args)
    return results


def fit_from_search_result(
    result: SearchResult,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> Any:
    X, y = split_xy(train_frame, feature_cols, cat_cols)
    if result.model_key == "catboost":
        X_cb = prepare_catboost_X(X, cat_cols)
        model = CatBoostRegressor(
            **result.best_params,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        cat_indices = [list(X_cb.columns).index(col) for col in cat_cols if col in X_cb.columns]
        fit_kwargs = {"cat_features": cat_indices} if cat_indices else {}
        model.fit(X_cb, y, **fit_kwargs)
        return model

    model = clone(result.estimator)
    model.fit(X, y)
    return model


def predict_model(model_key: str, model: Any, X: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    if model_key == "catboost":
        return np.asarray(model.predict(prepare_catboost_X(X, cat_cols)), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def bound_av_predictions(
    preds: np.ndarray,
    train: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, int]:
    """Keep model outputs in the feasible 2-year AV range seen at train time."""
    if args.no_prediction_clipping:
        return preds, float("nan"), 0

    upper = float(train["av_2yr"].quantile(args.prediction_upper_quantile))
    if not np.isfinite(upper) or upper <= 0:
        upper = float(train["av_2yr"].max())
    bounded = np.clip(preds, 0.0, upper)
    n_clipped = int(np.sum(~np.isclose(preds, bounded, equal_nan=True)))
    return bounded, upper, n_clipped


def add_pick_bin(df: pd.DataFrame, bin_size: int) -> pd.Series:
    pick = pd.to_numeric(df["pick"], errors="coerce").fillna(999).astype(int)
    return ((pick - 1) // bin_size) * bin_size + 1


def predict_pick_bin_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bin_size: int,
) -> np.ndarray:
    """Predict from historical AV for nearby pick bins, with round/global fallbacks."""
    train = train.copy()
    test = test.copy()
    train["_pick_bin_start"] = add_pick_bin(train, bin_size)
    test["_pick_bin_start"] = add_pick_bin(test, bin_size)

    bin_mean = train.groupby("_pick_bin_start")["av_2yr"].mean()
    overall_mean = float(train["av_2yr"].mean())
    preds = test["_pick_bin_start"].map(bin_mean).astype(float)

    if "round" in train.columns and "round" in test.columns:
        round_mean = train.groupby("round")["av_2yr"].mean()
        missing = preds.isna()
        if missing.any():
            preds.loc[missing] = test.loc[missing, "round"].map(round_mean).astype(float)

    return preds.fillna(overall_mean).to_numpy(dtype=float)


def resolve_first_test_year(complete: pd.DataFrame, args: argparse.Namespace) -> int:
    years = sorted(int(year) for year in complete["draft_season"].unique())
    if not years:
        raise ValueError("No complete target years are available for walk-forward evaluation.")
    if args.first_test_year is not None:
        return int(args.first_test_year)
    return years[min(args.min_train_years, len(years) - 1)]


def walk_forward(
    frame: pd.DataFrame,
    search_results: dict[str, SearchResult],
    feature_cols: list[str],
    cat_cols: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = frame[frame["target_complete"] & frame["av_2yr"].notna()].copy()
    years = sorted(complete["draft_season"].unique())
    first_test_year = resolve_first_test_year(complete, args)
    max_test_year = args.max_test_year or max(years)

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    print("\n[Walk-forward] Expanding-window evaluation ...")
    for test_year in years:
        if test_year < first_test_year or test_year > max_test_year:
            continue
        train = complete[complete["draft_season"] < test_year].copy()
        test = complete[complete["draft_season"] == test_year].copy()
        if train["draft_season"].nunique() < args.min_train_years or test.empty:
            continue

        X_test, y_test = split_xy(test, feature_cols, cat_cols)
        base_cols = [
            "draft_pick_id",
            "draft_season",
            "pick",
            "round",
            "team",
            "position",
            "position_group",
            "college",
            "pfr_player_name",
            "av_2yr",
        ]
        year_pred = test[[col for col in base_cols if col in test.columns]].copy()

        baseline_preds = predict_pick_bin_baseline(train, test, args.pick_bin_size)
        year_pred[f"pred_{PICK_BIN_BASELINE_KEY}"] = baseline_preds
        metric_rows.append(
            {
                "model": RESULT_LABELS[PICK_BIN_BASELINE_KEY],
                "model_key": PICK_BIN_BASELINE_KEY,
                "test_year": int(test_year),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "prediction_clip_upper": float("nan"),
                "n_predictions_clipped": 0,
                "mae": float(mean_absolute_error(y_test, baseline_preds)),
                "rmse": rmse(y_test, baseline_preds),
                "r2": float(r2_score(y_test, baseline_preds)),
                "spearman": safe_spearman(y_test, baseline_preds),
            }
        )
        print(
            f"  {test_year} {RESULT_LABELS[PICK_BIN_BASELINE_KEY]:<16} "
            f"RMSE={metric_rows[-1]['rmse']:.3f} "
            f"R2={metric_rows[-1]['r2']:.3f} "
            f"Spearman={metric_rows[-1]['spearman']:.3f}"
        )

        for model_key, result in search_results.items():
            model = fit_from_search_result(result, train, feature_cols, cat_cols)
            raw_preds = predict_model(model_key, model, X_test, cat_cols)
            preds, clip_upper, n_clipped = bound_av_predictions(raw_preds, train, args)
            year_pred[f"pred_{model_key}"] = preds
            year_pred[f"raw_pred_{model_key}"] = raw_preds
            metric_rows.append(
                {
                    "model": MODEL_LABELS[model_key],
                    "model_key": model_key,
                    "test_year": int(test_year),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "prediction_clip_upper": clip_upper,
                    "n_predictions_clipped": n_clipped,
                    "mae": float(mean_absolute_error(y_test, preds)),
                    "rmse": rmse(y_test, preds),
                    "r2": float(r2_score(y_test, preds)),
                    "spearman": safe_spearman(y_test, preds),
                }
            )
            print(
                f"  {test_year} {MODEL_LABELS[model_key]:<13} "
                f"RMSE={metric_rows[-1]['rmse']:.3f} "
                f"R2={metric_rows[-1]['r2']:.3f} "
                f"Spearman={metric_rows[-1]['spearman']:.3f}"
            )
        pred_rows.append(year_pred)

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def summarize_overall(walk_metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key, label in RESULT_LABELS.items():
        pred_col = f"pred_{model_key}"
        if pred_col not in predictions.columns:
            continue
        y = predictions["av_2yr"].to_numpy(dtype=float)
        pred = predictions[pred_col].to_numpy(dtype=float)
        rows.append(
            {
                "model": label,
                "model_key": model_key,
                "n_predictions": int(len(predictions)),
                "mae": float(mean_absolute_error(y, pred)),
                "rmse": rmse(y, pred),
                "r2": float(r2_score(y, pred)),
                "spearman": safe_spearman(y, pred),
                "mean_yearly_rmse": float(
                    walk_metrics.loc[walk_metrics["model_key"] == model_key, "rmse"].mean()
                ),
                "mean_yearly_r2": float(
                    walk_metrics.loc[walk_metrics["model_key"] == model_key, "r2"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)


def save_search_summary(search_results: dict[str, SearchResult], out_dir: Path) -> pd.DataFrame:
    rows = []
    for result in search_results.values():
        rows.append(
            {
                "model": MODEL_LABELS[result.model_key],
                "model_key": result.model_key,
                "cv_rmse": result.cv_rmse,
                "train_mae": result.train_mae,
                "train_rmse": result.train_rmse,
                "train_r2": result.train_r2,
                "train_spearman": result.train_spearman,
                "best_params": json.dumps(result.best_params, default=json_default),
            }
        )
    df = pd.DataFrame(rows).sort_values("cv_rmse")
    df.to_csv(out_dir / "tables" / "grid_search_summary.csv", index=False)
    (out_dir / "tables" / "grid_search_summary.json").write_text(
        json.dumps(rows, indent=2, default=json_default),
        encoding="utf-8",
    )
    return df


def save_models(search_results: dict[str, SearchResult], out_dir: Path) -> None:
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for result in search_results.values():
        if result.model_key == "catboost":
            path = model_dir / "catboost_best.cbm"
            result.estimator.save_model(str(path))
        else:
            joblib.dump(result.estimator, model_dir / f"{result.model_key}_best.pkl")


def refit_search_results_on_frame(
    search_results: dict[str, SearchResult],
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> None:
    for result in search_results.values():
        result.estimator = fit_from_search_result(result, train_frame, feature_cols, cat_cols)


def build_reference_grid(X_train: pd.DataFrame, cat_cols: list[str], picks: np.ndarray) -> pd.DataFrame:
    reference: dict[str, Any] = {}
    for col in X_train.columns:
        if col in cat_cols:
            mode = X_train[col].dropna().mode()
            reference[col] = mode.iloc[0] if len(mode) else "_missing_"
        else:
            med = pd.to_numeric(X_train[col], errors="coerce").median()
            reference[col] = 0.0 if pd.isna(med) else float(med)

    grid = pd.DataFrame([reference for _ in picks])
    grid["pick"] = picks
    if "round" in grid.columns:
        grid["round"] = np.ceil(picks / 32).clip(1, 8)
    return grid


def compute_pick_value_curves(
    frame: pd.DataFrame,
    search_results: dict[str, SearchResult],
    feature_cols: list[str],
    cat_cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    complete = frame[frame["target_complete"] & frame["av_2yr"].notna()].copy()
    complete_years = sorted(complete["draft_season"].unique())
    default_years = [
        year for year in [2010, 2015, 2020, max(complete_years)] if year in complete_years
    ]
    curve_years = args.curve_years or default_years
    picks = np.arange(1, 261)
    rows: list[pd.DataFrame] = []

    for cutoff in curve_years:
        train = complete[complete["draft_season"] <= cutoff].copy()
        if train.empty:
            continue
        X_train, _ = split_xy(train, feature_cols, cat_cols)
        grid = build_reference_grid(X_train, cat_cols, picks)
        baseline_preds = predict_pick_bin_baseline(train, grid, args.pick_bin_size)
        rows.append(
            pd.DataFrame(
                {
                    "train_through_year": cutoff,
                    "pick": picks,
                    "model": RESULT_LABELS[PICK_BIN_BASELINE_KEY],
                    "model_key": PICK_BIN_BASELINE_KEY,
                    "predicted_av_2yr": baseline_preds,
                }
            )
        )
        for model_key, result in search_results.items():
            model = fit_from_search_result(result, train, feature_cols, cat_cols)
            raw_preds = predict_model(model_key, model, grid, cat_cols)
            preds, _, _ = bound_av_predictions(raw_preds, train, args)
            rows.append(
                pd.DataFrame(
                    {
                        "train_through_year": cutoff,
                        "pick": picks,
                        "model": MODEL_LABELS[model_key],
                        "model_key": model_key,
                        "predicted_av_2yr": preds,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_actual_pick_curve(frame: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    complete = frame[frame["target_complete"] & frame["av_2yr"].notna()].copy()
    max_pick = int(complete["pick"].max())
    bins = np.arange(1, max_pick + bin_size + 1, bin_size)
    complete["pick_bin"] = pd.cut(complete["pick"], bins=bins, right=False, include_lowest=True)
    grouped = (
        complete.groupby("pick_bin", observed=True)
        .agg(
            pick_min=("pick", "min"),
            pick_max=("pick", "max"),
            pick_center=("pick", "median"),
            n_players=("av_2yr", "size"),
            mean_av_2yr=("av_2yr", "mean"),
            median_av_2yr=("av_2yr", "median"),
        )
        .reset_index(drop=True)
    )
    grouped["rolling_mean_av_2yr"] = (
        grouped["mean_av_2yr"].rolling(window=3, center=True, min_periods=1).mean()
    )
    return grouped


def plot_model_comparison(overall: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), dpi=220)
    metrics = [
        ("mae", "MAE", True),
        ("rmse", "RMSE", True),
        ("r2", "R2", False),
        ("spearman", "Spearman", False),
    ]
    for ax, (metric, title, lower_is_better) in zip(axes.flat, metrics):
        data = overall.sort_values(metric, ascending=lower_is_better)
        colors = [MODEL_COLORS.get(key, "#777777") for key in data["model_key"]]
        ax.barh(data["model"], data[metric], color=colors)
        ax.invert_yaxis()
        ax.set_title(f"{title} ({'lower' if lower_is_better else 'higher'} is better)", fontweight="bold")
        ax.axvline(0, color="#222222", linewidth=0.9, alpha=0.35)
        offset = 0.03 * max(float(data[metric].abs().max()), 1.0)
        for i, value in enumerate(data[metric]):
            ha = "left" if value >= 0 else "right"
            ax.text(
                value + (offset if value >= 0 else -offset),
                i,
                f"{value:.2f}",
                va="center",
                ha=ha,
                fontsize=8,
                clip_on=False,
            )
        ax.margins(x=0.16)
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Final Model Comparison - Walk-Forward Pooled Predictions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_comparison_overall.png", bbox_inches="tight")
    plt.close(fig)


def plot_walkforward_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=220)
    for ax, metric in zip(axes.flat, ["mae", "rmse", "r2", "spearman"]):
        for model_key, label in RESULT_LABELS.items():
            sub = metrics[metrics["model_key"] == model_key]
            if sub.empty:
                continue
            ax.plot(
                sub["test_year"],
                sub[metric],
                marker="o",
                linewidth=1.9,
                color=MODEL_COLORS.get(model_key, "#777777"),
                label=label,
            )
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.25)
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xlabel("Draft year")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Expanding-Window Walk-Forward Metrics", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_walkforward_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(predictions: pd.DataFrame, overall: pd.DataFrame, out_dir: Path) -> None:
    model_keys = [key for key in MODEL_LABELS if f"pred_{key}" in predictions.columns]
    n = len(model_keys)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.6 * rows), dpi=220)
    axes_arr = np.atleast_1d(axes).flat
    y = predictions["av_2yr"].to_numpy(dtype=float)
    for ax, model_key in zip(axes_arr, model_keys):
        pred = predictions[f"pred_{model_key}"].to_numpy(dtype=float)
        row = overall[overall["model_key"] == model_key].iloc[0]
        color = MODEL_COLORS.get(model_key, "#777777")
        ax.scatter(y, pred, s=16, alpha=0.35, color=color, edgecolor="none")
        lim = max(float(np.nanmax(y)), float(np.nanmax(pred))) * 1.08
        ax.plot([0, lim], [0, lim], color="#111111", linestyle="--", linewidth=1.1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(RESULT_LABELS[model_key], fontweight="bold")
        ax.set_xlabel("Actual 2-Year AV")
        ax.set_ylabel("Predicted 2-Year AV")
        ax.text(
            0.03,
            0.97,
            f"MAE={row['mae']:.2f}\nRMSE={row['rmse']:.2f}\nR2={row['r2']:.2f}\nSpearman={row['spearman']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#dddddd"),
        )
        ax.grid(alpha=0.22)
    for ax in list(axes_arr)[n:]:
        ax.axis("off")
    fig.suptitle("Walk-Forward Predicted vs Actual", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_predicted_vs_actual_models_2x2.png", bbox_inches="tight")
    plt.close(fig)

    baseline_key = PICK_BIN_BASELINE_KEY
    baseline_col = f"pred_{baseline_key}"
    if baseline_col in predictions.columns:
        fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=220)
        pred = predictions[baseline_col].to_numpy(dtype=float)
        row = overall[overall["model_key"] == baseline_key].iloc[0]
        y = predictions["av_2yr"].to_numpy(dtype=float)
        lim = max(float(np.nanmax(y)), float(np.nanmax(pred))) * 1.08
        ax.scatter(y, pred, s=16, alpha=0.35, color=MODEL_COLORS[baseline_key], edgecolor="none")
        ax.plot([0, lim], [0, lim], color="#111111", linestyle="--", linewidth=1.1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(RESULT_LABELS[baseline_key], fontweight="bold")
        ax.set_xlabel("Actual 2-Year AV")
        ax.set_ylabel("Predicted 2-Year AV")
        ax.text(
            0.03,
            0.97,
            f"MAE={row['mae']:.2f}\nRMSE={row['rmse']:.2f}\nR2={row['r2']:.2f}\nSpearman={row['spearman']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#dddddd"),
        )
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_predicted_vs_actual_pick_bin_baseline.png", bbox_inches="tight")
        plt.close(fig)


def plot_residuals(predictions: pd.DataFrame, out_dir: Path) -> None:
    model_keys = [key for key in RESULT_LABELS if f"pred_{key}" in predictions.columns]
    cols = 2
    rows = math.ceil(len(model_keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 * rows), dpi=220)
    axes_arr = np.atleast_1d(axes).flat
    y = predictions["av_2yr"].to_numpy(dtype=float)
    for ax, model_key in zip(axes_arr, model_keys):
        resid = y - predictions[f"pred_{model_key}"].to_numpy(dtype=float)
        sns.histplot(resid, bins=35, kde=True, ax=ax, color=MODEL_COLORS.get(model_key, "#777777"))
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{RESULT_LABELS[model_key]} residuals", fontweight="bold")
        ax.set_xlabel("Actual minus predicted AV")
    for ax in list(axes_arr)[len(model_keys):]:
        ax.axis("off")
    fig.suptitle("Residual Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_residual_distributions.png", bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(frame: pd.DataFrame, out_dir: Path) -> None:
    complete = frame[frame["target_complete"] & frame["av_2yr"].notna()]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=220)
    sns.histplot(complete["av_2yr"], bins=40, kde=True, ax=ax, color="#477ca8")
    ax.set_title("2-Year Approximate Value Distribution", fontweight="bold")
    ax.set_xlabel("2-Year AV")
    ax.set_ylabel("Players")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_actual_pick_curve(actual_curve: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=220)
    ax.scatter(
        actual_curve["pick_center"],
        actual_curve["mean_av_2yr"],
        s=np.clip(actual_curve["n_players"] / 2, 18, 85),
        color="#9fbad0",
        edgecolor="#335f7c",
        alpha=0.85,
        label="Pick-bin mean",
    )
    ax.plot(
        actual_curve["pick_center"],
        actual_curve["rolling_mean_av_2yr"],
        color="#143d59",
        linewidth=2.6,
        label="Smoothed mean",
    )
    ax.set_title("Empirical Pick-Value Curve", fontweight="bold")
    ax.set_xlabel("Draft pick")
    ax.set_ylabel("Mean 2-Year AV")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_actual_pick_value_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_pick_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    if curves.empty:
        return
    model_keys = [key for key in MODEL_LABELS if key in set(curves["model_key"])]
    cols = 2
    rows = math.ceil(len(model_keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.8 * rows), dpi=220)
    axes_arr = np.atleast_1d(axes).flat
    for ax, model_key in zip(axes_arr, model_keys):
        sub = curves[curves["model_key"] == model_key]
        for year, year_df in sub.groupby("train_through_year"):
            ax.plot(year_df["pick"], year_df["predicted_av_2yr"], linewidth=1.7, label=str(year))
        ax.set_title(RESULT_LABELS[model_key], fontweight="bold")
        ax.set_xlabel("Draft pick")
        ax.set_ylabel("Predicted 2-Year AV")
        ax.grid(alpha=0.25)
    for ax in list(axes_arr)[len(model_keys):]:
        ax.axis("off")
    axes_arr[0].legend(title="Train through", fontsize=8)
    fig.suptitle("Model Pick-Value Curve Approximation Over Time", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_pick_value_curves_over_time.png", bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(
    search_results: dict[str, SearchResult],
    out_dir: Path,
    top_n: int,
) -> None:
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    for key in ["xgboost", "catboost"]:
        result = search_results.get(key)
        if result is None:
            continue
        try:
            if key == "xgboost":
                pre = result.estimator.named_steps["pre"]
                model = result.estimator.named_steps["model"]
                names = [name.split("__", 1)[-1] for name in pre.get_feature_names_out()]
                importance = model.feature_importances_
            else:
                model = result.estimator
                names = list(model.feature_names_)
                importance = model.get_feature_importance()
            imp = (
                pd.DataFrame({"feature": names, "importance": importance})
                .sort_values("importance", ascending=False)
                .head(top_n)
            )
            imp.to_csv(table_dir / f"feature_importance_{key}.csv", index=False)
            fig, ax = plt.subplots(figsize=(9, 7), dpi=220)
            ax.barh(imp["feature"][::-1], imp["importance"][::-1], color=MODEL_COLORS[key])
            ax.set_title(f"{MODEL_LABELS[key]} Feature Importance", fontweight="bold")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig_feature_importance_{key}.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] Could not save {MODEL_LABELS[key]} feature importance: {exc}")


def make_plots(
    frame: pd.DataFrame,
    walk_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    actual_curve: pd.DataFrame,
    model_curves: pd.DataFrame,
    out_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig_dir = out_dir / "figures"
    plot_model_comparison(overall, fig_dir)
    plot_walkforward_metrics(walk_metrics, fig_dir)
    plot_predicted_vs_actual(predictions, overall, fig_dir)
    plot_residuals(predictions, fig_dir)
    plot_target_distribution(frame, fig_dir)
    plot_actual_pick_curve(actual_curve, fig_dir)
    plot_model_pick_curves(model_curves, fig_dir)


def write_data_artifacts(
    tables: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
    quality: dict[str, Any],
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    out_dir: Path,
) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    id_cols = [
        "draft_pick_id",
        "draft_season",
        "pfr_player_id",
        "pfr_player_name",
        "av_2yr",
        "target_complete",
        "label_year_max",
    ]
    safe_cols = list(dict.fromkeys([col for col in id_cols + feature_cols if col in frame.columns]))
    frame[safe_cols].to_csv(table_dir / "model_frame.csv", index=False)
    source_table_quality(tables).to_csv(table_dir / "source_table_quality.csv", index=False)
    (table_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "feature_count": len(feature_cols),
                "numeric_feature_count": len(num_cols),
                "categorical_feature_count": len(cat_cols),
                "features": feature_cols,
                "numeric_features": num_cols,
                "categorical_features": cat_cols,
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    (table_dir / "data_quality_summary.json").write_text(
        json.dumps(
            {
                **quality,
                "feature_count": len(feature_cols),
                "numeric_feature_count": len(num_cols),
                "categorical_feature_count": len(cat_cols),
                "numeric_features": num_cols,
                "categorical_features": cat_cols,
                "feature_groups": {
                    "draft": [col for col in feature_cols if not col.startswith(("cfb_", "board_", "context_"))],
                    "college_stats": [col for col in feature_cols if col.startswith("cfb_")],
                    "draft_board": [col for col in feature_cols if col.startswith("board_")],
                    "roster_trade_context": [col for col in feature_cols if col.startswith("context_")],
                },
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )


def write_run_readme(out_dir: Path, best_model: str, args: argparse.Namespace) -> None:
    selected = set(args.models)
    is_full_final = selected == set(ALL_MODELS) and not args.allow_partial_model_run
    run_label = "FULL FINAL RUN" if is_full_final else "PARTIAL DEVELOPMENT RUN"
    text = f"""# Final Poster Modeling Run

Run status: **{run_label}**

Generated by:

```bash
python src/final_poster_modeling/train_final_poster_models.py
```

Best model by pooled walk-forward RMSE: **{best_model}**.

## Contents

- `tables/model_frame.csv`: cleaned modeling frame with target and feature inputs.
- `tables/source_table_quality.csv`: basic row, duplicate-key, missing-value, and nonfinite checks for source exports.
- `tables/feature_columns.json`: exact feature columns passed into the models.
- `tables/grid_search_summary.csv`: CV/search metrics and selected hyperparameters.
- `tables/walkforward_metrics.csv`: year-by-year walk-forward metrics.
- `tables/overall_summary.csv`: pooled walk-forward performance by model.
- `tables/walkforward_predictions.csv`: player-level walk-forward predictions.
- `tables/actual_pick_value_curve.csv`: empirical pick-bin value curve.
- `tables/model_pick_value_curves.csv`: model-implied pick curves through selected training years.
- `figures/`: poster-ready plots.
- `models/`: fitted best estimators from the grid search stage.

## Modeling Notes

The frame joins draft info, college aggregate stats, prior-season roster context,
traded-pick context, and draft-board features. Raw roster/trade files are not
joined directly into the model because `draft_pick_context_features.csv` already
compresses them to one pick-safe row. Post-draft outcome columns are removed
from features. Walk-forward evaluation trains on all complete draft classes
before the test year and evaluates on that single draft class.

The tuned poster models are XGBoost, CatBoost, MLP Neural Net, and Spline Ridge.
`Pick-Bin Baseline` is included as a simple historical comparison, but it is not
eligible to be named the tuned best model.
MLP Neural Net and Spline Ridge use a nonnegative log1p target transform to
better handle the heavily right-skewed AV distribution.

Hyperparameters are tuned only on complete draft classes before the first
walk-forward test year. Each walk-forward year then refits on only prior draft
classes. Saved model artifacts are refit on all complete labels after evaluation.
Predictions are clipped to the feasible 2-year AV range observed in the
training window unless `--no-prediction-clipping` is supplied.

Configured models: `{", ".join(args.models)}`.
First walk-forward test year: `{args.first_test_year}`.
Prediction upper quantile: `{args.prediction_upper_quantile}`.
Random search iterations per model: `{args.n_iter}`.
CV folds: `{args.cv_folds}`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    prepare_output_dir(args.output_dir)

    tables = load_exports(args.data_dir)
    frame, quality = build_model_frame(tables, args.horizon)
    feature_cols, num_cols, cat_cols = select_feature_columns(frame)

    complete = frame[frame["target_complete"] & frame["av_2yr"].notna()].copy()
    if args.max_test_year is not None:
        complete = complete[complete["draft_season"] <= args.max_test_year].copy()

    write_data_artifacts(tables, frame, quality, feature_cols, num_cols, cat_cols, args.output_dir)

    print("Data ready:")
    print(f"  frame rows: {len(frame):,}")
    print(f"  complete target rows: {len(complete):,}")
    print(f"  features: {len(feature_cols):,} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"  complete years: {complete['draft_season'].min()}-{complete['draft_season'].max()}")
    print(f"  source rows: {quality['source_table_rows']}")

    if args.prepare_only:
        print("\nData preparation complete. Training was skipped because --prepare-only was set.")
        print(f"  Outputs: {args.output_dir}")
        return 0

    first_test_year = resolve_first_test_year(complete, args)
    tuning_frame = complete[complete["draft_season"] < first_test_year].copy()
    if tuning_frame["draft_season"].nunique() < args.min_train_years:
        raise SystemExit(
            f"Not enough pre-{first_test_year} complete seasons for tuning. "
            f"Found {tuning_frame['draft_season'].nunique()} years; need at least {args.min_train_years}."
        )
    print(
        f"\n[Tuning] Hyperparameter search uses {len(tuning_frame):,} rows from "
        f"{tuning_frame['draft_season'].min()}-{tuning_frame['draft_season'].max()}."
    )
    print(f"[Evaluation] Walk-forward test years start at {first_test_year}.")

    search_results = fit_searches(tuning_frame, feature_cols, num_cols, cat_cols, args)
    save_search_summary(search_results, args.output_dir)

    walk_metrics, predictions = walk_forward(frame, search_results, feature_cols, cat_cols, args)
    overall = summarize_overall(walk_metrics, predictions)
    tuned_overall = overall[overall["model_key"].isin(search_results.keys())].copy()
    best_row = tuned_overall.iloc[0]
    best_model_key = str(best_row["model_key"])
    best_model_label = str(best_row["model"])

    walk_metrics.to_csv(args.output_dir / "tables" / "walkforward_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "tables" / "walkforward_predictions.csv", index=False)
    overall.to_csv(args.output_dir / "tables" / "overall_summary.csv", index=False)

    actual_curve = compute_actual_pick_curve(frame, args.pick_bin_size)
    actual_curve.to_csv(args.output_dir / "tables" / "actual_pick_value_curve.csv", index=False)
    model_curves = compute_pick_value_curves(frame, search_results, feature_cols, cat_cols, args)
    model_curves.to_csv(args.output_dir / "tables" / "model_pick_value_curves.csv", index=False)

    # After honest walk-forward evaluation, refit the winning hyperparameters
    # on all complete labels so the saved artifacts are useful for poster demos.
    refit_search_results_on_frame(search_results, complete, feature_cols, cat_cols)
    save_models(search_results, args.output_dir)

    # Save a copy of the winning final refit model in a prominent folder.
    best_result = search_results[best_model_key]
    if best_model_key == "catboost":
        best_result.estimator.save_model(str(args.output_dir / "best_model" / "best_model_catboost.cbm"))
    else:
        joblib.dump(best_result.estimator, args.output_dir / "best_model" / f"best_model_{best_model_key}.pkl")
    (args.output_dir / "best_model" / "best_model_summary.json").write_text(
        json.dumps(best_row.to_dict(), indent=2, default=json_default),
        encoding="utf-8",
    )

    if not args.skip_plots:
        make_plots(frame, walk_metrics, predictions, overall, actual_curve, model_curves, args.output_dir)
        save_feature_importance(search_results, args.output_dir, args.top_n_features)

    write_run_readme(args.output_dir, best_model_label, args)
    print("\nFinal poster modeling complete.")
    print(f"  Best model: {best_model_label}")
    print(f"  Outputs: {args.output_dir}")
    print("\nOverall summary:")
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
