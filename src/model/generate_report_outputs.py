"""Generate report-ready tables and visualizations from model outputs.

This is a post-processing script: it expects ``python -m src.model.train_n_evaluate``
to have produced the model_v6 CSVs/PNGs in ``results/``. It intentionally writes
everything into one clean report folder.

Run:
    python -m src.model.generate_report_outputs
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data" / "supabase_exports"
REPORT_DIR = ROOT / "reports" / "final_report_modeling" / "latest"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
VECTOR_FIGURE_DIR = REPORT_DIR / "figures_vector"
GOOGLE_DOC_FIGURE_DIR = REPORT_DIR / "figures_google_doc"

MODEL_ORDER = ["spline", "xgb", "rf", "ftt", "pick_bin"]
REQUESTED_MODELS = ["pick_bin", "xgb", "rf", "ftt", "spline"]
MODEL_LABELS = {
    "spline": "Spline Ridge",
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "ftt": "FT-Transformer",
    "pick_bin": "Pick-Bin Baseline",
}
MODEL_COLORS = {
    "spline": "#3B6FB6",
    "xgb": "#1F7A5A",
    "rf": "#D5672A",
    "ftt": "#264653",
    "pick_bin": "#6C757D",
}
METRIC_ORDER = ["mae", "rmse", "spearman", "r2"]
PICK_BIN_SIZE = 16
APPENDIX_SELECTED_YEARS = [2007, 2011, 2015, 2019, 2024]
PICK_REGION_BANDS = [
    (1, 32, "Round 1"),
    (33, 100, "Rounds 2-3"),
    (101, 262, "Day 3"),
]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGURE_CAPTIONS = {
    "overall_model_performance_grid.png": (
        "Overall out-of-sample model performance across all walk-forward test years. "
        "MAE and RMSE summarize absolute prediction error, Spearman measures ranking quality, "
        "and R² measures variance explained."
    ),
    "walkforward_metrics_requested_models.png": (
        "Year-by-year walk-forward performance for the requested models. Each point is a held-out "
        "draft class, trained only on prior draft classes."
    ),
    "walkforward_rmse_teaser_requested_models.png": (
        "Year-by-year walk-forward RMSE for the requested models. This report-facing teaser focuses "
        "on absolute prediction error; the full MAE, Spearman, and R² metric grid is included in the appendix."
    ),
    "pred_vs_actual_grid_requested_models_2024.png": (
        "Predicted versus actual 2-year AV for the 2024 held-out draft class. Points closer to the "
        "diagonal represent more accurate predictions."
    ),
    "pred_vs_actual_grid_requested_models_all_years.png": (
        "Predicted versus actual 2-year AV across all walk-forward held-out draft classes. This "
        "shows the full out-of-sample prediction pattern by model."
    ),
    "pick_value_curves_oos_by_model.png": (
        "Report-ready draft pick-value comparison. The black curve is realized average 2-year AV by "
        "draft slot, while the colored curves show the Pick-Bin baseline and tree-model estimates."
    ),
    "pick_value_curves_model_facets.png": (
        "Appendix view of pick-value curves by model. Each panel compares one model's out-of-sample "
        "estimated AV curve with the same realized empirical draft value curve."
    ),
    "pick_value_curves_exact_pick_no_bins.png": (
        "Exact-pick value comparison using overall pick numbers. Points show realized average AV at "
        "each pick; curves use exact picks with enough history and monotonic smoothing for readability."
    ),
    "pick_value_curve_monotonic_report.png": (
        "Empirical draft pick-value curve. Gray points are observed 16-pick averages and the black "
        "line is a monotonic smooth showing the overall decline in expected 2-year AV as picks get later."
    ),
    "mae_by_pick_bin_requested_models.png": (
        "Mean absolute error by draft pick range. This diagnostic shows where each model is most "
        "and least accurate across the draft board."
    ),
    "calibration_by_prediction_decile_requested_models.png": (
        "Calibration by prediction decile. Curves close to the diagonal indicate that predicted "
        "AV levels match realized average AV levels."
    ),
    "residual_distributions_requested_models.png": (
        "Residual distributions by model, where residual equals actual AV minus predicted AV. "
        "Centering near zero indicates less systematic over- or under-prediction."
    ),
    "prediction_distribution_2024.png": (
        "Distribution of actual 2024 2-year AV compared with each model's predicted distribution. "
        "This shows whether models reproduce the skew and spread of draft outcomes."
    ),
    "feature_importance_xgb_top20.png": (
        "Top 20 XGBoost feature importances from the final training window. Higher bars indicate "
        "features used more strongly by the model."
    ),
    "feature_importance_rf_top20.png": (
        "Top 20 Random Forest feature importances from the final training window. Higher bars "
        "indicate features with larger average contribution to split quality."
    ),
    "tree_feature_importance_top5.png": (
        "Top five feature importances for XGBoost and Random Forest from the final training "
        "window, normalized within each model so the top feature equals 100. This compact version "
        "highlights the clearest tree-model signals for the main report."
    ),
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GOOGLE_DOC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename: str) -> None:
    """Save report figures as high-res PNG plus vector SVG/PDF."""
    png_path = FIGURE_DIR / filename
    stem = Path(filename).stem
    fig.savefig(png_path, bbox_inches="tight", dpi=320)
    fig.savefig(VECTOR_FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(VECTOR_FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")


def save_google_doc_figure(fig, filename: str) -> None:
    """Save a page-sized PNG that survives Google Docs downscaling better."""
    fig.savefig(GOOGLE_DOC_FIGURE_DIR / filename, bbox_inches="tight", dpi=450)


def nice_axes(ax) -> None:
    ax.grid(True, color="#E6EBEF", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB7C4")
    ax.spines["bottom"].set_color("#AAB7C4")


def add_pick_region_bands(ax) -> None:
    for i, (start, end, label) in enumerate(PICK_REGION_BANDS):
        if i % 2 == 0:
            ax.axvspan(start, end, color="#F5F7FA", alpha=0.65, zorder=0)
        ax.axvline(end + 0.5, color="#DDE5ED", linewidth=0.8, zorder=1)
        ax.text(
            (start + end) / 2,
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4B5563",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
        )


def eval_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {m: np.nan for m in METRIC_ORDER}
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "r2": r2_score(y_true, y_pred),
    }


def set_padded_ylim(ax, values, *, include_zero: bool = False, floor: float | None = None, ceiling: float | None = None) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    pad = span * 0.12
    bottom = lo - pad
    top = hi + pad
    if include_zero:
        bottom = min(bottom, 0.0)
        top = max(top, 0.0)
    if floor is not None:
        bottom = floor
    if ceiling is not None:
        top = ceiling
    ax.set_ylim(bottom, top)


def monotonic_smooth(x: pd.Series, y: pd.Series) -> np.ndarray:
    if len(x) < 3:
        return y.to_numpy(dtype=float)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    return iso.fit_transform(x.to_numpy(dtype=float), y.to_numpy(dtype=float))


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    walk = pd.read_csv(RESULTS_DIR / "model_v6_walkforward_results.csv")
    overall = pd.read_csv(RESULTS_DIR / "model_v6_overall_summary.csv")
    selected = pd.read_csv(RESULTS_DIR / "model_v6_selected_years_summary.csv")
    all_preds_path = RESULTS_DIR / "model_v6_all_predictions.csv"
    if all_preds_path.exists():
        preds = pd.read_csv(all_preds_path)
    else:
        preds = pd.read_csv(RESULTS_DIR / "model_v6_latest_year_predictions_2024.csv")
    return walk, overall, selected, preds


def walkforward_long(walk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in walk.iterrows():
        for model in MODEL_ORDER:
            if f"{model}_mae" not in walk.columns:
                continue
            rows.append({
                "test_year": int(row["test_year"]),
                "model_type": model,
                "model": MODEL_LABELS[model],
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
                "cfb_match_rate": float(row.get("cfb_match_rate", np.nan)),
                "mae": float(row[f"{model}_mae"]),
                "rmse": float(row[f"{model}_rmse"]),
                "spearman": float(row[f"{model}_spearman"]),
                "r2": float(row[f"{model}_r2"]),
            })
    return pd.DataFrame(rows)


def predictions_long(preds: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    id_cols = [c for c in preds.columns if c not in pred_cols]
    out = preds.melt(
        id_vars=id_cols,
        value_vars=pred_cols,
        var_name="model_type",
        value_name="prediction",
    )
    out["model_type"] = out["model_type"].str.replace("pred_", "", regex=False)
    out["model"] = out["model_type"].map(MODEL_LABELS).fillna(out["model_type"])
    out["residual"] = out["av_2yr"] - out["prediction"]
    out["absolute_error"] = out["residual"].abs()
    return out


def write_tables(walk: pd.DataFrame, overall: pd.DataFrame, selected: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    walk_long = walkforward_long(walk)
    pred_long = predictions_long(preds)

    overall_clean = overall.copy()
    overall_clean["model"] = overall_clean["model_type"].map(MODEL_LABELS).fillna(overall_clean["model_type"])
    overall_clean = overall_clean[
        ["model_type", "model", "n_predictions", "mae", "rmse", "spearman", "r2"]
    ].sort_values("rmse")

    requested_overall = overall_clean[overall_clean["model_type"].isin(REQUESTED_MODELS)].copy()

    pick_bin_rows = []
    pred_long["pick_bin_start"] = ((pred_long["pick"].astype(int) - 1) // PICK_BIN_SIZE) * PICK_BIN_SIZE + 1
    for (model, pick_bin), grp in pred_long.groupby(["model_type", "pick_bin_start"]):
        metrics = eval_metrics(grp["av_2yr"], grp["prediction"])
        pick_bin_rows.append({
            "model_type": model,
            "model": MODEL_LABELS.get(model, model),
            "pick_bin_start": int(pick_bin),
            "pick_bin_end": int(pick_bin + PICK_BIN_SIZE - 1),
            "n": len(grp),
            "actual_mean_av": grp["av_2yr"].mean(),
            "predicted_mean_av": grp["prediction"].mean(),
            **metrics,
        })
    pick_bin_perf = pd.DataFrame(pick_bin_rows)

    walk_long.to_csv(TABLE_DIR / "year_by_year_model_metrics.csv", index=False)
    overall_clean.to_csv(TABLE_DIR / "overall_model_performance.csv", index=False)
    requested_overall.to_csv(TABLE_DIR / "overall_requested_model_performance.csv", index=False)
    selected.to_csv(TABLE_DIR / "selected_years_model_metrics.csv", index=False)
    pred_long.to_csv(TABLE_DIR / "all_out_of_sample_predictions_long.csv", index=False)
    pick_bin_perf.to_csv(TABLE_DIR / "pick_bin_performance_by_model.csv", index=False)

    return pred_long


def round_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    integer_like_cols = {"Year", "N", "N Test"}
    for col in out.columns:
        if col not in integer_like_cols and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(2)
    return out


def write_copy_ready_metric_tables(walk_long: pd.DataFrame, overall: pd.DataFrame) -> None:
    model_rank = {model: i for i, model in enumerate(REQUESTED_MODELS)}

    overall_clean = overall[overall["model_type"].isin(REQUESTED_MODELS)].copy()
    overall_clean["model_order"] = overall_clean["model_type"].map(model_rank)
    overall_clean["Model"] = overall_clean["model_type"].map(MODEL_LABELS)
    overall_clean = overall_clean.sort_values("model_order")
    report_table = overall_clean.rename(columns={
        "n_predictions": "N",
        "mae": "MAE",
        "rmse": "RMSE",
        "spearman": "Spearman",
        "r2": "R2",
    })[["Model", "N", "MAE", "RMSE", "Spearman", "R2"]]
    round_metric_columns(report_table).to_csv(TABLE_DIR / "report_overall_metrics_copy_ready.csv", index=False)

    appendix_long = walk_long[walk_long["model_type"].isin(REQUESTED_MODELS)].copy()
    appendix_long["model_order"] = appendix_long["model_type"].map(model_rank)
    appendix_long = appendix_long.sort_values(["test_year", "model_order"])
    appendix_long = appendix_long.rename(columns={
        "test_year": "Year",
        "model": "Model",
        "n_test": "N Test",
        "mae": "MAE",
        "rmse": "RMSE",
        "spearman": "Spearman",
        "r2": "R2",
    })[["Year", "Model", "N Test", "MAE", "RMSE", "Spearman", "R2"]]
    round_metric_columns(appendix_long).to_csv(TABLE_DIR / "appendix_year_by_year_metrics_copy_ready.csv", index=False)

    metric_labels = [("MAE", "mae"), ("RMSE", "rmse"), ("Spearman", "spearman"), ("R2", "r2")]
    wide_rows = []
    for year in sorted(walk_long["test_year"].unique()):
        year_df = walk_long[(walk_long["test_year"] == year) & (walk_long["model_type"].isin(REQUESTED_MODELS))]
        for metric_label, metric_col in metric_labels:
            row = {"Year": int(year), "Metric": metric_label}
            for model in REQUESTED_MODELS:
                value = year_df.loc[year_df["model_type"] == model, metric_col]
                row[MODEL_LABELS[model]] = float(value.iloc[0]) if not value.empty else np.nan
            wide_rows.append(row)
    appendix_wide_full = pd.DataFrame(wide_rows)
    available_years = set(appendix_wide_full["Year"].unique())
    selected_years = [year for year in APPENDIX_SELECTED_YEARS if year in available_years]
    appendix_wide = appendix_wide_full[appendix_wide_full["Year"].isin(selected_years)].copy()
    round_metric_columns(appendix_wide).to_csv(TABLE_DIR / "appendix_year_metric_wide_copy_ready.csv", index=False)
    round_metric_columns(appendix_wide_full).to_csv(TABLE_DIR / "appendix_year_metric_wide_full_copy_ready.csv", index=False)
    compact_appendix = appendix_wide[appendix_wide["Metric"].isin(["RMSE", "R2"])].copy()
    round_metric_columns(compact_appendix).to_csv(TABLE_DIR / "appendix_rmse_r2_by_year_wide_copy_ready.csv", index=False)


def plot_overall_performance(overall: pd.DataFrame) -> None:
    overall = overall.copy()
    overall["model"] = overall["model_type"].map(MODEL_LABELS).fillna(overall["model_type"])
    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("spearman", "Spearman"),
        ("r2", "R²"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=220)
    axes = axes.flatten()
    for ax, (metric, label) in zip(axes, metrics):
        df = overall.sort_values(metric, ascending=metric in {"mae", "rmse"})
        colors = [MODEL_COLORS.get(m, "#777777") for m in df["model_type"]]
        ax.barh(df["model"], df[metric], color=colors, alpha=0.9)
        ax.set_title(label, fontsize=12, weight="bold")
        ax.set_xlabel(label)
        ax.tick_params(axis="y", labelsize=9)
        if metric in {"mae", "rmse", "spearman"}:
            ax.set_xlim(left=0)
        if metric == "spearman":
            ax.set_xlim(0, 1)
        if metric == "r2":
            ax.axvline(0, color="#111827", linewidth=1.0, alpha=0.75)
        nice_axes(ax)
    fig.suptitle("Overall Out-of-Sample Model Performance", fontsize=16, weight="bold", y=0.99)
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.96])
    save_figure(fig, "overall_model_performance_grid.png")
    plt.close(fig)


def plot_walkforward_metrics(walk_long: pd.DataFrame) -> None:
    plot_df = walk_long[walk_long["model_type"].isin(REQUESTED_MODELS)].copy()
    years = sorted(plot_df["test_year"].unique())
    tick_years = [year for year in years if year % 2 == 0] or years
    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("spearman", "Spearman"),
        ("r2", "R²"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), dpi=220, sharex=True)
    axes = axes.flatten()
    for ax, (metric, label) in zip(axes, metrics):
        for model in REQUESTED_MODELS:
            sub = plot_df[plot_df["model_type"] == model]
            ax.plot(
                sub["test_year"],
                sub[metric],
                marker="o",
                linewidth=2.1,
                markersize=4.3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set_title(label, fontsize=12, weight="bold")
        ax.set_ylabel(label)
        if years:
            ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
            ax.set_xticks(tick_years)
            ax.set_xticklabels([str(y) for y in tick_years], rotation=0)
        if metric in {"mae", "rmse"}:
            set_padded_ylim(ax, plot_df[metric])
        if metric == "spearman":
            ax.set_ylim(0, 1)
        if metric == "r2":
            ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.65)
            set_padded_ylim(ax, plot_df[metric], include_zero=True)
        nice_axes(ax)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Walk-Forward Metrics by Draft Year", fontsize=16, weight="bold", y=0.99)
    fig.supxlabel("Held-out draft year", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.96])
    save_figure(fig, "walkforward_metrics_requested_models.png")
    plt.close(fig)


def plot_walkforward_metrics_google_doc(walk_long: pd.DataFrame) -> None:
    """Page-sized walk-forward metric grid for Google Docs insertion."""
    plot_df = walk_long[walk_long["model_type"].isin(REQUESTED_MODELS)].copy()
    years = sorted(plot_df["test_year"].unique())
    tick_years = [year for year in years if year % 4 == 0] if years else []
    if years and max(years) not in tick_years:
        tick_years.append(max(years))
    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("spearman", "Spearman"),
        ("r2", "R²"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.8), dpi=220, sharex=True)
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None
    for ax, (metric, label) in zip(axes, metrics):
        for model in REQUESTED_MODELS:
            sub = plot_df[plot_df["model_type"] == model]
            ax.plot(
                sub["test_year"],
                sub[metric],
                marker="o",
                linewidth=2.2,
                markersize=4.2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(label, fontsize=14, weight="bold", pad=7)
        ax.set_ylabel(label, fontsize=11.5)
        if years:
            ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
            ax.set_xticks(tick_years)
            ax.set_xticklabels([str(y) for y in tick_years], rotation=0)
        if ax in axes[2:]:
            ax.set_xlabel("Held-out year", fontsize=10.5)
        if metric in {"mae", "rmse"}:
            set_padded_ylim(ax, plot_df[metric])
        if metric == "spearman":
            ax.set_ylim(0, 1)
        if metric == "r2":
            ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.65)
            set_padded_ylim(ax, plot_df[metric], include_zero=True)
        ax.grid(True, color="#E6EBEF", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#AAB7C4")
        ax.spines["bottom"].set_color("#AAB7C4")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            fontsize=10.5,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
        )
    fig.suptitle("Walk-Forward Metrics by Draft Year", fontsize=17.5, weight="bold", y=0.99)
    fig.tight_layout(rect=[0.03, 0.14, 1.0, 0.95], h_pad=1.05, w_pad=1.0)
    save_google_doc_figure(fig, "walkforward_metrics_requested_models_google_doc.png")
    plt.close(fig)


def plot_walkforward_rmse_teaser(walk_long: pd.DataFrame) -> None:
    plot_df = walk_long[walk_long["model_type"].isin(REQUESTED_MODELS)].copy()
    years = sorted(plot_df["test_year"].unique())
    tick_years = [year for year in years if year % 2 == 0] or years

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=220)
    for model in REQUESTED_MODELS:
        sub = plot_df[plot_df["model_type"] == model]
        ax.plot(
            sub["test_year"],
            sub["rmse"],
            marker="o",
            linewidth=2.4,
            markersize=4.8,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_title("Walk-Forward RMSE by Draft Year", fontsize=16, weight="bold")
    ax.set_xlabel("Held-out draft year")
    ax.set_ylabel("RMSE")
    if years:
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(y) for y in tick_years])
    set_padded_ylim(ax, plot_df["rmse"])
    ax.text(
        0.99,
        0.04,
        "See appendix for MAE, Spearman, and R²",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#4B5563",
    )
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    nice_axes(ax)
    fig.tight_layout()
    save_figure(fig, "walkforward_rmse_teaser_requested_models.png")
    plt.close(fig)


def plot_pred_vs_actual(pred_long: pd.DataFrame, latest_only: bool = False) -> None:
    plot_df = pred_long[pred_long["model_type"].isin(REQUESTED_MODELS)].copy()
    if latest_only:
        latest_year = int(plot_df["draft_season"].max())
        plot_df = plot_df[plot_df["draft_season"] == latest_year].copy()
        suffix = f"_{latest_year}"
        title_year = f"{latest_year} Draft Class"
    else:
        suffix = "_all_years"
        title_year = "All Walk-Forward Holdouts"
    scope_label = "2024 holdout" if latest_only else "all walk-forward holdouts"

    lo = 0.0
    hi = float(np.nanmax([plot_df["av_2yr"].max(), plot_df["prediction"].max(), 1.0]))
    hi = max(hi, 10.0)
    ncols = 2
    nrows = math.ceil(len(REQUESTED_MODELS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 5.2 * nrows), dpi=220, sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for ax, model in zip(axes, REQUESTED_MODELS):
        sub = plot_df[plot_df["model_type"] == model]
        metrics = eval_metrics(sub["av_2yr"], sub["prediction"])
        ax.scatter(
            sub["av_2yr"],
            sub["prediction"],
            s=22 if latest_only else 11,
            alpha=0.55 if latest_only else 0.23,
            color=MODEL_COLORS[model],
            edgecolor="none",
        )
        ax.plot([lo, hi], [lo, hi], color="#2F3E46", linewidth=1.4, alpha=0.8)
        ax.set_title(MODEL_LABELS[model], fontsize=15, weight="bold")
        ax.text(
            0.04, 0.96,
            f"MAE {metrics['mae']:.2f}\nRMSE {metrics['rmse']:.2f}\nR² {metrics['r2']:.2f}\nρ {metrics['spearman']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", alpha=0.94, edgecolor="#D9E0E6"),
        )
        nice_axes(ax)

        fig_single, ax_single = plt.subplots(figsize=(6.2, 5.4), dpi=220)
        ax_single.scatter(
            sub["av_2yr"], sub["prediction"], s=14 if not latest_only else 22,
            alpha=0.35 if not latest_only else 0.65,
            color=MODEL_COLORS[model], edgecolor="none",
        )
        ax_single.plot([lo, hi], [lo, hi], color="#2F3E46", linewidth=1.2)
        ax_single.set_title(
            f"{MODEL_LABELS[model]}: Predicted vs Actual ({scope_label})",
            fontsize=13,
            weight="bold",
        )
        ax_single.text(
            0.04, 0.96,
            f"MAE {metrics['mae']:.2f}  RMSE {metrics['rmse']:.2f}\nR² {metrics['r2']:.2f}  Spearman {metrics['spearman']:.2f}",
            transform=ax_single.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#D9E0E6"),
        )
        ax_single.set_xlabel("Actual 2-year AV")
        ax_single.set_ylabel("Predicted 2-year AV")
        ax_single.set_xlim(lo, hi)
        ax_single.set_ylim(lo, hi)
        nice_axes(ax_single)
        fig_single.tight_layout()
        save_figure(fig_single, f"pred_vs_actual_{model}{suffix}.png")
        plt.close(fig_single)

    for ax in axes[len(REQUESTED_MODELS):]:
        ax.set_visible(False)
    fig.suptitle(f"Predicted vs Actual 2-Year AV: {title_year}", fontsize=20, weight="bold", y=0.99)
    fig.supxlabel("Actual 2-year AV", fontsize=15)
    fig.supylabel("Predicted 2-year AV", fontsize=15)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.96])
    save_figure(fig, f"pred_vs_actual_grid_requested_models{suffix}.png")
    plt.close(fig)


def plot_pred_vs_actual_google_doc(pred_long: pd.DataFrame, latest_only: bool = False) -> None:
    """Page-sized pred-vs-actual grid for Google Docs insertion.

    The regular grid is intentionally large for slides/vector export. Google Docs
    often shrinks that image to page width, making labels look fuzzy. This version
    is closer to its final displayed size and uses larger text throughout.
    """
    plot_df = pred_long[pred_long["model_type"].isin(REQUESTED_MODELS)].copy()
    if latest_only:
        latest_year = int(plot_df["draft_season"].max())
        plot_df = plot_df[plot_df["draft_season"] == latest_year].copy()
        suffix = f"_{latest_year}"
        title_year = f"{latest_year} Draft Class"
    else:
        suffix = "_all_years"
        title_year = "All Walk-Forward Holdouts"

    lo = 0.0
    hi = float(np.nanmax([plot_df["av_2yr"].max(), plot_df["prediction"].max(), 1.0]))
    hi = max(hi, 10.0)

    fig, axes = plt.subplots(3, 2, figsize=(7.4, 10.2), dpi=220, sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    for ax, model in zip(axes, REQUESTED_MODELS):
        sub = plot_df[plot_df["model_type"] == model]
        metrics = eval_metrics(sub["av_2yr"], sub["prediction"])
        ax.scatter(
            sub["av_2yr"],
            sub["prediction"],
            s=13 if not latest_only else 22,
            alpha=0.24 if not latest_only else 0.6,
            color=MODEL_COLORS[model],
            edgecolor="none",
        )
        ax.plot([lo, hi], [lo, hi], color="#2F3E46", linewidth=1.35, alpha=0.85)
        ax.set_title(MODEL_LABELS[model], fontsize=13.5, weight="bold", pad=8)
        ax.text(
            0.04,
            0.95,
            f"MAE {metrics['mae']:.2f}\nRMSE {metrics['rmse']:.2f}\nR² {metrics['r2']:.2f}\nρ {metrics['spearman']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            linespacing=1.12,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.95, edgecolor="#D9E0E6"),
        )
        ax.grid(True, color="#E6EBEF", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#AAB7C4")
        ax.spines["bottom"].set_color("#AAB7C4")
    for ax in axes[len(REQUESTED_MODELS):]:
        ax.set_visible(False)

    fig.suptitle(f"Predicted vs Actual 2-Year AV: {title_year}", fontsize=17.5, weight="bold", y=0.988)
    fig.supxlabel("Actual 2-year AV", fontsize=13.5, y=0.035)
    fig.supylabel("Predicted 2-year AV", fontsize=13.5, x=0.025)
    fig.tight_layout(rect=[0.04, 0.05, 1.0, 0.955], h_pad=1.2, w_pad=1.0)
    save_google_doc_figure(fig, f"pred_vs_actual_grid_requested_models{suffix}_google_doc.png")
    plt.close(fig)


def plot_xgb_2024_pred_vs_actual_google_doc(pred_long: pd.DataFrame) -> None:
    """Single XGBoost 2024 holdout plot sized for Google Docs."""
    latest_year = int(pred_long["draft_season"].max())
    sub = pred_long[
        (pred_long["draft_season"] == latest_year)
        & (pred_long["model_type"] == "xgb")
    ].copy()
    if sub.empty:
        return

    metrics = eval_metrics(sub["av_2yr"], sub["prediction"])
    lo = 0.0
    hi = float(np.nanmax([sub["av_2yr"].max(), sub["prediction"].max(), 1.0]))
    hi = max(hi, 10.0)

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=220)
    ax.scatter(
        sub["av_2yr"],
        sub["prediction"],
        s=34,
        alpha=0.68,
        color=MODEL_COLORS["xgb"],
        edgecolor="none",
    )
    ax.plot([lo, hi], [lo, hi], color="#2F3E46", linewidth=1.6, alpha=0.85)
    ax.text(
        0.04,
        0.95,
        f"MAE {metrics['mae']:.2f}\nRMSE {metrics['rmse']:.2f}\nR² {metrics['r2']:.2f}\nρ {metrics['spearman']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.12,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, edgecolor="#D9E0E6"),
    )
    ax.set_title(f"XGBoost Predicted vs Actual 2-Year AV: {latest_year} Holdout", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Actual 2-year AV", fontsize=13)
    ax.set_ylabel("Predicted 2-year AV", fontsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, color="#E6EBEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB7C4")
    ax.spines["bottom"].set_color("#AAB7C4")
    fig.tight_layout()
    save_google_doc_figure(fig, f"pred_vs_actual_xgb_{latest_year}_google_doc.png")
    plt.close(fig)


def plot_pick_value_curves(pred_long: pd.DataFrame) -> None:
    plot_df = pred_long[pred_long["model_type"].isin(REQUESTED_MODELS)].copy()
    plot_df["pick_bin_start"] = ((plot_df["pick"].astype(int) - 1) // PICK_BIN_SIZE) * PICK_BIN_SIZE + 1
    by_bin = (
        plot_df.groupby(["model_type", "pick_bin_start"], as_index=False)
        .agg(
            pick=("pick", "mean"),
            actual_av=("av_2yr", "mean"),
            predicted_av=("prediction", "mean"),
            n=("prediction", "size"),
        )
        .sort_values(["model_type", "pick_bin_start"])
    )

    actual = by_bin.groupby("pick_bin_start", as_index=False).agg(
        pick=("pick", "mean"), actual_av=("actual_av", "mean"), n=("n", "max")
    )

    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=220)
    add_pick_region_bands(ax)
    ax.scatter(
        actual["pick"], actual["actual_av"],
        color="#6B7280", alpha=0.32, s=34,
        label="Observed 16-pick averages",
    )
    ax.plot(
        actual["pick"], monotonic_smooth(actual["pick"], actual["actual_av"]),
        color="#111827", linewidth=3.0,
        label="Actual pick-value curve",
    )
    for model in ["pick_bin", "xgb", "rf"]:
        sub = by_bin[by_bin["model_type"] == model]
        ax.plot(
            sub["pick"], monotonic_smooth(sub["pick"], sub["predicted_av"]),
            color=MODEL_COLORS[model],
            linewidth=2.3,
            alpha=0.92,
            label=MODEL_LABELS[model],
        )
    ax.text(
        0.02,
        0.07,
        "Curves are averaged by 16-pick groups; higher values mean more expected early-career AV.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="#E5E7EB"),
    )
    ax.set_title("Draft Pick Value Curve: Actual vs Model Estimates", fontsize=16, weight="bold")
    ax.set_xlabel("Overall pick number")
    ax.set_ylabel("Mean 2-year AV per 16-pick group")
    ax.set_xlim(1, 262)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, ncol=1, loc="center right")
    nice_axes(ax)
    fig.tight_layout()
    save_figure(fig, "pick_value_curves_oos_by_model.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=220, sharex=True, sharey=True)
    axes = axes.flatten()
    actual_smooth = monotonic_smooth(actual["pick"], actual["actual_av"])
    for ax, model in zip(axes, REQUESTED_MODELS):
        sub = by_bin[by_bin["model_type"] == model]
        add_pick_region_bands(ax)
        ax.plot(actual["pick"], actual_smooth, color="#111827", linewidth=2.2, label="Actual")
        ax.plot(
            sub["pick"],
            monotonic_smooth(sub["pick"], sub["predicted_av"]),
            color=MODEL_COLORS[model],
            linewidth=2.1,
            label=MODEL_LABELS[model],
        )
        ax.set_title(MODEL_LABELS[model], fontsize=12, weight="bold")
        ax.set_xlim(1, 262)
        ax.set_ylim(bottom=0)
        nice_axes(ax)
    for ax in axes[len(REQUESTED_MODELS):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Pick-Value Curves by Model", fontsize=16, weight="bold", y=0.99)
    fig.supxlabel("Overall pick number")
    fig.supylabel("Mean 2-year AV per 16-pick group")
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.96])
    save_figure(fig, "pick_value_curves_model_facets.png")
    plt.close(fig)
    by_bin.to_csv(TABLE_DIR / "pick_value_curves_oos_by_model.csv", index=False)


def plot_pick_value_curves_exact_pick(pred_long: pd.DataFrame) -> None:
    plot_models = ["pick_bin", "xgb", "rf", "ftt", "spline"]
    min_exact_pick_n = 3
    plot_df = pred_long[pred_long["model_type"].isin(plot_models)].copy()
    plot_df["pick"] = pd.to_numeric(plot_df["pick"], errors="coerce")
    plot_df = plot_df.dropna(subset=["pick", "av_2yr", "prediction"]).copy()
    plot_df["pick"] = plot_df["pick"].astype(int)

    id_cols = [c for c in ["draft_season", "pick", "pfr_player_name"] if c in plot_df.columns]
    actual_source = plot_df.drop_duplicates(id_cols) if id_cols else plot_df
    actual = (
        actual_source.groupby("pick", as_index=False)
        .agg(actual_av=("av_2yr", "mean"), n=("av_2yr", "size"))
        .sort_values("pick")
    )
    pred_by_pick = (
        plot_df.groupby(["model_type", "pick"], as_index=False)
        .agg(predicted_av=("prediction", "mean"), n=("prediction", "size"))
        .sort_values(["model_type", "pick"])
    )
    actual_curve = actual[actual["n"] >= min_exact_pick_n].copy()
    y_cap = max(20.0, float(np.nanpercentile(actual_curve["actual_av"], 99) + 0.5))
    actual_display = actual.copy()
    actual_display["display_av"] = actual_display["actual_av"].clip(upper=y_cap)
    actual_display = actual_display[actual_display["actual_av"] <= y_cap].copy()

    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=220)
    add_pick_region_bands(ax)
    ax.scatter(
        actual_display["pick"],
        actual_display["display_av"],
        color="#6B7280",
        alpha=0.18,
        s=np.clip(actual_display["n"] * 3.5, 8, 30),
        label="Observed exact-pick averages",
    )
    ax.plot(
        actual_curve["pick"],
        monotonic_smooth(actual_curve["pick"], actual_curve["actual_av"]),
        color="#111827",
        linewidth=3.0,
        label="Actual pick-value curve",
    )
    for model in plot_models:
        sub = pred_by_pick[pred_by_pick["model_type"] == model]
        sub_curve = sub[sub["n"] >= min_exact_pick_n]
        ax.plot(
            sub_curve["pick"],
            monotonic_smooth(sub_curve["pick"], sub_curve["predicted_av"]),
            color=MODEL_COLORS[model],
            linewidth=2.2,
            alpha=0.92,
            label=MODEL_LABELS[model],
        )
    ax.set_title("Draft Pick Value Curve by Exact Pick", fontsize=16, weight="bold")
    ax.set_xlabel("Overall pick number")
    ax.set_ylabel("Mean 2-year AV at exact pick")
    ax.set_xlim(1, 262)
    ax.set_ylim(0, y_cap + 0.5)
    ax.legend(frameon=False, fontsize=14, ncol=1, loc="center right")
    nice_axes(ax)
    fig.tight_layout()
    save_figure(fig, "pick_value_curves_exact_pick_no_bins.png")
    plt.close(fig)


def plot_monotonic_pick_curve() -> None:
    if not (DATA_DIR / "drafts.csv").exists() or not (DATA_DIR / "av.csv").exists():
        return

    drafts = pd.read_csv(DATA_DIR / "drafts.csv")[["draft_season", "pick", "pfr_player_id"]]
    av = pd.read_csv(DATA_DIR / "av.csv")[["season", "pfr_player_id", "av"]]
    drafts["draft_season"] = pd.to_numeric(drafts["draft_season"], errors="coerce")
    drafts["pick"] = pd.to_numeric(drafts["pick"], errors="coerce")
    av["season"] = pd.to_numeric(av["season"], errors="coerce")
    av["av"] = pd.to_numeric(av["av"], errors="coerce").fillna(0.0)
    drafts = drafts.dropna(subset=["draft_season", "pick"]).copy()
    drafts["draft_season"] = drafts["draft_season"].astype(int)
    drafts["pick"] = drafts["pick"].astype(int)
    av = av.dropna(subset=["season", "pfr_player_id"]).copy()
    av["season"] = av["season"].astype(int)
    av = av.groupby(["season", "pfr_player_id"], as_index=False)["av"].sum()

    complete = drafts[drafts["draft_season"] + 1 <= int(av["season"].max())].copy()
    y0 = av.rename(columns={"season": "draft_season", "av": "av_y0"})
    y1 = av.rename(columns={"season": "year_two", "av": "av_y1"})
    complete = complete.merge(y0[["draft_season", "pfr_player_id", "av_y0"]], on=["draft_season", "pfr_player_id"], how="left")
    complete["year_two"] = complete["draft_season"] + 1
    complete = complete.merge(y1[["year_two", "pfr_player_id", "av_y1"]], on=["year_two", "pfr_player_id"], how="left")
    complete["av_2yr"] = complete[["av_y0", "av_y1"]].fillna(0.0).sum(axis=1)
    complete["pick_bin_start"] = ((complete["pick"] - 1) // PICK_BIN_SIZE) * PICK_BIN_SIZE + 1
    actual = complete.groupby("pick_bin_start", as_index=False).agg(
        pick=("pick", "mean"), actual_av=("av_2yr", "mean"), n=("av_2yr", "size")
    )

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    smooth = iso.fit_transform(actual["pick"], actual["actual_av"])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=220)
    add_pick_region_bands(ax)
    ax.scatter(actual["pick"], actual["actual_av"], s=np.clip(actual["n"] / 1.8, 18, 80),
               color="#AAB7C4", alpha=0.68, edgecolor="white", linewidth=0.6,
               label="Observed 16-pick averages")
    ax.plot(actual["pick"], smooth, color="#111827", linewidth=2.8, label="Monotonic empirical curve")
    ax.text(
        0.02,
        0.08,
        "Interpretation: earlier picks have higher expected early-career AV, but the curve flattens later in the draft.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="#E5E7EB"),
    )
    ax.set_title("Empirical Draft Pick Value Curve", fontsize=16, weight="bold")
    ax.set_xlabel("Overall pick number")
    ax.set_ylabel("Expected 2-year AV")
    ax.set_xlim(1, 262)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    nice_axes(ax)
    fig.tight_layout()
    save_figure(fig, "pick_value_curve_monotonic_report.png")
    plt.close(fig)


def plot_error_diagnostics(pred_long: pd.DataFrame) -> None:
    plot_df = pred_long[pred_long["model_type"].isin(REQUESTED_MODELS)].copy()
    plot_df["pick_bin_start"] = ((plot_df["pick"].astype(int) - 1) // PICK_BIN_SIZE) * PICK_BIN_SIZE + 1

    mae_bin = plot_df.groupby(["model_type", "pick_bin_start"], as_index=False).agg(
        pick=("pick", "mean"), mae=("absolute_error", "mean")
    )
    fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
    for model in REQUESTED_MODELS:
        sub = mae_bin[mae_bin["model_type"] == model]
        ax.plot(sub["pick"], sub["mae"], color=MODEL_COLORS[model], linewidth=2.0, marker="o", markersize=3.5,
                label=MODEL_LABELS[model])
    ax.set_title("Mean Absolute Error by Pick Range", fontsize=16, weight="bold")
    ax.set_xlabel("Overall pick number")
    ax.set_ylabel("MAE by 16-pick bin")
    ax.set_xlim(1, 262)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    nice_axes(ax)
    fig.tight_layout()
    save_figure(fig, "mae_by_pick_bin_requested_models.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=220, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, model in zip(axes, REQUESTED_MODELS):
        sub = plot_df[plot_df["model_type"] == model]
        ax.hist(sub["residual"], bins=32, color=MODEL_COLORS[model], alpha=0.82)
        ax.axvline(0, color="#111827", linewidth=1.2)
        ax.set_title(MODEL_LABELS[model], fontsize=12, weight="bold")
        nice_axes(ax)
    for ax in axes[len(REQUESTED_MODELS):]:
        ax.set_visible(False)
    fig.suptitle("Residual Distributions", fontsize=16, weight="bold", y=0.99)
    fig.supxlabel("Actual - predicted 2-year AV")
    fig.supylabel("Players")
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.96])
    save_figure(fig, "residual_distributions_requested_models.png")
    plt.close(fig)


def plot_calibration(pred_long: pd.DataFrame) -> None:
    plot_df = pred_long[pred_long["model_type"].isin(REQUESTED_MODELS)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=220, sharex=True, sharey=True)
    axes = axes.flatten()
    hi = 0.0
    for ax, model in zip(axes, REQUESTED_MODELS):
        sub = plot_df[plot_df["model_type"] == model].copy()
        sub["pred_decile"] = pd.qcut(sub["prediction"], q=10, labels=False, duplicates="drop")
        cal = sub.groupby("pred_decile", as_index=False).agg(
            predicted=("prediction", "mean"), actual=("av_2yr", "mean")
        )
        hi = max(hi, float(cal[["predicted", "actual"]].max().max()))
        ax.plot(cal["predicted"], cal["actual"], marker="o", linewidth=2.0, color=MODEL_COLORS[model])
        ax.set_title(MODEL_LABELS[model], fontsize=12, weight="bold")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        nice_axes(ax)
    hi = max(hi, 1.0)
    for ax in axes[:len(REQUESTED_MODELS)]:
        ax.plot([0, hi], [0, hi], color="#111827", linewidth=1.0, linestyle="--")
    for ax in axes[len(REQUESTED_MODELS):]:
        ax.set_visible(False)
    fig.suptitle("Calibration by Prediction Decile", fontsize=16, weight="bold", y=0.99)
    fig.supxlabel("Mean predicted 2-year AV")
    fig.supylabel("Mean actual 2-year AV")
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.96])
    save_figure(fig, "calibration_by_prediction_decile_requested_models.png")
    plt.close(fig)


def plot_prediction_distributions(pred_long: pd.DataFrame) -> None:
    latest_year = int(pred_long["draft_season"].max())
    plot_df = pred_long[
        (pred_long["draft_season"] == latest_year)
        & (pred_long["model_type"].isin(REQUESTED_MODELS))
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=220)
    ax.hist(plot_df["av_2yr"].dropna(), bins=np.arange(0, max(40, plot_df["av_2yr"].max() + 2), 2),
            color="#111827", alpha=0.22, label="Actual")
    for model in REQUESTED_MODELS:
        sub = plot_df[plot_df["model_type"] == model]
        ax.hist(sub["prediction"], bins=np.arange(0, max(40, plot_df["prediction"].max() + 2), 2),
                histtype="step", linewidth=2.0, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_title(f"{latest_year} Actual vs Predicted AV Distribution", fontsize=16, weight="bold")
    ax.set_xlabel("2-year AV")
    ax.set_ylabel("Players")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    nice_axes(ax)
    fig.tight_layout()
    save_figure(fig, f"prediction_distribution_{latest_year}.png")
    plt.close(fig)


def plot_feature_importance() -> None:
    top5_by_model = {}
    for model in ["xgb", "rf"]:
        csv_path = RESULTS_DIR / f"{model}_feature_importance_v6.csv"
        if not csv_path.exists():
            continue
        full_df = pd.read_csv(csv_path).dropna(subset=["importance"])
        df = full_df.head(20)
        if df.empty:
            continue
        top5_by_model[model] = full_df.head(5)
        df = df.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=220)
        ax.barh(df["feature"], df["importance"], color=MODEL_COLORS[model], alpha=0.9)
        ax.set_title(f"{MODEL_LABELS[model]} Feature Importance", fontsize=15, weight="bold")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)
        nice_axes(ax)
        fig.tight_layout()
        save_figure(fig, f"feature_importance_{model}_top20.png")
        plt.close(fig)

    if top5_by_model:
        fig_width = max(9.5, 5.2 * len(top5_by_model))
        fig, axes = plt.subplots(1, len(top5_by_model), figsize=(fig_width, 4.8), dpi=220)
        axes = np.array(axes).flatten()
        for ax, (model, df) in zip(axes, top5_by_model.items()):
            df = df.copy()
            top_importance = df["importance"].max()
            if top_importance > 0:
                df["relative_importance"] = df["importance"] / top_importance * 100
            else:
                df["relative_importance"] = 0
            df = df.iloc[::-1]
            ax.barh(df["feature"], df["relative_importance"], color=MODEL_COLORS[model], alpha=0.9)
            ax.set_title(MODEL_LABELS[model], fontsize=12, weight="bold")
            ax.set_xlabel("Relative importance")
            ax.set_xlim(0, 105)
            ax.tick_params(axis="y", labelsize=8)
            nice_axes(ax)
        fig.suptitle("Top 5 Tree Model Features", fontsize=16, weight="bold", y=1.02)
        fig.tight_layout()
        save_figure(fig, "tree_feature_importance_top5.png")
        plt.close(fig)


def caption_for_figure(filename: str) -> str:
    if filename in FIGURE_CAPTIONS:
        return FIGURE_CAPTIONS[filename]

    if filename.startswith("pred_vs_actual_"):
        model_key = filename.replace("pred_vs_actual_", "").replace("_2024.png", "").replace("_all_years.png", "")
        model_label = MODEL_LABELS.get(model_key, model_key.replace("_", " ").title())
        scope = "the 2024 held-out draft class" if filename.endswith("_2024.png") else "all walk-forward held-out draft classes"
        return (
            f"Predicted versus actual 2-year AV for {model_label} on {scope}. "
            "The diagonal line represents perfect prediction; points closer to the line are more accurate."
        )

    return (
        "Report visualization generated from the final DraftSight walk-forward outputs. "
        "Axes and titles describe the plotted metric and population."
    )


def write_figure_captions() -> None:
    rows = []
    for path in sorted(FIGURE_DIR.glob("*.png")):
        rows.append({
            "figure_file": path.name,
            "caption": caption_for_figure(path.name),
        })
    captions = pd.DataFrame(rows)
    captions.to_csv(TABLE_DIR / "figure_captions.csv", index=False)

    md_lines = [
        "# Figure Captions",
        "",
        "Use or adapt these captions in the written report so every figure has a text explanation.",
        "",
    ]
    for row in rows:
        md_lines.append(f"## `{row['figure_file']}`")
        md_lines.append("")
        md_lines.append(row["caption"])
        md_lines.append("")
    (REPORT_DIR / "figure_captions.md").write_text("\n".join(md_lines))


def main() -> None:
    ensure_dirs()
    walk, overall, selected, preds = load_results()
    pred_long = write_tables(walk, overall, selected, preds)
    walk_long = walkforward_long(walk)
    write_copy_ready_metric_tables(walk_long, overall)

    plot_overall_performance(overall)
    plot_walkforward_metrics(walk_long)
    plot_walkforward_metrics_google_doc(walk_long)
    plot_walkforward_rmse_teaser(walk_long)
    plot_pred_vs_actual(pred_long, latest_only=False)
    plot_pred_vs_actual(pred_long, latest_only=True)
    plot_pred_vs_actual_google_doc(pred_long, latest_only=False)
    plot_pred_vs_actual_google_doc(pred_long, latest_only=True)
    plot_xgb_2024_pred_vs_actual_google_doc(pred_long)
    plot_pick_value_curves(pred_long)
    plot_pick_value_curves_exact_pick(pred_long)
    plot_monotonic_pick_curve()
    plot_error_diagnostics(pred_long)
    plot_calibration(pred_long)
    plot_prediction_distributions(pred_long)
    plot_feature_importance()
    write_figure_captions()

    print(f"Wrote tables to {TABLE_DIR}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
