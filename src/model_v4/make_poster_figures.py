"""Create poster-ready figures from model_v4 evaluation outputs.

The script intentionally consumes saved CSV artifacts instead of retraining.
That keeps poster figure generation fast, reproducible, and safe to rerun.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "poc_outputs_v4"
OUTPUT_DIR = ROOT / "poc_outputs"

OVERALL_SUMMARY = SOURCE_DIR / "model_v4_overall_summary.csv"
WALKFORWARD_RESULTS = SOURCE_DIR / "model_v4_walkforward_results.csv"
LATEST_PREDICTIONS = SOURCE_DIR / "model_v4_latest_year_predictions_2024.csv"

MODEL_ORDER = ["pick_cfb_bin", "spline", "xgb", "tabnet"]
MODEL_LABELS = {
    "pick_cfb_bin": "Pick + College Bin",
    "spline": "Spline",
    "xgb": "XGBoost",
    "tabnet": "TabNet",
}
PREDICTION_COLUMNS = {
    "pick_cfb_bin": "pred_pick_cfb_bin",
    "spline": "pred_spline",
    "xgb": "pred_xgb",
    "tabnet": "pred_tabnet",
}
MODEL_COLORS = {
    "pick_cfb_bin": "#2a9d8f",
    "spline": "#f4a261",
    "xgb": "#457b9d",
    "tabnet": "#e76f51",
}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def _normalize_higher_is_better(values: pd.Series) -> pd.Series:
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(0.5, index=values.index)
    return (values - min_value) / (max_value - min_value)


def load_all_labeled_players() -> pd.DataFrame:
    """Load the full labeled modeling frame without training any model."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.model_v4.train_n_evaluate import build_model_frame

    return build_model_frame()


def make_figure_1(overall: pd.DataFrame) -> None:
    """Poster Figure 1: compact model performance comparison."""

    summary = (
        overall.assign(
            model_label=lambda frame: frame["model_type"].map(MODEL_LABELS),
            mse=lambda frame: frame["rmse"] ** 2,
        )
        .set_index("model_type")
        .loc[MODEL_ORDER]
        .reset_index()
    )

    metric_frame = summary[["mse", "r2"]].copy()
    performance_score = pd.DataFrame(index=summary.index)
    performance_score["mse"] = 1 - _normalize_higher_is_better(metric_frame["mse"])
    performance_score["r2"] = _normalize_higher_is_better(metric_frame["r2"])

    cmap = LinearSegmentedColormap.from_list(
        "poster_worse_to_better",
        ["#cf4c45", "#f7f5ef", "#2f6fb0"],
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    im = ax.imshow(performance_score.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks([0, 1], labels=["MSE\n(lower is better)", "R^2\n(higher is better)"])
    ax.set_yticks(np.arange(len(summary)), labels=summary["model_label"])
    ax.tick_params(axis="both", length=0, labelsize=12)

    for row_index, row in summary.iterrows():
        values = [row["mse"], row["r2"]]
        labels = [f"{values[0]:.1f}", f"{values[1]:.3f}"]
        for column_index, label in enumerate(labels):
            score = performance_score.iloc[row_index, column_index]
            text_color = "white" if score < 0.18 or score > 0.82 else "#1b1b1b"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=16,
                fontweight="bold",
            )

    ax.set_title("Model Performance Summary", fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Walk-forward aggregate metrics from model_v4 outputs", fontsize=11, labelpad=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["Worse", "Middle", "Better"])
    colorbar.ax.tick_params(labelsize=10)

    csv_out = OUTPUT_DIR / "poster_figure_1_model_performance_mse_r2.csv"
    count_column = "n_predictions" if "n_predictions" in summary.columns else "n"
    summary[
        ["model_type", "model_label", "mse", "r2", "mae", "rmse", "spearman", count_column]
    ].to_csv(
        csv_out,
        index=False,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "poster_figure_1_model_performance_mse_r2.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "poster_figure_1_model_performance_mse_r2.pdf", bbox_inches="tight")
    plt.close(fig)


def make_signal_error_figure(overall: pd.DataFrame) -> None:
    """Poster-friendly replacement for the MSE/R2 heatmap."""

    summary = (
        overall.assign(model_label=lambda frame: frame["model_type"].map(MODEL_LABELS))
        .set_index("model_type")
        .loc[MODEL_ORDER]
        .reset_index()
    )
    colors = [MODEL_COLORS[model_type] for model_type in summary["model_type"]]
    labels = summary["model_label"].tolist()
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=300)

    axes[0].bar(x, summary["spearman"], color=colors, alpha=0.95)
    axes[0].axhline(0, color="#2f3e46", linewidth=1.0)
    axes[0].set_title("Ranking Signal", fontsize=15, fontweight="bold", pad=12)
    axes[0].set_ylabel("Spearman correlation", fontsize=11)
    axes[0].set_ylim(0, max(0.65, summary["spearman"].max() + 0.08))

    axes[1].bar(x, summary["mae"], color=colors, alpha=0.95)
    axes[1].set_title("Typical Prediction Error", fontsize=15, fontweight="bold", pad=12)
    axes[1].set_ylabel("Mean absolute error in 2-year AV", fontsize=11)
    axes[1].set_ylim(0, max(4.4, summary["mae"].max() + 0.45))

    for ax, metric_name in zip(axes, ["spearman", "mae"]):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", color="#e6e8eb", linewidth=0.9)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for idx, value in enumerate(summary[metric_name]):
            ax.text(idx, value + ax.get_ylim()[1] * 0.018, f"{value:.2f}", ha="center", fontsize=10)

    fig.suptitle("Out-of-Sample Model Comparison", fontsize=18, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        0.01,
        "Walk-forward predictions across 2010-2024; Spearman shows ordering ability, MAE shows average miss size.",
        ha="center",
        fontsize=10,
        color="#3b4652",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    summary[["model_type", "model_label", "spearman", "mae", "rmse", "r2", "n_predictions"]].to_csv(
        OUTPUT_DIR / "poster_figure_1_model_signal_error.csv",
        index=False,
    )
    fig.savefig(OUTPUT_DIR / "poster_figure_1_model_signal_error.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "poster_figure_1_model_signal_error.pdf", bbox_inches="tight")
    plt.close(fig)


def make_walkforward_signal_figure(walkforward: pd.DataFrame) -> None:
    """Show year-by-year rank signal, which is the cleanest model story."""

    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=300)
    for model_type in MODEL_ORDER:
        ax.plot(
            walkforward["test_year"],
            walkforward[f"{model_type}_spearman"],
            marker="o",
            linewidth=2.4,
            markersize=5,
            color=MODEL_COLORS[model_type],
            label=MODEL_LABELS[model_type],
            alpha=0.95,
        )

    ax.axhline(0, color="#2f3e46", linewidth=1.0)
    ax.set_title("Walk-Forward Ranking Signal by Draft Year", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Held-out draft year", fontsize=12)
    ax.set_ylabel("Spearman correlation", fontsize=12)
    ax.set_ylim(0, max(0.75, walkforward[[f"{model}_spearman" for model in MODEL_ORDER]].max().max() + 0.06))
    ax.grid(True, color="#e6e8eb", linewidth=0.9)
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", edgecolor="#d7dde4", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    keep_cols = ["test_year"] + [f"{model}_spearman" for model in MODEL_ORDER]
    walkforward[keep_cols].to_csv(OUTPUT_DIR / "poster_figure_1_walkforward_rank_signal.csv", index=False)
    fig.savefig(OUTPUT_DIR / "poster_figure_1_walkforward_rank_signal.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "poster_figure_1_walkforward_rank_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figure_2(predictions: pd.DataFrame) -> None:
    """Poster Figure 2: actual AV distribution vs each model's predicted distribution."""

    actual = predictions["av_2yr"].dropna()
    prediction_max = max(predictions[column].max() for column in PREDICTION_COLUMNS.values())
    max_value = max(actual.max(), prediction_max)
    bins = np.arange(0, np.ceil(max_value / 2) * 2 + 2.1, 2)

    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=300)

    ax.hist(
        actual,
        bins=bins,
        color="#cfd6de",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.88,
        label="Actual AV",
    )
    ax.axvline(
        actual.mean(),
        color="#28343f",
        linewidth=2.0,
        linestyle="--",
        alpha=0.85,
        label=f"Actual mean: {actual.mean():.1f}",
    )

    for model_type in MODEL_ORDER:
        pred_col = PREDICTION_COLUMNS[model_type]
        predicted = predictions[pred_col].dropna()
        model_label = MODEL_LABELS[model_type]
        model_color = MODEL_COLORS[model_type]

        ax.hist(
            predicted,
            bins=bins,
            histtype="step",
            linewidth=2.8,
            color=model_color,
            label=f"{model_label} predictions",
        )

    ax.set_title(
        "2024 Draft Class: Actual vs Predicted 2-Year AV Distributions",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("2-Year Approximate Value", fontsize=12)
    ax.set_ylabel("Player Count", fontsize=12)
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.9)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d7dde4", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    distribution_summary = []
    for model_type, pred_col in PREDICTION_COLUMNS.items():
        predicted = predictions[pred_col].dropna()
        distribution_summary.append(
            {
                "model_type": model_type,
                "model_label": MODEL_LABELS[model_type],
                "actual_mean": actual.mean(),
                "predicted_mean": predicted.mean(),
                "actual_std": actual.std(),
                "predicted_std": predicted.std(),
                "actual_max": actual.max(),
                "predicted_max": predicted.max(),
            }
        )
    pd.DataFrame(distribution_summary).to_csv(
        OUTPUT_DIR / "poster_figure_2_av_distribution_predictions_2024.csv",
        index=False,
    )
    fig.savefig(OUTPUT_DIR / "poster_figure_2_av_distribution_predictions_2024.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "poster_figure_2_av_distribution_predictions_2024.pdf", bbox_inches="tight")
    plt.close(fig)


def make_all_player_av_distribution(model_frame: pd.DataFrame) -> None:
    """Response-variable reminder: all labeled 2-year AV outcomes."""

    av = model_frame["av_2yr"].dropna().astype(float)
    upper = max(40, int(np.ceil(av.max() / 2) * 2))
    bins = np.arange(0, upper + 1, 1)

    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=300)
    ax.hist(av, bins=bins, color="#6fa3d2", edgecolor="white", linewidth=0.45, alpha=0.95)
    ax.axvline(av.median(), color="#253746", linewidth=2.0, linestyle="--", label=f"Median: {av.median():.1f}")
    ax.axvline(av.mean(), color="#c44e52", linewidth=2.0, linestyle=":", label=f"Mean: {av.mean():.1f}")

    ax.set_title("2-Year Approximate Value Distribution per Player", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Approximate Value in First 2 NFL Seasons", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)
    ax.set_xlim(0, upper)
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.9)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.text(
        0.99,
        0.93,
        f"n = {len(av):,} labeled draft picks",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#24313f",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d7dde4"},
    )
    fig.tight_layout()

    pd.DataFrame(
        [
            {
                "n_players": len(av),
                "mean_av_2yr": av.mean(),
                "median_av_2yr": av.median(),
                "p75_av_2yr": av.quantile(0.75),
                "p90_av_2yr": av.quantile(0.90),
                "max_av_2yr": av.max(),
            }
        ]
    ).to_csv(OUTPUT_DIR / "poster_figure_2_av_distribution_all_labeled_players.csv", index=False)
    fig.savefig(OUTPUT_DIR / "poster_figure_2_av_distribution_all_labeled_players.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "poster_figure_2_av_distribution_all_labeled_players.pdf", bbox_inches="tight")
    plt.close(fig)


def make_loess_value_curve(model_frame: pd.DataFrame) -> None:
    """Readable empirical draft value curve from all labeled players."""

    curve_df = model_frame[["pick", "av_2yr"]].dropna().copy()
    curve_df["pick"] = curve_df["pick"].astype(int)
    curve_df["pick_bin"] = ((curve_df["pick"] - 1) // 16).astype(int)
    binned = (
        curve_df.groupby("pick_bin", as_index=False)
        .agg(
            pick_center=("pick", "median"),
            mean_av=("av_2yr", "mean"),
            median_av=("av_2yr", "median"),
            n_players=("av_2yr", "size"),
        )
        .sort_values("pick_center")
    )
    smoothed = lowess(
        binned["mean_av"],
        binned["pick_center"],
        frac=0.28,
        it=1,
        return_sorted=True,
    )
    monotone_smooth = smoothed.copy()
    monotone_smooth[:, 1] = np.minimum.accumulate(monotone_smooth[:, 1])

    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=300)
    size = np.clip(binned["n_players"] * 0.55, 18, 70)
    ax.scatter(
        binned["pick_center"],
        binned["mean_av"],
        s=size,
        color="#9fb6c9",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        label="16-pick bin average",
    )
    ax.plot(
        monotone_smooth[:, 0],
        monotone_smooth[:, 1],
        color="#1f5f8b",
        linewidth=3.5,
        label="Monotone LOESS value curve",
    )

    ax.set_title("Empirical Draft Value Curve", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Pick Number", fontsize=12)
    ax.set_ylabel("2-Year Approximate Value", fontsize=12)
    ax.set_xlim(1, min(260, curve_df["pick"].max() + 2))
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e6e8eb", linewidth=0.9)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d7dde4", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    smoothed_df = pd.DataFrame(
        {
            "pick": smoothed[:, 0],
            "loess_mean_av_2yr": smoothed[:, 1],
            "monotone_loess_mean_av_2yr": monotone_smooth[:, 1],
        }
    )
    binned.merge(smoothed_df, how="left", left_on="pick_center", right_on="pick").drop(columns=["pick"]).to_csv(
        OUTPUT_DIR / "poster_figure_2_loess_draft_value_curve_all_labeled_players.csv",
        index=False,
    )
    fig.savefig(OUTPUT_DIR / "poster_figure_2_loess_draft_value_curve_all_labeled_players.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "poster_figure_2_loess_draft_value_curve_all_labeled_players.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _require_file(OVERALL_SUMMARY)
    _require_file(WALKFORWARD_RESULTS)
    _require_file(LATEST_PREDICTIONS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(OVERALL_SUMMARY)
    walkforward = pd.read_csv(WALKFORWARD_RESULTS)
    predictions = pd.read_csv(LATEST_PREDICTIONS)
    model_frame = load_all_labeled_players()

    make_figure_1(overall)
    make_signal_error_figure(overall)
    make_walkforward_signal_figure(walkforward)
    make_figure_2(predictions)
    make_all_player_av_distribution(model_frame)
    make_loess_value_curve(model_frame)

    print(f"Wrote poster figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
