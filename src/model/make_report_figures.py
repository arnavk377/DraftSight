"""Create report-ready figures from saved DraftSight artifacts.

This script is intentionally lightweight: it does not retrain the full model.
It uses the committed demo data plus ``pick_values.json`` to produce a clean
pick-value curve for reports/posters.

Run:
    python -m src.model.make_report_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "supabase_exports"
RESULTS_DIR = ROOT / "results"
PICK_VALUES_PATH = ROOT / "pick_values.json"


def load_two_year_av() -> pd.DataFrame:
    """Build one row per draft pick with first-two-season AV."""
    drafts = pd.read_csv(DATA_DIR / "drafts.csv")
    av = pd.read_csv(DATA_DIR / "av.csv")

    drafts = drafts[["draft_season", "pick", "pfr_player_id"]].copy()
    drafts["draft_season"] = pd.to_numeric(drafts["draft_season"], errors="coerce")
    drafts["pick"] = pd.to_numeric(drafts["pick"], errors="coerce")
    drafts = drafts.dropna(subset=["draft_season", "pick"]).copy()
    drafts["draft_season"] = drafts["draft_season"].astype(int)
    drafts["pick"] = drafts["pick"].astype(int)

    av = av[["season", "pfr_player_id", "av"]].copy()
    av["season"] = pd.to_numeric(av["season"], errors="coerce")
    av["av"] = pd.to_numeric(av["av"], errors="coerce").fillna(0.0)
    av = av.dropna(subset=["season", "pfr_player_id"]).copy()
    av["season"] = av["season"].astype(int)
    av = av.groupby(["season", "pfr_player_id"], as_index=False)["av"].sum()

    av_min = int(av["season"].min())
    av_max = int(av["season"].max())
    complete = drafts[
        (drafts["draft_season"] >= av_min)
        & (drafts["draft_season"] + 1 <= av_max)
    ].copy()

    rookie = av.rename(columns={"season": "draft_season", "av": "av_y0"})
    year_two = av.rename(columns={"season": "year_two", "av": "av_y1"})
    complete = complete.merge(
        rookie[["draft_season", "pfr_player_id", "av_y0"]],
        on=["draft_season", "pfr_player_id"],
        how="left",
    )
    complete["year_two"] = complete["draft_season"] + 1
    complete = complete.merge(
        year_two[["year_two", "pfr_player_id", "av_y1"]],
        on=["year_two", "pfr_player_id"],
        how="left",
    )
    complete["av_2yr"] = complete[["av_y0", "av_y1"]].fillna(0.0).sum(axis=1)
    return complete[["draft_season", "pick", "av_2yr"]]


def load_pick_values() -> pd.DataFrame:
    with PICK_VALUES_PATH.open() as f:
        raw = json.load(f)
    return pd.DataFrame(raw).rename(columns={"value": "model_value"})


def make_pick_value_curve() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)

    actual = load_two_year_av()
    model_curve = load_pick_values()

    bin_size = 16
    actual["pick_bin"] = ((actual["pick"] - 1) // bin_size) * bin_size + 1
    binned = (
        actual.groupby("pick_bin", as_index=False)
        .agg(
            pick=("pick", "mean"),
            actual_av=("av_2yr", "mean"),
            n=("av_2yr", "size"),
        )
        .sort_values("pick")
    )

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=220)
    ax.scatter(
        binned["pick"],
        binned["actual_av"],
        s=np.clip(binned["n"] / 1.8, 18, 80),
        color="#8DA0AE",
        alpha=0.62,
        edgecolor="white",
        linewidth=0.6,
        label=f"Actual 2-year AV, {bin_size}-pick bins",
        zorder=2,
    )
    ax.plot(
        model_curve["pick"],
        model_curve["model_value"],
        color="#1F6F8B",
        linewidth=3.0,
        label="XGBoost value curve, monotonic smoothed",
        zorder=3,
    )

    for boundary in [32, 64, 100, 135, 176, 215]:
        ax.axvline(boundary + 0.5, color="#DFE5EA", linewidth=0.8, zorder=1)

    ax.set_title("NFL Draft Pick Value Curve", fontsize=17, weight="bold", pad=12)
    subtitle = (
        f"Actual AV from {actual['draft_season'].min()}-{actual['draft_season'].max()} "
        "with model curve constrained so later picks do not exceed earlier picks"
    )
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=10.5, color="#52616B")
    ax.set_xlabel("Overall pick number", fontsize=12)
    ax.set_ylabel("Expected Approximate Value, first 2 NFL seasons", fontsize=12)
    ax.set_xlim(1, 262)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E9EEF2", linewidth=0.9)
    ax.grid(axis="x", color="#F2F5F7", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB7C4")
    ax.spines["bottom"].set_color("#AAB7C4")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()

    out_path = RESULTS_DIR / "plot_pick_value_curve_report.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    binned.to_csv(RESULTS_DIR / "pick_value_curve_actual_bins.csv", index=False)
    return out_path


def main() -> None:
    out_path = make_pick_value_curve()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
