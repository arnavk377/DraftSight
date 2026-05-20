"""Build a single draft-pick context feature table.

This merges leakage-safe roster context with structural traded-pick features.
It intentionally excludes selected-player information from trade data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROSTER_CONTEXT = (
    REPO_ROOT / "data" / "processed" / "rosters" / "draft_pick_roster_context_features.csv"
)
DEFAULT_TRADE_CONTEXT = REPO_ROOT / "data" / "processed" / "trades" / "draft_pick_trade_features.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "features" / "draft_pick_context_features.csv"

TRADE_COUNT_COLS = [
    "pick_trade_count",
    "pick_future_trade_count",
    "pick_same_year_trade_count",
    "pick_post_year_trade_count",
    "pick_trade_n_distinct_teams",
]
TRADE_BOOL_COLS = ["pick_was_traded", "pick_ever_conditional"]
TRADE_TEXT_COLS = [
    "pick_first_trade_date",
    "pick_last_trade_date",
    "pick_first_gave_team",
    "pick_first_received_team",
    "pick_last_gave_team",
    "pick_last_received_team",
    "pick_trade_team_chain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-context", type=Path, default=DEFAULT_ROSTER_CONTEXT)
    parser.add_argument("--trade-context", type=Path, default=DEFAULT_TRADE_CONTEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_context_features(roster_context: pd.DataFrame, trade_context: pd.DataFrame) -> pd.DataFrame:
    for frame in [roster_context, trade_context]:
        frame["draft_season"] = pd.to_numeric(frame["draft_season"], errors="coerce").astype("Int64")
        frame["pick"] = pd.to_numeric(frame["pick"], errors="coerce").astype("Int64")

    keep_trade_cols = [
        "draft_season",
        "pick",
        *TRADE_BOOL_COLS,
        *TRADE_COUNT_COLS,
        *TRADE_TEXT_COLS,
        "pick_approx_value_points",
    ]
    trade_context = trade_context[[col for col in keep_trade_cols if col in trade_context.columns]].copy()

    merged = roster_context.merge(
        trade_context,
        on=["draft_season", "pick"],
        how="left",
    )

    for col in TRADE_BOOL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].map(lambda value: bool(value) if pd.notna(value) else False)
    for col in TRADE_COUNT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    for col in TRADE_TEXT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    return merged.sort_values(["draft_season", "pick"])


def main() -> None:
    args = parse_args()
    roster_context = pd.read_csv(args.roster_context)
    trade_context = pd.read_csv(args.trade_context)
    features = build_context_features(roster_context, trade_context)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    print(f"Wrote {len(features):,} draft-pick context feature rows: {args.output}")
    print(f"Trade-feature coverage: {features['pick_was_traded'].mean():.1%} of picks marked traded")
    print(
        "Roster-context coverage since 2010: "
        f"{features.loc[features['draft_season'] >= 2010, 'roster_context_available'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
