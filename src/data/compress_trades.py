"""Compress asset-level NFL trade rows into analysis-friendly tables.

The raw trade file has one row per traded asset:
  - `gave` is the team giving the asset.
  - `received` is the team receiving the asset.
  - Draft-pick assets can include the eventual selected player in `pfr_*`.

This script creates:
  - one row per trade, with JSON-preserved asset/team summaries;
  - one row per team per trade, useful for team behavior analysis;
  - one row per traded pick asset, useful for joining into draft models.

Important modeling note: selected-player fields on draft-pick assets are
post-trade outcomes. Do not use those selected-player names/ids as model
features for draft-night prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "trades" / "trades.csv"
DEFAULT_TRADE_OUTPUT = REPO_ROOT / "data" / "trades" / "trades_compressed.csv"
DEFAULT_TEAM_LEDGER_OUTPUT = REPO_ROOT / "data" / "trades" / "team_trade_ledger.csv"
DEFAULT_PICK_LEDGER_OUTPUT = REPO_ROOT / "data" / "trades" / "traded_pick_ledger.csv"
DEFAULT_PICK_FEATURES_OUTPUT = REPO_ROOT / "data" / "trades" / "draft_pick_trade_features.csv"

# Approximation of common NFL draft trade-chart points. Values are for relative
# draft capital only; they are not labels and should not be interpreted as AV.
PICK_VALUE_ANCHORS = {
    1: 3000,
    2: 2600,
    3: 2200,
    4: 1800,
    5: 1700,
    6: 1600,
    7: 1500,
    8: 1400,
    9: 1350,
    10: 1300,
    11: 1250,
    12: 1200,
    13: 1150,
    14: 1100,
    15: 1050,
    16: 1000,
    17: 950,
    18: 900,
    19: 875,
    20: 850,
    21: 800,
    22: 780,
    23: 760,
    24: 740,
    25: 720,
    26: 700,
    27: 680,
    28: 660,
    29: 640,
    30: 620,
    31: 600,
    32: 590,
    64: 270,
    96: 116,
    128: 43,
    160: 28,
    192: 14,
    224: 4,
    256: 1,
}


@dataclass
class Asset:
    asset_type: str
    from_team: str
    to_team: str
    description: str
    pick_season: int | None
    pick_round: int | None
    pick_number: int | None
    conditional: bool | None
    selected_player_id: str | None
    selected_player_name: str | None
    approx_pick_value_points: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--trade-output", type=Path, default=DEFAULT_TRADE_OUTPUT)
    parser.add_argument("--team-ledger-output", type=Path, default=DEFAULT_TEAM_LEDGER_OUTPUT)
    parser.add_argument("--pick-ledger-output", type=Path, default=DEFAULT_PICK_LEDGER_OUTPUT)
    parser.add_argument("--pick-features-output", type=Path, default=DEFAULT_PICK_FEATURES_OUTPUT)
    return parser.parse_args()


def clean_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def clean_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value else None


def approx_pick_value(pick_number: int | None) -> float | None:
    """Return an interpolated draft-capital point estimate for known picks."""

    if pick_number is None or pick_number <= 0:
        return None
    if pick_number in PICK_VALUE_ANCHORS:
        return float(PICK_VALUE_ANCHORS[pick_number])

    anchors = sorted(PICK_VALUE_ANCHORS)
    if pick_number < anchors[0]:
        return float(PICK_VALUE_ANCHORS[anchors[0]])
    if pick_number > anchors[-1]:
        extra_picks = pick_number - anchors[-1]
        return float(max(0.25, PICK_VALUE_ANCHORS[anchors[-1]] * np.exp(-0.08 * extra_picks)))

    lo = max(anchor for anchor in anchors if anchor < pick_number)
    hi = min(anchor for anchor in anchors if anchor > pick_number)
    lo_value = float(PICK_VALUE_ANCHORS[lo])
    hi_value = float(PICK_VALUE_ANCHORS[hi])

    # Interpolate on log-values so late-round picks decay smoothly.
    weight = (pick_number - lo) / (hi - lo)
    log_value = np.log(lo_value) + weight * (np.log(hi_value) - np.log(lo_value))
    return float(np.exp(log_value))


def classify_asset(row: pd.Series) -> str:
    has_pick_context = any(pd.notna(row[col]) for col in ["pick_season", "pick_round", "pick_number"])
    if has_pick_context:
        return "draft_pick"
    if pd.notna(row.get("pfr_id")) or pd.notna(row.get("pfr_name")):
        return "player"
    return "unknown"


def describe_asset(row: pd.Series, asset_type: str) -> str:
    pfr_name = clean_str(row.get("pfr_name"))
    pfr_id = clean_str(row.get("pfr_id"))
    player_label = pfr_name or pfr_id

    if asset_type == "player":
        return player_label or "unknown player"

    if asset_type == "draft_pick":
        pick_season = clean_int(row.get("pick_season"))
        pick_round = clean_int(row.get("pick_round"))
        pick_number = clean_int(row.get("pick_number"))
        conditional = clean_int(row.get("conditional")) == 1 if pd.notna(row.get("conditional")) else False

        parts = []
        if pick_season is not None:
            parts.append(str(pick_season))
        if pick_round is not None:
            parts.append(f"R{pick_round}")
        if pick_number is not None:
            parts.append(f"#{pick_number}")
        if not parts:
            parts.append("draft pick")
        description = " ".join(parts)
        if conditional:
            description = f"conditional {description}"
        if player_label:
            description = f"{description} ({player_label})"
        return description

    return "unknown asset"


def row_to_asset(row: pd.Series) -> Asset:
    asset_type = classify_asset(row)
    pick_number = clean_int(row.get("pick_number"))
    conditional = clean_int(row.get("conditional")) == 1 if pd.notna(row.get("conditional")) else None
    return Asset(
        asset_type=asset_type,
        from_team=str(row["gave"]),
        to_team=str(row["received"]),
        description=describe_asset(row, asset_type),
        pick_season=clean_int(row.get("pick_season")),
        pick_round=clean_int(row.get("pick_round")),
        pick_number=pick_number,
        conditional=conditional,
        selected_player_id=clean_str(row.get("pfr_id")) if asset_type == "draft_pick" else None,
        selected_player_name=clean_str(row.get("pfr_name")) if asset_type == "draft_pick" else None,
        approx_pick_value_points=approx_pick_value(pick_number),
    )


def empty_team_summary(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "sent_assets": [],
        "received_assets": [],
        "sent_players": 0,
        "received_players": 0,
        "sent_picks": 0,
        "received_picks": 0,
        "sent_known_picks": 0,
        "received_known_picks": 0,
        "sent_conditional_picks": 0,
        "received_conditional_picks": 0,
        "sent_pick_value_points": 0.0,
        "received_pick_value_points": 0.0,
        "earliest_pick_sent": None,
        "earliest_pick_received": None,
        "sent_rounds": [],
        "received_rounds": [],
    }


def update_team_summary(summary: dict[str, Any], asset: Asset, direction: str) -> None:
    prefix = "sent" if direction == "sent" else "received"
    summary[f"{prefix}_assets"].append(asset.description)

    if asset.asset_type == "player":
        summary[f"{prefix}_players"] += 1
        return

    if asset.asset_type != "draft_pick":
        return

    summary[f"{prefix}_picks"] += 1
    if asset.pick_number is not None:
        summary[f"{prefix}_known_picks"] += 1
        earliest_key = f"earliest_pick_{prefix}"
        current = summary[earliest_key]
        summary[earliest_key] = asset.pick_number if current is None else min(current, asset.pick_number)
    if asset.conditional:
        summary[f"{prefix}_conditional_picks"] += 1
    if asset.approx_pick_value_points is not None:
        summary[f"{prefix}_pick_value_points"] += asset.approx_pick_value_points
    if asset.pick_round is not None:
        summary[f"{prefix}_rounds"].append(asset.pick_round)


def finalize_team_summary(summary: dict[str, Any]) -> dict[str, Any]:
    for key in ["sent_rounds", "received_rounds"]:
        summary[key] = sorted(set(summary[key]))
    summary["net_players"] = summary["received_players"] - summary["sent_players"]
    summary["net_picks"] = summary["received_picks"] - summary["sent_picks"]
    summary["net_known_picks"] = summary["received_known_picks"] - summary["sent_known_picks"]
    summary["net_pick_value_points"] = (
        summary["received_pick_value_points"] - summary["sent_pick_value_points"]
    )
    return summary


def pipe_join(values: list[str]) -> str:
    return " | ".join(values)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def infer_trade_type(assets: list[Asset], team_summaries: dict[str, dict[str, Any]]) -> str:
    n_teams = len(team_summaries)
    has_players = any(asset.asset_type == "player" for asset in assets)
    has_picks = any(asset.asset_type == "draft_pick" for asset in assets)
    if n_teams > 2:
        return "multi_team"
    if has_players and has_picks:
        return "player_for_draft_capital_or_mixed"
    if has_players:
        return "player_only"
    if has_picks:
        return "pick_only"
    return "unknown"


def max_positive_team(team_summaries: dict[str, dict[str, Any]], key: str) -> str | None:
    if not team_summaries:
        return None
    team, value = max(
        ((team, summary[key]) for team, summary in team_summaries.items()),
        key=lambda item: item[1],
    )
    return team if value > 0 else None


def compress_trade(group: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    group = group.sort_values(["gave", "received", "pick_season", "pick_number", "pfr_name"], na_position="last")
    assets = [row_to_asset(row) for _, row in group.iterrows()]
    teams = sorted(set(group["gave"].dropna()) | set(group["received"].dropna()))
    team_summaries = {team: empty_team_summary(team) for team in teams}

    for asset in assets:
        update_team_summary(team_summaries[asset.from_team], asset, "sent")
        update_team_summary(team_summaries[asset.to_team], asset, "received")

    team_summaries = {
        team: finalize_team_summary(summary)
        for team, summary in sorted(team_summaries.items())
    }

    trade_id = int(group["trade_id"].iloc[0])
    trade_date = str(group["trade_date"].iloc[0])
    season = int(group["season"].iloc[0])
    team_a = teams[0] if len(teams) >= 1 else None
    team_b = teams[1] if len(teams) >= 2 else None
    team_a_summary = team_summaries.get(team_a, empty_team_summary("")) if team_a else empty_team_summary("")
    team_b_summary = team_summaries.get(team_b, empty_team_summary("")) if team_b else empty_team_summary("")

    asset_dicts = [asdict(asset) for asset in assets]
    team_summary_dicts = list(team_summaries.values())
    asset_summary = "; ".join(
        f"{asset.from_team}->{asset.to_team}: {asset.description}"
        for asset in assets
    )

    trade_row = {
        "trade_id": trade_id,
        "season": season,
        "trade_date": trade_date,
        "n_assets": len(assets),
        "n_teams": len(teams),
        "teams": pipe_join(teams),
        "is_multiteam": len(teams) > 2,
        "trade_type": infer_trade_type(assets, team_summaries),
        "has_players": any(asset.asset_type == "player" for asset in assets),
        "has_draft_picks": any(asset.asset_type == "draft_pick" for asset in assets),
        "has_conditional_picks": any(asset.conditional for asset in assets if asset.asset_type == "draft_pick"),
        "n_player_assets": sum(asset.asset_type == "player" for asset in assets),
        "n_pick_assets": sum(asset.asset_type == "draft_pick" for asset in assets),
        "known_pick_value_points_total": sum(
            asset.approx_pick_value_points or 0.0 for asset in assets
        ),
        "player_receiver_team": max_positive_team(team_summaries, "net_players"),
        "pick_value_receiver_team": max_positive_team(team_summaries, "net_pick_value_points"),
        "team_a": team_a,
        "team_b": team_b,
        "team_a_assets_sent": pipe_join(team_a_summary["sent_assets"]),
        "team_a_assets_received": pipe_join(team_a_summary["received_assets"]),
        "team_a_sent_players": team_a_summary["sent_players"],
        "team_a_received_players": team_a_summary["received_players"],
        "team_a_sent_picks": team_a_summary["sent_picks"],
        "team_a_received_picks": team_a_summary["received_picks"],
        "team_a_net_players": team_a_summary["net_players"],
        "team_a_net_picks": team_a_summary["net_picks"],
        "team_a_net_pick_value_points": team_a_summary["net_pick_value_points"],
        "team_b_assets_sent": pipe_join(team_b_summary["sent_assets"]),
        "team_b_assets_received": pipe_join(team_b_summary["received_assets"]),
        "team_b_sent_players": team_b_summary["sent_players"],
        "team_b_received_players": team_b_summary["received_players"],
        "team_b_sent_picks": team_b_summary["sent_picks"],
        "team_b_received_picks": team_b_summary["received_picks"],
        "team_b_net_players": team_b_summary["net_players"],
        "team_b_net_picks": team_b_summary["net_picks"],
        "team_b_net_pick_value_points": team_b_summary["net_pick_value_points"],
        "asset_summary": asset_summary,
        "assets_json": json_dumps(asset_dicts),
        "team_summary_json": json_dumps(team_summary_dicts),
    }

    team_rows = []
    for team, summary in team_summaries.items():
        team_rows.append(
            {
                "trade_id": trade_id,
                "season": season,
                "trade_date": trade_date,
                "team": team,
                "n_teams": len(teams),
                "trade_type": trade_row["trade_type"],
                "counterparty_teams": pipe_join([other for other in teams if other != team]),
                "sent_assets": pipe_join(summary["sent_assets"]),
                "received_assets": pipe_join(summary["received_assets"]),
                "sent_players": summary["sent_players"],
                "received_players": summary["received_players"],
                "sent_picks": summary["sent_picks"],
                "received_picks": summary["received_picks"],
                "sent_known_picks": summary["sent_known_picks"],
                "received_known_picks": summary["received_known_picks"],
                "sent_conditional_picks": summary["sent_conditional_picks"],
                "received_conditional_picks": summary["received_conditional_picks"],
                "sent_pick_value_points": summary["sent_pick_value_points"],
                "received_pick_value_points": summary["received_pick_value_points"],
                "net_players": summary["net_players"],
                "net_picks": summary["net_picks"],
                "net_known_picks": summary["net_known_picks"],
                "net_pick_value_points": summary["net_pick_value_points"],
                "earliest_pick_sent": summary["earliest_pick_sent"],
                "earliest_pick_received": summary["earliest_pick_received"],
                "sent_rounds": pipe_join([str(round_) for round_ in summary["sent_rounds"]]),
                "received_rounds": pipe_join([str(round_) for round_ in summary["received_rounds"]]),
            }
        )

    pick_rows = []
    for asset in assets:
        if asset.asset_type != "draft_pick":
            continue
        pick_rows.append(
            {
                "trade_id": trade_id,
                "season": season,
                "trade_date": trade_date,
                "gave": asset.from_team,
                "received": asset.to_team,
                "pick_season": asset.pick_season,
                "pick_round": asset.pick_round,
                "pick_number": asset.pick_number,
                "conditional": asset.conditional,
                "approx_pick_value_points": asset.approx_pick_value_points,
                "selected_player_id": asset.selected_player_id,
                "selected_player_name": asset.selected_player_name,
                "description": asset.description,
            }
        )

    return trade_row, team_rows, pick_rows


def compress_trades(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for col in ["pick_season", "pick_round", "pick_number", "conditional"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    trade_rows = []
    team_rows = []
    pick_rows = []
    for _, group in raw.groupby("trade_id", sort=True):
        trade_row, trade_team_rows, trade_pick_rows = compress_trade(group)
        trade_rows.append(trade_row)
        team_rows.extend(trade_team_rows)
        pick_rows.extend(trade_pick_rows)

    return (
        pd.DataFrame(trade_rows).sort_values(["trade_date", "trade_id"]),
        pd.DataFrame(team_rows).sort_values(["trade_date", "trade_id", "team"]),
        pd.DataFrame(pick_rows).sort_values(["trade_date", "trade_id", "pick_season", "pick_number"]),
    )


def build_pick_trade_features(pick_ledger: pd.DataFrame) -> pd.DataFrame:
    """Aggregate traded-pick assets into draft-pick-level model features.

    These are safe structural features only. Selected-player columns are
    intentionally excluded because they are outcomes, not draft-night inputs.
    """

    known_picks = pick_ledger.dropna(subset=["pick_season", "pick_number"]).copy()
    if known_picks.empty:
        return pd.DataFrame()

    known_picks["pick_season"] = known_picks["pick_season"].astype(int)
    known_picks["pick_number"] = known_picks["pick_number"].astype(int)
    known_picks["trade_date"] = pd.to_datetime(known_picks["trade_date"], errors="coerce")
    known_picks["conditional"] = known_picks["conditional"].fillna(False).astype(bool)

    rows = []
    for (pick_season, pick_number), group in known_picks.groupby(["pick_season", "pick_number"], sort=True):
        group = group.sort_values(["trade_date", "trade_id"])
        teams = sorted(set(group["gave"].dropna()) | set(group["received"].dropna()))
        rows.append(
            {
                "draft_season": int(pick_season),
                "pick": int(pick_number),
                "pick_was_traded": True,
                "pick_trade_count": int(len(group)),
                "pick_future_trade_count": int((group["season"] < pick_season).sum()),
                "pick_same_year_trade_count": int((group["season"] == pick_season).sum()),
                "pick_post_year_trade_count": int((group["season"] > pick_season).sum()),
                "pick_ever_conditional": bool(group["conditional"].any()),
                "pick_first_trade_date": group["trade_date"].iloc[0].strftime("%Y-%m-%d"),
                "pick_last_trade_date": group["trade_date"].iloc[-1].strftime("%Y-%m-%d"),
                "pick_first_gave_team": group["gave"].iloc[0],
                "pick_first_received_team": group["received"].iloc[0],
                "pick_last_gave_team": group["gave"].iloc[-1],
                "pick_last_received_team": group["received"].iloc[-1],
                "pick_trade_team_chain": pipe_join(teams),
                "pick_trade_n_distinct_teams": int(len(teams)),
                "pick_approx_value_points": float(group["approx_pick_value_points"].dropna().max())
                if group["approx_pick_value_points"].notna().any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["draft_season", "pick"])


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    trades, team_ledger, pick_ledger = compress_trades(raw)
    pick_features = build_pick_trade_features(pick_ledger)

    args.trade_output.parent.mkdir(parents=True, exist_ok=True)
    args.team_ledger_output.parent.mkdir(parents=True, exist_ok=True)
    args.pick_ledger_output.parent.mkdir(parents=True, exist_ok=True)
    args.pick_features_output.parent.mkdir(parents=True, exist_ok=True)

    trades.to_csv(args.trade_output, index=False)
    team_ledger.to_csv(args.team_ledger_output, index=False)
    pick_ledger.to_csv(args.pick_ledger_output, index=False)
    pick_features.to_csv(args.pick_features_output, index=False)

    print(f"Wrote {len(trades):,} compressed trades: {args.trade_output}")
    print(f"Wrote {len(team_ledger):,} team-trade rows: {args.team_ledger_output}")
    print(f"Wrote {len(pick_ledger):,} traded-pick rows: {args.pick_ledger_output}")
    print(f"Wrote {len(pick_features):,} draft-pick feature rows: {args.pick_features_output}")


if __name__ == "__main__":
    main()
