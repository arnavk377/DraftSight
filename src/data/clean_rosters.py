"""Clean roster data and build draft-time roster context features.

Outputs are designed for downstream modeling, but this script does not modify
any model code. To avoid leakage, the draft-pick context table uses the team's
previous season roster only.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROSTER_INPUT = REPO_ROOT / "data" / "processed" / "rosters" / "rosters.csv"
DEFAULT_DRAFT_INPUT = REPO_ROOT / "data" / "raw" / "nfl" / "draft_picks.csv"
DEFAULT_CLEAN_OUTPUT = REPO_ROOT / "data" / "processed" / "rosters" / "rosters_cleaned.csv"
DEFAULT_TEAM_FEATURES_OUTPUT = REPO_ROOT / "data" / "processed" / "rosters" / "team_roster_features.csv"
DEFAULT_DRAFT_CONTEXT_OUTPUT = REPO_ROOT / "data" / "processed" / "rosters" / "draft_team_context_features.csv"
DEFAULT_PICK_CONTEXT_OUTPUT = (
    REPO_ROOT / "data" / "processed" / "rosters" / "draft_pick_roster_context_features.csv"
)

STATUS_PRIORITY = {
    "ACT": 0,
    "RES": 1,
    "INA": 2,
    "DEV": 3,
    "PUP": 4,
    "SUS": 5,
    "NWT": 6,
    "RSN": 7,
    "RSR": 8,
    "EXE": 9,
    "TRD": 10,
    "TRC": 11,
    "CUT": 12,
}

TEAM_TO_FRANCHISE = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "PHO": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "HST": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "RAM": "LAR",
    "SL": "LAR",
    "STL": "LAR",
    "LAC": "LAC",
    "SD": "LAC",
    "SDG": "LAC",
    "LV": "LVR",
    "LVR": "LVR",
    "OAK": "LVR",
    "RAI": "LVR",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
}

TEAM_TO_DRAFT_CODE = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "PHO": "PHO",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GNB",
    "GNB": "GNB",
    "HOU": "HOU",
    "HST": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "KAN",
    "KAN": "KAN",
    "LA": "LAR",
    "LAR": "LAR",
    "RAM": "RAM",
    "SL": "STL",
    "STL": "STL",
    "LAC": "LAC",
    "SD": "SDG",
    "SDG": "SDG",
    "LV": "LVR",
    "LVR": "LVR",
    "OAK": "OAK",
    "RAI": "RAI",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NWE",
    "NWE": "NWE",
    "NO": "NOR",
    "NOR": "NOR",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SFO",
    "SFO": "SFO",
    "TB": "TAM",
    "TAM": "TAM",
    "TEN": "TEN",
    "WAS": "WAS",
}

POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OL": "OL",
    "T": "OL",
    "OT": "OL",
    "G": "OL",
    "OG": "OL",
    "C": "OL",
    "DL": "DL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "DB": "DB",
    "CB": "DB",
    "S": "DB",
    "SS": "DB",
    "FS": "DB",
    "SAF": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
    "KR": "ST",
    "PR": "ST",
}
POSITION_GROUP_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST", "OTHER"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-input", type=Path, default=DEFAULT_ROSTER_INPUT)
    parser.add_argument("--draft-input", type=Path, default=DEFAULT_DRAFT_INPUT)
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_OUTPUT)
    parser.add_argument("--team-features-output", type=Path, default=DEFAULT_TEAM_FEATURES_OUTPUT)
    parser.add_argument("--draft-context-output", type=Path, default=DEFAULT_DRAFT_CONTEXT_OUTPUT)
    parser.add_argument("--pick-context-output", type=Path, default=DEFAULT_PICK_CONTEXT_OUTPUT)
    return parser.parse_args()


def clean_team_code(value: Any) -> str:
    if pd.isna(value):
        return "UNK"
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper()) or "UNK"


def franchise_id(value: Any) -> str:
    team = clean_team_code(value)
    return TEAM_TO_FRANCHISE.get(team, team)


def draft_team_code(value: Any) -> str:
    team = clean_team_code(value)
    return TEAM_TO_DRAFT_CODE.get(team, team)


def clean_position(value: Any) -> str:
    if pd.isna(value):
        return "UNK"
    position = re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())
    return position or "UNK"


def position_group(value: Any) -> str:
    return POSITION_GROUPS.get(clean_position(value), "OTHER")


def status_group(value: Any) -> str:
    status = clean_position(value)
    if status == "ACT":
        return "active"
    if status == "RES":
        return "reserve"
    if status == "DEV":
        return "development"
    if status == "CUT":
        return "cut"
    if status in {"INA", "PUP", "SUS", "NWT", "RSN", "RSR", "EXE"}:
        return "inactive_or_exception"
    if status in {"UFA", "RFA"}:
        return "free_agent"
    if status in {"TRD", "TRC", "TRT"}:
        return "transaction"
    return "other"


def draft_round_from_pick(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    pick = float(value)
    if pick <= 0:
        return np.nan
    return float(min(8, int(np.ceil(pick / 32.0))))


def first_not_null(series: pd.Series) -> Any:
    nonnull = series.dropna()
    return nonnull.iloc[0] if len(nonnull) else np.nan


def clean_roster(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [col.strip() for col in df.columns]

    for col in ["season", "age", "height", "weight", "years_exp", "entry_year", "rookie_year", "draft_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["season"] = df["season"].astype(int)
    df["team"] = df["team"].map(clean_team_code)
    df["team_draft_code"] = df["team"].map(draft_team_code)
    df["franchise_id"] = df["team"].map(franchise_id)
    df["position_raw"] = df["position"].map(clean_position)
    df["depth_chart_position_clean"] = df["depth_chart_position"].map(clean_position)
    df["roster_position"] = np.where(
        df["depth_chart_position_clean"].ne("UNK"),
        df["depth_chart_position_clean"],
        df["position_raw"],
    )
    df["position_group"] = df["roster_position"].map(position_group)
    df["status"] = df["status"].map(clean_position)
    df["status_group"] = df["status"].map(status_group)
    df["is_active"] = df["status"].eq("ACT")
    df["is_rookie"] = df["years_exp"].eq(0)
    df["is_young_player"] = df["years_exp"].between(0, 2, inclusive="both")
    df["is_veteran_5plus"] = df["years_exp"].ge(5)
    df["is_drafted"] = df["draft_number"].notna()
    df["is_top100_draftee"] = df["draft_number"].le(100)
    df["draft_round_est"] = df["draft_number"].map(draft_round_from_pick)
    df["bmi"] = np.where(df["height"].gt(0), 703 * df["weight"] / (df["height"] ** 2), np.nan)

    df["_status_priority"] = df["status"].map(STATUS_PRIORITY).fillna(99)
    df["_nonnull_count"] = df.notna().sum(axis=1)
    dedup_keys = ["season", "team", "player_id"]
    before = len(df)
    df = (
        df.sort_values(dedup_keys + ["_status_priority", "_nonnull_count"], ascending=[True, True, True, True, False])
        .drop_duplicates(dedup_keys, keep="first")
        .drop(columns=["_status_priority", "_nonnull_count"])
    )
    df.attrs["dropped_duplicate_rows"] = before - len(df)

    keep_cols = [
        "season",
        "team",
        "team_draft_code",
        "franchise_id",
        "player_id",
        "pfr_id",
        "esb_id",
        "smart_id",
        "player_name",
        "first_name",
        "last_name",
        "football_name",
        "position_raw",
        "depth_chart_position_clean",
        "roster_position",
        "position_group",
        "jersey_number",
        "status",
        "status_group",
        "is_active",
        "birth_date",
        "age",
        "height",
        "weight",
        "bmi",
        "college",
        "years_exp",
        "entry_year",
        "rookie_year",
        "draft_club",
        "draft_number",
        "draft_round_est",
        "is_rookie",
        "is_young_player",
        "is_veteran_5plus",
        "is_drafted",
        "is_top100_draftee",
    ]
    return df[[col for col in keep_cols if col in df.columns]].sort_values(["season", "franchise_id", "team", "player_name"])


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if series.notna().any() else np.nan


def aggregate_team_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, franchise, team), group in cleaned.groupby(["season", "franchise_id", "team_draft_code"], sort=True):
        active = group[group["is_active"]]
        drafted = group[group["is_drafted"]]
        row = {
            "season": int(season),
            "franchise_id": franchise,
            "team": team,
            "roster_players": int(len(group)),
            "roster_active_players": int(group["is_active"].sum()),
            "roster_reserve_players": int(group["status_group"].eq("reserve").sum()),
            "roster_development_players": int(group["status_group"].eq("development").sum()),
            "roster_cut_players": int(group["status_group"].eq("cut").sum()),
            "roster_avg_age": safe_mean(group["age"]),
            "roster_active_avg_age": safe_mean(active["age"]),
            "roster_avg_years_exp": safe_mean(group["years_exp"]),
            "roster_active_avg_years_exp": safe_mean(active["years_exp"]),
            "roster_avg_height": safe_mean(group["height"]),
            "roster_avg_weight": safe_mean(group["weight"]),
            "roster_avg_bmi": safe_mean(group["bmi"]),
            "roster_rookies": int(group["is_rookie"].sum()),
            "roster_young_players": int(group["is_young_player"].sum()),
            "roster_veterans_5plus": int(group["is_veteran_5plus"].sum()),
            "roster_drafted_players": int(group["is_drafted"].sum()),
            "roster_top100_draftees": int(group["is_top100_draftee"].sum()),
            "roster_avg_draft_number": safe_mean(drafted["draft_number"]),
        }

        for pos_group in POSITION_GROUP_ORDER:
            pos = group[group["position_group"].eq(pos_group)]
            pos_active = pos[pos["is_active"]]
            prefix = f"roster_group_{pos_group.lower()}"
            row[f"{prefix}_players"] = int(len(pos))
            row[f"{prefix}_active_players"] = int(pos["is_active"].sum())
            row[f"{prefix}_share"] = float(len(pos) / len(group)) if len(group) else np.nan
            row[f"{prefix}_active_share"] = (
                float(len(pos_active) / len(active)) if len(active) else np.nan
            )
            row[f"{prefix}_avg_age"] = safe_mean(pos["age"])
            row[f"{prefix}_avg_years_exp"] = safe_mean(pos["years_exp"])
            row[f"{prefix}_rookies"] = int(pos["is_rookie"].sum())
            row[f"{prefix}_young_players"] = int(pos["is_young_player"].sum())
            row[f"{prefix}_veterans_5plus"] = int(pos["is_veteran_5plus"].sum())
            row[f"{prefix}_drafted_players"] = int(pos["is_drafted"].sum())
            row[f"{prefix}_top100_draftees"] = int(pos["is_top100_draftee"].sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["season", "franchise_id", "team"])


def build_draft_team_context(team_features: pd.DataFrame) -> pd.DataFrame:
    context = team_features.copy()
    context = context.rename(columns={"season": "roster_context_season"})
    context["draft_season"] = context["roster_context_season"] + 1

    id_cols = ["draft_season", "roster_context_season", "franchise_id", "team"]
    feature_cols = [col for col in context.columns if col not in id_cols]
    context = context[id_cols + feature_cols].copy()
    rename_map = {col: f"prev_{col}" for col in feature_cols}
    return context.rename(columns=rename_map).sort_values(["draft_season", "franchise_id", "team"])


def build_draft_pick_context(draft_raw: pd.DataFrame, draft_context: pd.DataFrame) -> pd.DataFrame:
    draft = draft_raw.copy()
    if "draft_season" in draft.columns and "season" not in draft.columns:
        draft = draft.rename(columns={"draft_season": "season"})
    draft["draft_season"] = pd.to_numeric(draft["season"], errors="coerce").astype("Int64")
    draft["pick"] = pd.to_numeric(draft["pick"], errors="coerce").astype("Int64")
    draft["team"] = draft["team"].map(clean_team_code)
    draft["team_draft_code"] = draft["team"].map(draft_team_code)
    draft["franchise_id"] = draft["team"].map(franchise_id)
    draft["draft_position_group"] = draft["position"].map(position_group)

    merged = draft[["draft_season", "pick", "team", "team_draft_code", "franchise_id", "position", "draft_position_group"]].merge(
        draft_context,
        on=["draft_season", "franchise_id"],
        how="left",
        suffixes=("", "_context"),
    )
    merged["roster_context_available"] = merged["roster_context_season"].notna()

    metric_suffixes = [
        "players",
        "active_players",
        "share",
        "active_share",
        "avg_age",
        "avg_years_exp",
        "rookies",
        "young_players",
        "veterans_5plus",
        "drafted_players",
        "top100_draftees",
    ]

    for suffix in metric_suffixes:
        values = []
        for _, row in merged.iterrows():
            group = str(row["draft_position_group"]).lower()
            source_col = f"prev_roster_group_{group}_{suffix}"
            values.append(row[source_col] if source_col in merged.columns else np.nan)
        merged[f"prev_same_position_group_{suffix}"] = values

    keep_cols = [
        "draft_season",
        "pick",
        "team",
        "team_draft_code",
        "franchise_id",
        "position",
        "draft_position_group",
        "roster_context_available",
        "roster_context_season",
        "prev_roster_players",
        "prev_roster_active_players",
        "prev_roster_avg_age",
        "prev_roster_active_avg_age",
        "prev_roster_avg_years_exp",
        "prev_roster_active_avg_years_exp",
        "prev_roster_rookies",
        "prev_roster_young_players",
        "prev_roster_veterans_5plus",
        "prev_roster_drafted_players",
        "prev_roster_top100_draftees",
        "prev_roster_avg_draft_number",
    ] + [f"prev_same_position_group_{suffix}" for suffix in metric_suffixes]

    return merged[[col for col in keep_cols if col in merged.columns]].sort_values(["draft_season", "pick"])


def main() -> None:
    args = parse_args()
    raw_roster = pd.read_csv(args.roster_input)
    raw_draft = pd.read_csv(args.draft_input)

    cleaned = clean_roster(raw_roster)
    team_features = aggregate_team_features(cleaned)
    draft_context = build_draft_team_context(team_features)
    pick_context = build_draft_pick_context(raw_draft, draft_context)

    for path in [
        args.clean_output,
        args.team_features_output,
        args.draft_context_output,
        args.pick_context_output,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(args.clean_output, index=False)
    team_features.to_csv(args.team_features_output, index=False)
    draft_context.to_csv(args.draft_context_output, index=False)
    pick_context.to_csv(args.pick_context_output, index=False)

    dropped_dupes = cleaned.attrs.get("dropped_duplicate_rows", 0)
    print(f"Wrote cleaned roster rows: {len(cleaned):,} ({dropped_dupes:,} duplicate rows removed)")
    print(f"Wrote team-season roster features: {len(team_features):,} -> {args.team_features_output}")
    print(f"Wrote draft-team context features: {len(draft_context):,} -> {args.draft_context_output}")
    print(f"Wrote draft-pick roster context features: {len(pick_context):,} -> {args.pick_context_output}")
    print(
        "Draft-pick context coverage: "
        f"{pick_context['roster_context_available'].mean():.1%} overall, "
        f"{pick_context.loc[pick_context['draft_season'] >= 2010, 'roster_context_available'].mean():.1%} since 2010"
    )


if __name__ == "__main__":
    main()
