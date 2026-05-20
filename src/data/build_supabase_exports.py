"""Create Supabase-ready CSV exports for the main DraftSight datasets.

The export folder contains one canonical CSV for each source domain:
  - drafts.csv
  - draft_board.csv
  - av.csv
  - trades.csv
  - rosters.csv
  - college_stats.csv
  - college_player_seasons.csv

It also includes one derived modeling convenience table:
  - draft_pick_context_features.csv

The script assumes the trade and roster preprocessing scripts have been run.
If their outputs are missing, it builds them first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from ast import literal_eval
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clean_rosters import clean_position, draft_team_code, franchise_id, position_group


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = REPO_ROOT / "data" / "supabase_exports"

DRAFTS_INPUT = REPO_ROOT / "src" / "data" / "raw" / "draft_picks.csv"
DRAFT_BOARD_INPUT = REPO_ROOT / "data" / "raw_cfb" / "draft_picks_2000_2026.csv"
COLLEGE_AGG_INPUT = REPO_ROOT / "data" / "clean_cfb" / "05_19_player_stats_aggregated.csv"
COLLEGE_SEASON_INPUT = REPO_ROOT / "data" / "all_results_wide.csv"
COLLEGE_SEASON_DRAFT_OVERALL_INPUT = REPO_ROOT / "data" / "clean_cfb" / "all_results_wide.csv"
AV_DIR = REPO_ROOT / "scraping_av" / "data"
ROSTERS_CLEANED = REPO_ROOT / "data" / "roster" / "rosters_cleaned.csv"
TRADES_COMPRESSED = REPO_ROOT / "data" / "trades" / "trades_compressed.csv"
CONTEXT_FEATURES = REPO_ROOT / "data" / "features" / "draft_pick_context_features.csv"

DEPENDENCY_SCRIPTS = [
    REPO_ROOT / "src" / "data" / "compress_trades.py",
    REPO_ROOT / "src" / "data" / "clean_rosters.py",
    REPO_ROOT / "src" / "data" / "build_draft_context_features.py",
]

NFL_TEAM_NAME_TO_CODE = {
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Green Bay": "GB",
    "Houston": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas City": "KC",
    "LA Chargers": "LAC",
    "LA Rams": "LAR",
    "Las Vegas": "LVR",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "New England": "NE",
    "New Orleans": "NO",
    "NY Giants": "NYG",
    "NY Jets": "NYJ",
    "Oakland": "LVR",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "San Diego": "LAC",
    "San Francisco": "SF",
    "Seattle": "SEA",
    "St. Louis": "LAR",
    "Tampa Bay": "TB",
    "Tennessee": "TEN",
    "Washington": "WAS",
}

FULL_POSITION_GROUPS = {
    "QUARTERBACK": "QB",
    "RUNNINGBACK": "RB",
    "FULLBACK": "RB",
    "WIDERECEIVER": "WR",
    "TIGHTEND": "TE",
    "OFFENSIVETACKLE": "OL",
    "OFFENSIVEGUARD": "OL",
    "CENTER": "OL",
    "DEFENSIVEEND": "DL",
    "DEFENSIVEEDGE": "DL",
    "DEFENSIVETACKLE": "DL",
    "LINEBACKER": "LB",
    "OUTSIDELINEBACKER": "LB",
    "INSIDELINEBACKER": "LB",
    "CORNERBACK": "DB",
    "SAFETY": "DB",
    "DEFENSIVEBACK": "DB",
    "PLACEKICKER": "ST",
    "PUNTER": "ST",
    "LONGSNAPPER": "ST",
    "KICKRETURNER": "ST",
}


def snake_case(name: str) -> str:
    cleaned = (
        str(name)
        .strip()
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    return cleaned.lower()


def ensure_dependencies() -> None:
    required_outputs = [ROSTERS_CLEANED, TRADES_COMPRESSED, CONTEXT_FEATURES]
    if all(path.exists() for path in required_outputs):
        return

    for script in DEPENDENCY_SCRIPTS:
        subprocess.run([sys.executable, str(script)], check=True, cwd=REPO_ROOT)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def bool_to_int(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].where(out[col].notna(), "")
    return out


def broad_position_group(value: object) -> str:
    cleaned = clean_position(value)
    return FULL_POSITION_GROUPS.get(cleaned, position_group(cleaned))


def export_drafts() -> pd.DataFrame:
    df = pd.read_csv(DRAFTS_INPUT)
    df.columns = [snake_case(col) for col in df.columns]
    df = df.rename(
        columns={
            "season": "draft_season",
            "to": "last_nfl_season",
            "cfb_player_id": "sports_ref_cfb_player_id",
        }
    )

    numeric_cols = [
        "draft_season",
        "round",
        "pick",
        "age",
        "last_nfl_season",
        "allpro",
        "probowls",
        "seasons_started",
        "w_av",
        "car_av",
        "dr_av",
        "games",
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
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["draft_season"] = df["draft_season"].astype(int)
    df["pick"] = df["pick"].astype(int)
    df["draft_pick_id"] = df["draft_season"].astype(str) + "_" + df["pick"].astype(str).str.zfill(3)
    df["team_raw"] = df["team"].astype(str)
    df["team"] = df["team_raw"].map(draft_team_code)
    df["franchise_id"] = df["team_raw"].map(franchise_id)
    df["position"] = df["position"].map(clean_position)
    df["position_group"] = df["position"].map(broad_position_group)
    if "hof" in df.columns:
        df["hof"] = bool_to_int(df["hof"])

    ordered = [
        "draft_pick_id",
        "draft_season",
        "round",
        "pick",
        "team",
        "team_raw",
        "franchise_id",
        "gsis_id",
        "pfr_player_id",
        "sports_ref_cfb_player_id",
        "pfr_player_name",
        "hof",
        "position",
        "position_group",
        "category",
        "side",
        "college",
        "age",
    ]
    remaining = [col for col in df.columns if col not in ordered]
    return normalize_string_columns(df[[col for col in ordered if col in df.columns] + remaining])


def parse_hometown_info(value: object) -> str:
    if pd.isna(value) or value == "":
        return ""
    try:
        parsed = literal_eval(str(value))
    except (ValueError, SyntaxError):
        return str(value)
    return json.dumps(parsed, sort_keys=True)


def export_draft_board() -> pd.DataFrame:
    df = pd.read_csv(DRAFT_BOARD_INPUT)
    df.columns = [snake_case(col) for col in df.columns]
    df = df.rename(
        columns={
            "year": "draft_season",
            "overall": "draft_overall",
            "pick": "draft_pick_in_round",
            "name": "player",
            "nflteam": "nfl_team",
            "nflteamid": "nfl_team_id",
            "nflathleteid": "nfl_athlete_id",
            "collegeid": "college_id",
            "collegeteam": "college_team",
            "collegeconference": "college_conference",
            "collegeathleteid": "cfb_player_id",
            "hometowninfo": "hometown_info_json",
            "predraftranking": "pre_draft_ranking",
            "predraftpositionranking": "pre_draft_position_ranking",
            "predraftgrade": "pre_draft_grade",
        }
    )
    for col in [
        "nfl_athlete_id",
        "college_id",
        "nfl_team_id",
        "draft_season",
        "draft_overall",
        "round",
        "draft_pick_in_round",
        "cfb_player_id",
        "height",
        "weight",
        "pre_draft_ranking",
        "pre_draft_position_ranking",
        "pre_draft_grade",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team"] = df["nfl_team"].map(NFL_TEAM_NAME_TO_CODE).fillna("")
    df["position"] = df["position"].map(clean_position)
    df["position_group"] = df["position"].map(broad_position_group)
    df["draft_board_id"] = (
        df["draft_season"].astype("Int64").astype(str)
        + "_"
        + df["draft_overall"].astype("Int64").astype(str).str.zfill(3)
    )
    if "hometown_info_json" in df.columns:
        df["hometown_info_json"] = df["hometown_info_json"].map(parse_hometown_info)

    ordered = [
        "draft_board_id",
        "draft_season",
        "draft_overall",
        "round",
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
        "college_conference",
        "height",
        "weight",
        "pre_draft_ranking",
        "pre_draft_position_ranking",
        "pre_draft_grade",
        "hometown_info_json",
    ]
    remaining = [col for col in df.columns if col not in ordered]
    return normalize_string_columns(df[[col for col in ordered if col in df.columns] + remaining])


def export_av() -> pd.DataFrame:
    rows = []
    for path in sorted(AV_DIR.glob("*_av.csv")):
        df = pd.read_csv(path)
        df.columns = [snake_case(col) for col in df.columns]
        df = df.rename(
            columns={
                "year": "season",
                "playerid": "pfr_player_id",
                "player": "player_name",
                "team": "team_raw",
                "position": "position",
                "experience": "experience_raw",
            }
        )
        rows.append(df)

    av = pd.concat(rows, ignore_index=True)
    av["season"] = pd.to_numeric(av["season"], errors="coerce").astype(int)
    av["av"] = pd.to_numeric(av["av"], errors="coerce").fillna(0.0)
    av["team_raw"] = av["team_raw"].astype(str).str.upper()
    av["team"] = av["team_raw"].map(draft_team_code)
    av["franchise_id"] = av["team_raw"].map(franchise_id)
    av["position"] = av["position"].map(clean_position)
    av["position_group"] = av["position"].map(broad_position_group)
    av["experience"] = (
        av["experience_raw"]
        .replace({"Rook": 0, "rook": 0, "ROOK": 0})
        .pipe(pd.to_numeric, errors="coerce")
    )
    av["av_row_id"] = (
        av["season"].astype(str)
        + "_"
        + av["team"].astype(str)
        + "_"
        + av["pfr_player_id"].astype(str)
    )

    ordered = [
        "av_row_id",
        "season",
        "team",
        "team_raw",
        "franchise_id",
        "pfr_player_id",
        "player_name",
        "position",
        "position_group",
        "experience",
        "experience_raw",
        "av",
    ]
    return normalize_string_columns(av[[col for col in ordered if col in av.columns]])


def export_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_COMPRESSED)
    df.columns = [snake_case(col) for col in df.columns]
    bool_cols = ["is_multiteam", "has_players", "has_draft_picks", "has_conditional_picks"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = bool_to_int(df[col])
    return normalize_string_columns(df)


def export_rosters() -> pd.DataFrame:
    df = pd.read_csv(ROSTERS_CLEANED)
    df.columns = [snake_case(col) for col in df.columns]
    bool_cols = [
        "is_active",
        "is_rookie",
        "is_young_player",
        "is_veteran_5plus",
        "is_drafted",
        "is_top100_draftee",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = bool_to_int(df[col])
    df["roster_row_id"] = (
        df["season"].astype(str)
        + "_"
        + df["team"].astype(str)
        + "_"
        + df["player_id"].astype(str)
    )
    ordered = ["roster_row_id"] + [col for col in df.columns if col != "roster_row_id"]
    return normalize_string_columns(df[ordered])


def _prepare_college_seasons() -> pd.DataFrame:
    df = pd.read_csv(COLLEGE_SEASON_INPUT)
    df.columns = [snake_case(col) for col in df.columns]

    if COLLEGE_SEASON_DRAFT_OVERALL_INPUT.exists():
        overall = pd.read_csv(
            COLLEGE_SEASON_DRAFT_OVERALL_INPUT,
            usecols=["player_id", "season", "draft_overall"],
        )
        overall.columns = [snake_case(col) for col in overall.columns]
        df = df.merge(overall, on=["player_id", "season"], how="left")

    df = df.rename(
        columns={
            "player_id": "cfb_player_id",
            "team": "college_team",
            "draft_year": "draft_season",
            "draft_pick": "draft_pick_in_round",
        }
    )
    if "position" in df.columns:
        df["position"] = df["position"].map(clean_position)
        df["position_group"] = df["position"].map(broad_position_group)

    numeric_skip = {"player", "college_team", "conference", "position", "position_group"}
    for col in df.columns:
        if col not in numeric_skip:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                pass

    return df


def export_college_stats() -> pd.DataFrame:
    df = pd.read_csv(COLLEGE_AGG_INPUT)
    df.columns = [snake_case(col) for col in df.columns]
    df = df.rename(
        columns={
            "team": "college_team",
            "draft_year": "draft_season",
            "draft_pick": "draft_pick_in_round",
        }
    )

    season_df = _prepare_college_seasons()
    id_lookup = (
        season_df.sort_values("season")
        .groupby(["player", "draft_season", "draft_round", "draft_pick_in_round"], as_index=False)
        .tail(1)[
            [
                "player",
                "draft_season",
                "draft_round",
                "draft_pick_in_round",
                "cfb_player_id",
                "draft_overall",
            ]
        ]
    )
    df = df.merge(
        id_lookup,
        on=["player", "draft_season", "draft_round", "draft_pick_in_round"],
        how="left",
    )

    if "position" in df.columns:
        df["position"] = df["position"].map(clean_position)
        df["position_group"] = df["position"].map(broad_position_group)

    numeric_skip = {"player", "college_team", "conference", "position", "position_group"}
    for col in df.columns:
        if col not in numeric_skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["college_stats_id"] = (
        df["draft_season"].astype("Int64").astype(str)
        + "_"
        + df["draft_overall"].astype("Int64").astype(str).str.zfill(3)
        + "_"
        + df["cfb_player_id"].astype("Int64").astype(str)
    )
    ordered = [
        "college_stats_id",
        "cfb_player_id",
        "player",
        "college_team",
        "conference",
        "position",
        "position_group",
        "draft_season",
        "draft_round",
        "draft_pick_in_round",
        "draft_overall",
        "num_seasons",
    ]
    remaining = [col for col in df.columns if col not in ordered]
    return normalize_string_columns(df[[col for col in ordered if col in df.columns] + remaining])


def export_college_player_seasons() -> pd.DataFrame:
    df = _prepare_college_seasons()
    df["college_player_season_id"] = (
        df["cfb_player_id"].astype("Int64").astype(str)
        + "_"
        + df["season"].astype("Int64").astype(str)
    )
    ordered = [
        "college_player_season_id",
        "cfb_player_id",
        "player",
        "college_team",
        "conference",
        "season",
        "draft_season",
        "draft_round",
        "draft_pick_in_round",
        "draft_overall",
        "position",
        "position_group",
    ]
    remaining = [col for col in df.columns if col not in ordered]
    return normalize_string_columns(df[[col for col in ordered if col in df.columns] + remaining])


def export_context_features() -> pd.DataFrame:
    df = pd.read_csv(CONTEXT_FEATURES)
    df.columns = [snake_case(col) for col in df.columns]
    bool_cols = ["roster_context_available", "pick_was_traded", "pick_ever_conditional"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = bool_to_int(df[col])
    return normalize_string_columns(df)


def write_readme(manifest: list[dict]) -> None:
    lines = [
        "# Supabase Exports",
        "",
        "Generated by `python src/data/build_supabase_exports.py`.",
        "",
        "These CSVs are split into historical draft truth tables, prospect/college source tables, NFL context tables, and one derived feature table. For modeling, start with `drafts.csv`, build labels from `av.csv`, then add college features from `college_stats.csv` and draft-time context from `draft_pick_context_features.csv`.",
        "",
        "## Recommended Import Order",
        "",
        "1. `drafts.csv`",
        "2. `draft_board.csv`",
        "3. `av.csv`",
        "4. `college_stats.csv`",
        "5. `college_player_seasons.csv`",
        "6. `rosters.csv`",
        "7. `trades.csv`",
        "8. `draft_pick_context_features.csv`",
        "",
        "## Quick Table Guide",
        "",
        "### Draft Tables",
        "",
        "- `drafts.csv` is the historical actual-draft table. Use this as the base table for backtests because it has one row per NFL draft pick and PFR-style player identifiers for joining to AV outcomes. Treat career/outcome columns such as `car_av`, `w_av`, games, Pro Bowls, All-Pro counts, and seasons started as labels or diagnostics, not draft-time features.",
        "- `draft_board.csv` is the board/prospect-style draft table from 2000-2026. It includes board ranks, grades, and 2026 prospects, so it is useful for future/prospect scoring. It should not replace `drafts.csv` for historical AV training because it does not carry the same PFR outcome identifiers.",
        "",
        "### College Tables",
        "",
        "- `college_stats.csv` is the one-row-per-player college aggregate table. This is the easiest college table to use in models because each row is already collapsed to a prospect/player and includes `cfb_player_id`, `draft_season`, and `draft_overall`.",
        "- `college_player_seasons.csv` is the wide player-season college table. It can have multiple rows per player, so use it for deeper feature engineering such as final-season production, year-over-year trends, breakout age, peak season, or career trajectory features. Do not join it directly into a one-row-per-pick model frame without aggregating first.",
        "",
        "### NFL Outcome And Context Tables",
        "",
        "- `av.csv` is the player-season Approximate Value table. For the current 2-year target, join it to `drafts.csv` by `pfr_player_id` and sum AV from the draft season and the following season.",
        "- `rosters.csv` is the cleaned player-season-team roster table. It is most useful for building team need and roster-strength features, but draft-time modeling should only use prior-season roster information.",
        "- `trades.csv` is one row per NFL trade with JSON asset summaries preserved. Use it to study trade behavior directly, or to derive pick-trade context. Be careful not to use selected-player outcomes from a trade as pre-draft features.",
        "- `draft_pick_context_features.csv` is a derived, one-row-per-pick modeling table aligned to `drafts.csv`. It combines leakage-safe traded-pick structure and prior-season roster context, including fields such as whether the pick was traded, trade count, conditional-pick flags, roster size, prior AV, and same-position roster depth.",
        "",
        "## Core Join Keys",
        "",
        "- Draft pick joins: `draft_season`, `pick`",
        "- Draft board joins: `draft_board.draft_season + draft_overall` -> `drafts.draft_season + pick` for historical years",
        "- Player joins across draft/AV: `pfr_player_id`",
        "- College aggregate joins to draft: `college_stats.draft_season + draft_overall` -> `drafts.draft_season + pick`",
        "- College season joins to aggregate: `cfb_player_id`",
        "- Draft `sports_ref_cfb_player_id` is a Sports Reference slug, not the same id as college `cfb_player_id`",
        "- Team context joins: `draft_season`, `franchise_id`, `team`",
        "",
        "## Suggested Modeling Approach",
        "",
        "1. Build the historical training frame from `drafts.csv`, keeping one row per `draft_season + pick`.",
        "2. Create the target from `av.csv`, usually 2-year AV or ROI by summing the player's AV in draft year and draft year plus one.",
        "3. Add draft-night features from `drafts.csv`, but exclude known post-draft outcome columns.",
        "4. Add college aggregate features from `college_stats.csv` using `draft_season + draft_overall` -> `draft_season + pick`.",
        "5. Add team and trade context from `draft_pick_context_features.csv` using `draft_season + pick`.",
        "6. Use `college_player_seasons.csv` later to engineer stronger college trend features before joining those aggregates back to the one-row-per-pick frame.",
        "7. Use `draft_board.csv` for future/prospect scoring after the model is trained on historical rows, especially for 2026-style board data.",
        "",
        "Recommended ablations for the next modeling round: draft-only, draft plus college, draft plus context, draft plus college plus context, and model families compared against the pick-binning baseline.",
        "",
        "## Leakage Notes",
        "",
        "- Do not use post-draft outcome fields from `drafts.csv` as features.",
        "- Do not use current-season roster data for draft-time predictions; use prior-season roster context only.",
        "- Do not use selected-player fields from `trades.csv` as model features if those fields would not be known before the pick.",
        "- Keep `college_player_seasons.csv` as a source table until it is aggregated to one row per draft prospect or pick.",
        "",
        "## Export Manifest",
        "",
    ]
    for item in manifest:
        lines.append(
            f"- `{item['file']}`: {item['rows']:,} rows, {item['columns']:,} columns. {item['description']}"
        )
    lines.append("")
    lines.append(
        "Note: `draft_pick_context_features.csv` is derived and leakage-safe for draft-time modeling; "
        "trade selected-player fields are intentionally not included there."
    )
    (EXPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dependencies()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    exports = [
        ("drafts.csv", export_drafts(), "Draft pick table from PFR/nflverse-style draft data."),
        (
            "draft_board.csv",
            export_draft_board(),
            "CFBD/board-style draft table from 2000-2026, including 2026 prospects and pre-draft rankings.",
        ),
        ("av.csv", export_av(), "Player-season Approximate Value table combined from yearly AV files."),
        (
            "college_stats.csv",
            export_college_stats(),
            "Enriched college career aggregate table from the 05_19 player stats pull.",
        ),
        (
            "college_player_seasons.csv",
            export_college_player_seasons(),
            "Wide college player-season statistics table from the latest all_results pull.",
        ),
        ("rosters.csv", export_rosters(), "Cleaned player-season-team roster table."),
        ("trades.csv", export_trades(), "One row per NFL trade, with JSON asset/team summaries preserved."),
        (
            "draft_pick_context_features.csv",
            export_context_features(),
            "Derived draft-pick feature table combining roster context and traded-pick structure.",
        ),
    ]

    manifest = []
    for filename, df, description in exports:
        path = EXPORT_DIR / filename
        write_csv(df, path)
        manifest.append(
            {
                "file": filename,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "description": description,
            }
        )
        print(f"Wrote {filename}: {len(df):,} rows x {len(df.columns):,} columns")

    write_readme(manifest)
    print(f"Supabase export folder: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
