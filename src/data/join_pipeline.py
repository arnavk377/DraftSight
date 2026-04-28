"""Entity resolution and join logic to create master player table."""

import pandas as pd
import numpy as np
from pathlib import Path
from thefuzz import fuzz

from src.data.config import NFL_DATA_DIR, CFB_DATA_DIR, JOINED_DATA_DIR
from src.data.normalize_names import normalize_name, fuzzy_match_name


def load_nfl_data():
    """Load NFL datasets."""
    draft = pd.read_csv(NFL_DATA_DIR / "draft_picks.csv")
    combine = pd.read_csv(NFL_DATA_DIR / "combine.csv")

    rosters_path = NFL_DATA_DIR / "rosters.csv"
    if rosters_path.exists() and rosters_path.stat().st_size > 0:
        rosters = pd.read_csv(rosters_path)
    else:
        rosters = pd.DataFrame()

    return draft, combine, rosters


def load_cfb_data():
    """Load CFB datasets."""
    dfs = {}
    files = ["player_season_stats.csv", "recruiting.csv", "team_talent.csv", "sp_ratings.csv"]

    for fname in files:
        try:
            df = pd.read_csv(CFB_DATA_DIR / fname)
            dfs[fname.split(".")[0]] = df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            dfs[fname.split(".")[0]] = pd.DataFrame()

    return (
        dfs["player_season_stats"],
        dfs["recruiting"],
        dfs["team_talent"],
        dfs["sp_ratings"],
    )


def extract_final_season_stats(player_season_stats: pd.DataFrame) -> pd.DataFrame:
    """Extract final college season stats for each player."""
    if len(player_season_stats) == 0:
        return pd.DataFrame()

    final_season = (
        player_season_stats.sort_values("year")
        .groupby(["player_name", "school"])
        .last()
        .reset_index()
    )
    return final_season


def match_draft_to_combine(draft: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Match draft picks to combine data using name and season."""
    print("Matching draft picks to combine data...")

    if len(combine) == 0 or len(draft) == 0:
        print("  ! no combine or draft data found")
        return draft

    result = draft.copy()

    # Use correct column name for player name
    name_col = "pfr_player_name" if "pfr_player_name" in result.columns else "pfr_name"

    # Normalize for matching
    result["norm_name"] = result[name_col].apply(normalize_name)
    result["norm_school"] = result.get("college", result.get("school", "")).fillna("").apply(normalize_name)

    combine_norm = combine.copy()
    combine_norm["norm_name"] = combine_norm["player_name"].apply(normalize_name)
    combine_norm["norm_school"] = combine_norm.get("school", "").fillna("").apply(
        normalize_name
    )

    # Merge on exact normalized name match
    combine_cols = [
        "ht",
        "wt",
        "forty",
        "vertical",
        "bench_press",
        "broad_jump",
        "three_cone",
        "shuttle",
    ]
    available_combine_cols = [c for c in combine_cols if c in combine_norm.columns]

    result = result.merge(
        combine_norm[["norm_name"] + available_combine_cols],
        on="norm_name",
        how="left",
        suffixes=("", "_combine"),
    )

    matched_count = result[available_combine_cols[0]].notna().sum()
    print(f"  -> matched {matched_count}/{len(result)} draft picks to combine")

    result = result.drop(columns=["norm_name", "norm_school"], errors="ignore")
    return result


def match_draft_to_cfb(
    draft_with_combine: pd.DataFrame, player_season_stats: pd.DataFrame
) -> pd.DataFrame:
    """Match draft picks to college season stats."""
    print("Matching draft picks to college stats...")

    if len(player_season_stats) == 0:
        print("  ! no college stats found")
        return draft_with_combine

    # Get final season for each player
    final_stats = extract_final_season_stats(player_season_stats)

    result = draft_with_combine.copy()

    name_col = "pfr_player_name" if "pfr_player_name" in result.columns else "pfr_name"
    school_col = "college" if "college" in result.columns else "school"

    result["norm_name"] = result[name_col].apply(normalize_name)
    result["norm_school"] = result.get(school_col, "").fillna("").apply(normalize_name)

    final_stats["norm_name"] = final_stats["player_name"].apply(normalize_name)
    final_stats["norm_school"] = final_stats.get("school", "").fillna("").apply(
        normalize_name
    )

    # Merge on (name, school)
    stat_cols = [
        "pass_att",
        "pass_cmp",
        "pass_yds",
        "pass_td",
        "rush_att",
        "rush_yds",
        "rush_td",
        "rec",
        "rec_yds",
        "rec_td",
        "tackles",
        "sacks",
        "ints",
    ]
    available_stat_cols = [c for c in stat_cols if c in final_stats.columns]

    result = result.merge(
        final_stats[["norm_name", "norm_school"] + available_stat_cols],
        on=["norm_name", "norm_school"],
        how="left",
        suffixes=("", "_cfb"),
    )

    matched_count = (
        result[available_stat_cols[0]].notna().sum() if available_stat_cols else 0
    )
    print(f"  -> matched {matched_count}/{len(result)} draft picks to college stats")

    result = result.drop(columns=["norm_name", "norm_school"], errors="ignore")
    return result


def add_recruiting_context(
    draft_with_stats: pd.DataFrame, recruiting: pd.DataFrame
) -> pd.DataFrame:
    """Add recruiting pedigree to matched players."""
    print("Adding recruiting context...")

    if len(recruiting) == 0:
        print("  ! no recruiting data found")
        return draft_with_stats

    result = draft_with_stats.copy()

    name_col = "pfr_player_name" if "pfr_player_name" in result.columns else "pfr_name"
    result["norm_name"] = result[name_col].apply(normalize_name)

    recruiting_norm = recruiting.copy()
    recruiting_norm["norm_name"] = recruiting_norm["player_name"].apply(normalize_name)

    recruit_cols = ["rating", "stars", "rank_overall", "rank_position"]
    available_recruit_cols = [c for c in recruit_cols if c in recruiting_norm.columns]

    rename_dict = {c: f"recruiting_{c}" for c in available_recruit_cols}

    result = result.merge(
        recruiting_norm[["norm_name"] + available_recruit_cols].rename(
            columns=rename_dict
        ),
        on="norm_name",
        how="left",
    )

    matched_count = (
        result[[f"recruiting_{c}" for c in available_recruit_cols][0]].notna().sum()
        if available_recruit_cols
        else 0
    )
    print(f"  -> matched {matched_count}/{len(result)} to recruiting data")

    result = result.drop(columns=["norm_name"], errors="ignore")
    return result


def add_team_context(
    draft_with_context: pd.DataFrame,
    talent: pd.DataFrame,
    sp_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Add team talent and SP+ context."""
    print("Adding team context...")

    result = draft_with_context.copy()

    if len(talent) > 0:
        talent_norm = talent.copy()
        talent_norm["norm_school"] = talent_norm["school"].apply(normalize_name)

        # Get school column name (college or school)
        school_col = "college" if "college" in result.columns else "school"
        result["norm_school"] = result[school_col].fillna("").apply(normalize_name)

        # Use draft year - 1 as college season
        result["college_year"] = result["season"] - 1
        talent_norm = talent_norm.rename(columns={"year": "college_year"})

        result = result.merge(
            talent_norm[["norm_school", "college_year", "talent_rating"]],
            on=["norm_school", "college_year"],
            how="left",
        )

    if len(sp_ratings) > 0:
        sp_norm = sp_ratings.copy()
        sp_norm["norm_school"] = sp_norm["school"].apply(normalize_name)
        if "norm_school" not in result.columns:
            result["norm_school"] = result.get("school", "").fillna("").apply(
                normalize_name
            )
        if "college_year" not in result.columns:
            result["college_year"] = result["season"] - 1

        sp_norm = sp_norm.rename(columns={"year": "college_year"})

        result = result.merge(
            sp_norm[["norm_school", "college_year", "sp_rating"]],
            on=["norm_school", "college_year"],
            how="left",
        )

    result = result.drop(
        columns=["norm_school", "college_year"], errors="ignore"
    )

    print(f"  -> added team context")

    return result


def calculate_career_value(row) -> float:
    """Calculate career value metric from NFL stats."""
    value = 0

    # Games played
    for col in ["games", "g"]:
        if col in row and pd.notna(row[col]):
            try:
                value += int(float(row[col]))
                break
            except:
                pass

    # Passing stats
    for col in ["pass_yds", "pass_yards"]:
        if col in row and pd.notna(row[col]):
            try:
                value += int(float(row[col])) / 500
                break
            except:
                pass

    if pd.notna(row.get("pass_tds")):
        try:
            value += int(float(row["pass_tds"])) * 6
        except:
            pass

    # Rushing stats
    for col in ["rush_yds", "rush_yards"]:
        if col in row and pd.notna(row[col]):
            try:
                value += int(float(row[col])) / 100
                break
            except:
                pass

    if pd.notna(row.get("rush_tds")):
        try:
            value += int(float(row["rush_tds"])) * 6
        except:
            pass

    # Receiving stats
    for col in ["rec_yds", "rec_yards"]:
        if col in row and pd.notna(row[col]):
            try:
                value += int(float(row[col])) / 100
                break
            except:
                pass

    if pd.notna(row.get("rec_tds")):
        try:
            value += int(float(row["rec_tds"])) * 6
        except:
            pass

    # Defense
    if pd.notna(row.get("def_solo_tackles")):
        try:
            value += int(float(row["def_solo_tackles"])) * 0.5
        except:
            pass

    if pd.notna(row.get("def_sacks")):
        try:
            value += int(float(row["def_sacks"])) * 3
        except:
            pass

    if pd.notna(row.get("def_ints")):
        try:
            value += int(float(row["def_ints"])) * 4
        except:
            pass

    return max(0, value)


def build_master_table() -> pd.DataFrame:
    """Build the master player table by joining all data."""
    print("\n=== BUILDING MASTER PLAYER TABLE ===\n")

    # Load all data
    draft, combine, rosters = load_nfl_data()
    season_stats, recruiting, talent, sp_ratings = load_cfb_data()

    # Start with draft picks
    master = draft.copy()
    print(f"Starting with {len(master)} drafted players\n")

    # Add combine data
    master = match_draft_to_combine(master, combine)

    # Add college season stats
    master = match_draft_to_cfb(master, season_stats)

    # Add recruiting context
    master = add_recruiting_context(master, recruiting)

    # Add team context
    master = add_team_context(master, talent, sp_ratings)

    # Calculate career value metric from NFL stats
    print("Calculating career value metric...")
    master["career_value_metric"] = master.apply(calculate_career_value, axis=1)

    # Drop empty AV columns (car_av and dr_av are all NaN from nfl_data_py)
    master = master.drop(columns=["car_av", "dr_av"], errors="ignore")

    # Select final columns in order
    all_cols = list(master.columns)

    # Reorder to put key columns first
    priority_order = [
        "pfr_player_name",
        "college",
        "season",
        "round",
        "pick",
        "position",
        "career_value_metric",
    ]

    ordered_cols = (
        [c for c in priority_order if c in all_cols]
        + [c for c in all_cols if c not in priority_order]
    )

    master = master[ordered_cols]

    # Save
    JOINED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = JOINED_DATA_DIR / "master_player_table.csv"
    master.to_csv(output_path, index=False)

    print(f"\n=== MASTER TABLE SAVED ===")
    print(f"  {output_path}")
    print(f"  {len(master)} rows, {master.shape[1]} columns\n")

    return master


if __name__ == "__main__":
    build_master_table()
