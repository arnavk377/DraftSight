"""
Data loading for model_v5.1.

Sources:
  supabase_exports/  drafts.csv, college_stats.csv, av.csv
                     draft_pick_context_features.csv  (roster cols only — trade cols dropped)
  data/roster/       veteran_performance_features.csv  (vet_av, vet_age per prior season/team/pos)
                     draft_positional_context_features.csv  (pos_drafted, pos_first_overall)
                     pick_trade_flags.csv  (was_traded, was_primary_pick, trade_had_players)
"""

import numpy as np
import pandas as pd


# ── Feature column lists ──────────────────────────────────────────────────────

CFB_STAT_COLS = [
    "career_years",
    "passing_att", "passing_completions", "passing_int", "passing_td",
    "passing_yds", "passing_pct", "passing_ypa",
    "rushing_car", "rushing_td", "rushing_yds", "rushing_ypc",
    "receiving_rec", "receiving_td", "receiving_yds", "receiving_ypr",
    "defensive_pd", "defensive_qb_hur", "defensive_sacks", "defensive_solo",
    "defensive_td", "defensive_tfl", "defensive_tot",
    "interceptions_int", "interceptions_td", "interceptions_yds",
    "fumbles_fum", "fumbles_lost", "fumbles_rec",
    "kicking_fga", "kicking_fgm", "kicking_pts", "kicking_pct",
    "punting_no", "punting_yds", "punting_ypp",
    "kickreturns_no", "kickreturns_yds", "kickreturns_avg",
    "puntreturns_no", "puntreturns_yds", "puntreturns_avg",
]

DRAFT_NUM_COLS = ["pick", "round", "age"]
DRAFT_CAT_COLS = ["position", "position_group", "category", "team", "college", "side"]

# Roster depth from draft_pick_context_features.csv (trade cols dropped in v5.1)
CONTEXT_ROSTER_COLS = [
    "prev_roster_players", "prev_roster_active_players",
    "prev_roster_avg_age", "prev_roster_active_avg_age",
    "prev_roster_avg_years_exp", "prev_roster_active_avg_years_exp",
    "prev_roster_rookies", "prev_roster_young_players",
    "prev_roster_veterans_5plus", "prev_roster_drafted_players",
    "prev_roster_top100_draftees", "prev_roster_avg_draft_number",
    "prev_same_position_group_players", "prev_same_position_group_active_players",
    "prev_same_position_group_share", "prev_same_position_group_active_share",
    "prev_same_position_group_avg_age", "prev_same_position_group_avg_years_exp",
    "prev_same_position_group_rookies", "prev_same_position_group_young_players",
    "prev_same_position_group_veterans_5plus", "prev_same_position_group_drafted_players",
    "prev_same_position_group_top100_draftees",
]

# Trade flags from pick_trade_flags.csv
PICK_TRADE_COLS = ["was_traded", "was_primary_pick", "trade_had_players"]

# Same-draft team positional context from draft_positional_context_features.csv
DRAFT_POS_CONTEXT_COLS = ["pos_drafted", "pos_first_overall"]

# Veteran performance from veteran_performance_features.csv (prior-season, team, pos)
VET_PERF_COLS = ["vet_av", "vet_age"]

NUM_COLS = (
    DRAFT_NUM_COLS
    + CFB_STAT_COLS
    + CONTEXT_ROSTER_COLS
    + PICK_TRADE_COLS
    + DRAFT_POS_CONTEXT_COLS
    + VET_PERF_COLS
)
CAT_COLS = DRAFT_CAT_COLS


# ── Position → pos_category mapping ──────────────────────────────────────────

def derive_pos_category(position: pd.Series, position_group: pd.Series) -> pd.Series:
    """Map draft position/position_group to the pos_category used in veteran_performance_features.

    position_group values in drafts.csv: QB RB WR TE OL DL LB DB ST
    K and P players both appear under position_group='ST'; use position to distinguish them.
    """
    cat = position_group.str.upper().copy()
    is_st = cat == "ST"
    cat[is_st & (position.str.upper() == "K")] = "K"
    cat[is_st & (position.str.upper() == "P")] = "P"
    # remaining ST rows (LS, KR, PR, etc.) stay as ST
    known = {"QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P", "ST"}
    cat[~cat.isin(known)] = "OTHER"
    return cat


# ── Data loading ──────────────────────────────────────────────────────────────

def load_draft(drafts_csv: str) -> pd.DataFrame:
    df = pd.read_csv(drafts_csv)
    df = df.rename(columns={"draft_season": "season"})
    needed = [
        "season", "pick", "round", "team", "position", "position_group",
        "category", "side", "age", "college", "pfr_player_id", "pfr_player_name",
    ]
    df = df[[c for c in needed if c in df.columns]].copy()
    df["season"] = df["season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)
    for c in ["round", "age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["team", "position", "position_group", "category", "side", "college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def load_college_stats(college_stats_csv: str) -> pd.DataFrame:
    df = pd.read_csv(college_stats_csv)
    df = df.rename(columns={"num_seasons": "career_years"})
    stat_cols = [c for c in CFB_STAT_COLS if c in df.columns]
    keep = ["draft_season", "draft_overall"] + stat_cols
    df = df[[c for c in keep if c in df.columns]].copy()
    df["draft_season"] = df["draft_season"].astype(int)
    df["draft_overall"] = pd.to_numeric(df["draft_overall"], errors="coerce")
    df = df.dropna(subset=["draft_overall"]).copy()
    df["draft_overall"] = df["draft_overall"].astype(int)
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_av(av_csv: str) -> pd.DataFrame:
    df = pd.read_csv(av_csv)[["season", "pfr_player_id", "av"]].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["av"] = pd.to_numeric(df["av"], errors="coerce").fillna(0.0)
    return df.groupby(["season", "pfr_player_id"], as_index=False).agg(av=("av", "sum"))


def build_two_year_labels(av_long: pd.DataFrame) -> pd.DataFrame:
    a = av_long.rename(columns={"season": "draft_season", "av": "av_y"})
    b = av_long.rename(columns={"season": "next_season", "av": "av_y1"})[
        ["next_season", "pfr_player_id", "av_y1"]
    ]
    m = a.merge(b, on="pfr_player_id", how="left")
    m = m[m["next_season"] == m["draft_season"] + 1].copy()
    m["av_2yr"] = m["av_y"] + m["av_y1"].fillna(0.0)
    return m[["draft_season", "pfr_player_id", "av_2yr"]]


def join_college_stats(
    draft: pd.DataFrame,
    college_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    stat_cols = [c for c in CFB_STAT_COLS if c in college_stats.columns]
    cs_sub = college_stats[["draft_season", "draft_overall"] + stat_cols].copy()

    merged = draft.merge(
        cs_sub,
        left_on=["season", "pick"],
        right_on=["draft_season", "draft_overall"],
        how="left",
    ).drop(columns=["draft_season", "draft_overall"], errors="ignore")

    cs_min = int(college_stats["draft_season"].min())
    cs_max = int(college_stats["draft_season"].max())
    in_window = merged["season"].between(cs_min, cs_max)
    matched = merged[stat_cols[0]].notna() if stat_cols else pd.Series(False, index=merged.index)

    stats = {
        "total_picks": len(draft),
        "cs_window": (cs_min, cs_max),
        "picks_in_window": int(in_window.sum()),
        "matched_total": int(matched.sum()),
        "matched_in_window": int((in_window & matched).sum()),
        "match_rate_overall": float(matched.mean()),
        "match_rate_in_window": float(matched[in_window].mean()) if in_window.any() else 0.0,
    }
    return merged, stats


def load_roster_context(context_csv: str) -> pd.DataFrame:
    """Load draft_pick_context_features.csv keeping only leakage-safe roster depth cols."""
    df = pd.read_csv(context_csv)
    keep = ["draft_season", "pick"] + [c for c in CONTEXT_ROSTER_COLS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["draft_season"] = df["draft_season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)
    for c in CONTEXT_ROSTER_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_pick_trade_flags(trade_flags_csv: str) -> pd.DataFrame:
    """Load pick_trade_flags.csv — was_traded, was_primary_pick, trade_had_players."""
    df = pd.read_csv(trade_flags_csv)
    keep = ["season", "pick"] + [c for c in PICK_TRADE_COLS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={"season": "draft_season"})
    df["draft_season"] = df["draft_season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)
    for c in PICK_TRADE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_draft_pos_context(pos_context_csv: str) -> pd.DataFrame:
    """Load draft_positional_context_features.csv — pos_drafted, pos_first_overall."""
    df = pd.read_csv(pos_context_csv)
    keep = ["draft_season", "pick"] + [c for c in DRAFT_POS_CONTEXT_COLS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["draft_season"] = df["draft_season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)
    for c in DRAFT_POS_CONTEXT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_veteran_performance(vet_csv: str) -> pd.DataFrame:
    """Load veteran_performance_features.csv — vet_av, vet_age per (season, team, pos_category)."""
    df = pd.read_csv(vet_csv)
    keep = ["season", "team", "pos_category"] + [c for c in VET_PERF_COLS if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["team"] = df["team"].astype(str)
    df["pos_category"] = df["pos_category"].astype(str)
    for c in VET_PERF_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def join_veteran_performance(
    df: pd.DataFrame,
    vet: pd.DataFrame,
) -> pd.DataFrame:
    """Join prior-season veteran AV/age onto each draft pick.

    Derives pos_category from (position, position_group), then joins on
    (draft_season - 1, team, pos_category) so there is no leakage.
    """
    df = df.copy()
    df["pos_category"] = derive_pos_category(df["position"], df["position_group"])
    df["vet_season"] = df["draft_season"] - 1

    vet_renamed = vet.rename(columns={"season": "vet_season"})
    vet_renamed["vet_season"] = vet_renamed["vet_season"].astype(int)

    df = df.merge(
        vet_renamed[["vet_season", "team", "pos_category"] + VET_PERF_COLS],
        on=["vet_season", "team", "pos_category"],
        how="left",
    ).drop(columns=["vet_season", "pos_category"])

    return df
