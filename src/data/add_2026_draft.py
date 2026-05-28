"""
Normalize 2026 NFL draft picks from ESPN raw data and append to drafts.csv.

The ESPN source (data/raw_cfb/draft_picks_2000_2026.csv) already contains 2026
picks with player name, position, college, team, and pre-draft grades.
Career stats are left NaN — players haven't played yet.

Idempotent: exits cleanly if 2026 rows are already present in drafts.csv.

Run:
    python -m src.data.add_2026_draft
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

ESPN_SOURCE = REPO_ROOT / "data" / "raw_cfb" / "draft_picks_2000_2026.csv"
DRAFTS_CSV  = REPO_ROOT / "data" / "supabase_exports" / "drafts.csv"

DRAFT_YEAR  = 2026

# ── Position mapping ──────────────────────────────────────────────────────────
# ESPN full name → PFR-style short code
ESPN_TO_PFR_POS: dict[str, str] = {
    "Quarterback":      "QB",
    "Running Back":     "RB",
    "Full Back":        "FB",
    "Wide Receiver":    "WR",
    "Tight End":        "TE",
    "Offensive Tackle": "OT",
    "Offensive Guard":  "G",
    "Center":           "C",
    "Offensive Line":   "OL",
    "Defensive End":    "DE",
    "Defensive Edge":   "DE",
    "Defensive Tackle": "DT",
    "Defensive Line":   "DL",
    "Linebacker":       "LB",
    "Outside Linebacker": "OLB",
    "Inside Linebacker":  "ILB",
    "Middle Linebacker":  "MLB",
    "Cornerback":       "CB",
    "Safety":           "S",
    "Defensive Back":   "DB",
    "Place Kicker":     "K",
    "Kicker":           "K",
    "Punter":           "P",
    "Long Snapper":     "LS",
    "Kick Returner":    "KR",
}

# PFR position code → (position_group, category, side)
POS_META: dict[str, tuple[str, str, str]] = {
    "QB":  ("QB", "QB", "O"),
    "RB":  ("RB", "RB", "O"),
    "FB":  ("RB", "RB", "O"),
    "WR":  ("WR", "WR", "O"),
    "TE":  ("TE", "TE", "O"),
    "OT":  ("OL", "OL", "O"),
    "G":   ("OL", "OL", "O"),
    "C":   ("OL", "OL", "O"),
    "OL":  ("OL", "OL", "O"),
    "DE":  ("DL", "DL", "D"),
    "DT":  ("DL", "DL", "D"),
    "NT":  ("DL", "DL", "D"),
    "DL":  ("DL", "DL", "D"),
    "LB":  ("LB", "LB", "D"),
    "OLB": ("LB", "LB", "D"),
    "ILB": ("LB", "LB", "D"),
    "MLB": ("LB", "LB", "D"),
    "CB":  ("DB", "DB", "D"),
    "S":   ("DB", "DB", "D"),
    "SS":  ("DB", "DB", "D"),
    "FS":  ("DB", "DB", "D"),
    "DB":  ("DB", "DB", "D"),
    "K":   ("ST", "K",  "S"),
    "P":   ("ST", "P",  "S"),
    "LS":  ("ST", "LS", "S"),
    "KR":  ("ST", "ST", "S"),
}

# ESPN nflTeamId → team code used in drafts.csv
ESPN_TEAM_ID_TO_CODE: dict[int, str] = {
    22: "ARI",
    1:  "ATL",
    33: "BAL",
    2:  "BUF",
    29: "CAR",
    3:  "CHI",
    4:  "CIN",
    5:  "CLE",
    6:  "DAL",
    7:  "DEN",
    8:  "DET",
    9:  "GNB",
    34: "HOU",
    11: "IND",
    30: "JAX",
    12: "KAN",
    13: "LVR",
    14: "LAR",
    24: "LAC",
    15: "MIA",
    16: "MIN",
    17: "NWE",
    18: "NOR",
    19: "NYJ",
    20: "NYG",
    21: "PHI",
    23: "PIT",
    25: "SFO",
    26: "SEA",
    27: "TAM",
    10: "TEN",
    28: "WAS",
}

# ESPN college name → drafts.csv college name (only entries that differ)
COLLEGE_MAP: dict[str, str] = {
    "Ohio State":        "Ohio St.",
    "Penn State":        "Penn St.",
    "Michigan State":    "Michigan St.",
    "Florida State":     "Florida St.",
    "Kansas State":      "Kansas St.",
    "Mississippi State": "Mississippi St.",
    "Boise State":       "Boise St.",
    "Arizona State":     "Arizona St.",
    "Ohio State":        "Ohio St.",
    "Miami":             "Miami (FL)",
    "Miami (Ohio)":      "Miami (OH)",
    "USC":               "Southern California",
    "TCU":               "TCU",
    "SMU":               "SMU",
    "Notre Dame":        "Notre Dame",
    "BYU":               "BYU",
    "Utah":              "Utah",
    "Texas A&M":         "Texas A&M",
    "UCLA":              "UCLA",
    "LSU":               "LSU",
    "Ole Miss":          "Mississippi",
    "Mississippi":       "Mississippi",
}


def _map_college(name: str) -> str:
    return COLLEGE_MAP.get(str(name).strip(), str(name).strip())


def build_2026_rows(espn: pd.DataFrame) -> pd.DataFrame:
    df = espn[espn["year"] == DRAFT_YEAR].copy()
    df = df.sort_values("overall").reset_index(drop=True)

    # Core identifiers
    rows = pd.DataFrame()
    rows["draft_pick_id"]  = df["overall"].apply(lambda p: f"{DRAFT_YEAR}_{int(p):03d}")
    rows["draft_season"]   = DRAFT_YEAR
    rows["round"]          = df["round"].astype(int)
    rows["pick"]           = df["overall"].astype(int)
    rows["team_raw"]       = df["nflTeamId"].map(ESPN_TEAM_ID_TO_CODE).fillna("UNK")
    rows["team"]           = rows["team_raw"]
    rows["franchise_id"]   = rows["team_raw"]  # simplified; will be NaN-safe in model
    rows["gsis_id"]        = np.nan
    rows["pfr_player_id"]  = df["name"].str.lower().str.replace(r"\s+", "_", regex=True)
    rows["pfr_player_name"] = df["name"]

    # Map ESPN position → PFR code → group / category / side
    pfr_pos = df["position"].str.strip().map(ESPN_TO_PFR_POS).fillna(df["position"].str.strip())
    rows["position"] = pfr_pos

    pos_group   = pfr_pos.map(lambda p: POS_META.get(p, ("UNK", "UNK", np.nan))[0])
    pos_cat     = pfr_pos.map(lambda p: POS_META.get(p, ("UNK", "UNK", np.nan))[1])
    pos_side    = pfr_pos.map(lambda p: POS_META.get(p, ("UNK", "UNK", np.nan))[2])
    rows["position_group"] = pos_group
    rows["category"]       = pos_cat
    rows["side"]           = pos_side

    rows["college"] = df["collegeTeam"].apply(_map_college)
    rows["age"]     = df.get("age", pd.Series(np.nan, index=df.index))

    # Career stats — unknown for fresh draftees
    stat_cols = [
        "hof", "last_nfl_season", "allpro", "probowls", "seasons_started",
        "w_av", "car_av", "dr_av", "games",
        "pass_completions", "pass_attempts", "pass_yards", "pass_tds", "pass_ints",
        "rush_atts", "rush_yards", "rush_tds",
        "receptions", "rec_yards", "rec_tds",
        "def_solo_tackles", "def_ints", "def_sacks",
    ]
    for c in stat_cols:
        rows[c] = np.nan

    return rows


def main() -> None:
    espn   = pd.read_csv(ESPN_SOURCE)
    drafts = pd.read_csv(DRAFTS_CSV)

    if DRAFT_YEAR in drafts["draft_season"].values:
        n = (drafts["draft_season"] == DRAFT_YEAR).sum()
        print(f"{DRAFT_YEAR} already present in drafts.csv ({n} rows) — nothing to do.")
        return

    new_rows = build_2026_rows(espn)

    # Align columns to existing schema (add missing cols as NaN)
    for col in drafts.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    new_rows = new_rows[drafts.columns]

    updated = pd.concat([drafts, new_rows], ignore_index=True)
    updated.to_csv(DRAFTS_CSV, index=False)

    print(f"Appended {len(new_rows)} picks for {DRAFT_YEAR} to {DRAFTS_CSV}")
    print(f"Total rows now: {len(updated)}")
    print()
    print("Sample (picks 1-5):")
    print(
        new_rows[["draft_season", "pick", "pfr_player_name", "position",
                  "position_group", "team", "college"]]
        .head(5)
        .to_string(index=False)
    )
    print()
    print("Note: roster/context features for 2026 are NOT yet regenerated.")
    print("Run src/data/clean_rosters.py to rebuild context features from 2025 roster data.")


if __name__ == "__main__":
    main()
