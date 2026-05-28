"""Build and append 2026 draft-class context features.

Uses 2025 roster data (already in supabase exports) to compute leakage-safe
context features for the 2026 draft class, then appends them to the three
feature CSVs consumed by model_v6:

  data/supabase_exports/draft_pick_context_features.csv   (roster depth)
  data/roster/draft_pick_roster_context_features.csv      (same, local copy)
  data/roster/draft_positional_context_features.csv       (pos_drafted, pos_first_overall)
  data/roster/pick_trade_flags.csv                        (was_traded etc., all zeros)

Veteran performance (veteran_performance_features.csv) already covers the
2025 season and is joined on (draft_season - 1), so no update is needed there.

Idempotent: re-running drops and replaces any existing 2026 rows.

Run from repo root:
    python -m src.data.add_2026_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_rosters import (
    aggregate_team_features,
    build_draft_pick_context,
    build_draft_team_context,
    clean_roster,
)

REPO_ROOT    = Path(__file__).resolve().parents[2]
SUPABASE_DIR = REPO_ROOT / "data" / "supabase_exports"
ROSTER_DIR   = REPO_ROOT / "data" / "roster"
DRAFT_YEAR   = 2026


# ── Helpers ───────────────────────────────────────────────────────────────────

def drop_year(df: pd.DataFrame, year_col: str, year: int) -> pd.DataFrame:
    """Remove existing rows for `year` so they can be replaced cleanly."""
    if year_col in df.columns:
        return df[pd.to_numeric(df[year_col], errors="coerce") != year].copy()
    return df.copy()


def append_and_save(
    path: Path,
    new_rows: pd.DataFrame,
    year_col: str,
    sort_cols: list[str] | None = None,
) -> None:
    existing = pd.read_csv(path)
    existing = drop_year(existing, year_col, DRAFT_YEAR)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    sort_by  = sort_cols or ([year_col, "pick"] if "pick" in combined.columns else [year_col])
    combined = combined.sort_values(sort_by).reset_index(drop=True)
    combined.to_csv(path, index=False)
    n = int((pd.to_numeric(combined[year_col], errors="coerce") == DRAFT_YEAR).sum())
    print(f"  {path.name}: appended {n} rows for {DRAFT_YEAR}")


# ── Feature builders ──────────────────────────────────────────────────────────

def build_roster_context_2026(
    roster_csv: Path,
    drafts_csv: Path,
) -> pd.DataFrame:
    """Compute 2026 draft-pick roster context from 2025 roster data."""
    raw_roster = pd.read_csv(roster_csv)
    raw_draft  = pd.read_csv(drafts_csv)

    roster_prev = raw_roster[raw_roster["season"] == DRAFT_YEAR - 1].copy()
    if roster_prev.empty:
        raise ValueError(f"No roster rows found for season {DRAFT_YEAR - 1} in {roster_csv}")

    draft_2026 = raw_draft[
        pd.to_numeric(raw_draft.get("draft_season", raw_draft.get("season")), errors="coerce") == DRAFT_YEAR
    ].copy()
    if draft_2026.empty:
        raise ValueError(f"No draft picks found for {DRAFT_YEAR} in {drafts_csv}")

    cleaned       = clean_roster(roster_prev)
    team_features = aggregate_team_features(cleaned)
    draft_context = build_draft_team_context(team_features)
    pick_context  = build_draft_pick_context(draft_2026, draft_context)

    return pick_context


def build_positional_context_2026(drafts_csv: Path) -> pd.DataFrame:
    """Compute pos_drafted / pos_first_overall for each 2026 pick."""
    drafts = pd.read_csv(drafts_csv)
    d26 = (
        drafts[pd.to_numeric(drafts.get("draft_season", drafts.get("season")), errors="coerce") == DRAFT_YEAR]
        [["draft_season", "round", "pick", "team", "position_group"]]
        .copy()
        .sort_values("pick")
        .reset_index(drop=True)
    )

    d26["pos_first_overall"] = (
        d26.groupby(["draft_season", "team", "position_group"])["pick"]
        .transform(lambda s: s.shift(1).expanding().min())
    )
    d26["pos_drafted"] = d26["pos_first_overall"].notna().astype(int)

    return d26[[
        "draft_season", "pick", "round", "team", "position_group",
        "pos_drafted", "pos_first_overall",
    ]]


def build_trade_flags_2026(drafts_csv: Path) -> pd.DataFrame:
    """Trade flags for 2026 picks — all zeros (no 2026 trade data yet)."""
    drafts = pd.read_csv(drafts_csv)
    d26 = (
        drafts[pd.to_numeric(drafts.get("draft_season", drafts.get("season")), errors="coerce") == DRAFT_YEAR]
        [["draft_season", "pick", "round", "team"]]
        .copy()
        .rename(columns={"draft_season": "season"})
    )
    d26["was_traded"]        = 0
    d26["was_primary_pick"]  = 0
    d26["trade_had_players"] = 0
    return d26[["season", "pick", "round", "team",
                "was_traded", "was_primary_pick", "trade_had_players"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    drafts_csv = SUPABASE_DIR / "drafts.csv"
    roster_csv = ROSTER_DIR / "rosters.csv"

    print(f"Building {DRAFT_YEAR} context features...\n")

    # 1. Roster depth context
    print("1. Roster context (from 2025 rosters)...")
    rc = build_roster_context_2026(roster_csv, drafts_csv)
    append_and_save(SUPABASE_DIR / "draft_pick_context_features.csv",        rc, "draft_season")
    append_and_save(ROSTER_DIR   / "draft_pick_roster_context_features.csv", rc, "draft_season")

    # 2. Positional draft context
    print("2. Positional draft context...")
    pc = build_positional_context_2026(drafts_csv)
    append_and_save(ROSTER_DIR / "draft_positional_context_features.csv", pc, "draft_season")

    # 3. Pick trade flags (all zeros — no 2026 trade data in trades.csv)
    print("3. Pick trade flags (all zeros — no 2026 trade data yet)...")
    tf = build_trade_flags_2026(drafts_csv)
    append_and_save(ROSTER_DIR / "pick_trade_flags.csv", tf, "season")

    print(
        f"\nDone. veteran_performance_features.csv already covers the 2025 season\n"
        f"(joined as draft_season - 1 = {DRAFT_YEAR - 1}), so no update needed there."
    )


if __name__ == "__main__":
    main()
