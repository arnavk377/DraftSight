"""Fetch birth dates / ages for 2026 draft prospects and backfill drafts.csv.

Strategy (in order):
  1. nfl_data_py  — import_draft_picks(2026) already has `age` for ~215/257 picks
  2. ESPN college football athlete API — for remaining gaps, uses cfb_player_id
     from draft_board.csv (numeric ESPN IDs)

Age is computed as of the 2026 NFL Draft (April 23, 2026) for the ESPN path.
For the nfl_data_py path the age is taken as-is (PFR reports age at draft time).

Run from repo root:
    python -m src.data.fetch_2026_ages
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests

REPO_ROOT    = Path(__file__).resolve().parents[2]
SUPABASE_DIR = REPO_ROOT / "data" / "supabase_exports"

DRAFT_YEAR = 2026
DRAFT_DATE = date(2026, 4, 23)
ESPN_URL   = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/athletes/{cfb_id}"
RATE_SLEEP = 0.25


def age_on_draft_day(birth_str: str) -> float | None:
    try:
        born = date.fromisoformat(birth_str[:10])
        return round((DRAFT_DATE - born).days / 365.25, 1)
    except Exception:
        return None


def fetch_birth_date_espn(cfb_id: int, session: requests.Session) -> str | None:
    url = ESPN_URL.format(cfb_id=int(cfb_id))
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        dob = data.get("dateOfBirth") or data.get("born", {}).get("date")
        return dob[:10] if dob else None
    except Exception as exc:
        print(f"    WARNING: could not fetch cfb_id={cfb_id}: {exc}")
        return None


def main() -> None:
    # ── Step 1: nfl_data_py ages ──────────────────────────────────────────────
    print(f"Loading {DRAFT_YEAR} draft picks from nfl_data_py...")
    nfl_draft = nfl.import_draft_picks([DRAFT_YEAR])[["pick", "pfr_player_name", "age"]]
    covered = nfl_draft["age"].notna().sum()
    print(f"  nfl_data_py age coverage: {covered}/{len(nfl_draft)} picks\n")

    # ── Step 2: ESPN fallback for missing picks ───────────────────────────────
    missing_picks = set(nfl_draft.loc[nfl_draft["age"].isna(), "pick"].tolist())

    if missing_picks:
        db = pd.read_csv(SUPABASE_DIR / "draft_board.csv")
        db["draft_season_yr"] = db["draft_board_id"].str[:4].astype(int)
        db26 = (
            db[db["draft_season_yr"] == DRAFT_YEAR][["draft_overall", "cfb_player_id", "player"]]
            .rename(columns={"draft_overall": "pick"})
            .copy()
        )
        db26["cfb_player_id"] = pd.to_numeric(db26["cfb_player_id"], errors="coerce")
        db26 = db26[db26["pick"].isin(missing_picks) & db26["cfb_player_id"].notna()]

        print(f"Fetching ESPN ages for {len(db26)} remaining picks...")
        session = requests.Session()
        session.headers["User-Agent"] = "DraftSight research project"

        espn_ages: dict[int, float] = {}
        for i, (_, row) in enumerate(db26.iterrows(), 1):
            cfb_id = int(row["cfb_player_id"])
            pick   = int(row["pick"])
            dob    = fetch_birth_date_espn(cfb_id, session)
            age    = age_on_draft_day(dob) if dob else None
            if age:
                espn_ages[pick] = age
            status = f"{age:.1f} yrs (born {dob})" if age else "no birth date"
            print(f"  [{i:3d}/{len(db26)}] pick {pick:3d}  {row['player']:<22s}  {status}")
            time.sleep(RATE_SLEEP)

        # Patch nfl_draft with ESPN results
        for pick, age in espn_ages.items():
            nfl_draft.loc[nfl_draft["pick"] == pick, "age"] = age

        espn_filled = len(espn_ages)
        print(f"\nESPN filled {espn_filled} additional picks")

    final_covered = nfl_draft["age"].notna().sum()
    print(f"Total age coverage: {final_covered}/{len(nfl_draft)} picks\n")

    # ── Step 3: Backfill drafts.csv ───────────────────────────────────────────
    drafts = pd.read_csv(SUPABASE_DIR / "drafts.csv")
    mask   = drafts["draft_season"] == DRAFT_YEAR

    age_map = nfl_draft.set_index("pick")["age"].to_dict()
    drafts.loc[mask, "age"] = drafts.loc[mask, "pick"].map(age_map)

    age_filled = drafts.loc[mask, "age"].notna().sum()
    print(f"Age filled in drafts.csv for {age_filled}/{mask.sum()} 2026 picks")
    drafts.to_csv(SUPABASE_DIR / "drafts.csv", index=False)
    print(f"Saved → {SUPABASE_DIR / 'drafts.csv'}")


if __name__ == "__main__":
    main()
