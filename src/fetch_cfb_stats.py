"""
Fetch career college football stats from the College Football Data (CFBD) API
for draft picks that have no entry in college_stats.csv.

Free API key: https://collegefootballdata.com  (register → API key → free tier)

Output: data/supabase_exports/college_stats_scraped.csv
  Same (draft_season, draft_overall) key as college_stats.csv.
  To merge into the main file after reviewing:
      python src/fetch_cfb_stats.py --merge

Raw API responses are cached in data/cfbd_cache/ so re-runs cost no extra
requests. Delete that directory to force a full re-fetch.

Usage:
    export CFBD_API_KEY=your_key
    python src/fetch_cfb_stats.py                     # all missing 2006-2024
    python src/fetch_cfb_stats.py --years 2006 2016   # specific range
    python src/fetch_cfb_stats.py --dry-run           # list targets, no requests
    python src/fetch_cfb_stats.py --merge             # concat scraped → main csv
    python src/fetch_cfb_stats.py --show-categories   # print CFBD stat types
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── API key — set this directly or via environment variable ──────────────────
# Option 1: paste your key here
CFBD_API_KEY: str = "N4vhjsmreG239KWtr54k0eJw/Vj2c9tv312rNh8+jxnz07mX5DQSiT6WvLxr0l+y"
# Option 2: set CFBD_API_KEY in your shell and leave the line above empty

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DRAFTS_CSV  = REPO_ROOT / "data" / "supabase_exports" / "drafts.csv"
MAIN_CS_CSV = REPO_ROOT / "data" / "supabase_exports" / "college_stats.csv"
OUT_CSV     = REPO_ROOT / "data" / "supabase_exports" / "college_stats_scraped.csv"
CACHE_DIR   = REPO_ROOT / "data" / "cfbd_cache"

CFBD_BASE   = "https://api.collegefootballdata.com"
MODEL_YEARS = (2006, 2024)
RATE_DELAY  = 0.35   # seconds between requests (CFBD free tier: comfortable at ~3/s)

# ── CFBD categories → stat types we care about ───────────────────────────────
# Only counting stats are mapped; rate stats (pct, avg) are recomputed at end.

CFBD_CATEGORIES = [
    "passing", "rushing", "receiving",
    "defensive", "interceptions",
    "kicking", "punting",
    "kickReturns", "puntReturns",
]

# (cfbd_category, cfbd_statType) → our column name
COUNT_MAP: dict[tuple[str, str], str] = {
    # Passing
    ("passing", "completions"):         "passing_completions",
    ("passing", "attempts"):            "passing_att",
    ("passing", "yards"):               "passing_yds",
    ("passing", "touchdowns"):          "passing_td",
    ("passing", "interceptions"):       "passing_int",

    # Rushing
    ("rushing", "car"):                 "rushing_car",
    ("rushing", "carries"):             "rushing_car",
    ("rushing", "attempts"):            "rushing_car",
    ("rushing", "yards"):               "rushing_yds",
    ("rushing", "touchdowns"):          "rushing_td",

    # Receiving
    ("receiving", "receptions"):        "receiving_rec",
    ("receiving", "rec"):               "receiving_rec",
    ("receiving", "yards"):             "receiving_yds",
    ("receiving", "touchdowns"):        "receiving_td",

    # Defensive
    ("defensive", "totalTackles"):      "defensive_tot",
    ("defensive", "soloTackles"):       "defensive_solo",
    ("defensive", "sacks"):             "defensive_sacks",
    ("defensive", "tacklesForLoss"):    "defensive_tfl",
    ("defensive", "passesDeflected"):   "defensive_pd",
    ("defensive", "qbHurries"):         "defensive_qb_hur",
    ("defensive", "defensiveTDs"):      "defensive_td",
    ("defensive", "fumbles"):           "fumbles_rec",

    # Interceptions (separate CFBD category)
    ("interceptions", "interceptions"): "interceptions_int",
    ("interceptions", "yards"):         "interceptions_yds",
    ("interceptions", "touchdowns"):    "interceptions_td",

    # Kicking
    ("kicking", "fgAttempts"):          "kicking_fga",
    ("kicking", "fgMade"):              "kicking_fgm",
    ("kicking", "xpAttempts"):          "kicking_xpa",
    ("kicking", "xpMade"):              "kicking_xpm",
    ("kicking", "points"):              "kicking_pts",
    ("kicking", "kickoffs"):            None,   # skip

    # Punting
    ("punting", "punts"):               "punting_no",
    ("punting", "yards"):               "punting_yds",

    # Kick returns
    ("kickReturns", "returns"):         "kickreturns_no",
    ("kickReturns", "yards"):           "kickreturns_yds",

    # Punt returns
    ("puntReturns", "returns"):         "puntreturns_no",
    ("puntReturns", "yards"):           "puntreturns_yds",
}

# Output columns matching CFB_STAT_COLS + join keys
OUTPUT_COLS = [
    "draft_season", "draft_overall",
    "career_years",
    "passing_att", "passing_completions", "passing_int", "passing_td",
    "passing_yds", "passing_pct", "passing_ypa",
    "rushing_car", "rushing_yds", "rushing_td", "rushing_ypc",
    "receiving_rec", "receiving_yds", "receiving_td", "receiving_ypr",
    "defensive_solo", "defensive_tot", "defensive_tfl", "defensive_sacks",
    "defensive_pd", "defensive_qb_hur", "defensive_td",
    "interceptions_int", "interceptions_yds", "interceptions_td",
    "fumbles_rec",
    "kicking_fga", "kicking_fgm", "kicking_xpa", "kicking_xpm", "kicking_pts",
    "kicking_pct",
    "punting_no", "punting_yds", "punting_ypp",
    "kickreturns_no", "kickreturns_yds", "kickreturns_avg",
    "puntreturns_no", "puntreturns_yds", "puntreturns_avg",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove punctuation/suffixes for fuzzy match."""
    name = str(name)
    # Remove accents
    name = "".join(c for c in unicodedata.normalize("NFD", name)
                   if unicodedata.category(c) != "Mn")
    name = name.lower()
    # Remove common suffixes
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name)
    # Remove punctuation except spaces
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())


def _compute_derived(stats: dict) -> dict:
    """Compute rate stats from accumulated counting stats."""
    s = dict(stats)
    def safe_div(num_key, den_key, scale=1.0, digits=2):
        n = s.get(num_key, 0) or 0
        d = s.get(den_key, 0) or 0
        return round(n / d * scale, digits) if d > 0 else None

    s["passing_pct"]      = safe_div("passing_completions", "passing_att", 100, 1)
    s["passing_ypa"]      = safe_div("passing_yds", "passing_att", 1, 2)
    s["rushing_ypc"]      = safe_div("rushing_yds", "rushing_car", 1, 2)
    s["receiving_ypr"]    = safe_div("receiving_yds", "receiving_rec", 1, 2)
    s["punting_ypp"]      = safe_div("punting_yds", "punting_no", 1, 1)
    s["kickreturns_avg"]  = safe_div("kickreturns_yds", "kickreturns_no", 1, 1)
    s["puntreturns_avg"]  = safe_div("puntreturns_yds", "puntreturns_no", 1, 1)
    s["kicking_pct"]      = safe_div("kicking_fgm", "kicking_fga", 100, 1)
    return s


# ── CFBD API client ───────────────────────────────────────────────────────────

class CFBDClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
        })
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, year: int, category: str) -> Path:
        return CACHE_DIR / f"stats_{year}_{category}.json"

    def get_season_stats(self, year: int, category: str) -> list[dict]:
        """Fetch all player stats for a given year and category (cached)."""
        path = self._cache_path(year, category)
        if path.exists():
            with open(path) as f:
                return json.load(f)

        url = f"{CFBD_BASE}/stats/player/season"
        resp = self.session.get(url, params={"year": year, "category": category},
                                timeout=20)
        if resp.status_code == 429:
            print(f"  Rate limited, sleeping 30s...", flush=True)
            time.sleep(30)
            return self.get_season_stats(year, category)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {year}/{category}", file=sys.stderr)
            return []

        data = resp.json()
        with open(path, "w") as f:
            json.dump(data, f)
        time.sleep(RATE_DELAY)
        return data

    def show_categories(self, year: int = 2022):
        """Print all available stat types per category (for debugging)."""
        for cat in CFBD_CATEGORIES:
            data = self.get_season_stats(year, cat)
            types = sorted({r.get("statType", "") for r in data})
            print(f"  {cat:15s} → {types}")


# ── Stats cache: year × category → player_id → {statType: value} ─────────────

def build_stats_lookup(
    client: CFBDClient, years: list[int]
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Returns {year: {normalized_name|team: {col: value}}} for fast lookup.
    Actually returns {year: list_of_records} and we match at query time.
    """
    cache: dict[int, list[dict]] = {}
    total = len(years) * len(CFBD_CATEGORIES)
    done = 0
    for year in years:
        cache[year] = []
        for cat in CFBD_CATEGORIES:
            records = client.get_season_stats(year, cat)
            cache[year].extend(records)
            done += 1
            if done % 20 == 0:
                print(f"  Cached {done}/{total} year-category combos", flush=True)
    return cache


# ── Player matching ───────────────────────────────────────────────────────────

def find_player_stats(
    name: str,
    college: str,
    draft_year: int,
    stats_cache: dict[int, list[dict]],
) -> Optional[dict]:
    """
    Search the stats cache for a player's career stats.
    Looks in college years = [draft_year-5 .. draft_year-1].
    Matches by normalized name, with college as a tiebreaker.
    Returns aggregated counting stats dict, or None if not found.
    """
    norm_name = _normalize_name(name)
    norm_college = _normalize_name(college) if pd.notna(college) else ""

    college_years = [y for y in range(draft_year - 5, draft_year)
                     if y in stats_cache]

    # Accumulate counting stats across all matching seasons
    accumulated: dict[str, float] = {}
    seasons_found = 0
    matched_school: Optional[str] = None

    for year in college_years:
        year_records = stats_cache.get(year, [])

        # Find records matching this player's name
        candidates = [
            r for r in year_records
            if _normalize_name(r.get("player", "")) == norm_name
        ]

        if not candidates:
            continue

        # If multiple players share a name, use college to pick the right one
        if len(candidates) > 1 and norm_college:
            college_match = [
                r for r in candidates
                if norm_college in _normalize_name(r.get("team", ""))
                or _normalize_name(r.get("team", "")) in norm_college
            ]
            if college_match:
                candidates = college_match

        # Use the first match's school to filter consistently across years
        if matched_school is None and candidates:
            matched_school = candidates[0].get("team", "")

        if matched_school:
            candidates = [r for r in candidates
                          if r.get("team", "") == matched_school]

        for record in candidates:
            cat  = record.get("category", "")
            stype = record.get("statType", "")
            col  = COUNT_MAP.get((cat, stype))
            if col is None:
                continue
            try:
                val = float(record.get("stat", 0) or 0)
            except (TypeError, ValueError):
                continue
            accumulated[col] = accumulated.get(col, 0) + val

        if candidates:
            seasons_found += 1

    if not accumulated:
        return None

    result = _compute_derived(accumulated)
    result["career_years"] = seasons_found
    return result


# ── Build target list ─────────────────────────────────────────────────────────

def build_targets(year_start: int, year_end: int) -> pd.DataFrame:
    """Draft picks in [year_start, year_end] that are missing from college_stats.csv."""
    drafts = pd.read_csv(DRAFTS_CSV)
    existing = pd.read_csv(MAIN_CS_CSV)

    picks = drafts[
        drafts["draft_season"].between(year_start, year_end)
    ][["draft_season", "pick", "pfr_player_name", "position_group", "college"]].copy()
    picks = picks.rename(columns={"pick": "draft_overall"})

    already = set(zip(existing["draft_season"].astype(int),
                      existing["draft_overall"].astype(int)))
    picks["already_have"] = picks.apply(
        lambda r: (int(r["draft_season"]), int(r["draft_overall"])) in already, axis=1
    )
    return picks[~picks["already_have"]].reset_index(drop=True)


# ── Merge helper ─────────────────────────────────────────────────────────────

def do_merge():
    if not OUT_CSV.exists():
        print(f"Nothing to merge — {OUT_CSV} does not exist.")
        return
    old = pd.read_csv(MAIN_CS_CSV)
    new = pd.read_csv(OUT_CSV)
    combined = (
        pd.concat([old, new], ignore_index=True)
        .drop_duplicates(subset=["draft_season", "draft_overall"], keep="first")
        .sort_values(["draft_season", "draft_overall"])
        .reset_index(drop=True)
    )
    combined.to_csv(MAIN_CS_CSV, index=False)
    print(f"Merged: {len(old)} + {len(new)} → {len(combined)} rows in {MAIN_CS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch CFBD college stats for draft picks")
    parser.add_argument("--years", nargs=2, type=int, default=list(MODEL_YEARS),
                        metavar=("START", "END"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print targets without making API calls")
    parser.add_argument("--merge", action="store_true",
                        help="merge college_stats_scraped.csv into college_stats.csv")
    parser.add_argument("--show-categories", action="store_true",
                        help="print available CFBD stat types per category")
    args = parser.parse_args()

    if args.merge:
        do_merge()
        return

    # Resolve key: module-level constant wins; fall back to env var
    api_key = CFBD_API_KEY or os.environ.get("CFBD_API_KEY", "")

    if not api_key and not args.dry_run:
        print("ERROR: No API key found.")
        print("  Option 1: set CFBD_API_KEY = 'your_key' at the top of src/fetch_cfb_stats.py")
        print("  Option 2: export CFBD_API_KEY=your_key in your shell")
        print("  Get a free key at: https://collegefootballdata.com")
        sys.exit(1)

    year_start, year_end = args.years
    targets = build_targets(year_start, year_end)

    print(f"Target window:   {year_start}–{year_end}")
    print(f"Missing picks:   {len(targets):,}")

    if args.dry_run:
        pd.set_option("display.max_rows", 50)
        print(targets[["draft_season", "draft_overall", "pfr_player_name",
                        "position_group", "college"]].head(50).to_string(index=False))
        print(f"\n(showing first 50 of {len(targets)} — run without --dry-run to fetch)")
        return

    client = CFBDClient(api_key)

    if args.show_categories:
        print("Available CFBD stat types by category (2022 sample):")
        client.show_categories()
        return

    # Pull all stats for the college years covered by our target drafts
    college_year_min = year_start - 5   # players typically entered college 4-5 years before draft
    college_year_max = year_end - 1
    all_years = list(range(max(college_year_min, 2002), college_year_max + 1))

    print(f"College years:   {all_years[0]}–{all_years[-1]}  ({len(all_years)} years × {len(CFBD_CATEGORIES)} categories)")
    print(f"Cached in:       {CACHE_DIR}")
    print(f"\nPulling stats cache...", flush=True)

    stats_cache = build_stats_lookup(client, all_years)
    print(f"Cache ready. Matching {len(targets):,} draft picks...\n")

    records = []
    n_found = n_miss = 0

    for idx, row in targets.iterrows():
        season  = int(row["draft_season"])
        pick    = int(row["draft_overall"])
        name    = str(row["pfr_player_name"])
        college = str(row.get("college", ""))
        pos     = str(row.get("position_group", ""))

        stats = find_player_stats(name, college, season, stats_cache)

        if stats is None:
            n_miss += 1
            if (idx + 1) % 100 == 0 or n_miss <= 5:
                print(f"  [{idx+1}/{len(targets)}] NOT FOUND  {season} pk{pick:3d} {name} ({pos})")
        else:
            n_found += 1
            record = {"draft_season": season, "draft_overall": pick, **stats}
            records.append(record)
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(targets)}] found={n_found}  miss={n_miss}", flush=True)

    print(f"\nResults:  {n_found} found  |  {n_miss} not found  |  {len(targets)} total")

    if not records:
        print("No stats found — check API key and --show-categories output.")
        return

    out_df = pd.DataFrame(records)
    for col in OUTPUT_COLS:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df[OUTPUT_COLS].sort_values(["draft_season", "draft_overall"])
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved to:  {OUT_CSV}")

    # Print coverage summary
    by_year = out_df.groupby("draft_season").size().reset_index(name="scraped")
    total_by_year = targets.groupby("draft_season").size().reset_index(name="total_missing")
    summary = total_by_year.merge(by_year, on="draft_season", how="left").fillna(0)
    summary["scraped"] = summary["scraped"].astype(int)
    summary["match_rate"] = (summary["scraped"] / summary["total_missing"] * 100).round(1)
    print("\nScrape coverage by year:")
    print(summary.to_string(index=False))

    print(f"\nTo merge into college_stats.csv:")
    print(f"    python src/fetch_cfb_stats.py --merge")


if __name__ == "__main__":
    main()
