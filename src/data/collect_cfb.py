"""Collect college football data from CFBD API."""

import cfbd
import pandas as pd
from pathlib import Path
import time

from src.data.config import CFB_DATA_DIR, CFBD_API_KEY


def _get_api_client():
    """Initialize CFBD API client."""
    config = cfbd.Configuration()
    config.access_token = CFBD_API_KEY
    return cfbd.ApiClient(config)


def _flatten_stat_dict(stat_dict):
    """Flatten nested stat categories into columns."""
    if stat_dict is None:
        return {}
    flat = {}
    for key, val in stat_dict.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}_{subkey}"] = subval
        else:
            flat[key] = val
    return flat


def collect_player_season_stats(force: bool = False) -> pd.DataFrame:
    """Fetch player season stats from CFBD."""
    output_path = CFB_DATA_DIR / "player_season_stats.csv"
    CFB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        try:
            return pd.read_csv(output_path)
        except pd.errors.EmptyDataError:
            pass

    print("Fetching player season stats (2000-2025)...")
    api_client = _get_api_client()
    api = cfbd.StatsApi(api_client)

    all_rows = []
    for year in range(2000, 2026):
        try:
            players = api.get_player_season_stats(year=year)
            for p in players:
                row = {
                    "year": year,
                    "player_name": p.player_name,
                    "position": p.position,
                    "school": p.school,
                    "conference": p.conference if hasattr(p, "conference") else None,
                }
                if hasattr(p, "passing") and p.passing:
                    row.update(_flatten_stat_dict(p.passing.to_dict()))
                if hasattr(p, "rushing") and p.rushing:
                    row.update(_flatten_stat_dict(p.rushing.to_dict()))
                if hasattr(p, "receiving") and p.receiving:
                    row.update(_flatten_stat_dict(p.receiving.to_dict()))
                if hasattr(p, "defensive") and p.defensive:
                    row.update(_flatten_stat_dict(p.defensive.to_dict()))
                all_rows.append(row)
            print(f"  -> fetched year {year} ({len([p for p in players])} rows)")
        except Exception as e:
            print(f"  ! warning: failed to fetch player stats for {year}: {e}")

        time.sleep(1)

    if not all_rows:
        print("  ! no player season stats fetched")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(output_path, index=False)
    print(f"  -> saved {output_path} ({len(df)} rows, {len(df.columns) if len(df) > 0 else 0} cols)")
    return df


def collect_player_usage(force: bool = False) -> pd.DataFrame:
    """Fetch player usage and PPA from CFBD."""
    output_path = CFB_DATA_DIR / "player_usage.csv"
    CFB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"Player usage already exists at {output_path}, skipping...")
        return pd.read_csv(output_path)

    print("Fetching player usage (2000-2025)...")
    api_client = _get_api_client()
    api = cfbd.PlayersApi(api_client)

    all_rows = []
    for year in range(2000, 2026):
        try:
            players = api.get_player_usage(year=year)
            for p in players:
                # Handle different possible attribute names
                player_name = getattr(p, "player_name", None) or getattr(p, "name", None)
                position = getattr(p, "position", None)
                school = getattr(p, "school", None)

                if not player_name:
                    continue

                row = {
                    "year": year,
                    "player_name": player_name,
                    "position": position,
                    "school": school,
                }
                if hasattr(p, "usage"):
                    row.update(_flatten_stat_dict(p.usage.to_dict() if p.usage else {}))
                if hasattr(p, "ppa"):
                    row.update(_flatten_stat_dict(p.ppa.to_dict() if p.ppa else {}))
                all_rows.append(row)
            print(f"  -> fetched year {year} ({len([p for p in players])} rows)")
        except Exception as e:
            print(f"  ! warning: failed to fetch player usage for {year}: {e}")

        time.sleep(1)

    if not all_rows:
        print("  ! no player usage data fetched, creating empty file")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(output_path, index=False)
    print(f"  -> saved {output_path} ({len(df)} rows, {df.shape[1] if len(df) > 0 else 0} cols)")
    return df


def collect_recruiting(force: bool = False) -> pd.DataFrame:
    """Fetch recruiting data from CFBD."""
    output_path = CFB_DATA_DIR / "recruiting.csv"
    CFB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        try:
            df = pd.read_csv(output_path)
            if len(df) > 0:
                print(f"Recruiting data already exists at {output_path}, skipping...")
                return df
        except pd.errors.EmptyDataError:
            pass

    print("Fetching recruiting data (2000-2025)...")
    api_client = _get_api_client()
    api = cfbd.RecruitingApi(api_client)

    all_rows = []
    for year in range(2000, 2026):
        try:
            players = api.get_recruits(year=year)
            for p in players:
                row = {
                    "recruit_year": year,
                    "player_name": p.recruit_name if hasattr(p, "recruit_name") else p.name,
                    "school": p.school,
                    "committed_to": p.committed_to if hasattr(p, "committed_to") else None,
                    "rating": p.rating if hasattr(p, "rating") else None,
                    "stars": p.stars if hasattr(p, "stars") else None,
                    "position": p.position if hasattr(p, "position") else None,
                    "rank_overall": p.rank_overall if hasattr(p, "rank_overall") else None,
                    "rank_position": p.rank_position if hasattr(p, "rank_position") else None,
                }
                all_rows.append(row)
            print(f"  -> fetched year {year} ({len([p for p in players])} rows)")
        except Exception as e:
            print(f"  ! warning: failed to fetch recruiting for {year}: {e}")

        time.sleep(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"  -> saved {output_path} ({len(df)} rows, {df.shape[1]} cols)")
    return df


def collect_team_talent(force: bool = False) -> pd.DataFrame:
    """Fetch team talent composite from CFBD."""
    output_path = CFB_DATA_DIR / "team_talent.csv"
    CFB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        try:
            return pd.read_csv(output_path)
        except pd.errors.EmptyDataError:
            pass

    print("Fetching team talent (2000-2025)...")
    api_client = _get_api_client()
    api = cfbd.TeamsApi(api_client)

    all_rows = []
    for year in range(2000, 2026):
        try:
            talents = api.get_talent(year=year)
            for t in talents:
                school = getattr(t, "school", None) or getattr(t, "team", None)
                if not school:
                    continue
                row = {
                    "year": year,
                    "school": school,
                    "talent_rating": getattr(t, "talent", None),
                }
                all_rows.append(row)
            print(f"  -> fetched year {year} ({len([t for t in talents])} rows)")
        except Exception as e:
            print(f"  ! warning: failed to fetch team talent for {year}: {e}")

        time.sleep(1)

    if not all_rows:
        print("  ! no team talent data fetched")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(output_path, index=False)
    print(f"  -> saved {output_path} ({len(df)} rows, {len(df.columns) if len(df) > 0 else 0} cols)")
    return df


def collect_sp_ratings(force: bool = False) -> pd.DataFrame:
    """Fetch SP+ ratings from CFBD."""
    output_path = CFB_DATA_DIR / "sp_ratings.csv"
    CFB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        try:
            return pd.read_csv(output_path)
        except pd.errors.EmptyDataError:
            pass

    print("Fetching SP+ ratings (2000-2025)...")
    api_client = _get_api_client()
    api = cfbd.RatingsApi(api_client)

    all_rows = []

    for year in range(2000, 2026):
        try:
            ratings = api.get_sp(year=year)
            for r in ratings:
                row = {
                    "year": year,
                    "school": getattr(r, "team", None),
                    "sp_rating": getattr(r, "rating", None),
                    "conference": getattr(r, "conference", None),
                }
                all_rows.append(row)
            print(f"  -> fetched year {year} ({len([r for r in ratings])} rows)")
        except Exception as e:
            print(f"  ! warning: failed to fetch SP+ ratings for {year}: {e}")

        time.sleep(1)

    if not all_rows:
        print("  ! no SP+ ratings fetched")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(output_path, index=False)
    print(f"  -> saved {output_path} ({len(df)} rows, {len(df.columns) if len(df) > 0 else 0} cols)")
    return df


def collect_all_cfb(force: bool = False) -> dict[str, pd.DataFrame]:
    """Collect all college football datasets."""
    print("\n=== COLLECTING COLLEGE FOOTBALL DATA ===\n")

    season_stats = collect_player_season_stats(force=force)
    usage = collect_player_usage(force=force)
    recruiting = collect_recruiting(force=force)
    talent = collect_team_talent(force=force)
    sp_ratings = collect_sp_ratings(force=force)

    return {
        "season_stats": season_stats,
        "usage": usage,
        "recruiting": recruiting,
        "talent": talent,
        "sp_ratings": sp_ratings,
    }


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    collect_all_cfb(force=force)
