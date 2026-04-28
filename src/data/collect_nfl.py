"""Collect NFL data: draft picks, combine results, and rosters."""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
import time

from src.data.config import NFL_DATA_DIR


def collect_draft_picks(force: bool = False) -> pd.DataFrame:
    """Fetch draft picks since 2000."""
    output_path = NFL_DATA_DIR / "draft_picks.csv"
    NFL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    df = nfl.import_draft_picks()
    df = df[df["season"] >= 2000].reset_index(drop=True)

    df.to_csv(output_path, index=False)
    return df


def collect_combine(force: bool = False) -> pd.DataFrame:
    """Fetch combine results."""
    output_path = NFL_DATA_DIR / "combine.csv"
    NFL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return pd.read_csv(output_path)

    df = nfl.import_combine_data()

    df.to_csv(output_path, index=False)
    return df


def collect_rosters(force: bool = False) -> pd.DataFrame:
    """Fetch rosters for 2000-2025."""
    output_path = NFL_DATA_DIR / "rosters.csv"
    NFL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"Rosters already exist at {output_path}, skipping...")
        return pd.read_csv(output_path)

    print("Fetching rosters (2000-2025)...")
    years = list(range(2000, 2026))
    dfs = []

    for year in years:
        try:
            df = nfl.import_rosters(years=[year])
            df["season"] = year
            dfs.append(df)
            print(f"  -> fetched year {year} ({len(df)} rows)")
        except AttributeError:
            print(f"  ! warning: nfl_data_py.import_rosters not available, skipping rosters")
            return pd.DataFrame()
        except Exception as e:
            print(f"  ! warning: failed to fetch rosters for {year}: {e}")

    if not dfs:
        print("  ! no rosters fetched, continuing without roster data")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"  -> saved {output_path} ({len(df)} rows, {df.shape[1]} cols)")
    return df


def collect_all_nfl(force: bool = False) -> dict[str, pd.DataFrame]:
    """Collect all NFL datasets."""
    draft = collect_draft_picks(force=force)
    combine = collect_combine(force=force)
    rosters = collect_rosters(force=force)

    return {"draft": draft, "combine": combine, "rosters": rosters}


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    collect_all_nfl(force=force)
