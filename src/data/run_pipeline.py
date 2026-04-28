"""Orchestrator: run the entire pipeline end-to-end."""

import sys
from pathlib import Path

from src.data.collect_nfl import collect_all_nfl
from src.data.collect_cfb import collect_all_cfb
from src.data.join_pipeline import build_master_table


def run_pipeline(force: bool = False):
    """Run the complete data pipeline."""
    print("\n[DraftSight] Starting data pipeline...")

    # Step 1: Collect NFL data
    nfl_data = collect_all_nfl(force=force)

    # Step 2: Collect CFB data
    cfb_data = collect_all_cfb(force=force)

    # Step 3: Build master table
    master = build_master_table()

    print("[DraftSight] Pipeline complete. Output: data/joined/master_player_table.csv\n")

    return master


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_pipeline(force=force)
