# Data Layout

This folder is the canonical home for project data.

## Folders

- `raw/`: Source-like inputs that should be preserved with minimal changes.
- `processed/`: Cleaned, joined, or feature-engineered tables used by scripts and models.
- `exports/`: Shareable output bundles, including Supabase-ready CSVs.

## Important Tables

- `raw/nfl/draft_picks.csv`: Historical NFL draft picks. This is the base table for model backtests.
- `raw/av/*_av.csv`: Yearly Pro Football Reference Approximate Value pulls.
- `raw/cfb/draft_picks_2000_2026.csv`: Draft board/prospect-style college football data, including 2026 prospects.
- `processed/college/05_04_all_players_2004_2024.csv`: College career aggregate table used by model v3/v4.
- `processed/college/05_19_player_stats_aggregated.csv`: Newer college aggregate table used for Supabase exports.
- `processed/college/all_results_wide.csv`: Wide college player-season table.
- `processed/college/all_results_wide_with_draft_overall.csv`: Player-season table with draft-overall context.
- `processed/rosters/`: Cleaned roster data and prior-season roster context features.
- `processed/trades/`: Compressed trade tables and traded-pick features.
- `processed/features/draft_pick_context_features.csv`: Leakage-safe, one-row-per-pick context features.
- `exports/supabase/`: Final CSV bundle for loading into Supabase.

## Modeling Guidance

For historical modeling, start with the draft table, build the 2-year AV target from yearly AV files, join college aggregates, then join `processed/features/draft_pick_context_features.csv` by `draft_season + pick`.

Keep player-season college data as source material for future feature engineering. Aggregate it to one row per player or pick before adding it to a model frame.
