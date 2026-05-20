# DraftSight

DraftSight is an NFL draft modeling project that estimates early-career player value from information available around draft time. The current target is 2-year Approximate Value (AV), with models evaluated through walk-forward backtests so each draft class is scored using only prior years of training data.

## Repository Layout

- `data/`: Canonical project data, organized into raw inputs, processed modeling tables, and export-ready CSVs.
- `src/data/`: Data cleaning, trade compression, roster context, and Supabase export scripts.
- `src/model_v1` through `src/model_v4/`: Iterative modeling experiments, with `model_v4` as the current presentation-ready version.
- `reports/`: Generated model outputs, poster figures, diagnostics, and result CSVs.
- `notebooks/`: Exploratory notebooks kept separate from production scripts.

## Current Data Flow

1. Raw NFL, AV, and college files live under `data/raw/`.
2. Cleaned and feature-engineered tables live under `data/processed/`.
3. Supabase-ready tables are generated in `data/exports/supabase/`.
4. Model and poster outputs are generated in `reports/`.

Run the current export pipeline:

```bash
python src/data/build_supabase_exports.py
```

Run the current model iteration:

```bash
python src/model_v4/train_n_evaluate.py
```

Regenerate poster figures from saved model outputs:

```bash
python src/model_v4/make_poster_figures.py
```

## Modeling Notes

The most reliable modeling frame starts from `data/raw/nfl/draft_picks.csv`, joins AV labels from `data/raw/av/`, adds college production from `data/processed/college/`, and adds leakage-safe roster/trade context from `data/processed/features/draft_pick_context_features.csv`.

Use `data/exports/supabase/` when sharing with the app/database side of the project. It contains one clean CSV per major entity: drafts, draft board, AV, college stats, college player seasons, rosters, trades, and draft-pick context features.
