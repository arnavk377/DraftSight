# DraftSight

Pre-draft player value prediction model for the NFL. Combines college performance, combine measurables, recruiting rankings, and draft capital to predict career NFL success.

## Data Pipeline

Collects and aggregates data from:
- **NFL Data** (nfl_data_py): Draft picks, combine results, rosters (2000-2025)
- **College Data** (CFBD API): Player stats, recruiting rankings (2000-2025)

Produces `data/joined/master_player_table.csv` with:
- 9,300+ drafted players
- 47 features (college stats, combine measurements, recruiting, draft info)
- Target variable: career value metric (NFL production)

## Setup

```bash
pip install -r requirements.txt
```

Add `CFBD_API_KEY` to `.env` (get free key at https://api.collegefootballdata.com)

## Usage

Run the data pipeline:
```bash
python -m src.data.run_pipeline
```

See `DATASET_OVERVIEW.md` for schema and data quality details.
