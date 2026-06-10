# DraftSight

DraftSight predicts early NFL career value from draft-night information, college production, team roster context, and trade context. The target is each player's first two seasons of Pro Football Reference Approximate Value (2-year AV).

The repository includes a fast demonstration notebook for grading and presentation, plus the fuller walk-forward modeling code used during development.

## Quick Start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

View the pre-rendered notebook output without running code:

```bash
open project.html
```

Run the demo notebook interactively:

```bash
jupyter notebook notebooks/demo.ipynb
```

Regenerate the required HTML submission artifact:

```bash
jupyter nbconvert --to html --execute notebooks/demo.ipynb --output project.html --output-dir .
```

The demo notebook is designed to run in under one minute. It trains lightweight Spline Ridge and XGBoost models, evaluates a 2024 holdout, shows examples of good and bad predictions, and visualizes the learned pick-value curve.

## Files

| File | Purpose |
|---|---|
| `README.md` | Project overview, file guide, and run instructions. |
| `requirements.txt` | Python packages needed to run the demo notebook and modeling scripts. |
| `notebooks/demo.ipynb` | Main runnable project notebook. It loads the included data, engineers features, uses college stats through `CollegePerformanceScorer`, trains demo models, evaluates predictions, and creates plots. |
| `project.html` | Pre-executed HTML version of `notebooks/demo.ipynb` with cell outputs already shown. |
| `pick_values.json` | Precomputed pick-value curve used by the notebook so the demo stays fast. |
| `export_pick_values.py` | Optional utility for rebuilding `pick_values.json` from trained model outputs. |
| `src/model/data_loader.py` | Shared data loading and joining utilities for drafts, AV labels, college stats, roster context, trade context, and veteran team context. |
| `src/model/train_n_evaluate.py` | Full walk-forward modeling script. It runs Spline Ridge, XGBoost, CatBoost, Random Forest, FT-Transformer, a stacked ensemble, and a pick-bin baseline. |
| `results/` | Saved model results and poster-ready plots from the larger modeling run. |
| `data/supabase_exports/drafts.csv` | Draft-pick backbone table. Models start from one row per NFL draft pick. |
| `data/supabase_exports/av.csv` | Pro Football Reference Approximate Value data used to build the 2-year AV target. |
| `data/supabase_exports/college_stats.csv` | Career college production features joined by draft season and overall pick. |
| `data/supabase_exports/draft_pick_context_features.csv` | Draft-time team roster context features. |
| `data/roster/pick_trade_flags.csv` | Pick-level trade flags used as simple trade context features. |
| `data/roster/draft_positional_context_features.csv` | Same-draft positional context for each team and pick. |
| `data/roster/veteran_performance_features.csv` | Prior-season veteran context by team and position group. |

## Data Used by the Demo

The notebook only requires the seven CSVs listed above plus `pick_values.json`. Those files are small enough for the course repository requirement when raw/intermediate data dumps are not included.

The feature join is:

```text
drafts.csv
  + 2-year AV labels from av.csv
  + college production from college_stats.csv
  + roster context from draft_pick_context_features.csv
  + trade flags from pick_trade_flags.csv
  + positional context from draft_positional_context_features.csv
  + veteran context from veteran_performance_features.csv
```

College stats are used in the demo through two paths: raw college stat columns and a standardized `college_perf_score` feature computed inside the notebook.

## Full Modeling

Run the larger walk-forward experiment:

```bash
python -m src.model.train_n_evaluate
```

By default, the full script includes college production features. To reproduce the older sensitivity run that excluded college features, use:

```bash
DRAFTSIGHT_EXCLUDE_COLLEGE_STATS=1 python -m src.model.train_n_evaluate
```

The full modeling run can take longer than the submission notebook. For grading, use `notebooks/demo.ipynb` or the already-rendered `project.html`.

## Notes

No API keys are needed to run the demo notebook from the included CSVs. A CollegeFootballData API key is only needed if rebuilding raw college-football data from scratch.
