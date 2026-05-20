# Reports

Generated figures, diagnostics, and model result CSVs live here instead of at the repository root.

## Folders

- `model_v3/`: Earlier college-feature model outputs and diagnostics.
- `model_v4/`: Current presentation-ready model outputs, walk-forward summaries, and diagnostic plots.
- `poster_figures/`: Poster-ready figures and matching CSVs.

Model v1 and v2 outputs are generated into `reports/model_v1/` and `reports/model_v2/` if those scripts are rerun.

## Common Commands

Run model v4:

```bash
python src/model_v4/train_n_evaluate.py
```

Regenerate poster figures from saved model v4 outputs:

```bash
python src/model_v4/make_poster_figures.py
```
