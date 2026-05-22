# Final Poster Modeling

This directory is reserved for the final capstone/poster modeling run. The
pipeline is intentionally strict: the default run trains all four poster models
instead of letting a one-model dev run look like a final comparison.

Prepare and validate the modeling frame only:

```bash
python src/final_poster_modeling/prepare_final_poster_data.py
```

Equivalent training-script shortcut: `python src/final_poster_modeling/train_final_poster_models.py --prepare-only`.

Run the full poster modeling pipeline:

```bash
python src/final_poster_modeling/train_final_poster_models.py
```

The full run trains XGBoost, CatBoost, MLP Neural Net, and Spline Ridge. It also
includes a pick-bin historical baseline for comparison. The model frame joins
draft info, college aggregate stats, draft-board features, traded-pick context,
and prior-season roster context. It writes tables, fitted models, and
poster-ready figures under `latest/`.

MLP Neural Net and Spline Ridge use a nonnegative `log1p` target transform
because 2-year AV is strongly right-skewed and can contain rare negative values.

By default, hyperparameters are tuned on complete pre-2010 draft classes, then
walk-forward evaluation starts in 2010 and refits each year using only prior
draft classes. Saved model artifacts are refit on all complete labels after the
evaluation tables are written.

Predictions are clipped to the feasible 2-year AV range observed in each
training window. That keeps unstable neural-net/spline extrapolations from
creating impossible negative or thousand-AV predictions in early walk-forward
years.

If CatBoost or XGBoost is missing locally, install project dependencies first:

```bash
pip install -r requirements.txt
```
