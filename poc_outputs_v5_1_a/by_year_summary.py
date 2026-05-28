"""
Pivot the walk-forward results into clean by-year tables and save to CSV.
Run from the repo root:
    python poc_outputs_v5_1_a/by_year_summary.py
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "model_v5_1_a_walkforward_results.csv"))

MODELS = ["spline", "xgb", "catboost", "rf", "mlpe", "ftt", "tabnet", "stack"]
METRICS = ["mae", "rmse", "spearman", "r2"]

for metric in METRICS:
    cols = ["test_year"] + [f"{m}_{metric}" for m in MODELS if f"{m}_{metric}" in df.columns]
    pivot = df[cols].copy()
    pivot.columns = ["year"] + [m for m in MODELS if f"{m}_{metric}" in df.columns]
    pivot = pivot.sort_values("year").reset_index(drop=True)

    # Round for readability
    for c in pivot.columns[1:]:
        pivot[c] = pivot[c].round(3)

    out = os.path.join(HERE, f"by_year_{metric}.csv")
    pivot.to_csv(out, index=False)
    print(f"\n=== {metric.upper()} by year ===")
    print(pivot.to_string(index=False))
    print(f"Saved: {out}")
