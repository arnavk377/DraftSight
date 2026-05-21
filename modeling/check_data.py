import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/andrewhahm/Documents/DraftSight")
DATA_DIR = ROOT / "data" / "supabase_exports"

drafts  = pd.read_csv(DATA_DIR / "drafts.csv")
av      = pd.read_csv(DATA_DIR / "av.csv")
college = pd.read_csv(DATA_DIR / "college_stats.csv")
context = pd.read_csv(DATA_DIR / "draft_pick_context_features.csv")

print("CSVs loaded")

# Build merged frame
frame = drafts.copy()
college_join = college.rename(columns={"draft_overall": "pick"})
frame = frame.merge(college_join, on=["draft_season", "pick"], how="left")
frame = frame.merge(context, on=["draft_season", "pick"], how="left")

print(f"Frame shape: {frame.shape}")

# Check numeric columns
num_df = frame.select_dtypes(include=[np.number])
inf_cols = [c for c in num_df.columns if np.isinf(num_df[c]).any()]
big_cols = [c for c in num_df.columns if (num_df[c].abs() > 1e15).any()]

print(f"\nINF columns ({len(inf_cols)}): {inf_cols}")
print(f"Too-large columns ({len(big_cols)}): {big_cols}")