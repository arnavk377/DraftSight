"""Build the final poster modeling frame without training models.

This is a thin, data-first entry point around the shared preparation code in
`train_final_poster_models.py`. Use it when you want to inspect joins, missing
coverage, duplicate keys, feature groups, and label completeness before running
the full model search.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .train_final_poster_models import (
        DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT_DIR,
        build_model_frame,
        load_exports,
        select_feature_columns,
        write_data_artifacts,
    )
except ImportError:  # Allows `python src/final_poster_modeling/prepare_final_poster_data.py`.
    from train_final_poster_models import (
        DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT_DIR,
        build_model_frame,
        load_exports,
        select_feature_columns,
        write_data_artifacts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables = load_exports(args.data_dir)
    frame, quality = build_model_frame(tables, args.horizon)
    feature_cols, num_cols, cat_cols = select_feature_columns(frame)
    write_data_artifacts(tables, frame, quality, feature_cols, num_cols, cat_cols, args.output_dir)

    complete = frame[frame["target_complete"] & frame["av_2yr"].notna()]
    print("Final poster data is ready:")
    print(f"  output: {args.output_dir}")
    print(f"  rows: {len(frame):,}")
    print(f"  complete target rows: {len(complete):,}")
    print(f"  complete years: {complete['draft_season'].min()}-{complete['draft_season'].max()}")
    print(f"  features: {len(feature_cols):,} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"  nonfinite values after cleaning: {quality['nonfinite_numeric_values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
