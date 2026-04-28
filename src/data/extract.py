"""Pull tables from the Supabase analytics schema into local parquet files."""

from pathlib import Path

import pandas as pd
from supabase import create_client

from src.data.config import SUPABASE_KEY, SUPABASE_URL, SCHEMA

TABLES = ["draft_features", "trade_assets", "player_week"]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_table(table_name: str, page_size: int = 1000) -> pd.DataFrame:
    """Paginate through a Supabase table and return a DataFrame."""
    client = get_client()
    all_rows = []
    offset = 0
    while True:
        response = (
            client.schema(SCHEMA)
            .table(table_name)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        all_rows.extend(response.data)
        if len(response.data) < page_size:
            break
        offset += page_size
    return pd.DataFrame(all_rows)


def fetch_all(save: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch every table. Optionally persist to data/raw/ as parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        print(f"Fetching {table} ...")
        df = fetch_table(table)
        frames[table] = df
        if save:
            out = RAW_DIR / f"{table}.parquet"
            df.to_parquet(out, index=False)
            print(f"  -> saved {out}  ({len(df)} rows, {df.shape[1]} cols)")
    return frames


def load_raw(table_name: str) -> pd.DataFrame:
    """Load a previously extracted parquet from data/raw/."""
    path = RAW_DIR / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.extract` first."
        )
    return pd.read_parquet(path)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all raw parquets."""
    return {t: load_raw(t) for t in TABLES}


if __name__ == "__main__":
    dataframes = fetch_all(save=True)
    for name, df in dataframes.items():
        print(f"\n--- {name} ({len(df)} rows, {df.shape[1]} cols) ---")
        print(df.head())
