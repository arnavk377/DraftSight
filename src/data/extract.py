"""Pull tables from the Supabase analytics schema into DataFrames."""

import pandas as pd
from supabase import create_client

from src.data.config import SUPABASE_KEY, SUPABASE_URL, SCHEMA

TABLES = ["draft_features", "trade_assets"]


def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_table(table_name: str, page_size: int = 1000) -> pd.DataFrame:
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


def fetch_all() -> dict[str, pd.DataFrame]:
    return {table: fetch_table(table) for table in TABLES}


if __name__ == "__main__":
    dataframes = fetch_all()
    for name, df in dataframes.items():
        print(f"\n--- {name} ({len(df)} rows) ---")
        print(df.head())
