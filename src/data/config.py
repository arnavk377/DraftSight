import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SCHEMA = "analytics"
CFBD_API_KEY = os.environ.get("CFBD_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DATA_DIR = DATA_DIR / "exports"

NFL_DATA_DIR = RAW_DATA_DIR / "nfl"
AV_DATA_DIR = RAW_DATA_DIR / "av"
CFB_DATA_DIR = RAW_DATA_DIR / "cfb"
COLLEGE_PROCESSED_DIR = PROCESSED_DATA_DIR / "college"
REPORTS_DIR = PROJECT_ROOT / "reports"
