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
NFL_DATA_DIR = DATA_DIR / "nfl"
CFB_DATA_DIR = DATA_DIR / "cfb"
JOINED_DATA_DIR = DATA_DIR / "joined"
