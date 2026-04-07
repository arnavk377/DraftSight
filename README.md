# DraftSight
UCI Data Science Capstone 2026

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file from the template and fill in your Supabase credentials:
   ```bash
   cp .env.example .env
   ```

## Usage

Pull data from the Supabase `analytics` schema:
```bash
python -m src.data.extract
```
