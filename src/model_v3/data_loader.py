"""
Data loading and CFB-to-draft joining for model_v3.

Joins CFB career stats onto draft picks via:
  - player name  (exact or fuzzy)
  - cfb.last_year + 1 == draft.season
  - optionally: cfb.latest_team == draft.college
"""

import glob
import os
import re

import numpy as np
import pandas as pd


# ── Feature column lists ─────────────────────────────────────────────────────

DRAFT_NUM_COLS = ["pick", "round", "age"]
DRAFT_CAT_COLS = ["position", "category", "team", "college", "side"]

CFB_STAT_COLS = [
    "career_years",
    "passing_att", "passing_completions", "passing_int", "passing_td",
    "passing_yds", "passing_pct", "passing_ypa",
    "rushing_car", "rushing_td", "rushing_yds", "rushing_ypc",
    "receiving_rec", "receiving_td", "receiving_yds", "receiving_ypr",
    "defensive_pd", "defensive_qb_hur", "defensive_sacks", "defensive_solo",
    "defensive_td", "defensive_tfl", "defensive_tot",
    "interceptions_int", "interceptions_td", "interceptions_yds",
    "fumbles_fum", "fumbles_lost", "fumbles_rec",
    "kicking_fga", "kicking_fgm", "kicking_pts", "kicking_pct",
    "punting_no", "punting_yds", "punting_ypp",
    "kickreturns_no", "kickreturns_yds", "kickreturns_avg",
    "puntreturns_no", "puntreturns_yds", "puntreturns_avg",
]

# Combined feature lists used by models
NUM_COLS = DRAFT_NUM_COLS + CFB_STAT_COLS
CAT_COLS = DRAFT_CAT_COLS


# ── Normalization helpers ────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    """Lowercase, strip suffixes and punctuation for name matching."""
    name = str(name).lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\.?\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    return " ".join(name.split())


def _norm_college(name: str) -> str:
    """Lowercase, strip generic institutional words for college matching."""
    name = str(name).lower().strip()
    name = re.sub(r"\b(university|college|the|of|at|&|and)\b", "", name)
    name = re.sub(r"[^a-z ]", "", name)
    return " ".join(name.split())


# ── Data loading ─────────────────────────────────────────────────────────────

def load_cfb(cfb_csv: str) -> pd.DataFrame:
    """Load college football career stats and add join keys."""
    df = pd.read_csv(cfb_csv)
    df.columns = [c.strip() for c in df.columns]
    df["draft_season"] = df["last_year"].astype(int) + 1
    df["_name_norm"] = df["player"].apply(_norm_name)
    df["_college_norm"] = df["latest_team"].apply(_norm_college)
    for col in CFB_STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_draft(draft_csv: str) -> pd.DataFrame:
    """Load and clean draft picks, adding join keys."""
    df = pd.read_csv(draft_csv)
    df.columns = [c.strip() for c in df.columns]
    if "draft_season" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"draft_season": "season"})

    needed = ["season", "pick", "round", "team", "position", "category",
              "side", "age", "college", "pfr_player_id", "pfr_player_name"]
    df = df[[c for c in needed if c in df.columns]].copy()

    df["season"] = df["season"].astype(int)
    df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
    df = df.dropna(subset=["pick"]).copy()
    df["pick"] = df["pick"].astype(int)
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    for c in ["team", "position", "category", "side", "college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    df["_name_norm"] = df["pfr_player_name"].apply(_norm_name)
    df["_college_norm"] = df["college"].apply(_norm_college)
    return df


def load_av_from_year_files(av_dir: str) -> pd.DataFrame:
    """Load per-season AV from yearly *_av.csv files."""
    paths = sorted(glob.glob(os.path.join(av_dir, "*_av.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_av.csv files in {av_dir}")
    dfs = []
    for fp in paths:
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Year": "season", "PlayerID": "pfr_player_id",
                                  "Player": "player_name", "AV": "av"})
        df = df[["season", "pfr_player_id", "player_name", "av"]].copy()
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["av"] = pd.to_numeric(df["av"], errors="coerce").fillna(0.0)
        dfs.append(df)
    av = pd.concat(dfs, ignore_index=True)
    return av.groupby(["season", "pfr_player_id"], as_index=False).agg(
        player_name=("player_name", "first"), av=("av", "sum"))


def build_two_year_labels(av_long: pd.DataFrame) -> pd.DataFrame:
    a = av_long.rename(columns={"season": "draft_season", "av": "av_y"})
    b = av_long.rename(columns={"season": "next_season", "av": "av_y1"})[
        ["next_season", "pfr_player_id", "av_y1"]]
    m = a.merge(b, on="pfr_player_id", how="left")
    m = m[m["next_season"] == m["draft_season"] + 1].copy()
    m["av_2yr"] = m["av_y"] + m["av_y1"].fillna(0.0)
    return m[["draft_season", "pfr_player_id", "av_2yr"]]


# ── CFB joining ───────────────────────────────────────────────────────────────

def join_cfb_to_draft(
    draft: pd.DataFrame,
    cfb: pd.DataFrame,
    join_on_college: bool = False,
    use_fuzzy: bool = False,
    fuzzy_threshold: int = 85,
) -> tuple[pd.DataFrame, dict]:
    """Join CFB career stats to draft picks.

    Matches on normalized player name + cfb.draft_season == draft.season.
    Optionally also requires normalized college name to match.
    Supports exact or fuzzy name matching.

    Returns (merged_df, join_stats).
    join_stats contains match counts for analysis.
    """
    stat_cols = [c for c in CFB_STAT_COLS if c in cfb.columns]

    if use_fuzzy:
        merged = _fuzzy_merge(draft, cfb, join_on_college, fuzzy_threshold, stat_cols)
    else:
        merged = _exact_merge(draft, cfb, join_on_college, stat_cols)

    cfb_min_year = int(cfb["draft_season"].min())
    cfb_max_year = int(cfb["draft_season"].max())
    in_window = merged["season"].between(cfb_min_year, cfb_max_year)
    matched = merged[stat_cols[0]].notna() if stat_cols else pd.Series(False, index=merged.index)

    stats = {
        "total_picks": len(draft),
        "cfb_window": (cfb_min_year, cfb_max_year),
        "picks_in_window": int(in_window.sum()),
        "matched_total": int(matched.sum()),
        "matched_in_window": int((in_window & matched).sum()),
        "match_rate_overall": float(matched.mean()),
        "match_rate_in_window": float(matched[in_window].mean()) if in_window.any() else 0.0,
        "join_on_college": join_on_college,
        "use_fuzzy": use_fuzzy,
    }
    return merged, stats


def _exact_merge(
    draft: pd.DataFrame,
    cfb: pd.DataFrame,
    join_on_college: bool,
    stat_cols: list,
) -> pd.DataFrame:
    keep = ["_name_norm", "_college_norm", "draft_season"] + stat_cols
    cfb_sub = cfb[[c for c in keep if c in cfb.columns]].copy()

    join_keys = ["_name_norm", "draft_season"]
    if join_on_college:
        join_keys.append("_college_norm")

    draft_tmp = draft.rename(columns={"season": "draft_season"})
    merged = draft_tmp.merge(cfb_sub, on=join_keys, how="left", suffixes=("", "_cfb"))
    merged = merged.rename(columns={"draft_season": "season"})
    # drop duplicate _college_norm brought in from cfb when join_on_college=False
    merged = merged.drop(columns=["_college_norm_cfb"], errors="ignore")
    return merged


def _fuzzy_merge(
    draft: pd.DataFrame,
    cfb: pd.DataFrame,
    join_on_college: bool,
    threshold: int,
    stat_cols: list,
) -> pd.DataFrame:
    try:
        from rapidfuzz import fuzz, process as rfprocess
        _rapidfuzz = True
    except ImportError:
        from difflib import SequenceMatcher
        _rapidfuzz = False

    draft_tmp = draft.rename(columns={"season": "draft_season"}).reset_index(drop=True)

    # Build per-season lookup list
    cfb_lookup: dict[int, list[dict]] = {}
    for _, row in cfb.iterrows():
        s = int(row["draft_season"])
        entry = {"name": row["_name_norm"], "college": row.get("_college_norm", "")}
        for c in stat_cols:
            entry[c] = row.get(c, np.nan)
        cfb_lookup.setdefault(s, []).append(entry)

    rows = []
    for _, drow in draft_tmp.iterrows():
        season = int(drow["draft_season"])
        name = drow["_name_norm"]
        col_norm = drow.get("_college_norm", "")
        empty = {c: np.nan for c in stat_cols}

        candidates = cfb_lookup.get(season, [])
        if not candidates:
            rows.append(empty)
            continue

        cand_names = [e["name"] for e in candidates]

        # Exact match first
        if name in cand_names:
            idx = cand_names.index(name)
        else:
            # Fuzzy fallback
            if _rapidfuzz:
                result = rfprocess.extractOne(
                    name, cand_names, scorer=fuzz.WRatio, score_cutoff=threshold)
                if result is None:
                    rows.append(empty)
                    continue
                idx = result[2]
            else:
                best_score, best_idx = 0.0, -1
                for i, cand in enumerate(cand_names):
                    score = SequenceMatcher(None, name, cand).ratio() * 100
                    if score > best_score:
                        best_score, best_idx = score, i
                if best_score < threshold or best_idx < 0:
                    rows.append(empty)
                    continue
                idx = best_idx

        entry = candidates[idx]
        if join_on_college and entry["college"] != col_norm:
            rows.append(empty)
            continue
        rows.append({c: entry.get(c, np.nan) for c in stat_cols})

    cfb_df = pd.DataFrame(rows, index=draft_tmp.index)
    merged = pd.concat([draft_tmp, cfb_df], axis=1)
    return merged.rename(columns={"draft_season": "season"})


# ── Join analysis helpers ─────────────────────────────────────────────────────

def compare_join_strategies(
    draft: pd.DataFrame,
    cfb: pd.DataFrame,
    fuzzy_threshold: int = 85,
) -> pd.DataFrame:
    """Run all four join strategies and return a summary DataFrame.

    Strategies:
      A - exact name + season
      B - exact name + season + college
      C - fuzzy name + season
      D - fuzzy name + season + college
    """
    strategies = [
        ("A: exact", False, False),
        ("B: exact + college", True, False),
        ("C: fuzzy", False, True),
        ("D: fuzzy + college", True, True),
    ]
    rows = []
    for label, joc, fuzz in strategies:
        _, stats = join_cfb_to_draft(draft, cfb, join_on_college=joc,
                                      use_fuzzy=fuzz, fuzzy_threshold=fuzzy_threshold)
        rows.append({
            "strategy": label,
            "total_picks": stats["total_picks"],
            "picks_in_window": stats["picks_in_window"],
            "matched": stats["matched_total"],
            "matched_in_window": stats["matched_in_window"],
            "match_rate_overall": f"{stats['match_rate_overall']:.1%}",
            "match_rate_in_window": f"{stats['match_rate_in_window']:.1%}",
        })
    return pd.DataFrame(rows)
