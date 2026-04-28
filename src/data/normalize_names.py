"""Utilities for name normalization and fuzzy matching."""

import re
from thefuzz import fuzz


def normalize_name(name: str) -> str:
    """Normalize a player name for matching.

    - lowercase
    - remove suffixes (Jr., Sr., III, II, etc.)
    - remove periods
    - strip extra whitespace
    """
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()

    # Remove common suffixes
    suffixes = [r"\s+(jr\.?|sr\.?|iii|ii|iv|v)$", r"\s+(jr|sr|iii|ii|iv|v)$"]
    for suffix in suffixes:
        name = re.sub(suffix, "", name)

    # Remove periods
    name = name.replace(".", "")

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name)

    return name


def fuzzy_match_name(name1: str, name2: str, threshold: int = 85) -> bool:
    """Fuzzy match two names using partial_ratio.

    Returns True if match score >= threshold.
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    if not norm1 or not norm2:
        return False

    score = fuzz.partial_ratio(norm1, norm2)
    return score >= threshold


def match_player_to_cfb(
    pfr_name: str,
    pfr_school: str,
    cfb_players_df,
    name_col: str = "player_name",
    school_col: str = "school",
) -> tuple:
    """Find a matching CFB player for a PFR player.

    Returns (matched_row, match_type) where match_type in ['exact', 'fuzzy', None]
    """
    # Normalize inputs
    norm_pfr_name = normalize_name(pfr_name)
    norm_pfr_school = normalize_name(pfr_school) if pfr_school else ""

    # Exact match on (name, school)
    for _, row in cfb_players_df.iterrows():
        cfb_name = normalize_name(row[name_col])
        cfb_school = normalize_name(row[school_col]) if row[school_col] else ""

        if norm_pfr_name == cfb_name and norm_pfr_school == cfb_school:
            return row, "exact"

    # Fuzzy match on (name, school) with high threshold
    best_match = None
    best_score = 0

    for _, row in cfb_players_df.iterrows():
        cfb_name = normalize_name(row[name_col])
        cfb_school = normalize_name(row[school_col]) if row[school_col] else ""

        # Weight school match heavily
        name_score = fuzz.partial_ratio(norm_pfr_name, cfb_name)
        school_match = norm_pfr_school == cfb_school

        if school_match:
            combined_score = name_score
        else:
            combined_score = name_score * 0.7  # Penalize school mismatch

        if combined_score >= 85 and combined_score > best_score:
            best_match = row
            best_score = combined_score

    if best_match is not None:
        return best_match, "fuzzy"

    return None, None
