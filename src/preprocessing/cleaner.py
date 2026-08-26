"""
src/preprocessing/cleaner.py

Reusable preprocessing for all job fields.
Keeps original raw values alongside normalised versions.
Does NOT modify the source dataframe in-place — always returns a new df.
"""

import re
import numpy as np
import pandas as pd
from backend.logger import get_logger

log = get_logger("preprocessing")

# ── Stopwords for description cleaning ───────────────────────────────────────
_DESC_STOPWORDS = {
    "will", "work", "experience", "team", "role", "position", "job",
    "company", "including", "required", "ability", "skills", "years",
    "strong", "working", "provide", "ensure", "support", "responsible",
    "must", "also", "new", "using", "within", "across", "help", "us",
    "our", "we", "you", "the", "and", "to", "of", "in", "a", "for",
    "with", "is", "are", "be", "as", "an", "or", "at", "on", "by",
    "this", "that", "have", "has", "from", "not", "all", "their",
    "other", "may", "well", "high", "level", "related", "business",
    "please", "apply", "equal", "opportunity", "employer",
    "qualified", "candidates", "without", "regard", "race", "color",
    "religion", "sex", "national", "origin", "disability", "veteran",
}

# ── Experience level normalisation map ───────────────────────────────────────
_EXP_MAP = {
    "internship"      : "Internship",
    "entry level"     : "Entry",
    "entry"           : "Entry",
    "associate"       : "Associate",
    "mid-senior level": "Mid-Senior",
    "mid-senior"      : "Mid-Senior",
    "mid senior"      : "Mid-Senior",
    "senior"          : "Mid-Senior",
    "director"        : "Director",
    "executive"       : "Executive",
}

# ── Work type normalisation ───────────────────────────────────────────────────
_WORK_TYPE_MAP = {
    "full-time" : "Full-time",
    "fulltime"  : "Full-time",
    "part-time" : "Part-time",
    "parttime"  : "Part-time",
    "contract"  : "Contract",
    "temporary" : "Temporary",
    "temp"      : "Temporary",
    "internship": "Internship",
    "volunteer" : "Volunteer",
    "other"     : "Other",
}

# ── US state abbreviation → full name (for location normalisation) ────────────
_STATE_ABBR = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
}


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────

def normalise_title(title: str) -> str:
    """Lowercase, strip noise, collapse whitespace."""
    t = str(title).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s\-/&]", "", t)
    return t.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Location
# ─────────────────────────────────────────────────────────────────────────────

def normalise_location(loc: str) -> dict:
    """
    Parse location string into {city, state, country, is_remote}.
    Returns dict so callers can use whichever field they need.
    """
    if pd.isna(loc) or str(loc).strip() == "":
        return {"city": None, "state": None, "country": None, "location_raw": None}

    raw = str(loc).strip()
    parts = [p.strip() for p in raw.split(",")]

    city    = parts[0] if len(parts) >= 1 else None
    state   = None
    country = None

    if len(parts) == 2:
        candidate = parts[1].strip()
        # Could be state abbr, full state name, or country
        if candidate.upper() in _STATE_ABBR:
            state   = _STATE_ABBR[candidate.upper()]
            country = "United States"
        elif candidate in _STATE_ABBR.values():
            state   = candidate
            country = "United States"
        else:
            country = candidate

    elif len(parts) >= 3:
        candidate = parts[1].strip()
        if candidate.upper() in _STATE_ABBR:
            state = _STATE_ABBR[candidate.upper()]
        else:
            state = candidate
        country = parts[2].strip()

    return {
        "city"        : city,
        "state"       : state,
        "country"     : country,
        "location_raw": raw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experience level
# ─────────────────────────────────────────────────────────────────────────────

def normalise_experience(exp: str) -> str:
    if pd.isna(exp):
        return "Not Specified"
    key = str(exp).strip().lower()
    return _EXP_MAP.get(key, "Not Specified")


# ─────────────────────────────────────────────────────────────────────────────
# Work type
# ─────────────────────────────────────────────────────────────────────────────

def normalise_work_type(wt: str) -> str:
    if pd.isna(wt):
        return "Not Specified"
    key = str(wt).strip().lower()
    return _WORK_TYPE_MAP.get(key, str(wt).strip())


# ─────────────────────────────────────────────────────────────────────────────
# Salary
# ─────────────────────────────────────────────────────────────────────────────

def normalise_salary(
    min_sal, max_sal, pay_period: str,
    min_valid: float = 10_000,
    max_valid: float = 600_000,
) -> dict:
    """
    Convert salary to yearly and validate range.
    Returns dict with salary_min, salary_max, salary_mid, salary_yearly_valid.
    """
    multipliers = {
        "HOURLY": 2080, "MONTHLY": 12, "WEEKLY": 52,
        "BIWEEKLY": 26, "YEARLY": 1,
    }
    mult = multipliers.get(str(pay_period).upper(), 1)

    s_min = float(min_sal) * mult if pd.notna(min_sal) else np.nan
    s_max = float(max_sal) * mult if pd.notna(max_sal) else np.nan

    # Validate range
    def valid(v):
        return pd.notna(v) and min_valid <= v <= max_valid

    s_min = s_min if valid(s_min) else np.nan
    s_max = s_max if valid(s_max) else np.nan
    s_mid = (s_min + s_max) / 2 if (pd.notna(s_min) and pd.notna(s_max)) else np.nan

    return {
        "salary_min"         : s_min,
        "salary_max"         : s_max,
        "salary_mid"         : s_mid,
        "salary_yearly_valid": pd.notna(s_mid),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Description
# ─────────────────────────────────────────────────────────────────────────────

def clean_description(text: str, max_chars: int = 2000) -> str:
    """Lowercase, remove URLs/HTML/special chars, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)  # keep letters, digits, +, #
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def get_description_stopwords() -> set:
    return _DESC_STOPWORDS.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Full dataframe pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_postings(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalisation steps to a raw postings dataframe.
    Keeps original columns alongside normalised ones.
    Returns a new dataframe — does not modify df_raw.
    """
    log.info("Preprocessing %d postings…", len(df_raw))
    df = df_raw.copy()

    # ── Drop rows with no description ────────────────────────
    before = len(df)
    df = df[df["description"].notna()].reset_index(drop=True)
    log.info("Dropped %d rows with missing description", before - len(df))

    # ── Title ─────────────────────────────────────────────────
    df["title_clean"] = df["title"].apply(normalise_title)

    # ── Location ──────────────────────────────────────────────
    loc_parsed = df["location"].apply(normalise_location).apply(pd.Series)
    df["city"]         = loc_parsed["city"]
    df["state"]        = loc_parsed["state"]
    df["country"]      = loc_parsed["country"]
    df["location_raw"] = loc_parsed["location_raw"]

    # ── Experience ────────────────────────────────────────────
    df["experience_level"] = df["formatted_experience_level"].apply(normalise_experience)

    # ── Work type ─────────────────────────────────────────────
    df["work_type"] = df["formatted_work_type"].apply(normalise_work_type)

    # ── Remote flag ───────────────────────────────────────────
    df["is_remote"] = df["remote_allowed"].fillna(0).astype(int)

    # ── Description ───────────────────────────────────────────
    df["description_clean"] = df["description"].apply(clean_description)

    # ── Data quality report ───────────────────────────────────
    _report_quality(df)

    log.info("Preprocessing complete — %d jobs retained", len(df))
    return df


def _report_quality(df: pd.DataFrame) -> None:
    """Log a data quality summary."""
    log.info("── Data Quality Report ──────────────────────────")
    log.info("Total rows            : %d", len(df))
    log.info("Duplicate job_ids     : %d", df["job_id"].duplicated().sum())
    log.info("Missing title         : %d", df["title"].isna().sum())
    log.info("Missing description   : %d", df["description"].isna().sum())
    log.info("Missing location      : %d", df["location"].isna().sum())
    log.info("Missing experience    : %d", (df["experience_level"] == "Not Specified").sum())
    log.info("Missing work_type     : %d", (df["work_type"] == "Not Specified").sum())
    log.info("Short descriptions (<100 chars): %d",
             (df["description"].str.len() < 100).sum())
    log.info("Experience distribution:\n%s", df["experience_level"].value_counts().to_string())
    log.info("Work type distribution:\n%s", df["work_type"].value_counts().to_string())
    log.info("────────────────────────────────────────────────")
