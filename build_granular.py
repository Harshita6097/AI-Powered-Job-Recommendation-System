"""
build_granular.py — Stage 1 offline build script.

Runs preprocessing + granular skill extraction on all 123,842 jobs.
Saves: data/processed/jobs_granular.csv

This does NOT replace the existing pipeline.
The existing jobs_features.csv and tfidf artifacts remain untouched.

Usage:
    python build_granular.py
"""

import json
import os
import time
import pandas as pd
import numpy as np

from config import (
    POSTINGS_CSV, JOB_SKILLS_CSV, SKILLS_MAP_CSV,
    SALARIES_CSV, JOB_INDUSTRIES_CSV, INDUSTRIES_MAP_CSV,
    GRANULAR_JOBS_CSV, DATA_PROC, DATA_INDEX,
    SALARY_MIN_VALID, SALARY_MAX_VALID, PAY_PERIOD_MULTIPLIERS,
    EXP_ORDINAL,
)
from src.preprocessing.cleaner import preprocess_postings
from src.preprocessing.skill_extractor import extract_skills_from_job
from backend.logger import get_logger

log = get_logger("build_granular")
os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(DATA_INDEX, exist_ok=True)


def load_raw_data() -> dict:
    log.info("Loading raw data files…")
    return {
        "postings"      : pd.read_csv(POSTINGS_CSV),
        "job_skills"    : pd.read_csv(JOB_SKILLS_CSV),
        "skills_map"    : pd.read_csv(SKILLS_MAP_CSV),
        "salaries"      : pd.read_csv(SALARIES_CSV),
        "job_industries": pd.read_csv(JOB_INDUSTRIES_CSV),
        "industries_map": pd.read_csv(INDUSTRIES_MAP_CSV),
    }


def attach_broad_skills(df: pd.DataFrame, job_skills: pd.DataFrame,
                        skills_map: pd.DataFrame) -> pd.DataFrame:
    """Attach the 35 broad LinkedIn skill categories (kept as metadata)."""
    skills_full = job_skills.merge(skills_map, on="skill_abr")
    grouped = (
        skills_full.groupby("job_id")["skill_name"]
        .apply(list).reset_index()
    )
    grouped.columns = ["job_id", "broad_skills_raw"]
    df = df.merge(grouped, on="job_id", how="left")
    df["broad_skills_raw"] = df["broad_skills_raw"].apply(
        lambda x: x if isinstance(x, list) else ["Other"]
    )
    df["broad_skills_str"] = df["broad_skills_raw"].apply(lambda x: "|".join(x))
    return df


def attach_industry(df: pd.DataFrame, job_industries: pd.DataFrame,
                    industries_map: pd.DataFrame) -> pd.DataFrame:
    primary = (
        job_industries.merge(industries_map, on="industry_id")
        .groupby("job_id")["industry_name"].first().reset_index()
    )
    primary.columns = ["job_id", "industry"]
    df = df.merge(primary, on="job_id", how="left")
    df["industry"] = df["industry"].fillna("Not Specified")
    return df


def attach_salary(df: pd.DataFrame, salaries: pd.DataFrame) -> pd.DataFrame:
    sal = salaries.copy()

    def to_yearly(row):
        mult = PAY_PERIOD_MULTIPLIERS.get(str(row["pay_period"]).upper(), 1)
        mn = float(row["min_salary"]) * mult if pd.notna(row["min_salary"]) else np.nan
        mx = float(row["max_salary"]) * mult if pd.notna(row["max_salary"]) else np.nan
        # Validate
        mn = mn if (pd.notna(mn) and SALARY_MIN_VALID <= mn <= SALARY_MAX_VALID) else np.nan
        mx = mx if (pd.notna(mx) and SALARY_MIN_VALID <= mx <= SALARY_MAX_VALID) else np.nan
        return pd.Series({"salary_min": mn, "salary_max": mx})

    sal[["salary_min", "salary_max"]] = sal.apply(to_yearly, axis=1)
    sal_agg = (
        sal.groupby("job_id")
        .agg(salary_min=("salary_min", "min"), salary_max=("salary_max", "max"))
        .reset_index()
    )
    sal_agg["salary_mid"] = (sal_agg["salary_min"] + sal_agg["salary_max"]) / 2
    df = df.merge(sal_agg, on="job_id", how="left")
    df["has_salary"] = df["salary_mid"].notna().astype(int)
    return df


def extract_granular_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Run granular skill extraction on title + description for every job."""
    log.info("Extracting granular skills from %d jobs…", len(df))
    t0 = time.time()

    results = df.apply(
        lambda row: extract_skills_from_job(
            str(row["title_clean"]),
            str(row["description_clean"])
        ),
        axis=1
    )

    df["granular_skills"]  = results.apply(lambda x: "|".join(x["granular_skills"]))
    df["broad_categories"] = results.apply(lambda x: "|".join(x["broad_categories"]))
    df["n_granular_skills"]= df["granular_skills"].apply(
        lambda x: len(x.split("|")) if x else 0
    )

    elapsed = time.time() - t0
    log.info("Skill extraction complete in %.1fs", elapsed)
    log.info("Avg granular skills per job : %.2f",
             df["n_granular_skills"].mean())
    log.info("Jobs with 0 granular skills : %d",
             (df["n_granular_skills"] == 0).sum())
    log.info("Jobs with 5+ granular skills: %d",
             (df["n_granular_skills"] >= 5).sum())
    return df


def build_feature_text_granular(row) -> str:
    """
    Build feature_text using granular skills instead of broad categories.
    Granular skills repeated 3x for TF-IDF weight boost.
    """
    gran  = row["granular_skills"].replace("|", " ") if row["granular_skills"] else ""
    title = str(row["title_clean"])
    exp   = str(row["experience_level"]).lower().replace("-", " ")
    ind   = str(row["industry"]).lower().replace("/", " ")
    desc  = str(row["description_clean"])[:1000]
    return f"{gran} {gran} {gran} {title} {title} {exp} {ind} {desc}"


def main():
    log.info("=" * 60)
    log.info("BUILD GRANULAR — Stage 1")
    log.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────
    raw = load_raw_data()
    log.info("Postings loaded: %d", len(raw["postings"]))

    # ── Preprocess ────────────────────────────────────────────
    df = preprocess_postings(raw["postings"])

    # ── Attach supporting data ────────────────────────────────
    df = attach_broad_skills(df, raw["job_skills"], raw["skills_map"])
    df = attach_industry(df, raw["job_industries"], raw["industries_map"])
    df = attach_salary(df, raw["salaries"])

    # ── Granular skill extraction ─────────────────────────────
    df = extract_granular_skills(df)

    # ── Experience ordinal ────────────────────────────────────
    df["experience_encoded"] = df["experience_level"].map(EXP_ORDINAL)

    # ── Granular feature text ─────────────────────────────────
    df["feature_text_granular"] = df.apply(build_feature_text_granular, axis=1)

    # ── Select output columns ─────────────────────────────────
    out_cols = [
        "job_id", "title_clean", "experience_level", "experience_encoded",
        "work_type", "is_remote", "city", "state", "country", "location_raw",
        "industry", "broad_skills_str", "broad_categories",
        "granular_skills", "n_granular_skills",
        "salary_min", "salary_max", "salary_mid", "has_salary",
        "description_clean", "feature_text_granular",
    ]
    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    out = df[out_cols].reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────
    out.to_csv(GRANULAR_JOBS_CSV, index=False)
    log.info("Saved: %s  (%d rows × %d cols)", GRANULAR_JOBS_CSV, *out.shape)

    # ── Summary stats ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 1 COMPLETE")
    log.info("=" * 60)
    log.info("Total jobs              : %d", len(out))
    log.info("Jobs with granular skills: %d (%.1f%%)",
             (out["n_granular_skills"] > 0).sum(),
             (out["n_granular_skills"] > 0).mean() * 100)
    log.info("Avg granular skills/job : %.2f", out["n_granular_skills"].mean())
    log.info("Top 20 most common granular skills:")

    from collections import Counter
    all_skills = []
    for s in out["granular_skills"].dropna():
        all_skills.extend([x.strip() for x in s.split("|") if x.strip()])
    top20 = Counter(all_skills).most_common(20)
    for skill, count in top20:
        log.info("  %-35s %d jobs", skill, count)

    # Save summary to JSON for reference
    summary = {
        "total_jobs"              : int(len(out)),
        "jobs_with_granular_skills": int((out["n_granular_skills"] > 0).sum()),
        "avg_granular_skills"     : round(float(out["n_granular_skills"].mean()), 2),
        "top_20_skills"           : [{"skill": s, "count": c} for s, c in top20],
    }
    summary_path = DATA_PROC / "granular_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
