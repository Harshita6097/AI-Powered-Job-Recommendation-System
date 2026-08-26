"""
Feature Engineering — AI-Powered Job Recommendation System
Produces: data/processed/jobs_features.csv  (master feature table)
          data/processed/tfidf_matrix.npz   (TF-IDF sparse matrix)
          data/processed/tfidf_vectorizer.pkl
          data/processed/skill_binarizer.pkl
          data/processed/feature_metadata.json
"""

import pandas as pd
import numpy as np
import re
import json
import os
import pickle
import warnings
from scipy.sparse import save_npz, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING RAW DATA")
print("=" * 60)

postings       = pd.read_csv("linkedin_dataset/postings.csv")
job_skills     = pd.read_csv("linkedin_dataset/jobs/job_skills.csv")
skills_map     = pd.read_csv("linkedin_dataset/mappings/skills.csv")
salaries       = pd.read_csv("linkedin_dataset/jobs/salaries.csv")
job_industries = pd.read_csv("linkedin_dataset/jobs/job_industries.csv")
industries_map = pd.read_csv("linkedin_dataset/mappings/industries.csv")

print(f"postings       : {postings.shape}")
print(f"job_skills     : {job_skills.shape}")
print(f"salaries       : {salaries.shape}")
print(f"job_industries : {job_industries.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — CLEAN POSTINGS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — CLEANING POSTINGS")
print("=" * 60)

df = postings[[
    "job_id", "title", "description", "location",
    "formatted_experience_level", "formatted_work_type", "remote_allowed"
]].copy()

# Drop rows with no description (only 7 rows)
df = df[df["description"].notna()].reset_index(drop=True)

# ── 2a. Clean title ──────────────────────────────────────────
def clean_title(t):
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)           # collapse whitespace
    t = re.sub(r"[^a-zA-Z0-9\s\-/&]", "", t)  # remove special chars
    return t.title()

df["title_clean"] = df["title"].apply(clean_title)

# ── 2b. Extract state from location ─────────────────────────
# Format is mostly "City, ST" or "City, State" or "Region"
def extract_state(loc):
    if pd.isna(loc):
        return "Unknown"
    parts = str(loc).split(",")
    if len(parts) >= 2:
        return parts[-1].strip()
    return parts[0].strip()

df["state"] = df["location"].apply(extract_state)

# ── 2c. Normalise experience level ──────────────────────────
exp_map = {
    "Internship"      : "Internship",
    "Entry level"     : "Entry",
    "Associate"       : "Associate",
    "Mid-Senior level": "Mid-Senior",
    "Director"        : "Director",
    "Executive"       : "Executive",
}
df["experience_level"] = df["formatted_experience_level"].map(exp_map).fillna("Not Specified")

# ── 2d. Normalise work type ──────────────────────────────────
df["work_type"] = df["formatted_work_type"].fillna("Not Specified")

# ── 2e. Remote flag ─────────────────────────────────────────
df["is_remote"] = df["remote_allowed"].fillna(0).astype(int)

print(f"Rows after cleaning : {len(df):,}")
print(f"Experience levels   : {df['experience_level'].value_counts().to_dict()}")
print(f"Work types          : {df['work_type'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — CLEAN & ATTACH SKILLS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — BUILDING SKILL FEATURES")
print("=" * 60)

# Map abbreviations → full names
skills_full = job_skills.merge(skills_map, on="skill_abr")

# Group skills per job as a list
skills_grouped = (
    skills_full
    .groupby("job_id")["skill_name"]
    .apply(list)
    .reset_index()
)
skills_grouped.columns = ["job_id", "skills"]

# Merge onto main df
df = df.merge(skills_grouped, on="job_id", how="left")
df["skills"] = df["skills"].apply(lambda x: x if isinstance(x, list) else [])

# Jobs with no skill tags get "Other" as fallback
df["skills"] = df["skills"].apply(lambda x: x if len(x) > 0 else ["Other"])

print(f"Jobs with skill tags    : {(df['skills'].apply(len) > 1).sum():,}")
print(f"Jobs with fallback only : {(df['skills'].apply(lambda x: x == ['Other'])).sum():,}")
print(f"Unique skill categories : {skills_map.shape[0]}")

# ── Multi-label binarize skills ─────────────────────────────
mlb = MultiLabelBinarizer()
skill_matrix = mlb.fit_transform(df["skills"])
skill_cols   = [f"skill_{s.replace('/', '_').replace(' ', '_')}" for s in mlb.classes_]
skill_df     = pd.DataFrame(skill_matrix, columns=skill_cols, index=df.index)

print(f"Skill binary matrix shape : {skill_matrix.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — ATTACH INDUSTRY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — ATTACHING INDUSTRY")
print("=" * 60)

# A job can have multiple industries — take the first (primary)
industry_primary = (
    job_industries
    .merge(industries_map, on="industry_id")
    .groupby("job_id")["industry_name"]
    .first()
    .reset_index()
)
industry_primary.columns = ["job_id", "industry"]

df = df.merge(industry_primary, on="job_id", how="left")
df["industry"] = df["industry"].fillna("Not Specified")

print(f"Jobs with industry tag : {(df['industry'] != 'Not Specified').sum():,}")
print(f"Top 5 industries       : {df['industry'].value_counts().head(5).to_dict()}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — SALARY FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — SALARY FEATURES")
print("=" * 60)

# Normalise all salaries to yearly
def to_yearly(row):
    period_multipliers = {
        "HOURLY"   : 2080,
        "MONTHLY"  : 12,
        "WEEKLY"   : 52,
        "BIWEEKLY" : 26,
        "YEARLY"   : 1,
    }
    mult = period_multipliers.get(str(row["pay_period"]).upper(), 1)
    mn   = row["min_salary"] * mult if pd.notna(row["min_salary"]) else np.nan
    mx   = row["max_salary"] * mult if pd.notna(row["max_salary"]) else np.nan
    return pd.Series({"salary_min": mn, "salary_max": mx})

sal = salaries.copy()
sal[["salary_min", "salary_max"]] = sal.apply(to_yearly, axis=1)

# Sanity filter — remove outliers
sal = sal[(sal["salary_min"].isna() | (sal["salary_min"].between(10_000, 600_000))) &
          (sal["salary_max"].isna() | (sal["salary_max"].between(10_000, 600_000)))]

# Keep one salary row per job (take max salary_max if duplicates)
sal_agg = (
    sal.groupby("job_id")
    .agg(salary_min=("salary_min", "min"),
         salary_max=("salary_max", "max"))
    .reset_index()
)
sal_agg["salary_mid"] = (sal_agg["salary_min"] + sal_agg["salary_max"]) / 2

df = df.merge(sal_agg[["job_id", "salary_min", "salary_max", "salary_mid"]],
              on="job_id", how="left")

# Fill missing salary with median (for model use — flagged separately)
salary_median = df["salary_mid"].median()
df["salary_mid_filled"] = df["salary_mid"].fillna(salary_median)
df["has_salary"]        = df["salary_mid"].notna().astype(int)

print(f"Jobs with salary data : {df['has_salary'].sum():,}")
print(f"Salary median (yearly): ${salary_median:,.0f}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — CLEAN DESCRIPTION + TF-IDF
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — DESCRIPTION CLEANING + TF-IDF")
print("=" * 60)

STOPWORDS = {
    "will", "work", "experience", "team", "role", "position", "job",
    "company", "including", "required", "ability", "skills", "years",
    "strong", "working", "provide", "ensure", "support", "responsible",
    "must", "also", "new", "using", "within", "across", "help", "us",
    "our", "we", "you", "the", "and", "to", "of", "in", "a", "for",
    "with", "is", "are", "be", "as", "an", "or", "at", "on", "by",
    "this", "that", "have", "has", "from", "not", "all", "their",
    "other", "may", "well", "high", "level", "related", "business",
    "please", "apply", "equal", "opportunity", "employer", "including",
    "qualified", "candidates", "without", "regard", "race", "color",
    "religion", "sex", "national", "origin", "disability", "veteran"
}

def clean_description(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["description_clean"] = df["description"].apply(clean_description)

# TF-IDF on cleaned descriptions
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    stop_words=list(STOPWORDS),
    sublinear_tf=True          # log(1+tf) — reduces impact of very frequent terms
)
tfidf_matrix = tfidf.fit_transform(df["description_clean"])

print(f"TF-IDF matrix shape : {tfidf_matrix.shape}")
print(f"Vocabulary size     : {len(tfidf.vocabulary_):,}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — BUILD COMBINED FEATURE TEXT
#          (used as the primary matching surface)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — COMBINED FEATURE TEXT")
print("=" * 60)

# Concatenate: skills + title + experience + industry + description
# Skills and title are repeated to boost their weight in TF-IDF
def build_feature_text(row):
    skills_str = " ".join(row["skills"]).lower().replace("/", " ").replace("-", " ")
    title_str  = row["title_clean"].lower()
    exp_str    = row["experience_level"].lower().replace("-", " ")
    ind_str    = str(row["industry"]).lower().replace("/", " ").replace(",", " ")
    desc_str   = row["description_clean"][:1000]   # cap at 1000 chars

    # Repeat skills x3 and title x2 to upweight them
    return f"{skills_str} {skills_str} {skills_str} {title_str} {title_str} {exp_str} {ind_str} {desc_str}"

df["feature_text"] = df.apply(build_feature_text, axis=1)

# TF-IDF on combined feature text — this is the PRIMARY vector for cosine similarity
tfidf_combined = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    stop_words=list(STOPWORDS),
    sublinear_tf=True
)
tfidf_combined_matrix = tfidf_combined.fit_transform(df["feature_text"])

print(f"Combined TF-IDF shape    : {tfidf_combined_matrix.shape}")
print(f"Combined vocabulary size : {len(tfidf_combined.vocabulary_):,}")

# ─────────────────────────────────────────────────────────────
# STEP 8 — ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — ENCODING CATEGORICAL FEATURES")
print("=" * 60)

# Experience level ordinal encoding
exp_ordinal = {
    "Internship"   : 0,
    "Entry"        : 1,
    "Associate"    : 2,
    "Mid-Senior"   : 3,
    "Director"     : 4,
    "Executive"    : 5,
    "Not Specified": 2,   # default to middle
}
df["experience_encoded"] = df["experience_level"].map(exp_ordinal)

# Work type one-hot
work_dummies = pd.get_dummies(df["work_type"], prefix="wt").astype(int)

print(f"Experience encoded range : {df['experience_encoded'].min()} – {df['experience_encoded'].max()}")
print(f"Work type dummies        : {list(work_dummies.columns)}")

# ─────────────────────────────────────────────────────────────
# STEP 9 — SCALE NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — SCALING NUMERIC FEATURES")
print("=" * 60)

scaler = MinMaxScaler()
numeric_cols = ["salary_mid_filled", "experience_encoded"]
df[["salary_scaled", "exp_scaled"]] = scaler.fit_transform(df[numeric_cols])

print(f"Salary scaled range     : {df['salary_scaled'].min():.3f} – {df['salary_scaled'].max():.3f}")
print(f"Experience scaled range : {df['exp_scaled'].min():.3f} – {df['exp_scaled'].max():.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 10 — ASSEMBLE MASTER FEATURE TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — ASSEMBLING MASTER FEATURE TABLE")
print("=" * 60)

# Core columns to keep in the CSV
core_cols = [
    "job_id", "title_clean", "experience_level", "work_type",
    "is_remote", "state", "industry", "skills",
    "salary_min", "salary_max", "salary_mid", "salary_mid_filled",
    "has_salary", "salary_scaled", "exp_scaled",
    "experience_encoded", "description_clean", "feature_text"
]

master = pd.concat([
    df[core_cols].reset_index(drop=True),
    skill_df.reset_index(drop=True),
    work_dummies.reset_index(drop=True)
], axis=1)

# Convert skills list to string for CSV storage
master["skills"] = master["skills"].apply(lambda x: "|".join(x))

print(f"Master table shape : {master.shape}")
print(f"Columns            : {len(master.columns)}")
print(f"Sample row:")
print(master[["job_id", "title_clean", "experience_level",
              "industry", "skills", "salary_mid"]].head(3).to_string())

# ─────────────────────────────────────────────────────────────
# STEP 11 — SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — SAVING ARTIFACTS")
print("=" * 60)

# 1. Master feature table
master.to_csv("data/processed/jobs_features.csv", index=False)
print("Saved: data/processed/jobs_features.csv")

# 2. Combined TF-IDF matrix (primary similarity matrix)
save_npz("data/processed/tfidf_matrix.npz", tfidf_combined_matrix)
print("Saved: data/processed/tfidf_matrix.npz")

# 3. TF-IDF vectorizer (needed to transform user input at inference)
with open("data/processed/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_combined, f)
print("Saved: data/processed/tfidf_vectorizer.pkl")

# 4. Skill binarizer (needed to encode user skills at inference)
with open("data/processed/skill_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)
print("Saved: data/processed/skill_binarizer.pkl")

# 5. Metadata — needed by the model and API
metadata = {
    "total_jobs"         : len(master),
    "skill_categories"   : list(mlb.classes_),
    "experience_levels"  : list(exp_ordinal.keys()),
    "work_types"         : list(df["work_type"].unique()),
    "tfidf_vocab_size"   : len(tfidf_combined.vocabulary_),
    "tfidf_features"     : tfidf_combined_matrix.shape[1],
    "skill_binary_cols"  : skill_cols,
    "salary_median"      : round(salary_median, 2),
    "exp_ordinal_map"    : exp_ordinal,
}
with open("data/processed/feature_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved: data/processed/feature_metadata.json")

# ─────────────────────────────────────────────────────────────
# STEP 12 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12 — SUMMARY")
print("=" * 60)

print(f"Total jobs processed         : {len(master):,}")
print(f"Skill binary features        : {len(skill_cols)}")
print(f"TF-IDF features (combined)   : {tfidf_combined_matrix.shape[1]:,}")
print(f"Work type dummy features     : {len(work_dummies.columns)}")
print(f"Total feature columns in CSV : {master.shape[1]}")
print(f"\nFeature engineering complete. Artifacts in data/processed/")
