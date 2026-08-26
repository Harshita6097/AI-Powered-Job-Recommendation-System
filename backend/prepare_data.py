"""
prepare_data.py
Processes raw LinkedIn dataset into pipeline-ready CSVs.

Run once locally:
    python prepare_data.py

Output -> backend/data/processed/
    jobs.csv         : job_id, job_title, skills, description, experience_level, work_type, location
    users.csv        : user_id, user_skills, user_location
    interactions.csv : user_id, job_id, interaction_score
"""
import pandas as pd
import numpy as np
import random
import re
from pathlib import Path

random.seed(42)
np.random.seed(42)

RAW_DIR       = Path(__file__).parent.parent / "linkedin_dataset"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("Loading raw LinkedIn data...")
postings = pd.read_csv(
    RAW_DIR / "postings.csv",
    usecols=["job_id", "title", "description", "location",
             "formatted_experience_level", "formatted_work_type", "normalized_salary"]
)
job_skills_df = pd.read_csv(RAW_DIR / "jobs" / "job_skills.csv")
skills_map    = pd.read_csv(RAW_DIR / "mappings" / "skills.csv")
print(f"  Postings: {len(postings)} | Jobs with skills: {job_skills_df['job_id'].nunique()}")

# ── 2. Map skill abbreviations → full names ───────────────────────────────────
merged  = job_skills_df.merge(skills_map, on="skill_abr")
grouped = (
    merged.groupby("job_id")["skill_name"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"skill_name": "skills"})
)

# ── 3. Join and clean ─────────────────────────────────────────────────────────
jobs_raw = postings.merge(grouped, on="job_id")
jobs_raw = jobs_raw.dropna(subset=["title", "description", "location", "skills"])
jobs_raw["location"] = (
    jobs_raw["location"].str.split(",").str[:2].str.join(",").str.strip()
)
# Keep US locations only
jobs_raw = jobs_raw[jobs_raw["location"].str.contains(r"[A-Z]{2}$", regex=True, na=False)]
jobs_raw = jobs_raw.rename(columns={
    "title": "job_title",
    "formatted_experience_level": "experience_level",
    "formatted_work_type": "work_type"
})
jobs_raw["experience_level"] = jobs_raw["experience_level"].fillna("Not Specified")
jobs_raw["work_type"]        = jobs_raw["work_type"].fillna("Full-time")

# Clean description — strip excessive whitespace
jobs_raw["description"] = jobs_raw["description"].str.replace(r"\s+", " ", regex=True).str.strip()

print(f"  After cleaning: {len(jobs_raw)} jobs")

# ── 4. Sample jobs — balanced across top titles ───────────────────────────────
N_JOBS     = 1000
top_titles = jobs_raw["job_title"].value_counts().head(100).index
jobs_filtered = jobs_raw[jobs_raw["job_title"].isin(top_titles)].copy()

per_title = max(1, N_JOBS // len(top_titles))
sampled_parts = []
for title, grp in jobs_filtered.groupby("job_title"):
    sampled_parts.append(grp.sample(min(len(grp), per_title), random_state=42))

jobs_sampled = pd.concat(sampled_parts).head(N_JOBS).reset_index(drop=True)
jobs_sampled["job_id"] = [f"job_{i+1}" for i in range(len(jobs_sampled))]
jobs_final = jobs_sampled[[
    "job_id", "job_title", "skills", "description", "experience_level", "work_type", "location"
]].copy()

print(f"  Sampled: {len(jobs_final)} jobs across {jobs_final['job_title'].nunique()} unique titles")

# ── 5. Skill categories for user generation ───────────────────────────────────
skill_categories = {
    "Engineering":    ["Engineering", "Manufacturing", "Quality Assurance", "Production"],
    "IT":             ["Information Technology"],
    "Finance":        ["Finance", "Accounting/Auditing"],
    "Marketing":      ["Marketing", "Advertising", "Public Relations", "Writing/Editing"],
    "Sales":          ["Sales", "Business Development", "Customer Service"],
    "Management":     ["Management", "Strategy/Planning", "Project Management", "General Business"],
    "Design":         ["Design", "Art/Creative"],
    "HR":             ["Human Resources", "Training", "Education"],
    "Research":       ["Research", "Science", "Analyst"],
    "Legal":          ["Legal", "Consulting"],
    "Supply Chain":   ["Supply Chain", "Distribution", "Purchasing"],
    "Product":        ["Product Management"],
    "Administrative": ["Administrative"],
}

domain_affinity = {
    "Engineering":    {"Engineering": 5, "IT": 4, "Research": 3, "Management": 2},
    "IT":             {"IT": 5, "Engineering": 4, "Research": 3, "Product": 2},
    "Finance":        {"Finance": 5, "Management": 4, "Research": 3, "Administrative": 2},
    "Marketing":      {"Marketing": 5, "Sales": 4, "Design": 3, "Management": 2},
    "Sales":          {"Sales": 5, "Marketing": 4, "Management": 3, "Administrative": 2},
    "Management":     {"Management": 5, "Product": 4, "Finance": 3, "HR": 2},
    "Design":         {"Design": 5, "Marketing": 4, "Product": 3, "IT": 2},
    "HR":             {"HR": 5, "Management": 4, "Administrative": 3, "Legal": 2},
    "Research":       {"Research": 5, "Engineering": 4, "IT": 3, "Finance": 2},
    "Legal":          {"Legal": 5, "Management": 4, "Finance": 3, "HR": 2},
    "Supply Chain":   {"Supply Chain": 5, "Engineering": 4, "Management": 3, "Administrative": 2},
    "Product":        {"Product": 5, "Management": 4, "IT": 3, "Design": 2},
    "Administrative": {"Administrative": 5, "HR": 4, "Management": 3, "Finance": 2},
}

def get_job_domain(skills_str):
    job_skill_set = set(s.strip() for s in skills_str.split(","))
    best, best_n = "Administrative", 0
    for domain, skills in skill_categories.items():
        n = len(job_skill_set & set(skills))
        if n > best_n:
            best_n, best = n, domain
    return best

jobs_final["_domain"] = jobs_final["skills"].apply(get_job_domain)
locations = jobs_final["location"].value_counts().head(20).index.tolist()

# ── 6. Generate 500 users ─────────────────────────────────────────────────────
print("Generating users...")
users, user_domains = [], []
cat_names = list(skill_categories.keys())

for i in range(1, 501):
    primary   = random.choice(cat_names)
    secondary = random.choice([c for c in cat_names if c != primary])
    p_skills  = random.sample(skill_categories[primary],
                              min(len(skill_categories[primary]), random.randint(2, 3)))
    s_skills  = random.sample(skill_categories[secondary],
                              min(len(skill_categories[secondary]), random.randint(1, 2)))
    users.append({
        "user_id":       f"user_{i}",
        "user_skills":   ", ".join(list(dict.fromkeys(p_skills + s_skills))),
        "user_location": random.choice(locations),
    })
    user_domains.append(primary)

users_df = pd.DataFrame(users)

# ── 7. Generate interactions with strong domain signal ────────────────────────
print("Generating interactions...")
interactions = []
for idx, user in users_df.iterrows():
    primary  = user_domains[idx]
    affinity = domain_affinity.get(primary, {})

    preferred = jobs_final[jobs_final["_domain"].isin(affinity.keys())]
    other     = jobs_final[~jobs_final["_domain"].isin(affinity.keys())]

    n_total = random.randint(20, 30)
    n_pref  = int(n_total * 0.65)
    n_other = n_total - n_pref

    sampled = pd.concat([
        preferred.sample(min(len(preferred), n_pref),  random_state=random.randint(0, 99999)),
        other.sample(    min(len(other),     n_other), random_state=random.randint(0, 99999)),
    ])

    for _, job in sampled.iterrows():
        base  = affinity.get(job["_domain"], 1)
        noise = np.random.choice([-1, 0, 0, 0, 1], p=[0.05, 0.35, 0.35, 0.15, 0.10])
        interactions.append({
            "user_id":           user["user_id"],
            "job_id":            job["job_id"],
            "interaction_score": int(np.clip(base + noise, 1, 5)),
        })

interactions_df = (
    pd.DataFrame(interactions)
    .drop_duplicates(subset=["user_id", "job_id"])
    .reset_index(drop=True)
)

# ── 8. Save ───────────────────────────────────────────────────────────────────
jobs_final.drop(columns=["_domain"]).to_csv(PROCESSED_DIR / "jobs.csv",         index=False)
users_df.to_csv(                            PROCESSED_DIR / "users.csv",         index=False)
interactions_df.to_csv(                     PROCESSED_DIR / "interactions.csv",  index=False)

print("\n" + "="*52)
print("  DATA PREPARATION COMPLETE")
print("="*52)
print(f"  Jobs           : {len(jobs_final)}")
print(f"  Unique titles  : {jobs_final['job_title'].nunique()}")
print(f"  Users          : {len(users_df)}")
print(f"  Interactions   : {len(interactions_df)}")
print(f"  Avg per user   : {len(interactions_df)/len(users_df):.1f}")
print(f"  Avg score      : {interactions_df['interaction_score'].mean():.2f} "
      f"± {interactions_df['interaction_score'].std():.2f}")
print(f"  Output         : backend/data/processed/")
print("="*52)
