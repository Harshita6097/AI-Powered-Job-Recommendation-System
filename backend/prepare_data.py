"""
prepare_data.py
Transforms raw LinkedIn job postings into pipeline-ready CSVs.
Run once locally before train.py:
    python prepare_data.py

Output → backend/data/processed/
    jobs.csv         : job_id, job_title, skills, location
    users.csv        : user_id, user_skills, user_location
    interactions.csv : user_id, job_id, interaction_score
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

RAW_DIR       = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load and join raw LinkedIn data ────────────────────────────────────────
print("Loading raw data...")
postings   = pd.read_csv(RAW_DIR / "postings.csv", usecols=["job_id", "title", "location"])
job_skills = pd.read_csv(RAW_DIR / "jobs" / "job_skills.csv")
skills_map = pd.read_csv(RAW_DIR / "mappings" / "skills.csv")

merged = job_skills.merge(skills_map, on="skill_abr")
grouped = (
    merged.groupby("job_id")["skill_name"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"skill_name": "skills"})
)

jobs_raw = postings.merge(grouped, on="job_id").dropna(subset=["title", "location", "skills"])
print(f"  Raw jobs with skills: {len(jobs_raw)}")

# ── 2. Clean and sample jobs ──────────────────────────────────────────────────
jobs_raw["location"] = jobs_raw["location"].str.split(",").str[:2].str.join(",").str.strip()
jobs_raw = jobs_raw[jobs_raw["location"].str.contains(r"[A-Z]{2}$", regex=True, na=False)]
jobs_clean = (
    jobs_raw
    .drop_duplicates(subset=["title", "location"])
    .rename(columns={"title": "job_title"})
    .reset_index(drop=True)
)

N_JOBS     = 500
top_titles = jobs_clean["job_title"].value_counts().head(80).index
jobs_filtered = jobs_clean[jobs_clean["job_title"].isin(top_titles)].copy()

per_title = max(1, N_JOBS // len(top_titles))
sampled_parts = []
for title, grp in jobs_filtered.groupby("job_title"):
    sampled_parts.append(grp.sample(min(len(grp), per_title), random_state=42))

jobs_sampled = pd.concat(sampled_parts).head(N_JOBS).reset_index(drop=True)
jobs_sampled["job_id"] = [f"job_{i+1}" for i in range(len(jobs_sampled))]
jobs_final = jobs_sampled[["job_id", "job_title", "skills", "location"]].copy()
print(f"  Sampled jobs: {len(jobs_final)} across {jobs_final['job_title'].nunique()} unique titles")

# ── 3. Skill categories and domain → job title affinity ──────────────────────
# Each domain maps to job titles it strongly prefers (score 4-5)
# and weakly prefers (score 2-3) — everything else gets score 1-2
# This creates REAL signal for SVD to learn latent factors from

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

# Domain affinity: primary → {other_domain: score_base}
# High score = user in this domain strongly likes jobs in that domain
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

def get_job_domain(job_skills_str):
    """Map a job's skills to its closest domain."""
    job_skill_set = set(s.strip() for s in job_skills_str.split(","))
    best_domain, best_overlap = "Administrative", 0
    for domain, skills in skill_categories.items():
        overlap = len(job_skill_set & set(skills))
        if overlap > best_overlap:
            best_overlap = overlap
            best_domain = domain
    return best_domain

# Pre-compute domain for each job
jobs_final["_domain"] = jobs_final["skills"].apply(get_job_domain)

locations = jobs_final["location"].value_counts().head(20).index.tolist()

# ── 4. Generate 500 users ─────────────────────────────────────────────────────
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
    user_skills = ", ".join(list(dict.fromkeys(p_skills + s_skills)))
    users.append({
        "user_id":       f"user_{i}",
        "user_skills":   user_skills,
        "user_location": random.choice(locations),
    })
    user_domains.append(primary)

users_df = pd.DataFrame(users)

# ── 5. Generate interactions with strong domain signal ────────────────────────
# Key fix: users interact MORE with jobs in their domain and the scores
# are strongly correlated with domain affinity — not random noise
# This gives SVD real latent structure to learn
print("Generating interactions...")

interactions = []
for idx, user in users_df.iterrows():
    primary_domain = user_domains[idx]
    affinity       = domain_affinity.get(primary_domain, {})

    # Split job pool: 60% from preferred domains, 40% random
    preferred_jobs = jobs_final[jobs_final["_domain"].isin(affinity.keys())]
    other_jobs     = jobs_final[~jobs_final["_domain"].isin(affinity.keys())]

    n_total     = random.randint(20, 28)
    n_preferred = int(n_total * 0.65)
    n_other     = n_total - n_preferred

    sampled_preferred = preferred_jobs.sample(
        min(len(preferred_jobs), n_preferred),
        random_state=random.randint(0, 99999)
    )
    sampled_other = other_jobs.sample(
        min(len(other_jobs), n_other),
        random_state=random.randint(0, 99999)
    )
    sampled = pd.concat([sampled_preferred, sampled_other])

    for _, job in sampled.iterrows():
        job_domain = job["_domain"]
        base_score = affinity.get(job_domain, 1)
        # Small noise ±1 with low probability to keep signal strong
        noise = np.random.choice([-1, 0, 0, 0, 1], p=[0.05, 0.35, 0.35, 0.15, 0.10])
        score = int(np.clip(base_score + noise, 1, 5))
        interactions.append({
            "user_id":           user["user_id"],
            "job_id":            job["job_id"],
            "interaction_score": score,
        })

interactions_df = (
    pd.DataFrame(interactions)
    .drop_duplicates(subset=["user_id", "job_id"])
    .reset_index(drop=True)
)

# ── 6. Save ───────────────────────────────────────────────────────────────────
jobs_final[["job_id", "job_title", "skills", "location"]].to_csv(
    PROCESSED_DIR / "jobs.csv", index=False)
users_df.to_csv(PROCESSED_DIR / "users.csv",              index=False)
interactions_df.to_csv(PROCESSED_DIR / "interactions.csv", index=False)

avg_score = interactions_df["interaction_score"].mean()
score_std  = interactions_df["interaction_score"].std()

print("\n" + "="*50)
print("  DATA PREPARATION COMPLETE")
print("="*50)
print(f"  Jobs           : {len(jobs_final)}")
print(f"  Unique titles  : {jobs_final['job_title'].nunique()}")
print(f"  Users          : {len(users_df)}")
print(f"  Interactions   : {len(interactions_df)}")
print(f"  Avg per user   : {len(interactions_df)/len(users_df):.1f}")
print(f"  Avg score      : {avg_score:.2f} ± {score_std:.2f}")
print(f"  Output         : backend/data/processed/")
print("="*50)
