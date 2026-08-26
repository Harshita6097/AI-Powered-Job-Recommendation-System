"""
EDA — AI-Powered Job Recommendation System
LinkedIn Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("eda_outputs", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})

# ─────────────────────────────────────────────
# 1. LOAD ALL FILES
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING DATA")
print("=" * 60)

postings       = pd.read_csv("linkedin_dataset/postings.csv")
job_skills     = pd.read_csv("linkedin_dataset/jobs/job_skills.csv")
skills_map     = pd.read_csv("linkedin_dataset/mappings/skills.csv")
salaries       = pd.read_csv("linkedin_dataset/jobs/salaries.csv")
job_industries = pd.read_csv("linkedin_dataset/jobs/job_industries.csv")
industries_map = pd.read_csv("linkedin_dataset/mappings/industries.csv")
companies      = pd.read_csv("linkedin_dataset/companies/companies.csv")

print(f"postings       : {postings.shape}")
print(f"job_skills     : {job_skills.shape}")
print(f"skills_map     : {skills_map.shape}")
print(f"salaries       : {salaries.shape}")
print(f"job_industries : {job_industries.shape}")
print(f"industries_map : {industries_map.shape}")
print(f"companies      : {companies.shape}")

# ─────────────────────────────────────────────
# 2. BASIC OVERVIEW
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — BASIC OVERVIEW")
print("=" * 60)

print("\n--- postings dtypes ---")
print(postings.dtypes.to_string())
print("\n--- postings describe (numeric) ---")
print(postings.describe().to_string())

# ─────────────────────────────────────────────
# 3. MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — MISSING VALUE ANALYSIS")
print("=" * 60)

null_pct = (postings.isnull().sum() / len(postings) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
print(null_pct.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
null_pct.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Missing Value % per Column — postings.csv", fontsize=14, fontweight="bold")
ax.set_ylabel("Missing %")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_outputs/01_missing_values.png")
plt.close()
print("Saved: eda_outputs/01_missing_values.png")

# ─────────────────────────────────────────────
# 4. JOB TITLE DISTRIBUTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — TOP JOB TITLES")
print("=" * 60)

top_titles = postings["title"].value_counts().head(20)
print(top_titles.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
top_titles.sort_values().plot(kind="barh", ax=ax, color="teal", edgecolor="white")
ax.set_title("Top 20 Job Titles by Posting Count", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Postings")
plt.tight_layout()
plt.savefig("eda_outputs/02_top_job_titles.png")
plt.close()
print("Saved: eda_outputs/02_top_job_titles.png")

# ─────────────────────────────────────────────
# 5. EXPERIENCE LEVEL DISTRIBUTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — EXPERIENCE LEVEL DISTRIBUTION")
print("=" * 60)

exp = postings["formatted_experience_level"].value_counts()
print(exp.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
exp.plot(kind="bar", ax=axes[0], color="coral", edgecolor="white")
axes[0].set_title("Experience Level — Count", fontsize=13, fontweight="bold")
axes[0].tick_params(axis="x", rotation=30)
axes[1].pie(exp, labels=exp.index, autopct="%1.1f%%", startangle=140,
            colors=sns.color_palette("pastel"))
axes[1].set_title("Experience Level — Share", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_outputs/03_experience_level.png")
plt.close()
print("Saved: eda_outputs/03_experience_level.png")

# ─────────────────────────────────────────────
# 6. WORK TYPE DISTRIBUTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — WORK TYPE DISTRIBUTION")
print("=" * 60)

wt = postings["formatted_work_type"].value_counts()
print(wt.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
wt.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="white")
ax.set_title("Work Type Distribution", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("eda_outputs/04_work_type.png")
plt.close()
print("Saved: eda_outputs/04_work_type.png")

# ─────────────────────────────────────────────
# 7. SKILLS DISTRIBUTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — SKILLS DISTRIBUTION")
print("=" * 60)

skills_full = job_skills.merge(skills_map, on="skill_abr")
top_skills  = skills_full["skill_name"].value_counts().head(20)
print(top_skills.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
top_skills.sort_values().plot(kind="barh", ax=ax, color="darkorange", edgecolor="white")
ax.set_title("Top 20 Skills Across All Job Postings", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Job Postings Requiring This Skill")
plt.tight_layout()
plt.savefig("eda_outputs/05_top_skills.png")
plt.close()
print("Saved: eda_outputs/05_top_skills.png")

skills_per_job = job_skills.groupby("job_id")["skill_abr"].count()
print(f"\nAvg skills per job : {skills_per_job.mean():.2f}")
print(f"Max skills per job : {skills_per_job.max()}")
print(f"Min skills per job : {skills_per_job.min()}")

fig, ax = plt.subplots(figsize=(10, 5))
skills_per_job.value_counts().sort_index().plot(kind="bar", ax=ax,
                                                 color="steelblue", edgecolor="white")
ax.set_title("Distribution of Number of Skills per Job Posting", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Skills Tagged")
ax.set_ylabel("Number of Jobs")
plt.tight_layout()
plt.savefig("eda_outputs/06_skills_per_job.png")
plt.close()
print("Saved: eda_outputs/06_skills_per_job.png")

# ─────────────────────────────────────────────
# 8. INDUSTRY DISTRIBUTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — INDUSTRY DISTRIBUTION")
print("=" * 60)

industries_full = job_industries.merge(industries_map, on="industry_id")
top_industries  = industries_full["industry_name"].value_counts().head(20)
print(top_industries.to_string())

fig, ax = plt.subplots(figsize=(12, 8))
top_industries.sort_values().plot(kind="barh", ax=ax, color="seagreen", edgecolor="white")
ax.set_title("Top 20 Industries by Job Posting Count", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Job Postings")
plt.tight_layout()
plt.savefig("eda_outputs/07_top_industries.png")
plt.close()
print("Saved: eda_outputs/07_top_industries.png")

# ─────────────────────────────────────────────
# 9. SALARY ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — SALARY ANALYSIS")
print("=" * 60)

sal = salaries.copy()
sal["yearly_min"] = sal.apply(
    lambda r: r["min_salary"] * 2080 if r["pay_period"] == "HOURLY" else r["min_salary"], axis=1)
sal["yearly_max"] = sal.apply(
    lambda r: r["max_salary"] * 2080 if r["pay_period"] == "HOURLY" else r["max_salary"], axis=1)

yearly = sal[(sal["yearly_min"] > 10000) & (sal["yearly_max"] < 500000)].dropna(
    subset=["yearly_min", "yearly_max"])

print(f"Salary records after cleaning : {len(yearly)}")
print(yearly[["yearly_min", "yearly_max"]].describe().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(yearly["yearly_min"].dropna(), bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_title("Distribution of Min Yearly Salary", fontsize=13, fontweight="bold")
axes[0].set_xlabel("USD / Year")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].hist(yearly["yearly_max"].dropna(), bins=50, color="coral", edgecolor="white", alpha=0.8)
axes[1].set_title("Distribution of Max Yearly Salary", fontsize=13, fontweight="bold")
axes[1].set_xlabel("USD / Year")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig("eda_outputs/08_salary_distribution.png")
plt.close()
print("Saved: eda_outputs/08_salary_distribution.png")

# ─────────────────────────────────────────────
# 10. SALARY BY EXPERIENCE LEVEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — SALARY BY EXPERIENCE LEVEL")
print("=" * 60)

exp_order = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
sal_exp = sal.merge(postings[["job_id", "formatted_experience_level"]], on="job_id")
sal_exp = sal_exp[
    (sal_exp["yearly_max"] < 500000) &
    (sal_exp["yearly_max"] > 10000) &
    sal_exp["formatted_experience_level"].notna()
]

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=sal_exp, x="formatted_experience_level", y="yearly_max",
            order=exp_order, palette="Set2", ax=ax)
ax.set_title("Max Salary Distribution by Experience Level", fontsize=14, fontweight="bold")
ax.set_xlabel("Experience Level")
ax.set_ylabel("Max Yearly Salary (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("eda_outputs/09_salary_by_experience.png")
plt.close()
print("Saved: eda_outputs/09_salary_by_experience.png")

# ─────────────────────────────────────────────
# 11. SKILLS HEATMAP — TOP SKILLS vs EXPERIENCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11 — SKILLS vs EXPERIENCE HEATMAP")
print("=" * 60)

top15_skills = skills_full["skill_name"].value_counts().head(15).index.tolist()
jobs_with_exp = postings[["job_id", "formatted_experience_level"]].dropna()
skill_exp = skills_full.merge(jobs_with_exp, on="job_id")
skill_exp = skill_exp[skill_exp["skill_name"].isin(top15_skills)]
skill_exp = skill_exp[skill_exp["formatted_experience_level"].isin(exp_order)]

heatmap_data = skill_exp.groupby(
    ["formatted_experience_level", "skill_name"]
).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(exp_order).fillna(0)

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
ax.set_title("Skill Demand by Experience Level (Top 15 Skills)", fontsize=14, fontweight="bold")
ax.set_xlabel("Skill")
ax.set_ylabel("Experience Level")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("eda_outputs/10_skills_experience_heatmap.png")
plt.close()
print("Saved: eda_outputs/10_skills_experience_heatmap.png")

# ─────────────────────────────────────────────
# 12. REMOTE vs ON-SITE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12 — REMOTE vs ON-SITE")
print("=" * 60)

remote = postings["remote_allowed"].fillna(0).astype(int).value_counts()
remote.index = ["On-site / Not Specified", "Remote Allowed"]
print(remote.to_string())

fig, ax = plt.subplots(figsize=(7, 5))
remote.plot(kind="bar", ax=ax, color=["steelblue", "seagreen"], edgecolor="white")
ax.set_title("Remote vs On-site Job Postings", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/11_remote_vs_onsite.png")
plt.close()
print("Saved: eda_outputs/11_remote_vs_onsite.png")

# ─────────────────────────────────────────────
# 13. WORD CLOUD — JOB DESCRIPTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13 — WORD CLOUD FROM JOB DESCRIPTIONS")
print("=" * 60)

desc_text = " ".join(postings["description"].dropna().sample(
    min(5000, len(postings)), random_state=42).tolist())

stopwords_extra = {
    "will", "work", "experience", "team", "role", "position", "job",
    "company", "including", "required", "ability", "skills", "years",
    "strong", "working", "provide", "ensure", "support", "responsible",
    "must", "also", "new", "using", "within", "across", "help", "us",
    "our", "we", "you", "the", "and", "to", "of", "in", "a", "for",
    "with", "is", "are", "be", "as", "an", "or", "at", "on", "by",
    "this", "that", "have", "has", "from", "not", "all", "their",
    "other", "may", "well", "high", "level", "related", "business"
}

wc = WordCloud(
    width=1400, height=700, background_color="white",
    colormap="viridis", stopwords=stopwords_extra,
    max_words=150, collocations=False
).generate(desc_text)

fig, ax = plt.subplots(figsize=(16, 7))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Most Common Words in Job Descriptions", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_outputs/12_wordcloud_descriptions.png")
plt.close()
print("Saved: eda_outputs/12_wordcloud_descriptions.png")

# ─────────────────────────────────────────────
# 14. TOP HIRING COMPANIES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 14 — TOP HIRING COMPANIES")
print("=" * 60)

top_companies = postings["company_name"].value_counts().head(20)
print(top_companies.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
top_companies.sort_values().plot(kind="barh", ax=ax, color="slateblue", edgecolor="white")
ax.set_title("Top 20 Companies by Job Postings", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Postings")
plt.tight_layout()
plt.savefig("eda_outputs/13_top_companies.png")
plt.close()
print("Saved: eda_outputs/13_top_companies.png")

# ─────────────────────────────────────────────
# 15. SKILL CO-OCCURRENCE MATRIX
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 15 — SKILL CO-OCCURRENCE MATRIX (Top 12)")
print("=" * 60)

top12 = skills_full["skill_name"].value_counts().head(12).index.tolist()
skill_pivot = skills_full[skills_full["skill_name"].isin(top12)].copy()
skill_pivot["value"] = 1
skill_matrix = skill_pivot.pivot_table(
    index="job_id", columns="skill_name", values="value", fill_value=0)
cooccurrence = skill_matrix.T.dot(skill_matrix).astype(int)
np.fill_diagonal(cooccurrence.values, 0)

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
ax.set_title("Skill Co-occurrence Matrix (Top 12 Skills)", fontsize=14, fontweight="bold")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/14_skill_cooccurrence.png")
plt.close()
print("Saved: eda_outputs/14_skill_cooccurrence.png")

# ─────────────────────────────────────────────
# 16. DESCRIPTION LENGTH ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 16 — JOB DESCRIPTION LENGTH")
print("=" * 60)

postings["desc_length"] = postings["description"].fillna("").apply(lambda x: len(x.split()))
print(postings["desc_length"].describe().to_string())

fig, ax = plt.subplots(figsize=(11, 5))
postings[postings["desc_length"] < 2000]["desc_length"].hist(
    bins=60, ax=ax, color="cadetblue", edgecolor="white")
ax.set_title("Distribution of Job Description Word Count", fontsize=13, fontweight="bold")
ax.set_xlabel("Word Count")
ax.set_ylabel("Number of Postings")
plt.tight_layout()
plt.savefig("eda_outputs/15_description_length.png")
plt.close()
print("Saved: eda_outputs/15_description_length.png")

# ─────────────────────────────────────────────
# 17. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 17 — FINAL SUMMARY")
print("=" * 60)

print(f"Total job postings          : {len(postings):,}")
print(f"Unique job titles           : {postings['title'].nunique():,}")
print(f"Unique companies            : {postings['company_name'].nunique():,}")
print(f"Unique skills (categories)  : {skills_map.shape[0]}")
print(f"Total skill-job mappings    : {len(job_skills):,}")
print(f"Jobs with salary data       : {len(salaries):,}")
print(f"Jobs with industry data     : {len(job_industries):,}")
print(f"Jobs with experience level  : {postings['formatted_experience_level'].notna().sum():,}")
print(f"Jobs with description       : {postings['description'].notna().sum():,}")
print(f"\nAll EDA plots saved to: eda_outputs/")
