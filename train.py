"""
Model Training — AI-Powered Job Recommendation System

Architecture:
  - Content-based filtering via cosine similarity on TF-IDF vectors
  - Skill gap analysis via set difference on skill categories
  - Evaluation via simulated held-out test set (precision@k, recall@k, ndcg@k)

Outputs:
  data/processed/model_index.pkl     — full model bundle (vectorizer + matrix + df)
  data/processed/eval_results.json   — evaluation metrics
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from scipy.sparse import load_npz, vstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING ARTIFACTS")
print("=" * 60)

df     = pd.read_csv("data/processed/jobs_features.csv")
tfidf  = load_npz("data/processed/tfidf_matrix.npz")

with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("data/processed/skill_binarizer.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("data/processed/feature_metadata.json") as f:
    meta = json.load(f)

# Parse skills back to list
df["skills_list"] = df["skills"].apply(lambda x: x.split("|") if isinstance(x, str) else [])

print(f"Jobs loaded        : {len(df):,}")
print(f"TF-IDF matrix      : {tfidf.shape}")
print(f"Skill categories   : {len(meta['skill_categories'])}")
print(f"Vectorizer vocab   : {meta['tfidf_vocab_size']:,}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — CORE RECOMMENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — BUILDING RECOMMENDER")
print("=" * 60)

SKILL_CATEGORIES = meta["skill_categories"]
EXP_ORDINAL      = meta["exp_ordinal_map"]
SALARY_MEDIAN    = meta["salary_median"]


def encode_user_input(user_skills: list, experience_level: str = None) -> str:
    """
    Convert raw user skills + experience into the same feature_text
    format used during training so the vectorizer can transform it.
    """
    skills_clean = [s.strip().title() for s in user_skills]
    # Match to known categories (fuzzy fallback to raw input)
    matched = [s for s in skills_clean if s in SKILL_CATEGORIES]
    if not matched:
        matched = skills_clean   # use as-is, TF-IDF will handle unknown terms

    skills_str = " ".join(matched).lower().replace("/", " ").replace("-", " ")
    exp_str    = str(experience_level or "").lower().replace("-", " ")

    # Mirror the feature_text construction from FE (skills x3, no desc)
    return f"{skills_str} {skills_str} {skills_str} {exp_str}"


def recommend(
    user_skills      : list,
    experience_level : str  = None,
    top_n            : int  = 10,
    filter_exp       : bool = False,
    filter_remote    : bool = False,
) -> pd.DataFrame:
    """
    Returns top_n job recommendations ranked by cosine similarity.

    Parameters
    ----------
    user_skills      : list of skill strings e.g. ['Python', 'Machine Learning']
    experience_level : optional filter e.g. 'Entry', 'Mid-Senior'
    top_n            : number of results to return
    filter_exp       : if True, only return jobs matching experience_level
    filter_remote    : if True, only return remote jobs
    """
    # Build user vector
    user_text   = encode_user_input(user_skills, experience_level)
    user_vector = vectorizer.transform([user_text])   # (1, 8000)

    # Candidate pool — optionally filtered
    candidate_df  = df.copy()
    candidate_idx = np.arange(len(df))

    if filter_exp and experience_level:
        mask          = candidate_df["experience_level"] == experience_level
        candidate_df  = candidate_df[mask].reset_index(drop=True)
        candidate_idx = np.where(mask)[0]

    if filter_remote:
        mask          = candidate_df["is_remote"] == 1
        candidate_df  = candidate_df[mask].reset_index(drop=True)
        candidate_idx = candidate_idx[candidate_df.index] if filter_exp else np.where(mask)[0]

    if len(candidate_df) == 0:
        return pd.DataFrame()

    # Cosine similarity against candidate pool
    candidate_matrix = tfidf[candidate_idx]
    scores           = cosine_similarity(user_vector, candidate_matrix).flatten()

    # Get top_n indices
    top_idx    = np.argsort(scores)[::-1][:top_n]
    top_scores = scores[top_idx]

    results = candidate_df.iloc[top_idx][[
        "job_id", "title_clean", "experience_level",
        "industry", "skills", "salary_mid",
        "work_type", "is_remote", "state"
    ]].copy()
    results["match_score"] = np.round(top_scores * 100, 2)

    return results.reset_index(drop=True)


def skill_gap(user_skills: list, job_skills_str: str) -> dict:
    """
    Returns skills the user is missing for a given job,
    and skills the user has that match.
    """
    job_skills_list  = [s.strip() for s in job_skills_str.split("|")]
    user_skills_norm = [s.strip().title() for s in user_skills]

    matched  = [s for s in job_skills_list if s in user_skills_norm]
    missing  = [s for s in job_skills_list if s not in user_skills_norm]

    return {
        "matched_skills" : matched,
        "missing_skills" : missing,
        "match_pct"      : round(len(matched) / max(len(job_skills_list), 1) * 100, 1)
    }


print("Recommender functions defined.")

# ─────────────────────────────────────────────────────────────
# STEP 3 — SMOKE TEST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — SMOKE TEST")
print("=" * 60)

test_skills = ["Information Technology", "Project Management", "Analyst"]
results     = recommend(test_skills, experience_level="Mid-Senior", top_n=5)

print(f"Query skills : {test_skills}")
print(f"Results      : {len(results)} jobs")
print()
print(results[["title_clean", "experience_level", "industry",
               "skills", "match_score"]].to_string())

print()
# Skill gap for top result
if len(results) > 0:
    gap = skill_gap(test_skills, results.iloc[0]["skills"])
    print(f"Skill gap for '{results.iloc[0]['title_clean']}':")
    print(f"  Matched  : {gap['matched_skills']}")
    print(f"  Missing  : {gap['missing_skills']}")
    print(f"  Match %  : {gap['match_pct']}%")

# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT + EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — TRAIN / TEST SPLIT + EVALUATION")
print("=" * 60)

"""
Evaluation strategy for content-based filtering (no user data):

We simulate a held-out test by:
1. Sampling N 'query' jobs as pseudo-users
2. Their skills = ground truth relevant skills
3. We query the model with those skills
4. We check if the original job appears in top-k results
   AND how many results share the same skill set (precision proxy)

Metrics:
  - Hit Rate @ k    : did the original job appear in top-k?
  - Precision @ k   : fraction of top-k that share ≥1 skill with query
  - Recall @ k      : fraction of all jobs sharing skills that appear in top-k
  - NDCG @ k        : position-weighted relevance score
  - Mean Similarity : average cosine score of top-k results
"""

np.random.seed(42)
N_TEST = 500   # number of query jobs to evaluate

# Sample test jobs — only jobs that have real skill tags (not just 'Other')
test_pool = df[df["skills_list"].apply(lambda x: x != ["Other"] and len(x) > 0)]
test_jobs = test_pool.sample(N_TEST, random_state=42).reset_index(drop=True)
test_idx  = test_jobs.index.tolist()

print(f"Test pool size : {len(test_pool):,}")
print(f"Test queries   : {N_TEST}")

K_VALUES = [5, 10, 20]

def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return np.sum(relevances / np.log2(positions + 1))

def ndcg_at_k(relevances, k):
    dcg  = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


metrics = {k: {"hit_rate": [], "precision": [], "recall": [],
               "ndcg": [], "mean_sim": []} for k in K_VALUES}

for i, row in test_jobs.iterrows():
    query_skills = row["skills_list"]
    query_exp    = row["experience_level"] if row["experience_level"] != "Not Specified" else None
    true_job_id  = row["job_id"]

    # Get top-20 recommendations (covers all K values)
    recs = recommend(query_skills, experience_level=query_exp, top_n=20)
    if len(recs) == 0:
        continue

    # Ground truth: jobs that share at least 1 skill with query
    query_skill_set = set(query_skills)
    relevant_mask   = df["skills_list"].apply(
        lambda x: len(set(x) & query_skill_set) > 0
    )
    n_relevant = relevant_mask.sum()

    for k in K_VALUES:
        top_k = recs.head(k)

        # Hit rate — did original job appear?
        hit = int(true_job_id in top_k["job_id"].values)

        # Precision — fraction of top-k sharing ≥1 skill with query
        top_k_skills = top_k["skills"].apply(
            lambda x: set(x.split("|")) if isinstance(x, str) else set()
        )
        relevant_in_topk = top_k_skills.apply(
            lambda s: int(len(s & query_skill_set) > 0)
        ).sum()
        precision = relevant_in_topk / k

        # Recall
        recall = relevant_in_topk / max(n_relevant, 1)

        # NDCG
        relevances = top_k_skills.apply(
            lambda s: 1 if len(s & query_skill_set) > 0 else 0
        ).tolist()
        ndcg = ndcg_at_k(relevances, k)

        # Mean similarity
        mean_sim = top_k["match_score"].mean() / 100

        metrics[k]["hit_rate"].append(hit)
        metrics[k]["precision"].append(precision)
        metrics[k]["recall"].append(recall)
        metrics[k]["ndcg"].append(ndcg)
        metrics[k]["mean_sim"].append(mean_sim)

# ─────────────────────────────────────────────────────────────
# STEP 5 — PRINT METRICS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — EVALUATION RESULTS")
print("=" * 60)

eval_results = {}
print(f"\n{'Metric':<20} {'@5':>10} {'@10':>10} {'@20':>10}")
print("-" * 52)

for metric in ["hit_rate", "precision", "recall", "ndcg", "mean_sim"]:
    row_vals = {}
    row_str  = f"{metric:<20}"
    for k in K_VALUES:
        val = np.mean(metrics[k][metric])
        row_vals[f"@{k}"] = round(float(val), 4)
        row_str += f"{val:>10.4f}"
    eval_results[metric] = row_vals
    print(row_str)

print()
print("Interpretation:")
print(f"  Precision@10 : of every 10 recommendations, "
      f"{eval_results['precision']['@10']*10:.1f} share skills with the query")
print(f"  Hit Rate@10  : {eval_results['hit_rate']['@10']*100:.1f}% of the time the "
      f"original job appears in top-10")
print(f"  NDCG@10      : {eval_results['ndcg']['@10']:.4f} (1.0 = perfect ranking)")

# ─────────────────────────────────────────────────────────────
# STEP 6 — SAVE MODEL BUNDLE + EVAL RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — SAVING MODEL BUNDLE")
print("=" * 60)

model_bundle = {
    "vectorizer"  : vectorizer,
    "mlb"         : mlb,
    "metadata"    : meta,
}
with open("data/processed/model_index.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
print("Saved: data/processed/model_index.pkl")

# Save eval results
eval_output = {
    "n_test_queries" : N_TEST,
    "k_values"       : K_VALUES,
    "metrics"        : eval_results,
}
with open("data/processed/eval_results.json", "w") as f:
    json.dump(eval_output, f, indent=2)
print("Saved: data/processed/eval_results.json")

# ─────────────────────────────────────────────────────────────
# STEP 7 — END-TO-END INFERENCE DEMO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — END-TO-END INFERENCE DEMO")
print("=" * 60)

demo_cases = [
    {
        "skills"    : ["Information Technology", "Engineering", "Project Management"],
        "exp"       : "Mid-Senior",
        "label"     : "Software / IT Professional"
    },
    {
        "skills"    : ["Sales", "Marketing", "Business Development"],
        "exp"       : "Entry",
        "label"     : "Entry-level Sales/Marketing"
    },
    {
        "skills"    : ["Health Care Provider", "Science", "Research"],
        "exp"       : "Associate",
        "label"     : "Healthcare / Research"
    },
    {
        "skills"    : ["Finance", "Accounting/Auditing", "Analyst"],
        "exp"       : "Mid-Senior",
        "label"     : "Finance / Accounting"
    },
]

for case in demo_cases:
    print(f"\n--- {case['label']} ---")
    print(f"Input skills : {case['skills']}")
    print(f"Experience   : {case['exp']}")

    recs = recommend(case["skills"], experience_level=case["exp"], top_n=5)
    print(f"\nTop 5 Recommendations:")
    print(recs[["title_clean", "experience_level", "industry",
                "match_score"]].to_string(index=False))

    # Skill gap for top result
    if len(recs) > 0:
        gap = skill_gap(case["skills"], recs.iloc[0]["skills"])
        print(f"\nSkill gap for top result '{recs.iloc[0]['title_clean']}':")
        print(f"  Matched : {gap['matched_skills']}")
        print(f"  Missing : {gap['missing_skills']}")
        print(f"  Match % : {gap['match_pct']}%")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"  Model bundle : data/processed/model_index.pkl")
print(f"  Eval results : data/processed/eval_results.json")
print(f"  TF-IDF matrix: data/processed/tfidf_matrix.npz  (used at inference)")
print(f"  Features CSV : data/processed/jobs_features.csv (used at inference)")
