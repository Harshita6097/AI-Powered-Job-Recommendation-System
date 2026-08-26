import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def load_data():
    jobs = pd.read_csv(DATA_DIR / "jobs.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")
    return jobs, users, interactions


def build_models(jobs: pd.DataFrame, interactions: pd.DataFrame):
    # --- Content-based: TF-IDF on job skills + location ---
    jobs["_content"] = jobs["skills"] + " " + jobs["location"]
    vectorizer = TfidfVectorizer(stop_words="english")
    job_vectors = vectorizer.fit_transform(jobs["_content"])

    # --- Collaborative: SVD on user-job interaction matrix ---
    matrix = pd.pivot_table(
        interactions, index="user_id", columns="job_id",
        values="interaction_score", fill_value=0
    )
    n_components = min(10, matrix.shape[0] - 1, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix.values)
    job_factors = svd.components_.T
    svd_predictions = pd.DataFrame(
        np.dot(user_factors, job_factors.T),
        index=matrix.index,
        columns=matrix.columns
    )

    return vectorizer, job_vectors, svd_predictions, matrix


def _content_scores(user_skills: str, vectorizer, job_vectors, jobs: pd.DataFrame) -> pd.Series:
    user_vec = vectorizer.transform([user_skills])
    scores = cosine_similarity(user_vec, job_vectors).flatten()
    return pd.Series(scores, index=jobs["job_id"].values)


def recommend(
    jobs: pd.DataFrame,
    vectorizer,
    job_vectors,
    svd_predictions: pd.DataFrame,
    interaction_matrix: pd.DataFrame,
    user_id: str = None,
    user_skills: str = None,
    top_k: int = 8
) -> list[dict]:
    """
    Hybrid recommendation:
    - Known user with history  → blend SVD (collaborative) + TF-IDF (content)
    - New / cold-start user    → pure TF-IDF content-based on provided skills
    """
    known_user = user_id and user_id in svd_predictions.index
    has_skills = bool(user_skills and user_skills.strip())

    if not known_user and not has_skills:
        raise ValueError("Provide either a known user_id or user_skills.")

    all_job_ids = jobs["job_id"].values

    if known_user:
        collab = svd_predictions.loc[user_id].reindex(all_job_ids, fill_value=0)
        # Normalise to [0, 1]
        c_min, c_max = collab.min(), collab.max()
        collab_norm = (collab - c_min) / (c_max - c_min + 1e-9)

        # How many interactions does this user have? → drives alpha
        n_interactions = (interaction_matrix.loc[user_id] > 0).sum() if user_id in interaction_matrix.index else 0
        alpha = min(0.85, 0.4 + 0.05 * n_interactions)  # more history → trust collab more

        if has_skills:
            content = _content_scores(user_skills, vectorizer, job_vectors, jobs)
        else:
            # Derive skills from user's top interacted jobs
            top_jobs = interaction_matrix.loc[user_id].nlargest(5).index.tolist()
            inferred_skills = " ".join(
                jobs[jobs["job_id"].isin(top_jobs)]["skills"].tolist()
            )
            content = _content_scores(inferred_skills, vectorizer, job_vectors, jobs)

        final_scores = alpha * collab_norm + (1 - alpha) * content

    else:
        # Cold start — pure content
        final_scores = _content_scores(user_skills, vectorizer, job_vectors, jobs)

    final_scores = final_scores.reindex(all_job_ids, fill_value=0)
    top_ids = final_scores.nlargest(top_k).index.tolist()

    result = []
    for job_id in top_ids:
        row = jobs[jobs["job_id"] == job_id].iloc[0]
        result.append({
            "job_id": job_id,
            "job_title": row["job_title"],
            "skills": row["skills"],
            "location": row["location"],
            "score": round(float(final_scores[job_id]), 4),
            "mode": "hybrid" if known_user else "content-based"
        })

    return result
