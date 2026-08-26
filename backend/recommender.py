import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).parent / "model"


def load_model():
    vectorizer       = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    job_vectors      = joblib.load(MODEL_DIR / "job_vectors.pkl")
    svd_predictions  = joblib.load(MODEL_DIR / "svd_predictions.pkl")
    interaction_matrix = joblib.load(MODEL_DIR / "interaction_matrix.pkl")
    jobs             = pd.read_pickle(MODEL_DIR / "jobs.pkl")
    return vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs


def _content_scores(user_skills: str, vectorizer, job_vectors, job_ids) -> pd.Series:
    user_vec = vectorizer.transform([user_skills])
    scores = cosine_similarity(user_vec, job_vectors).flatten()
    return pd.Series(scores, index=job_ids)


def recommend(
    vectorizer,
    job_vectors,
    svd_predictions: pd.DataFrame,
    interaction_matrix: pd.DataFrame,
    jobs: pd.DataFrame,
    user_id: str = None,
    user_skills: str = None,
    top_k: int = 8
) -> list[dict]:
    """
    Hybrid recommendation:
    - Known user  → α × SVD_score + (1−α) × TF-IDF_score
    - New user    → pure TF-IDF content-based on provided skills
    α scales with number of past interactions (more history = trust collab more)
    """
    known_user = user_id and user_id in svd_predictions.index
    has_skills = bool(user_skills and user_skills.strip())

    if not known_user and not has_skills:
        raise ValueError("Provide either a valid user_id or user_skills.")

    all_job_ids = jobs["job_id"].values

    if known_user:
        collab = svd_predictions.loc[user_id].reindex(all_job_ids, fill_value=0)
        c_min, c_max = collab.min(), collab.max()
        collab_norm = (collab - c_min) / (c_max - c_min + 1e-9)

        n_interactions = int((interaction_matrix.loc[user_id] > 0).sum()) \
            if user_id in interaction_matrix.index else 0
        alpha = min(0.85, 0.4 + 0.05 * n_interactions)

        if has_skills:
            content = _content_scores(user_skills, vectorizer, job_vectors, all_job_ids)
        else:
            top_jobs = interaction_matrix.loc[user_id].nlargest(5).index.tolist()
            inferred = " ".join(jobs[jobs["job_id"].isin(top_jobs)]["skills"].tolist())
            content = _content_scores(inferred, vectorizer, job_vectors, all_job_ids)

        final_scores = alpha * collab_norm + (1 - alpha) * content
        mode = "hybrid"

    else:
        final_scores = _content_scores(user_skills, vectorizer, job_vectors, all_job_ids)
        mode = "content-based"

    final_scores = final_scores.reindex(all_job_ids, fill_value=0)
    top_ids = final_scores.nlargest(top_k).index.tolist()

    return [
        {
            "job_id": jid,
            "job_title": jobs.loc[jobs["job_id"] == jid, "job_title"].values[0],
            "skills": jobs.loc[jobs["job_id"] == jid, "skills"].values[0],
            "location": jobs.loc[jobs["job_id"] == jid, "location"].values[0],
            "score": round(float(final_scores[jid]), 4),
            "mode": mode
        }
        for jid in top_ids
    ]
