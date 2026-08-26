"""
train.py
Trains SVD and NMF collaborative filtering models, evaluates both on a
held-out test set, picks the better model, and saves all artifacts.

Run before starting the server:
    python train.py
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from pathlib import Path
import joblib

DATA_DIR  = Path(__file__).parent / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
jobs         = pd.read_csv(DATA_DIR / "jobs.csv")
users        = pd.read_csv(DATA_DIR / "users.csv")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")
print(f"Users: {len(users)} | Jobs: {len(jobs)} | Interactions: {len(interactions)}")

# ── TF-IDF content-based model ────────────────────────────────────────────────
jobs["_content"] = jobs["skills"] + " " + jobs["location"]
vectorizer  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
job_vectors = vectorizer.fit_transform(jobs["_content"])
print(f"TF-IDF matrix: {job_vectors.shape}")

# ── Train / test split (80/20) on interactions ────────────────────────────────
train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)

def build_matrix(df, all_users, all_jobs):
    mat = pd.pivot_table(df, index="user_id", columns="job_id",
                         values="interaction_score", fill_value=0)
    mat = mat.reindex(index=all_users, columns=all_jobs, fill_value=0)
    return mat

all_users = interactions["user_id"].unique()
all_jobs  = interactions["job_id"].unique()

train_matrix = build_matrix(train_df, all_users, all_jobs)
full_matrix  = build_matrix(interactions, all_users, all_jobs)

n_components = min(50, len(all_users) - 1, len(all_jobs) - 1)

# ── Helper: evaluate on held-out test rows ────────────────────────────────────
def rmse_on_test(pred_matrix, test_df, train_matrix):
    """RMSE only on test interactions — proper held-out evaluation."""
    test_filtered = test_df[
        test_df["user_id"].isin(train_matrix.index) &
        test_df["job_id"].isin(train_matrix.columns)
    ]
    actual, predicted = [], []
    for _, row in test_filtered.iterrows():
        u_idx = pred_matrix.index.get_loc(row["user_id"])
        j_idx = pred_matrix.columns.get_loc(row["job_id"])
        actual.append(row["interaction_score"])
        predicted.append(pred_matrix.iloc[u_idx, j_idx])
    return sqrt(mean_squared_error(actual, predicted))

def precision_at_k_holdout(pred_matrix, test_df, k=10, threshold=3):
    """
    Precision@K on held-out test set.
    threshold=3 means score>=3 counts as relevant (realistic for 1-5 scale).
    """
    test_liked = (
        test_df[test_df["interaction_score"] >= threshold]
        .groupby("user_id")["job_id"]
        .apply(set)
    )
    precisions = []
    for user in pred_matrix.index:
        if user not in test_liked.index:
            continue
        liked = test_liked[user]
        top_k = pred_matrix.loc[user].nlargest(k).index.tolist()
        hits  = len([j for j in top_k if j in liked])
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0

def recall_at_k_holdout(pred_matrix, test_df, k=10, threshold=3):
    """Recall@K on held-out test set."""
    test_liked = (
        test_df[test_df["interaction_score"] >= threshold]
        .groupby("user_id")["job_id"]
        .apply(set)
    )
    recalls = []
    for user in pred_matrix.index:
        if user not in test_liked.index:
            continue
        liked = test_liked[user]
        if not liked:
            continue
        top_k = pred_matrix.loc[user].nlargest(k).index.tolist()
        hits  = len([j for j in top_k if j in liked])
        recalls.append(hits / len(liked))
    return np.mean(recalls) if recalls else 0.0

def ndcg_at_k_holdout(pred_matrix, test_df, k=10):
    """
    NDCG@K — industry standard ranking metric.
    Rewards hitting relevant items AND hitting them higher in the list.
    """
    test_scores = test_df.groupby(["user_id","job_id"])["interaction_score"].first()
    ndcgs = []
    for user in pred_matrix.index:
        if user not in test_scores.index.get_level_values(0):
            continue
        user_test = test_scores[user]
        top_k = pred_matrix.loc[user].nlargest(k).index.tolist()
        dcg, idcg = 0.0, 0.0
        ideal = sorted(user_test.values, reverse=True)[:k]
        for i, job in enumerate(top_k):
            rel = user_test.get(job, 0)
            dcg  += rel / np.log2(i + 2)
        for i, rel in enumerate(ideal):
            idcg += rel / np.log2(i + 2)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs) if ndcgs else 0.0

# ── Train SVD ─────────────────────────────────────────────────────────────────
print("\nTraining SVD...")
svd          = TruncatedSVD(n_components=n_components, random_state=42)
svd_u        = svd.fit_transform(train_matrix.values)
svd_pred_val = np.dot(svd_u, svd.components_)
svd_pred_df  = pd.DataFrame(svd_pred_val, index=train_matrix.index, columns=train_matrix.columns)

svd_rmse_train = sqrt(mean_squared_error(
    train_matrix.values[train_matrix.values > 0],
    svd_pred_val[train_matrix.values > 0]
))
svd_rmse_test  = rmse_on_test(svd_pred_df, test_df, train_matrix)
svd_p5         = precision_at_k_holdout(svd_pred_df, test_df, k=5)
svd_p10        = precision_at_k_holdout(svd_pred_df, test_df, k=10)
svd_r10        = recall_at_k_holdout(svd_pred_df, test_df, k=10)
svd_ndcg       = ndcg_at_k_holdout(svd_pred_df, test_df, k=10)
svd_var        = svd.explained_variance_ratio_.sum()

# ── Train NMF ─────────────────────────────────────────────────────────────────
print("Training NMF...")
nmf          = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=500)
nmf_u        = nmf.fit_transform(train_matrix.values)
nmf_pred_val = np.dot(nmf_u, nmf.components_)
nmf_pred_df  = pd.DataFrame(nmf_pred_val, index=train_matrix.index, columns=train_matrix.columns)

nmf_rmse_train = sqrt(mean_squared_error(
    train_matrix.values[train_matrix.values > 0],
    nmf_pred_val[train_matrix.values > 0]
))
nmf_rmse_test  = rmse_on_test(nmf_pred_df, test_df, train_matrix)
nmf_p5         = precision_at_k_holdout(nmf_pred_df, test_df, k=5)
nmf_p10        = precision_at_k_holdout(nmf_pred_df, test_df, k=10)
nmf_r10        = recall_at_k_holdout(nmf_pred_df, test_df, k=10)
nmf_ndcg       = ndcg_at_k_holdout(nmf_pred_df, test_df, k=10)

# ── Pick best model by test RMSE ──────────────────────────────────────────────
best_model_name = "SVD" if svd_rmse_test <= nmf_rmse_test else "NMF"

# Retrain winner on FULL interaction matrix for production
print(f"\nBest model: {best_model_name} — retraining on full data...")
if best_model_name == "SVD":
    final_model   = TruncatedSVD(n_components=n_components, random_state=42)
    final_u       = final_model.fit_transform(full_matrix.values)
    final_pred    = np.dot(final_u, final_model.components_)
else:
    final_model   = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=500)
    final_u       = final_model.fit_transform(full_matrix.values)
    final_pred    = np.dot(final_u, final_model.components_)

best_predictions = pd.DataFrame(final_pred, index=full_matrix.index, columns=full_matrix.columns)

# ── Print evaluation report ───────────────────────────────────────────────────
print("\n" + "="*57)
print("  MODEL EVALUATION REPORT")
print("="*57)
print(f"  Dataset")
print(f"    Users            : {len(users)}")
print(f"    Jobs             : {len(jobs)}")
print(f"    Interactions     : {len(interactions)}")
print(f"    Avg per user     : {len(interactions)/len(users):.1f}")
print(f"    Train / Test     : {len(train_df)} / {len(test_df)}")
print()
print(f"  {'Metric':<28} {'SVD':>8} {'NMF':>8}")
print(f"  {'-'*44}")
print(f"  {'Components':<28} {n_components:>8} {n_components:>8}")
print(f"  {'Variance explained (SVD)':<28} {svd_var:>7.2%} {'N/A':>8}")
print(f"  {'RMSE (train)':<28} {svd_rmse_train:>8.4f} {nmf_rmse_train:>8.4f}")
print(f"  {'RMSE (test 20%)':<28} {svd_rmse_test:>8.4f} {nmf_rmse_test:>8.4f}")
print(f"  {'Precision@5  (held-out)':<28} {svd_p5:>8.4f} {nmf_p5:>8.4f}")
print(f"  {'Precision@10 (held-out)':<28} {svd_p10:>8.4f} {nmf_p10:>8.4f}")
print(f"  {'Recall@10    (held-out)':<28} {svd_r10:>8.4f} {nmf_r10:>8.4f}")
print(f"  {'NDCG@10      (held-out)':<28} {svd_ndcg:>8.4f} {nmf_ndcg:>8.4f}")
print(f"  {'-'*44}")
print(f"  Selected model: {best_model_name} (lower test RMSE)")
print("="*57 + "\n")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(vectorizer,       MODEL_DIR / "tfidf_vectorizer.pkl")
joblib.dump(job_vectors,      MODEL_DIR / "job_vectors.pkl")
joblib.dump(best_predictions, MODEL_DIR / "svd_predictions.pkl")   # name kept for API compat
joblib.dump(full_matrix,      MODEL_DIR / "interaction_matrix.pkl")
jobs[["job_id", "job_title", "skills", "location"]].to_pickle(MODEL_DIR / "jobs.pkl")

print("Artifacts saved to model/")
