"""
evaluate.py — Benchmark all retrieval modes against the same 500-query test set.

Modes evaluated: tfidf | bm25 | dense | hybrid
Protocol (mirrors train.py):
  - Sample 500 jobs from jobs_granular.csv (seed=42)
  - Query = job's own granular_skills (new modes) or broad skills (tfidf)
  - Relevance = result shares >= 1 skill with query
  - Metrics: Hit Rate, Precision, NDCG, Mean Score at k=5,10,20

Output:
  data/processed/eval_results_all_modes.json
"""

import json
import time
import numpy as np
import pandas as pd

import config
from config import (
    GRANULAR_JOBS_CSV, EVAL_N_QUERIES, EVAL_K_VALUES, EVAL_RANDOM_SEED,
    BM25_TOP_K, DENSE_TOP_K,
)
from src.evaluation.evaluator import compute_metrics, aggregate

# ── load granular jobs (used by all new modes) ─────────────────────────────────
print("Loading jobs_granular.csv ...")
df_g = pd.read_csv(GRANULAR_JOBS_CSV, dtype={"job_id": str})
df_g["granular_list"] = df_g["granular_skills"].apply(
    lambda x: [s.strip() for s in x.split("|") if s.strip()] if isinstance(x, str) else []
)
print(f"  {len(df_g):,} jobs loaded")

# ── sample test set ────────────────────────────────────────────────────────────
np.random.seed(EVAL_RANDOM_SEED)
test_pool = df_g[df_g["granular_list"].apply(lambda x: len(x) > 0)]
test_jobs = test_pool.sample(min(EVAL_N_QUERIES, len(test_pool)), random_state=EVAL_RANDOM_SEED)
test_jobs = test_jobs.reset_index(drop=True)
print(f"  Test queries: {len(test_jobs)}")


# ── helpers ────────────────────────────────────────────────────────────────────

def _fmt_results_granular(job_ids: list, scores: list) -> list[dict]:
    """Build result dicts from job_id list + scores using granular lookup."""
    lookup = df_g.set_index("job_id")["granular_skills"].to_dict()
    return [
        {"job_id": jid, "granular_skills": lookup.get(jid, ""), "match_score": float(s)}
        for jid, s in zip(job_ids, scores)
    ]


def _gs_lookup():
    """Pre-built granular_skills lookup — NaN -> empty string."""
    return {str(k): (v if isinstance(v, str) else "") for k, v in
            df_g.set_index("job_id")["granular_skills"].to_dict().items()}


def run_bm25_eval(test_jobs: pd.DataFrame, k_values: list) -> list[dict]:
    from src.retrieval.bm25_retriever import load_index, tokenise_query, query_bm25
    print("  Loading BM25 index ...")
    bm25, bm25_ids = load_index()
    gs = _gs_lookup()
    per_query = []
    for _, row in test_jobs.iterrows():
        skills = row["granular_list"]
        tokens = tokenise_query(", ".join(skills))
        results = query_bm25(bm25, bm25_ids, tokens, top_k=max(k_values))
        fmt = [{"job_id": r["job_id"], "granular_skills": gs.get(r["job_id"], ""),
                "match_score": r["bm25_score"]} for r in results]
        per_query.append(compute_metrics(fmt, row["job_id"], set(skills), "granular_skills", k_values))
    return per_query


def run_dense_eval(test_jobs: pd.DataFrame, k_values: list) -> list[dict]:
    from src.retrieval.dense_retriever import load_index, query_dense
    print("  Loading FAISS index ...")
    faiss_idx, _, dense_ids = load_index()
    gs = _gs_lookup()
    per_query = []
    for _, row in test_jobs.iterrows():
        skills = row["granular_list"]
        query_text = ", ".join(skills)
        results = query_dense(faiss_idx, dense_ids, query_text, top_k=max(k_values))
        fmt = [{"job_id": r["job_id"], "granular_skills": gs.get(r["job_id"], ""),
                "match_score": r["dense_score"]} for r in results]
        per_query.append(compute_metrics(fmt, row["job_id"], set(skills), "granular_skills", k_values))
    return per_query


def run_hybrid_eval(test_jobs: pd.DataFrame, k_values: list) -> list[dict]:
    from src.retrieval.bm25_retriever import load_index as load_bm25, tokenise_query, query_bm25
    from src.retrieval.dense_retriever import load_index as load_dense, query_dense
    from src.retrieval.rrf import fuse
    from src.ranking.reranker import rerank
    print("  Loading BM25 + FAISS indexes ...")
    bm25, bm25_ids = load_bm25()
    faiss_idx, _, dense_ids = load_dense()
    lookup = df_g.set_index("job_id").to_dict("index")
    gs = _gs_lookup()
    per_query = []
    for _, row in test_jobs.iterrows():
        skills = row["granular_list"]
        tokens = tokenise_query(", ".join(skills))
        query_text = ", ".join(skills)
        bm25_res  = query_bm25(bm25, bm25_ids, tokens, top_k=BM25_TOP_K)
        dense_res = query_dense(faiss_idx, dense_ids, query_text, top_k=DENSE_TOP_K)
        rrf_pool  = fuse([bm25_res, dense_res], list_names=["bm25", "dense"], top_n=20)
        final     = rerank(query_text, rrf_pool, lookup, top_k=max(k_values))
        fmt = [{"job_id": r["job_id"], "granular_skills": gs.get(r["job_id"], ""),
                "match_score": r["reranker_score"]} for r in final]
        per_query.append(compute_metrics(fmt, row["job_id"], set(skills), "granular_skills", k_values))
    return per_query


def run_tfidf_eval(test_jobs: pd.DataFrame, k_values: list) -> list[dict]:
    import pickle
    from scipy.sparse import load_npz
    from sklearn.metrics.pairwise import cosine_similarity
    from config import JOBS_FEATURES_CSV, TFIDF_MATRIX_NPZ, TFIDF_VECTORIZER_PKL
    print("  Loading TF-IDF artifacts ...")
    df_f = pd.read_csv(JOBS_FEATURES_CSV, dtype={"job_id": str})
    df_f["skills_list"] = df_f["skills"].apply(
        lambda x: [s.strip() for s in x.split("|") if s.strip()] if isinstance(x, str) else []
    )
    tfidf_mat = load_npz(TFIDF_MATRIX_NPZ)
    with open(TFIDF_VECTORIZER_PKL, "rb") as f:
        vectorizer = pickle.load(f)
    # build job_id -> index map for tfidf df
    id_to_idx = {str(jid): i for i, jid in enumerate(df_f["job_id"])}
    skills_lookup = df_f.set_index("job_id")["skills"].to_dict()

    per_query = []
    for _, row in test_jobs.iterrows():
        skills = row["granular_list"]
        # map granular -> broad for tfidf query (best effort: use raw skills)
        query_text = " ".join(skills).lower()
        user_vec = vectorizer.transform([query_text])
        scores = cosine_similarity(user_vec, tfidf_mat).flatten()
        top_idx = np.argsort(scores)[::-1][:max(k_values)]
        fmt = [
            {"job_id": str(df_f.iloc[i]["job_id"]),
             "granular_skills": skills_lookup.get(str(df_f.iloc[i]["job_id"]), ""),
             "match_score": float(scores[i] * 100)}
            for i in top_idx
        ]
        per_query.append(compute_metrics(fmt, row["job_id"], set(skills), "granular_skills", k_values))
    return per_query


# ── run all modes ──────────────────────────────────────────────────────────────

MODES = ["tfidf", "bm25", "dense", "hybrid"]
all_results = {}
import os

for mode in MODES:
    print(f"\n{'='*55}")
    print(f"  Evaluating: {mode.upper()}")
    print(f"{'='*55}")
    t0 = time.time()

    if mode == "tfidf":
        per_query = run_tfidf_eval(test_jobs, EVAL_K_VALUES)
    elif mode == "bm25":
        per_query = run_bm25_eval(test_jobs, EVAL_K_VALUES)
    elif mode == "dense":
        per_query = run_dense_eval(test_jobs, EVAL_K_VALUES)
    else:
        per_query = run_hybrid_eval(test_jobs, EVAL_K_VALUES)

    elapsed = time.time() - t0
    agg = aggregate(per_query, EVAL_K_VALUES)
    all_results[mode] = {"metrics": agg, "elapsed_s": round(elapsed, 1), "n_queries": len(per_query)}
    print(f"  Done in {elapsed:.1f}s")


# ── print comparison table ─────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("  EVALUATION RESULTS -- all modes, 500 queries")
print(f"{'='*75}")

all_modes_ordered = [m for m in ["tfidf", "bm25", "dense", "hybrid"] if m in all_results]
for metric in ["hit_rate", "precision", "ndcg", "mean_score"]:
    print(f"\n  {metric.upper()}")
    header = f"  {'Mode':<10}" + "".join(f"  @{k:<6}" for k in EVAL_K_VALUES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode in all_modes_ordered:
        row = f"  {mode:<10}"
        for k in EVAL_K_VALUES:
            # JSON round-trips int keys as strings
            key = k if k in all_results[mode]["metrics"] else str(k)
            val = all_results[mode]["metrics"][key][metric]
            row += f"  {val:<8.4f}"
        print(row)

print(f"\n  Runtimes: " + "  |  ".join(f"{m}: {all_results[m]['elapsed_s']}s" for m in all_modes_ordered))

# ── save ───────────────────────────────────────────────────────────────────────

out_path = "data/processed/eval_results_all_modes.json"
with open(out_path, "w") as f:
    json.dump({
        "n_queries": len(test_jobs),
        "k_values":  EVAL_K_VALUES,
        "modes":     all_results,
    }, f, indent=2)
print(f"\n  Saved: {out_path}")
