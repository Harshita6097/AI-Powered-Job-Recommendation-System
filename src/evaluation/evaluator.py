"""
src/evaluation/evaluator.py

Evaluation metrics for retrieval systems.

Metrics
-------
- Hit Rate @ k   : did the original job appear in top-k?
- Precision @ k  : fraction of top-k sharing >= 1 skill with query
- NDCG @ k       : position-weighted relevance
- Mean Score @ k : average raw match score of top-k results
"""

import numpy as np


def dcg_at_k(relevances: list, k: int) -> float:
    r = np.array(relevances[:k], dtype=float)
    if len(r) == 0:
        return 0.0
    return float(np.sum(r / np.log2(np.arange(1, len(r) + 1) + 1)))


def ndcg_at_k(relevances: list, k: int) -> float:
    dcg  = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    results: list[dict],
    true_job_id: str,
    query_skill_set: set,
    skill_field: str,
    k_values: list[int],
) -> dict:
    """
    Compute hit_rate, precision, ndcg, mean_score at each k.

    Parameters
    ----------
    results       : list of result dicts, each with "job_id", skill_field, "match_score"
    true_job_id   : the ground-truth job id for hit rate
    query_skill_set : set of skill strings used as query
    skill_field   : key in result dict containing pipe-separated skills
    k_values      : list of k values to evaluate at
    """
    max_k = max(k_values)
    top_results = results[:max_k]

    out = {}
    for k in k_values:
        top_k = top_results[:k]
        result_ids = [r["job_id"] for r in top_k]

        # hit rate
        hit = int(true_job_id in result_ids)

        # per-result relevance (shares >= 1 skill with query)
        relevances = []
        for r in top_k:
            skills_str = r.get(skill_field, "") or ""
            job_skills = set(s.strip() for s in skills_str.split("|") if s.strip())
            relevances.append(1 if job_skills & query_skill_set else 0)

        precision  = sum(relevances) / k
        ndcg       = ndcg_at_k(relevances, k)
        mean_score = float(np.mean([r["match_score"] for r in top_k])) if top_k else 0.0

        out[k] = {
            "hit_rate":   hit,
            "precision":  precision,
            "ndcg":       ndcg,
            "mean_score": mean_score,
        }
    return out


def aggregate(per_query: list[dict], k_values: list[int]) -> dict:
    """Average per-query metrics across all queries."""
    agg = {}
    for k in k_values:
        rows = [q[k] for q in per_query if k in q]
        agg[k] = {
            metric: round(float(np.mean([r[metric] for r in rows])), 4)
            for metric in ["hit_rate", "precision", "ndcg", "mean_score"]
        }
    return agg
