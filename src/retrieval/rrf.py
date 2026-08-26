"""
src/retrieval/rrf.py

Reciprocal Rank Fusion — merges ranked candidate lists from BM25 and dense
retrieval into a single fused ranking.

Formula:  rrf_score(d) = Σ  1 / (k + rank_i(d))
          where rank_i is 1-based position in list i, k is a smoothing constant.

Public API
----------
fuse(ranked_lists, k, top_n)  -> list[dict]

Each input list is a list of dicts with at least {"job_id": str, "rank": int}.
Output dicts: {"job_id": str, "rrf_score": float, "rank": int,
               "bm25_rank": int|None, "dense_rank": int|None}
"""

from config import RRF_K, RRF_CANDIDATE_POOL


def fuse(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    top_n: int = RRF_CANDIDATE_POOL,
    list_names: list[str] | None = None,
) -> list[dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of ranked result lists.
                   Each item must have "job_id" and "rank" (1-based).
    k            : RRF smoothing constant (default 60).
    top_n        : number of results to return.
    list_names   : optional names for each list (e.g. ["bm25", "dense"]).
                   Used to populate per-source rank fields in output.

    Returns
    -------
    list of dicts sorted by rrf_score descending, length <= top_n.
    """
    if list_names is None:
        list_names = [f"list_{i}" for i in range(len(ranked_lists))]

    scores: dict[str, float] = {}
    per_source: dict[str, dict[str, int]] = {name: {} for name in list_names}

    for name, ranked in zip(list_names, ranked_lists):
        for item in ranked:
            jid = item["job_id"]
            r = item["rank"]
            scores[jid] = scores.get(jid, 0.0) + 1.0 / (k + r)
            per_source[name][jid] = r

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for rank, (jid, rrf_score) in enumerate(fused, start=1):
        entry = {"job_id": jid, "rrf_score": round(rrf_score, 6), "rank": rank}
        for name in list_names:
            entry[f"{name}_rank"] = per_source[name].get(jid)
        results.append(entry)

    return results
