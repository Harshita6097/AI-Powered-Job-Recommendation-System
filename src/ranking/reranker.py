"""
src/ranking/reranker.py

Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Takes the top-N RRF candidates and rescores each (query, job_text) pair.
The cross-encoder sees both query and document together — much richer signal
than the independent BM25/dense scores.

Public API
----------
rerank(query_text, candidates, df_lookup, top_k)  -> list[dict]
"""

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANKER_TOP_K, RERANKER_BATCH_SIZE
from src.retrieval.dense_retriever import _job_text

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(
    query_text: str,
    candidates: list[dict],
    df_lookup: dict,
    top_k: int = RERANKER_TOP_K,
) -> list[dict]:
    """
    Rerank RRF candidates with a cross-encoder.

    Parameters
    ----------
    query_text : the user's query string (free text or skill list).
    candidates : RRF output — list of dicts with at least {"job_id": str}.
    df_lookup  : dict keyed by job_id → row dict (needs title_clean,
                 granular_skills, industry, experience_level).
    top_k      : number of results to return after reranking.

    Returns
    -------
    list of dicts with original candidate fields + {"reranker_score": float,
    "rank": int}, sorted by reranker_score descending.
    """
    model = _get_model()

    # build (query, doc) pairs — skip candidates with no lookup entry
    pairs, valid = [], []
    for c in candidates:
        row = df_lookup.get(c["job_id"])
        if row is None:
            continue
        pairs.append((query_text, _job_text(row)))
        valid.append(c)

    if not pairs:
        return []

    scores = model.predict(pairs, batch_size=RERANKER_BATCH_SIZE)

    ranked = sorted(
        zip(scores, valid),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    return [
        {**c, "reranker_score": round(float(score), 6), "rank": rank + 1}
        for rank, (score, c) in enumerate(ranked)
    ]
