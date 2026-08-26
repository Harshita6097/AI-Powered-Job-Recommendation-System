"""
src/inference/engine.py

Hybrid inference engine — single entry point for job recommendations.

Routes by RETRIEVAL_MODE (config.py):
  "tfidf"  -> legacy TF-IDF cosine similarity (backend/recommender.py)
  "bm25"   -> BM25 only, no reranker
  "dense"  -> dense only, no reranker
  "hybrid" -> BM25 + dense -> RRF -> cross-encoder reranker  (full pipeline)

All artifacts are lazy-loaded once and reused across calls.

Public API
----------
recommend(skills, experience_level, top_n, filter_exp, filter_remote) -> dict
skill_gap(user_skills, job_skills_str) -> dict
load_all()   # call at app startup to pre-warm everything
"""

import time
import pandas as pd
from backend.logger import get_logger
import config
from config import (
    GRANULAR_JOBS_CSV,
    BM25_TOP_K, DENSE_TOP_K, RRF_CANDIDATE_POOL, RERANKER_TOP_K,
)
from src.retrieval.bm25_retriever import (
    load_index as _load_bm25, tokenise_query, query_bm25,
)
from src.retrieval.dense_retriever import (
    load_index as _load_dense, query_dense,
)
from src.retrieval.rrf import fuse
from src.ranking.reranker import rerank
from src.preprocessing.skill_extractor import extract_skills

log = get_logger("engine")

# ── artifact state ─────────────────────────────────────────────────────────────
_df: pd.DataFrame | None = None          # jobs_granular
_lookup: dict | None = None              # job_id -> row dict
_bm25 = None
_bm25_ids: list[str] | None = None
_faiss_idx = None
_dense_ids: list[str] | None = None


def _load_granular():
    global _df, _lookup
    if _df is None:
        log.info("Loading jobs_granular.csv ...")
        _df = pd.read_csv(GRANULAR_JOBS_CSV, dtype={"job_id": str})
        _lookup = _df.set_index("job_id").to_dict("index")
        log.info("Loaded %d jobs", len(_df))


def _load_bm25_artifacts():
    global _bm25, _bm25_ids
    if _bm25 is None:
        log.info("Loading BM25 index ...")
        _bm25, _bm25_ids = _load_bm25()
        log.info("BM25 index loaded (%d docs)", len(_bm25_ids))


def _load_dense_artifacts():
    global _faiss_idx, _dense_ids
    if _faiss_idx is None:
        log.info("Loading FAISS index ...")
        _faiss_idx, _, _dense_ids = _load_dense()
        log.info("FAISS index loaded (%d docs)", len(_dense_ids))


def load_all():
    """Pre-warm all artifacts for the configured RETRIEVAL_MODE."""
    _load_granular()
    if config.RETRIEVAL_MODE in ("bm25", "hybrid"):
        _load_bm25_artifacts()
    if config.RETRIEVAL_MODE in ("dense", "hybrid"):
        _load_dense_artifacts()
    log.info("Engine ready -- mode=%s", config.RETRIEVAL_MODE)


# ── query text builder ─────────────────────────────────────────────────────────

def _build_query_text(skills: list[str]) -> str:
    """Free-text query for dense encoder and cross-encoder."""
    return ", ".join(skills)


# ── result formatter ───────────────────────────────────────────────────────────

def _format_results(job_ids: list[str], scores: list[float],
                    score_key: str) -> list[dict]:
    rows = []
    for jid, score in zip(job_ids, scores):
        row = _lookup.get(jid)
        if row is None:
            continue
        rows.append({
            "job_id":           jid,
            "title":            row.get("title_clean", ""),
            "experience_level": row.get("experience_level", ""),
            "industry":         row.get("industry", ""),
            "skills":           row.get("granular_skills", ""),
            "broad_skills":     row.get("broad_skills_str", ""),
            "salary_mid":       row.get("salary_mid") if row.get("has_salary") else None,
            "work_type":        row.get("work_type", ""),
            "is_remote":        bool(row.get("is_remote", False)),
            "state":            row.get("state"),
            score_key:          round(float(score), 4),
        })
    return rows


# ── retrieval modes ────────────────────────────────────────────────────────────

def _run_bm25(query_tokens: list[str], top_n: int) -> list[dict]:
    results = query_bm25(_bm25, _bm25_ids, query_tokens, top_k=max(top_n, BM25_TOP_K))
    job_ids = [r["job_id"] for r in results[:top_n]]
    scores  = [r["bm25_score"] for r in results[:top_n]]
    return _format_results(job_ids, scores, "match_score")


def _run_dense(query_text: str, top_n: int) -> list[dict]:
    results = query_dense(_faiss_idx, _dense_ids, query_text, top_k=max(top_n, DENSE_TOP_K))
    job_ids = [r["job_id"] for r in results[:top_n]]
    scores  = [r["dense_score"] for r in results[:top_n]]
    return _format_results(job_ids, scores, "match_score")


def _run_hybrid(query_tokens: list[str], query_text: str, top_n: int) -> list[dict]:
    bm25_results  = query_bm25(_bm25, _bm25_ids, query_tokens, top_k=BM25_TOP_K)
    dense_results = query_dense(_faiss_idx, _dense_ids, query_text, top_k=DENSE_TOP_K)

    rrf_pool = fuse(
        [bm25_results, dense_results],
        list_names=["bm25", "dense"],
        top_n=RRF_CANDIDATE_POOL,
    )

    # reranker takes top min(pool, RERANKER_TOP_K) candidates
    rerank_pool = rrf_pool[:RERANKER_TOP_K]
    final = rerank(query_text, rerank_pool, _lookup, top_k=top_n)

    job_ids = [r["job_id"] for r in final]
    scores  = [r["reranker_score"] for r in final]
    return _format_results(job_ids, scores, "match_score")


# ── public API ─────────────────────────────────────────────────────────────────

def recommend(
    skills: list[str],
    experience_level: str | None = None,
    top_n: int = 10,
    filter_exp: bool = False,
    filter_remote: bool = False,
) -> dict:
    """
    Returns
    -------
    {
      "results":              list[dict],
      "query_skills_raw":     list[str],
      "query_skills_mapped":  list[str],   # granular skills found in SKILL_DICT
      "unmapped_skills":      list[str],
      "retrieval_mode":       str,
      "latency_ms":           float,
    }
    """
    t0 = time.time()
    _load_granular()

    # extract granular skills from the user's input text
    query_text_raw = ", ".join(skills)
    granular  = list(extract_skills(query_text_raw))   # returns tuple of matched skill strings
    unmapped  = [s for s in skills if s.lower() not in {g.lower() for g in granular}]

    # fall back to raw skills if extractor found nothing
    query_skills = granular if granular else skills

    query_tokens = tokenise_query(", ".join(query_skills))
    query_text   = _build_query_text(query_skills)

    log.info("recommend | mode=%s raw=%s granular=%s exp=%s",
             config.RETRIEVAL_MODE, skills, query_skills, experience_level)

    if config.RETRIEVAL_MODE == "bm25":
        _load_bm25_artifacts()
        results = _run_bm25(query_tokens, top_n)

    elif config.RETRIEVAL_MODE == "dense":
        _load_dense_artifacts()
        results = _run_dense(query_text, top_n)

    elif config.RETRIEVAL_MODE == "hybrid":
        _load_bm25_artifacts()
        _load_dense_artifacts()
        results = _run_hybrid(query_tokens, query_text, top_n)

    else:
        # "tfidf" — delegate to legacy recommender
        from backend.recommender import recommend as legacy_recommend
        df_results, mapped, unmap = legacy_recommend(
            skills, experience_level, top_n, filter_exp, filter_remote
        )
        return {
            "results":             df_results.to_dict("records"),
            "query_skills_raw":    skills,
            "query_skills_mapped": mapped,
            "unmapped_skills":     unmap,
            "retrieval_mode":      "tfidf",
            "latency_ms":          round((time.time() - t0) * 1000, 1),
        }

    # apply post-retrieval filters (bm25 / dense / hybrid paths)
    if filter_exp and experience_level:
        results = [r for r in results if r.get("experience_level") == experience_level]
    if filter_remote:
        results = [r for r in results if r.get("is_remote")]

    latency = round((time.time() - t0) * 1000, 1)
    log.info("recommend | returned %d results in %sms", len(results), latency)

    return {
        "results":             results[:top_n],
        "query_skills_raw":    skills,
        "query_skills_mapped": query_skills,
        "unmapped_skills":     unmapped,
        "retrieval_mode":      config.RETRIEVAL_MODE,
        "latency_ms":          latency,
    }


def skill_gap(user_skills: list[str], job_skills_str: str) -> dict:
    """
    Granular skill gap analysis.
    job_skills_str: pipe-separated granular skills from a job result.
    """
    extracted   = list(extract_skills(", ".join(user_skills)))
    user_set    = set(s.lower() for s in extracted) if extracted else {s.lower() for s in user_skills}
    job_skills  = [s.strip() for s in job_skills_str.split("|") if s.strip()]

    matched = [s for s in job_skills if s.lower() in user_set]
    missing = [s for s in job_skills if s.lower() not in user_set]

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_pct":      round(len(matched) / max(len(job_skills), 1) * 100, 1),
    }
