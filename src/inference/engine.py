"""
src/inference/engine.py

Hybrid inference engine — single entry point for job recommendations.

Routes by RETRIEVAL_MODE (config.py / .env):
  "tfidf"  -> legacy TF-IDF cosine similarity (backend/recommender.py)
  "bm25"   -> BM25 only
  "dense"  -> dense FAISS only
  "hybrid" -> BM25 + dense -> RRF -> cross-encoder reranker

Features
--------
- Pre-retrieval filtering: exp level + remote filter applied before scoring
  so top_n results are always returned even with strict filters
- MMR diversity reranking: reduces redundant results in final list
- LRU cache: repeated identical queries skip the full pipeline
- Lazy artifact loading: indexes loaded once, reused across requests
"""

import time
import functools
import pandas as pd
from backend.logger import get_logger
import config
from config import (
    GRANULAR_JOBS_CSV,
    BM25_TOP_K, DENSE_TOP_K, RRF_CANDIDATE_POOL, RERANKER_TOP_K,
    MMR_LAMBDA, MMR_FINAL_K,
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
from src.utils.filters import build_allowed_ids, apply_allowed
from src.utils.mmr import mmr_rerank

log = get_logger("engine")

# ── artifact state ─────────────────────────────────────────────────────────────
_df: pd.DataFrame | None = None
_lookup: dict | None = None
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


# ── result formatter ───────────────────────────────────────────────────────────

def _format_results(job_ids: list[str], scores: list[float],
                    score_key: str) -> list[dict]:
    rows = []
    for jid, score in zip(job_ids, scores):
        row = _lookup.get(jid)
        if row is None:
            continue
        state = row.get("state")
        salary = row.get("salary_mid") if row.get("has_salary") else None
        rows.append({
            "job_id":           jid,
            "title":            row.get("title_clean", ""),
            "experience_level": row.get("experience_level", ""),
            "industry":         row.get("industry", ""),
            "skills":           row.get("granular_skills", ""),
            "broad_skills":     row.get("broad_skills_str", ""),
            "salary_mid":       float(salary) if salary == salary and salary is not None else None,
            "work_type":        row.get("work_type", ""),
            "is_remote":        bool(row.get("is_remote", False)),
            "state":            state if isinstance(state, str) else None,
            score_key:          round(float(score), 4),
        })
    return rows


# ── retrieval modes ────────────────────────────────────────────────────────────

def _run_bm25(query_tokens: list[str], allowed: set | None,
              top_n: int) -> list[dict]:
    # fetch more candidates when filtering to ensure top_n survive
    fetch_k = BM25_TOP_K if allowed is None else min(len(allowed), BM25_TOP_K * 3)
    results = query_bm25(_bm25, _bm25_ids, query_tokens, top_k=fetch_k)
    results = apply_allowed(results, allowed)
    job_ids = [r["job_id"] for r in results[:top_n]]
    scores  = [r["bm25_score"] for r in results[:top_n]]
    return _format_results(job_ids, scores, "match_score")


def _run_dense(query_text: str, allowed: set | None,
               top_n: int) -> list[dict]:
    fetch_k = DENSE_TOP_K if allowed is None else min(len(allowed), DENSE_TOP_K * 3)
    results = query_dense(_faiss_idx, _dense_ids, query_text, top_k=fetch_k)
    results = apply_allowed(results, allowed)
    job_ids = [r["job_id"] for r in results[:top_n]]
    scores  = [r["dense_score"] for r in results[:top_n]]
    return _format_results(job_ids, scores, "match_score")


def _run_hybrid(query_tokens: list[str], query_text: str,
                allowed: set | None, top_n: int) -> list[dict]:
    fetch_k = BM25_TOP_K if allowed is None else min(len(allowed), BM25_TOP_K * 3)
    bm25_results  = query_bm25(_bm25, _bm25_ids, query_tokens, top_k=fetch_k)
    dense_results = query_dense(_faiss_idx, _dense_ids, query_text, top_k=fetch_k)

    # apply filter before RRF so the fused pool only contains valid candidates
    bm25_results  = apply_allowed(bm25_results, allowed)
    dense_results = apply_allowed(dense_results, allowed)

    # re-rank within filtered lists before fusing
    for i, r in enumerate(bm25_results):
        r["rank"] = i + 1
    for i, r in enumerate(dense_results):
        r["rank"] = i + 1

    rrf_pool    = fuse([bm25_results, dense_results],
                       list_names=["bm25", "dense"], top_n=RRF_CANDIDATE_POOL)
    rerank_pool = rrf_pool[:RERANKER_TOP_K]
    final       = rerank(query_text, rerank_pool, _lookup, top_k=top_n)

    job_ids = [r["job_id"] for r in final]
    scores  = [r["reranker_score"] for r in final]
    return _format_results(job_ids, scores, "match_score")


# ── LRU cache ──────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def _cached_pipeline(skills_key: tuple, experience_level: str | None,
                     top_n: int, filter_exp: bool, filter_remote: bool,
                     salary_min: float | None, salary_max: float | None,
                     mode: str) -> tuple:
    """
    Cached inner pipeline call. Returns a tuple (results_tuple, mapped, unmapped).
    All args must be hashable — skills passed as sorted tuple.
    """
    result = _run_pipeline(
        skills=list(skills_key),
        experience_level=experience_level,
        top_n=top_n,
        filter_exp=filter_exp,
        filter_remote=filter_remote,
        salary_min=salary_min,
        salary_max=salary_max,
    )
    # freeze results for caching
    return (tuple(result["results"]), tuple(result["query_skills_mapped"]),
            tuple(result["unmapped_skills"]))


def cache_info() -> str:
    info = _cached_pipeline.cache_info()
    return f"hits={info.hits} misses={info.misses} size={info.currsize}/256"


def cache_clear() -> None:
    _cached_pipeline.cache_clear()


# ── core pipeline (uncached) ───────────────────────────────────────────────────

def _run_pipeline(skills: list[str], experience_level: str | None,
                  top_n: int, filter_exp: bool, filter_remote: bool,
                  salary_min: float | None = None,
                  salary_max: float | None = None) -> dict:
    """Run the retrieval pipeline for the current RETRIEVAL_MODE."""
    granular = list(extract_skills(", ".join(skills)))
    unmapped = [s for s in skills if s.lower() not in {g.lower() for g in granular}]
    query_skills = granular if granular else skills

    query_tokens = tokenise_query(", ".join(query_skills))
    query_text   = ", ".join(query_skills)

    # build pre-filter set (None = no filter = all jobs allowed)
    allowed = build_allowed_ids(_lookup, filter_exp, experience_level, filter_remote,
                                salary_min, salary_max)

    if config.RETRIEVAL_MODE == "bm25":
        _load_bm25_artifacts()
        results = _run_bm25(query_tokens, allowed, top_n)

    elif config.RETRIEVAL_MODE == "dense":
        _load_dense_artifacts()
        results = _run_dense(query_text, allowed, top_n)

    elif config.RETRIEVAL_MODE == "hybrid":
        _load_bm25_artifacts()
        _load_dense_artifacts()
        results = _run_hybrid(query_tokens, query_text, allowed, top_n)

    else:
        # tfidf — no pre-filtering, handled by legacy recommender
        from backend.recommender import recommend as legacy_recommend
        df_results, mapped, unmap = legacy_recommend(
            skills, experience_level, top_n, filter_exp, filter_remote
        )
        return {
            "results":             df_results.to_dict("records"),
            "query_skills_mapped": mapped,
            "unmapped_skills":     unmap,
        }

    # MMR diversity reranking on final results
    if MMR_LAMBDA < 1.0 and len(results) > 1:
        results = mmr_rerank(results, top_k=top_n, lam=MMR_LAMBDA)

    return {
        "results":             results[:top_n],
        "query_skills_mapped": query_skills,
        "unmapped_skills":     unmapped,
    }


# ── public API ─────────────────────────────────────────────────────────────────

def recommend(
    skills: list[str],
    experience_level: str | None = None,
    top_n: int = 10,
    filter_exp: bool = False,
    filter_remote: bool = False,
    salary_min: float | None = None,
    salary_max: float | None = None,
) -> dict:
    t0 = time.time()
    _load_granular()

    mode = config.RETRIEVAL_MODE
    log.info("recommend | mode=%s skills=%s exp=%s filter_exp=%s remote=%s salary=[%s,%s]",
             mode, skills, experience_level, filter_exp, filter_remote, salary_min, salary_max)

    # cache key: sorted skills tuple so order doesn't matter
    skills_key = tuple(sorted(s.lower() for s in skills))

    try:
        results_t, mapped_t, unmapped_t = _cached_pipeline(
            skills_key, experience_level, top_n, filter_exp, filter_remote,
            salary_min, salary_max, mode
        )
        results  = list(results_t)
        mapped   = list(mapped_t)
        unmapped = list(unmapped_t)
    except Exception as e:
        log.warning("Cache miss / pipeline error: %s — running uncached", e)
        out      = _run_pipeline(skills, experience_level, top_n, filter_exp, filter_remote,
                                 salary_min, salary_max)
        results  = out["results"]
        mapped   = out["query_skills_mapped"]
        unmapped = out["unmapped_skills"]

    latency = round((time.time() - t0) * 1000, 1)
    log.info("recommend | returned %d results in %sms | cache=%s",
             len(results), latency, cache_info())

    return {
        "results":             results,
        "query_skills_raw":    skills,
        "query_skills_mapped": mapped,
        "unmapped_skills":     unmapped,
        "retrieval_mode":      mode,
        "latency_ms":          latency,
    }


def skill_gap(user_skills: list[str], job_skills_str: str) -> dict:
    """Granular skill gap analysis."""
    extracted = list(extract_skills(", ".join(user_skills)))
    user_set  = set(s.lower() for s in extracted) if extracted else {s.lower() for s in user_skills}
    job_skills = [s.strip() for s in job_skills_str.split("|") if s.strip()]
    matched = [s for s in job_skills if s.lower() in user_set]
    missing = [s for s in job_skills if s.lower() not in user_set]
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_pct":      round(len(matched) / max(len(job_skills), 1) * 100, 1),
    }
