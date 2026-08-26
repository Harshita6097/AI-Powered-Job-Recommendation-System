"""
src/retrieval/bm25_retriever.py

BM25 retriever over jobs_granular.csv.

Corpus token field: granular_skills (pipe-separated) + title_clean words.
Each document = skill tokens + title tokens (title weighted ×2 by repetition).

Public API
----------
build_bm25_index(df)  -> (BM25Okapi, list[str])   # job_ids in corpus order
query_bm25(bm25, job_ids, query_tokens, top_k)  -> list[dict]
tokenise_query(text)  -> list[str]
"""

import re
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config import BM25_K1, BM25_B, BM25_TOP_K, BM25_INDEX_PKL


# ── tokeniser ─────────────────────────────────────────────────────────────────

def _skill_tokens(granular_skills: str) -> list[str]:
    """Split pipe-separated skill string into tokens."""
    if not granular_skills or (isinstance(granular_skills, float)):
        return []
    return [s.strip().lower().replace(" ", "_") for s in granular_skills.split("|") if s.strip()]


def _title_tokens(title: str) -> list[str]:
    if not title or (isinstance(title, float)):
        return []
    return re.findall(r"[a-z0-9]+", title.lower())


def _doc_tokens(granular_skills: str, title: str) -> list[str]:
    """Skills + title repeated twice (mild title boost)."""
    skills = _skill_tokens(granular_skills)
    title_toks = _title_tokens(title)
    return skills + title_toks + title_toks   # title ×2


def tokenise_query(text: str) -> list[str]:
    """
    Convert free-text query into BM25 tokens.
    Skill phrases (multi-word) are joined with underscore to match corpus tokens.
    Single words are kept as-is.
    """
    text = text.lower().strip()
    # split on pipe or comma first (structured skill input)
    parts = re.split(r"[|,]", text)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # multi-word skill → underscore token
        if " " in part:
            tokens.append(part.replace(" ", "_"))
        else:
            tokens.append(part)
    return tokens


# ── build ──────────────────────────────────────────────────────────────────────

def build_bm25_index(df) -> tuple:
    """
    Build BM25Okapi index from a jobs_granular DataFrame.

    Returns
    -------
    bm25     : BM25Okapi
    job_ids  : list[str]  — corpus-order job IDs (index → job_id)
    """
    corpus = [
        _doc_tokens(row["granular_skills"], row["title_clean"])
        for _, row in df.iterrows()
    ]
    job_ids = df["job_id"].astype(str).tolist()

    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    return bm25, job_ids


# ── query ──────────────────────────────────────────────────────────────────────

def query_bm25(bm25: BM25Okapi, job_ids: list[str], query_tokens: list[str],
               top_k: int = BM25_TOP_K) -> list[dict]:
    """
    Run BM25 query and return top_k results.

    Returns list of {"job_id": str, "bm25_score": float, "rank": int}
    """
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        {"job_id": job_ids[i], "bm25_score": float(scores[i]), "rank": rank + 1}
        for rank, i in enumerate(top_indices)
        if scores[i] > 0
    ]


# ── save / load ────────────────────────────────────────────────────────────────

def save_index(bm25: BM25Okapi, job_ids: list[str], path: Path = BM25_INDEX_PKL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "job_ids": job_ids}, f, protocol=5)


def load_index(path: Path = BM25_INDEX_PKL) -> tuple:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["job_ids"]
