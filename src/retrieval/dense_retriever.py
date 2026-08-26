"""
src/retrieval/dense_retriever.py

Dense retrieval using sentence-transformers + FAISS (cosine similarity via
L2-normalised inner product).

Corpus text per job:
    "{title} | {granular_skills} | {industry} | {experience_level}"

Public API
----------
build_dense_index(df)  -> (faiss.Index, np.ndarray, list[str])
query_dense(index, job_ids, query_text, top_k)  -> list[dict]
encode_texts(texts)    -> np.ndarray   (L2-normalised, float32)
save_index(...)  /  load_index(...)
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL, EMBED_BATCH_SIZE, DENSE_TOP_K,
    FAISS_INDEX_BIN, EMBEDDINGS_NPY, JOB_ID_MAP_JSON, EMBED_CONFIG_JSON,
)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ── text builder ───────────────────────────────────────────────────────────────

def _job_text(row) -> str:
    skills = row["granular_skills"] if isinstance(row["granular_skills"], str) else ""
    skills = skills.replace("|", ", ")
    industry = row["industry"] if isinstance(row["industry"], str) else ""
    exp = row["experience_level"] if isinstance(row["experience_level"], str) else ""
    return f"{row['title_clean']} | {skills} | {industry} | {exp}"


# ── encode ─────────────────────────────────────────────────────────────────────

def encode_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE,
                 show_progress: bool = False) -> np.ndarray:
    """Encode texts → L2-normalised float32 embeddings."""
    model = _get_model()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 norm → inner product = cosine
    )
    return embs.astype(np.float32)


# ── build ──────────────────────────────────────────────────────────────────────

def build_dense_index(df, show_progress: bool = True) -> tuple:
    """
    Encode all jobs and build a FAISS IndexFlatIP index.

    Returns
    -------
    index    : faiss.IndexFlatIP
    embeddings : np.ndarray  shape (N, dim)
    job_ids  : list[str]
    """
    texts = [_job_text(row) for _, row in df.iterrows()]
    job_ids = df["job_id"].astype(str).tolist()

    print(f"  Encoding {len(texts):,} texts with {EMBED_MODEL} …")
    embeddings = encode_texts(texts, show_progress=show_progress)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings, job_ids


# ── query ──────────────────────────────────────────────────────────────────────

def query_dense(index: faiss.IndexFlatIP, job_ids: list[str],
                query_text: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    """
    Encode query_text and search FAISS index.

    Returns list of {"job_id": str, "dense_score": float, "rank": int}
    """
    q_emb = encode_texts([query_text])          # (1, dim)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue
        results.append({"job_id": job_ids[idx], "dense_score": float(score), "rank": rank + 1})
    return results


# ── save / load ────────────────────────────────────────────────────────────────

def save_index(index: faiss.IndexFlatIP, embeddings: np.ndarray,
               job_ids: list[str]) -> None:
    FAISS_INDEX_BIN.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_BIN))
    np.save(str(EMBEDDINGS_NPY), embeddings)

    with open(JOB_ID_MAP_JSON, "w") as f:
        json.dump(job_ids, f)

    dim = embeddings.shape[1]
    with open(EMBED_CONFIG_JSON, "w") as f:
        json.dump({"model": EMBED_MODEL, "dim": dim, "n_jobs": len(job_ids)}, f)


def load_index() -> tuple:
    """Returns (faiss.Index, embeddings np.ndarray, job_ids list[str])"""
    index = faiss.read_index(str(FAISS_INDEX_BIN))
    embeddings = np.load(str(EMBEDDINGS_NPY))
    with open(JOB_ID_MAP_JSON) as f:
        job_ids = json.load(f)
    return index, embeddings, job_ids
