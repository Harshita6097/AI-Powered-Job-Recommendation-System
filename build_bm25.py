"""
build_bm25.py — Offline script to build and save the BM25 index.

Usage:
    python build_bm25.py

Input  : data/processed/jobs_granular.csv
Output : data/indexes/bm25_index.pkl

Runtime: ~2-4 minutes on 123,842 jobs.
"""

import time
import pandas as pd

from config import GRANULAR_JOBS_CSV, BM25_INDEX_PKL
from src.retrieval.bm25_retriever import build_bm25_index, save_index, query_bm25, tokenise_query

print("Loading jobs_granular.csv …")
df = pd.read_csv(GRANULAR_JOBS_CSV, dtype={"job_id": str})
print(f"  {len(df):,} jobs loaded")

print("Building BM25 index …")
t0 = time.time()
bm25, job_ids = build_bm25_index(df)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

print(f"Saving to {BM25_INDEX_PKL} …")
save_index(bm25, job_ids)
print(f"  Saved — {BM25_INDEX_PKL.stat().st_size / 1e6:.1f} MB")

# ── smoke test ────────────────────────────────────────────────────────────────
print("\nSmoke test — query: 'python machine_learning sql'")
tokens = tokenise_query("python, machine learning, sql")
results = query_bm25(bm25, job_ids, tokens, top_k=5)
for r in results:
    row = df[df["job_id"] == r["job_id"]].iloc[0]
    print(f"  [{r['rank']}] score={r['bm25_score']:.3f}  {row['title_clean'][:50]}  skills={row['granular_skills'][:60]}")

print("\nBM25 index build complete.")
