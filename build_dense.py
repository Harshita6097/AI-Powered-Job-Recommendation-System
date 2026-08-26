"""
build_dense.py — Offline script to build and save the FAISS dense index.

Usage:
    python build_dense.py

Input  : data/processed/jobs_granular.csv
Output : data/indexes/faiss_index.bin   (~190 MB)
         data/indexes/embeddings.npy    (~190 MB)
         data/indexes/job_id_map.json
         data/indexes/embed_config.json

Runtime: ~25-40 min on CPU (123,842 jobs × 384-dim embeddings).
"""

import time
import pandas as pd

from config import GRANULAR_JOBS_CSV, FAISS_INDEX_BIN, EMBEDDINGS_NPY
from src.retrieval.dense_retriever import build_dense_index, save_index, query_dense

print("Loading jobs_granular.csv …")
df = pd.read_csv(GRANULAR_JOBS_CSV, dtype={"job_id": str})
print(f"  {len(df):,} jobs loaded")

t0 = time.time()
index, embeddings, job_ids = build_dense_index(df, show_progress=True)
elapsed = time.time() - t0
print(f"  Encoding + indexing done in {elapsed/60:.1f} min")
print(f"  Embeddings shape: {embeddings.shape}")

print("Saving index …")
save_index(index, embeddings, job_ids)
print(f"  faiss_index.bin : {FAISS_INDEX_BIN.stat().st_size / 1e6:.1f} MB")
print(f"  embeddings.npy  : {EMBEDDINGS_NPY.stat().st_size / 1e6:.1f} MB")

# ── smoke test ────────────────────────────────────────────────────────────────
print("\nSmoke test — query: 'python machine learning sql data scientist'")
results = query_dense(index, job_ids, "python machine learning sql data scientist", top_k=5)
for r in results:
    row = df[df["job_id"] == r["job_id"]].iloc[0]
    print(f"  [{r['rank']}] score={r['dense_score']:.4f}  {row['title_clean'][:50]}  skills={row['granular_skills'][:60]}")

print("\nDense index build complete.")
