# AI-Powered Job Recommendation System

A hybrid job recommendation system built on the LinkedIn Job Postings dataset (123,842 jobs).  
Users type skills or upload a PDF resume and receive ranked recommendations with skill gap analysis.

The system supports four retrieval modes switchable via a single config line:

| Mode | Pipeline |
|---|---|
| `tfidf` | TF-IDF cosine similarity (legacy baseline) |
| `bm25` | BM25 exact skill token matching |
| `dense` | Sentence-transformer semantic search (FAISS) |
| `hybrid` | BM25 + Dense → RRF fusion → Cross-encoder reranker |

---

## Project Structure

```
├── linkedin_dataset/           # Raw data (local only — not tracked in git)
│   ├── postings.csv            #   123,849 job postings (~493 MB)
│   ├── jobs/                   #   job_skills, salaries, job_industries, benefits
│   ├── companies/              #   companies.csv
│   └── mappings/               #   skills.csv, industries.csv
│
├── config.py                   # Central config — all paths, modes, hyperparameters
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py          #   Title / location / experience normalisation
│   │   └── skill_extractor.py  #   300+ granular skills, 100+ aliases, regex pass
│   ├── retrieval/
│   │   ├── bm25_retriever.py   #   BM25Okapi index builder + query
│   │   ├── dense_retriever.py  #   Sentence-transformer encoder + FAISS IndexFlatIP
│   │   └── rrf.py              #   Reciprocal Rank Fusion
│   ├── ranking/
│   │   └── reranker.py         #   Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── inference/
│   │   └── engine.py           #   Single recommend() entry point — routes by mode
│   └── evaluation/
│       └── evaluator.py        #   Hit Rate / Precision / NDCG / Mean Score metrics
│
├── build_granular.py           # Offline: build jobs_granular.csv (~38 min)
├── build_bm25.py               # Offline: build BM25 index (~6 s)
├── build_dense.py              # Offline: build FAISS index (~11 min on CPU)
├── evaluate.py                 # Benchmark all 4 modes on 500 queries
│
├── data/processed/             # Generated artifacts (local only — not tracked in git)
│   ├── jobs_features.csv       #   123,842 × 60  (legacy TF-IDF features)
│   ├── tfidf_matrix.npz        #   Sparse matrix 123,842 × 8,000
│   ├── tfidf_vectorizer.pkl
│   ├── skill_binarizer.pkl
│   ├── feature_metadata.json
│   ├── model_index.pkl
│   ├── jobs_granular.csv       #   123,842 × 21  (granular skill features)
│   └── eval_results_all_modes.json
│
├── data/indexes/               # Generated indexes (local only — not tracked in git)
│   ├── bm25_index.pkl          #   14 MB
│   ├── faiss_index.bin         #   190 MB
│   ├── embeddings.npy          #   190 MB
│   ├── job_id_map.json
│   └── embed_config.json
│
├── backend/
│   ├── main.py                 #   FastAPI app — 4 endpoints, routes by RETRIEVAL_MODE
│   ├── recommender.py          #   Legacy TF-IDF inference (150+ skill mappings)
│   ├── schemas.py              #   Pydantic models (includes retrieval_mode field)
│   └── logger.py               #   Rotating file + console logging
│
├── frontend/
│   └── index.html              #   Single-page UI — pipeline badge, skill chips, score bar
│
├── eda.py                      # 17-step EDA, saves plots to eda_outputs/
├── feature_engineering.py      # Legacy TF-IDF feature pipeline
├── train.py                    # Legacy TF-IDF training + evaluation
├── requirements.txt
└── README.md
```

---

## Architecture

### Hybrid Pipeline (recommended)

```
User skills / Resume PDF
        │
        ▼
  Skill Extractor          300+ granular skills, 100+ aliases, single regex pass
        │
   ┌────┴────┐
   ▼         ▼
 BM25       Dense          BM25: exact token match on granular skills + title
 top-100   top-100         Dense: all-MiniLM-L6-v2 → FAISS IndexFlatIP (cosine)
   │         │
   └────┬────┘
        ▼
   RRF Fusion              1/(k+rank) across both lists, k=60, pool=200
        │
        ▼
  Cross-Encoder            ms-marco-MiniLM-L-6-v2 rescores top-20 (query, job) pairs
        │
        ▼
   Top-N Results           + skill gap analysis on granular skills
```

### Why this design?

- **No user interaction data** — collaborative filtering is impossible. Content-based retrieval is the only valid approach.
- **BM25 alone** is strong for exact skill matching but misses semantic variants (`ML` vs `machine learning`).
- **Dense alone** captures semantics but can miss exact skill tokens.
- **RRF** combines both candidate pools without requiring score normalisation.
- **Cross-encoder** sees the full (query, document) pair — far richer signal than independent scores.

---

## Evaluation Results

500 test queries, 123,842 job pool, seed=42.  
Query = job's own granular skills. Relevance = result shares ≥1 skill with query.

### Hit Rate (did the original job appear in top-k?)

| Mode | @5 | @10 | @20 |
|---|---|---|---|
| TF-IDF | 3.6% | 4.6% | 7.0% |
| Dense | 14.8% | 19.8% | 22.2% |
| Hybrid | 33.0% | 37.8% | 40.8% |
| **BM25** | **35.4%** | **40.6%** | **45.8%** |

### Precision (fraction of top-k sharing ≥1 skill with query)

| Mode | @5 | @10 | @20 |
|---|---|---|---|
| TF-IDF | ~0% | ~0% | ~0% |
| Dense | 98.6% | 98.4% | 97.9% |
| Hybrid | **99.8%** | **99.8%** | **99.6%** |
| BM25 | 100% | 100% | 100% |

### NDCG (position-weighted relevance, 1.0 = perfect)

| Mode | @5 | @10 | @20 |
|---|---|---|---|
| TF-IDF | ~0 | ~0 | ~0 |
| Dense | 0.993 | 0.993 | 0.993 |
| Hybrid | **0.999** | **0.999** | **0.999** |
| BM25 | 1.000 | 1.000 | 1.000 |

### Notes

- **TF-IDF precision ≈ 0%** in the new eval because the test queries use granular skills (`python`, `docker`) but TF-IDF was trained on broad categories (`Information Technology`). TF-IDF still works correctly when queried with its own broad-category vocabulary (original eval: Precision@10 = 99.9%).
- **BM25 Hit@10 = 40.6%** on 123,842 jobs = **5,000× better than random (0.008%)**.
- **Hybrid NDCG = 0.999** — the cross-encoder produces the best-ordered results even when it doesn't always surface the exact original job.
- BM25 wins on raw hit rate; Hybrid wins on ranking quality (NDCG) and is the recommended production mode.

---

## Dataset

LinkedIn Job Postings — 123,849 postings, 35 broad skill categories, 422 industries.  
Stored locally, excluded from git (`postings.csv` is ~493 MB).

---

## Running Locally

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Build indexes (one-time, ~50 min total on CPU)

```bash
# Legacy TF-IDF artifacts (needed for tfidf mode)
python feature_engineering.py   # ~5 min
python train.py                  # ~2 min

# New hybrid pipeline artifacts
python build_granular.py         # ~38 min  →  data/processed/jobs_granular.csv
python build_bm25.py             # ~6 s     →  data/indexes/bm25_index.pkl
python build_dense.py            # ~11 min  →  data/indexes/faiss_index.bin
```

### Step 3 — Choose retrieval mode

Edit `config.py`:

```python
RETRIEVAL_MODE = "hybrid"   # "tfidf" | "bm25" | "dense" | "hybrid"
```

### Step 4 — Start the server

```bash
python -m uvicorn backend.main:app --reload
```

### Step 5 — Open the app

```
http://localhost:8000
```

Swagger UI (API docs):
```
http://localhost:8000/docs
```

### Step 6 — Run evaluation (optional)

```bash
python evaluate.py   # benchmarks all 4 modes, ~10 min total
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/recommend` | Recommend jobs from skill list |
| POST | `/resume` | Recommend jobs from uploaded PDF resume |
| POST | `/skill-gap` | Skill gap analysis for a specific job |

### POST `/recommend`

```json
{
  "skills": ["python", "machine learning", "sql"],
  "experience_level": "Mid-Senior",
  "top_n": 10,
  "filter_exp": false,
  "filter_remote": false
}
```

Response includes `retrieval_mode` field indicating which pipeline served the request.

### POST `/resume`

Multipart form upload — send a PDF file.  
In hybrid/bm25/dense modes, skills are extracted via the 300+ skill dictionary.  
In tfidf mode, skills are matched against the 150+ SKILL_MAP keyword list.

### POST `/skill-gap`

```json
{
  "user_skills": ["python", "sql"],
  "job_skills": "python|machine learning|sql|docker|kubernetes"
}
```

Returns `matched_skills`, `missing_skills`, `match_pct`.

---

## Build Stages

| Stage | File | Description |
|---|---|---|
| 1 | `src/preprocessing/`, `build_granular.py` | Config, cleaner, granular skill extractor (300+ skills), `jobs_granular.csv` |
| 2 | `src/retrieval/bm25_retriever.py`, `build_bm25.py` | BM25 index (6s build, 14 MB) |
| 3 | `src/retrieval/dense_retriever.py`, `build_dense.py` | FAISS dense index (11 min, 190 MB) |
| 4 | `src/retrieval/rrf.py` | Reciprocal Rank Fusion |
| 5 | `src/ranking/reranker.py` | Cross-encoder reranker |
| 6 | `src/inference/engine.py` | Hybrid inference engine — routes by `RETRIEVAL_MODE` |
| 7 | `backend/main.py` | FastAPI wired to engine |
| 8 | `frontend/index.html` | Pipeline badge, skill chips, score normalisation |
| 9 | `evaluate.py`, `src/evaluation/evaluator.py` | Multi-mode benchmark |
| 10 | `README.md` | This document |

---

## Tech Stack

- **Retrieval:** rank-bm25, sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Data & ML:** Python, Pandas, Scikit-learn, NumPy, SciPy
- **Backend:** FastAPI, Uvicorn, PyMuPDF, Pydantic
- **Frontend:** HTML / CSS / JavaScript (no framework)
- **Visualisation:** Matplotlib, Seaborn, WordCloud
- **Logging:** Python logging — rotating file handler (5 MB × 3 backups)
