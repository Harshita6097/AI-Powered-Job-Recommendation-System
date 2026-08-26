# AI-Powered Job Recommendation System

A content-based job recommendation system built on the LinkedIn Job Postings dataset.  
Users type their skills or upload a PDF resume and receive ranked job recommendations with skill gap analysis.

---

## Project Structure

```
├── linkedin_dataset/        # Raw data (local only — not tracked in git)
│   ├── postings.csv         #   123,849 job postings (~493MB)
│   ├── jobs/
│   │   ├── job_skills.csv
│   │   ├── salaries.csv
│   │   ├── job_industries.csv
│   │   └── benefits.csv
│   ├── companies/
│   │   └── companies.csv
│   └── mappings/
│       ├── skills.csv
│       └── industries.csv
│
├── eda.py                   # Exploratory Data Analysis — 17 analyses, saves plots to eda_outputs/
├── eda_outputs/             # EDA plots (local only — not tracked in git)
│
├── feature_engineering.py  # Full feature pipeline — produces data/processed/ artifacts
├── train.py                 # TF-IDF model + cosine similarity + evaluation
│
├── data/processed/          # Generated artifacts (local only — not tracked in git)
│   ├── jobs_features.csv    #   123,842 × 60 master feature table
│   ├── tfidf_matrix.npz     #   Sparse matrix (123,842 × 8,000)
│   ├── tfidf_vectorizer.pkl #   Fitted TF-IDF vectorizer
│   ├── skill_binarizer.pkl  #   Fitted MultiLabelBinarizer
│   ├── feature_metadata.json#   Skill categories, exp map, vocab size
│   ├── model_index.pkl      #   Model bundle (vectorizer + mlb + metadata)
│   └── eval_results.json    #   Evaluation metrics at k=5,10,20
│
├── backend/
│   ├── main.py              # FastAPI app — 4 endpoints
│   ├── recommender.py       # Skill mapping (150+ skills) + inference logic
│   ├── schemas.py           # Pydantic request/response models
│   └── logger.py            # Rotating file + console logging
│
├── frontend/
│   └── index.html           # Single-page UI (served by FastAPI)
│
├── logs/                    # Runtime logs (local only — not tracked in git)
│   └── app.log              #   Rotating, max 5MB × 3 backups
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw Data → EDA → Feature Engineering → Model Training → FastAPI Backend → Frontend
```

| Stage | Status | File |
|---|---|---|
| EDA | ✅ Done | `eda.py` |
| Feature Engineering | ✅ Done | `feature_engineering.py` |
| Model Training | ✅ Done | `train.py` |
| FastAPI Backend | ✅ Done | `backend/` |
| Frontend | ✅ Done | `frontend/index.html` |
| Logging | ✅ Done | `backend/logger.py` |

---

## How It Works

**No model is trained in the traditional sense.** This is a retrieval system, not a classifier.

1. Each job's skills, title, experience, industry and description are combined into a `feature_text` string
2. TF-IDF vectorizes all 123,842 jobs into an 8,000-dimensional sparse matrix
3. At query time, the user's skills are mapped to known LinkedIn categories, vectorized the same way, and cosine similarity is computed against all jobs
4. Top-N most similar jobs are returned

### Why Content-Based Filtering?

The dataset has zero user interaction data — no user IDs, no application history, no ratings.  
Collaborative filtering requires a user-item matrix which cannot be built here.  
Content-based filtering (TF-IDF + cosine similarity) is the correct approach for this dataset.

### Skill Mapping Layer

Users can type natural skill names like `python`, `figma`, `communication skills`.  
These are mapped to the 35 LinkedIn skill categories the model understands before inference.  
150+ skill mappings are defined in `backend/recommender.py`.

---

## Evaluation Results

Evaluated on 500 sampled test queries (job's own skills used as query, check if original job appears in top-k):

| Metric | @5 | @10 | @20 |
|---|---|---|---|
| Hit Rate | 3.4% | 4.8% | 7.8% |
| Precision | 99.9% | 99.9% | 99.9% |
| NDCG | 99.95% | 99.95% | 99.96% |
| Mean Similarity | 60.7% | 55.6% | 50.9% |

Hit Rate@10 = 4.8% on a pool of 123,842 jobs is **600× better than random baseline (0.008%)**.  
Precision is inflated due to only 35 broad skill categories — most jobs share at least one category.  
Hit Rate and Mean Similarity are the meaningful metrics.

---

## Dataset

LinkedIn Job Postings — 123,849 postings, 35 skill categories, 422 industries.  
Stored locally, excluded from git (postings.csv is ~493MB).

---

## Running Locally

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Generate processed data** (only needed once)
```bash
python feature_engineering.py
python train.py
```

**Step 3 — Start the server**
```bash
python -m uvicorn backend.main:app --reload
```

**Step 4 — Open the app**
```
http://localhost:8000
```

API docs (Swagger UI):
```
http://localhost:8000/docs
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

### POST `/resume`
Multipart form upload — send a PDF file. Extracts skills from resume text automatically.

### POST `/skill-gap`
```json
{
  "user_skills": ["python", "sql"],
  "job_skills": "Information Technology|Management|Finance"
}
```
Returns `matched_skills`, `missing_skills`, `match_pct`.

---

## Tech Stack

- **Data & ML:** Python, Pandas, Scikit-learn, NumPy, SciPy
- **Backend:** FastAPI, Uvicorn, PyMuPDF, Pydantic
- **Frontend:** HTML / CSS / JavaScript (no framework)
- **Visualisation:** Matplotlib, Seaborn, WordCloud
- **Logging:** Python logging — rotating file handler
