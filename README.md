# AI-Powered Job Recommendation System

An end-to-end job recommendation engine built with Python and FastAPI, leveraging a **hybrid approach** that combines collaborative filtering (SVD) and content-based filtering (TF-IDF) to deliver accurate, personalized job matches — including handling the cold-start problem for new users.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| ML / Recommendation | scikit-learn (TruncatedSVD, TF-IDF, Cosine Similarity) |
| Data | Pandas, NumPy |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Dataset | LinkedIn Job Postings (Kaggle) |

---

## Recommendation Techniques

| Technique | Purpose |
|---|---|
| TruncatedSVD | Collaborative filtering via matrix factorization on user–job interaction data |
| TF-IDF + Cosine Similarity | Content-based filtering on job skills |
| Hybrid Blending | Weighted combination — `α × SVD_score + (1−α) × TF-IDF_score` |
| Cold-Start Handling | Falls back to pure content-based filtering when no interaction history exists |

> α scales dynamically with the number of past interactions — the more history a user has, the more the model trusts collaborative signals.

---

## Project Structure

```
├── backend/
│   ├── data/
│   │   ├── raw/                   # Raw LinkedIn dataset (local only, gitignored)
│   │   └── processed/             # Cleaned pipeline-ready CSVs (committed)
│   │       ├── jobs.csv
│   │       ├── users.csv
│   │       └── interactions.csv
│   ├── model/                     # Saved model artifacts (gitignored, regenerate locally)
│   ├── prepare_data.py            # Cleans raw LinkedIn data → processed/
│   ├── train.py                   # Trains SVD + TF-IDF, saves artifacts, prints eval report
│   ├── recommender.py             # Hybrid recommendation engine
│   ├── main.py                    # FastAPI app and API routes
│   └── requirements.txt
├── frontend/
│   ├── index.html                 # User input form (existing user / new user)
│   └── recommendations.html      # Job recommendation results page
├── tests/
│   └── test_recommender.py        # Unit tests
├── .gitignore
└── README.md
```

---

## How It Works

### Existing User (Hybrid Mode)
- Builds a user–job interaction matrix and applies **TruncatedSVD** to learn latent factors
- Computes **TF-IDF cosine similarity** on job skills
- Blends both scores: `score = α × collaborative + (1−α) × content`
- α increases with interaction count — more history means more trust in collaborative filtering

### New User (Cold Start)
- No interaction history required
- User enters their skills directly
- Pure **TF-IDF cosine similarity** matches skills against all job listings

---

## Setup & Run

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. (First time only) Prepare data from raw LinkedIn dataset
python prepare_data.py

# 3. Train the model
python train.py

# 4. Start the server
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Model Evaluation

Running `python train.py` prints a full evaluation report:

```
=======================================================
  MODEL EVALUATION REPORT
=======================================================
  Dataset
    Users            : 500
    Jobs             : 480
    Interactions     : 10,739
    Avg per user     : 21.5

  SVD (Collaborative Filtering)
    Components       : 50
    Variance explained: 33.59%
    RMSE (train)     : 1.7220
    RMSE (test 20%)  : 2.5314

  Ranking Metrics
    Precision@5      : 0.4668  (46.7%)
    Precision@10     : 0.3242  (32.4%)
=======================================================
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend landing page |
| GET | `/api/users` | Returns list of all users |
| POST | `/api/recommend` | Returns top-N job recommendations |

### POST `/api/recommend` — Request Body

```json
{
  "user_id": "user_1",
  "user_skills": "Python, Machine Learning",
  "top_k": 8
}
```

Pass either `user_id` (existing user) or `user_skills` (new user) — not required to pass both.

---

## Dataset

- **480 real job listings** from the LinkedIn Job Postings dataset (Kaggle) across 80 unique roles
- **500 users** with domain-based skill profiles
- **10,739 interaction scores** (1–5) covering all jobs
- Raw dataset is excluded from version control — `prepare_data.py` regenerates processed CSVs locally

---

## Notes

- `backend/model/` artifacts are gitignored — run `python train.py` to regenerate locally
- `backend/data/raw/` is gitignored — download the LinkedIn dataset from Kaggle and place it there
- `backend/data/processed/` CSVs are committed so the model can be trained without the raw dataset
