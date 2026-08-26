# AI-Powered Job Recommendation System

A full-stack job recommendation engine built with Python and FastAPI, leveraging a **hybrid approach** that combines collaborative filtering (SVD) and content-based filtering (TF-IDF) to deliver accurate, personalized job matches — including handling the cold-start problem for new users.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| ML / Recommendation | scikit-learn (TruncatedSVD, TF-IDF, Cosine Similarity) |
| Data | Pandas, NumPy |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

## Recommendation Techniques

| Technique | Purpose |
|---|---|
| TruncatedSVD | Collaborative filtering via matrix factorization on user–job interaction data |
| TF-IDF + Cosine Similarity | Content-based filtering on job skills for new users |
| Hybrid Blending | Weighted combination — `α × SVD_score + (1−α) × TF-IDF_score` |
| Cold-Start Handling | Falls back to pure content-based filtering when no interaction history exists |

> α scales dynamically with the number of past interactions — the more history a user has, the more the model trusts collaborative signals.

---

## Project Structure

```
├── backend/
│   ├── data/
│   │   ├── users.csv          # User profiles and skills
│   │   ├── jobs.csv           # Job listings with required skills and location
│   │   └── interactions.csv   # User–job interaction scores
│   ├── recommender.py         # Hybrid recommendation engine (SVD + TF-IDF)
│   ├── main.py                # FastAPI application and API routes
│   └── requirements.txt
├── frontend/
│   ├── index.html             # User input form (existing user / new user)
│   └── recommendations.html  # Job recommendation results page
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
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

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

- **108 job listings** across roles like Data Scientist, ML Engineer, DevOps, UX Designer, etc.
- **20 users** with skill profiles
- **Interaction scores** (1–5) representing user engagement with jobs
