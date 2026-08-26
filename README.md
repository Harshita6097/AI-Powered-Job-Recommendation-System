# AI-Powered Job Recommendation System

An end-to-end job recommendation engine using a **hybrid approach** — SVD-based collaborative filtering combined with TF-IDF content-based filtering to handle the cold-start problem.

## Techniques Used

| Technique | Purpose |
|---|---|
| TruncatedSVD | Collaborative filtering via matrix factorization on user–job interactions |
| TF-IDF + Cosine Similarity | Content-based filtering on job skills |
| Hybrid Blending | Weighted combination of both scores; α scales with interaction history |
| Cold-Start Handling | Falls back to pure content-based when user has no interaction history |

## Project Structure

```
├── backend/
│   ├── data/
│   │   ├── users.csv
│   │   ├── jobs.csv
│   │   └── interactions.csv
│   ├── recommender.py   # Hybrid engine (SVD + TF-IDF)
│   ├── main.py          # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── index.html           # User input form
│   └── recommendations.html # Results page
└── README.md
```

## Setup & Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

## How It Works

- **Existing user** → hybrid score = `α × SVD_score + (1−α) × TF-IDF_score`  
  α increases with the number of past interactions (more history = trust collaborative more)
- **New user (cold start)** → pure TF-IDF cosine similarity on entered skills
