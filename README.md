# AI-Powered Job Recommendation System

> **Branch:** `job-recommender` — Active development  
> **Previous version:** [`main`](../../tree/main)

A content-based job recommendation system built on the LinkedIn Job Postings dataset. Users input their skills or upload a resume and receive ranked job role recommendations along with a skill gap analysis showing what to learn next.

---

## Project Structure

```
├── linkedin_dataset/        # Raw data (local only — not tracked in git)
│   ├── postings.csv
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
├── eda.py                   # Exploratory Data Analysis (17 analyses)
├── eda_outputs/             # EDA plots (local only — not tracked in git)
│
├── notebooks/               # Step-by-step development notebooks
│
├── backend/                 # FastAPI backend (coming soon)
├── frontend/                # Web UI (coming soon)
├── tests/                   # Unit tests (coming soon)
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
| Feature Engineering | 🔄 Next | `feature_engineering.py` |
| Model Training | ⏳ Pending | `train.py` |
| FastAPI Backend | ⏳ Pending | `backend/` |
| Frontend | ⏳ Pending | `frontend/` |

---

## Why Content-Based Filtering?

This dataset has no user interaction data (no application history, no user IDs, no ratings). Collaborative filtering requires a user-item matrix which cannot be built here. Content-based filtering — matching user skills to job skill requirements via cosine similarity — is the correct approach for this dataset.

---

## Dataset

LinkedIn Job Postings dataset — 123,849 postings, 35 skill categories, 422 industries.  
Dataset is stored locally and excluded from git due to file size (postings.csv is ~493MB).

---

## Tech Stack

- **Data & ML:** Python, Pandas, Scikit-learn, NumPy
- **Backend:** FastAPI
- **Frontend:** HTML / CSS / JavaScript
- **Visualisation:** Matplotlib, Seaborn, WordCloud
