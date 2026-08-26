from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from recommender import load_model, recommend

app = FastAPI(title="AI Job Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DATA_DIR = Path(__file__).parent / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).parent / "data"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Load pre-trained model artifacts once at startup
vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = load_model()
users = pd.read_csv(DATA_DIR / "users.csv")


class RecommendRequest(BaseModel):
    user_id: str | None = None
    user_skills: str | None = None
    top_k: int = 8


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/recommendations")
def serve_recommendations():
    return FileResponse(str(FRONTEND_DIR / "recommendations.html"))


@app.get("/api/users")
def get_users():
    return users[["user_id", "user_skills", "user_location"]].to_dict(orient="records")


@app.post("/api/recommend")
def get_recommendations(req: RecommendRequest):
    try:
        results = recommend(
            vectorizer=vectorizer,
            job_vectors=job_vectors,
            svd_predictions=svd_predictions,
            interaction_matrix=interaction_matrix,
            jobs=jobs,
            user_id=req.user_id,
            user_skills=req.user_skills,
            top_k=req.top_k,
        )
        return {"recommendations": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
