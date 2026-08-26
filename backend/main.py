from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from recommender import load_data, build_models, recommend

app = FastAPI(title="AI Job Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Load data and build models once at startup
jobs, users, interactions = load_data()
vectorizer, job_vectors, svd_predictions, interaction_matrix = build_models(jobs, interactions)


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
            jobs=jobs,
            vectorizer=vectorizer,
            job_vectors=job_vectors,
            svd_predictions=svd_predictions,
            interaction_matrix=interaction_matrix,
            user_id=req.user_id,
            user_skills=req.user_skills,
            top_k=req.top_k,
        )
        return {"recommendations": results, "count": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
