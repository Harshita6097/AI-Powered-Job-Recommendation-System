import io
import pymupdf as fitz
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional

from backend.recommender import load_artifacts, recommend, skill_gap, map_skills
from backend.schemas import (
    RecommendRequest, RecommendResponse, JobResult,
    SkillGapRequest, SkillGapResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="AI Job Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(req: RecommendRequest):
    if not req.skills:
        raise HTTPException(status_code=400, detail="skills list cannot be empty")

    results_df, mapped_skills, unmapped = recommend(
        raw_skills=req.skills,
        experience_level=req.experience_level,
        top_n=req.top_n,
        filter_exp=req.filter_exp,
        filter_remote=req.filter_remote,
    )

    if results_df.empty:
        raise HTTPException(status_code=404, detail="No results found for given filters")

    jobs = [
        JobResult(
            job_id=str(row.job_id),
            title=row.title_clean,
            experience_level=row.experience_level,
            industry=row.industry,
            skills=row.skills,
            salary_mid=row.salary_mid if str(row.salary_mid) != "nan" else None,
            work_type=row.work_type,
            is_remote=bool(row.is_remote),
            state=row.state if str(row.state) != "nan" else None,
            match_score=row.match_score,
        )
        for row in results_df.itertuples()
    ]

    return RecommendResponse(
        query_skills_raw=req.skills,
        query_skills_mapped=mapped_skills,
        unmapped_skills=unmapped,
        results=jobs,
    )


@app.post("/resume", response_model=RecommendResponse)
async def recommend_from_resume(
    file: UploadFile = File(...),
    experience_level: Optional[str] = Form(None),
    filter_remote: bool = Form(False),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
    text = " ".join(page.get_text() for page in doc).lower()
    doc.close()

    # Extract skills by checking which SKILL_MAP keys appear in resume text
    from backend.recommender import SKILL_MAP, SKILL_CATEGORIES
    found_raw = [k for k in SKILL_MAP if k in text]
    # Also check direct category names
    found_raw += [c for c in SKILL_CATEGORIES if c.lower() in text]

    if not found_raw:
        raise HTTPException(status_code=422, detail="No recognisable skills found in resume")

    results_df, mapped_skills, unmapped = recommend(
        raw_skills=found_raw,
        experience_level=experience_level,
        top_n=10,
        filter_remote=filter_remote,
    )

    if results_df.empty:
        raise HTTPException(status_code=404, detail="No results found")

    jobs = [
        JobResult(
            job_id=str(row.job_id),
            title=row.title_clean,
            experience_level=row.experience_level,
            industry=row.industry,
            skills=row.skills,
            salary_mid=row.salary_mid if str(row.salary_mid) != "nan" else None,
            work_type=row.work_type,
            is_remote=bool(row.is_remote),
            state=row.state if str(row.state) != "nan" else None,
            match_score=row.match_score,
        )
        for row in results_df.itertuples()
    ]

    return RecommendResponse(
        query_skills_raw=found_raw[:10],
        query_skills_mapped=mapped_skills,
        unmapped_skills=unmapped,
        results=jobs,
    )


@app.post("/skill-gap", response_model=SkillGapResult)
def get_skill_gap(req: SkillGapRequest):
    if not req.user_skills:
        raise HTTPException(status_code=400, detail="user_skills cannot be empty")
    if not req.job_skills:
        raise HTTPException(status_code=400, detail="job_skills cannot be empty")

    gap = skill_gap(req.user_skills, req.job_skills)
    return SkillGapResult(**gap)
