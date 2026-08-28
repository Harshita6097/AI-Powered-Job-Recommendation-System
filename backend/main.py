import io
import time
import pymupdf as fitz
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Optional

import config
from backend.recommender import load_artifacts
from backend.recommender import recommend as legacy_recommend
from backend.recommender import skill_gap as legacy_skill_gap
from backend.schemas import (
    RecommendRequest, RecommendResponse, JobResult,
    SkillGapRequest, SkillGapResult,
)
from backend.logger import get_logger
from src.inference.engine import recommend as engine_recommend
from src.inference.engine import skill_gap as engine_skill_gap
from src.inference.engine import load_all, cache_info, cache_clear
from src.preprocessing.skill_extractor import extract_skills

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== AI Job Recommender API starting up ===")
    if config.RETRIEVAL_MODE == "tfidf":
        load_artifacts()
    else:
        load_all()
    log.info("=== Startup complete — mode=%s — ready to serve ===", config.RETRIEVAL_MODE)
    yield
    log.info("=== API shutting down ===")


app = FastAPI(
    title="AI Job Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s → %d (%.3fs)",
             request.method, request.url.path,
             response.status_code, time.time() - t0)
    return response


@app.get("/health")
def health():
    log.debug("Health check")
    return {"status": "ok", "mode": config.RETRIEVAL_MODE, "cache": cache_info()}


@app.get("/cache")
def get_cache_info():
    return {"cache": cache_info(), "mode": config.RETRIEVAL_MODE}


@app.delete("/cache")
def clear_cache():
    cache_clear()
    log.info("Recommendation cache cleared")
    return {"status": "cleared"}


@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(req: RecommendRequest):
    if not req.skills:
        raise HTTPException(status_code=400, detail="skills list cannot be empty")

    if config.RETRIEVAL_MODE != "tfidf":
        out = engine_recommend(
            skills=req.skills,
            experience_level=req.experience_level,
            top_n=req.top_n + req.offset,
            filter_exp=req.filter_exp,
            filter_remote=req.filter_remote,
            salary_min=req.salary_min,
            salary_max=req.salary_max,
        )
        if not out["results"]:
            raise HTTPException(status_code=404, detail="No results found for given filters")
        page = out["results"][req.offset: req.offset + req.top_n]
        jobs = [
            JobResult(
                job_id=str(r["job_id"]),
                title=r["title"],
                experience_level=r.get("experience_level") or "",
                industry=r.get("industry") or "",
                skills=r.get("skills") or "",
                salary_mid=r.get("salary_mid") if r.get("salary_mid") == r.get("salary_mid") else None,
                work_type=r.get("work_type") or "",
                is_remote=bool(r.get("is_remote", False)),
                state=r.get("state") if isinstance(r.get("state"), str) else None,
                match_score=r["match_score"],
            )
            for r in page
        ]
        return RecommendResponse(
            query_skills_raw=out["query_skills_raw"],
            query_skills_mapped=out["query_skills_mapped"],
            unmapped_skills=out["unmapped_skills"],
            retrieval_mode=out["retrieval_mode"],
            total=len(out["results"]),
            offset=req.offset,
            results=jobs,
        )

    # legacy tfidf path
    results_df, mapped_skills, unmapped = legacy_recommend(
        raw_skills=req.skills,
        experience_level=req.experience_level,
        top_n=req.top_n,
        filter_exp=req.filter_exp,
        filter_remote=req.filter_remote,
    )
    if results_df.empty:
        log.warning("No results for skills=%s exp=%s", req.skills, req.experience_level)
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
        retrieval_mode="tfidf",
        total=len(results_df),
        offset=req.offset,
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

    if config.RETRIEVAL_MODE != "tfidf":
        found_raw = list(extract_skills(text))
        if not found_raw:
            raise HTTPException(status_code=422, detail="No recognisable skills found in resume")
        log.info("Resume '%s' — extracted %d granular skills", file.filename, len(found_raw))
        out = engine_recommend(
            skills=found_raw,
            experience_level=experience_level,
            top_n=10,
            filter_remote=filter_remote,
        )
        if not out["results"]:
            raise HTTPException(status_code=404, detail="No results found")
        jobs = [
            JobResult(
                job_id=str(r["job_id"]),
                title=r["title"],
                experience_level=r.get("experience_level") or "",
                industry=r.get("industry") or "",
                skills=r.get("skills") or "",
                salary_mid=r.get("salary_mid") if r.get("salary_mid") == r.get("salary_mid") else None,
                work_type=r.get("work_type") or "",
                is_remote=bool(r.get("is_remote", False)),
                state=r.get("state") if isinstance(r.get("state"), str) else None,
                match_score=r["match_score"],
            )
            for r in out["results"]
        ]
        return RecommendResponse(
            query_skills_raw=found_raw[:10],
            query_skills_mapped=out["query_skills_mapped"],
            unmapped_skills=out["unmapped_skills"],
            retrieval_mode=out["retrieval_mode"],
            results=jobs,
        )

    # legacy tfidf path
    from backend.recommender import SKILL_MAP, SKILL_CATEGORIES
    found_raw = [k for k in SKILL_MAP if k in text]
    found_raw += [c for c in SKILL_CATEGORIES if c.lower() in text]
    if not found_raw:
        log.warning("Resume '%s' — no recognisable skills found", file.filename)
        raise HTTPException(status_code=422, detail="No recognisable skills found in resume")
    log.info("Resume '%s' — extracted %d skill signals", file.filename, len(found_raw))
    results_df, mapped_skills, unmapped = legacy_recommend(
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
        retrieval_mode="tfidf",
        results=jobs,
    )


@app.post("/skill-gap", response_model=SkillGapResult)
def get_skill_gap(req: SkillGapRequest):
    if not req.user_skills:
        raise HTTPException(status_code=400, detail="user_skills cannot be empty")
    if not req.job_skills:
        raise HTTPException(status_code=400, detail="job_skills cannot be empty")

    if config.RETRIEVAL_MODE != "tfidf":
        gap = engine_skill_gap(req.user_skills, req.job_skills)
    else:
        gap = legacy_skill_gap(req.user_skills, req.job_skills)
    return SkillGapResult(**gap)


# Must be mounted last — catches all unmatched routes for the frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
