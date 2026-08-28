from pydantic import BaseModel
from typing import List, Optional


class RecommendRequest(BaseModel):
    skills: List[str]
    experience_level: Optional[str] = None   # Internship | Entry | Associate | Mid-Senior | Director | Executive
    top_n: int = 10
    offset: int = 0                          # skip first N results (for pagination)
    filter_exp: bool = False
    filter_remote: bool = False
    salary_min: Optional[float] = None       # yearly minimum salary filter
    salary_max: Optional[float] = None       # yearly maximum salary filter


class JobResult(BaseModel):
    job_id: str
    title: str
    experience_level: str
    industry: str
    skills: str
    salary_mid: Optional[float]
    work_type: str
    is_remote: bool
    state: Optional[str]
    match_score: float
    retrieval_mode: Optional[str] = None


class SkillGapResult(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    match_pct: float


class RecommendResponse(BaseModel):
    query_skills_raw: List[str]
    query_skills_mapped: List[str]
    unmapped_skills: List[str]
    retrieval_mode: str = "tfidf"
    total: int = 0          # total results fetched before pagination slice
    offset: int = 0         # offset used in this response
    results: List[JobResult]


class SkillGapRequest(BaseModel):
    user_skills: List[str]
    job_skills: str   # pipe-separated string from job result
