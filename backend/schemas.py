from pydantic import BaseModel
from typing import List, Optional


class RecommendRequest(BaseModel):
    skills: List[str]
    experience_level: Optional[str] = None   # Internship | Entry | Associate | Mid-Senior | Director | Executive
    top_n: int = 10
    filter_exp: bool = False
    filter_remote: bool = False


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
    results: List[JobResult]


class SkillGapRequest(BaseModel):
    user_skills: List[str]
    job_skills: str   # pipe-separated string from job result
