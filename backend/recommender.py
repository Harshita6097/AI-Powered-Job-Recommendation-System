import pickle
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ── Skill normalisation map ───────────────────────────────────────────────────
# Maps granular user-typed skills → one of the 35 LinkedIn skill categories
SKILL_MAP: dict[str, str] = {
    # Design / Creative
    "figma": "Design", "ui/ux": "Design", "ux": "Design", "ui": "Design",
    "sketch": "Design", "adobe xd": "Design", "wireframing": "Design",
    "prototyping": "Design", "user experience": "Design", "user interface": "Design",
    "web design": "Design", "graphic design": "Art/Creative",
    "illustration": "Art/Creative", "drawing": "Art/Creative",
    "photoshop": "Art/Creative", "illustrator": "Art/Creative",
    "indesign": "Art/Creative", "canva": "Art/Creative",
    "animation": "Art/Creative", "video editing": "Art/Creative",
    "photography": "Art/Creative", "branding": "Art/Creative",

    # Information Technology
    "python": "Information Technology", "java": "Information Technology",
    "javascript": "Information Technology", "typescript": "Information Technology",
    "c++": "Information Technology", "c#": "Information Technology",
    "sql": "Information Technology", "nosql": "Information Technology",
    "react": "Information Technology", "angular": "Information Technology",
    "vue": "Information Technology", "node.js": "Information Technology",
    "django": "Information Technology", "flask": "Information Technology",
    "fastapi": "Information Technology", "aws": "Information Technology",
    "azure": "Information Technology", "gcp": "Information Technology",
    "docker": "Information Technology", "kubernetes": "Information Technology",
    "git": "Information Technology", "linux": "Information Technology",
    "machine learning": "Information Technology", "deep learning": "Information Technology",
    "nlp": "Information Technology", "computer vision": "Information Technology",
    "data science": "Information Technology", "data engineering": "Information Technology",
    "software development": "Information Technology", "backend": "Information Technology",
    "frontend": "Information Technology", "full stack": "Information Technology",
    "devops": "Information Technology", "cybersecurity": "Information Technology",
    "networking": "Information Technology", "database": "Information Technology",
    "api": "Information Technology", "rest api": "Information Technology",
    "html": "Information Technology", "css": "Information Technology",
    "tensorflow": "Information Technology", "pytorch": "Information Technology",
    "spark": "Information Technology", "hadoop": "Information Technology",
    "tableau": "Analyst", "power bi": "Analyst",

    # Analyst
    "data analysis": "Analyst", "data analytics": "Analyst",
    "business analysis": "Analyst", "excel": "Analyst",
    "statistics": "Analyst", "r": "Analyst", "spss": "Analyst",
    "reporting": "Analyst", "dashboards": "Analyst",

    # Finance / Accounting
    "accounting": "Accounting/Auditing", "auditing": "Accounting/Auditing",
    "bookkeeping": "Accounting/Auditing", "tax": "Accounting/Auditing",
    "financial analysis": "Finance", "financial modelling": "Finance",
    "budgeting": "Finance", "forecasting": "Finance",
    "investment": "Finance", "banking": "Finance",
    "risk management": "Finance", "portfolio management": "Finance",

    # Sales / Marketing
    "sales": "Sales", "cold calling": "Sales", "lead generation": "Sales",
    "crm": "Sales", "salesforce": "Sales", "b2b": "Sales", "b2c": "Sales",
    "marketing": "Marketing", "digital marketing": "Marketing",
    "seo": "Marketing", "sem": "Marketing", "social media": "Marketing",
    "content marketing": "Marketing", "email marketing": "Marketing",
    "google ads": "Marketing", "facebook ads": "Marketing",
    "brand management": "Marketing", "market research": "Marketing",

    # Communication / Soft skills
    "communication": "General Business", "communication skills": "General Business",
    "presentation": "General Business", "public speaking": "General Business",
    "teamwork": "General Business", "collaboration": "General Business",
    "problem solving": "General Business", "critical thinking": "General Business",
    "time management": "General Business", "leadership": "Management",
    "team management": "Management", "people management": "Management",
    "stakeholder management": "Management",

    # Project / Product Management
    "project management": "Project Management", "agile": "Project Management",
    "scrum": "Project Management", "jira": "Project Management",
    "product management": "Product Management", "product roadmap": "Product Management",
    "product strategy": "Product Management",

    # HR
    "recruitment": "Human Resources", "talent acquisition": "Human Resources",
    "hr": "Human Resources", "human resources": "Human Resources",
    "onboarding": "Human Resources", "payroll": "Human Resources",

    # Healthcare
    "nursing": "Health Care Provider", "clinical": "Health Care Provider",
    "patient care": "Health Care Provider", "medical": "Health Care Provider",
    "healthcare": "Health Care Provider", "pharmacy": "Health Care Provider",

    # Research / Science
    "research": "Research", "data collection": "Research",
    "lab": "Science", "biology": "Science", "chemistry": "Science",
    "physics": "Science",

    # Writing
    "writing": "Writing/Editing", "copywriting": "Writing/Editing",
    "editing": "Writing/Editing", "content writing": "Writing/Editing",
    "technical writing": "Writing/Editing", "blogging": "Writing/Editing",

    # Education / Training
    "teaching": "Education", "curriculum": "Education",
    "training": "Training", "coaching": "Training", "mentoring": "Training",

    # Customer Service
    "customer service": "Customer Service", "customer support": "Customer Service",
    "help desk": "Customer Service",

    # Supply Chain / Operations
    "supply chain": "Supply Chain", "logistics": "Supply Chain",
    "procurement": "Purchasing", "inventory": "Supply Chain",
    "operations": "General Business",

    # Legal
    "legal": "Legal", "compliance": "Legal", "contracts": "Legal",

    # Strategy
    "strategy": "Strategy/Planning", "business strategy": "Strategy/Planning",
    "consulting": "Consulting", "business consulting": "Consulting",

    # PR / Advertising
    "public relations": "Public Relations", "pr": "Public Relations",
    "advertising": "Advertising", "media": "Advertising",

    # Engineering
    "engineering": "Engineering", "mechanical engineering": "Engineering",
    "electrical engineering": "Engineering", "civil engineering": "Engineering",
    "chemical engineering": "Engineering",

    # Manufacturing / QA
    "manufacturing": "Manufacturing", "quality assurance": "Quality Assurance",
    "qa": "Quality Assurance", "testing": "Quality Assurance",
}

SKILL_CATEGORIES: list[str] = []
_df: pd.DataFrame = None
_tfidf = None
_vectorizer = None
_meta: dict = {}


def load_artifacts(base_path: str = "data/processed"):
    global SKILL_CATEGORIES, _df, _tfidf, _vectorizer, _meta

    _df = pd.read_csv(f"{base_path}/jobs_features.csv")
    _df["skills_list"] = _df["skills"].apply(
        lambda x: x.split("|") if isinstance(x, str) else []
    )
    _tfidf = load_npz(f"{base_path}/tfidf_matrix.npz")

    with open(f"{base_path}/tfidf_vectorizer.pkl", "rb") as f:
        _vectorizer = pickle.load(f)
    with open(f"{base_path}/feature_metadata.json") as f:
        _meta = json.load(f)

    SKILL_CATEGORIES = _meta["skill_categories"]


def map_skills(raw_skills: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Maps raw user skills to known LinkedIn skill categories.
    Returns (mapped, unmapped, all_mapped_deduplicated).
    """
    mapped, unmapped = [], []
    for s in raw_skills:
        key = s.strip().lower()
        if key in SKILL_MAP:
            mapped.append(SKILL_MAP[key])
        elif s.strip().title() in SKILL_CATEGORIES:
            mapped.append(s.strip().title())
        else:
            unmapped.append(s)

    # Deduplicate while preserving order
    seen = set()
    deduped = [x for x in mapped if not (x in seen or seen.add(x))]
    return deduped, unmapped, mapped


def _build_feature_text(skills: list[str], experience_level: str = None) -> str:
    skills_str = " ".join(skills).lower().replace("/", " ").replace("-", " ")
    exp_str = str(experience_level or "").lower().replace("-", " ")
    return f"{skills_str} {skills_str} {skills_str} {exp_str}"


def recommend(
    raw_skills: list[str],
    experience_level: str = None,
    top_n: int = 10,
    filter_exp: bool = False,
    filter_remote: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    mapped_skills, unmapped, _ = map_skills(raw_skills)

    # Fall back to raw input if nothing mapped
    query_skills = mapped_skills if mapped_skills else [s.strip().title() for s in raw_skills]

    user_text = _build_feature_text(query_skills, experience_level)
    user_vector = _vectorizer.transform([user_text])

    candidate_df = _df.copy()
    candidate_idx = np.arange(len(_df))

    if filter_exp and experience_level:
        mask = candidate_df["experience_level"] == experience_level
        candidate_df = candidate_df[mask].reset_index(drop=True)
        candidate_idx = np.where(mask)[0]

    if filter_remote:
        mask = candidate_df["is_remote"] == 1
        candidate_df = candidate_df[mask].reset_index(drop=True)
        candidate_idx = candidate_idx[candidate_df.index] if filter_exp else np.where(mask)[0]

    if len(candidate_df) == 0:
        return pd.DataFrame(), mapped_skills, unmapped

    scores = cosine_similarity(user_vector, _tfidf[candidate_idx]).flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]

    results = candidate_df.iloc[top_idx][[
        "job_id", "title_clean", "experience_level",
        "industry", "skills", "salary_mid",
        "work_type", "is_remote", "state"
    ]].copy()
    results["match_score"] = np.round(scores[top_idx] * 100, 2)

    return results.reset_index(drop=True), mapped_skills, unmapped


def skill_gap(user_skills: list[str], job_skills_str: str) -> dict:
    mapped_skills, _, _ = map_skills(user_skills)
    query_set = set(mapped_skills) if mapped_skills else {s.strip().title() for s in user_skills}

    job_skills_list = [s.strip() for s in job_skills_str.split("|")]
    matched = [s for s in job_skills_list if s in query_set]
    missing = [s for s in job_skills_list if s not in query_set]

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_pct": round(len(matched) / max(len(job_skills_list), 1) * 100, 1),
    }
