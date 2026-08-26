"""
src/preprocessing/skill_extractor.py

Granular skill extraction from job text (title + description).

Strategy:
  1. Normalise aliases  (k8s → kubernetes, ml → machine learning, etc.)
  2. Phrase-match against a curated skill dictionary (300+ skills)
  3. Return granular_skills list + broad_categories list per job

The 35 broad LinkedIn categories are RETAINED as metadata.
Granular skills are the PRIMARY representation for retrieval.
"""

import re
from functools import lru_cache
from backend.logger import get_logger

log = get_logger("skill_extractor")

# ─────────────────────────────────────────────────────────────────────────────
# ALIAS MAP  — normalise before matching
# Maps common abbreviations / variants → canonical skill name
# ─────────────────────────────────────────────────────────────────────────────
ALIAS_MAP: dict[str, str] = {
    # Python ecosystem
    "py"            : "python",
    "python3"       : "python",
    "python 3"      : "python",

    # ML / DL
    "ml"            : "machine learning",
    "dl"            : "deep learning",
    "ai"            : "artificial intelligence",
    "gen ai"        : "generative ai",
    "genai"         : "generative ai",
    "llm"           : "large language models",
    "llms"          : "large language models",
    "nlp"           : "natural language processing",
    "cv"            : "computer vision",
    "rl"            : "reinforcement learning",
    "xgb"           : "xgboost",
    "lgbm"          : "lightgbm",
    "sklearn"       : "scikit-learn",
    "torch"         : "pytorch",
    "tf"            : "tensorflow",
    "hf"            : "hugging face",
    "huggingface"   : "hugging face",

    # Cloud
    "aws"           : "aws",
    "amazon web services": "aws",
    "gcp"           : "gcp",
    "google cloud"  : "gcp",
    "google cloud platform": "gcp",
    "azure"         : "azure",
    "microsoft azure": "azure",

    # DevOps / infra
    "k8s"           : "kubernetes",
    "kube"          : "kubernetes",
    "docker compose": "docker",
    "ci/cd"         : "ci/cd",
    "cicd"          : "ci/cd",
    "github actions": "ci/cd",
    "jenkins"       : "ci/cd",

    # Databases
    "postgres"      : "postgresql",
    "pg"            : "postgresql",
    "mongo"         : "mongodb",
    "mssql"         : "sql server",
    "ms sql"        : "sql server",
    "mysql"         : "mysql",
    "nosql"         : "nosql",
    "elasticsearch" : "elasticsearch",
    "elastic"       : "elasticsearch",
    "redis"         : "redis",

    # Data
    "bi"            : "business intelligence",
    "etl"           : "etl",
    "elt"           : "etl",
    "powerbi"       : "power bi",
    "power-bi"      : "power bi",
    "tableau"       : "tableau",
    "looker"        : "looker",
    "dbt"           : "dbt",
    "airflow"       : "apache airflow",
    "spark"         : "apache spark",
    "pyspark"       : "apache spark",
    "kafka"         : "apache kafka",
    "hadoop"        : "hadoop",

    # Languages
    "js"            : "javascript",
    "ts"            : "typescript",
    "golang"        : "golang",
    "c sharp"       : "c#",
    "dotnet"        : ".net",
    ".net core"     : ".net",
    "asp.net"       : ".net",
    "node"          : "node.js",
    "nodejs"        : "node.js",
    "node js"       : "node.js",
    "reactjs"       : "react",
    "react.js"      : "react",
    "react js"      : "react",
    "vuejs"         : "vue.js",
    "vue js"        : "vue.js",
    "angularjs"     : "angular",
    "nextjs"        : "next.js",
    "next js"       : "next.js",

    # Design
    "ux"            : "ux design",
    "ui"            : "ui design",
    "ui/ux"         : "ui/ux design",
    "ux/ui"         : "ui/ux design",
    "figma"         : "figma",
    "adobe xd"      : "adobe xd",
    "sketch"        : "sketch",

    # Office / productivity
    "ms excel"      : "excel",
    "microsoft excel": "excel",
    "ms office"     : "microsoft office",
    "ms word"       : "microsoft office",
    "ms powerpoint" : "microsoft office",
    "google sheets" : "google workspace",
    "google docs"   : "google workspace",

    # Project management
    "pm"            : "project management",
    "pmp"           : "project management",
    "scrum master"  : "scrum",
    "agile/scrum"   : "agile",

    # Soft skills
    "comm"          : "communication",
    "comms"         : "communication",
}

# ─────────────────────────────────────────────────────────────────────────────
# SKILL DICTIONARY — canonical skill name → broad category
# Ordered longest-phrase-first to ensure multi-word skills match before subsets
# ─────────────────────────────────────────────────────────────────────────────
SKILL_DICT: dict[str, str] = {
    # ── Programming languages ─────────────────────────────────
    "python"                    : "Information Technology",
    "java"                      : "Information Technology",
    "javascript"                : "Information Technology",
    "typescript"                : "Information Technology",
    "c++"                       : "Information Technology",
    "c#"                        : "Information Technology",
    "golang"                    : "Information Technology",
    "rust"                      : "Information Technology",
    "scala"                     : "Information Technology",
    "kotlin"                    : "Information Technology",
    "swift"                     : "Information Technology",
    "ruby"                      : "Information Technology",
    "php"                       : "Information Technology",
    "r programming"             : "Analyst",
    "matlab"                    : "Information Technology",
    "bash"                      : "Information Technology",
    "shell scripting"           : "Information Technology",
    "perl"                      : "Information Technology",

    # ── ML / AI ───────────────────────────────────────────────
    "machine learning"          : "Information Technology",
    "deep learning"             : "Information Technology",
    "artificial intelligence"   : "Information Technology",
    "natural language processing": "Information Technology",
    "computer vision"           : "Information Technology",
    "reinforcement learning"    : "Information Technology",
    "generative ai"             : "Information Technology",
    "large language models"     : "Information Technology",
    "transformers"              : "Information Technology",
    "hugging face"              : "Information Technology",
    "neural networks"           : "Information Technology",
    "feature engineering"       : "Information Technology",
    "model deployment"          : "Information Technology",
    "mlops"                     : "Information Technology",
    "a/b testing"               : "Analyst",
    "statistical modeling"      : "Analyst",
    "time series"               : "Analyst",
    "forecasting"               : "Finance",

    # ── ML libraries ─────────────────────────────────────────
    "pytorch"                   : "Information Technology",
    "tensorflow"                : "Information Technology",
    "keras"                     : "Information Technology",
    "scikit-learn"              : "Information Technology",
    "xgboost"                   : "Information Technology",
    "lightgbm"                  : "Information Technology",
    "catboost"                  : "Information Technology",
    "pandas"                    : "Information Technology",
    "numpy"                     : "Information Technology",
    "scipy"                     : "Information Technology",
    "matplotlib"                : "Information Technology",
    "seaborn"                   : "Information Technology",
    "plotly"                    : "Information Technology",
    "opencv"                    : "Information Technology",
    "nltk"                      : "Information Technology",
    "spacy"                     : "Information Technology",
    "langchain"                 : "Information Technology",
    "llamaindex"                : "Information Technology",

    # ── Cloud ─────────────────────────────────────────────────
    "aws"                       : "Information Technology",
    "azure"                     : "Information Technology",
    "gcp"                       : "Information Technology",
    "aws lambda"                : "Information Technology",
    "aws s3"                    : "Information Technology",
    "aws ec2"                   : "Information Technology",
    "aws sagemaker"             : "Information Technology",
    "azure ml"                  : "Information Technology",
    "google bigquery"           : "Information Technology",
    "snowflake"                 : "Information Technology",
    "databricks"                : "Information Technology",

    # ── DevOps / infra ────────────────────────────────────────
    "docker"                    : "Information Technology",
    "kubernetes"                : "Information Technology",
    "ci/cd"                     : "Information Technology",
    "terraform"                 : "Information Technology",
    "ansible"                   : "Information Technology",
    "linux"                     : "Information Technology",
    "git"                       : "Information Technology",
    "github"                    : "Information Technology",
    "gitlab"                    : "Information Technology",
    "nginx"                     : "Information Technology",
    "microservices"             : "Information Technology",
    "rest api"                  : "Information Technology",
    "graphql"                   : "Information Technology",
    "grpc"                      : "Information Technology",

    # ── Databases ─────────────────────────────────────────────
    "sql"                       : "Information Technology",
    "nosql"                     : "Information Technology",
    "postgresql"                : "Information Technology",
    "mysql"                     : "Information Technology",
    "mongodb"                   : "Information Technology",
    "sql server"                : "Information Technology",
    "oracle"                    : "Information Technology",
    "redis"                     : "Information Technology",
    "elasticsearch"             : "Information Technology",
    "cassandra"                 : "Information Technology",
    "dynamodb"                  : "Information Technology",
    "sqlite"                    : "Information Technology",

    # ── Web / frontend ────────────────────────────────────────
    "react"                     : "Information Technology",
    "angular"                   : "Information Technology",
    "vue.js"                    : "Information Technology",
    "next.js"                   : "Information Technology",
    "node.js"                   : "Information Technology",
    "html"                      : "Information Technology",
    "css"                       : "Information Technology",
    "sass"                      : "Information Technology",
    "tailwind"                  : "Information Technology",
    "webpack"                   : "Information Technology",

    # ── Backend frameworks ────────────────────────────────────
    "django"                    : "Information Technology",
    "flask"                     : "Information Technology",
    "fastapi"                   : "Information Technology",
    "spring boot"               : "Information Technology",
    "express.js"                : "Information Technology",
    ".net"                      : "Information Technology",
    "laravel"                   : "Information Technology",
    "rails"                     : "Information Technology",

    # ── Data engineering ──────────────────────────────────────
    "apache spark"              : "Information Technology",
    "apache kafka"              : "Information Technology",
    "apache airflow"            : "Information Technology",
    "hadoop"                    : "Information Technology",
    "etl"                       : "Information Technology",
    "dbt"                       : "Information Technology",
    "data pipelines"            : "Information Technology",
    "data warehousing"          : "Information Technology",
    "data modeling"             : "Information Technology",

    # ── Analytics / BI ────────────────────────────────────────
    "tableau"                   : "Analyst",
    "power bi"                  : "Analyst",
    "looker"                    : "Analyst",
    "excel"                     : "Analyst",
    "google analytics"          : "Marketing",
    "data analysis"             : "Analyst",
    "data visualization"        : "Analyst",
    "business intelligence"     : "Analyst",
    "statistics"                : "Analyst",
    "quantitative analysis"     : "Analyst",

    # ── Design ────────────────────────────────────────────────
    "figma"                     : "Design",
    "adobe xd"                  : "Design",
    "sketch"                    : "Design",
    "ui/ux design"              : "Design",
    "ux design"                 : "Design",
    "ui design"                 : "Design",
    "wireframing"               : "Design",
    "prototyping"               : "Design",
    "user research"             : "Design",
    "design systems"            : "Design",
    "responsive design"         : "Design",
    "web design"                : "Design",
    "graphic design"            : "Art/Creative",
    "adobe photoshop"           : "Art/Creative",
    "adobe illustrator"         : "Art/Creative",
    "adobe indesign"            : "Art/Creative",
    "canva"                     : "Art/Creative",
    "animation"                 : "Art/Creative",
    "video editing"             : "Art/Creative",
    "photography"               : "Art/Creative",
    "branding"                  : "Art/Creative",
    "illustration"              : "Art/Creative",

    # ── Finance / Accounting ──────────────────────────────────
    "financial modeling"        : "Finance",
    "financial analysis"        : "Finance",
    "valuation"                 : "Finance",
    "investment banking"        : "Finance",
    "private equity"            : "Finance",
    "portfolio management"      : "Finance",
    "risk management"           : "Finance",
    "budgeting"                 : "Finance",
    "accounting"                : "Accounting/Auditing",
    "auditing"                  : "Accounting/Auditing",
    "tax"                       : "Accounting/Auditing",
    "bookkeeping"               : "Accounting/Auditing",
    "gaap"                      : "Accounting/Auditing",
    "ifrs"                      : "Accounting/Auditing",
    "quickbooks"                : "Accounting/Auditing",
    "sap"                       : "General Business",
    "erp"                       : "General Business",

    # ── Sales / Marketing ─────────────────────────────────────
    "sales"                     : "Sales",
    "b2b sales"                 : "Sales",
    "b2c sales"                 : "Sales",
    "lead generation"           : "Sales",
    "cold calling"              : "Sales",
    "salesforce"                : "Sales",
    "crm"                       : "Sales",
    "account management"        : "Sales",
    "business development"      : "Business Development",
    "digital marketing"         : "Marketing",
    "seo"                       : "Marketing",
    "sem"                       : "Marketing",
    "social media marketing"    : "Marketing",
    "content marketing"         : "Marketing",
    "email marketing"           : "Marketing",
    "google ads"                : "Marketing",
    "facebook ads"              : "Marketing",
    "marketing analytics"       : "Marketing",
    "brand management"          : "Marketing",
    "market research"           : "Marketing",
    "growth hacking"            : "Marketing",
    "product marketing"         : "Marketing",

    # ── Project / Product management ──────────────────────────
    "project management"        : "Project Management",
    "agile"                     : "Project Management",
    "scrum"                     : "Project Management",
    "kanban"                    : "Project Management",
    "jira"                      : "Project Management",
    "confluence"                : "Project Management",
    "product management"        : "Product Management",
    "product roadmap"           : "Product Management",
    "product strategy"          : "Product Management",
    "stakeholder management"    : "Management",
    "program management"        : "Project Management",

    # ── HR ────────────────────────────────────────────────────
    "recruitment"               : "Human Resources",
    "talent acquisition"        : "Human Resources",
    "human resources"           : "Human Resources",
    "onboarding"                : "Human Resources",
    "payroll"                   : "Human Resources",
    "hris"                      : "Human Resources",
    "performance management"    : "Human Resources",
    "employee relations"        : "Human Resources",

    # ── Healthcare ────────────────────────────────────────────
    "nursing"                   : "Health Care Provider",
    "clinical research"         : "Health Care Provider",
    "patient care"              : "Health Care Provider",
    "electronic health records" : "Health Care Provider",
    "ehr"                       : "Health Care Provider",
    "hipaa"                     : "Health Care Provider",
    "medical coding"            : "Health Care Provider",
    "pharmacy"                  : "Health Care Provider",

    # ── Research / Science ────────────────────────────────────
    "research"                  : "Research",
    "data collection"           : "Research",
    "literature review"         : "Research",
    "biology"                   : "Science",
    "chemistry"                 : "Science",
    "physics"                   : "Science",
    "genomics"                  : "Science",
    "bioinformatics"            : "Science",

    # ── Writing / Communication ───────────────────────────────
    "technical writing"         : "Writing/Editing",
    "copywriting"               : "Writing/Editing",
    "content writing"           : "Writing/Editing",
    "editing"                   : "Writing/Editing",
    "communication"             : "General Business",
    "public speaking"           : "General Business",
    "presentation skills"       : "General Business",

    # ── Education / Training ──────────────────────────────────
    "teaching"                  : "Education",
    "curriculum development"    : "Education",
    "instructional design"      : "Training",
    "training"                  : "Training",
    "coaching"                  : "Training",
    "mentoring"                 : "Training",

    # ── Customer service ──────────────────────────────────────
    "customer service"          : "Customer Service",
    "customer support"          : "Customer Service",
    "help desk"                 : "Customer Service",
    "zendesk"                   : "Customer Service",
    "technical support"         : "Customer Service",

    # ── Supply chain / Operations ─────────────────────────────
    "supply chain"              : "Supply Chain",
    "logistics"                 : "Supply Chain",
    "procurement"               : "Purchasing",
    "inventory management"      : "Supply Chain",
    "operations management"     : "General Business",
    "lean"                      : "Manufacturing",
    "six sigma"                 : "Quality Assurance",
    "quality assurance"         : "Quality Assurance",
    "quality control"           : "Quality Assurance",

    # ── Legal / Compliance ────────────────────────────────────
    "legal"                     : "Legal",
    "compliance"                : "Legal",
    "contract management"       : "Legal",
    "intellectual property"     : "Legal",
    "gdpr"                      : "Legal",

    # ── Strategy / Consulting ─────────────────────────────────
    "strategy"                  : "Strategy/Planning",
    "management consulting"     : "Consulting",
    "business strategy"         : "Strategy/Planning",
    "strategic planning"        : "Strategy/Planning",

    # ── Engineering ───────────────────────────────────────────
    "mechanical engineering"    : "Engineering",
    "electrical engineering"    : "Engineering",
    "civil engineering"         : "Engineering",
    "chemical engineering"      : "Engineering",
    "systems engineering"       : "Engineering",
    "embedded systems"          : "Engineering",
    "autocad"                   : "Engineering",
    "solidworks"                : "Engineering",

    # ── Soft skills ───────────────────────────────────────────
    "leadership"                : "Management",
    "team management"           : "Management",
    "problem solving"           : "General Business",
    "critical thinking"         : "General Business",
    "time management"           : "General Business",
    "collaboration"             : "General Business",
    "microsoft office"          : "General Business",
    "google workspace"          : "General Business",
}

# Pre-sort by phrase length descending so longer phrases match first
_SORTED_SKILLS = sorted(SKILL_DICT.keys(), key=len, reverse=True)

# Build a single compiled combined regex for fast vectorised matching
# Each skill becomes a named group — one pass over the text finds all skills
_SKILL_PATTERN = re.compile(
    "|".join(r"(?P<s{}>\b{}\b)".format(i, re.escape(s))
             for i, s in enumerate(_SORTED_SKILLS)),
    re.IGNORECASE
)
_IDX_TO_SKILL = {i: s for i, s in enumerate(_SORTED_SKILLS)}

# Pre-compile alias patterns
_ALIAS_PATTERNS = [
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), canonical)
    for alias, canonical in ALIAS_MAP.items()
]


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction functions
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, keep alphanumeric + key punctuation."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\+\#\./\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_aliases(text: str) -> str:
    """Replace known aliases with canonical forms using pre-compiled patterns."""
    for pattern, canonical in _ALIAS_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


@lru_cache(maxsize=8192)
def extract_skills(text: str) -> tuple[str, ...]:
    """
    Extract granular skills from a text string.
    Returns a tuple of canonical skill names (hashable for caching).
    Single-pass regex over the full text — much faster than per-skill loops.
    """
    if not text or not str(text).strip():
        return ()

    normalised = _normalise_text(text)
    normalised = _apply_aliases(normalised)

    found: set[str] = set()
    consumed_spans: list[tuple[int, int]] = []

    for m in _SKILL_PATTERN.finditer(normalised):
        start, end = m.start(), m.end()
        # Skip overlapping spans (longer phrases already consumed this region)
        if any(s <= start < e or s < end <= e for s, e in consumed_spans):
            continue
        # Find which group matched
        matched_skill = m.group(0).lower()
        # Map back to canonical skill name
        canonical = next(
            (s for s in _SORTED_SKILLS if s == matched_skill or
             re.fullmatch(re.escape(s), matched_skill, re.IGNORECASE)),
            matched_skill
        )
        if canonical in SKILL_DICT:
            found.add(canonical)
            consumed_spans.append((start, end))

    return tuple(sorted(found))


def extract_broad_categories(granular_skills: list[str]) -> list[str]:
    """Map granular skills → unique broad categories."""
    cats = {SKILL_DICT[s] for s in granular_skills if s in SKILL_DICT}
    return sorted(cats)


def extract_skills_from_job(title: str, description: str) -> dict:
    """
    Extract granular skills and broad categories from a job's title + description.
    Returns dict with granular_skills and broad_categories.
    """
    combined = f"{title} {description}"
    granular = list(extract_skills(combined))
    broad    = extract_broad_categories(granular)
    return {
        "granular_skills" : granular,
        "broad_categories": broad,
    }
