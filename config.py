"""
config.py — Central configuration for the Job Recommendation System.

All tunable parameters live here.
Import with:
    from config import CFG
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "linkedin_dataset"
DATA_PROC   = ROOT / "data" / "processed"
DATA_INDEX  = ROOT / "data" / "indexes"
LOGS_DIR    = ROOT / "logs"

# Raw dataset files
POSTINGS_CSV       = DATA_RAW / "postings.csv"
JOB_SKILLS_CSV     = DATA_RAW / "jobs" / "job_skills.csv"
SKILLS_MAP_CSV     = DATA_RAW / "mappings" / "skills.csv"
SALARIES_CSV       = DATA_RAW / "jobs" / "salaries.csv"
JOB_INDUSTRIES_CSV = DATA_RAW / "jobs" / "job_industries.csv"
INDUSTRIES_MAP_CSV = DATA_RAW / "mappings" / "industries.csv"

# Processed artifacts (legacy — kept for backward compatibility)
JOBS_FEATURES_CSV    = DATA_PROC / "jobs_features.csv"
TFIDF_MATRIX_NPZ     = DATA_PROC / "tfidf_matrix.npz"
TFIDF_VECTORIZER_PKL = DATA_PROC / "tfidf_vectorizer.pkl"
SKILL_BINARIZER_PKL  = DATA_PROC / "skill_binarizer.pkl"
FEATURE_METADATA_JSON= DATA_PROC / "feature_metadata.json"
MODEL_INDEX_PKL      = DATA_PROC / "model_index.pkl"

# New index artifacts
GRANULAR_JOBS_CSV    = DATA_PROC / "jobs_granular.csv"
BM25_INDEX_PKL       = DATA_INDEX / "bm25_index.pkl"
FAISS_INDEX_BIN      = DATA_INDEX / "faiss_index.bin"
EMBEDDINGS_NPY       = DATA_INDEX / "embeddings.npy"
JOB_ID_MAP_JSON      = DATA_INDEX / "job_id_map.json"
EMBED_CONFIG_JSON    = DATA_INDEX / "embed_config.json"

# ── Retrieval mode ────────────────────────────────────────────────────────────
# Options: "tfidf" | "bm25" | "dense" | "hybrid"
RETRIEVAL_MODE = "tfidf"   # default — keeps existing system working

# ── TF-IDF (legacy baseline) ──────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 3
TFIDF_MAX_DF       = 0.90

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_K1              = 1.5
BM25_B               = 0.75
BM25_TOP_K           = 100

# ── Dense retrieval ───────────────────────────────────────────────────────────
EMBED_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE     = 256
DENSE_TOP_K          = 100

# ── RRF ───────────────────────────────────────────────────────────────────────
RRF_K                = 60
RRF_CANDIDATE_POOL   = 200

# ── Cross-encoder reranking ───────────────────────────────────────────────────
RERANKER_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K       = 20
RERANKER_BATCH_SIZE  = 32

# ── Diversity / MMR ───────────────────────────────────────────────────────────
MMR_LAMBDA           = 0.7    # 1.0 = pure relevance, 0.0 = pure diversity
MMR_FINAL_K          = 10

# ── Scoring weights ───────────────────────────────────────────────────────────
# Initial weights — not tuned, configurable
SCORE_WEIGHTS = {
    "skill"      : 0.35,
    "semantic"   : 0.20,
    "title"      : 0.15,
    "experience" : 0.10,
    "location"   : 0.08,
    "work_type"  : 0.07,
    "salary"     : 0.05,
}

# ── Experience compatibility ──────────────────────────────────────────────────
EXP_ORDINAL = {
    "Internship"   : 0,
    "Entry"        : 1,
    "Associate"    : 2,
    "Mid-Senior"   : 3,
    "Director"     : 4,
    "Executive"    : 5,
    "Not Specified": 2,
}
# Score decay per level of distance
EXP_COMPATIBILITY = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.2, 4: 0.1, 5: 0.0}

# ── Salary ────────────────────────────────────────────────────────────────────
SALARY_MIN_VALID = 10_000
SALARY_MAX_VALID = 600_000
PAY_PERIOD_MULTIPLIERS = {
    "HOURLY"  : 2080,
    "MONTHLY" : 12,
    "WEEKLY"  : 52,
    "BIWEEKLY": 26,
    "YEARLY"  : 1,
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_N_QUERIES   = 500
EVAL_K_VALUES    = [5, 10, 20]
EVAL_RANDOM_SEED = 42

# ── Description ───────────────────────────────────────────────────────────────
DESC_MAX_CHARS   = 1000   # chars used from description in feature_text
