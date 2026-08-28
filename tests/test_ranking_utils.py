"""
tests/test_ranking_utils.py

Unit tests for:
  - src/ranking/reranker.py  (cross-encoder mocked)
  - src/utils/filters.py
  - src/utils/mmr.py
"""

import pytest
from unittest.mock import patch, MagicMock

from src.utils.filters import build_allowed_ids, apply_allowed
from src.utils.mmr import mmr_rerank


# ── filters ───────────────────────────────────────────────────────────────────

@pytest.fixture
def lookup():
    return {
        "j1": {"experience_level": "Mid-Senior", "is_remote": 1},
        "j2": {"experience_level": "Entry",      "is_remote": 0},
        "j3": {"experience_level": "Mid-Senior", "is_remote": 1},
        "j4": {"experience_level": "Director",   "is_remote": 0},
        "j5": {"experience_level": "Entry",      "is_remote": 1},
    }


class TestBuildAllowedIds:
    def test_no_filters_returns_none(self, lookup):
        assert build_allowed_ids(lookup) is None

    def test_exp_filter(self, lookup):
        allowed = build_allowed_ids(lookup, filter_exp=True, experience_level="Mid-Senior")
        assert allowed == {"j1", "j3"}

    def test_remote_filter(self, lookup):
        allowed = build_allowed_ids(lookup, filter_remote=True)
        assert allowed == {"j1", "j3", "j5"}

    def test_combined_filter(self, lookup):
        allowed = build_allowed_ids(
            lookup, filter_exp=True, experience_level="Entry", filter_remote=True
        )
        assert allowed == {"j5"}

    def test_no_match_returns_empty_set(self, lookup):
        allowed = build_allowed_ids(lookup, filter_exp=True, experience_level="Executive")
        assert allowed == set()

    def test_filter_exp_without_level_passes_all(self, lookup):
        # filter_exp=True but no experience_level — no exp filter applied
        allowed = build_allowed_ids(lookup, filter_exp=True, experience_level=None)
        assert allowed == set(lookup.keys())


class TestApplyAllowed:
    def test_none_allowed_is_noop(self):
        results = [{"job_id": "j1"}, {"job_id": "j2"}]
        assert apply_allowed(results, None) == results

    def test_filters_correctly(self):
        results = [{"job_id": "j1"}, {"job_id": "j2"}, {"job_id": "j3"}]
        allowed = {"j1", "j3"}
        filtered = apply_allowed(results, allowed)
        assert [r["job_id"] for r in filtered] == ["j1", "j3"]

    def test_empty_allowed_returns_empty(self):
        results = [{"job_id": "j1"}, {"job_id": "j2"}]
        assert apply_allowed(results, set()) == []

    def test_preserves_order(self):
        results = [{"job_id": f"j{i}"} for i in range(5)]
        allowed = {"j4", "j2", "j0"}
        filtered = apply_allowed(results, allowed)
        assert [r["job_id"] for r in filtered] == ["j0", "j2", "j4"]


# ── mmr ───────────────────────────────────────────────────────────────────────

def _make_results(specs):
    """specs: list of (job_id, score, skills_pipe_str)"""
    return [
        {"job_id": jid, "match_score": score, "skills": skills}
        for jid, score, skills in specs
    ]


class TestMmrRerank:
    def test_lambda_1_preserves_order(self):
        results = _make_results([
            ("j1", 0.9, "python|sql"),
            ("j2", 0.8, "python|sql"),
            ("j3", 0.7, "python|sql"),
        ])
        reranked = mmr_rerank(results, top_k=3, lam=1.0)
        assert [r["job_id"] for r in reranked] == ["j1", "j2", "j3"]

    def test_top_k_respected(self):
        results = _make_results([
            ("j1", 0.9, "python"),
            ("j2", 0.8, "sql"),
            ("j3", 0.7, "docker"),
            ("j4", 0.6, "java"),
        ])
        reranked = mmr_rerank(results, top_k=2, lam=0.7)
        assert len(reranked) == 2

    def test_diversity_promotes_different_skills(self):
        # j1 and j2 are identical skills; j3 is very different
        # with strong diversity (lam=0.1), j3 should be preferred over j2 in position 2
        # use top_k=2 so MMR actually runs (len > top_k required)
        results = _make_results([
            ("j1", 1.0,  "python|sql|pandas"),
            ("j2", 0.99, "python|sql|pandas"),   # near-duplicate of j1
            ("j3", 0.50, "docker|kubernetes|ci/cd"),  # diverse but lower score
        ])
        reranked = mmr_rerank(results, top_k=2, lam=0.1)
        ids = [r["job_id"] for r in reranked]
        # j1 always first (highest score); j3 should beat j2 due to diversity
        assert ids[0] == "j1"
        assert ids[1] == "j3"

    def test_fewer_results_than_top_k(self):
        results = _make_results([("j1", 0.9, "python"), ("j2", 0.8, "sql")])
        reranked = mmr_rerank(results, top_k=10, lam=0.7)
        assert len(reranked) == 2

    def test_empty_input(self):
        assert mmr_rerank([], top_k=5, lam=0.7) == []

    def test_single_result(self):
        results = _make_results([("j1", 0.9, "python")])
        reranked = mmr_rerank(results, top_k=5, lam=0.7)
        assert len(reranked) == 1
        assert reranked[0]["job_id"] == "j1"

    def test_empty_skills_handled(self):
        results = _make_results([
            ("j1", 0.9, ""),
            ("j2", 0.8, ""),
            ("j3", 0.7, ""),
        ])
        reranked = mmr_rerank(results, top_k=3, lam=0.5)
        assert len(reranked) == 3


# ── reranker (mocked) ─────────────────────────────────────────────────────────

class TestReranker:
    def test_rerank_returns_sorted_by_score(self):
        from src.ranking.reranker import rerank

        candidates = [
            {"job_id": "j1", "rrf_score": 0.9, "rank": 1},
            {"job_id": "j2", "rrf_score": 0.8, "rank": 2},
            {"job_id": "j3", "rrf_score": 0.7, "rank": 3},
        ]
        lookup = {
            "j1": {"title_clean": "data scientist", "granular_skills": "python|sql",
                   "industry": "Tech", "experience_level": "Mid-Senior"},
            "j2": {"title_clean": "ml engineer", "granular_skills": "python|pytorch",
                   "industry": "Tech", "experience_level": "Mid-Senior"},
            "j3": {"title_clean": "analyst", "granular_skills": "sql|excel",
                   "industry": "Finance", "experience_level": "Entry"},
        }

        mock_scores = [0.5, 0.9, 0.3]  # j2 should rank first after reranking

        with patch("src.ranking.reranker._get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_get.return_value = mock_model

            result = rerank("python machine learning", candidates, lookup, top_k=3)

        assert result[0]["job_id"] == "j2"
        scores = [r["reranker_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_top_k_respected(self):
        from src.ranking.reranker import rerank

        candidates = [{"job_id": f"j{i}", "rank": i + 1} for i in range(5)]
        lookup = {
            f"j{i}": {"title_clean": f"job {i}", "granular_skills": "python",
                      "industry": "Tech", "experience_level": "Entry"}
            for i in range(5)
        }

        with patch("src.ranking.reranker._get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
            mock_get.return_value = mock_model

            result = rerank("python", candidates, lookup, top_k=2)

        assert len(result) == 2

    def test_rerank_missing_lookup_skipped(self):
        from src.ranking.reranker import rerank

        candidates = [
            {"job_id": "j1", "rank": 1},
            {"job_id": "j_missing", "rank": 2},
        ]
        lookup = {
            "j1": {"title_clean": "engineer", "granular_skills": "python",
                   "industry": "Tech", "experience_level": "Entry"},
        }

        with patch("src.ranking.reranker._get_model") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.8]
            mock_get.return_value = mock_model

            result = rerank("python", candidates, lookup, top_k=5)

        assert len(result) == 1
        assert result[0]["job_id"] == "j1"


class TestSalaryFilter:
    @pytest.fixture
    def salary_lookup(self):
        return {
            "j1": {"experience_level": "Mid-Senior", "is_remote": 0, "salary_mid": 80000.0},
            "j2": {"experience_level": "Entry",      "is_remote": 0, "salary_mid": 50000.0},
            "j3": {"experience_level": "Mid-Senior", "is_remote": 1, "salary_mid": 150000.0},
            "j4": {"experience_level": "Director",   "is_remote": 0, "salary_mid": None},
            "j5": {"experience_level": "Entry",      "is_remote": 1, "salary_mid": 60000.0},
        }

    def test_salary_min_filter(self, salary_lookup):
        allowed = build_allowed_ids(salary_lookup, salary_min=70000.0)
        assert allowed == {"j1", "j3"}

    def test_salary_max_filter(self, salary_lookup):
        allowed = build_allowed_ids(salary_lookup, salary_max=60000.0)
        assert allowed == {"j2", "j5"}

    def test_salary_range_filter(self, salary_lookup):
        allowed = build_allowed_ids(salary_lookup, salary_min=55000.0, salary_max=100000.0)
        assert allowed == {"j1", "j5"}

    def test_no_salary_data_excluded(self, salary_lookup):
        # j4 has salary_mid=None — excluded when salary filter is active
        allowed = build_allowed_ids(salary_lookup, salary_min=0.0)
        assert "j4" not in allowed

    def test_salary_combined_with_remote(self, salary_lookup):
        allowed = build_allowed_ids(salary_lookup, filter_remote=True, salary_min=70000.0)
        assert allowed == {"j3"}
