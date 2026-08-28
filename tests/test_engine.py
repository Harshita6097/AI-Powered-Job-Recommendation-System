"""
tests/test_engine.py

Unit tests for src/inference/engine.py
All heavy artifacts (BM25, FAISS, CSV) are mocked so tests run without indexes.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_lookup(n=5):
    return {
        f"j{i}": {
            "title_clean":      f"job title {i}",
            "experience_level": "Mid-Senior" if i % 2 == 0 else "Entry",
            "industry":         "Tech",
            "granular_skills":  "python|sql" if i < 3 else "docker|kubernetes",
            "broad_skills_str": "Information Technology",
            "salary_mid":       80000.0,
            "has_salary":       True,
            "work_type":        "Full-time",
            "is_remote":        bool(i % 2),
            "state":            "California",
        }
        for i in range(n)
    }


def _make_df(n=5):
    rows = []
    for i in range(n):
        rows.append({
            "job_id":           f"j{i}",
            "title_clean":      f"job title {i}",
            "experience_level": "Mid-Senior" if i % 2 == 0 else "Entry",
            "industry":         "Tech",
            "granular_skills":  "python|sql" if i < 3 else "docker|kubernetes",
            "broad_skills_str": "Information Technology",
            "salary_mid":       80000.0,
            "has_salary":       True,
            "work_type":        "Full-time",
            "is_remote":        i % 2,
            "state":            "California",
        })
    return pd.DataFrame(rows)


def _bm25_results(job_ids):
    return [{"job_id": jid, "bm25_score": 1.0 - i * 0.1, "rank": i + 1}
            for i, jid in enumerate(job_ids)]


def _dense_results(job_ids):
    return [{"job_id": jid, "dense_score": 0.9 - i * 0.1, "rank": i + 1}
            for i, jid in enumerate(job_ids)]


# ── skill_gap ─────────────────────────────────────────────────────────────────

class TestSkillGap:
    def test_matched_and_missing(self):
        from src.inference.engine import skill_gap
        result = skill_gap(["python", "sql"], "python|sql|docker|kubernetes")
        assert "python" in result["matched_skills"]
        assert "sql" in result["matched_skills"]
        assert "docker" in result["missing_skills"]
        assert "kubernetes" in result["missing_skills"]

    def test_match_pct_full(self):
        from src.inference.engine import skill_gap
        result = skill_gap(["python", "sql"], "python|sql")
        assert result["match_pct"] == 100.0

    def test_match_pct_zero(self):
        from src.inference.engine import skill_gap
        result = skill_gap(["java"], "python|sql|docker")
        assert result["match_pct"] == 0.0

    def test_match_pct_partial(self):
        from src.inference.engine import skill_gap
        result = skill_gap(["python"], "python|sql|docker")
        assert result["match_pct"] == pytest.approx(33.3, abs=0.1)

    def test_empty_job_skills(self):
        from src.inference.engine import skill_gap
        result = skill_gap(["python"], "")
        assert result["match_pct"] == 0.0


# ── cache utilities ───────────────────────────────────────────────────────────

class TestCacheUtils:
    def test_cache_info_format(self):
        from src.inference.engine import cache_info
        info = cache_info()
        assert "hits=" in info
        assert "misses=" in info
        assert "size=" in info

    def test_cache_clear(self):
        from src.inference.engine import cache_clear, cache_info
        cache_clear()
        info = cache_info()
        assert "size=0" in info


# ── recommend (bm25 mode, mocked) ─────────────────────────────────────────────

class TestRecommendBm25:
    def _patch_and_run(self, skills, mode="bm25", **kwargs):
        import src.inference.engine as eng

        lookup = _make_lookup()
        df = _make_df()
        job_ids = list(lookup.keys())

        with patch.object(eng, "_df", df), \
             patch.object(eng, "_lookup", lookup), \
             patch.object(eng, "_bm25", MagicMock()), \
             patch.object(eng, "_bm25_ids", job_ids), \
             patch("src.inference.engine.config") as mock_cfg, \
             patch("src.inference.engine.query_bm25",
                   return_value=_bm25_results(job_ids)):
            mock_cfg.RETRIEVAL_MODE = mode
            mock_cfg.MMR_LAMBDA = 1.0   # disable MMR for determinism
            eng.cache_clear()
            result = eng.recommend(skills, **kwargs)

        return result

    def test_returns_dict_keys(self):
        result = self._patch_and_run(["python"])
        assert "results" in result
        assert "retrieval_mode" in result
        assert "latency_ms" in result

    def test_top_n_respected(self):
        result = self._patch_and_run(["python"], top_n=3)
        assert len(result["results"]) <= 3

    def test_result_fields(self):
        result = self._patch_and_run(["python"])
        if result["results"]:
            r = result["results"][0]
            assert "job_id" in r
            assert "title" in r
            assert "match_score" in r

    def test_empty_skills_still_runs(self):
        # engine should handle empty skills gracefully
        result = self._patch_and_run([])
        assert "results" in result

    def test_mode_in_response(self):
        result = self._patch_and_run(["python"], mode="bm25")
        assert result["retrieval_mode"] == "bm25"


# ── recommend (hybrid mode, mocked) ───────────────────────────────────────────

class TestRecommendHybrid:
    def test_hybrid_mode(self):
        import src.inference.engine as eng

        lookup = _make_lookup()
        df = _make_df()
        job_ids = list(lookup.keys())

        rrf_pool = [{"job_id": jid, "rrf_score": 0.9 - i * 0.1, "rank": i + 1,
                     "bm25_rank": i + 1, "dense_rank": i + 1}
                    for i, jid in enumerate(job_ids)]
        reranked = [{**r, "reranker_score": r["rrf_score"]} for r in rrf_pool[:3]]

        with patch.object(eng, "_df", df), \
             patch.object(eng, "_lookup", lookup), \
             patch.object(eng, "_bm25", MagicMock()), \
             patch.object(eng, "_bm25_ids", job_ids), \
             patch.object(eng, "_faiss_idx", MagicMock()), \
             patch.object(eng, "_dense_ids", job_ids), \
             patch("src.inference.engine.config") as mock_cfg, \
             patch("src.inference.engine.query_bm25",
                   return_value=_bm25_results(job_ids)), \
             patch("src.inference.engine.query_dense",
                   return_value=_dense_results(job_ids)), \
             patch("src.inference.engine.fuse", return_value=rrf_pool), \
             patch("src.inference.engine.rerank", return_value=reranked):
            mock_cfg.RETRIEVAL_MODE = "hybrid"
            mock_cfg.MMR_LAMBDA = 1.0
            eng.cache_clear()
            result = eng.recommend(["python", "sql"])

        assert result["retrieval_mode"] == "hybrid"
        assert len(result["results"]) > 0
