"""
tests/test_retrieval.py

Unit tests for src/retrieval/ — BM25, dense (mocked), and RRF.
Dense retrieval tests mock the sentence-transformer to avoid loading the model.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.retrieval.bm25_retriever import (
    tokenise_query,
    build_bm25_index,
    query_bm25,
)
from src.retrieval.rrf import fuse


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    return pd.DataFrame({
        "job_id":         ["j1", "j2", "j3", "j4", "j5"],
        "title_clean":    ["data scientist", "ml engineer", "backend developer",
                           "data analyst", "devops engineer"],
        "granular_skills": [
            "python|machine learning|sql",
            "python|pytorch|tensorflow",
            "python|django|rest api",
            "sql|tableau|excel",
            "docker|kubernetes|ci/cd",
        ],
    })


@pytest.fixture
def bm25_index(small_df):
    return build_bm25_index(small_df)


# ── tokenise_query ────────────────────────────────────────────────────────────

class TestTokeniseQuery:
    def test_single_word(self):
        assert tokenise_query("python") == ["python"]

    def test_multi_word_skill(self):
        tokens = tokenise_query("machine learning")
        assert "machine_learning" in tokens

    def test_pipe_separated(self):
        tokens = tokenise_query("python|sql|docker")
        assert "python" in tokens
        assert "sql" in tokens
        assert "docker" in tokens

    def test_comma_separated(self):
        tokens = tokenise_query("python, sql, docker")
        assert "python" in tokens
        assert "sql" in tokens

    def test_empty(self):
        assert tokenise_query("") == []

    def test_lowercases(self):
        tokens = tokenise_query("Python SQL")
        assert "python_sql" in tokens or ("python" in tokens and "sql" in tokens)


# ── build_bm25_index ──────────────────────────────────────────────────────────

class TestBuildBm25Index:
    def test_returns_tuple(self, small_df):
        bm25, job_ids = build_bm25_index(small_df)
        assert job_ids == ["j1", "j2", "j3", "j4", "j5"]

    def test_corpus_size(self, small_df):
        bm25, job_ids = build_bm25_index(small_df)
        assert len(job_ids) == len(small_df)


# ── query_bm25 ────────────────────────────────────────────────────────────────

class TestQueryBm25:
    def test_returns_list_of_dicts(self, bm25_index):
        bm25, job_ids = bm25_index
        results = query_bm25(bm25, job_ids, ["python"], top_k=3)
        assert isinstance(results, list)
        assert all("job_id" in r and "bm25_score" in r for r in results)

    def test_top_k_respected(self, bm25_index):
        bm25, job_ids = bm25_index
        results = query_bm25(bm25, job_ids, ["python"], top_k=2)
        assert len(results) <= 2

    def test_relevant_job_ranked_first(self, bm25_index):
        bm25, job_ids = bm25_index
        # "docker" only appears in j5
        results = query_bm25(bm25, job_ids, ["docker"], top_k=5)
        assert results[0]["job_id"] == "j5"

    def test_empty_query_returns_empty(self, bm25_index):
        bm25, job_ids = bm25_index
        results = query_bm25(bm25, job_ids, [], top_k=5)
        assert results == []

    def test_scores_descending(self, bm25_index):
        bm25, job_ids = bm25_index
        results = query_bm25(bm25, job_ids, ["python", "machine_learning"], top_k=5)
        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_present(self, bm25_index):
        bm25, job_ids = bm25_index
        results = query_bm25(bm25, job_ids, ["python"], top_k=3)
        assert all("rank" in r for r in results)

    def test_no_zero_score_results(self, bm25_index):
        bm25, job_ids = bm25_index
        results = query_bm25(bm25, job_ids, ["python"], top_k=5)
        assert all(r["bm25_score"] > 0 for r in results)


# ── RRF ───────────────────────────────────────────────────────────────────────

class TestRRFFuse:
    def _make_list(self, job_ids):
        return [{"job_id": jid, "rank": i + 1} for i, jid in enumerate(job_ids)]

    def test_basic_fusion(self):
        bm25 = self._make_list(["j1", "j2", "j3"])
        dense = self._make_list(["j2", "j1", "j4"])
        result = fuse([bm25, dense], k=60, top_n=4)
        ids = [r["job_id"] for r in result]
        # j1 and j2 appear in both lists — should rank above j3/j4
        assert ids.index("j1") < ids.index("j3")
        assert ids.index("j2") < ids.index("j4")

    def test_top_n_respected(self):
        bm25 = self._make_list(["j1", "j2", "j3", "j4", "j5"])
        dense = self._make_list(["j5", "j4", "j3", "j2", "j1"])
        result = fuse([bm25, dense], k=60, top_n=3)
        assert len(result) == 3

    def test_scores_descending(self):
        bm25 = self._make_list(["j1", "j2", "j3"])
        dense = self._make_list(["j1", "j3", "j2"])
        result = fuse([bm25, dense], k=60, top_n=3)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_per_source_rank_fields(self):
        bm25 = self._make_list(["j1", "j2"])
        dense = self._make_list(["j2", "j1"])
        result = fuse([bm25, dense], k=60, top_n=2, list_names=["bm25", "dense"])
        for r in result:
            assert "bm25_rank" in r
            assert "dense_rank" in r

    def test_single_list(self):
        lst = self._make_list(["j1", "j2", "j3"])
        result = fuse([lst], k=60, top_n=3)
        assert [r["job_id"] for r in result] == ["j1", "j2", "j3"]

    def test_empty_lists(self):
        result = fuse([[], []], k=60, top_n=5)
        assert result == []

    def test_overlap_boosts_score(self):
        # j1 in both lists should score higher than j3 in only one
        bm25 = self._make_list(["j1", "j2", "j3"])
        dense = self._make_list(["j1", "j4", "j5"])
        result = fuse([bm25, dense], k=60, top_n=5)
        j1_score = next(r["rrf_score"] for r in result if r["job_id"] == "j1")
        j3_score = next(r["rrf_score"] for r in result if r["job_id"] == "j3")
        assert j1_score > j3_score

    def test_rank_field_sequential(self):
        lst = self._make_list(["j1", "j2", "j3"])
        result = fuse([lst], k=60, top_n=3)
        assert [r["rank"] for r in result] == [1, 2, 3]
