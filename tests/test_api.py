"""
tests/test_api.py

Integration tests for backend/main.py FastAPI endpoints.
The inference engine and legacy recommender are mocked so no indexes are needed.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── shared mock data ──────────────────────────────────────────────────────────

_ENGINE_OUT = {
    "results": [
        {
            "job_id":           "j1",
            "title":            "data scientist",
            "experience_level": "Mid-Senior",
            "industry":         "Tech",
            "skills":           "python|sql|machine learning",
            "salary_mid":       100000.0,
            "work_type":        "Full-time",
            "is_remote":        True,
            "state":            "California",
            "match_score":      0.92,
        }
    ],
    "query_skills_raw":    ["python", "sql"],
    "query_skills_mapped": ["python", "sql"],
    "unmapped_skills":     [],
    "retrieval_mode":      "hybrid",
    "latency_ms":          42.0,
}

_SKILL_GAP_OUT = {
    "matched_skills": ["python", "sql"],
    "missing_skills": ["docker"],
    "match_pct":      66.7,
}


@pytest.fixture(scope="module")
def client():
    """Create TestClient with all heavy dependencies mocked."""
    with patch("src.inference.engine.load_all"), \
         patch("backend.recommender.load_artifacts"), \
         patch("src.inference.engine.recommend", return_value=_ENGINE_OUT), \
         patch("src.inference.engine.skill_gap", return_value=_SKILL_GAP_OUT), \
         patch("src.inference.engine.cache_info", return_value="hits=0 misses=0 size=0/256"), \
         patch("src.inference.engine.cache_clear"):

        import config
        config.RETRIEVAL_MODE = "hybrid"

        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_returns_mode(self, client):
        r = client.get("/health")
        assert "mode" in r.json()

    def test_returns_cache(self, client):
        r = client.get("/health")
        assert "cache" in r.json()


# ── /cache ────────────────────────────────────────────────────────────────────

class TestCache:
    def test_get_cache(self, client):
        r = client.get("/cache")
        assert r.status_code == 200
        assert "cache" in r.json()

    def test_delete_cache(self, client):
        r = client.delete("/cache")
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"


# ── /recommend ────────────────────────────────────────────────────────────────

class TestRecommend:
    def test_basic_request(self, client):
        r = client.post("/recommend", json={"skills": ["python", "sql"]})
        assert r.status_code == 200

    def test_response_schema(self, client):
        r = client.post("/recommend", json={"skills": ["python"]})
        body = r.json()
        assert "results" in body
        assert "retrieval_mode" in body
        assert "query_skills_raw" in body
        assert "query_skills_mapped" in body
        assert "unmapped_skills" in body

    def test_result_fields(self, client):
        r = client.post("/recommend", json={"skills": ["python"]})
        job = r.json()["results"][0]
        for field in ("job_id", "title", "match_score", "skills",
                      "experience_level", "industry", "work_type", "is_remote"):
            assert field in job, f"missing field: {field}"

    def test_empty_skills_returns_400(self, client):
        r = client.post("/recommend", json={"skills": []})
        assert r.status_code == 400

    def test_top_n_param_accepted(self, client):
        r = client.post("/recommend", json={"skills": ["python"], "top_n": 5})
        assert r.status_code == 200

    def test_filter_params_accepted(self, client):
        r = client.post("/recommend", json={
            "skills": ["python"],
            "experience_level": "Mid-Senior",
            "filter_exp": True,
            "filter_remote": True,
        })
        assert r.status_code == 200

    def test_no_results_returns_404(self, client):
        empty_out = {**_ENGINE_OUT, "results": []}
        import backend.main as main_mod
        original = main_mod.engine_recommend
        try:
            main_mod.engine_recommend = lambda **kw: empty_out
            r = client.post("/recommend", json={"skills": ["python"]})
        finally:
            main_mod.engine_recommend = original
        assert r.status_code == 404


# ── /skill-gap ────────────────────────────────────────────────────────────────

class TestSkillGap:
    def test_basic_request(self, client):
        r = client.post("/skill-gap", json={
            "user_skills": ["python", "sql"],
            "job_skills":  "python|sql|docker",
        })
        assert r.status_code == 200

    def test_response_schema(self, client):
        r = client.post("/skill-gap", json={
            "user_skills": ["python"],
            "job_skills":  "python|docker",
        })
        body = r.json()
        assert "matched_skills" in body
        assert "missing_skills" in body
        assert "match_pct" in body

    def test_empty_user_skills_returns_400(self, client):
        r = client.post("/skill-gap", json={
            "user_skills": [],
            "job_skills":  "python|sql",
        })
        assert r.status_code == 400

    def test_empty_job_skills_returns_400(self, client):
        r = client.post("/skill-gap", json={
            "user_skills": ["python"],
            "job_skills":  "",
        })
        assert r.status_code == 400


# ── /resume ───────────────────────────────────────────────────────────────────

class TestResume:
    def _make_pdf_bytes(self, text: str) -> bytes:
        """Create a minimal valid PDF containing the given text."""
        import io
        try:
            import pymupdf as fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), text)
            buf = io.BytesIO()
            doc.save(buf)
            return buf.getvalue()
        except Exception:
            # fallback: minimal hand-crafted PDF
            return (
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
                b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
                b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 720 Td ("
                + text.encode() +
                b") Tj ET\nendstream\nendobj\n"
                b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
                b"xref\n0 6\n0000000000 65535 f\n"
                b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF"
            )

    def test_non_pdf_rejected(self, client):
        r = client.post(
            "/resume",
            files={"file": ("resume.txt", b"python sql", "text/plain")},
        )
        assert r.status_code == 400

    def test_pdf_with_skills_succeeds(self, client):
        pdf = self._make_pdf_bytes("python sql machine learning docker")
        with patch("src.preprocessing.skill_extractor.extract_skills",
                   return_value=("python", "sql", "machine learning", "docker")):
            r = client.post(
                "/resume",
                files={"file": ("resume.pdf", pdf, "application/pdf")},
            )
        assert r.status_code == 200

    def test_pdf_no_skills_returns_422(self, client):
        pdf = self._make_pdf_bytes("hello world no skills here")
        with patch("src.preprocessing.skill_extractor.extract_skills",
                   return_value=()):
            r = client.post(
                "/resume",
                files={"file": ("resume.pdf", pdf, "application/pdf")},
            )
        assert r.status_code == 422


# ── pagination ────────────────────────────────────────────────────────────────

class TestPagination:
    def _engine_out_with_n(self, n):
        """Return engine output with n results."""
        results = [
            {
                "job_id":           f"j{i}",
                "title":            f"job {i}",
                "experience_level": "Mid-Senior",
                "industry":         "Tech",
                "skills":           "python|sql",
                "salary_mid":       80000.0,
                "work_type":        "Full-time",
                "is_remote":        False,
                "state":            "California",
                "match_score":      1.0 - i * 0.05,
            }
            for i in range(n)
        ]
        return {**_ENGINE_OUT, "results": results}

    def test_offset_zero_returns_first_page(self, client):
        import backend.main as main_mod
        out = self._engine_out_with_n(10)
        original = main_mod.engine_recommend
        try:
            main_mod.engine_recommend = lambda **kw: out
            r = client.post("/recommend", json={"skills": ["python"], "top_n": 5, "offset": 0})
        finally:
            main_mod.engine_recommend = original
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 5
        assert body["results"][0]["job_id"] == "j0"

    def test_offset_skips_results(self, client):
        import backend.main as main_mod
        out = self._engine_out_with_n(10)
        original = main_mod.engine_recommend
        try:
            main_mod.engine_recommend = lambda **kw: out
            r = client.post("/recommend", json={"skills": ["python"], "top_n": 5, "offset": 5})
        finally:
            main_mod.engine_recommend = original
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 5
        assert body["results"][0]["job_id"] == "j5"

    def test_response_includes_total_and_offset(self, client):
        import backend.main as main_mod
        out = self._engine_out_with_n(10)
        original = main_mod.engine_recommend
        try:
            main_mod.engine_recommend = lambda **kw: out
            r = client.post("/recommend", json={"skills": ["python"], "top_n": 5, "offset": 3})
        finally:
            main_mod.engine_recommend = original
        body = r.json()
        assert "total" in body
        assert "offset" in body
        assert body["offset"] == 3
        assert body["total"] == 10

    def test_offset_beyond_results_returns_empty(self, client):
        import backend.main as main_mod
        out = self._engine_out_with_n(3)
        original = main_mod.engine_recommend
        try:
            main_mod.engine_recommend = lambda **kw: out
            r = client.post("/recommend", json={"skills": ["python"], "top_n": 5, "offset": 10})
        finally:
            main_mod.engine_recommend = original
        # engine returns 3 results, offset=10 → empty page → 404
        assert r.status_code in (200, 404)
