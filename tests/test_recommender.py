"""
tests/test_recommender.py
Run locally : pytest tests/
Run by CI   : automatically on every git push via GitHub Actions
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from recommender import load_model, recommend


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_loads(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    assert vectorizer is not None
    assert job_vectors.shape[0] > 0
    assert not svd_predictions.empty
    assert not jobs.empty


def test_existing_user_returns_results(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    user_id = svd_predictions.index[0]
    results = recommend(vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs,
                        user_id=user_id, top_k=5)
    assert len(results) == 5
    assert results[0]["mode"] == "hybrid"


def test_cold_start_returns_results(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    results = recommend(vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs,
                        user_skills="Python, Machine Learning, Data Analysis", top_k=5)
    assert len(results) == 5
    assert results[0]["mode"] == "content-based"


def test_recommendation_fields(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    results = recommend(vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs,
                        user_skills="Finance, Accounting/Auditing", top_k=3)
    for r in results:
        assert "job_id" in r
        assert "job_title" in r
        assert "skills" in r
        assert "location" in r
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.5


def test_scores_are_descending(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    results = recommend(vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs,
                        user_skills="Sales, Marketing", top_k=8)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_no_input_raises_error(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    with pytest.raises(ValueError):
        recommend(vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs)


def test_top_k_respected(model):
    vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs = model
    for k in [1, 5, 10]:
        results = recommend(vectorizer, job_vectors, svd_predictions, interaction_matrix, jobs,
                            user_skills="Management, Strategy/Planning", top_k=k)
        assert len(results) == k
