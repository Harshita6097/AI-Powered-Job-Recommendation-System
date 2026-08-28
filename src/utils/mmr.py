"""
src/utils/mmr.py

Maximal Marginal Relevance (MMR) diversity reranking.

Reduces redundancy in the final result list by penalising candidates that
are too similar to already-selected results.

Formula:
    MMR(d) = lambda * relevance(d) - (1 - lambda) * max_similarity(d, selected)

lambda=1.0 -> pure relevance order (no diversity)
lambda=0.0 -> pure diversity (greedy furthest-first)

Uses granular skill Jaccard similarity as the diversity measure — fast,
interpretable, no extra model needed.
"""

from __future__ import annotations
from config import MMR_LAMBDA, MMR_FINAL_K


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _skill_set(skills_str: str) -> set[str]:
    if not skills_str or not isinstance(skills_str, str):
        return set()
    return {s.strip().lower() for s in skills_str.split("|") if s.strip()}


def mmr_rerank(
    results: list[dict],
    top_k: int = MMR_FINAL_K,
    lam: float = MMR_LAMBDA,
    score_key: str = "match_score",
    skills_key: str = "skills",
) -> list[dict]:
    """
    Apply MMR to a ranked result list.

    Parameters
    ----------
    results    : list of result dicts, already sorted by relevance descending.
    top_k      : number of results to return.
    lam        : lambda — trade-off between relevance and diversity.
    score_key  : field name for the relevance score.
    skills_key : field name for pipe-separated skill string.

    Returns
    -------
    Re-ranked list of length min(top_k, len(results)).
    """
    if lam >= 1.0 or len(results) <= top_k:
        return results[:top_k]

    # normalise scores to [0, 1]
    scores = [r[score_key] for r in results]
    max_s, min_s = max(scores), min(scores)
    rng = max_s - min_s or 1.0
    norm = [(s - min_s) / rng for s in scores]

    skill_sets = [_skill_set(r.get(skills_key, "")) for r in results]

    selected_idx: list[int] = []
    selected_skills: list[set] = []

    remaining = list(range(len(results)))

    for _ in range(min(top_k, len(results))):
        best_idx, best_score = -1, float("-inf")
        for i in remaining:
            rel = norm[i]
            if not selected_skills:
                sim = 0.0
            else:
                sim = max(_jaccard(skill_sets[i], s) for s in selected_skills)
            mmr_score = lam * rel - (1 - lam) * sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i

        selected_idx.append(best_idx)
        selected_skills.append(skill_sets[best_idx])
        remaining.remove(best_idx)

    return [results[i] for i in selected_idx]
