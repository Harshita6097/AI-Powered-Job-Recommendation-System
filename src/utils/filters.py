"""
src/utils/filters.py

Pre-retrieval candidate filtering.

Builds a set of allowed job_ids from jobs_granular before retrieval runs,
so BM25/dense/hybrid only score jobs that pass the filter — ensuring top_n
results are always returned even with strict filters.
"""

from __future__ import annotations


def build_allowed_ids(
    lookup: dict,
    filter_exp: bool = False,
    experience_level: str | None = None,
    filter_remote: bool = False,
    salary_min: float | None = None,
    salary_max: float | None = None,
) -> set[str] | None:
    """
    Return a set of job_ids that pass all active filters, or None if no
    filters are active (meaning all jobs are allowed — skip the check).

    Parameters
    ----------
    lookup           : job_id -> row dict (from engine._lookup)
    filter_exp       : if True, restrict to jobs matching experience_level
    experience_level : target experience level string
    filter_remote    : if True, restrict to remote jobs only
    salary_min       : if set, exclude jobs with salary_mid below this value
    salary_max       : if set, exclude jobs with salary_mid above this value
    """
    if not filter_exp and not filter_remote and salary_min is None and salary_max is None:
        return None   # no filtering — caller skips the check entirely

    allowed = set()
    for jid, row in lookup.items():
        if filter_exp and experience_level:
            if row.get("experience_level") != experience_level:
                continue
        if filter_remote:
            if not row.get("is_remote"):
                continue
        if salary_min is not None or salary_max is not None:
            mid = row.get("salary_mid")
            # jobs without salary data are excluded when a salary filter is active
            if mid is None or (isinstance(mid, float) and mid != mid):  # NaN check
                continue
            if salary_min is not None and mid < salary_min:
                continue
            if salary_max is not None and mid > salary_max:
                continue
        allowed.add(jid)
    return allowed


def apply_allowed(results: list[dict], allowed: set[str] | None) -> list[dict]:
    """Filter a result list to only allowed job_ids. No-op if allowed is None."""
    if allowed is None:
        return results
    return [r for r in results if r["job_id"] in allowed]
