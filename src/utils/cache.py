"""
src/utils/cache.py

Simple LRU cache for the recommend() function.

Key = (skills_tuple, experience_level, top_n, filter_exp, filter_remote, mode)
Avoids re-running the full BM25+dense+reranker pipeline for repeated queries.

Cache is per-process (in-memory). Cleared on server restart.
"""

from functools import lru_cache
from config import MMR_FINAL_K

# Max entries — each entry holds ~top_n result dicts, small memory footprint
_CACHE_SIZE = 256


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_recommend(
    skills_key: tuple,          # hashable tuple of skill strings
    experience_level: str | None,
    top_n: int,
    filter_exp: bool,
    filter_remote: bool,
    mode: str,
    _fn,                        # the actual recommend callable (makes key unique per fn)
) -> tuple:
    """Internal cached wrapper — returns a tuple of result dicts (hashable-safe)."""
    result = _fn(
        skills=list(skills_key),
        experience_level=experience_level,
        top_n=top_n,
        filter_exp=filter_exp,
        filter_remote=filter_remote,
    )
    return result


def cached_recommend(recommend_fn, skills: list, experience_level, top_n,
                     filter_exp, filter_remote, mode: str) -> dict:
    """
    Call recommend_fn with LRU caching.
    Returns the same dict structure as recommend_fn.
    """
    key = tuple(sorted(s.lower() for s in skills))
    return _cached_recommend(key, experience_level, top_n, filter_exp,
                             filter_remote, mode, recommend_fn)


def cache_info() -> str:
    info = _cached_recommend.cache_info()
    return f"hits={info.hits} misses={info.misses} size={info.currsize}/{_CACHE_SIZE}"


def cache_clear() -> None:
    _cached_recommend.cache_clear()
