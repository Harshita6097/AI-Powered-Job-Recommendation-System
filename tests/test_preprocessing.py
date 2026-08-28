"""
tests/test_preprocessing.py

Unit tests for src/preprocessing/cleaner.py and skill_extractor.py
"""

import pytest
from src.preprocessing.cleaner import (
    normalise_title,
    normalise_location,
    normalise_experience,
    normalise_work_type,
    normalise_salary,
    clean_description,
)
from src.preprocessing.skill_extractor import (
    extract_skills,
    extract_broad_categories,
    extract_skills_from_job,
    SKILL_DICT,
)


# ── cleaner ───────────────────────────────────────────────────────────────────

class TestNormaliseTitle:
    def test_lowercases(self):
        assert normalise_title("Senior Data Scientist") == "senior data scientist"

    def test_strips_whitespace(self):
        assert normalise_title("  ML Engineer  ") == "ml engineer"

    def test_collapses_spaces(self):
        assert normalise_title("Data  Science  Lead") == "data science lead"

    def test_removes_special_chars(self):
        assert normalise_title("Engineer @ FAANG!") == "engineer  faang"

    def test_empty_string(self):
        assert normalise_title("") == ""


class TestNormaliseLocation:
    def test_city_state_abbr(self):
        result = normalise_location("San Francisco, CA")
        assert result["city"] == "San Francisco"
        assert result["state"] == "California"
        assert result["country"] == "United States"

    def test_city_country(self):
        result = normalise_location("London, UK")
        assert result["city"] == "London"
        assert result["country"] == "UK"

    def test_three_parts(self):
        result = normalise_location("Austin, TX, United States")
        assert result["state"] == "Texas"
        assert result["country"] == "United States"

    def test_empty(self):
        result = normalise_location("")
        assert result["city"] is None

    def test_nan(self):
        import numpy as np
        result = normalise_location(np.nan)
        assert result["city"] is None


class TestNormaliseExperience:
    @pytest.mark.parametrize("raw,expected", [
        ("Mid-Senior level", "Mid-Senior"),
        ("Entry level",      "Entry"),
        ("Internship",       "Internship"),
        ("Director",         "Director"),
        ("unknown",          "Not Specified"),
        ("",                 "Not Specified"),
    ])
    def test_mapping(self, raw, expected):
        assert normalise_experience(raw) == expected


class TestNormaliseWorkType:
    @pytest.mark.parametrize("raw,expected", [
        ("Full-time",  "Full-time"),
        ("fulltime",   "Full-time"),
        ("Part-time",  "Part-time"),
        ("Contract",   "Contract"),
        ("Temporary",  "Temporary"),
        ("temp",       "Temporary"),
    ])
    def test_mapping(self, raw, expected):
        assert normalise_work_type(raw) == expected


class TestNormaliseSalary:
    def test_yearly_passthrough(self):
        result = normalise_salary(80000, 120000, "YEARLY")
        assert result["salary_min"] == 80000
        assert result["salary_max"] == 120000
        assert result["salary_mid"] == 100000
        assert result["salary_yearly_valid"] is True

    def test_hourly_conversion(self):
        result = normalise_salary(50, 70, "HOURLY")
        assert result["salary_min"] == pytest.approx(50 * 2080)
        assert result["salary_max"] == pytest.approx(70 * 2080)

    def test_out_of_range_invalidated(self):
        result = normalise_salary(1, 2, "YEARLY")
        assert result["salary_yearly_valid"] is False

    def test_missing_values(self):
        import numpy as np
        result = normalise_salary(np.nan, np.nan, "YEARLY")
        assert result["salary_yearly_valid"] is False


class TestCleanDescription:
    def test_removes_html(self):
        assert "<b>" not in clean_description("<b>Python</b> developer")

    def test_removes_urls(self):
        assert "http" not in clean_description("visit http://example.com for details")

    def test_truncates(self):
        long_text = "a " * 2000
        assert len(clean_description(long_text, max_chars=100)) <= 100

    def test_lowercases(self):
        assert clean_description("Python SQL") == "python sql"


# ── skill_extractor ───────────────────────────────────────────────────────────

class TestExtractSkills:
    def test_basic_skills(self):
        skills = extract_skills("python sql machine learning")
        assert "python" in skills
        assert "sql" in skills
        assert "machine learning" in skills

    def test_alias_resolution(self):
        # "ml" should resolve to "machine learning"
        skills = extract_skills("experience with ml and k8s")
        assert "machine learning" in skills
        assert "kubernetes" in skills

    def test_returns_tuple(self):
        assert isinstance(extract_skills("python"), tuple)

    def test_empty_text(self):
        assert extract_skills("") == ()
        assert extract_skills("   ") == ()

    def test_no_false_positive_go(self):
        # bare "go" should NOT match golang
        skills = extract_skills("go to the office")
        assert "golang" not in skills

    def test_no_false_positive_r(self):
        # bare "r" should NOT match r programming
        skills = extract_skills("r is a letter")
        assert "r programming" not in skills

    def test_golang_explicit(self):
        skills = extract_skills("golang developer")
        assert "golang" in skills

    def test_r_programming_explicit(self):
        skills = extract_skills("r programming and statistics")
        assert "r programming" in skills

    def test_case_insensitive(self):
        skills = extract_skills("PYTHON SQL DOCKER")
        assert "python" in skills
        assert "sql" in skills
        assert "docker" in skills

    def test_deduplication(self):
        skills = extract_skills("python python python")
        assert skills.count("python") == 1

    def test_lru_cache(self):
        # same input should return identical object (cached)
        r1 = extract_skills("python docker")
        r2 = extract_skills("python docker")
        assert r1 is r2


class TestExtractBroadCategories:
    def test_maps_to_category(self):
        cats = extract_broad_categories(["python", "sql"])
        assert "Information Technology" in cats

    def test_unknown_skill_ignored(self):
        cats = extract_broad_categories(["nonexistent_skill_xyz"])
        assert cats == []

    def test_deduplicates_categories(self):
        cats = extract_broad_categories(["python", "java", "javascript"])
        assert cats.count("Information Technology") == 1


class TestExtractSkillsFromJob:
    def test_returns_dict_keys(self):
        result = extract_skills_from_job("Data Scientist", "python machine learning sql")
        assert "granular_skills" in result
        assert "broad_categories" in result

    def test_granular_is_list(self):
        result = extract_skills_from_job("Engineer", "docker kubernetes")
        assert isinstance(result["granular_skills"], list)

    def test_skill_dict_coverage(self):
        # Skills extractable from their own name — skip those with special chars
        # or those containing alias substrings (e.g. 'generative ai' has 'ai' alias)
        import re
        from src.preprocessing.skill_extractor import ALIAS_MAP
        alias_words = set(ALIAS_MAP.keys())
        sample = [
            s for s in list(SKILL_DICT.keys())[:40]
            if re.fullmatch(r"[a-z0-9 /\-\.]+", s)
            and not any(w in s.split() for w in alias_words)
        ]
        for skill in sample:
            found = extract_skills(skill)
            assert skill in found, f"'{skill}' not extracted from its own name"
