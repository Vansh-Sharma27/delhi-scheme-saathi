"""Tests for deterministic scheme matching logic."""

from unittest.mock import AsyncMock

import pytest

from src.db import scheme_repo
from src.models.scheme import EligibilityCriteria, Scheme, SchemeMatch
from src.models.session import UserProfile
from src.services import scheme_matcher


def _make_housing_scheme() -> Scheme:
    return Scheme(
        id="SCH-DELHI-001",
        name="PMAY-U 2.0",
        name_hindi="पीएमएवाई-यू 2.0",
        department="Housing Department",
        department_hindi="आवास विभाग",
        level="central",
        description="Housing scheme for urban families.",
        description_hindi="शहरी परिवारों के लिए आवास योजना।",
        benefits_amount=250000,
        benefits_frequency="one-time",
        eligibility=EligibilityCriteria(
            categories=["EWS", "LIG", "MIG"],
            income_by_category={
                "EWS": 300000,
                "LIG": 600000,
                "MIG": 900000,
            },
        ),
        life_events=["HOUSING"],
        documents_required=[],
        rejection_rules=[],
    )


def _make_education_scheme(*, life_events: list[str] | None = None) -> Scheme:
    return Scheme(
        id="SCH-DELHI-006",
        name="Education Loan Scheme - Delhi",
        name_hindi="शिक्षा ऋण योजना - दिल्ली",
        department="Education Department",
        department_hindi="शिक्षा विभाग",
        level="state",
        description="Education loan support for higher education students.",
        description_hindi="उच्च शिक्षा के लिए शिक्षा ऋण सहायता।",
        benefits_amount=750000,
        benefits_frequency="installments",
        eligibility=EligibilityCriteria(
            max_income=500000,
            categories=["SC", "ST", "OBC"],
        ),
        life_events=life_events or ["EDUCATION"],
        tags=["education", "loan", "higher_education"],
        documents_required=[],
        rejection_rules=[],
    )


def _make_widow_scheme() -> Scheme:
    return Scheme(
        id="SCH-DELHI-003",
        name="Widow Pension",
        name_hindi="विधवा पेंशन",
        department="WCD",
        department_hindi="महिला एवं बाल विकास",
        level="state",
        description="Monthly pension for widows in Delhi.",
        description_hindi="दिल्ली की विधवाओं के लिए मासिक पेंशन।",
        benefits_amount=2500,
        benefits_frequency="monthly",
        eligibility=EligibilityCriteria(
            min_age=18,
            genders=["female"],
            categories=["all"],
            max_income=100000,
        ),
        life_events=["DEATH_IN_FAMILY"],
        tags=["widow", "pension", "women_in_distress"],
        documents_required=[],
        rejection_rules=[],
    )


def test_calculate_eligibility_match_uses_income_segments_not_caste_category() -> None:
    """Housing income bands should not be compared against caste category."""
    scheme = _make_housing_scheme()
    profile = UserProfile(
        life_event="HOUSING",
        age=25,
        category="OBC",
        annual_income=500000,
    )

    match = scheme_repo.calculate_eligibility_match(scheme, profile)

    assert "category" not in match
    assert match["income_segment"] is True
    assert match["income"] is True


def test_calculate_eligibility_match_treats_all_category_as_unrestricted() -> None:
    """Schemes declaring category=all should not reject OBC/SC/etc users."""
    scheme = _make_widow_scheme()
    profile = UserProfile(
        life_event="DEATH_IN_FAMILY",
        age=23,
        gender="female",
        category="OBC",
        annual_income=50000,
    )

    match = scheme_repo.calculate_eligibility_match(scheme, profile)

    assert match["age"] is True
    assert match["gender"] is True
    assert "category" not in match
    assert match["income"] is True


@pytest.mark.asyncio
async def test_match_schemes_keeps_housing_scheme_for_obc_user_with_lig_income(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Income-band schemes should survive deterministic post-filtering."""
    scheme = _make_housing_scheme()
    profile = UserProfile(
        life_event="HOUSING",
        age=25,
        category="OBC",
        annual_income=500000,
    )
    eligibility_match = scheme_repo.calculate_eligibility_match(scheme, profile)

    async def fake_hybrid_search(**kwargs):  # type: ignore[no-untyped-def]
        return [
            SchemeMatch(
                scheme=scheme,
                similarity=0.5,
                eligibility_match=eligibility_match,
                deterministic_score=0.6,
            )
        ]

    monkeypatch.setattr(scheme_matcher, "hybrid_search", fake_hybrid_search)

    matches = await scheme_matcher.match_schemes(
        pool=AsyncMock(),  # type: ignore[arg-type]
        profile=profile,
        query_text=None,
    )

    assert [match.scheme.id for match in matches] == ["SCH-DELHI-001"]


@pytest.mark.asyncio
async def test_match_schemes_filters_cross_domain_candidate_even_if_db_tag_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strong name/tag signals should prevent an education scheme leaking into housing."""
    housing = _make_housing_scheme()
    leaked_education = _make_education_scheme(life_events=["HOUSING"])
    profile = UserProfile(
        life_event="HOUSING",
        age=25,
        category="OBC",
        annual_income=500000,
    )

    async def fake_hybrid_search(**kwargs):  # type: ignore[no-untyped-def]
        return [
            SchemeMatch(
                scheme=housing,
                similarity=0.8,
                eligibility_match=scheme_repo.calculate_eligibility_match(housing, profile),
                deterministic_score=0.8,
            ),
            SchemeMatch(
                scheme=leaked_education,
                similarity=0.7,
                eligibility_match=scheme_repo.calculate_eligibility_match(leaked_education, profile),
                deterministic_score=0.7,
            ),
        ]

    monkeypatch.setattr(scheme_matcher, "hybrid_search", fake_hybrid_search)

    matches = await scheme_matcher.match_schemes(
        pool=AsyncMock(),  # type: ignore[arg-type]
        profile=profile,
        query_text=None,
    )

    assert [match.scheme.id for match in matches] == ["SCH-DELHI-001"]


@pytest.mark.asyncio
async def test_match_schemes_keeps_valid_multi_life_event_scheme_via_canonical_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical bundled tags should preserve valid multi-event schemes."""
    pmay = _make_housing_scheme()
    profile = UserProfile(
        life_event="CHILDBIRTH",
        age=25,
        category="OBC",
        annual_income=500000,
    )

    async def fake_hybrid_search(**kwargs):  # type: ignore[no-untyped-def]
        return [
            SchemeMatch(
                scheme=pmay,
                similarity=0.8,
                eligibility_match=scheme_repo.calculate_eligibility_match(pmay, profile),
                deterministic_score=0.8,
            )
        ]

    monkeypatch.setattr(scheme_matcher, "hybrid_search", fake_hybrid_search)

    matches = await scheme_matcher.match_schemes(
        pool=AsyncMock(),  # type: ignore[arg-type]
        profile=profile,
        query_text=None,
    )

    assert [match.scheme.id for match in matches] == ["SCH-DELHI-001"]
