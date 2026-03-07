"""Tests for AI relevance gating over deterministic scheme matches."""

from src.models.scheme import EligibilityCriteria, Scheme, SchemeMatch
from src.services.scheme_relevance import apply_relevance_judgement


def _make_match() -> SchemeMatch:
    return SchemeMatch(
        scheme=Scheme(
            id="SCH-DELHI-001",
            name="Pradhan Mantri Awas Yojana - Urban 2.0 (PMAY-U 2.0)",
            name_hindi="प्रधानमंत्री आवास योजना - शहरी 2.0",
            department="Housing Department",
            department_hindi="आवास विभाग",
            level="central",
            description="Housing scheme",
            description_hindi="आवास योजना",
            benefits_amount=250000,
            eligibility=EligibilityCriteria(categories=["EWS", "LIG", "MIG"]),
            life_events=["HOUSING"],
            documents_required=[],
            rejection_rules=[],
        ),
        similarity=0.4,
        eligibility_match={"age": True, "income": True},
        deterministic_score=0.8,
    )


def test_fallback_llm_unavailable_does_not_force_clarification() -> None:
    """If AI judging is unavailable, strong deterministic matches should still present."""
    match = _make_match()

    result = apply_relevance_judgement(
        [match],
        {
            "error": "LLM service unavailable",
            "overall_confidence": 0.5,
            "candidate_scores": [
                {
                    "scheme_id": "SCH-DELHI-001",
                    "relevance_score": 0.8,
                    "topic_match": None,
                    "reason": None,
                }
            ],
        },
        language="en",
        life_event="HOUSING",
    )

    assert result["should_clarify"] is False
    assert [item.scheme.id for item in result["matches"]] == ["SCH-DELHI-001"]
