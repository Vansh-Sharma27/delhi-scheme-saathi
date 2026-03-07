"""AI-assisted scheme relevance judging and confidence gating."""

from __future__ import annotations

from typing import Any

from src.models.scheme import SchemeMatch

PRESENT_CONFIDENCE_THRESHOLD = 0.6
CLARIFY_CONFIDENCE_THRESHOLD = 0.45


def build_candidate_payload(matches: list[SchemeMatch]) -> list[dict[str, Any]]:
    """Convert matches into a compact payload for LLM judging."""
    payload: list[dict[str, Any]] = []
    for match in matches[:5]:
        scheme = match.scheme
        payload.append(
            {
                "scheme_id": scheme.id,
                "name": scheme.name,
                "name_hindi": scheme.name_hindi,
                "description": scheme.description[:400],
                "life_events": scheme.life_events,
                "tags": scheme.tags[:12],
                "benefits_summary": (scheme.benefits_summary or "")[:240],
                "eligibility": {
                    "caste_categories": scheme.eligibility.caste_categories,
                    "income_segments": scheme.eligibility.income_segments,
                    "max_income": scheme.eligibility.max_income,
                },
                "deterministic_score": round(match.deterministic_score, 4),
            }
        )
    return payload


def _default_clarification_question(language: str, life_event: str | None) -> str:
    if language == "hi":
        if life_event == "HOUSING":
            return "मैंने आपकी जरूरत को आवास सहायता समझा है। क्या आप housing schemes ही देखना चाहते हैं?"
        return "मैं सही योजना चुनना चाहता हूँ। कृपया एक बार बताएं कि आपको किस तरह की सहायता चाहिए।"
    if language == "hinglish":
        if life_event == "HOUSING":
            return "Maine aapki need housing assistance samjhi hai. Kya aap housing schemes hi dekhna chahte hain?"
        return "Main sahi scheme choose karna chahta hoon. Please ek baar batayiye ki aapko kis tarah ki help chahiye."
    if life_event == "HOUSING":
        return "I understood that you want housing assistance. Do you want housing schemes specifically?"
    return "I want to be sure I choose the right scheme. Please confirm what kind of assistance you need."


def apply_relevance_judgement(
    matches: list[SchemeMatch],
    judgement: dict[str, Any] | None,
    language: str,
    life_event: str | None,
) -> dict[str, Any]:
    """Apply LLM judging output to deterministic matches and decide clarify/present."""
    if not matches:
        return {
            "matches": [],
            "should_clarify": False,
            "clarification_question": None,
            "overall_confidence": 0.0,
        }

    judgement = judgement or {}
    llm_available = bool(judgement) and not judgement.get("error")
    scores_by_id = {
        item.get("scheme_id"): item
        for item in judgement.get("candidate_scores", [])
        if item.get("scheme_id")
    }

    enriched: list[SchemeMatch] = []
    for match in matches:
        score_item = scores_by_id.get(match.scheme.id, {})
        ai_score = score_item.get("relevance_score")
        if not isinstance(ai_score, (int, float)):
            ai_score = match.deterministic_score or 0.5
        enriched.append(
            match.model_copy(
                update={
                    "ai_relevance_score": max(0.0, min(float(ai_score), 1.0)),
                    "ai_relevance_reason": score_item.get("reason"),
                    "topic_match": score_item.get("topic_match"),
                }
            )
        )

    enriched.sort(
        key=lambda match: (
            match.ai_relevance_score if match.ai_relevance_score is not None else 0.0,
            match.deterministic_score,
            match.similarity,
        ),
        reverse=True,
    )

    topic_pruned = [match for match in enriched if match.topic_match is not False]
    if topic_pruned:
        enriched = topic_pruned
    else:
        return {
            "matches": [],
            "should_clarify": True,
            "clarification_question": judgement.get("clarification_question")
            or _default_clarification_question(language, life_event),
            "overall_confidence": 0.0,
        }

    top_match = enriched[0]
    overall_confidence = judgement.get("overall_confidence")
    if not isinstance(overall_confidence, (int, float)):
        overall_confidence = (
            top_match.ai_relevance_score
            if llm_available and top_match.ai_relevance_score is not None
            else top_match.deterministic_score or 0.7
        )
    overall_confidence = max(0.0, min(float(overall_confidence), 1.0))

    should_clarify = bool(judgement.get("should_clarify"))
    if top_match.topic_match is False:
        should_clarify = True
    if llm_available:
        if (top_match.ai_relevance_score or 0.0) < CLARIFY_CONFIDENCE_THRESHOLD:
            should_clarify = True
        if overall_confidence < PRESENT_CONFIDENCE_THRESHOLD:
            should_clarify = True

    clarification_question = judgement.get("clarification_question")
    if should_clarify and not clarification_question:
        clarification_question = _default_clarification_question(language, life_event)

    return {
        "matches": enriched,
        "should_clarify": should_clarify,
        "clarification_question": clarification_question,
        "overall_confidence": overall_confidence,
    }
