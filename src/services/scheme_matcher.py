"""Scheme matching service with 3-stage hybrid retrieval."""

import logging
from typing import Any

import asyncpg

from src.db.scheme_repo import hybrid_search, get_schemes_by_life_event
from src.integrations.embedding_client import EMBEDDING_DIM, get_embedding_client
from src.models.scheme import Scheme, SchemeMatch
from src.models.session import UserProfile
from src.utils.scheme_catalog import get_canonical_life_events

logger = logging.getLogger(__name__)


def is_topic_consistent(scheme: Scheme, requested_life_event: str | None) -> bool:
    """Check whether the scheme's canonical topic aligns with the requested one."""
    if not requested_life_event:
        return True

    canonical_life_events = get_canonical_life_events(scheme.id)
    if canonical_life_events:
        return requested_life_event in canonical_life_events
    return requested_life_event in scheme.life_events


async def match_schemes(
    pool: asyncpg.Pool,
    profile: UserProfile,
    query_text: str | None = None,
    limit: int = 5,
) -> list[SchemeMatch]:
    """Match schemes to user profile using 3-stage hybrid retrieval.

    Stage 1: Filter by life event
    Stage 2: Filter by eligibility (age, income, category)
    Stage 3: Rank by semantic similarity (if query text provided)
    """
    logger.info(
        "Starting deterministic scheme matching for life_event=%s age=%s category=%s income=%s",
        profile.life_event,
        profile.age,
        profile.category,
        profile.annual_income,
    )

    # Get query embedding if text provided
    query_embedding = None
    if query_text:
        try:
            embedding_client = get_embedding_client()
            embedding = await embedding_client.get_embedding(query_text)
            if embedding and len(embedding) == EMBEDDING_DIM:
                query_embedding = embedding
            elif embedding:
                logger.warning(
                    f"Skipping vector ranking: expected {EMBEDDING_DIM}-dim embedding, "
                    f"received {len(embedding)}"
                )
            else:
                logger.warning("Skipping vector ranking: embedding unavailable from all providers")
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")

    # Run hybrid search
    fetch_limit = max(limit * 3, 10)
    matches = await hybrid_search(
        pool=pool,
        life_event=profile.life_event,
        profile=profile,
        query_embedding=query_embedding,
        limit=fetch_limit,
    )

    # Post-filter any scheme that still fails a hard eligibility check. This
    # catches non-SQL restrictions such as caste categories and normalized
    # income bands like EWS/LIG/MIG.
    if matches:
        filtered = []
        for match in matches:
            if not is_topic_consistent(match.scheme, profile.life_event):
                logger.info(
                    "Filtered out scheme %s for topic mismatch: requested=%s canonical_life_events=%s runtime_life_events=%s",
                    match.scheme.id,
                    profile.life_event,
                    get_canonical_life_events(match.scheme.id),
                    match.scheme.life_events,
                )
                continue
            failing_fields = [
                field
                for field, is_match in match.eligibility_match.items()
                if is_match is False
            ]
            if failing_fields:
                logger.info(
                    "Filtered out scheme %s after deterministic checks: failed=%s life_events=%s",
                    match.scheme.id,
                    ",".join(failing_fields),
                    match.scheme.life_events,
                )
                continue
            filtered.append(match)
        matches = rank_schemes(filtered)[:limit]

    logger.info(
        "Deterministic scheme matches complete: life_event=%s matched_ids=%s",
        profile.life_event,
        [match.scheme.id for match in matches],
    )

    return matches


async def get_schemes_for_life_event(
    pool: asyncpg.Pool,
    life_event: str,
    limit: int = 5,
) -> list[SchemeMatch]:
    """Simple scheme lookup by life event (no profile filtering)."""
    schemes = await get_schemes_by_life_event(pool, life_event, limit)

    return [
        SchemeMatch(scheme=scheme, similarity=0.0, eligibility_match={})
        for scheme in schemes
    ]


def format_scheme_for_display(
    match: SchemeMatch,
    language: str = "hi",
) -> dict[str, Any]:
    """Format scheme match for display in conversation."""
    scheme = match.scheme

    # Build eligibility status
    eligibility_text = []
    for field, is_match in match.eligibility_match.items():
        icon = "✅" if is_match else "❌"
        field_display = {
            "age": "आयु" if language == "hi" else "Age",
            "gender": "लिंग" if language == "hi" else "Gender",
            "category": "श्रेणी" if language == "hi" else "Category",
            "income_segment": "आय वर्ग" if language == "hi" else "Income band",
            "income": "आय" if language == "hi" else "Income",
        }.get(field, field)
        eligibility_text.append(f"{icon} {field_display}")

    # Format benefits
    benefits = scheme.benefits_summary or ""
    if scheme.benefits_amount:
        amount_str = f"₹{scheme.benefits_amount:,}"
        if scheme.benefits_amount >= 100000:
            amount_str = f"₹{scheme.benefits_amount / 100000:.1f} लाख"

    return {
        "id": scheme.id,
        "name": scheme.name_hindi if language == "hi" else scheme.name,
        "department": scheme.department_hindi if language == "hi" else scheme.department,
        "benefits_amount": scheme.benefits_amount,
        "benefits_display": amount_str if scheme.benefits_amount else None,
        "benefits_summary": benefits[:200] if benefits else None,
        "eligibility_match": match.eligibility_match,
        "eligibility_text": " | ".join(eligibility_text) if eligibility_text else None,
        "similarity_score": round(match.similarity, 2),
    }


def rank_schemes(matches: list[SchemeMatch]) -> list[SchemeMatch]:
    """Re-rank schemes by a combined score."""
    def score(match: SchemeMatch) -> float:
        # Base similarity score
        score = match.similarity * 0.4

        # Eligibility match bonus
        if match.eligibility_match:
            match_rate = sum(match.eligibility_match.values()) / len(match.eligibility_match)
            score += match_rate * 0.4

        # Benefits amount bonus (normalized)
        if match.scheme.benefits_amount:
            # Normalize to 0-1 range (assuming max 10 lakh)
            normalized_benefit = min(match.scheme.benefits_amount / 1000000, 1.0)
            score += normalized_benefit * 0.2

        return score

    ranked: list[SchemeMatch] = []
    for match in matches:
        ranked.append(
            match.model_copy(update={"deterministic_score": score(match)})
        )
    return sorted(ranked, key=lambda match: match.deterministic_score, reverse=True)
