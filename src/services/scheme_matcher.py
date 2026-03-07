"""Scheme matching service with 3-stage hybrid retrieval."""

import logging
from typing import Any

import asyncpg

from src.db.scheme_repo import hybrid_search, get_schemes_by_life_event
from src.integrations.embedding_client import EMBEDDING_DIM, get_embedding_client
from src.models.scheme import SchemeMatch
from src.models.session import UserProfile

logger = logging.getLogger(__name__)


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
    matches = await hybrid_search(
        pool=pool,
        life_event=profile.life_event,
        profile=profile,
        query_embedding=query_embedding,
        limit=limit,
    )

    # Post-filter: remove schemes where user's category is explicitly ineligible.
    # If a scheme lists specific categories and the user's category isn't among
    # them, the user cannot apply — showing it would be misleading.
    if profile.category and matches:
        filtered = []
        for match in matches:
            elig_cats = match.scheme.eligibility.categories
            if not elig_cats:
                # Empty list = no category restriction (all categories eligible)
                filtered.append(match)
            elif profile.category.upper() in [c.upper() for c in elig_cats]:
                filtered.append(match)
            else:
                logger.info(
                    "Filtered out scheme %s: user category '%s' not in %s",
                    match.scheme.id,
                    profile.category,
                    elig_cats,
                )
        matches = filtered

    logger.info(
        f"Matched {len(matches)} schemes for life_event={profile.life_event}, "
        f"age={profile.age}, income={profile.annual_income}"
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

    return sorted(matches, key=score, reverse=True)
