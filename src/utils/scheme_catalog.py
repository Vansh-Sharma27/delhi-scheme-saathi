"""Access canonical scheme metadata bundled with the repository."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "all_schemes.json"
_INCOME_SEGMENT_KEYS = {"EWS", "LIG", "MIG", "HIG"}


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, dict[str, Any]]:
    """Load bundled scheme metadata keyed by scheme id."""
    try:
        schemes = json.loads(_CATALOG_PATH.read_text())
    except FileNotFoundError:
        logger.warning("Canonical scheme catalog not found at %s", _CATALOG_PATH)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse canonical scheme catalog: %s", exc)
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    for scheme in schemes:
        scheme_id = str(scheme.get("id", "")).strip()
        if scheme_id:
            catalog[scheme_id] = scheme
    return catalog


def get_canonical_scheme_record(scheme_id: str) -> dict[str, Any] | None:
    """Return bundled scheme metadata for a scheme id, if present."""
    return _load_catalog().get(scheme_id)


def get_canonical_scheme_ids_for_life_event(life_event: str | None) -> list[str]:
    """Return bundled scheme ids mapped to a life event."""
    if not life_event:
        return []

    matching_ids: list[str] = []
    for scheme_id, scheme in _load_catalog().items():
        if life_event in scheme.get("life_events", []):
            matching_ids.append(scheme_id)
    return matching_ids


def get_canonical_life_events(scheme_id: str) -> list[str]:
    """Return bundled life events for a scheme id."""
    record = get_canonical_scheme_record(scheme_id)
    if not record:
        return []
    return [str(value) for value in record.get("life_events", []) if str(value)]


def get_canonical_tags(scheme_id: str) -> list[str]:
    """Return bundled tags for a scheme id."""
    record = get_canonical_scheme_record(scheme_id)
    if not record:
        return []
    return [str(value) for value in record.get("tags", []) if str(value)]


@lru_cache(maxsize=32)
def get_required_profile_fields_for_life_event(life_event: str | None) -> tuple[str, ...]:
    """Return the minimum profile fields worth collecting for a life event.

    The bot always needs the topic first. Beyond that, age and annual income
    are kept as shared core filters, while category/gender are only requested
    when at least one canonical scheme for the life event actually uses them.
    """
    if not life_event:
        return ("life_event",)

    required_fields = ["life_event", "age", "annual_income"]
    needs_category = False
    needs_gender = False

    for scheme in _load_catalog().values():
        if life_event not in scheme.get("life_events", []):
            continue

        eligibility = scheme.get("eligibility") or {}
        raw_categories = {
            str(value).strip().upper()
            for value in eligibility.get("categories", [])
            if str(value).strip()
        }
        explicit_caste_categories = {
            str(value).strip().upper()
            for value in eligibility.get("caste_categories", [])
            if str(value).strip()
        }
        genders = {
            str(value).strip().lower()
            for value in eligibility.get("genders", ["all"])
            if str(value).strip()
        }
        income_by_category = {
            str(key).strip().upper()
            for key in (eligibility.get("income_by_category") or {})
            if str(key).strip()
        }

        income_segment_categories = (raw_categories | income_by_category) & _INCOME_SEGMENT_KEYS
        normalized_caste_categories = explicit_caste_categories
        if (
            not normalized_caste_categories
            and raw_categories
            and not income_segment_categories
            and raw_categories != {"ALL"}
        ):
            normalized_caste_categories = raw_categories

        if normalized_caste_categories:
            needs_category = True
        if genders and genders != {"all"}:
            needs_gender = True

    if needs_gender:
        required_fields.append("gender")
    if needs_category:
        required_fields.append("category")
    return tuple(required_fields)
