"""Profile extraction service."""

import logging
import re
from typing import Any

from src.integrations.llm_client import get_llm_client
from src.models.session import UserProfile
from src.prompts.loader import get_system_prompt

logger = logging.getLogger(__name__)


def extract_by_patterns(text: str) -> dict[str, Any]:
    """Rule-based extraction using regex patterns."""
    extracted = {}
    text_lower = text.lower()

    # Age extraction
    age_patterns = [
        r"(\d{1,3})\s*(saal|साल|years?\s*old|year|वर्ष)",
        r"(age|umar|उम्र|आयु)\s*[:=-]?\s*(\d{1,3})",
        r"main\s*(\d{1,3})\s*(saal|साल)",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            age = int(match.group(1)) if match.group(1).isdigit() else int(match.group(2))
            if 18 <= age <= 120:
                extracted["age"] = age
                break

    # Income extraction (annual)
    income_patterns = [
        r"(\d+(?:\.\d+)?)\s*(lakh|lac|लाख)\s*(per\s*year|yearly|annual|सालाना)?",
        r"income\s*[:=-]?\s*(\d+(?:,\d{3})*)",
        r"(\d+(?:,\d{3})*)\s*(per\s*month|monthly|mahina|महीना)",
    ]
    for pattern in income_patterns:
        match = re.search(pattern, text_lower)
        if match:
            amount = match.group(1).replace(",", "")
            try:
                income = float(amount)
                # Check if in lakhs
                if "lakh" in text_lower or "lac" in text_lower or "लाख" in text_lower:
                    income *= 100000
                # Check if monthly - convert to annual
                if "month" in text_lower or "mahina" in text_lower or "महीना" in text_lower:
                    income *= 12
                extracted["annual_income"] = int(income)
                break
            except ValueError:
                pass

    # Category extraction
    category_map = {
        "sc": "SC", "scheduled caste": "SC", "अनुसूचित जाति": "SC",
        "st": "ST", "scheduled tribe": "ST", "अनुसूचित जनजाति": "ST",
        "obc": "OBC", "other backward": "OBC", "अन्य पिछड़ा": "OBC", "पिछड़ा वर्ग": "OBC",
        "ews": "EWS", "economically weaker": "EWS", "आर्थिक रूप से कमज़ोर": "EWS",
        "general": "General", "सामान्य": "General",
    }
    for keyword, category in category_map.items():
        if keyword in text_lower:
            extracted["category"] = category
            break

    # Gender extraction
    gender_indicators = {
        "male": ["main aadmi", "i am male", "ladka", "लड़का", "पुरुष"],
        "female": ["main aurat", "i am female", "ladki", "लड़की", "महिला", "widow", "vidhwa", "विधवा"],
    }
    for gender, keywords in gender_indicators.items():
        if any(kw in text_lower for kw in keywords):
            extracted["gender"] = gender
            break

    # Marital status
    marital_map = {
        "widow": "widowed", "vidhwa": "widowed", "विधवा": "widowed", "विधुर": "widowed",
        "married": "married", "शादीशुदा": "married", "विवाहित": "married",
        "single": "single", "unmarried": "single", "अविवाहित": "single",
        "divorced": "divorced", "तलाकशुदा": "divorced",
        "separated": "separated", "अलग": "separated",
    }
    for keyword, status in marital_map.items():
        if keyword in text_lower:
            extracted["marital_status"] = status
            break

    # Employment status
    employment_map = {
        "unemployed": "unemployed", "बेरोजगार": "unemployed", "no job": "unemployed",
        "self-employed": "self-employed", "स्वरोजगार": "self-employed", "business": "self-employed",
        "employed": "employed", "नौकरी": "employed", "job": "employed",
        "student": "student", "पढ़ाई": "student", "studying": "student",
    }
    for keyword, status in employment_map.items():
        if keyword in text_lower:
            extracted["employment_status"] = status
            break

    # BPL card
    if "bpl" in text_lower or "गरीबी रेखा" in text_lower:
        extracted["has_bpl_card"] = True
    elif "no bpl" in text_lower or "bpl nahi" in text_lower:
        extracted["has_bpl_card"] = False

    return extracted


async def extract_profile(
    user_message: str,
    conversation_history: list[dict[str, str]],
    current_profile: UserProfile,
) -> UserProfile:
    """Extract profile information from user message.

    Combines rule-based extraction with LLM for complex cases.
    Returns new UserProfile merged with current profile.
    """
    # Try rule-based extraction first
    pattern_extracted = extract_by_patterns(user_message)

    # Use LLM for additional extraction
    llm_extracted = {}
    try:
        llm = get_llm_client()
        analysis = await llm.analyze_message(
            user_message=user_message,
            conversation_history=conversation_history,
            current_state="UNDERSTANDING",
            user_profile=current_profile.model_dump(),
            system_prompt=get_system_prompt(),
        )

        llm_extracted = analysis.get("extracted_fields", {})

        # Also get life event if detected
        if analysis.get("life_event"):
            llm_extracted["life_event"] = analysis["life_event"]

    except Exception as e:
        logger.error(f"LLM profile extraction failed: {e}")

    # Merge extractions (rule-based takes precedence for explicit matches)
    merged = {**llm_extracted, **pattern_extracted}

    # Filter out None values
    merged = {k: v for k, v in merged.items() if v is not None}

    if not merged:
        return current_profile

    # Create new profile from extracted data
    extracted_profile = UserProfile(**merged)

    # Merge with current profile
    return current_profile.merge_with(extracted_profile)


def get_missing_fields(profile: UserProfile) -> list[str]:
    """Get list of missing fields that should be collected."""
    missing = []

    # Priority order for field collection
    fields = [
        ("life_event", "life situation"),
        ("age", "age"),
        ("category", "caste category (SC/ST/OBC/General/EWS)"),
        ("annual_income", "annual income"),
        ("gender", "gender"),
        ("employment_status", "employment status"),
    ]

    for field, display_name in fields:
        if getattr(profile, field) is None:
            missing.append(display_name)

    return missing


def get_next_question(profile: UserProfile, language: str = "hi") -> str | None:
    """Get the next question to ask based on missing profile fields."""
    questions = {
        "life_event": {
            "hi": "आप मुझे बताएं, आज आपको किस तरह की सहायता चाहिए? (जैसे: घर, स्वास्थ्य, शिक्षा, रोजगार)",
            "en": "Please tell me, what kind of assistance do you need today? (e.g., housing, health, education, employment)",
        },
        "age": {
            "hi": "आपकी उम्र कितनी है?",
            "en": "What is your age?",
        },
        "category": {
            "hi": "आप किस श्रेणी में आते हैं? (SC/ST/OBC/General/EWS)",
            "en": "What is your caste category? (SC/ST/OBC/General/EWS)",
        },
        "annual_income": {
            "hi": "आपकी वार्षिक पारिवारिक आय लगभग कितनी है?",
            "en": "What is your approximate annual family income?",
        },
        "gender": {
            "hi": "आप पुरुष हैं या महिला?",
            "en": "Are you male or female?",
        },
    }

    # Find first missing field
    for field, field_questions in questions.items():
        if getattr(profile, field) is None:
            return field_questions.get(language, field_questions["en"])

    return None
