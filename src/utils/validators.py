"""Input validation and sanitization utilities."""

import re
from typing import Any


def sanitize_input(text: str | None) -> str:
    """Sanitize user input text.

    - Strip whitespace
    - Remove excessive whitespace
    - Remove control characters
    - Limit length
    """
    if not text:
        return ""

    # Strip and normalize whitespace
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Remove control characters except newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Limit length (2000 chars max for messages)
    if len(text) > 2000:
        text = text[:2000]

    return text


def validate_age(age: Any) -> int | None:
    """Validate age value."""
    try:
        age_int = int(age)
        if 0 <= age_int <= 120:
            return age_int
    except (ValueError, TypeError):
        pass
    return None


def validate_income(income: Any) -> int | None:
    """Validate income value."""
    try:
        income_int = int(income)
        if income_int >= 0:
            return income_int
    except (ValueError, TypeError):
        pass
    return None


def validate_category(category: str | None) -> str | None:
    """Validate and normalize category."""
    if not category:
        return None

    valid_categories = ["SC", "ST", "OBC", "GENERAL", "EWS"]
    upper = category.upper().strip()

    if upper in valid_categories:
        return upper
    if upper == "GEN":
        return "GENERAL"

    return None


def validate_gender(gender: str | None) -> str | None:
    """Validate and normalize gender."""
    if not gender:
        return None

    gender_lower = gender.lower().strip()

    if gender_lower in ["male", "m", "पुरुष"]:
        return "male"
    if gender_lower in ["female", "f", "महिला", "स्त्री"]:
        return "female"
    if gender_lower in ["other", "अन्य"]:
        return "other"

    return None


def validate_marital_status(status: str | None) -> str | None:
    """Validate and normalize marital status."""
    if not status:
        return None

    status_lower = status.lower().strip()
    status_map = {
        "single": "single",
        "unmarried": "single",
        "अविवाहित": "single",
        "married": "married",
        "विवाहित": "married",
        "शादीशुदा": "married",
        "widowed": "widowed",
        "widow": "widowed",
        "widower": "widowed",
        "विधवा": "widowed",
        "विधुर": "widowed",
        "divorced": "divorced",
        "तलाकशुदा": "divorced",
        "separated": "separated",
        "अलग": "separated",
    }

    return status_map.get(status_lower)


def validate_employment_status(status: str | None) -> str | None:
    """Validate and normalize employment status."""
    if not status:
        return None

    status_lower = status.lower().strip()
    status_map = {
        "employed": "employed",
        "नौकरी": "employed",
        "job": "employed",
        "unemployed": "unemployed",
        "बेरोजगार": "unemployed",
        "self-employed": "self-employed",
        "self employed": "self-employed",
        "स्वरोजगार": "self-employed",
        "business": "self-employed",
        "student": "student",
        "छात्र": "student",
        "studying": "student",
    }

    return status_map.get(status_lower)


def is_valid_telegram_message(update: dict) -> bool:
    """Check if Telegram update contains a valid message."""
    if not update:
        return False

    message = update.get("message") or update.get("callback_query", {}).get("message")
    if not message:
        return False

    # Check for text or voice
    return bool(message.get("text") or message.get("voice"))


def extract_telegram_user_id(update: dict) -> str | None:
    """Extract user ID from Telegram update."""
    if update.get("message"):
        return str(update["message"].get("from", {}).get("id"))
    if update.get("callback_query"):
        return str(update["callback_query"].get("from", {}).get("id"))
    return None
