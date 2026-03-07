"""Profile extraction service."""

import re
from typing import Any

from src.models.session import UserProfile

FIELD_QUESTION_ORDER = ("life_event", "age", "gender", "category", "annual_income")


def get_required_matching_fields(profile: UserProfile) -> tuple[str, ...]:
    """Return the scheme-aware profile fields that matter before matching."""
    return profile.required_fields_for_matching()


def _has_income_context(text_lower: str, current_field: str | None) -> bool:
    """Return True when the user is likely answering the income question."""
    if current_field == "annual_income":
        return True

    income_cues = (
        r"\bincome\b",
        r"\bfamily income\b",
        r"\bhousehold income\b",
        r"\bannual income\b",
        r"\bmonthly income\b",
        r"\bsalary\b",
        r"\bearning[s]?\b",
        r"\bearn\b",
        r"\bper month\b",
        r"\bmonthly\b",
        r"\bper year\b",
        r"\byearly\b",
        r"\bannually\b",
        r"\bmahina\b",
        r"महीना",
    )
    return any(re.search(pattern, text_lower) for pattern in income_cues)


def extract_by_patterns(
    text: str,
    *,
    current_field: str | None = None,
) -> dict[str, Any]:
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
    if _has_income_context(text_lower, current_field):
        income_patterns = [
            r"(\d+(?:\.\d+)?)\s*(lakh|lac|लाख)\s*(per\s*year|yearly|annual|सालाना)?",
            r"(\d+(?:\.\d+)?)\s*(k|thousand|हजार)\s*(?:inr|rs\.?|rupees?)?",
            r"income\s*[:=-]?\s*(\d+(?:,\d{3})*)",
            r"(?:₹|rs\.?|rupees?|inr)\s*(\d+(?:,\d{3})*)",
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
                    elif re.search(r"\b\d+(?:\.\d+)?\s*(k|thousand|हजार)\b", text_lower):
                        income *= 1000
                    # Check if monthly - convert to annual
                    if "month" in text_lower or "mahina" in text_lower or "महीना" in text_lower:
                        income *= 12
                    extracted["annual_income"] = int(income)
                    break
                except ValueError:
                    pass

    bare_number = re.fullmatch(r"\s*(\d[\d,]*)\s*", text)
    if bare_number:
        value = int(bare_number.group(1).replace(",", ""))
        if "age" not in extracted and 18 <= value <= 120:
            extracted["age"] = value
        elif (
            "annual_income" not in extracted
            and value > 1000
            and current_field == "annual_income"
        ):
            extracted["annual_income"] = value

    # Category extraction
    # Use word-boundary/phrase matching to avoid false positives:
    # e.g. "study" should NOT map to "ST", "schemes" should NOT map to "SC".
    category_patterns: list[tuple[str, str]] = [
        (r"\bscheduled\s+caste\b|अनुसूचित\s*जाति|\bsc\b", "SC"),
        (r"\bscheduled\s+tribe\b|अनुसूचित\s*जनजाति|\bst\b", "ST"),
        (r"\bother\s+backward\b|अन्य\s*पिछड़ा|पिछड़ा\s*वर्ग|\bobc\b", "OBC"),
        (r"\beconomically\s+weaker\b|आर्थिक\s*रूप\s*से\s*कमज़ोर|\bews\b", "EWS"),
        (r"\bgeneral\b|सामान्य", "General"),
    ]
    for pattern, category in category_patterns:
        if re.search(pattern, text_lower):
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

def get_missing_fields(profile: UserProfile) -> list[str]:
    """Get list of missing fields that should be collected."""
    missing = []

    display_names = {
        "life_event": "life situation",
        "age": "age",
        "category": "caste category (SC/ST/OBC/General/EWS)",
        "annual_income": "annual income",
        "gender": "gender",
        "employment_status": "employment status",
    }

    for field in get_required_matching_fields(profile):
        if getattr(profile, field) is None:
            missing.append(display_names.get(field, field))

    return missing


def get_next_missing_field(
    profile: UserProfile,
    skipped_fields: list[str] | None = None,
) -> str | None:
    """Return the name of the next missing profile field, or None if all filled."""
    skipped = set(skipped_fields or [])
    required_fields = set(get_required_matching_fields(profile))
    for field in FIELD_QUESTION_ORDER:
        if field not in required_fields:
            continue
        if field in skipped:
            continue
        if getattr(profile, field) is None:
            return field
    return None


def get_next_question(
    profile: UserProfile,
    language: str = "hi",
    skipped_fields: list[str] | None = None,
) -> str | None:
    """Get the next question to ask based on missing profile fields."""
    questions = {
        "life_event": {
            "hi": "आप मुझे बताएं, आज आपको किस तरह की सहायता चाहिए? (जैसे: घर, स्वास्थ्य, शिक्षा, रोजगार)",
            "en": "Please tell me, what kind of assistance do you need today? (e.g., housing, health, education, employment)",
            "hinglish": "Batayiye, aaj aapko kis tarah ki madad chahiye? (jaise housing, health, education, employment)",
        },
        "age": {
            "hi": "आवेदक (या लाभार्थी) की उम्र कितनी है?",
            "en": "What is the applicant/beneficiary age?",
            "hinglish": "Applicant ya beneficiary ki age kitni hai?",
        },
        "category": {
            "hi": "आवेदक की श्रेणी क्या है? (SC/ST/OBC/General/EWS)",
            "en": "What is the applicant's caste category? (SC/ST/OBC/General/EWS)",
            "hinglish": "Applicant ki category kya hai? (SC/ST/OBC/General/EWS)",
        },
        "annual_income": {
            "hi": "आवेदक के परिवार की वार्षिक आय लगभग कितनी है?",
            "en": "What is the applicant's approximate annual family income?",
            "hinglish": "Applicant ke family ki approx annual income kitni hai?",
        },
        "gender": {
            "hi": "आप पुरुष हैं या महिला?",
            "en": "Are you male or female?",
            "hinglish": "Applicant male hai ya female?",
        },
    }

    skipped = set(skipped_fields or [])
    required_fields = set(get_required_matching_fields(profile))
    for field in FIELD_QUESTION_ORDER:
        if field not in required_fields:
            continue
        if field in skipped:
            continue
        if getattr(profile, field) is None:
            field_questions = questions[field]
            return field_questions.get(language, field_questions["en"])

    return None


# ---------------------------------------------------------------------------
# Input validation for profile fields
# ---------------------------------------------------------------------------

def validate_field_response(
    field: str,
    user_message: str,
    extracted_fields: dict,
) -> tuple[bool, str | None]:
    """Validate if user's response is appropriate for the field being asked.

    Returns (is_valid, validation_error_type) where:
    - is_valid: True if response contains valid data or is clearly unrelated
    - validation_error_type: None if valid, otherwise one of:
      - "invalid_age": looks like age attempt but invalid value (e.g., "200", "0")
      - "invalid_income": looks like income attempt but couldn't parse
      - "invalid_category": mentions category keywords but doesn't match known values
      - "unclear_response": seems like an attempt to answer but unclear

    We only flag "invalid" when the user CLEARLY tried to answer the question
    but gave an invalid value. If they're asking something else entirely,
    we let the conversation flow naturally.
    """
    text = user_message.strip().lower()

    if field == "age":
        # Check if user already provided valid age
        if "age" in extracted_fields:
            return True, None

        # Check if message looks like an age attempt
        # Pure number that's out of range (not 1-120)
        if re.match(r"^\d{1,3}$", text):
            num = int(text)
            if num < 1 or num > 120:
                return False, "invalid_age"
            # Valid number but wasn't extracted (handled elsewhere)
            return True, None

        # Text that looks like age attempt but invalid
        age_attempt_patterns = [
            r"(\d{4,})\s*(saal|साल|years?|वर्ष)",  # 4+ digit year (birth year confusion)
            r"(age|umar|उम्र)\s*[:=-]?\s*(\d{4,})",  # age: 2005 (birth year)
            r"born\s+in\s+(\d{4})",  # "born in 2005"
        ]
        for pattern in age_attempt_patterns:
            if re.search(pattern, text):
                return False, "invalid_age"

        # If message contains age-related keywords but no number extracted
        if any(kw in text for kw in ["years", "saal", "साल", "year", "age", "उम्र"]) and not any(
            c.isdigit() for c in text
        ):
            return False, "unclear_response"

    elif field == "category":
        if "category" in extracted_fields:
            return True, None

        # Check if user tried to give category but used wrong terms
        unclear_category_patterns = [
            r"\b(open|unreserved|ur)\b",  # Common confusion for General
            r"\b(backward|bc)\b(?!\s*(class|caste))",  # BC without OBC
        ]
        for pattern in unclear_category_patterns:
            if re.search(pattern, text):
                return False, "invalid_category"

    elif field == "annual_income":
        if "annual_income" in extracted_fields:
            return True, None

        # Check for clearly invalid income values
        if re.match(r"^\d+$", text):
            num = int(text)
            # If it looks like a reasonable income wasn't extracted,
            # the extraction logic handles it. Only flag truly nonsensical values.
            if num == 0:
                return False, "invalid_income"
        elif re.search(
            r"\d"
            r".*(?:₹|rs\.?|rupees?|inr|lakh|lac|thousand|monthly|per\s*month|mahina|महीना|हजार)"
            r"|(?:₹|rs\.?|rupees?|inr|income|salary|lakh|lac|thousand|monthly|per\s*month|mahina|महीना|हजार).*\d"
            r"|(?:₹|rs\.?|rupees?|inr|income|salary|lakh|lac|thousand|monthly|per\s*month|mahina|महीना|हजार)",
            text,
        ):
            return False, "invalid_income"

    # Default: assume valid (let conversation flow naturally)
    return True, None


def get_validation_re_prompt(
    field: str,
    error_type: str,
    language: str = "hi",
) -> str:
    """Generate a helpful re-prompt when user's input is invalid.

    Uses friendly, non-judgmental language that guides user to provide
    correct information.
    """
    prompts = {
        "invalid_age": {
            "hi": (
                "मैं उम्र समझ नहीं पाया। कृपया उम्र संख्या में बताएं, "
                "जैसे: 25 साल या 45 years।"
            ),
            "en": (
                "I couldn't understand the age. Please tell me the age as a number, "
                "for example: 25 years or 45 saal."
            ),
            "hinglish": (
                "Main age samajh nahi paaya. Please age number mein batayiye, "
                "jaise 25 years ya 45 saal."
            ),
        },
        "invalid_income": {
            "hi": (
                "आय की जानकारी स्पष्ट नहीं हुई। कृपया अनुमानित वार्षिक आय बताएं, "
                "जैसे: 3 लाख या 50000 रुपये।"
            ),
            "en": (
                "I couldn't understand the income. Please share approximate annual income, "
                "for example: 3 lakh or 50000 rupees."
            ),
            "hinglish": (
                "Income clear nahi hua. Please approx annual income batayiye, "
                "jaise 3 lakh ya 50000 rupees."
            ),
        },
        "invalid_category": {
            "hi": (
                "कृपया इनमें से एक श्रेणी बताएं:\n"
                "• SC (अनुसूचित जाति)\n"
                "• ST (अनुसूचित जनजाति)\n"
                "• OBC (अन्य पिछड़ा वर्ग)\n"
                "• General (सामान्य)\n"
                "• EWS (आर्थिक रूप से कमज़ोर)"
            ),
            "en": (
                "Please specify one of these categories:\n"
                "• SC (Scheduled Caste)\n"
                "• ST (Scheduled Tribe)\n"
                "• OBC (Other Backward Class)\n"
                "• General\n"
                "• EWS (Economically Weaker Section)"
            ),
            "hinglish": (
                "Please inmein se ek category batayiye:\n"
                "• SC\n"
                "• ST\n"
                "• OBC\n"
                "• General\n"
                "• EWS"
            ),
        },
        "unclear_response": {
            "hi": "मैं आपकी बात समझ नहीं पाया। कृपया दोबारा बताएं।",
            "en": "I couldn't understand your response. Could you please tell me again?",
            "hinglish": "Main aapki baat samajh nahi paaya. Please dobara batayiye.",
        },
    }

    error_prompts = prompts.get(error_type, prompts["unclear_response"])
    return error_prompts.get(language, error_prompts["en"])
