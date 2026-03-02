"""Life event classification service."""

import logging
from typing import Any

from src.integrations.llm_client import get_llm_client
from src.prompts.loader import get_system_prompt

logger = logging.getLogger(__name__)

# Valid life event categories
LIFE_EVENTS = [
    "HOUSING",
    "MARRIAGE",
    "CHILDBIRTH",
    "EDUCATION",
    "HEALTH_CRISIS",
    "DEATH_IN_FAMILY",
    "MARITAL_DISTRESS",
    "JOB_LOSS",
    "BUSINESS_STARTUP",
    "WOMEN_EMPOWERMENT",
]

# Hindi keyword mappings for rule-based fallback
HINDI_KEYWORDS = {
    "HOUSING": ["घर", "मकान", "किराया", "जमीन", "प्लॉट", "फ्लैट", "अपार्टमेंट", "आवास", "pmay", "awas"],
    "MARRIAGE": ["शादी", "विवाह", "ब्याह", "wedding", "marriage"],
    "CHILDBIRTH": ["गर्भवती", "प्रेग्नेंट", "बच्चा", "शिशु", "माँ बनना", "डिलीवरी", "pregnant", "baby", "child"],
    "EDUCATION": ["पढ़ाई", "स्कूल", "कॉलेज", "शिक्षा", "डिग्री", "कोर्स", "education", "study", "loan"],
    "HEALTH_CRISIS": ["बीमार", "अस्पताल", "इलाज", "ऑपरेशन", "दवाई", "चिकित्सा", "hospital", "medical", "surgery", "illness"],
    "DEATH_IN_FAMILY": ["मृत्यु", "गुजर गए", "मर गए", "निधन", "विधवा", "widow", "death", "passed away"],
    "MARITAL_DISTRESS": ["तलाक", "छोड़ दिया", "अलग हो गए", "divorce", "separated", "abandoned"],
    "JOB_LOSS": ["नौकरी छूटी", "बेरोजगार", "काम नहीं", "निकाल दिया", "unemployed", "job loss", "fired"],
    "BUSINESS_STARTUP": ["व्यापार", "दुकान", "बिज़नेस", "स्वरोजगार", "business", "self-employed", "startup"],
    "WOMEN_EMPOWERMENT": ["महिला", "लड़की", "बेटी", "women", "girl child"],
}


def classify_by_keywords(text: str) -> str | None:
    """Rule-based fallback classification using keywords."""
    text_lower = text.lower()

    # Check each category's keywords
    matches = {}
    for event, keywords in HINDI_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in text_lower)
        if count > 0:
            matches[event] = count

    if matches:
        # Return category with most keyword matches
        return max(matches, key=matches.get)

    return None


async def classify_life_event(
    user_message: str,
    conversation_history: list[dict[str, str]],
    current_profile: dict[str, Any],
) -> str | None:
    """Classify the user's life event from their message.

    Uses LLM for nuanced understanding with keyword fallback.
    Returns life event category or None if unclear.
    """
    # Try rule-based first for common cases
    keyword_match = classify_by_keywords(user_message)
    if keyword_match:
        logger.debug(f"Life event classified by keywords: {keyword_match}")
        return keyword_match

    # Use LLM for more complex cases
    try:
        llm = get_llm_client()
        analysis = await llm.analyze_message(
            user_message=user_message,
            conversation_history=conversation_history,
            current_state="UNDERSTANDING",
            user_profile=current_profile,
            system_prompt=get_system_prompt(),
        )

        life_event = analysis.get("life_event")
        if life_event and life_event in LIFE_EVENTS:
            logger.debug(f"Life event classified by LLM: {life_event}")
            return life_event

    except Exception as e:
        logger.error(f"LLM life event classification failed: {e}")

    return None


def get_life_event_display_name(event: str, language: str = "hi") -> str:
    """Get display name for a life event."""
    names = {
        "HOUSING": {"hi": "आवास एवं संपत्ति", "en": "Housing & Property"},
        "MARRIAGE": {"hi": "विवाह", "en": "Marriage"},
        "CHILDBIRTH": {"hi": "जन्म एवं पालन-पोषण", "en": "Childbirth & Parenting"},
        "EDUCATION": {"hi": "शिक्षा", "en": "Education"},
        "HEALTH_CRISIS": {"hi": "स्वास्थ्य आपातकाल", "en": "Health Emergency"},
        "DEATH_IN_FAMILY": {"hi": "परिवार में मृत्यु", "en": "Death in Family"},
        "MARITAL_DISTRESS": {"hi": "वैवाहिक कठिनाई", "en": "Marital Distress"},
        "JOB_LOSS": {"hi": "नौकरी छूटना एवं बेरोजगारी", "en": "Job Loss & Unemployment"},
        "BUSINESS_STARTUP": {"hi": "व्यवसाय शुरू करना", "en": "Starting a Business"},
        "WOMEN_EMPOWERMENT": {"hi": "महिला सशक्तिकरण", "en": "Women Empowerment"},
    }
    event_names = names.get(event, {"hi": event, "en": event})
    return event_names.get(language, event_names["en"])
