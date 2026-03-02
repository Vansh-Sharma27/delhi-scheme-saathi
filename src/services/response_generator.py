"""Response generation service."""

import logging
from typing import Any

from src.integrations.llm_client import get_llm_client
from src.models.session import ConversationState, Session
from src.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


async def generate_response(
    session: Session,
    context: dict[str, Any],
) -> str:
    """Generate natural language response using LLM and database context.

    Context includes:
    - matched_schemes: list of scheme matches
    - current_scheme: selected scheme details
    - documents: required documents
    - rejection_warnings: applicable rules
    - nearest_offices: nearby CSCs
    """
    # Load response generation prompt
    try:
        system_prompt = load_prompt("generate_response")
    except FileNotFoundError:
        system_prompt = "Generate a helpful response based on the context."

    # Add state-specific context
    context["conversation_state"] = session.state.value
    context["user_profile"] = session.user_profile.model_dump()
    context["language"] = session.language_preference

    # Generate response via LLM
    llm = get_llm_client()
    response = await llm.generate_response(
        context=context,
        system_prompt=system_prompt,
        user_language=session.language_preference,
    )

    return response


def generate_greeting_response(language: str = "hi") -> str:
    """Generate greeting response without LLM."""
    greetings = {
        "hi": (
            "नमस्ते! 🙏 मैं दिल्ली स्कीम साथी हूं।\n\n"
            "मैं आपको सरकारी योजनाओं की जानकारी देने में मदद कर सकता हूं:\n"
            "• घर खरीदना/बनाना\n"
            "• स्वास्थ्य सहायता\n"
            "• शिक्षा ऋण\n"
            "• पेंशन योजनाएं\n"
            "• रोजगार सहायता\n\n"
            "आप मुझे बताएं, आज आपको किस तरह की सहायता चाहिए?"
        ),
        "en": (
            "Namaste! 🙏 I am Delhi Scheme Saathi.\n\n"
            "I can help you with government welfare schemes for:\n"
            "• Housing assistance\n"
            "• Health support\n"
            "• Education loans\n"
            "• Pension schemes\n"
            "• Employment support\n\n"
            "Please tell me, what kind of assistance do you need today?"
        ),
    }
    return greetings.get(language, greetings["en"])


def generate_clarification_response(
    missing_field: str,
    language: str = "hi",
) -> str:
    """Generate response asking for missing information."""
    questions = {
        "life_event": {
            "hi": "आप मुझे बताएं, आज आपको किस तरह की सहायता चाहिए? (जैसे: घर, स्वास्थ्य, शिक्षा, रोजगार)",
            "en": "Please tell me, what kind of assistance do you need? (e.g., housing, health, education, employment)",
        },
        "age": {
            "hi": "योजनाओं की पात्रता जाँचने के लिए, कृपया अपनी उम्र बताएं।",
            "en": "To check scheme eligibility, please tell me your age.",
        },
        "category": {
            "hi": "आप किस श्रेणी में आते हैं? (SC/ST/OBC/General/EWS)",
            "en": "What is your category? (SC/ST/OBC/General/EWS)",
        },
        "annual_income": {
            "hi": "आपकी वार्षिक पारिवारिक आय लगभग कितनी है?",
            "en": "What is your approximate annual family income?",
        },
    }

    field_questions = questions.get(missing_field, questions["life_event"])
    return field_questions.get(language, field_questions["en"])


def generate_no_schemes_response(language: str = "hi") -> str:
    """Generate response when no schemes match."""
    responses = {
        "hi": (
            "मुझे खेद है, आपके प्रोफाइल के अनुसार कोई योजना नहीं मिली।\n\n"
            "कुछ सुझाव:\n"
            "• अपनी जानकारी दोबारा जाँचें\n"
            "• नजदीकी CSC केंद्र से संपर्क करें\n\n"
            "क्या आप कोई और जानकारी जोड़ना चाहेंगे?"
        ),
        "en": (
            "I'm sorry, no schemes matched your profile.\n\n"
            "Suggestions:\n"
            "• Please verify your information\n"
            "• Contact nearest CSC center\n\n"
            "Would you like to provide any additional information?"
        ),
    }
    return responses.get(language, responses["en"])


def generate_scheme_selection_response(language: str = "hi") -> str:
    """Generate response asking user to select a scheme."""
    responses = {
        "hi": "इनमें से कौन सी योजना के बारे में आप विस्तार से जानना चाहते हैं?",
        "en": "Which scheme would you like to know more about?",
    }
    return responses.get(language, responses["en"])


def generate_application_guidance(
    scheme_name: str,
    application_url: str | None,
    offline_process: str | None,
    language: str = "hi",
) -> str:
    """Generate application guidance response."""
    if language == "hi":
        response = f"*{scheme_name}* के लिए आवेदन:\n\n"
        if application_url:
            response += f"🌐 *ऑनलाइन आवेदन:*\n{application_url}\n\n"
        if offline_process:
            response += f"🏛️ *ऑफलाइन आवेदन:*\n{offline_process}\n\n"
        response += "क्या आपको दस्तावेजों की जानकारी चाहिए?"
    else:
        response = f"*{scheme_name}* Application:\n\n"
        if application_url:
            response += f"🌐 *Online Application:*\n{application_url}\n\n"
        if offline_process:
            response += f"🏛️ *Offline Application:*\n{offline_process}\n\n"
        response += "Would you like information about required documents?"

    return response
