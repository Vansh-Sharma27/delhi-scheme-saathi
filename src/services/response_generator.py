"""Response generation service."""

import logging
from typing import Any

from src.integrations.llm_client import get_llm_client
from src.models.session import ConversationState, Session
from src.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


def _pick_language_text(
    language: str,
    hi: str,
    en: str,
    hinglish: str | None = None,
) -> str:
    """Pick a text variant for the requested language."""
    if language == "hi":
        return hi
    if language == "hinglish":
        return hinglish or en
    return en


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
        "hinglish": (
            "Namaste! 🙏 Main Delhi Scheme Saathi hoon.\n\n"
            "Main aapko government welfare schemes mein help kar sakta hoon:\n"
            "• Housing assistance\n"
            "• Health support\n"
            "• Education loans\n"
            "• Pension schemes\n"
            "• Employment support\n\n"
            "Batayiye, aaj aapko kis tarah ki madad chahiye?"
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
            "hinglish": "Batayiye, aaj aapko kis tarah ki madad chahiye? (jaise housing, health, education, employment)",
        },
        "age": {
            "hi": "योजनाओं की पात्रता जाँचने के लिए, कृपया अपनी उम्र बताएं।",
            "en": "To check scheme eligibility, please tell me your age.",
            "hinglish": "Scheme eligibility check karne ke liye, please age batayiye.",
        },
        "category": {
            "hi": "आप किस श्रेणी में आते हैं? (SC/ST/OBC/General/EWS)",
            "en": "What is your category? (SC/ST/OBC/General/EWS)",
            "hinglish": "Aapki category kya hai? (SC/ST/OBC/General/EWS)",
        },
        "annual_income": {
            "hi": "आपकी वार्षिक पारिवारिक आय लगभग कितनी है?",
            "en": "What is your approximate annual family income?",
            "hinglish": "Approx annual family income kitni hai?",
        },
    }

    field_questions = questions.get(missing_field, questions["life_event"])
    return field_questions.get(language, field_questions["en"])


def generate_no_schemes_response(language: str = "hi") -> str:
    """Generate response when no schemes match the user's profile.

    Honest and helpful — doesn't blame the user, suggests actionable alternatives.
    """
    responses = {
        "hi": (
            "मुझे खेद है, आपकी जानकारी के अनुसार फ़िलहाल कोई योजना उपलब्ध नहीं है।\n\n"
            "आप ये कर सकते हैं:\n"
            "• किसी और विषय में योजना देखें (जैसे: घर, स्वास्थ्य, रोजगार, पेंशन)\n"
            "• अगर कोई जानकारी बदलनी हो तो बताएं\n"
            "• /start दबाकर नया विषय चुनें\n\n"
            "आप क्या करना चाहेंगे?"
        ),
        "en": (
            "Unfortunately, we don't currently have schemes matching your profile "
            "for this category.\n\n"
            "You can:\n"
            "• Explore a different area (housing, health, employment, pension)\n"
            "• Update your details if something has changed\n"
            "• Send /start to begin a fresh search\n\n"
            "What would you like to do?"
        ),
        "hinglish": (
            "Abhi aapki profile ke hisaab se is category mein matching scheme nahi mili.\n\n"
            "Aap ye kar sakte hain:\n"
            "• Kisi aur area mein dekh sakte hain (housing, health, employment, pension)\n"
            "• Agar koi detail badalni ho toh batayiye\n"
            "• /start bhejkar fresh search shuru kariye\n\n"
            "Aap kya karna chahenge?"
        ),
    }
    return responses.get(language, responses["en"])


def generate_farewell_response(language: str = "hi") -> str:
    """Generate farewell response when the user ends the conversation."""
    farewells = {
        "hi": (
            "धन्यवाद! 🙏 आपसे बात करके अच्छा लगा।\n\n"
            "जब भी सरकारी योजनाओं की जानकारी चाहिए, /start भेजें।\n"
            "शुभकामनाएं! 😊"
        ),
        "en": (
            "Thank you for using Delhi Scheme Saathi! 🙏\n\n"
            "Whenever you need help with government schemes, just send /start.\n"
            "Take care! 😊"
        ),
        "hinglish": (
            "Delhi Scheme Saathi use karne ke liye shukriya! 🙏\n\n"
            "Jab bhi government schemes mein help chahiye ho, bas /start bhej dijiye.\n"
            "Take care! 😊"
        ),
    }
    return farewells.get(language, farewells["en"])


def generate_scheme_selection_response(language: str = "hi") -> str:
    """Generate response asking user to select a scheme."""
    responses = {
        "hi": "इनमें से कौन सी योजना के बारे में आप विस्तार से जानना चाहते हैं?",
        "en": "Which scheme would you like to know more about?",
        "hinglish": "Inmein se kis scheme ke baare mein detail chahiye?",
    }
    return responses.get(language, responses["en"])


def generate_field_reason_response(field: str, language: str = "hi") -> str:
    """Explain why a specific field matters and re-ask it."""
    reasons = {
        "life_event": {
            "hi": "आपकी स्थिति जानने से मैं सही योजनाएं ढूँढ सकता हूं। कृपया बताएं कि आपको किस तरह की सहायता चाहिए?",
            "en": "Knowing your situation helps me find the right schemes. What kind of assistance do you need?",
            "hinglish": "Aapki situation samajhne se main sahi schemes dhoondh sakta hoon. Batayiye, kis tarah ki madad chahiye?",
        },
        "age": {
            "hi": "उम्र से मैं सही पात्रता जाँच सकता हूं। कृपया आवेदक की उम्र बताएं।",
            "en": "Age helps me check the right eligibility rules. Please share the applicant's age.",
            "hinglish": "Age se main sahi eligibility check kar sakta hoon. Please applicant ki age batayiye.",
        },
        "category": {
            "hi": "श्रेणी से सही योजनाएं और पात्रता तय होती हैं। कृपया SC/ST/OBC/General/EWS बताएं।",
            "en": "Category affects eligibility for many schemes. Please share SC/ST/OBC/General/EWS.",
            "hinglish": "Category se kai schemes ki eligibility decide hoti hai. Please SC/ST/OBC/General/EWS batayiye.",
        },
        "annual_income": {
            "hi": "आय से मैं सही पात्रता और लाभ जाँच सकता हूं। कृपया अनुमानित वार्षिक आय बताएं।",
            "en": "Income helps me check the right eligibility and benefits. Please share the approximate annual income.",
            "hinglish": "Income se main sahi eligibility aur benefits check kar sakta hoon. Please approx annual income batayiye.",
        },
        "gender": {
            "hi": "कुछ योजनाएं लिंग के आधार पर अलग होती हैं। कृपया बताएं आवेदक पुरुष हैं या महिला।",
            "en": "Some schemes differ by gender. Please tell me whether the applicant is male or female.",
            "hinglish": "Kuch schemes gender ke hisaab se alag hoti hain. Please batayiye applicant male hai ya female.",
        },
    }
    field_reasons = reasons.get(field, reasons["life_event"])
    return field_reasons.get(language, field_reasons["en"])


def generate_application_guidance(
    scheme_name: str,
    application_url: str | None,
    offline_process: str | None,
    language: str = "hi",
) -> str:
    """Generate application guidance response."""
    if language == "hi":
        response = f"📝 {scheme_name} के लिए आवेदन:\n\n"
        if application_url:
            response += f"🌐 ऑनलाइन आवेदन:\n{application_url}\n\n"
        if offline_process:
            response += f"🏛️ ऑफलाइन आवेदन:\n{offline_process}\n\n"
        response += "क्या आपको दस्तावेजों की जानकारी चाहिए, या कोई और सवाल है?"
        return response

    if language == "hinglish":
        response = f"📝 {scheme_name} ke liye apply kaise karein:\n\n"
        if application_url:
            response += f"🌐 Online apply:\n{application_url}\n\n"
        if offline_process:
            response += f"🏛️ Offline process:\n{offline_process}\n\n"
        response += "Kya aapko documents mein help chahiye, ya koi aur sawaal hai?"
        return response

    response = f"📝 How to Apply for {scheme_name}:\n\n"
    if application_url:
        response += f"🌐 Online:\n{application_url}\n\n"
    if offline_process:
        response += f"🏛️ Offline:\n{offline_process}\n\n"
    response += "Would you like help with documents, or do you have any other questions?"
    return response
