"""Response generation service."""

import logging
import re
from typing import Any

from src.db import scheme_repo
from src.integrations.llm_client import FallbackLLMClient
from src.models.scheme import Scheme
from src.models.session import Session, UserProfile
from src.prompts.loader import get_generate_response_prompt
from src.services.ai_orchestrator import get_ai_orchestrator

logger = logging.getLogger(__name__)

LANGUAGE_NORMALIZATION_PROMPT = """
You rewrite Delhi government welfare bot replies into the requested output language.

Rules:
- Rewrite only the text provided in context.source_text.
- Preserve all facts, numbers, rupee amounts, URLs, phone numbers, office names, addresses, and IDs.
- Keep the same order, bullets, and step structure.
- Do not add advice, omissions, or new facts.
- If a term is already a proper noun or official title, keep it intact and rewrite the surrounding text.
- For Hindi, output natural Hindi in Devanagari script.
- For Hinglish, output natural Roman-script Hinglish only. Do not use Devanagari.
- For English, output plain English only.
- Return only the rewritten reply text.
""".strip()

ELIGIBILITY_QUESTION_PATTERNS = (
    r"\beligib(?:le|ility)\b",
    r"\bqualif(?:y|ies|ied)\b",
    r"\bcriteria\b",
    r"\bcan (?:i|we|she|he|they) apply\b",
    r"\bwho can apply\b",
    r"\bam i eligible\b",
    r"\bdo(?:es)? .* qualify\b",
    r"\bpatar(?:ta|ता)\b",
    r"\bयोग्य\b",
    r"\bपात्र\b",
    r"qualify",
)
JUSTIFICATION_QUESTION_PATTERNS = (
    r"\bjustify\b",
    r"\bwhy (?:this|that) scheme\b",
    r"\bwhy did you suggest\b",
    r"\bwhy did you recommend\b",
    r"\bwhy was this shown\b",
    r"क्यों सुझा",
    r"क्यों recommend",
)


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


def _count_latin_tokens(text: str) -> int:
    """Count meaningful Latin-script words in a response."""
    return len(re.findall(r"[A-Za-z]{3,}", text))


def _count_devanagari_tokens(text: str) -> int:
    """Count meaningful Devanagari-script words in a response."""
    return len(re.findall(r"[\u0900-\u097F]{2,}", text))


def _count_hinglish_markers(text: str) -> int:
    """Count common Roman Hindi / Hinglish markers."""
    text_lower = text.lower()
    markers = (
        "aap", "apni", "apna", "hai", "hain", "kya", "kaise", "kyon",
        "kyu", "batayiye", "madad", "liye", "ke", "ki", "ka", "kar",
        "karna", "pooch", "dekhiye", "chahein", "yojana", "scheme",
    )
    return sum(
        1
        for marker in markers
        if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text_lower)
    )


def _needs_grounded_translation(text: str, language: str) -> bool:
    """Return True when deterministic grounded text still leaks source language."""
    if language not in {"hi", "en", "hinglish"}:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    latin_tokens = _count_latin_tokens(stripped)
    devanagari_tokens = _count_devanagari_tokens(stripped)

    if language == "hi":
        return latin_tokens > 0
    if language == "en":
        return devanagari_tokens > 0
    if devanagari_tokens > 0:
        return True
    return latin_tokens >= 4 and _count_hinglish_markers(stripped) < 2


def _needs_language_normalization(text: str, language: str) -> bool:
    """Return True when reply text drifts away from the requested language/script."""
    if language not in {"hi", "en", "hinglish"}:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    latin_tokens = _count_latin_tokens(stripped)
    devanagari_tokens = _count_devanagari_tokens(stripped)

    if language == "hi":
        if latin_tokens == 0:
            return False
        if devanagari_tokens == 0:
            return True
        return latin_tokens > devanagari_tokens

    if language == "en":
        return devanagari_tokens > 0

    if devanagari_tokens > 0:
        return True
    if latin_tokens == 0:
        return False

    hinglish_markers = _count_hinglish_markers(stripped)
    return latin_tokens >= 6 and hinglish_markers < 2


async def ensure_response_language(
    session: Session,
    text: str,
    language: str,
) -> str:
    """Rewrite a reply into the requested language when it drifted off target."""
    if not text.strip() or not _needs_language_normalization(text, language):
        return text

    return await _rewrite_response_language(session, text, language)


async def _rewrite_response_language(
    session: Session,
    text: str,
    language: str,
) -> str:
    """Rewrite a response using the shared Bedrock/Grok translation path."""
    if not text.strip():
        return text

    translated = await get_ai_orchestrator().generate_response(
        session=session,
        context={
            "response_mode": "language_normalization",
            "source_text": text,
        },
        system_prompt=LANGUAGE_NORMALIZATION_PROMPT,
        user_language=language,
    )

    if not translated.strip():
        return text

    safe_fallback = FallbackLLMClient._safe_generation_text(language)
    if translated.strip() == safe_fallback.strip():
        return text

    return translated.strip()


async def translate_grounded_text_if_needed(
    session: Session,
    text: str,
    language: str,
) -> str:
    """Faithfully translate grounded text when the target language needs it.

    This is only for already-grounded deterministic text. If the translation
    path is unavailable, we keep the original grounded text instead of
    returning an error stub.
    """
    if not text.strip() or not _needs_grounded_translation(text, language):
        return text
    return await _rewrite_response_language(session, text, language)


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
        system_prompt = get_generate_response_prompt()
    except FileNotFoundError:
        system_prompt = "Generate a helpful response based on the context."

    # Add state-specific context
    context["conversation_state"] = session.state.value
    context["user_profile"] = session.user_profile.model_dump()
    context["language"] = session.language_preference

    # Generate response via LLM
    response = await get_ai_orchestrator().generate_response(
        session=session,
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


def generate_help_response(
    language: str = "auto",
    *,
    has_active_scheme: bool = False,
) -> str:
    """Generate a deterministic help guide for users and judges."""
    generic_responses = {
        "hi": (
            "दिल्ली स्कीम साथी आपकी स्थिति के हिसाब से उपयोगी दिल्ली सरकारी योजनाएं समझने में मदद करता है।\n\n"
            "इसे कैसे उपयोग करें:\n"
            "• अपनी जरूरत एक सरल वाक्य में बताएं\n"
            '  उदाहरण: "मुझे विधवा पेंशन की जानकारी चाहिए"\n'
            "• अगर पूछा जाए तो उम्र, श्रेणी, आय, लिंग या स्थिति बताएं\n"
            "• किसी योजना पर आगे पूछें: पात्रता, लाभ, दस्तावेज, आवेदन प्रक्रिया, या यह योजना क्यों दिखाई गई\n"
            "• विषय बदलना हो तो सीधे कहें: अब मुझे किसी और मदद की जरूरत है\n\n"
            "उपयोगी कमांड और वाक्य:\n"
            "• /help : यह गाइड फिर से देखें\n"
            "• /language : भाषा बदलें\n"
            "• /start : नई खोज शुरू करें\n"
            '• "start over" या "restart" : चैट अपने-आप रीसेट करें\n'
            '• "bye" : बातचीत समाप्त करें\n\n'
            "अभी यह बॉट क्या कर सकता है:\n"
            "• दिल्ली की संबंधित योजनाएं ढूंढना\n"
            "• योजना को सरल भाषा में समझाना\n"
            "• आपकी दी हुई जानकारी के आधार पर संभावित पात्रता बताना\n"
            "• दस्तावेज, आवेदन चरण, और rejection warnings बताना\n"
            "• हिंदी, अंग्रेज़ी, और हिंग्लिश में जवाब देना\n\n"
            "महत्वपूर्ण:\n"
            "यह मार्गदर्शन के लिए है। अंतिम पात्रता और मंजूरी सरकारी नियमों और दस्तावेज़ सत्यापन पर निर्भर करती है।"
        ),
        "en": (
            "Delhi Scheme Saathi helps you understand which Delhi government welfare schemes may fit your situation.\n\n"
            "How to use it:\n"
            "• Start with one simple need\n"
            '  Example: "I need widow pension help"\n'
            "• Answer short questions like age, category, income, gender, or situation\n"
            "• Ask follow-up questions about eligibility, benefits, documents, application steps, or why a scheme was suggested\n"
            "• If your need changes, just say so and the bot can switch topics\n\n"
            "Useful commands and phrases:\n"
            "• /help : show this guide again\n"
            "• /language : change language\n"
            "• /start : begin a fresh search\n"
            '• "start over" or "restart" : reset the chat automatically\n'
            '• "bye" : end the conversation\n\n'
            "Current capabilities:\n"
            "• Find relevant Delhi welfare schemes\n"
            "• Explain scheme details in simple language\n"
            "• Share likely eligibility based on the details you provide\n"
            "• List documents, application steps, and common rejection warnings\n"
            "• Reply in English, Hindi, or Hinglish\n\n"
            "Important:\n"
            "This bot is a guidance tool. Final eligibility and approval depend on official government rules and document verification."
        ),
        "hinglish": (
            "Delhi Scheme Saathi aapki situation ke hisaab se useful Delhi government schemes samajhne mein help karta hai.\n\n"
            "Use kaise karein:\n"
            "• Apni need ek simple sentence mein batayiye\n"
            '  Example: "Mujhe widow pension help chahiye"\n'
            "• Agar poocha jaye toh age, category, income, gender ya situation batayiye\n"
            "• Scheme ke baare mein follow-up pooch sakte hain: eligibility, benefits, documents, application steps, ya yeh scheme kyon dikhayi gayi\n"
            "• Topic badalna ho toh seedha bol dijiye\n\n"
            "Useful commands aur phrases:\n"
            "• /help : yeh guide phir se dekhiye\n"
            "• /language : language badaliye\n"
            "• /start : fresh search shuru kijiye\n"
            '• "start over" ya "restart" : chat automatically reset ho jayegi\n'
            '• "bye" : conversation khatam kijiye\n\n'
            "Current capabilities:\n"
            "• Relevant Delhi welfare schemes dhoondhna\n"
            "• Scheme details simple language mein samjhana\n"
            "• Aapki details ke basis par likely eligibility batana\n"
            "• Documents, application steps, aur common rejection warnings batana\n"
            "• English, Hindi, aur Hinglish mein reply karna\n\n"
            "Important:\n"
            "Yeh guidance tool hai. Final eligibility aur approval official rules aur document verification par depend karta hai."
        ),
    }
    scheme_context_responses = {
        "hi": (
            "आप इस समय एक चुनी हुई योजना देख रहे हैं। इसी योजना पर आप सीधे ये सवाल पूछ सकते हैं:\n"
            "• क्या मैं पात्र हूं?\n"
            "• यह योजना क्यों दिखाई गई?\n"
            "• कौन-कौन से दस्तावेज चाहिए?\n"
            "• आवेदन कैसे करें?\n"
            "• किन गलतियों से बचना चाहिए?\n\n"
            "उपयोगी कमांड:\n"
            "• /language : भाषा बदलें\n"
            "• /start : नई खोज शुरू करें\n"
            '• "start over" : बातचीत रीसेट करें\n'
            '• "bye" : बातचीत समाप्त करें'
        ),
        "en": (
            "You are currently viewing a selected scheme. You can directly ask:\n"
            "• Am I eligible?\n"
            "• Why was this scheme suggested?\n"
            "• What documents are needed?\n"
            "• How do I apply?\n"
            "• What mistakes should I avoid?\n\n"
            "Useful commands:\n"
            "• /language : change language\n"
            "• /start : begin a fresh search\n"
            '• "start over" : reset the conversation\n'
            '• "bye" : end the conversation'
        ),
        "hinglish": (
            "Aap is waqt ek selected scheme dekh rahe hain. Aap seedha pooch sakte hain:\n"
            "• Kya main eligible hoon?\n"
            "• Yeh scheme kyon dikhayi gayi?\n"
            "• Kaun se documents chahiye?\n"
            "• Apply kaise karna hai?\n"
            "• Kin mistakes se bachna chahiye?\n\n"
            "Useful commands:\n"
            "• /language : language badaliye\n"
            "• /start : fresh search shuru kijiye\n"
            '• "start over" : conversation reset kijiye\n'
            '• "bye" : conversation khatam kijiye'
        ),
    }

    responses = scheme_context_responses if has_active_scheme else generic_responses

    if language not in responses:
        return (
            "Delhi Scheme Saathi Help\n\n"
            "ENGLISH\n"
            f"{generic_responses['en']}\n\n"
            "हिंदी\n"
            f"{generic_responses['hi']}"
        )
    return responses[language]


def generate_language_selection_response(language: str = "auto") -> str:
    """Prompt the user to choose a conversation language."""
    responses = {
        "hi": (
            "कृपया अपनी पसंदीदा भाषा चुनें।\n"
            "आप कभी भी /language दबाकर भाषा बदल सकते हैं।"
        ),
        "en": (
            "Choose your preferred language.\n"
            "You can switch again anytime with /language."
        ),
        "hinglish": (
            "Apni preferred language choose kijiye.\n"
            "Aap kabhi bhi /language se language badal sakte hain."
        ),
    }
    if language not in responses:
        return (
            "Choose your preferred language / अपनी पसंदीदा भाषा चुनें.\n"
            "You can switch again anytime with /language."
        )
    return responses[language]


def generate_language_changed_response(
    language: str,
    *,
    has_active_scheme: bool = False,
) -> str:
    """Confirm language change and suggest what to do next."""
    if has_active_scheme:
        return _pick_language_text(
            language,
            "भाषा बदल दी गई है। अब आप इसी योजना के बारे में पात्रता, दस्तावेज, लाभ, या आवेदन पूछ सकते हैं।",
            "Language updated. You can now continue asking about this scheme's eligibility, documents, benefits, or application steps.",
            "Language update ho gayi hai. Ab aap isi scheme ke baare mein eligibility, documents, benefits, ya application steps pooch sakte hain.",
        )
    return _pick_language_text(
        language,
        "भाषा बदल दी गई है। अब आप अपनी जरूरत बताइए या /help देखिए।",
        "Language updated. You can now tell me your need or use /help.",
        "Language update ho gayi hai. Ab aap apni need batayiye ya /help dekhiye.",
    )


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


def generate_field_help_response(field: str, language: str = "hi") -> str:
    """Explain how to answer a field question and re-ask it."""
    help_text = {
        "life_event": {
            "hi": "आप जिस मदद की तलाश कर रहे हैं, वही बताइए, जैसे: housing, widow pension, health treatment, education loan. आपको किस तरह की सहायता चाहिए?",
            "en": "Please tell me the kind of help you need, for example: housing, widow pension, health treatment, or education loan. What assistance are you looking for?",
            "hinglish": "Aapko kis type ki help chahiye woh batayiye, jaise housing, widow pension, health treatment ya education loan. Aap kis assistance ki talash mein hain?",
        },
        "age": {
            "hi": "कृपया पूरी उम्र सालों में बताएं, जैसे 24 या 45 years.",
            "en": "Please share the completed age in years, for example 24 or 45 years.",
            "hinglish": "Please completed age years mein batayiye, jaise 24 ya 45 years.",
        },
        "category": {
            "hi": "अगर पता हो तो इनमें से एक बताएं: SC, ST, OBC, General, EWS. अगर निश्चित न हों तो 'skip' भी लिख सकते हैं.",
            "en": "If you know it, please choose one: SC, ST, OBC, General, or EWS. If you are not sure, you can also say 'skip'.",
            "hinglish": "Agar pata ho to inmein se ek batayiye: SC, ST, OBC, General ya EWS. Agar sure nahi hain to 'skip' bhi bol sakte hain.",
        },
        "annual_income": {
            "hi": "अनुमानित वार्षिक पारिवारिक आय बताएं. अगर मासिक आय पता है, तो उसका 12 गुना बता सकते हैं, जैसे 50,000 monthly मतलब लगभग 6 लाख yearly.",
            "en": "Please share approximate annual family income. If you only know the monthly amount, you can multiply it by 12, for example 50,000 monthly is about 6 lakh yearly.",
            "hinglish": "Approx annual family income batayiye. Agar sirf monthly amount pata ho to uska 12 times bata sakte hain, jaise 50,000 monthly matlab roughly 6 lakh yearly.",
        },
        "gender": {
            "hi": "कृपया बताएं आवेदक male हैं या female. अगर योजना महिला-विशेष है, तो यह जानकारी जरूरी हो सकती है.",
            "en": "Please tell me whether the applicant is male or female. Some schemes are gender-specific, so this can matter.",
            "hinglish": "Please batayiye applicant male hai ya female. Kuch schemes gender-specific hoti hain, isliye yeh zaroori ho sakta hai.",
        },
    }
    field_help = help_text.get(field, help_text["life_event"])
    return field_help.get(language, field_help["en"])


def _format_currency(amount: int | float | None) -> str | None:
    """Format rupee values compactly for free-form answers."""
    if amount is None:
        return None
    if amount >= 100000:
        lakhs = amount / 100000
        return f"₹{lakhs:.1f} lakh" if lakhs != int(lakhs) else f"₹{int(lakhs)} lakh"
    return f"₹{amount:,.0f}"


def _infer_income_segment(income_limits: dict[str, int], annual_income: int | None) -> str | None:
    """Infer the first matching income band from configured cutoffs."""
    if annual_income is None:
        return None

    normalized_limits: list[tuple[str, int]] = []
    for segment, raw_limit in income_limits.items():
        try:
            normalized_limits.append((str(segment).upper(), int(raw_limit)))
        except (TypeError, ValueError):
            continue

    for segment, limit in sorted(normalized_limits, key=lambda item: item[1]):
        if annual_income <= limit:
            return segment
    return None


def _maybe_generate_scheme_term_response(
    scheme: Scheme,
    profile: UserProfile,
    user_question: str,
    language: str,
) -> str | None:
    """Answer common deterministic scheme-term questions without repeating the card."""
    text_lower = user_question.lower()
    asks_income_band = any(
        phrase in text_lower
        for phrase in ("income band", "lig", "mig", "ews", "income category")
    )
    if not asks_income_band:
        return None

    income_limits = scheme.eligibility.income_by_category
    if not income_limits:
        return None

    ordered_limits = []
    for segment, raw_limit in income_limits.items():
        try:
            ordered_limits.append((str(segment).upper(), int(raw_limit)))
        except (TypeError, ValueError):
            continue
    if not ordered_limits:
        return None

    ordered_limits.sort(key=lambda item: item[1])
    limit_text = ", ".join(
        f"{segment} up to {_format_currency(limit)}"
        for segment, limit in ordered_limits
    )
    user_segment = _infer_income_segment(income_limits, profile.annual_income)
    user_segment_text = None
    if user_segment and profile.annual_income is not None:
        user_segment_text = (
            f"At about {_format_currency(profile.annual_income)} annual income, you fit the {user_segment} band."
        )

    variants = {
        "hi": (
            "इस योजना में income band का मतलब वार्षिक पारिवारिक आय के आधार पर वर्ग है। "
            f"यहाँ bands हैं: {limit_text}. "
            + (
                f"आपकी करीब {_format_currency(profile.annual_income)} आय के हिसाब से आप {user_segment} band में आते हैं."
                if user_segment_text and profile.annual_income is not None
                else "अगर आप चाहें तो मैं बता सकता हूँ कि आपके लिए कौन सा band लागू होता है।"
            )
        ),
        "hinglish": (
            "Is scheme mein income band ka matlab annual family income ke hisaab se group hota hai. "
            f"Yahan bands hain: {limit_text}. "
            + (
                f"Aapki roughly {_format_currency(profile.annual_income)} income ke hisaab se aap {user_segment} band mein aate hain."
                if user_segment_text and profile.annual_income is not None
                else "Agar chahein to main aapke income ke hisaab se relevant band bhi bata sakta hoon."
            )
        ),
        "en": (
            "In this scheme, income band means the annual family income bracket used to decide which segment applies. "
            f"Here the bands are: {limit_text}. "
            + (
                user_segment_text
                if user_segment_text
                else "If you want, I can also tell you which band your income falls into."
            )
        ),
    }
    return variants.get(language, variants["en"])


def _is_eligibility_question(user_question: str) -> bool:
    """Return True when the user is asking about eligibility or qualification."""
    return any(
        re.search(pattern, user_question, re.IGNORECASE)
        for pattern in ELIGIBILITY_QUESTION_PATTERNS
    )


def _is_justification_question(user_question: str) -> bool:
    """Return True when the user asks why the scheme was suggested."""
    return any(
        re.search(pattern, user_question, re.IGNORECASE)
        for pattern in JUSTIFICATION_QUESTION_PATTERNS
    )


def _eligibility_field_label(field: str, language: str) -> str:
    """Render a field label in the response language."""
    labels = {
        "age": {"hi": "उम्र", "en": "age", "hinglish": "age"},
        "gender": {"hi": "लिंग", "en": "gender", "hinglish": "gender"},
        "income": {"hi": "आय", "en": "income", "hinglish": "income"},
        "category": {"hi": "श्रेणी", "en": "category", "hinglish": "category"},
    }
    field_labels = labels.get(field, labels["age"])
    return field_labels.get(language, field_labels["en"])


def _build_eligibility_rule_text(scheme: Scheme, language: str) -> list[str]:
    """Build concise human-readable rule bullets from structured eligibility data."""
    elig = scheme.eligibility
    rules: list[str] = []

    if elig.min_age is not None and elig.max_age is not None:
        rules.append(
            _pick_language_text(
                language,
                f"उम्र {elig.min_age} से {elig.max_age} वर्ष के बीच",
                f"age between {elig.min_age} and {elig.max_age}",
                f"age {elig.min_age} se {elig.max_age} ke beech",
            )
        )
    elif elig.min_age is not None:
        rules.append(
            _pick_language_text(
                language,
                f"उम्र कम से कम {elig.min_age} वर्ष",
                f"age {elig.min_age} or above",
                f"age kam se kam {elig.min_age} years",
            )
        )
    elif elig.max_age is not None:
        rules.append(
            _pick_language_text(
                language,
                f"उम्र {elig.max_age} वर्ष तक",
                f"age up to {elig.max_age}",
                f"age {elig.max_age} tak",
            )
        )

    restricted_genders = [gender for gender in elig.genders if gender.lower() != "all"]
    if restricted_genders:
        if len(restricted_genders) == 1 and restricted_genders[0].lower() == "female":
            rules.append(
                _pick_language_text(
                    language,
                    "महिला आवेदक",
                    "women applicants",
                    "female applicants",
                )
            )
        elif len(restricted_genders) == 1 and restricted_genders[0].lower() == "male":
            rules.append(
                _pick_language_text(
                    language,
                    "पुरुष आवेदक",
                    "male applicants",
                    "male applicants",
                )
            )
        else:
            gender_text = ", ".join(restricted_genders)
            rules.append(
                _pick_language_text(
                    language,
                    f"लिंग शर्त: {gender_text}",
                    f"gender condition: {gender_text}",
                    f"gender condition: {gender_text}",
                )
            )

    if elig.max_income is not None:
        income_text = _format_currency(elig.max_income)
        rules.append(
            _pick_language_text(
                language,
                f"वार्षिक पारिवारिक आय {income_text} तक",
                f"annual family income up to {income_text}",
                f"annual family income {income_text} tak",
            )
        )
    elif elig.income_by_category:
        limits = ", ".join(
            f"{segment.upper()} {_format_currency(limit)}"
            for segment, limit in sorted(
                (
                    (str(segment), int(limit))
                    for segment, limit in elig.income_by_category.items()
                ),
                key=lambda item: item[1],
            )
        )
        rules.append(
            _pick_language_text(
                language,
                f"आय सीमा band के अनुसार: {limits}",
                f"income limits depend on the band: {limits}",
                f"income limit band ke hisaab se hai: {limits}",
            )
        )

    if elig.caste_categories and not any(category.upper() == "ALL" for category in elig.caste_categories):
        category_text = ", ".join(elig.caste_categories)
        rules.append(
            _pick_language_text(
                language,
                f"श्रेणी शर्त: {category_text}",
                f"category condition: {category_text}",
                f"category condition: {category_text}",
            )
        )

    return rules


def _maybe_generate_eligibility_response(
    scheme: Scheme,
    profile: UserProfile,
    user_question: str,
    language: str,
) -> str | None:
    """Answer eligibility questions from structured scheme data without using the LLM."""
    if not _is_eligibility_question(user_question):
        return None

    rule_text = _build_eligibility_rule_text(scheme, language)
    if not rule_text:
        return None

    elig = scheme.eligibility
    match = scheme_repo.calculate_eligibility_match(scheme, profile)

    failed_fields: list[str] = []
    missing_fields: list[str] = []
    checked_fields: list[str] = []

    if elig.min_age is not None or elig.max_age is not None:
        if profile.age is None:
            missing_fields.append("age")
        else:
            checked_fields.append("age")
            if not match.get("age", True):
                failed_fields.append("age")

    restricted_genders = [gender for gender in elig.genders if gender.lower() != "all"]
    if restricted_genders:
        if profile.gender is None:
            missing_fields.append("gender")
        else:
            checked_fields.append("gender")
            if not match.get("gender", True):
                failed_fields.append("gender")

    if elig.max_income is not None or elig.income_by_category:
        if profile.annual_income is None:
            missing_fields.append("income")
        else:
            checked_fields.append("income")
            if not match.get("income", True):
                failed_fields.append("income")

    has_category_restriction = (
        bool(elig.caste_categories)
        and not any(category.upper() == "ALL" for category in elig.caste_categories)
    )
    if has_category_restriction:
        if profile.category is None:
            missing_fields.append("category")
        else:
            checked_fields.append("category")
            if not match.get("category", True):
                failed_fields.append("category")

    checked_fields_text = ", ".join(
        _eligibility_field_label(field, language)
        for field in checked_fields
    )
    missing_fields_text = ", ".join(
        _eligibility_field_label(field, language)
        for field in missing_fields
    )
    failed_fields_text = ", ".join(
        _eligibility_field_label(field, language)
        for field in failed_fields
    )

    category_note = None
    if profile.category and not has_category_restriction:
        category_note = _pick_language_text(
            language,
            f"{profile.category} श्रेणी अपने-आप में समस्या नहीं है, क्योंकि मेरे पास जो scheme data है उसमें caste-category restriction नहीं दिख रही।",
            f"{profile.category} category does not disqualify the applicant because this scheme does not have a caste-category restriction in the data I have.",
            f"{profile.category} category se problem nahi hai, kyunki mere paas jo scheme data hai usmein caste-category restriction nahi dikh rahi.",
        )
    elif profile.category and has_category_restriction and "category" in failed_fields:
        allowed_categories = ", ".join(elig.caste_categories)
        category_note = _pick_language_text(
            language,
            f"मेरे पास जो data है उसके अनुसार यह scheme केवल {allowed_categories} श्रेणी के लिए है, इसलिए {profile.category} match नहीं करती।",
            f"In the scheme data I have, this scheme is limited to {allowed_categories}, so {profile.category} does not match the category rule.",
            f"Mere paas jo scheme data hai uske hisaab se yeh scheme sirf {allowed_categories} ke liye hai, isliye {profile.category} category match nahi karti.",
        )

    if failed_fields:
        status_text = _pick_language_text(
            language,
            f"अभी साझा की गई जानकारी के आधार पर applicant {failed_fields_text} check पर fit नहीं लगते।",
            f"Based on the details shared so far, the applicant does not appear eligible on the {failed_fields_text} check.",
            f"Ab tak ki details ke hisaab se applicant {failed_fields_text} check par fit nahi lagte.",
        )
    elif missing_fields:
        status_text = _pick_language_text(
            language,
            f"अभी तक की जानकारी के आधार पर applicant eligible हो सकते हैं, लेकिन {missing_fields_text} confirm करना बाकी है।",
            f"Based on the details shared so far, the applicant may qualify, but {missing_fields_text} still needs to be confirmed.",
            f"Ab tak ki details ke hisaab se applicant qualify kar sakte hain, lekin {missing_fields_text} abhi confirm karna baaki hai.",
        )
    else:
        status_text = _pick_language_text(
            language,
            f"अभी साझा की गई जानकारी के आधार पर applicant {checked_fields_text} checks पर eligible लगते हैं।",
            f"Based on the details shared so far, the applicant appears eligible on the available {checked_fields_text} checks.",
            f"Ab tak ki details ke hisaab se applicant available {checked_fields_text} checks par eligible lagte hain.",
        )

    final_note = _pick_language_text(
        language,
        "अंतिम मंजूरी दस्तावेज़ जाँच और विभाग की बाकी शर्तों पर भी निर्भर करेगी।",
        "Final approval will still depend on document verification and any other departmental checks.",
        "Final approval documents verification aur department ke baaki checks par bhi depend karega.",
    )

    lines = [
        _pick_language_text(
            language,
            f"मेरे पास जो scheme data है उसके अनुसार मुख्य eligibility checks हैं: {'; '.join(rule_text)}.",
            f"From the scheme data I have, the main eligibility checks are: {'; '.join(rule_text)}.",
            f"Mere paas jo scheme data hai uske hisaab se main eligibility checks hain: {'; '.join(rule_text)}.",
        ),
    ]
    if category_note:
        lines.append(category_note)
    lines.append(status_text)
    lines.append(final_note)
    return " ".join(lines)


def _maybe_generate_scheme_justification_response(
    scheme: Scheme,
    profile: UserProfile,
    user_question: str,
    language: str,
) -> str | None:
    """Answer grounded why-this-scheme questions without free-form generation."""
    if not _is_justification_question(user_question):
        return None

    reasons = _build_matching_reason_context(scheme, profile)
    if not reasons:
        return _pick_language_text(
            language,
            "मैं इस scheme के बारे में सिर्फ वही कारण बताना चाहता हूँ जो data में साफ़ दिखते हैं। अभी मेरे पास इतना grounded match data नहीं है कि मैं भरोसे से reason बता सकूँ। अगर आप चाहें तो मैं उम्र, आय, श्रेणी या दूसरी eligibility details दोबारा check कर सकता हूँ।",
            "I only want to explain this scheme using grounded facts from the data. Right now I do not have enough matched rule evidence to justify it confidently. If you want, I can re-check the age, income, category, or other eligibility details.",
            "Main is scheme ko sirf grounded data ke basis par explain karna chahta hoon. Abhi mere paas itna matched rule evidence nahi hai ki main confidently reason bata sakoon. Agar chahein to main age, income, category ya doosri eligibility details dobara check kar sakta hoon.",
        )

    reasons_text = "; ".join(reasons[:3])
    return _pick_language_text(
        language,
        f"मैंने यह scheme इन grounded कारणों से दिखाई: {reasons_text} अगर चाहें तो मैं इसमें documents या application steps भी समझा सकता हूँ।",
        f"I suggested this scheme for grounded reasons from the current data: {reasons_text} If you want, I can also explain the documents or application steps.",
        f"Maine yeh scheme current data ke grounded reasons ki wajah se dikhayi: {reasons_text} Agar chahein to main documents ya application steps bhi samjha sakta hoon.",
    )


def _last_assistant_response(session: Session) -> str | None:
    """Return the most recent assistant reply for translation/rephrase follow-ups."""
    for message in reversed(session.messages):
        if message.role == "assistant" and message.content.strip():
            return message.content.strip()[:1200]
    return None


def _build_matching_reason_context(scheme: Scheme, profile: UserProfile) -> list[str]:
    """Collect grounded reasons the scheme could fit the current profile."""
    reasons: list[str] = []
    elig = scheme.eligibility
    eligibility_match = scheme_repo.calculate_eligibility_match(scheme, profile)
    scheme_text = " ".join(
        [
            scheme.name,
            scheme.name_hindi,
            scheme.description[:400],
            scheme.description_hindi[:400],
            " ".join(scheme.tags[:12]),
        ]
    ).lower()

    if profile.life_event and profile.life_event in scheme.life_events:
        reasons.append(f"The scheme is tagged for the same need area: {profile.life_event}.")

    if (
        profile.marital_status == "widowed"
        and any(keyword in scheme_text for keyword in ("widow", "widowed", "vidhwa", "विधवा"))
    ):
        reasons.append("The scheme itself is specifically framed for widowed women.")

    if (
        profile.gender
        and eligibility_match.get("gender")
        and elig.genders
        and "all" not in [gender.lower() for gender in elig.genders]
    ):
        reasons.append(f"The scheme is gender-restricted and the user profile says {profile.gender}.")

    if (
        profile.age is not None
        and eligibility_match.get("age")
        and (elig.min_age is not None or elig.max_age is not None)
    ):
        reasons.append(
            f"Age rule in context: {elig.min_age or 18}-{elig.max_age or 'no upper limit'}; user age: {profile.age}."
        )

    if profile.annual_income is not None and eligibility_match.get("income"):
        income_text = _format_currency(profile.annual_income)
        if elig.max_income is not None:
            reasons.append(f"Scheme max income in context: {_format_currency(elig.max_income)}; user income: {income_text}.")
        elif elig.income_by_category and eligibility_match.get("income_segment"):
            reasons.append(f"User income: {income_text}; the scheme uses income bands {', '.join(sorted(elig.income_by_category))}.")

    if (
        profile.category
        and eligibility_match.get("category")
        and elig.caste_categories
        and not any(category.upper() == "ALL" for category in elig.caste_categories)
    ):
        reasons.append(f"Scheme category condition in context: {', '.join(elig.caste_categories)}; user category: {profile.category}.")

    if elig.special_focus_groups:
        reasons.append(f"Special focus groups in context: {', '.join(elig.special_focus_groups[:8])}.")

    return reasons


async def generate_scheme_question_response(
    session: Session,
    scheme: Scheme,
    profile: UserProfile,
    user_question: str,
    language: str,
    *,
    active_view: str | None = None,
) -> str:
    """Answer a follow-up question about a selected scheme using grounded context."""
    term_response = _maybe_generate_scheme_term_response(scheme, profile, user_question, language)
    if term_response:
        return term_response
    justification_response = _maybe_generate_scheme_justification_response(
        scheme,
        profile,
        user_question,
        language,
    )
    if justification_response:
        return justification_response
    eligibility_response = _maybe_generate_eligibility_response(
        scheme,
        profile,
        user_question,
        language,
    )
    if eligibility_response:
        return eligibility_response

    current_scheme = scheme.model_dump(mode="json")
    current_scheme["description"] = scheme.description[:1200]
    current_scheme["description_hindi"] = scheme.description_hindi[:1200]

    context = {
        "response_mode": "scheme_question_answer",
        "active_view": active_view or session.state.value,
        "user_question": user_question,
        "current_scheme": current_scheme,
        "last_assistant_response": _last_assistant_response(session),
        "matching_reasons": _build_matching_reason_context(scheme, profile),
        "answer_style_rules": [
            "Answer the exact question first.",
            "Do not repeat the full scheme card unless the user asked for a full overview.",
            "If the user asks why this scheme was suggested, cite only matching_reasons and current_scheme facts.",
            "If the user asks for a term meaning, explain it plainly and practically.",
            "If the user asks for the same information in another language, translate or restate the last_assistant_response when it is relevant.",
        ],
    }
    return await generate_response(session, context)


def generate_application_guidance(
    scheme_name: str,
    application_url: str | None,
    offline_process: str | None,
    application_steps: list[str] | None = None,
    processing_time: str | None = None,
    helpline_phone: str | None = None,
    language: str = "hi",
) -> str:
    """Generate application guidance response."""
    if language == "hi":
        response = f"📝 {scheme_name} के लिए आवेदन:\n\n"
        if application_steps:
            response += "📋 आवेदन के चरण:\n"
            response += "\n".join(application_steps) + "\n\n"
        if application_url:
            response += f"🌐 ऑनलाइन आवेदन:\n{application_url}\n\n"
        if offline_process:
            response += f"🏛️ ऑफलाइन आवेदन:\n{offline_process}\n\n"
        if processing_time:
            response += f"⏱️ प्रक्रिया समय:\n{processing_time}\n\n"
        if helpline_phone:
            response += f"📞 संपर्क:\n{helpline_phone}\n\n"
        response += "क्या आपको दस्तावेजों की जानकारी चाहिए, या कोई और सवाल है?"
        return response

    if language == "hinglish":
        response = f"📝 {scheme_name} ke liye apply kaise karein:\n\n"
        if application_steps:
            response += "📋 Step-by-step process:\n"
            response += "\n".join(application_steps) + "\n\n"
        if application_url:
            response += f"🌐 Online apply:\n{application_url}\n\n"
        if offline_process:
            response += f"🏛️ Offline process:\n{offline_process}\n\n"
        if processing_time:
            response += f"⏱️ Processing time:\n{processing_time}\n\n"
        if helpline_phone:
            response += f"📞 Contact:\n{helpline_phone}\n\n"
        response += "Kya aapko documents mein help chahiye, ya koi aur sawaal hai?"
        return response

    response = f"📝 How to Apply for {scheme_name}:\n\n"
    if application_steps:
        response += "📋 Step-by-step process:\n"
        response += "\n".join(application_steps) + "\n\n"
    if application_url:
        response += f"🌐 Online:\n{application_url}\n\n"
    if offline_process:
        response += f"🏛️ Offline:\n{offline_process}\n\n"
    if processing_time:
        response += f"⏱️ Processing time:\n{processing_time}\n\n"
    if helpline_phone:
        response += f"📞 Contact:\n{helpline_phone}\n\n"
    response += "Would you like help with documents, or do you have any other questions?"
    return response
