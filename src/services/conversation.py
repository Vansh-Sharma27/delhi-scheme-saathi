"""Main conversation orchestrator.

This is the brain of Delhi Scheme Saathi. It:
1. Loads/creates session
2. Analyzes user input via LLM
3. Updates user profile
4. Executes FSM transitions
5. Runs state-specific logic (matching, document resolution, etc.)
6. Generates structured response using formatters (not LLM)
7. Saves session state
"""

import logging
from typing import Any

import asyncpg

from src.db import scheme_repo, office_repo
from src.integrations.llm_client import get_llm_client
from src.models.api import ChatRequest, ChatResponse
from src.models.document import DocumentChain
from src.models.scheme import SchemeMatch
from src.models.session import ConversationState, Session, UserProfile
from src.prompts.loader import get_system_prompt
from src.services import (
    fsm,
    life_event_classifier,
    profile_extractor,
    scheme_matcher,
    document_resolver,
    rejection_engine,
    response_generator,
    session_manager,
)
from src.utils.validators import sanitize_input
from src.utils.formatters import format_inline_keyboard

logger = logging.getLogger(__name__)

# Icons for life event categories
LIFE_EVENT_ICONS = {
    "HOUSING": "🏠",
    "HEALTH_CRISIS": "🏥",
    "EDUCATION": "📚",
    "DEATH_IN_FAMILY": "🙏",
    "MARITAL_DISTRESS": "🙏",
    "BUSINESS_STARTUP": "💼",
    "JOB_LOSS": "💼",
    "WOMEN_EMPOWERMENT": "👩",
    "CHILDBIRTH": "👶",
    "MARRIAGE": "💒",
}


# ---------------------------------------------------------------------------
# Plain-text formatting helpers (no MarkdownV2 — sent as plain Telegram text)
# ---------------------------------------------------------------------------

def _format_currency_plain(amount: int | float | None, language: str = "hi") -> str:
    """Format amount as Indian currency with lakh/crore notation."""
    if amount is None:
        return ""
    if amount >= 10000000:
        crores = amount / 10000000
        lbl = "करोड़" if language == "hi" else "Cr"
        return f"₹{crores:.1f} {lbl}" if crores != int(crores) else f"₹{int(crores)} {lbl}"
    if amount >= 100000:
        lakhs = amount / 100000
        lbl = "लाख" if language == "hi" else "lakh"
        return f"₹{lakhs:.1f} {lbl}" if lakhs != int(lakhs) else f"₹{int(lakhs)} {lbl}"
    return f"₹{amount:,.0f}"


def _build_scheme_list_text(
    schemes: list[SchemeMatch],
    profile: UserProfile,
    language: str,
) -> str:
    """Build a numbered, plain-text scheme list with eligibility info."""
    if not schemes:
        return "कोई योजना नहीं मिली।" if language == "hi" else "No matching schemes found."

    header = "🎯 आपके लिए ये योजनाएं मिली हैं:" if language == "hi" else "🎯 Found these schemes for you:"
    lines = [header, ""]

    for i, match in enumerate(schemes[:5], 1):
        scheme = match.scheme

        icon = "📋"
        for event in scheme.life_events:
            if event in LIFE_EVENT_ICONS:
                icon = LIFE_EVENT_ICONS[event]
                break

        name = scheme.name_hindi if language == "hi" else scheme.name
        lines.append(f"{i}. {icon} {name}")

        # Benefits
        if scheme.benefits_amount:
            amount_str = _format_currency_plain(scheme.benefits_amount, language)
            freq_map = {
                "monthly": "मासिक" if language == "hi" else "/month",
                "yearly": "वार्षिक" if language == "hi" else "/year",
                "one-time": "एकमुश्त" if language == "hi" else "one-time",
                "installments": "किश्तों में" if language == "hi" else "in installments",
            }
            freq_display = freq_map.get(scheme.benefits_frequency or "", "")
            benefit_label = "लाभ" if language == "hi" else "Benefit"
            lines.append(f"   💰 {benefit_label}: {amount_str} {freq_display}".rstrip())

        # Department
        dept = scheme.department_hindi if language == "hi" else scheme.department
        if len(dept) > 40:
            dept = dept[:37] + "..."
        dept_label = "विभाग" if language == "hi" else "Dept"
        lines.append(f"   🏛️ {dept_label}: {dept}")

        # Eligibility match
        if match.eligibility_match:
            field_labels = {
                "age": ("आयु", "Age"),
                "income": ("आय", "Income"),
                "category": ("श्रेणी", "Category"),
                "gender": ("लिंग", "Gender"),
            }
            parts = []
            for field, is_match in match.eligibility_match.items():
                hi_lbl, en_lbl = field_labels.get(field, (field, field))
                lbl = hi_lbl if language == "hi" else en_lbl
                mark = "✓" if is_match else "✗"
                parts.append(f"{lbl} {mark}")

            all_match = all(match.eligibility_match.values())
            prefix = ("✅ पात्र" if language == "hi" else "✅ Eligible") if all_match else ("⚠️ जाँचें" if language == "hi" else "⚠️ Check")
            lines.append(f"   {prefix}: {' • '.join(parts)}")

        lines.append("")  # blank line between schemes

    footer = "👆 नीचे बटन दबाएं या नंबर बताएं।" if language == "hi" else "👆 Tap a button below or type the number."
    lines.append(footer)
    return "\n".join(lines)


async def _build_scheme_details_text(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile,
    language: str,
) -> str:
    """Build rich plain-text scheme details with documents and warnings."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return "योजना नहीं मिली।" if language == "hi" else "Scheme not found."

    icon = "📋"
    for event in scheme.life_events:
        if event in LIFE_EVENT_ICONS:
            icon = LIFE_EVENT_ICONS[event]
            break

    name = scheme.name_hindi if language == "hi" else scheme.name
    lines = [f"{icon} {name}", ""]

    # Short description
    desc = scheme.description_hindi if language == "hi" else scheme.description
    if len(desc) > 250:
        desc = desc[:247] + "..."
    lines.append(desc)
    lines.append("")

    # Benefits
    if scheme.benefits_amount:
        amount_str = _format_currency_plain(scheme.benefits_amount, language)
        benefit_label = "लाभ राशि" if language == "hi" else "Benefit"
        lines.append(f"💰 {benefit_label}: {amount_str}")

    # Eligibility summary
    elig = scheme.eligibility
    elig_parts = []
    if elig.min_age or elig.max_age:
        age_lbl = "आयु" if language == "hi" else "Age"
        elig_parts.append(f"{age_lbl}: {elig.min_age or 18}-{elig.max_age or '∞'}")
    if elig.max_income:
        income_str = _format_currency_plain(elig.max_income, language)
        income_lbl = "अधिकतम आय" if language == "hi" else "Max income"
        elig_parts.append(f"{income_lbl}: {income_str}")
    if elig.categories:
        cat_lbl = "श्रेणी" if language == "hi" else "Category"
        elig_parts.append(f"{cat_lbl}: {', '.join(elig.categories)}")
    if elig_parts:
        elig_label = "पात्रता" if language == "hi" else "Eligibility"
        lines.append(f"✅ {elig_label}: {' | '.join(elig_parts)}")
    lines.append("")

    # Documents
    documents = await document_resolver.resolve_documents_for_scheme(
        pool, scheme.documents_required
    )
    if documents:
        doc_header = "📄 आवश्यक दस्तावेज:" if language == "hi" else "📄 Required Documents:"
        lines.append(doc_header)
        for idx, chain in enumerate(documents[:6], 1):
            doc = chain.document
            doc_name = doc.name_hindi if language == "hi" else doc.name
            lines.append(f"  {idx}. {doc_name}")

            where_lbl = "कहाँ से" if language == "hi" else "Where"
            extra = ""
            if doc.fee:
                fee_lbl = "शुल्क" if language == "hi" else "Fee"
                fee_val = f"₹{doc.fee}" if doc.fee.isdigit() else doc.fee
                extra += f", {fee_lbl}: {fee_val}"
            if doc.processing_time:
                time_lbl = "समय" if language == "hi" else "Time"
                extra += f", {time_lbl}: {doc.processing_time}"
            lines.append(f"     🏛️ {where_lbl}: {doc.issuing_authority}{extra}")

            if doc.online_portal:
                online_lbl = "ऑनलाइन" if language == "hi" else "Online"
                lines.append(f"     🌐 {online_lbl}: {doc.online_portal}")

            if doc.common_mistakes:
                warn_lbl = "ध्यान" if language == "hi" else "Note"
                lines.append(f"     ⚠️ {warn_lbl}: {doc.common_mistakes[0][:80]}")
        lines.append("")

    # Rejection warnings
    warnings = await rejection_engine.get_rejection_warnings(pool, scheme_id, profile)
    if warnings:
        warn_header = "⚠️ अस्वीकृति से बचें:" if language == "hi" else "⚠️ Avoid Rejection:"
        lines.append(warn_header)
        severity_icons = {"critical": "🔴", "high": "🟠", "warning": "🟡"}
        for rule in sorted(warnings[:5], key=lambda r: r.severity_order):
            sev_icon = severity_icons.get(rule.severity, "⚠️")
            rule_desc = rule.description_hindi if language == "hi" else rule.description
            lines.append(f"  {sev_icon} {rule_desc[:80]}")
            if rule.prevention_tip:
                tip_lbl = "बचाव" if language == "hi" else "Tip"
                lines.append(f"     ✅ {tip_lbl}: {rule.prevention_tip[:80]}")
        lines.append("")

    # Application info
    if scheme.application_url:
        apply_lbl = "ऑनलाइन आवेदन" if language == "hi" else "Apply online"
        lines.append(f"🔗 {apply_lbl}: {scheme.application_url}")
    if scheme.offline_process:
        offline_lbl = "ऑफलाइन आवेदन" if language == "hi" else "Offline"
        lines.append(f"🏛️ {offline_lbl}: {scheme.offline_process[:120]}")

    lines.append("")
    cta = "क्या आप इस योजना के लिए आवेदन करना चाहते हैं?" if language == "hi" else "Would you like to apply for this scheme?"
    lines.append(cta)
    return "\n".join(lines)


async def _build_handoff_text(
    pool: asyncpg.Pool,
    profile: UserProfile,
    language: str,
) -> str:
    """Build handoff text with nearby office info."""
    offices = []
    if profile.latitude and profile.longitude:
        offices = await office_repo.get_nearest_offices(
            pool, profile.latitude, profile.longitude, 3, "CSC"
        )
    elif profile.district:
        offices = await office_repo.get_offices_by_district(
            pool, profile.district, 3
        )

    if language == "hi":
        lines = ["🏛️ आपकी और सहायता के लिए नजदीकी सेवा केंद्र:", ""]
    else:
        lines = ["🏛️ Nearest service centers for further help:", ""]

    if offices:
        for office in offices[:3]:
            lines.append(f"📍 {office.name}")
            if office.address:
                lines.append(f"   📫 {office.address[:60]}")
            if office.phone:
                lines.append(f"   📞 {office.phone}")
            if office.working_hours:
                hrs_lbl = "समय" if language == "hi" else "Hours"
                lines.append(f"   🕐 {hrs_lbl}: {office.working_hours}")
            lines.append("")
    else:
        no_office = "नजदीकी केंद्र की जानकारी उपलब्ध नहीं है।" if language == "hi" else "No nearby center information available."
        lines.append(no_office)
        lines.append("")

    cta = "कृपया अपने सभी दस्तावेज लेकर जाएं।" if language == "hi" else "Please carry all your documents."
    lines.append(cta)
    return "\n".join(lines)


def _resolve_scheme_from_text(session: Session, text: str) -> str | None:
    """Try to match user text to a previously presented scheme by number or name."""
    presented = session.metadata.get("presented_schemes", [])
    if not presented:
        return None

    text_lower = text.strip().lower()

    # Match by number: "1", "2", "first", "पहला", etc.
    number_map = {
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "पहला": 0, "पहली": 0, "दूसरा": 1, "दूसरी": 1,
        "तीसरा": 2, "तीसरी": 2, "चौथा": 3, "पांचवां": 4,
    }
    for keyword, index in number_map.items():
        if keyword in text_lower and index < len(presented):
            return presented[index]["id"]

    # Match by scheme name (partial match on keywords > 3 chars)
    for scheme_info in presented:
        name_lower = scheme_info.get("name", "").lower()
        name_hindi = scheme_info.get("name_hindi", "")
        # Full name in text or text in full name
        if name_lower and (name_lower in text_lower or text_lower in name_lower):
            return scheme_info["id"]
        if name_hindi and (name_hindi in text or text.strip() in name_hindi):
            return scheme_info["id"]
        # Match significant words from English name
        for word in name_lower.split():
            if len(word) > 3 and word in text_lower:
                return scheme_info["id"]

    return None


def _store_presented_schemes(session: Session, schemes: list[SchemeMatch]) -> Session:
    """Store presented scheme info in session metadata for text-based selection."""
    presented = [
        {"id": m.scheme.id, "name": m.scheme.name, "name_hindi": m.scheme.name_hindi}
        for m in schemes[:5]
    ]
    metadata = dict(session.metadata)
    metadata["presented_schemes"] = presented
    return Session(
        user_id=session.user_id,
        state=session.state,
        user_profile=session.user_profile,
        messages=session.messages,
        conversation_summary=session.conversation_summary,
        discussed_schemes=session.discussed_schemes,
        selected_scheme_id=session.selected_scheme_id,
        language_preference=session.language_preference,
        created_at=session.created_at,
        updated_at=session.updated_at,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Conversation service
# ---------------------------------------------------------------------------

class ConversationService:
    """Main conversation orchestrator."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool
        self.llm = get_llm_client()

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        """Handle incoming user message and return response.

        Main flow:
        1. Load/create session
        2. Sanitize input
        3. LLM analysis (intent + life_event + entities in one call)
        4. Update profile (immutable merge)
        5. FSM transition
        6. Execute state-specific logic with formatted responses
        7. Save session
        """
        # 1. Load or create session
        session = await session_manager.get_or_create_session(request.user_id)

        # 2. Sanitize input
        user_message = sanitize_input(request.message)
        if not user_message:
            return ChatResponse(
                text="मुझे आपका संदेश समझ नहीं आया। कृपया दोबारा लिखें।"
            )

        # Handle /start command — always reset and greet
        if user_message.strip().lower() in ("/start", "/shuru"):
            session = session_manager.reset_session(session)
            lang = session.language_preference if session.language_preference != "auto" else "hi"
            response_text = response_generator.generate_greeting_response(lang)
            session = await session_manager.add_message(session, "user", user_message)
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)
            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
            )

        # Handle callback data (scheme selection via inline keyboard)
        if request.message_type == "callback" and request.callback_data:
            return await self._handle_callback(session, request.callback_data)

        # 3. Analyze message via LLM (intent + entities only)
        conversation_history = session_manager.get_conversation_history(
            session,
            include_assistant=False,
        )
        analysis = await self.llm.analyze_message(
            user_message=user_message,
            conversation_history=conversation_history,
            current_state=session.state.value,
            user_profile=session.user_profile.model_dump(),
            system_prompt=get_system_prompt(),
        )

        intent = analysis.get("intent", "unknown")
        detected_life_event = analysis.get("life_event")
        extracted_fields = analysis.get("extracted_fields", {})
        detected_language = analysis.get("language", "hi")
        selected_scheme_id = analysis.get("selected_scheme_id")

        # Deterministic extraction fallback for common profile fields.
        rule_based_fields = profile_extractor.extract_by_patterns(user_message)
        extracted_fields = {**extracted_fields, **rule_based_fields}

        # Deterministic life-event fallback when LLM returns null/unknown.
        if not detected_life_event:
            detected_life_event = life_event_classifier.classify_by_keywords(user_message)

        # Update language preference
        if detected_language and detected_language != "auto":
            if detected_language != session.language_preference:
                session = session_manager.set_language(session, detected_language)

        # Explicitly reset full session context on goodbye.
        if intent == "goodbye":
            session = session_manager.reset_session(session)
            if detected_language and detected_language != "auto":
                session = session_manager.set_language(session, detected_language)

            response_text = response_generator.generate_greeting_response(
                session.language_preference
            )
            session = await session_manager.add_message(session, "user", user_message)
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
            )

        # 4. Update profile with extracted fields
        if extracted_fields:
            new_profile = UserProfile(**{
                k: v for k, v in extracted_fields.items() if v is not None
            })
            session = session_manager.update_profile(session, new_profile)

        # Add life event to profile if detected
        if detected_life_event and not session.user_profile.life_event:
            session = session_manager.update_profile(
                session,
                UserProfile(life_event=detected_life_event)
            )

        # 5. Determine next FSM state
        profile = session.user_profile
        next_state = fsm.determine_next_state(
            current_state=session.state,
            profile=profile,
            intent=intent,
            selected_scheme_id=selected_scheme_id,
            has_selected_scheme=bool(session.selected_scheme_id),
        )

        # 6. Execute state-specific logic
        lang = session.language_preference if session.language_preference != "auto" else "hi"
        schemes: list[SchemeMatch] = []
        documents: list[DocumentChain] = []
        warnings = []
        offices = []
        inline_keyboard = None
        response_text = ""

        match next_state:

            case ConversationState.GREETING:
                # Deliver warm greeting (now reachable for greeting intent)
                if session.state == ConversationState.HANDOFF:
                    session = session_manager.reset_session(session)
                response_text = response_generator.generate_greeting_response(lang)

            case ConversationState.UNDERSTANDING:
                # Ask for next missing profile field
                next_question = profile_extractor.get_next_question(profile, lang)
                if next_question:
                    response_text = next_question
                else:
                    # All fields filled but FSM didn't trigger MATCHING
                    # (edge case safety) — run matching inline
                    next_state, response_text, schemes, inline_keyboard, session = (
                        await self._run_matching(profile, user_message, session, lang)
                    )

            case ConversationState.MATCHING:
                # Run scheme matching and build formatted response
                next_state, response_text, schemes, inline_keyboard, session = (
                    await self._run_matching(profile, user_message, session, lang)
                )

            case ConversationState.PRESENTING:
                # Try text-based scheme selection first
                resolved_id = selected_scheme_id or _resolve_scheme_from_text(
                    session, user_message
                )
                if resolved_id:
                    # Transition to DETAILS — build details inline (no fallthrough)
                    session = session_manager.select_scheme(session, resolved_id)
                    next_state = ConversationState.DETAILS
                    response_text = await _build_scheme_details_text(
                        self.pool, resolved_id, profile, lang
                    )
                else:
                    # Re-present schemes with keyboard
                    schemes = await scheme_matcher.match_schemes(
                        pool=self.pool, profile=profile
                    )
                    if schemes:
                        session = _store_presented_schemes(session, schemes)
                        inline_keyboard = format_inline_keyboard(schemes, lang)
                    select_prompt = (
                        "इनमें से कौन सी योजना के बारे में जानना चाहते हैं? नंबर बताएं या बटन दबाएं।"
                        if lang == "hi"
                        else "Which scheme would you like to know more about? Type the number or tap a button."
                    )
                    response_text = select_prompt

            case ConversationState.DETAILS:
                scheme_id = session.selected_scheme_id or selected_scheme_id
                if not scheme_id:
                    # No scheme selected — fall back to presenting
                    next_state = ConversationState.PRESENTING
                    select_prompt = (
                        "कृपया पहले एक योजना चुनें।"
                        if lang == "hi"
                        else "Please select a scheme first."
                    )
                    response_text = select_prompt
                else:
                    response_text = await _build_scheme_details_text(
                        self.pool, scheme_id, profile, lang
                    )

            case ConversationState.APPLICATION:
                scheme_id = session.selected_scheme_id
                if scheme_id:
                    scheme = await scheme_repo.get_scheme_by_id(self.pool, scheme_id)
                    if scheme:
                        response_text = response_generator.generate_application_guidance(
                            scheme.name_hindi if lang == "hi" else scheme.name,
                            scheme.application_url,
                            scheme.offline_process,
                            lang,
                        )
                    else:
                        response_text = (
                            "आवेदन जानकारी उपलब्ध नहीं है।"
                            if lang == "hi"
                            else "Application info not available."
                        )
                else:
                    response_text = (
                        "कृपया पहले एक योजना चुनें।"
                        if lang == "hi"
                        else "Please select a scheme first."
                    )

            case ConversationState.HANDOFF:
                response_text = await _build_handoff_text(self.pool, profile, lang)

            case _:
                response_text = response_generator.generate_greeting_response(lang)

        # 7. Update session state and save
        session = session_manager.update_state(session, next_state)
        session = await session_manager.add_message(session, "user", user_message)
        session = await session_manager.add_message(session, "assistant", response_text)
        await session_manager.save_session(session)

        return ChatResponse(
            text=response_text,
            schemes=schemes,
            documents=documents,
            rejection_warnings=warnings,
            offices=offices,
            inline_keyboard=inline_keyboard,
            next_state=next_state.value,
        )

    async def _run_matching(
        self,
        profile: UserProfile,
        user_message: str,
        session: Session,
        lang: str,
    ) -> tuple[ConversationState, str, list[SchemeMatch], list | None, Session]:
        """Run scheme matching and return (next_state, text, schemes, keyboard, session)."""
        schemes = await scheme_matcher.match_schemes(
            pool=self.pool,
            profile=profile,
            query_text=user_message,
        )

        if schemes:
            session = _store_presented_schemes(session, schemes)
            response_text = _build_scheme_list_text(schemes, profile, lang)
            inline_keyboard = format_inline_keyboard(schemes, lang)
            return ConversationState.PRESENTING, response_text, schemes, inline_keyboard, session

        no_match_text = response_generator.generate_no_schemes_response(lang)
        return ConversationState.HANDOFF, no_match_text, [], None, session

    async def _handle_callback(
        self,
        session: Session,
        callback_data: str,
    ) -> ChatResponse:
        """Handle callback query (scheme selection from inline keyboard)."""
        lang = session.language_preference if session.language_preference != "auto" else "hi"

        if callback_data.startswith("scheme:"):
            scheme_id = callback_data.replace("scheme:", "")

            # Select scheme and transition to DETAILS
            session = session_manager.select_scheme(session, scheme_id)
            session = session_manager.update_state(session, ConversationState.DETAILS)

            # Build rich details response
            response_text = await _build_scheme_details_text(
                self.pool, scheme_id, session.user_profile, lang
            )

            # Save session
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.DETAILS.value,
            )

        return ChatResponse(text="अमान्य चयन।" if lang == "hi" else "Invalid selection.")
