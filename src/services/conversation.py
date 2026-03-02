"""Main conversation orchestrator.

This is the brain of Delhi Scheme Saathi. It:
1. Loads/creates session
2. Analyzes user input via LLM
3. Updates user profile
4. Executes FSM transitions
5. Runs state-specific logic (matching, document resolution, etc.)
6. Generates natural language response
7. Saves session state
"""

import logging
from typing import Any

import asyncpg

from src.db import scheme_repo, document_repo, office_repo, rejection_rule_repo
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
from src.utils.formatters import format_scheme_card, format_inline_keyboard

logger = logging.getLogger(__name__)


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
        6. Execute state-specific logic
        7. Generate response
        8. Save session
        """
        # 1. Load or create session
        session = await session_manager.get_or_create_session(request.user_id)

        # 2. Sanitize input
        user_message = sanitize_input(request.message)
        if not user_message:
            return ChatResponse(
                text="मुझे आपका संदेश समझ नहीं आया। कृपया दोबारा लिखें।"
            )

        # Handle callback data (scheme selection)
        if request.message_type == "callback" and request.callback_data:
            return await self._handle_callback(session, request.callback_data)

        # 3. Analyze message via LLM
        conversation_history = session_manager.get_conversation_history(session)
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

        # Update language preference
        if detected_language != session.language_preference:
            session = session_manager.set_language(session, detected_language)

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
        )

        # 6. Execute state-specific logic
        context: dict[str, Any] = {
            "user_profile": profile.model_dump(),
            "language": session.language_preference,
        }
        schemes: list[SchemeMatch] = []
        documents: list[DocumentChain] = []
        warnings = []
        offices = []
        inline_keyboard = None

        match next_state:
            case ConversationState.GREETING:
                # Reset session if coming from HANDOFF
                if session.state == ConversationState.HANDOFF:
                    session = session_manager.reset_session(session)
                response_text = response_generator.generate_greeting_response(
                    session.language_preference
                )

            case ConversationState.UNDERSTANDING:
                # Check what information we still need
                missing = profile_extractor.get_missing_fields(profile)
                if missing:
                    next_question = profile_extractor.get_next_question(
                        profile, session.language_preference
                    )
                    context["missing_fields"] = missing
                    response_text = next_question or await response_generator.generate_response(
                        session, context
                    )
                else:
                    # Ready for matching
                    next_state = ConversationState.MATCHING
                    response_text = ""  # Will be set in MATCHING

            case ConversationState.MATCHING:
                # Run scheme matching
                schemes = await scheme_matcher.match_schemes(
                    pool=self.pool,
                    profile=profile,
                    query_text=user_message,
                )

                if schemes:
                    next_state = ConversationState.PRESENTING
                    context["matched_schemes"] = [
                        scheme_matcher.format_scheme_for_display(m, session.language_preference)
                        for m in schemes
                    ]

                    # Generate inline keyboard for scheme selection
                    inline_keyboard = format_inline_keyboard(schemes, session.language_preference)
                    response_text = await response_generator.generate_response(session, context)
                else:
                    next_state = ConversationState.HANDOFF
                    response_text = response_generator.generate_no_schemes_response(
                        session.language_preference
                    )

            case ConversationState.PRESENTING:
                # User might select a scheme or ask for more info
                if selected_scheme_id:
                    # Handle scheme selection
                    session = session_manager.select_scheme(session, selected_scheme_id)
                    next_state = ConversationState.DETAILS
                    # Fall through to DETAILS handling
                else:
                    # Re-present schemes or ask for clarification
                    schemes = await scheme_matcher.match_schemes(
                        pool=self.pool,
                        profile=profile,
                    )
                    context["matched_schemes"] = [
                        scheme_matcher.format_scheme_for_display(m, session.language_preference)
                        for m in schemes
                    ]
                    inline_keyboard = format_inline_keyboard(schemes, session.language_preference)
                    response_text = response_generator.generate_scheme_selection_response(
                        session.language_preference
                    )

            case ConversationState.DETAILS:
                # Get full scheme details
                scheme_id = session.selected_scheme_id or selected_scheme_id
                if not scheme_id:
                    # Fall back to presenting
                    next_state = ConversationState.PRESENTING
                    response_text = response_generator.generate_scheme_selection_response(
                        session.language_preference
                    )
                else:
                    scheme = await scheme_repo.get_scheme_by_id(self.pool, scheme_id)
                    if scheme:
                        # Get documents
                        documents = await document_resolver.resolve_documents_for_scheme(
                            self.pool, scheme.documents_required
                        )

                        # Get rejection warnings
                        warnings = await rejection_engine.get_rejection_warnings(
                            self.pool, scheme_id, profile
                        )

                        # Get nearest offices if user has location
                        if profile.latitude and profile.longitude:
                            offices = await office_repo.get_nearest_offices(
                                self.pool, profile.latitude, profile.longitude, 3
                            )

                        context["current_scheme"] = scheme.model_dump()
                        context["documents"] = [
                            document_resolver.format_document_guide(d, session.language_preference)
                            for d in documents
                        ]
                        context["rejection_warnings"] = [
                            rejection_engine.format_rejection_warning(w, session.language_preference)
                            for w in warnings[:5]
                        ]
                        context["nearest_offices"] = [o.model_dump() for o in offices]

                        response_text = await response_generator.generate_response(
                            session, context
                        )
                    else:
                        response_text = "योजना नहीं मिली।" if session.language_preference == "hi" else "Scheme not found."

            case ConversationState.APPLICATION:
                # Application guidance
                scheme_id = session.selected_scheme_id
                if scheme_id:
                    scheme = await scheme_repo.get_scheme_by_id(self.pool, scheme_id)
                    if scheme:
                        response_text = response_generator.generate_application_guidance(
                            scheme.name_hindi if session.language_preference == "hi" else scheme.name,
                            scheme.application_url,
                            scheme.offline_process,
                            session.language_preference,
                        )
                    else:
                        response_text = "आवेदन जानकारी उपलब्ध नहीं है।"
                else:
                    response_text = "कृपया पहले एक योजना चुनें।"

            case ConversationState.HANDOFF:
                # Connect to CSC
                if profile.latitude and profile.longitude:
                    offices = await office_repo.get_nearest_offices(
                        self.pool, profile.latitude, profile.longitude, 3, "CSC"
                    )
                elif profile.district:
                    offices = await office_repo.get_offices_by_district(
                        self.pool, profile.district, 3
                    )

                context["nearest_offices"] = [o.model_dump() for o in offices]
                response_text = await response_generator.generate_response(session, context)

            case _:
                response_text = await response_generator.generate_response(session, context)

        # 7. Update session state
        session = session_manager.update_state(session, next_state)

        # Add messages to session
        session = await session_manager.add_message(session, "user", user_message)
        session = await session_manager.add_message(session, "assistant", response_text)

        # 8. Save session
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

    async def _handle_callback(
        self,
        session: Session,
        callback_data: str,
    ) -> ChatResponse:
        """Handle callback query (scheme selection from inline keyboard)."""
        # Parse callback data (format: "scheme:{scheme_id}")
        if callback_data.startswith("scheme:"):
            scheme_id = callback_data.replace("scheme:", "")

            # Select scheme and transition to DETAILS
            session = session_manager.select_scheme(session, scheme_id)
            session = session_manager.update_state(session, ConversationState.DETAILS)

            # Get scheme details
            scheme = await scheme_repo.get_scheme_by_id(self.pool, scheme_id)
            if not scheme:
                return ChatResponse(text="योजना नहीं मिली।")

            # Get documents and warnings
            documents = await document_resolver.resolve_documents_for_scheme(
                self.pool, scheme.documents_required
            )
            warnings = await rejection_engine.get_rejection_warnings(
                self.pool, scheme_id
            )

            # Build context
            context = {
                "current_scheme": scheme.model_dump(),
                "documents": [
                    document_resolver.format_document_guide(d, session.language_preference)
                    for d in documents
                ],
                "rejection_warnings": [
                    rejection_engine.format_rejection_warning(w, session.language_preference)
                    for w in warnings[:5]
                ],
                "language": session.language_preference,
            }

            # Generate response
            response_text = await response_generator.generate_response(session, context)

            # Save session
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                documents=documents,
                rejection_warnings=warnings,
                next_state=ConversationState.DETAILS.value,
            )

        return ChatResponse(text="अमान्य चयन।")
