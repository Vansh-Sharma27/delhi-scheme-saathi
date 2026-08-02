"""The conversation orchestrator.

One user message is one turn, and every turn runs the same pipeline:

1. load the session and sanitize the message
2. short-circuit commands and inline-keyboard callbacks
3. analyse the message with the LLM, then correct that analysis with
   deterministic extraction and action overrides (:meth:`_analyze_turn`)
4. settle the response language (:meth:`_resolve_language`)
5. apply extracted fields to the profile (:meth:`_apply_profile_updates`)
6. pick the next FSM state (:meth:`_decide_next_state`)
7. render that state (:meth:`_render_state`)
8. enforce the response language, persist, reply (:meth:`_finalize_turn`)

The LLM proposes; the deterministic layers dispose. Anything the LLM says
that conflicts with what the user's own words plainly mean is overridden,
because a wrong-but-fluent reply is worse here than a plain one.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import asyncpg

from src.config import get_settings
from src.models.api import ChatRequest, ChatResponse
from src.models.scheme import SchemeMatch
from src.models.session import ConversationState, Session, UserProfile
from src.prompts.loader import get_analysis_system_prompt
from src.services import (
    fsm,
    life_event_classifier,
    profile_extractor,
    response_generator,
    scheme_matcher,
    scheme_relevance,
    session_manager,
)
from src.services.ai_background import enqueue_memory_refresh
from src.services.ai_orchestrator import get_ai_orchestrator
from src.services.conversation import intents, language, scheme_reference, turn_policy, views
from src.services.conversation_memory import should_refresh_working_memory
from src.utils.keyboards import (
    format_inline_keyboard,
    format_language_keyboard,
    format_presented_scheme_keyboard,
)
from src.utils.validators import sanitize_input

logger = logging.getLogger(__name__)

_CALLBACK_LANGUAGE_PREFIX = "lang:"
_CALLBACK_SCHEME_PREFIX = "scheme:"

# Scheme views that render a single selected scheme and share the same
# "resolve a scheme id first" preamble.
_SCHEME_VIEW_STATES = {
    ConversationState.SCHEME_DETAILS,
    ConversationState.DOCUMENT_GUIDANCE,
    ConversationState.REJECTION_WARNINGS,
    ConversationState.APPLICATION_HELP,
}


@dataclass(frozen=True)
class TurnAnalysis:
    """What the analysis phase concluded about a single user message."""

    intent: str
    action: str | None
    detected_life_event: str | None
    extracted_fields: dict[str, Any]
    llm_response_text: str | None
    resolved_scheme_id: str | None
    explicit_language: str | None
    explicit_topic_switch: bool
    detected_language: str
    # Language read off the raw message, and whether a low-context reply means
    # the session should keep its current language instead of following it.
    inferred_turn_language: str
    preserve_unlocked_language: bool


@dataclass(frozen=True)
class ProfileUpdate:
    """How this turn changed the stored profile."""

    before_profile: UserProfile
    changed_fields: set[str]

    @property
    def profile_changed(self) -> bool:
        """True when any match-relevant field took a new value."""
        return bool(self.changed_fields)

    @property
    def matching_inputs_changed(self) -> bool:
        """True when a field the scheme search depends on took a new value."""
        return bool(self.changed_fields & turn_policy.MATCH_RELEVANT_FIELDS)


@dataclass
class RenderResult:
    """The reply produced for one FSM state, plus any session edits it made."""

    session: Session
    state: ConversationState
    text: str
    schemes: list[SchemeMatch] = field(default_factory=list)
    inline_keyboard: list[list[dict[str, str]]] | None = None


class ConversationService:
    """Main conversation orchestrator."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool
        self.settings = get_settings()
        self.ai = get_ai_orchestrator()
        # Keep the raw client reachable for existing tests and narrow mocks.
        self.llm = self.ai.llm_client

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        """Handle one incoming user message and return the reply."""
        session = await session_manager.get_or_create_session(request.user_id)

        user_message = sanitize_input(request.message)
        if not user_message:
            return ChatResponse(
                text="मुझे आपका संदेश समझ नहीं आया। कृपया दोबारा लिखें।",
                language=(
                    session.language_preference
                    if session.language_preference != "auto"
                    else "hi"
                ),
            )

        command = intents.extract_supported_command(user_message)
        if command:
            return await self._handle_command(session, command, user_message)

        if request.message_type == "callback" and request.callback_data:
            return await self._handle_callback(session, request.callback_data)

        analysis = await self._analyze_turn(session, user_message)
        session, lang, language_changed = self._resolve_language(session, analysis)

        early_reply = await self._handle_turn_reset(
            session,
            analysis,
            user_message=user_message,
            lang=lang,
            language_changed=language_changed,
        )
        if early_reply is not None:
            return early_reply

        session, profile_update = self._apply_profile_updates(session, analysis, user_message)
        profile = session.user_profile

        next_state, requested_state = self._decide_next_state(
            session,
            analysis,
            profile_update,
            user_message,
        )

        result = await self._render_state(
            session=session,
            next_state=next_state,
            requested_state=requested_state,
            analysis=analysis,
            profile_update=profile_update,
            user_message=user_message,
            lang=lang,
        )

        response_text = result.text
        if (
            profile.life_event == "DEATH_IN_FAMILY"
            and "life_event" in profile_update.changed_fields
            and profile_update.before_profile.life_event is None
        ):
            response_text = language.prepend_death_in_family_empathy(response_text, lang)

        return await self._finalize_turn(
            session=result.session,
            next_state=result.state,
            user_message=user_message,
            response_text=response_text,
            schemes=result.schemes,
            inline_keyboard=result.inline_keyboard,
            lang=lang,
        )

    # ------------------------------------------------------------------
    # Commands and callbacks
    # ------------------------------------------------------------------

    async def _handle_command(
        self,
        session: Session,
        command: str,
        user_message: str,
    ) -> ChatResponse:
        """Answer /start, /help and /language without calling the LLM."""
        if command == "start":
            session = session_manager.reset_session(session)
            lang = session.language_preference if session.language_preference != "auto" else "hi"
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_generator.generate_greeting_response(lang),
                language=lang,
            )

        if command == "help":
            response_text = response_generator.generate_help_response(
                session.language_preference,
                has_active_scheme=bool(
                    session.selected_scheme_id
                    and session.state in scheme_reference.SCHEME_CONTEXT_STATES
                ),
            )
        else:
            response_text = response_generator.generate_language_selection_response(
                session.language_preference
            )

        return await self._build_command_response(
            session,
            user_message=user_message,
            response_text=response_text,
            language=language.command_response_language(session),
            inline_keyboard=format_language_keyboard(session.language_preference),
        )

    async def _handle_callback(
        self,
        session: Session,
        callback_data: str,
    ) -> ChatResponse:
        """Handle a tap on an inline keyboard button."""
        lang = session.language_preference if session.language_preference != "auto" else "hi"

        if callback_data.startswith(_CALLBACK_LANGUAGE_PREFIX):
            return await self._handle_language_callback(
                session,
                callback_data.removeprefix(_CALLBACK_LANGUAGE_PREFIX),
                lang,
            )

        if callback_data.startswith(_CALLBACK_SCHEME_PREFIX):
            return await self._handle_scheme_callback(
                session,
                callback_data.removeprefix(_CALLBACK_SCHEME_PREFIX),
                lang,
            )

        return ChatResponse(
            text="अमान्य चयन।" if lang == "hi" else "Invalid selection.",
            language=lang,
        )

    async def _handle_language_callback(
        self,
        session: Session,
        requested_language: str,
        lang: str,
    ) -> ChatResponse:
        """Switch language and re-render the user's current context in it."""
        if requested_language not in language.SUPPORTED_LANGUAGES:
            return ChatResponse(
                text="अमान्य भाषा चयन।" if lang == "hi" else "Invalid language selection.",
                language=lang,
            )

        session = session_manager.set_language(session, requested_language, locked=True)
        state_text, inline_keyboard = await self._render_state_snapshot(
            session,
            requested_language,
        )
        response_text = (
            response_generator.generate_language_changed_response(
                requested_language,
                has_active_scheme=bool(
                    session.selected_scheme_id
                    and session.state in scheme_reference.SCHEME_CONTEXT_STATES
                ),
            )
            + "\n\n"
            + state_text
        )
        response_text = await response_generator.ensure_response_language(
            session,
            response_text,
            requested_language,
        )

        session = await session_manager.add_message(session, "assistant", response_text)
        await session_manager.save_session(session)

        return ChatResponse(
            text=response_text,
            next_state=session.state.value,
            language=requested_language,
            inline_keyboard=inline_keyboard,
        )

    async def _handle_scheme_callback(
        self,
        session: Session,
        scheme_id: str,
        lang: str,
    ) -> ChatResponse:
        """Open the scheme behind a tapped button."""
        session = session_manager.select_scheme(session, scheme_id)
        session = session_manager.update_state(session, ConversationState.SCHEME_DETAILS)

        response_text = await views.build_scheme_details_text(
            self.pool, scheme_id, session.user_profile, lang
        )
        response_text = await response_generator.ensure_response_language(
            session,
            response_text,
            lang,
        )

        session = await session_manager.add_message(session, "assistant", response_text)
        await session_manager.save_session(session)

        return ChatResponse(
            text=response_text,
            next_state=ConversationState.SCHEME_DETAILS.value,
            language=lang,
        )

    # ------------------------------------------------------------------
    # Phase 3 — analysis
    # ------------------------------------------------------------------

    async def _analyze_turn(
        self,
        session: Session,
        user_message: str,
    ) -> TurnAnalysis:
        """Run LLM analysis, then correct it with the deterministic layers."""
        explicit_language = language.detect_explicit_language_request(user_message)
        explicit_topic_switch = intents.is_explicit_topic_switch(user_message)
        inferred_turn_language = explicit_language or language.infer_text_language(user_message)
        preserve_unlocked_language = language.should_preserve_unlocked_session_language(
            session,
            user_message,
            inferred_turn_language,
        )
        llm_session_language = (
            session.language_preference
            if (session.language_locked and session.language_preference != "auto")
            or preserve_unlocked_language
            else inferred_turn_language
        )

        analysis = await self.ai.analyze_message(
            session=session,
            user_message=user_message,
            conversation_history=session_manager.get_conversation_history(
                session,
                include_assistant=bool(session.currently_asking),
            ),
            system_prompt=get_analysis_system_prompt(),
            session_language=llm_session_language,
        )

        extracted_fields = self._merge_extracted_fields(
            session,
            user_message,
            llm_fields=analysis.get("extracted_fields", {}),
        )
        detected_life_event = self._resolve_life_event(
            analysis.get("life_event"),
            user_message,
            explicit_topic_switch=explicit_topic_switch,
            current_life_event=session.user_profile.life_event,
        )
        resolved_scheme_id = self._resolve_scheme_id(
            session,
            user_message,
            llm_scheme_id=analysis.get("selected_scheme_id"),
            explicit_topic_switch=explicit_topic_switch,
        )
        action = self._resolve_action(
            session,
            user_message,
            llm_action=analysis.get("action"),
            resolved_scheme_id=resolved_scheme_id,
            explicit_topic_switch=explicit_topic_switch,
        )

        return TurnAnalysis(
            intent=analysis.get("intent", "unknown"),
            action=action,
            detected_life_event=detected_life_event,
            extracted_fields=extracted_fields,
            llm_response_text=analysis.get("response_text"),
            resolved_scheme_id=resolved_scheme_id,
            explicit_language=explicit_language,
            explicit_topic_switch=explicit_topic_switch,
            detected_language=language.normalize_language(
                analysis.get("language", llm_session_language)
            ),
            inferred_turn_language=inferred_turn_language,
            preserve_unlocked_language=preserve_unlocked_language,
        )

    def _merge_extracted_fields(
        self,
        session: Session,
        user_message: str,
        *,
        llm_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine LLM extraction with the rule-based and contextual layers.

        Rule-based values win over the LLM's, because they come straight from
        the user's wording; the contextual layer then catches a bare number
        that both of the other two missed.
        """
        rule_based_fields = profile_extractor.extract_by_patterns(
            user_message,
            current_field=session.currently_asking,
        )
        merged = turn_policy.sanitize_extracted_fields(
            user_message,
            {**llm_fields, **rule_based_fields},
            rule_based_fields,
        )

        currently_asking = session.currently_asking
        if currently_asking and currently_asking not in merged:
            contextual = turn_policy.contextual_field_value(currently_asking, user_message)
            if contextual:
                field_name, value = contextual
                merged[field_name] = value

        return merged

    def _resolve_life_event(
        self,
        llm_life_event: str | None,
        user_message: str,
        *,
        explicit_topic_switch: bool,
        current_life_event: str | None,
    ) -> str | None:
        """Settle the life event for this turn, falling back to keywords."""
        detected = llm_life_event
        if explicit_topic_switch:
            classified = life_event_classifier.classify_by_keywords(user_message)
            if classified and classified != current_life_event:
                return classified
        if not detected:
            detected = life_event_classifier.classify_by_keywords(user_message)
        return detected

    def _resolve_scheme_id(
        self,
        session: Session,
        user_message: str,
        *,
        llm_scheme_id: str | None,
        explicit_topic_switch: bool,
    ) -> str | None:
        """Decide which scheme, if any, this turn refers to."""
        if explicit_topic_switch:
            # The user is leaving this topic, so any scheme name still in the
            # message is context, not a selection.
            return None
        validated = scheme_reference.validated_selected_scheme_id(session, llm_scheme_id)
        return validated or scheme_reference.resolve_scheme_from_text(session, user_message)

    def _resolve_action(
        self,
        session: Session,
        user_message: str,
        *,
        llm_action: str | None,
        resolved_scheme_id: str | None,
        explicit_topic_switch: bool,
    ) -> str | None:
        """Settle the turn's action, with deterministic overrides winning."""
        action = (
            intents.detect_action_override(
                user_message,
                session.state,
                session.currently_asking,
                resolved_scheme_id,
                session.selected_scheme_id,
            )
            or llm_action
        )
        if explicit_topic_switch and turn_policy.should_preserve_scheme_context_action(action):
            return None

        has_scheme_context = bool(
            resolved_scheme_id or session.selected_scheme_id or session.presented_schemes
        )
        if turn_policy.should_answer_scheme_question(
            user_message,
            session.state,
            action,
            resolved_scheme_id,
            session.selected_scheme_id,
            has_scheme_context,
        ):
            return "answer_scheme_question"
        return action

    # ------------------------------------------------------------------
    # Phase 4 — language
    # ------------------------------------------------------------------

    def _resolve_language(
        self,
        session: Session,
        analysis: TurnAnalysis,
    ) -> tuple[Session, str, bool]:
        """Settle the reply language and record it on the session.

        An explicit request locks the language. Otherwise the session keeps
        following whatever the user last wrote, without locking.
        """
        if analysis.explicit_language:
            language_changed = session.language_preference != analysis.explicit_language
            session = session_manager.set_language(
                session,
                analysis.explicit_language,
                locked=True,
            )
            return session, analysis.explicit_language, language_changed

        if session.language_locked and session.language_preference != "auto":
            return session, session.language_preference, False

        preferred = (
            session.language_preference
            if analysis.preserve_unlocked_language and session.language_preference != "auto"
            else language.preferred_turn_language(
                analysis.inferred_turn_language,
                analysis.detected_language,
            )
        )
        previous_language = session.language_preference
        language_changed = False
        if preferred != previous_language or previous_language == "auto":
            session = session_manager.set_language(session, preferred, locked=False)
            language_changed = previous_language != preferred

        lang = (
            session.language_preference
            if session.language_preference != "auto"
            else preferred
        )
        return session, lang, language_changed

    async def _handle_turn_reset(
        self,
        session: Session,
        analysis: TurnAnalysis,
        *,
        user_message: str,
        lang: str,
        language_changed: bool,
    ) -> ChatResponse | None:
        """Handle the turns that end the conversation flow instead of advancing it."""
        # A language switch before the user has said anything substantive gets
        # a fresh greeting, so the conversation restarts in the new language
        # rather than continuing mid-question.
        if (
            language_changed
            and session.state == ConversationState.GREETING
            and not analysis.detected_life_event
            and not analysis.extracted_fields
        ):
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_generator.generate_greeting_response(lang),
                language=lang,
            )

        if analysis.action == "start_over":
            session = session_manager.reset_session(session, preserve_language=True)
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_generator.generate_greeting_response(lang),
                language=lang,
            )

        if analysis.intent == "goodbye" or analysis.action == "goodbye":
            # Farewell, not a greeting — the user said goodbye.
            response_text = response_generator.generate_farewell_response(lang)
            session = session_manager.reset_session(session, preserve_language=True)
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_text,
                language=lang,
            )

        return None

    # ------------------------------------------------------------------
    # Phase 5 — profile
    # ------------------------------------------------------------------

    def _apply_profile_updates(
        self,
        session: Session,
        analysis: TurnAnalysis,
        user_message: str,
    ) -> tuple[Session, ProfileUpdate]:
        """Merge extracted fields into the profile and clear stale scheme state."""
        before_profile = session.user_profile

        if analysis.extracted_fields:
            session = session_manager.update_profile(
                session,
                UserProfile(
                    **{k: v for k, v in analysis.extracted_fields.items() if v is not None}
                ),
            )

        if turn_policy.should_update_life_event(
            session,
            analysis.detected_life_event,
            analysis.extracted_fields,
            analysis.action,
            user_message,
        ):
            session = session_manager.update_profile(
                session,
                UserProfile(life_event=analysis.detected_life_event),
            )

        update = ProfileUpdate(
            before_profile=before_profile,
            changed_fields=turn_policy.matching_field_changes(
                before_profile, session.user_profile
            ),
        )
        if update.profile_changed:
            session = self._clear_stale_scheme_state(session, analysis, update)
        return session, update

    def _clear_stale_scheme_state(
        self,
        session: Session,
        analysis: TurnAnalysis,
        update: ProfileUpdate,
    ) -> Session:
        """Drop match results that the new profile facts have invalidated."""
        # New facts arrived, so the "no results, wait for more input" guard no
        # longer applies and previously skipped fields become askable again.
        session = session_manager.set_awaiting_profile_change(session, False)
        session = session_manager.set_skipped_fields(
            session,
            [f for f in session.skipped_fields if f not in update.changed_fields],
        )

        if "life_event" in update.changed_fields and update.before_profile.life_event is not None:
            # The topic itself changed: nothing about the old search survives.
            session = session_manager.clear_selection(session)
            session = session_manager.set_presented_schemes(session, [])
            session = session_manager.set_currently_asking(session, None)
            session = session_manager.set_skipped_fields(session, [])
        elif (
            update.matching_inputs_changed
            and session.selected_scheme_id
            and not turn_policy.should_preserve_scheme_context_action(analysis.action)
        ):
            session = session_manager.clear_selection(session)
            session = session_manager.set_presented_schemes(session, [])

        return session

    # ------------------------------------------------------------------
    # Phase 6 — state
    # ------------------------------------------------------------------

    def _decide_next_state(
        self,
        session: Session,
        analysis: TurnAnalysis,
        update: ProfileUpdate,
        user_message: str,
    ) -> tuple[ConversationState, ConversationState | None]:
        """Pick the next FSM state and the scheme view the user asked for."""
        profile = session.user_profile

        requested_state = turn_policy.requested_scheme_view(
            user_message,
            analysis.action,
            session.state,
            resolved_scheme_id=analysis.resolved_scheme_id,
            active_scheme_id=session.selected_scheme_id,
        )
        if analysis.explicit_topic_switch and (
            requested_state in scheme_reference.SCHEME_CONTEXT_STATES
        ):
            requested_state = None

        next_state = fsm.determine_next_state(
            current_state=session.state,
            profile=profile,
            intent=analysis.intent,
            selected_scheme_id=analysis.resolved_scheme_id,
            has_selected_scheme=bool(session.selected_scheme_id),
            action=analysis.action,
            requested_state=requested_state,
        )

        # Do not re-run a search that already returned nothing unless the user
        # has actually told us something new; otherwise the bot loops on
        # "no schemes found" instead of guiding them.
        if (
            next_state == ConversationState.SCHEME_MATCHING
            and session.awaiting_profile_change
            and not update.profile_changed
        ):
            next_state = turn_policy.collection_state_for_profile(profile)

        if turn_policy.should_refresh_matches_after_profile_change(
            session=session,
            profile=profile,
            matching_inputs_changed=update.matching_inputs_changed,
            action=analysis.action,
            requested_state=requested_state,
        ):
            next_state = ConversationState.SCHEME_MATCHING

        # "Yes" to a scheme follow-up means "show me how to apply".
        if (
            next_state == ConversationState.SCHEME_DETAILS
            and session.state in {
                ConversationState.SCHEME_DETAILS,
                ConversationState.DOCUMENT_GUIDANCE,
                ConversationState.REJECTION_WARNINGS,
            }
            and intents.is_affirmative(user_message)
        ):
            next_state = ConversationState.APPLICATION_HELP

        return next_state, requested_state

    # ------------------------------------------------------------------
    # Phase 7 — rendering
    # ------------------------------------------------------------------

    async def _render_state(
        self,
        *,
        session: Session,
        next_state: ConversationState,
        requested_state: ConversationState | None,
        analysis: TurnAnalysis,
        profile_update: ProfileUpdate,
        user_message: str,
        lang: str,
    ) -> RenderResult:
        """Produce the reply for the state this turn landed in."""
        profile = session.user_profile

        if next_state == ConversationState.GREETING:
            if session.state == ConversationState.CSC_HANDOFF:
                session = session_manager.reset_session(session)
            return RenderResult(
                session,
                next_state,
                response_generator.generate_greeting_response(lang),
            )

        if next_state == ConversationState.SITUATION_UNDERSTANDING:
            return await self._render_situation_understanding(
                session, analysis, user_message, lang
            )

        if next_state == ConversationState.PROFILE_COLLECTION:
            return await self._render_profile_collection(
                session, analysis, profile_update, user_message, lang
            )

        if next_state == ConversationState.SCHEME_MATCHING:
            outcome = await self._run_matching(profile, user_message, session, lang)
            session = outcome.session
            # Clear field tracking only when we actually reached presentation;
            # the clarification and no-match paths still need it.
            if outcome.state == ConversationState.SCHEME_PRESENTATION:
                session = session_manager.set_currently_asking(session, None)
            return RenderResult(
                session,
                outcome.state,
                outcome.text,
                outcome.schemes,
                outcome.inline_keyboard,
            )

        if next_state == ConversationState.SCHEME_PRESENTATION:
            return await self._render_scheme_presentation(
                session, analysis, requested_state, user_message, lang
            )

        if next_state in _SCHEME_VIEW_STATES:
            return await self._render_scheme_view(
                session, analysis, next_state, user_message, lang
            )

        if next_state == ConversationState.CSC_HANDOFF:
            # The LLM reply is more natural when it has one; the office list
            # is the fallback for when it does not.
            text = analysis.llm_response_text or await views.build_handoff_text(
                self.pool, profile, lang
            )
            return RenderResult(session, next_state, text)

        return RenderResult(
            session,
            next_state,
            response_generator.generate_greeting_response(lang),
        )

    async def _render_situation_understanding(
        self,
        session: Session,
        analysis: TurnAnalysis,
        user_message: str,
        lang: str,
    ) -> RenderResult:
        """Ask what the user needs, or move on once the topic is known."""
        profile = session.user_profile

        if not profile.life_event:
            text = analysis.llm_response_text or response_generator.generate_clarification_response(
                "life_event", lang
            )
            session = session_manager.set_currently_asking(session, "life_event")
            return RenderResult(session, ConversationState.SITUATION_UNDERSTANDING, text)

        if profile.is_complete_for_matching:
            outcome = await self._run_matching(profile, user_message, session, lang)
            return RenderResult(
                outcome.session,
                outcome.state,
                outcome.text,
                outcome.schemes,
                outcome.inline_keyboard,
            )

        text = (
            profile_extractor.get_next_question(profile, lang, session.skipped_fields)
            or response_generator.generate_clarification_response("age", lang)
        )
        session = session_manager.set_currently_asking(
            session,
            profile_extractor.get_next_missing_field(profile, session.skipped_fields),
        )
        return RenderResult(session, ConversationState.PROFILE_COLLECTION, text)

    async def _render_profile_collection(
        self,
        session: Session,
        analysis: TurnAnalysis,
        profile_update: ProfileUpdate,
        user_message: str,
        lang: str,
    ) -> RenderResult:
        """Collect the next profile field, handling skips and bad answers."""
        profile = session.user_profile

        if not profile.life_event:
            session = session_manager.set_currently_asking(session, "life_event")
            return RenderResult(
                session,
                ConversationState.SITUATION_UNDERSTANDING,
                response_generator.generate_clarification_response("life_event", lang),
            )

        previously_asking = session.currently_asking
        # The last search came back empty, the profile is as complete as it
        # gets, and this turn added nothing new — running matching again would
        # only produce the same empty result.
        no_new_matches_possible = (
            session.awaiting_profile_change
            and not profile_update.profile_changed
            and profile.is_complete_for_matching
        )

        if analysis.action in {"ask_field_reason", "clarify_field"} and previously_asking:
            text = (
                response_generator.generate_field_reason_response(previously_asking, lang)
                if analysis.action == "ask_field_reason"
                else response_generator.generate_field_help_response(previously_asking, lang)
            )
            session = session_manager.set_currently_asking(session, previously_asking)
            return RenderResult(session, ConversationState.PROFILE_COLLECTION, text)

        if analysis.action == "skip_field" and previously_asking:
            return await self._render_skipped_field(
                session,
                previously_asking,
                user_message,
                lang,
                no_new_matches_possible=no_new_matches_possible,
            )

        return await self._render_field_question(
            session,
            analysis,
            profile_update,
            user_message,
            lang,
            previously_asking=previously_asking,
            no_new_matches_possible=no_new_matches_possible,
        )

    async def _render_skipped_field(
        self,
        session: Session,
        skipped_field: str,
        user_message: str,
        lang: str,
        *,
        no_new_matches_possible: bool,
    ) -> RenderResult:
        """Record a skipped field and ask the next one, or match with what we have."""
        profile = session.user_profile
        skipped = list(session.skipped_fields)
        if skipped_field not in skipped:
            skipped.append(skipped_field)
        session = session_manager.set_skipped_fields(session, skipped)

        next_unskipped = profile_extractor.get_next_missing_field(profile, skipped)
        if next_unskipped:
            session = session_manager.set_currently_asking(session, next_unskipped)
            return RenderResult(
                session,
                ConversationState.PROFILE_COLLECTION,
                profile_extractor.get_next_question(profile, lang, skipped) or "",
            )

        if no_new_matches_possible:
            session = session_manager.set_currently_asking(session, None)
            return RenderResult(
                session,
                ConversationState.PROFILE_COLLECTION,
                response_generator.generate_no_schemes_response(lang),
            )

        outcome = await self._run_matching(profile, user_message, session, lang)
        session = session_manager.set_currently_asking(outcome.session, None)
        return RenderResult(
            session,
            outcome.state,
            outcome.text,
            outcome.schemes,
            outcome.inline_keyboard,
        )

    async def _render_field_question(
        self,
        session: Session,
        analysis: TurnAnalysis,
        profile_update: ProfileUpdate,
        user_message: str,
        lang: str,
        *,
        previously_asking: str | None,
        no_new_matches_possible: bool,
    ) -> RenderResult:
        """Normal collection turn: validate the answer, then ask the next field."""
        profile = session.user_profile
        next_question = profile_extractor.get_next_question(
            profile, lang, session.skipped_fields
        )
        next_field = profile_extractor.get_next_missing_field(profile, session.skipped_fields)
        scope_followup = intents.is_multi_beneficiary_scope_followup(
            user_message, previously_asking
        )

        validation_error = None
        if previously_asking and previously_asking not in analysis.extracted_fields:
            is_valid, error_type = profile_extractor.validate_field_response(
                previously_asking, user_message, analysis.extracted_fields
            )
            if not is_valid and error_type:
                validation_error = error_type

        # An explicit language switch mid-question is answered by re-asking
        # the same question in the new language, not by moving on.
        translated_reask = (
            analysis.explicit_language is not None
            and previously_asking is not None
            and not analysis.extracted_fields
            and not analysis.detected_life_event
        )

        state = ConversationState.PROFILE_COLLECTION
        schemes: list[SchemeMatch] = []
        inline_keyboard = None

        if validation_error:
            text = profile_extractor.get_validation_re_prompt(
                previously_asking, validation_error, lang
            )
        elif scope_followup:
            text = views.build_multi_beneficiary_scope_response(lang)
        elif translated_reask:
            text = profile_extractor.get_next_question(profile, lang, session.skipped_fields) or ""
        elif self._should_use_llm_reply(
            analysis,
            previously_asking=previously_asking,
            next_question=next_question,
            validation_error=validation_error,
            user_message=user_message,
            profile_changed=profile_update.profile_changed,
        ):
            text = analysis.llm_response_text
        elif next_question:
            text = next_question
        elif no_new_matches_possible:
            text = response_generator.generate_no_schemes_response(lang)
        else:
            # Every field is filled but the FSM did not route to matching.
            outcome = await self._run_matching(profile, user_message, session, lang)
            session, state, text = outcome.session, outcome.state, outcome.text
            schemes, inline_keyboard = outcome.schemes, outcome.inline_keyboard

        # Track the field being asked so the next turn can interpret a bare
        # answer in context.
        if validation_error or translated_reask:
            session = session_manager.set_currently_asking(session, previously_asking)
        elif scope_followup:
            session = session_manager.set_currently_asking(session, "life_event")
        else:
            session = session_manager.set_currently_asking(session, next_field)

        return RenderResult(session, state, text, schemes, inline_keyboard)

    def _should_use_llm_reply(
        self,
        analysis: TurnAnalysis,
        *,
        previously_asking: str | None,
        next_question: str | None,
        validation_error: str | None,
        user_message: str,
        profile_changed: bool,
    ) -> bool:
        """Decide whether the LLM's own wording can be used for this turn.

        Its reply is generated before the rule-based layers run, so it can be
        stale: re-asking for a field we just captured, wandering off the
        pending question, or flipping husband/wife relative to what the user
        said.
        """
        if not analysis.llm_response_text:
            return False

        captured_pending_field = (
            profile_changed
            and previously_asking
            and previously_asking in analysis.extracted_fields
            and next_question
        )
        if captured_pending_field:
            return False

        still_waiting_for_field = (
            previously_asking
            and previously_asking not in analysis.extracted_fields
            and analysis.action
            not in {
                "ask_field_reason",
                "clarify_field",
                "skip_field",
                "change_language",
                "start_over",
                "request_handoff",
            }
            and not validation_error
        )
        if still_waiting_for_field:
            return False

        return not language.response_conflicts_with_spouse_reference(
            user_message,
            analysis.llm_response_text,
        )

    async def _render_scheme_presentation(
        self,
        session: Session,
        analysis: TurnAnalysis,
        requested_state: ConversationState | None,
        user_message: str,
        lang: str,
    ) -> RenderResult:
        """Show the scheme list, or open the scheme the user just referenced."""
        profile = session.user_profile
        scheme_id = analysis.resolved_scheme_id or scheme_reference.default_scheme_from_session(
            session, requested_state
        )

        if scheme_id:
            session = session_manager.select_scheme(session, scheme_id)
            if analysis.action == "answer_scheme_question":
                return RenderResult(
                    session,
                    ConversationState.SCHEME_DETAILS,
                    await views.build_scheme_question_answer_text(
                        self.pool,
                        session,
                        scheme_id,
                        profile,
                        user_message,
                        lang,
                        active_view=ConversationState.SCHEME_DETAILS.value,
                    ),
                )

            view_state = (
                requested_state
                if requested_state in _SCHEME_VIEW_STATES
                else ConversationState.SCHEME_DETAILS
            )
            return RenderResult(
                session,
                view_state,
                await self._build_scheme_view_text(
                    session, view_state, scheme_id, profile, user_message, lang, action=None
                ),
            )

        schemes = await scheme_matcher.match_schemes(
            pool=self.pool,
            profile=profile,
            query_text=user_message,
        )
        inline_keyboard = None
        if schemes:
            session = scheme_reference.store_presented_schemes(session, schemes)
            inline_keyboard = format_inline_keyboard(schemes, lang)
        return RenderResult(
            session,
            ConversationState.SCHEME_PRESENTATION,
            response_generator.generate_scheme_selection_response(lang),
            schemes,
            inline_keyboard,
        )

    async def _render_scheme_view(
        self,
        session: Session,
        analysis: TurnAnalysis,
        view_state: ConversationState,
        user_message: str,
        lang: str,
    ) -> RenderResult:
        """Render one of the four single-scheme views.

        All four need a scheme id first, and fall back to the list when there
        is none. Naming a different scheme while in application help means the
        user changed their mind, so they get that scheme's overview instead of
        application steps for something they have not seen.
        """
        profile = session.user_profile
        scheme_id = analysis.resolved_scheme_id or scheme_reference.default_scheme_from_session(
            session, view_state
        )

        if not scheme_id:
            return RenderResult(
                session,
                ConversationState.SCHEME_PRESENTATION,
                views.build_select_scheme_first_text(lang),
            )

        switched_scheme = scheme_id != session.selected_scheme_id
        if switched_scheme:
            session = session_manager.select_scheme(session, scheme_id)

        if view_state == ConversationState.APPLICATION_HELP and switched_scheme:
            return RenderResult(
                session,
                ConversationState.SCHEME_DETAILS,
                await views.build_scheme_details_text(self.pool, scheme_id, profile, lang),
            )

        return RenderResult(
            session,
            view_state,
            await self._build_scheme_view_text(
                session,
                view_state,
                scheme_id,
                profile,
                user_message,
                lang,
                action=analysis.action,
            ),
        )

    async def _build_scheme_view_text(
        self,
        session: Session,
        view_state: ConversationState,
        scheme_id: str,
        profile: UserProfile,
        user_message: str,
        lang: str,
        *,
        action: str | None,
    ) -> str:
        """Dispatch to the renderer for one scheme view."""
        if action == "answer_scheme_question":
            return await views.build_scheme_question_answer_text(
                self.pool, session, scheme_id, profile, user_message, lang
            )
        if view_state == ConversationState.DOCUMENT_GUIDANCE:
            return await views.build_document_guidance_text(
                self.pool, session, scheme_id, lang
            )
        if view_state == ConversationState.REJECTION_WARNINGS:
            return await views.build_rejection_warnings_text(
                self.pool, scheme_id, profile, lang
            )
        if view_state == ConversationState.APPLICATION_HELP:
            return await views.build_application_help_text(
                self.pool, session, scheme_id, lang
            )
        return await views.build_scheme_details_text(self.pool, scheme_id, profile, lang)

    async def _render_state_snapshot(
        self,
        session: Session,
        lang: str,
    ) -> tuple[str, list[list[dict[str, str]]] | None]:
        """Render the user's current context in a chosen language."""
        profile = session.user_profile
        state = session.state

        if state == ConversationState.GREETING:
            return response_generator.generate_greeting_response(lang), None

        if state in {
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
        }:
            next_question = profile_extractor.get_next_question(
                profile,
                lang,
                session.skipped_fields,
            )
            if next_question:
                return next_question, None
            if state == ConversationState.SITUATION_UNDERSTANDING or not profile.life_event:
                return response_generator.generate_clarification_response("life_event", lang), None
            return response_generator.generate_help_response(lang), None

        if state == ConversationState.SCHEME_PRESENTATION:
            selection_text = views.build_presented_scheme_selection_text(
                session.presented_schemes,
                lang,
            )
            if selection_text:
                return selection_text, format_presented_scheme_keyboard(
                    session.presented_schemes,
                    lang,
                )
            return response_generator.generate_scheme_selection_response(lang), None

        if state in _SCHEME_VIEW_STATES and session.selected_scheme_id:
            return await self._build_scheme_view_text(
                session,
                state,
                session.selected_scheme_id,
                profile,
                "",
                lang,
                action=None,
            ), None

        if state == ConversationState.CSC_HANDOFF:
            return await views.build_handoff_text(self.pool, profile, lang), None

        return response_generator.generate_help_response(lang), None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    async def _run_matching(
        self,
        profile: UserProfile,
        user_message: str,
        session: Session,
        lang: str,
    ) -> RenderResult:
        """Run scheme matching and render the resulting list, or the no-match reply."""
        logger.info(
            "Running scheme matching for user=%s state=%s profile.life_event=%s age=%s category=%s income=%s",
            session.user_id,
            session.state.value,
            profile.life_event,
            profile.age,
            profile.category,
            profile.annual_income,
        )
        low_context_turn = turn_policy.is_low_context_matching_turn(session, user_message)
        matching_query_text = (
            turn_policy.build_matching_focus_text(profile, user_message)
            if low_context_turn
            else user_message
        )
        schemes = await scheme_matcher.match_schemes(
            pool=self.pool,
            profile=profile,
            query_text=matching_query_text,
        )

        if not schemes:
            return self._no_match_result(session, profile, lang)

        relevance = await self._judge_relevance(
            session, profile, schemes, user_message, lang, low_context_turn
        )
        schemes = relevance["matches"]

        if relevance["should_clarify"]:
            session = session_manager.set_awaiting_profile_change(session, False)
            session = session_manager.clear_selection(session)
            session = session_manager.set_presented_schemes(session, [])
            session = session_manager.set_currently_asking(session, "life_event")
            return RenderResult(
                session,
                ConversationState.SITUATION_UNDERSTANDING,
                relevance["clarification_question"],
            )

        session = session_manager.set_awaiting_profile_change(session, False)
        session = scheme_reference.store_presented_schemes(session, schemes)
        return RenderResult(
            session,
            ConversationState.SCHEME_PRESENTATION,
            views.build_scheme_list_text(schemes, profile, lang),
            schemes,
            format_inline_keyboard(schemes, lang),
        )

    async def _judge_relevance(
        self,
        session: Session,
        profile: UserProfile,
        schemes: list[SchemeMatch],
        user_message: str,
        lang: str,
        low_context_turn: bool,
    ) -> dict[str, Any]:
        """Run the AI relevance gate over deterministic candidates."""
        judgement = None
        if self.ai.should_run_relevance_judge(schemes):
            try:
                judgement = await self.ai.judge_scheme_relevance(
                    session=session,
                    user_message=turn_policy.build_matching_focus_text(profile, user_message),
                    conversation_history=session_manager.get_conversation_history(session),
                    candidate_schemes=scheme_relevance.build_candidate_payload(schemes),
                    session_language=lang,
                )
            except Exception as exc:
                # A failed judgement degrades to the deterministic ranking
                # rather than losing the user's results entirely.
                logger.warning(
                    "Scheme relevance judging failed for user=%s: %s",
                    session.user_id,
                    exc,
                )
        else:
            logger.info(
                "Skipping AI relevance judge for user=%s top_score=%.3f",
                session.user_id,
                schemes[0].deterministic_score,
            )

        relevance = scheme_relevance.apply_relevance_judgement(
            schemes,
            judgement,
            lang,
            profile.life_event,
        )
        if low_context_turn and relevance["should_clarify"]:
            # A bare "yes" or "50000" is not enough context for the judge to
            # ask a good clarifying question, so keep the results instead.
            logger.info(
                "Ignoring low-context relevance clarification for user=%s currently_asking=%s",
                session.user_id,
                session.currently_asking,
            )
            relevance["should_clarify"] = False
            relevance["clarification_question"] = None

        logger.info(
            "Scheme relevance gate: should_clarify=%s overall_confidence=%s top_ids=%s",
            relevance["should_clarify"],
            relevance["overall_confidence"],
            [match.scheme.id for match in relevance["matches"][:3]],
        )
        return relevance

    def _no_match_result(
        self,
        session: Session,
        profile: UserProfile,
        lang: str,
    ) -> RenderResult:
        """Return to collection so the user can adjust, not to a dead end."""
        session = session_manager.set_awaiting_profile_change(session, True)
        session = session_manager.clear_selection(session)
        session = session_manager.set_presented_schemes(session, [])
        return RenderResult(
            session,
            turn_policy.collection_state_for_profile(profile),
            response_generator.generate_no_schemes_response(lang),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _finalize_turn(
        self,
        *,
        session: Session,
        next_state: ConversationState,
        user_message: str,
        response_text: str,
        schemes: list[SchemeMatch],
        inline_keyboard: list[list[dict[str, str]]] | None,
        lang: str,
    ) -> ChatResponse:
        """Enforce the response language, persist the turn, and build the reply."""
        response_text = await response_generator.ensure_response_language(
            session,
            response_text,
            lang,
        )
        session = session_manager.update_state(session, next_state)
        await self._save_completed_turn(
            session,
            user_message=user_message,
            response_text=response_text,
        )

        return ChatResponse(
            text=response_text,
            schemes=schemes,
            inline_keyboard=inline_keyboard,
            next_state=next_state.value,
            language=lang,
        )

    async def _save_completed_turn(
        self,
        session: Session,
        *,
        user_message: str,
        response_text: str,
    ) -> Session:
        """Persist a completed user-assistant turn and enqueue memory refresh if due."""
        session = await session_manager.add_message(session, "user", user_message)
        session = await session_manager.add_message(session, "assistant", response_text)
        session = session_manager.mark_turn_completed(session)

        refresh_due = should_refresh_working_memory(
            session,
            trigger_turns=self.settings.ai_memory_refresh_turns,
            trigger_tokens=self.settings.ai_memory_refresh_token_threshold,
        )
        if refresh_due:
            session = session_manager.set_pending_memory_job(session, True)

        await session_manager.save_session(session)

        if not refresh_due:
            return session

        if await enqueue_memory_refresh(session.user_id, session.completed_turn_count):
            return session

        # The job never made it onto the queue, so clear the marker; leaving it
        # set would block every future refresh for this user.
        logger.warning(
            "Memory refresh queue unavailable for user=%s turn=%s",
            session.user_id,
            session.completed_turn_count,
        )
        session = session_manager.set_pending_memory_job(session, False)
        await session_manager.save_session(session)
        return session

    async def _build_command_response(
        self,
        session: Session,
        *,
        user_message: str,
        response_text: str,
        language: str,
        inline_keyboard: list[list[dict[str, str]]] | None = None,
    ) -> ChatResponse:
        """Persist a deterministic turn and return a response."""
        await self._save_completed_turn(
            session,
            user_message=user_message,
            response_text=response_text,
        )
        return ChatResponse(
            text=response_text,
            next_state=session.state.value,
            language=language,
            inline_keyboard=inline_keyboard,
        )
