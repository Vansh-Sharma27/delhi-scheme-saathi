"""Decisions that need session and profile state, not just the message.

These are the rules that keep a turn from doing something surprising:
not re-running matching when nothing relevant changed, not abandoning an
open scheme because the user asked a follow-up question, and not letting a
newly detected life event overwrite the topic mid-collection.
"""

import re
from typing import Any

from src.models.session import ConversationState, Session, UserProfile
from src.services.conversation import intents, scheme_reference

# Profile fields that feed the SQL eligibility filter or the semantic query.
# A change to any of them invalidates the previous match results.
MATCH_RELEVANT_FIELDS = {"life_event", "age", "category", "annual_income", "gender"}

# Actions that mean "keep working with the scheme that is already open",
# so profile edits in the same turn must not clear the selection.
SCHEME_CONTEXT_ACTIONS = frozenset(
    {
        "answer_scheme_question",
        "request_details",
        "request_application",
        "request_handoff",
        "select_scheme",
        "switch_scheme",
    }
)

# Actions with an explicit destination of their own; a follow-up question
# should not be read into them.
_NON_QUESTION_ACTIONS = frozenset(
    {
        "start_over",
        "goodbye",
        "skip_field",
        "ask_field_reason",
        "clarify_field",
        "request_application",
        "request_handoff",
        "select_scheme",
        "switch_scheme",
    }
)

_FIELD_ANSWER_ACTIONS = frozenset({"answer_field", "skip_field", "ask_field_reason"})

_COLLECTION_STATES = {
    ConversationState.GREETING,
    ConversationState.SITUATION_UNDERSTANDING,
    ConversationState.PROFILE_COLLECTION,
}

# Wording that marks the speaker as a guardian acting for someone else, so
# "my daughter" must not be read as evidence that the speaker is married.
_GUARDIAN_MARKERS = (
    "mother",
    "father",
    "maa",
    "mom",
    "mummy",
    "parent",
    "uski maa",
    "uski mother",
    "meri beti",
    "my daughter",
    "my son",
    "mera beta",
)
_BENEFICIARY_MARKERS = (
    "beti",
    "beta",
    "daughter",
    "son",
    "child",
    "applicant",
)
_EXPLICIT_MARITAL_MARKERS = (
    "married",
    "शादीशुदा",
    "विवाहित",
)


def collection_state_for_profile(profile: UserProfile) -> ConversationState:
    """Return the active collection state based on whether the topic is known."""
    if profile.life_event:
        return ConversationState.PROFILE_COLLECTION
    return ConversationState.SITUATION_UNDERSTANDING


def should_preserve_scheme_context_action(action: str | None) -> bool:
    """Return True when an explicit scheme-flow action should keep the active scheme."""
    return action in SCHEME_CONTEXT_ACTIONS


def matching_field_changes(
    before_profile: UserProfile,
    after_profile: UserProfile,
) -> set[str]:
    """Return search-relevant profile fields that changed value."""
    return {
        field
        for field in MATCH_RELEVANT_FIELDS
        if getattr(before_profile, field) != getattr(after_profile, field)
    }


def should_refresh_matches_after_profile_change(
    *,
    session: Session,
    profile: UserProfile,
    matching_inputs_changed: bool,
    action: str | None,
    requested_state: ConversationState | None,
) -> bool:
    """Decide when updated profile facts should trigger a fresh scheme match."""
    if not matching_inputs_changed or not profile.is_complete_for_matching:
        return False
    if session.state not in scheme_reference.SCHEME_CONTEXT_STATES:
        return False
    if requested_state in scheme_reference.SCHEME_CONTEXT_STATES | {
        ConversationState.CSC_HANDOFF
    }:
        return False
    return not should_preserve_scheme_context_action(action)


def should_answer_scheme_question(
    text: str,
    current_state: ConversationState,
    action: str | None,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
    has_scheme_context: bool,
) -> bool:
    """Detect scheme follow-up questions that deserve an answer, not a card replay."""
    if current_state not in scheme_reference.SCHEME_CONTEXT_STATES:
        return False
    if not has_scheme_context:
        return False
    if resolved_scheme_id and not scheme_reference.resolved_scheme_matches_active_scheme(
        current_state,
        resolved_scheme_id,
        active_scheme_id,
    ):
        return False
    if (
        current_state in {
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.APPLICATION_HELP,
        }
        and intents.looks_like_scheme_question(text)
        and not intents.is_navigation_only_scheme_followup(text)
    ):
        return True
    if action in _NON_QUESTION_ACTIONS:
        return False
    if intents.wants_scheme_list_again(text):
        return False
    if intents.matches_any_pattern(text, intents.DOCUMENT_REQUEST_PATTERNS):
        return False
    if intents.matches_any_pattern(text, intents.REJECTION_REQUEST_PATTERNS):
        return False
    return intents.looks_like_scheme_question(text)


def requested_scheme_view(
    text: str,
    action: str | None,
    current_state: ConversationState,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
) -> ConversationState | None:
    """Resolve scheme-area navigation within the 10-state FSM."""
    same_active_scheme = scheme_reference.resolved_scheme_matches_active_scheme(
        current_state,
        resolved_scheme_id,
        active_scheme_id,
    )
    if intents.wants_scheme_list_again(text):
        return ConversationState.SCHEME_PRESENTATION
    if action == "request_handoff":
        return ConversationState.CSC_HANDOFF
    # Checked before the keyword patterns below: an analytical question keeps
    # the user where they are instead of jumping to whichever view a stray
    # keyword happens to name.
    if action == "answer_scheme_question":
        return (
            current_state
            if current_state in scheme_reference.SCHEME_CONTEXT_STATES
            else ConversationState.SCHEME_DETAILS
        )
    if action == "request_application" or intents.matches_any_pattern(
        text, intents.APPLICATION_REQUEST_PATTERNS
    ):
        return ConversationState.APPLICATION_HELP
    if intents.matches_any_pattern(text, intents.DOCUMENT_REQUEST_PATTERNS):
        return ConversationState.DOCUMENT_GUIDANCE
    if intents.matches_any_pattern(text, intents.REJECTION_REQUEST_PATTERNS):
        return ConversationState.REJECTION_WARNINGS
    if action == "request_details" or intents.matches_any_pattern(
        text, intents.JUSTIFICATION_PATTERNS
    ):
        return ConversationState.SCHEME_DETAILS
    if action in {"select_scheme", "switch_scheme"} or (
        resolved_scheme_id and not same_active_scheme
    ):
        return ConversationState.SCHEME_DETAILS
    return None


def sanitize_extracted_fields(
    user_message: str,
    extracted_fields: dict[str, Any],
    rule_based_fields: dict[str, Any],
) -> dict[str, Any]:
    """Drop relationship over-inference that is not directly supported by the text.

    A parent asking about a scholarship for their daughter is not stating
    their own marital status, but the LLM tends to infer "married" from the
    mention of a child. The rule-based extractor is trusted, so a value it
    produced is never dropped.
    """
    sanitized = dict(extracted_fields)
    text_lower = user_message.lower()

    if (
        sanitized.get("marital_status") == "married"
        and "marital_status" not in rule_based_fields
        and any(marker in text_lower for marker in _GUARDIAN_MARKERS)
        and any(marker in text_lower for marker in _BENEFICIARY_MARKERS)
        and not any(marker in text_lower for marker in _EXPLICIT_MARITAL_MARKERS)
    ):
        sanitized.pop("marital_status", None)

    return sanitized


def should_update_life_event(
    session: Session,
    detected_life_event: str | None,
    extracted_fields: dict[str, Any],
    action: str | None,
    user_message: str,
) -> bool:
    """Decide whether the current turn is allowed to replace the active topic."""
    current_life_event = session.user_profile.life_event
    if not detected_life_event or detected_life_event == current_life_event:
        return False
    if current_life_event is None:
        return True
    if session.currently_asking == "life_event":
        return True
    if intents.is_explicit_topic_switch(user_message):
        return True

    # Mid scheme flow the topic is settled; a scheme keyword that happens to
    # classify as another life event must not restart the search.
    if session.state in scheme_reference.SCHEME_CONTEXT_STATES | {
        ConversationState.CSC_HANDOFF
    } and (
        action in SCHEME_CONTEXT_ACTIONS
        or session.selected_scheme_id
        or session.presented_schemes
    ):
        return False

    # While collecting a specific field, treat this turn as a field answer
    # unless the user explicitly changed topic.
    if session.currently_asking and session.currently_asking != "life_event":
        if session.currently_asking in extracted_fields:
            return False
        if extracted_fields:
            return False
        if action in _FIELD_ANSWER_ACTIONS:
            return False

    return (
        session.state in _COLLECTION_STATES
        and session.currently_asking is None
        and not extracted_fields
    )


def is_low_context_matching_turn(session: Session, user_message: str) -> bool:
    """Return True when matching was triggered by a field reply or short confirmation.

    These turns usually contain bare values like "5 lakhs" or confirmations like
    "yes", so the active profile is a better signal than the raw message text.
    The AI relevance gate is also more likely to over-clarify on these inputs.
    """
    if session.currently_asking is not None:
        return True
    return bool(session.user_profile.life_event and intents.is_affirmative(user_message))


def build_matching_focus_text(profile: UserProfile, user_message: str) -> str:
    """Build a stable intent summary for AI relevance judging.

    The judge should evaluate the active scheme search goal, not only the latest
    collection turn such as a bare income answer.
    """
    focus_parts = []
    if profile.life_event:
        focus_parts.append(f"Need area: {profile.life_event}")
    if profile.marital_status:
        focus_parts.append(f"Marital status: {profile.marital_status}")
    if profile.gender:
        focus_parts.append(f"Gender: {profile.gender}")
    if profile.age is not None:
        focus_parts.append(f"Age: {profile.age}")
    if profile.category:
        focus_parts.append(f"Category: {profile.category}")
    if profile.annual_income is not None:
        focus_parts.append(f"Annual income: ₹{profile.annual_income}")
    if user_message.strip():
        focus_parts.append(f"Latest reply: {user_message.strip()}")
    return " | ".join(focus_parts) if focus_parts else user_message


def contextual_field_value(currently_asking: str | None, user_message: str) -> tuple[str, int] | None:
    """Interpret a bare number as the answer to the field being asked.

    The deterministic guardrail for the case where both the LLM and the
    regex extractor missed an obvious contextual answer such as "19" to
    "how old are you?".
    """
    if not currently_asking:
        return None

    bare_match = re.fullmatch(r"\s*(\d+)\s*", user_message.strip())
    if not bare_match:
        return None

    value = int(bare_match.group(1))
    if currently_asking == "age" and 1 <= value <= 120:
        return "age", value
    if currently_asking == "annual_income" and value > 0:
        return "annual_income", value
    return None
