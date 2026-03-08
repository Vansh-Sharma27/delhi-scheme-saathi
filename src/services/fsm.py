"""Finite State Machine for conversation flow."""

from src.models.session import ConversationState, Session, UserProfile


class FSMTransitionError(Exception):
    """Invalid state transition attempted."""


def get_valid_transitions(state: ConversationState) -> list[ConversationState]:
    """Get valid next states from current state."""
    transitions = {
        ConversationState.GREETING: [
            ConversationState.GREETING,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
        ],
        ConversationState.SITUATION_UNDERSTANDING: [
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
            ConversationState.SCHEME_MATCHING,
            ConversationState.GREETING,
        ],
        ConversationState.PROFILE_COLLECTION: [
            ConversationState.PROFILE_COLLECTION,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.SCHEME_MATCHING,
            ConversationState.GREETING,
        ],
        ConversationState.SCHEME_MATCHING: [
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
        ],
        ConversationState.SCHEME_PRESENTATION: [
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.SCHEME_DETAILS,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
            ConversationState.CSC_HANDOFF,
        ],
        ConversationState.SCHEME_DETAILS: [
            ConversationState.SCHEME_DETAILS,
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.APPLICATION_HELP,
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.CSC_HANDOFF,
        ],
        ConversationState.DOCUMENT_GUIDANCE: [
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.SCHEME_DETAILS,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.APPLICATION_HELP,
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.CSC_HANDOFF,
        ],
        ConversationState.REJECTION_WARNINGS: [
            ConversationState.REJECTION_WARNINGS,
            ConversationState.SCHEME_DETAILS,
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.APPLICATION_HELP,
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.CSC_HANDOFF,
        ],
        ConversationState.APPLICATION_HELP: [
            ConversationState.APPLICATION_HELP,
            ConversationState.SCHEME_DETAILS,
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.CSC_HANDOFF,
            ConversationState.GREETING,
        ],
        ConversationState.CSC_HANDOFF: [
            ConversationState.CSC_HANDOFF,
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.SCHEME_DETAILS,
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.APPLICATION_HELP,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
            ConversationState.GREETING,
        ],
    }
    return transitions.get(state, [])


def can_transition(current: ConversationState, target: ConversationState) -> bool:
    """Check if transition from current to target is valid."""
    return target in get_valid_transitions(current)


def transition(session: Session, target: ConversationState) -> Session:
    """Transition session to new state."""
    if not can_transition(session.state, target):
        raise FSMTransitionError(f"Invalid transition from {session.state} to {target}")
    return session.with_state(target)


def should_auto_match(profile: UserProfile) -> bool:
    """Check if profile has enough info to trigger matching."""
    return profile.is_complete_for_matching


def determine_next_state(
    current_state: ConversationState,
    profile: UserProfile,
    intent: str,
    has_schemes: bool | None = None,
    selected_scheme_id: str | None = None,
    has_selected_scheme: bool = False,
    action: str | None = None,
    requested_state: ConversationState | None = None,
) -> ConversationState:
    """Determine the next state from the current flow inputs."""
    if intent == "goodbye":
        return ConversationState.GREETING

    if requested_state and can_transition(current_state, requested_state):
        return requested_state

    match current_state:
        case ConversationState.GREETING:
            if intent == "greeting":
                return ConversationState.GREETING
            if profile.life_event:
                if should_auto_match(profile):
                    return ConversationState.SCHEME_MATCHING
                return ConversationState.PROFILE_COLLECTION
            return ConversationState.SITUATION_UNDERSTANDING

        case ConversationState.SITUATION_UNDERSTANDING:
            if not profile.life_event:
                return ConversationState.SITUATION_UNDERSTANDING
            if should_auto_match(profile):
                return ConversationState.SCHEME_MATCHING
            return ConversationState.PROFILE_COLLECTION

        case ConversationState.PROFILE_COLLECTION:
            if not profile.life_event:
                return ConversationState.SITUATION_UNDERSTANDING
            if should_auto_match(profile):
                return ConversationState.SCHEME_MATCHING
            return ConversationState.PROFILE_COLLECTION

        case ConversationState.SCHEME_MATCHING:
            if has_schemes is None:
                return ConversationState.SCHEME_MATCHING
            if has_schemes:
                return ConversationState.SCHEME_PRESENTATION
            if profile.life_event:
                return ConversationState.PROFILE_COLLECTION
            return ConversationState.SITUATION_UNDERSTANDING

        case ConversationState.SCHEME_PRESENTATION:
            if selected_scheme_id:
                return ConversationState.SCHEME_DETAILS
            if action == "request_handoff":
                return ConversationState.CSC_HANDOFF
            if action == "request_details" and has_selected_scheme:
                return ConversationState.SCHEME_DETAILS
            if intent == "clarification":
                return ConversationState.PROFILE_COLLECTION
            return ConversationState.SCHEME_PRESENTATION

        case ConversationState.SCHEME_DETAILS:
            if action == "request_application":
                return ConversationState.APPLICATION_HELP
            if action == "request_handoff":
                return ConversationState.CSC_HANDOFF
            if selected_scheme_id or action in {"request_details", "switch_scheme"}:
                return ConversationState.SCHEME_DETAILS
            return ConversationState.SCHEME_DETAILS

        case ConversationState.DOCUMENT_GUIDANCE:
            if action == "request_application":
                return ConversationState.APPLICATION_HELP
            if action == "request_handoff":
                return ConversationState.CSC_HANDOFF
            if selected_scheme_id:
                return ConversationState.SCHEME_DETAILS
            return ConversationState.DOCUMENT_GUIDANCE

        case ConversationState.REJECTION_WARNINGS:
            if action == "request_application":
                return ConversationState.APPLICATION_HELP
            if action == "request_handoff":
                return ConversationState.CSC_HANDOFF
            if selected_scheme_id:
                return ConversationState.SCHEME_DETAILS
            return ConversationState.REJECTION_WARNINGS

        case ConversationState.APPLICATION_HELP:
            if action == "request_handoff":
                return ConversationState.CSC_HANDOFF
            if selected_scheme_id:
                return ConversationState.SCHEME_DETAILS
            return ConversationState.APPLICATION_HELP

        case ConversationState.CSC_HANDOFF:
            if selected_scheme_id:
                return ConversationState.SCHEME_DETAILS
            if profile.life_event and should_auto_match(profile):
                return ConversationState.SCHEME_PRESENTATION
            if profile.life_event:
                return ConversationState.PROFILE_COLLECTION
            return ConversationState.SITUATION_UNDERSTANDING

    return current_state


def get_state_prompt_context(state: ConversationState) -> dict[str, str]:
    """Get context-specific prompting hints for each state."""
    contexts = {
        ConversationState.GREETING: {
            "goal": "Welcome the user and establish the conversation",
            "actions": ["greet warmly", "explain capabilities", "ask about their situation"],
        },
        ConversationState.SITUATION_UNDERSTANDING: {
            "goal": "Understand the user's life event or type of help needed",
            "actions": ["identify the main need", "clarify ambiguous topics", "confirm understanding"],
        },
        ConversationState.PROFILE_COLLECTION: {
            "goal": "Collect the minimum profile needed for eligibility matching",
            "actions": ["ask for missing fields", "handle corrections", "avoid re-asking known details"],
        },
        ConversationState.SCHEME_MATCHING: {
            "goal": "Run deterministic and semantic scheme matching",
            "actions": ["filter by life event", "apply eligibility constraints", "rank candidates"],
        },
        ConversationState.SCHEME_PRESENTATION: {
            "goal": "Present the best matched schemes",
            "actions": ["show scheme options", "highlight why they fit", "ask which one to open"],
        },
        ConversationState.SCHEME_DETAILS: {
            "goal": "Explain the selected scheme and why it was suggested",
            "actions": ["summarize benefits", "explain fit", "offer next-step navigation"],
        },
        ConversationState.DOCUMENT_GUIDANCE: {
            "goal": "Guide the user through required documents",
            "actions": ["list documents", "explain where to get them", "mention fees and timing"],
        },
        ConversationState.REJECTION_WARNINGS: {
            "goal": "Warn the user about common rejection reasons",
            "actions": ["surface major rejection risks", "give prevention tips", "keep advice actionable"],
        },
        ConversationState.APPLICATION_HELP: {
            "goal": "Walk the user through the application process",
            "actions": ["show online/offline steps", "share links", "suggest document and rejection checks"],
        },
        ConversationState.CSC_HANDOFF: {
            "goal": "Hand the user off to human support when needed",
            "actions": ["share nearby CSC details", "explain when to visit", "allow return to scheme flow"],
        },
    }
    return contexts.get(state, {"goal": "Unknown", "actions": []})
