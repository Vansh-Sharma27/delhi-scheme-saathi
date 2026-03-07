"""Finite State Machine for conversation flow.

Simplified 7-state model for MVP:
- GREETING: Welcome and introduction
- UNDERSTANDING: Discover life event + collect profile (collapsed from 2 states)
- MATCHING: Transient state for running scheme retrieval
- PRESENTING: Show matched schemes
- DETAILS: Deep dive into scheme + docs + rejections (collapsed from 3 states)
- APPLICATION: Step-by-step application guidance
- HANDOFF: Connect to human help at CSC
"""

from src.models.session import ConversationState, Session, UserProfile


class FSMTransitionError(Exception):
    """Invalid state transition attempted."""
    pass


def get_valid_transitions(state: ConversationState) -> list[ConversationState]:
    """Get valid next states from current state."""
    transitions = {
        ConversationState.GREETING: [
            ConversationState.GREETING,  # Stay for greeting response
            ConversationState.UNDERSTANDING,
        ],
        ConversationState.UNDERSTANDING: [
            ConversationState.MATCHING,
            ConversationState.UNDERSTANDING,  # Stay for more info
        ],
        ConversationState.MATCHING: [
            ConversationState.PRESENTING,  # Schemes found
            ConversationState.HANDOFF,  # No schemes found
            ConversationState.UNDERSTANDING,  # Need more info
        ],
        ConversationState.PRESENTING: [
            ConversationState.DETAILS,  # User selects scheme
            ConversationState.UNDERSTANDING,  # User refines criteria
            ConversationState.HANDOFF,  # User needs help
        ],
        ConversationState.DETAILS: [
            ConversationState.APPLICATION,  # User ready to apply
            ConversationState.PRESENTING,  # Back to list
            ConversationState.HANDOFF,  # Need human help
        ],
        ConversationState.APPLICATION: [
            ConversationState.HANDOFF,  # Need CSC help
            ConversationState.DETAILS,  # Back to details
            ConversationState.GREETING,  # Start over
        ],
        ConversationState.HANDOFF: [
            ConversationState.GREETING,  # Start over
            ConversationState.PRESENTING,  # Back to schemes
            ConversationState.UNDERSTANDING,  # User asks a new question
        ],
    }
    return transitions.get(state, [])


def can_transition(current: ConversationState, target: ConversationState) -> bool:
    """Check if transition from current to target is valid."""
    return target in get_valid_transitions(current)


def transition(session: Session, target: ConversationState) -> Session:
    """Transition session to new state.

    Returns new session with updated state (immutable).
    Raises FSMTransitionError if transition is invalid.
    """
    if not can_transition(session.state, target):
        raise FSMTransitionError(
            f"Invalid transition from {session.state} to {target}"
        )
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
) -> ConversationState:
    """Determine the next state based on current state, profile, and intent.

    This encapsulates the FSM logic for automatic transitions.
    """
    if intent == "goodbye":
        return ConversationState.GREETING

    match current_state:
        case ConversationState.GREETING:
            if intent == "greeting":
                return ConversationState.GREETING
            return ConversationState.UNDERSTANDING

        case ConversationState.UNDERSTANDING:
            if should_auto_match(profile):
                return ConversationState.MATCHING
            return ConversationState.UNDERSTANDING

        case ConversationState.MATCHING:
            # If matching has not run yet in this turn, stay in MATCHING.
            if has_schemes is None:
                return ConversationState.MATCHING
            if has_schemes:
                return ConversationState.PRESENTING
            return ConversationState.HANDOFF

        case ConversationState.PRESENTING:
            if selected_scheme_id:
                return ConversationState.DETAILS
            if action == "request_handoff":
                return ConversationState.HANDOFF
            if intent == "clarification":
                return ConversationState.UNDERSTANDING
            return ConversationState.PRESENTING

        case ConversationState.DETAILS:
            if action == "request_application":
                return ConversationState.APPLICATION
            if action == "request_handoff":
                return ConversationState.HANDOFF
            if intent == "selection":
                if selected_scheme_id is None and not has_selected_scheme:
                    return ConversationState.PRESENTING
                return ConversationState.DETAILS
            if action == "request_details":
                return ConversationState.DETAILS
            if intent in ["question", "clarification"]:
                return ConversationState.DETAILS
            return ConversationState.DETAILS

        case ConversationState.APPLICATION:
            if action == "request_handoff":
                return ConversationState.HANDOFF
            if selected_scheme_id:
                return ConversationState.DETAILS
            if action == "request_details":
                return ConversationState.DETAILS
            return ConversationState.APPLICATION

        case ConversationState.HANDOFF:
            if intent == "greeting":
                return ConversationState.GREETING
            if selected_scheme_id:
                return ConversationState.DETAILS
            if intent in ("question", "clarification", "selection"):
                return ConversationState.UNDERSTANDING
            return ConversationState.HANDOFF

    return current_state


def get_state_prompt_context(state: ConversationState) -> dict[str, str]:
    """Get context-specific prompting hints for each state."""
    contexts = {
        ConversationState.GREETING: {
            "goal": "Welcome user and identify their need",
            "actions": ["greet warmly", "explain capabilities", "ask about situation"],
        },
        ConversationState.UNDERSTANDING: {
            "goal": "Understand life event and collect profile",
            "actions": ["identify life event", "ask for age/income/category", "show empathy"],
        },
        ConversationState.MATCHING: {
            "goal": "Run scheme matching",
            "actions": ["search database", "filter by eligibility", "rank results"],
        },
        ConversationState.PRESENTING: {
            "goal": "Present matching schemes",
            "actions": ["show scheme cards", "highlight benefits", "explain eligibility"],
        },
        ConversationState.DETAILS: {
            "goal": "Deep dive into selected scheme",
            "actions": ["show full details", "list documents", "warn about rejections"],
        },
        ConversationState.APPLICATION: {
            "goal": "Guide through application",
            "actions": ["step-by-step process", "online/offline options", "timeline"],
        },
        ConversationState.HANDOFF: {
            "goal": "Connect to human help",
            "actions": ["show nearest CSC", "provide contact", "explain services"],
        },
    }
    return contexts.get(state, {"goal": "Unknown", "actions": []})
