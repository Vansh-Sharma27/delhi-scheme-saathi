"""Helpers for compact conversation memory and refresh decisions."""

from __future__ import annotations

from src.models.session import ConversationMemory, Session


def _format_income(income: int) -> str:
    """Format income for compact memory strings."""
    if income >= 100000:
        lakhs = income / 100000
        return f"₹{lakhs:.1f} lakh" if lakhs != int(lakhs) else f"₹{int(lakhs)} lakh"
    return f"₹{income:,}"


def build_profile_facts(session: Session) -> list[str]:
    """Build deterministic facts to carry across long conversations."""
    profile = session.user_profile
    facts: list[str] = []

    if profile.life_event:
        facts.append(f"Need area: {profile.life_event}")
    if profile.age is not None:
        facts.append(f"Age: {profile.age}")
    if profile.gender:
        facts.append(f"Gender: {profile.gender}")
    if profile.category:
        facts.append(f"Category: {profile.category}")
    if profile.marital_status:
        facts.append(f"Marital status: {profile.marital_status}")
    if profile.annual_income is not None:
        facts.append(f"Annual income: {_format_income(profile.annual_income)}")
    if session.currently_asking:
        facts.append(f"Pending field: {session.currently_asking}")

    return facts


def build_working_memory(
    session: Session,
    summary_text: str | None,
) -> ConversationMemory:
    """Construct working memory from deterministic state and LLM summary."""
    presented_scheme_ids = [
        str(item["id"])
        for item in session.presented_schemes[:3]
        if isinstance(item, dict) and item.get("id")
    ]

    return ConversationMemory(
        summary=(summary_text or session.working_memory.summary or None),
        profile_facts=build_profile_facts(session),
        active_scheme_ids=list(dict.fromkeys(
            [
                *presented_scheme_ids,
                *( [session.selected_scheme_id] if session.selected_scheme_id else [] ),
                *session.discussed_schemes[-3:],
            ]
        )),
        pending_action=session.currently_asking or session.state.value,
        last_user_goal=session.user_profile.life_event,
    )


def working_memory_payload(session: Session) -> dict[str, object] | None:
    """Return prompt-ready memory payload when it has useful content."""
    memory = session.working_memory
    if not memory.is_meaningful():
        return None
    return memory.model_dump(exclude_none=True)


def estimate_context_tokens(session: Session) -> int:
    """Estimate token usage for recent context and memory using a 4-char heuristic."""
    total_chars = sum(len(message.content) for message in session.messages)
    total_chars += len(session.working_memory.summary or "")
    total_chars += sum(len(fact) for fact in session.working_memory.profile_facts)
    return max(total_chars // 4, 0)


def should_refresh_working_memory(
    session: Session,
    *,
    trigger_turns: int,
    trigger_tokens: int,
) -> bool:
    """Return True when the session should enqueue a memory refresh."""
    if session.pending_memory_job:
        return False
    if session.completed_turn_count == 0:
        return False

    turns_since_refresh = session.completed_turn_count - session.last_memory_refresh_turn
    if turns_since_refresh >= trigger_turns:
        return True

    return estimate_context_tokens(session) >= trigger_tokens
