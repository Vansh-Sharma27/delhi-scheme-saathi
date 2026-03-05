"""Session management service."""

import logging
from datetime import datetime

from src.db.session_store import get_session_store
from src.integrations.llm_client import get_llm_client
from src.models.session import ConversationState, Message, Session, UserProfile

logger = logging.getLogger(__name__)

# Summarize conversation every N messages
SUMMARY_INTERVAL = 5


async def get_or_create_session(user_id: str) -> Session:
    """Get existing session or create new one."""
    store = get_session_store()
    session = await store.get(user_id)

    if session is None:
        session = Session(
            user_id=user_id,
            state=ConversationState.GREETING,
            user_profile=UserProfile(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await store.save(session)
        logger.info(f"Created new session for user {user_id}")

    return session


async def save_session(session: Session) -> None:
    """Save session to store."""
    store = get_session_store()
    await store.save(session)


async def delete_session(user_id: str) -> None:
    """Delete session."""
    store = get_session_store()
    await store.delete(user_id)


async def add_message(
    session: Session,
    role: str,
    content: str,
) -> Session:
    """Add message to session and potentially summarize."""
    new_session = session.add_message(role, content)

    # Check if we should summarize
    if len(new_session.messages) >= SUMMARY_INTERVAL:
        message_count_since_summary = len(new_session.messages)
        if message_count_since_summary % SUMMARY_INTERVAL == 0:
            new_session = await update_summary(new_session)

    return new_session


async def update_summary(session: Session) -> Session:
    """Update conversation summary."""
    if not session.messages:
        return session

    try:
        llm = get_llm_client()
        messages_dicts = [
            {"role": m.role, "content": m.content}
            for m in session.messages
        ]

        new_summary = await llm.summarize_conversation(
            messages=messages_dicts,
            current_summary=session.conversation_summary,
        )

        return Session(
            user_id=session.user_id,
            state=session.state,
            user_profile=session.user_profile,
            messages=session.messages,
            conversation_summary=new_summary,
            discussed_schemes=session.discussed_schemes,
            selected_scheme_id=session.selected_scheme_id,
            language_preference=session.language_preference,
            created_at=session.created_at,
            updated_at=datetime.utcnow(),
            metadata=session.metadata,
        )

    except Exception as e:
        logger.error(f"Failed to update summary: {e}")
        return session


def update_profile(session: Session, new_profile: UserProfile) -> Session:
    """Update session with new profile (immutable merge)."""
    merged_profile = session.user_profile.merge_with(new_profile)
    return session.with_profile(merged_profile)


def update_state(session: Session, new_state: ConversationState) -> Session:
    """Update session state."""
    return session.with_state(new_state)


def add_discussed_scheme(session: Session, scheme_id: str) -> Session:
    """Add scheme to discussed list."""
    if scheme_id not in session.discussed_schemes:
        discussed = list(session.discussed_schemes)
        discussed.append(scheme_id)
        return Session(
            user_id=session.user_id,
            state=session.state,
            user_profile=session.user_profile,
            messages=session.messages,
            conversation_summary=session.conversation_summary,
            discussed_schemes=discussed,
            selected_scheme_id=session.selected_scheme_id,
            language_preference=session.language_preference,
            created_at=session.created_at,
            updated_at=datetime.utcnow(),
            metadata=session.metadata,
        )
    return session


def select_scheme(session: Session, scheme_id: str) -> Session:
    """Set selected scheme and add to discussed."""
    session = add_discussed_scheme(session, scheme_id)
    return Session(
        user_id=session.user_id,
        state=session.state,
        user_profile=session.user_profile,
        messages=session.messages,
        conversation_summary=session.conversation_summary,
        discussed_schemes=session.discussed_schemes,
        selected_scheme_id=scheme_id,
        language_preference=session.language_preference,
        created_at=session.created_at,
        updated_at=datetime.utcnow(),
        metadata=session.metadata,
    )


def set_language(session: Session, language: str) -> Session:
    """Set language preference."""
    return Session(
        user_id=session.user_id,
        state=session.state,
        user_profile=session.user_profile,
        messages=session.messages,
        conversation_summary=session.conversation_summary,
        discussed_schemes=session.discussed_schemes,
        selected_scheme_id=session.selected_scheme_id,
        language_preference=language,
        created_at=session.created_at,
        updated_at=datetime.utcnow(),
        metadata=session.metadata,
    )


def get_conversation_history(
    session: Session,
    include_assistant: bool = True,
) -> list[dict[str, str]]:
    """Get conversation history as list of dicts."""
    messages = session.messages
    if not include_assistant:
        messages = [m for m in messages if m.role == "user"]

    return [{"role": m.role, "content": m.content} for m in messages]


def reset_session(session: Session) -> Session:
    """Reset session to initial state (keep user_id)."""
    return Session(
        user_id=session.user_id,
        state=ConversationState.GREETING,
        user_profile=UserProfile(),
        created_at=session.created_at,
        updated_at=datetime.utcnow(),
    )
