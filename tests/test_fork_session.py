"""Tests for the session inspect/fork script."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts import fork_session
from src.db.session_store import InMemorySessionStore
from src.models.session import ConversationState, Message, Session, UserProfile


class _FakeStore:
    """Stand-in for a shared store, deliberately not InMemorySessionStore."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    async def get(self, user_id: str) -> Session | None:
        return self.sessions.get(user_id)

    async def save(self, session: Session) -> None:
        self.sessions[session.user_id] = session

    async def delete(self, user_id: str) -> None:
        self.sessions.pop(user_id, None)


def _telegram_session(user_id: str = "780045592") -> Session:
    return Session(
        user_id=user_id,
        state=ConversationState.SCHEME_PRESENTATION,
        user_profile=UserProfile(age=62, annual_income=180000, category="EWS"),
        messages=[Message(role="user", content="widow pension chahiye")],
        completed_turn_count=4,
    )


@pytest.fixture
def store():
    """A populated fake store wired into the script."""
    fake = _FakeStore()
    with patch.object(fork_session, "_configure_session_store", lambda: None), patch.object(
        fork_session, "get_session_store", lambda: fake
    ):
        yield fake


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("repro-12", "api:repro-12"),
        ("  repro-12  ", "api:repro-12"),
        # A Telegram ID passed as the label must still land in the api keyspace.
        ("780045592", "api:780045592"),
        # Already-prefixed input must not double up.
        ("api:780045592", "api:780045592"),
    ],
)
def test_fork_id_always_lands_in_chat_keyspace(label: str, expected: str) -> None:
    assert fork_session._fork_id(label) == expected


def test_fork_id_rejects_empty_label() -> None:
    with pytest.raises(SystemExit):
        fork_session._fork_id("   ")


@pytest.mark.asyncio
async def test_fork_copies_without_touching_source(store: _FakeStore) -> None:
    """The point of forking: the studied conversation must not be mutated."""
    source = _telegram_session()
    await store.save(source)

    await fork_session.fork("780045592", "repro-turn-12", force=False)

    assert sorted(store.sessions) == ["780045592", "api:repro-turn-12"]
    assert store.sessions["780045592"] == source

    forked = store.sessions["api:repro-turn-12"]
    assert forked.state is source.state
    assert forked.user_profile.age == 62
    assert len(forked.messages) == len(source.messages)


@pytest.mark.asyncio
async def test_fork_never_writes_to_a_bare_telegram_id(store: _FakeStore) -> None:
    """Even a numeric label must not overwrite the real session."""
    source = _telegram_session()
    await store.save(source)

    await fork_session.fork("780045592", "780045592", force=False)

    assert store.sessions["780045592"] == source
    assert "api:780045592" in store.sessions


@pytest.mark.asyncio
async def test_fork_refuses_to_overwrite_existing_target(store: _FakeStore) -> None:
    await store.save(_telegram_session())
    await fork_session.fork("780045592", "repro", force=False)
    first = store.sessions["api:repro"]

    with pytest.raises(SystemExit):
        await fork_session.fork("780045592", "repro", force=False)

    assert store.sessions["api:repro"] is first


@pytest.mark.asyncio
async def test_fork_overwrites_with_force(store: _FakeStore) -> None:
    await store.save(_telegram_session())
    await fork_session.fork("780045592", "repro", force=False)
    first = store.sessions["api:repro"]

    await fork_session.fork("780045592", "repro", force=True)

    assert store.sessions["api:repro"] is not first


@pytest.mark.asyncio
async def test_missing_source_session_exits(store: _FakeStore) -> None:
    with pytest.raises(SystemExit):
        await fork_session.fork("does-not-exist", "repro", force=False)


@pytest.mark.asyncio
async def test_show_omits_raw_messages_unless_requested(
    store: _FakeStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Conversation text is the most sensitive field, so it is opt-in."""
    await store.save(_telegram_session())

    await fork_session.show("780045592", include_messages=False)
    assert "widow pension chahiye" not in capsys.readouterr().out

    await fork_session.show("780045592", include_messages=True)
    assert "widow pension chahiye" in capsys.readouterr().out


def test_in_memory_store_fails_loudly() -> None:
    """A per-process store cannot see the app's sessions; say so, don't 404."""
    with patch.object(fork_session, "_configure_session_store", lambda: None), patch.object(
        fork_session, "get_session_store", lambda: InMemorySessionStore()
    ), pytest.raises(SystemExit):
        fork_session._require_shared_store()
