"""Session store - in-memory for MVP, DynamoDB for production."""

from datetime import datetime
from typing import Protocol

from src.models.session import Session


class SessionStore(Protocol):
    """Session store protocol for dependency injection."""

    async def get(self, user_id: str) -> Session | None:
        """Get session by user ID."""
        ...

    async def save(self, session: Session) -> None:
        """Save or update session."""
        ...

    async def delete(self, user_id: str) -> None:
        """Delete session."""
        ...


class InMemorySessionStore:
    """In-memory session store for local development."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, user_id: str) -> Session | None:
        """Get session by user ID."""
        session = self._sessions.get(user_id)
        return session.model_copy(deep=True) if session else None

    async def save(self, session: Session) -> None:
        """Save or update session."""
        updated = session.copy_with(updated_at=datetime.utcnow())
        self._sessions[session.user_id] = updated

    async def delete(self, user_id: str) -> None:
        """Delete session."""
        self._sessions.pop(user_id, None)

    def clear(self) -> None:
        """Clear all sessions (for testing)."""
        self._sessions.clear()


class DynamoDBSessionStore:
    """DynamoDB session store for AWS deployment (Phase 6)."""

    def __init__(self, table_name: str, region: str = "ap-south-1") -> None:
        import boto3
        self._table_name = table_name
        self._dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = self._dynamodb.Table(table_name)

    async def get(self, user_id: str) -> Session | None:
        """Get session by user ID."""
        import asyncio

        def _get():
            response = self._table.get_item(
                Key={"user_id": user_id},
                ConsistentRead=True,
            )
            return response.get("Item")

        item = await asyncio.get_event_loop().run_in_executor(None, _get)
        if item:
            return Session.from_dynamodb_item(item)
        return None

    async def save(self, session: Session) -> None:
        """Save or update session."""
        import asyncio

        item = session.to_dynamodb_item()

        def _put():
            self._table.put_item(Item=item)

        await asyncio.get_event_loop().run_in_executor(None, _put)

    async def delete(self, user_id: str) -> None:
        """Delete session."""
        import asyncio

        def _delete():
            self._table.delete_item(Key={"user_id": user_id})

        await asyncio.get_event_loop().run_in_executor(None, _delete)


# Global session store instance (configured at startup)
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get the configured session store."""
    global _session_store
    if _session_store is None:
        # Default to in-memory for local development
        _session_store = InMemorySessionStore()
    return _session_store


def configure_session_store(store: SessionStore) -> None:
    """Configure the session store (called at startup)."""
    global _session_store
    _session_store = store
