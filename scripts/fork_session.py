"""Inspect a conversation session, or fork one into the /api/chat keyspace.

``POST /api/chat`` namespaces caller-supplied IDs under ``api:`` so an
unauthenticated request cannot address a Telegram user's session. That is
deliberate, but it leaves a real need unserved: reproducing a bad response a
live user actually hit.

Forking serves it without the hazard. A fork is an independent copy, so replay
and prompt iteration never mutate the conversation being studied — which the
old shared-keyspace behaviour did, advancing the user's FSM state and
overwriting their extracted profile while you tested.

Usage:
    python scripts/fork_session.py show 780045592
    python scripts/fork_session.py show 780045592 --messages
    python scripts/fork_session.py fork 780045592 --to repro-turn-12

Then drive the fork normally:
    curl -X POST http://localhost:8000/api/chat \\
      -H "Content-Type: application/json" \\
      -d '{"user_id": "repro-turn-12", "message": "..."}'

Output contains real personal data — age, income, caste category and, with
--messages, the conversation itself. Do not paste it into issues or logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, NoReturn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session_store import InMemorySessionStore, get_session_store

# CHAT_SESSION_PREFIX and the store-selection rule are imported rather than
# redeclared: they must match the running application exactly, or this script
# reads the wrong place and reports a session as missing when it is not.
from src.main import CHAT_SESSION_PREFIX, _configure_session_store
from src.models.session import Session

IN_MEMORY_HELP = """\
The configured session store is in-memory, which is per-process: sessions live
inside the running application, so this script sees an empty store and every
lookup would report 'not found'.

Point it at a real store instead:
  - AWS:   set SESSION_TABLE_NAME (and AWS credentials/AWS_REGION) so the
           DynamoDB store is selected, as src/main.py:_configure_session_store
           decides it.
  - Local: sessions are not reachable from a separate process at all. Reproduce
           by driving POST /api/chat directly, or run against the deployed
           table.
"""


def _fail(message: str) -> NoReturn:
    """Exit non-zero with a diagnosis rather than a silent empty result."""
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _require_shared_store() -> None:
    """Refuse to run against a store that cannot hold the app's sessions."""
    _configure_session_store()
    if isinstance(get_session_store(), InMemorySessionStore):
        _fail(IN_MEMORY_HELP)


def _fork_id(label: str) -> str:
    """Normalize a label into an /api/chat session ID.

    Always prefixed, so a fork can never be written over a Telegram session
    even if the caller passes a numeric Telegram ID as the label.
    """
    cleaned = label.strip()
    if not cleaned:
        _fail("--to must not be empty")
    if cleaned.startswith(CHAT_SESSION_PREFIX):
        cleaned = cleaned[len(CHAT_SESSION_PREFIX):]
    return f"{CHAT_SESSION_PREFIX}{cleaned}"


def _summarize(session: Session, include_messages: bool) -> dict[str, Any]:
    """Build a readable view, omitting raw conversation text by default."""
    data: dict[str, Any] = {
        "user_id": session.user_id,
        "state": session.state.value,
        "profile": session.user_profile.model_dump(exclude_none=True),
        "message_count": len(session.messages),
        "completed_turn_count": session.completed_turn_count,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }
    if include_messages:
        data["messages"] = [
            {"role": message.role, "content": message.content}
            for message in session.messages
        ]
    return data


async def show(user_id: str, include_messages: bool) -> None:
    """Print one session without modifying it."""
    _require_shared_store()

    session = await get_session_store().get(user_id)
    if session is None:
        _fail(f"No session found for user_id={user_id!r}")

    print(json.dumps(_summarize(session, include_messages), ensure_ascii=False, indent=2))


async def fork(source_user_id: str, label: str, force: bool) -> None:
    """Copy a session into the /api/chat keyspace for safe replay."""
    _require_shared_store()

    store = get_session_store()
    source = await store.get(source_user_id)
    if source is None:
        _fail(f"No session found for user_id={source_user_id!r}")

    target_id = _fork_id(label)
    if target_id == source_user_id:
        _fail("Refusing to fork a session onto itself")

    if not force and await store.get(target_id) is not None:
        _fail(
            f"Fork target {target_id!r} already exists. Re-run with --force to "
            f"overwrite it, or choose a different --to label."
        )

    await store.save(source.copy_with(user_id=target_id))

    chat_id = target_id[len(CHAT_SESSION_PREFIX):]
    print(f"Forked {source_user_id!r} -> {target_id!r}")
    print(f"State: {source.state.value}, {len(source.messages)} messages retained")
    print(
        "\nDrive it with:\n"
        '  curl -X POST http://localhost:8000/api/chat \\\n'
        '    -H "Content-Type: application/json" \\\n'
        f"    -d '{json.dumps({'user_id': chat_id, 'message': '...'})}'"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Print a session")
    show_parser.add_argument("user_id", help="Session ID, e.g. a Telegram user ID")
    show_parser.add_argument(
        "--messages",
        action="store_true",
        help="Include raw conversation text (personal data)",
    )

    fork_parser = subparsers.add_parser("fork", help="Copy a session for replay")
    fork_parser.add_argument("user_id", help="Source session ID to copy from")
    fork_parser.add_argument(
        "--to",
        required=True,
        metavar="LABEL",
        help=f"Fork label; stored as {CHAT_SESSION_PREFIX}LABEL",
    )
    fork_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the fork target if it already exists",
    )

    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()

    if args.command == "show":
        asyncio.run(show(args.user_id, args.messages))
    else:
        asyncio.run(fork(args.user_id, args.to, args.force))


if __name__ == "__main__":
    main()
