"""Conversation orchestration.

The turn pipeline lives in :mod:`service`; the layers it draws on are split
by what they need to know:

- :mod:`language` — what language to read and reply in
- :mod:`intents` — what the user is asking for, from the message alone
- :mod:`scheme_reference` — which scheme the user means
- :mod:`turn_policy` — decisions that also need session and profile state
- :mod:`views` — the plain text that gets sent back

Dependencies run in that order; nothing imports back up the list.
"""

from src.services.conversation.service import ConversationService

__all__ = ["ConversationService"]
