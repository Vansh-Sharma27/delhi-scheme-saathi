#!/usr/bin/env python3
"""Interactive CLI test for conversation flow.

Simulates Telegram interaction via command line.
Usage: python -m scripts.test_conversation_flow
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv


async def main():
    """Run interactive conversation test."""
    load_dotenv()

    print("=" * 60)
    print("Delhi Scheme Saathi - Conversation Test CLI")
    print("=" * 60)
    print("Type your messages to simulate Telegram chat.")
    print("Commands: /reset - restart, /profile - show profile, /quit - exit")
    print("-" * 60)

    # Check for required environment variables
    if not os.getenv("XAI_API_KEY"):
        print("⚠️  XAI_API_KEY not set - LLM features will use fallbacks")
    if not os.getenv("DATABASE_URL"):
        print("⚠️  DATABASE_URL not set - using default localhost")

    # Import after env is loaded
    from src.db.session_store import InMemorySessionStore, configure_session_store
    from src.models.api import ChatRequest
    from src.services.conversation import ConversationService
    from src.db.connection import init_pool, close_pool

    # Use in-memory session store
    configure_session_store(InMemorySessionStore())

    # Initialize database
    try:
        pool = await init_pool()
        print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Make sure PostgreSQL is running with the scheme data")
        return

    # Create conversation service
    conversation = ConversationService(pool)
    user_id = "cli-test-user"

    print("\n🤖 Bot: Ready! Type 'Namaste' to start.\n")

    try:
        while True:
            try:
                user_input = input("👤 You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/reset":
                from src.services import session_manager
                await session_manager.delete_session(user_id)
                print("🔄 Session reset. Type 'Namaste' to start fresh.\n")
                continue

            if user_input.lower() == "/profile":
                from src.services import session_manager
                session = await session_manager.get_or_create_session(user_id)
                profile = session.user_profile
                print(f"📋 Current Profile:")
                print(f"   State: {session.state.value}")
                print(f"   Life Event: {profile.life_event}")
                print(f"   Age: {profile.age}")
                print(f"   Gender: {profile.gender}")
                print(f"   Category: {profile.category}")
                print(f"   Income: {profile.annual_income}")
                print(f"   Completeness: {profile.completeness_score}/10")
                print()
                continue

            # Process message
            request = ChatRequest(
                user_id=user_id,
                message=user_input,
                message_type="text",
            )

            try:
                response = await conversation.handle_message(request)
                print(f"\n🤖 Bot: {response.text}")

                if response.schemes:
                    print("\n📋 Matched Schemes:")
                    for match in response.schemes:
                        scheme = match.scheme
                        print(f"   • {scheme.name_hindi}")

                if response.inline_keyboard:
                    print("\n   [Keyboard buttons would appear here]")

                print()

            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    finally:
        await close_pool()
        print("\n👋 Session ended.")


if __name__ == "__main__":
    asyncio.run(main())
