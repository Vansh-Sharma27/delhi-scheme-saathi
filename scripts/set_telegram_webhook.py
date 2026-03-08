#!/usr/bin/env python3
"""Set Telegram bot webhook URL.

Usage:
    python scripts/set_telegram_webhook.py --url https://your-domain.com/webhook/telegram
    python scripts/set_telegram_webhook.py --commands  # Register default bot commands
    python scripts/set_telegram_webhook.py --delete  # Remove webhook
    python scripts/set_telegram_webhook.py --info    # Get current webhook info
"""

import argparse
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

from src.integrations.telegram import DEFAULT_BOT_COMMANDS


async def set_webhook(token: str, url: str) -> None:
    """Set webhook URL."""
    api_url = f"https://api.telegram.org/bot{token}/setWebhook"

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json={"url": url})
        result = response.json()

    if result.get("ok"):
        print("✅ Webhook set successfully!")
        print(f"   URL: {url}")
    else:
        print(f"❌ Failed to set webhook: {result.get('description')}")


async def delete_webhook(token: str) -> None:
    """Delete webhook."""
    api_url = f"https://api.telegram.org/bot{token}/deleteWebhook"

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url)
        result = response.json()

    if result.get("ok"):
        print("✅ Webhook deleted successfully!")
    else:
        print(f"❌ Failed to delete webhook: {result.get('description')}")


async def set_my_commands(
    token: str,
    commands: list[dict[str, str]] | None = None,
) -> None:
    """Register Telegram bot commands."""
    api_url = f"https://api.telegram.org/bot{token}/setMyCommands"
    payload = {"commands": commands or DEFAULT_BOT_COMMANDS}

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json=payload)
        result = response.json()

    if result.get("ok"):
        print("✅ Bot commands synced successfully!")
        for command in payload["commands"]:
            print(f"   /{command['command']} - {command['description']}")
    else:
        print(f"❌ Failed to sync commands: {result.get('description')}")


async def get_webhook_info(token: str) -> None:
    """Get current webhook info."""
    api_url = f"https://api.telegram.org/bot{token}/getWebhookInfo"

    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        result = response.json()

    if result.get("ok"):
        info = result["result"]
        print("📋 Webhook Info:")
        print(f"   URL: {info.get('url') or '(not set)'}")
        print(f"   Pending updates: {info.get('pending_update_count', 0)}")
        if info.get("last_error_message"):
            print(f"   ⚠️ Last error: {info['last_error_message']}")
    else:
        print(f"❌ Failed to get webhook info: {result.get('description')}")


async def get_my_commands(token: str) -> None:
    """Get current Telegram bot commands."""
    api_url = f"https://api.telegram.org/bot{token}/getMyCommands"

    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        result = response.json()

    if result.get("ok"):
        print("📋 Bot Commands:")
        commands = result.get("result", [])
        if not commands:
            print("   (not set)")
            return
        for command in commands:
            print(f"   /{command.get('command')} - {command.get('description')}")
    else:
        print(f"❌ Failed to get commands: {result.get('description')}")


async def get_me(token: str) -> None:
    """Get bot info."""
    api_url = f"https://api.telegram.org/bot{token}/getMe"

    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        result = response.json()

    if result.get("ok"):
        bot = result["result"]
        print("🤖 Bot Info:")
        print(f"   Name: {bot.get('first_name')}")
        print(f"   Username: @{bot.get('username')}")
        print(f"   ID: {bot.get('id')}")
    else:
        print(f"❌ Failed to get bot info: {result.get('description')}")


def main():
    parser = argparse.ArgumentParser(description="Manage Telegram bot webhook")
    parser.add_argument("--url", help="Webhook URL to set")
    parser.add_argument(
        "--commands",
        action="store_true",
        help="Register default Telegram bot commands",
    )
    parser.add_argument("--delete", action="store_true", help="Delete webhook")
    parser.add_argument("--info", action="store_true", help="Get webhook info")
    parser.add_argument("--token", help="Bot token (or set TELEGRAM_BOT_TOKEN env var)")
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Error: TELEGRAM_BOT_TOKEN not set")
        print("   Set via --token or TELEGRAM_BOT_TOKEN environment variable")
        sys.exit(1)

    # Run async operation
    if args.delete:
        asyncio.run(delete_webhook(token))
    elif args.info:
        asyncio.run(get_me(token))
        asyncio.run(get_webhook_info(token))
        asyncio.run(get_my_commands(token))
    elif args.commands:
        asyncio.run(set_my_commands(token))
    elif args.url:
        asyncio.run(set_webhook(token, args.url))
        asyncio.run(set_my_commands(token))
    else:
        # Default: show info
        asyncio.run(get_me(token))
        asyncio.run(get_webhook_info(token))
        asyncio.run(get_my_commands(token))


if __name__ == "__main__":
    main()
