# Telegram Help And Language Commands Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add discoverable Telegram commands, state-aware help, and explicit language selection controls without breaking existing scheme flows.

**Architecture:** Extend the existing deterministic command path in `ConversationService` so `/start`, `/help`, and `/language` stay out of the LLM path. Add a small Telegram Bot API command-registration surface plus callback handling for language buttons, then wire deployment tooling and tests around those deterministic behaviors.

**Tech Stack:** Python 3.11, FastAPI services, Telegram Bot API, pytest, ruff

---

### Task 1: Register Telegram Bot Commands

**Files:**
- Modify: `src/integrations/telegram.py`
- Modify: `scripts/set_telegram_webhook.py`
- Modify: `scripts/rapid_redeploy.sh`
- Test: `tests/test_webhook.py`

**Step 1: Write the failing test**

Add a test for command registration payload formatting and script-level coverage where practical.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_webhook.py -k command -v`

**Step 3: Write minimal implementation**

Add Telegram client support for `setMyCommands`, expose `/start`, `/help`, `/language`, and let the webhook utility register those commands after setting webhook.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_webhook.py -k command -v`

**Step 5: Commit**

```bash
git add src/integrations/telegram.py scripts/set_telegram_webhook.py scripts/rapid_redeploy.sh tests/test_webhook.py
git commit -m "feat: register telegram bot commands"
```

### Task 2: Add State-Aware Help And Language Controls

**Files:**
- Modify: `src/services/conversation.py`
- Modify: `src/services/response_generator.py`
- Modify: `src/utils/formatters.py`
- Modify: `src/models/api.py` only if response shape needs extension
- Test: `tests/test_conversation_regressions.py`
- Test: `tests/test_webhook.py`

**Step 1: Write the failing tests**

Add regressions for:
- `/help` showing a short scheme-context help when a scheme is active
- `/language` returning a language chooser
- language callback updating session language and re-rendering help/greeting safely

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_conversation_regressions.py tests/test_webhook.py -k 'help or language' -v`

**Step 3: Write minimal implementation**

Add deterministic help variants and language picker buttons, extend callback handling beyond `scheme:<id>`, and keep current state/session context intact.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_conversation_regressions.py tests/test_webhook.py -k 'help or language' -v`

**Step 5: Commit**

```bash
git add src/services/conversation.py src/services/response_generator.py src/utils/formatters.py tests/test_conversation_regressions.py tests/test_webhook.py
git commit -m "feat: add state-aware help and language controls"
```

### Task 3: Document And Verify Deployment Behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/WHAT_THIS_BOT_CAN_DO.md`
- Test: local targeted pytest + ruff

**Step 1: Update docs**

Document `/help`, `/language`, and the fact that Telegram commands are registered for discoverability.

**Step 2: Run verification**

Run:
- `pytest tests/test_conversation_regressions.py tests/test_webhook.py -k 'help or language or start_over' -q`
- `ruff check src/services/conversation.py src/services/response_generator.py src/integrations/telegram.py tests/test_conversation_regressions.py tests/test_webhook.py`

**Step 3: Commit**

```bash
git add README.md docs/WHAT_THIS_BOT_CAN_DO.md
git commit -m "docs: describe telegram help and language commands"
```
