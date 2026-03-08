# Language Normalization, Voice Echo, And Application Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Hindi, Hinglish, and English outputs stay consistent with the user's selected language, fix application-procedure routing, and make voice transcript echoes respect the user's language setting.

**Architecture:** Keep the existing FSM and scheme-answer paths, but add one shared response-language normalization layer that uses the Bedrock/Grok LLM clients as a bounded translator/re-writer when output text leaks into the wrong language or script. Tighten deterministic action detection for application/procedure turns, and base voice transcript echo prefixes on the locked session language instead of transcript language alone.

**Tech Stack:** FastAPI bot services, async Python 3.11, pytest, existing AI orchestrator over Bedrock/Grok, Telegram webhook handlers.

---

### Task 1: Add failing regressions for language drift

**Files:**
- Modify: `tests/test_conversation_regressions.py`
- Test: `tests/test_conversation_regressions.py`

**Step 1: Write the failing tests**

Add regressions for:
- locked Hinglish session receiving Hindi `llm_response_text`
- locked English active-scheme session receiving Hindi text after an application/procedure ask
- procedure/process wording routing to application help

**Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'hinglish_lock or english_scheme_language_lock or procedure_routes_to_application'`
Expected: FAIL in current language handling / routing.

**Step 3: Write minimal implementation**

Add shared language-normalization helpers and tighten procedure/application action detection.

**Step 4: Run tests to verify they pass**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'hinglish_lock or english_scheme_language_lock or procedure_routes_to_application'`
Expected: PASS

### Task 2: Normalize final response language with LLM-backed translation

**Files:**
- Modify: `src/services/response_generator.py`
- Modify: `src/services/conversation.py`
- Test: `tests/test_conversation_regressions.py`

**Step 1: Write the failing tests**

Cover:
- Hindi output with Latin leakage should be normalized
- Hinglish output with Devanagari / pure English drift should be normalized
- English output with Devanagari leakage should be normalized

**Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'language_normalization'`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:
- a shared prompt for faithful language normalization / translation
- heuristics to decide when text needs rewriting for `hi`, `hinglish`, or `en`
- a final `ensure_response_language(...)` pass in `ConversationService.handle_message(...)`

**Step 4: Run tests to verify they pass**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'language_normalization'`
Expected: PASS

### Task 3: Fix voice transcript echo language

**Files:**
- Modify: `src/webhook/handler.py`
- Modify: `tests/test_webhook.py`
- Test: `tests/test_webhook.py`

**Step 1: Write the failing tests**

Add regressions for:
- locked Hindi session with English transcript still echoing `आपने कहा:`
- locked Hinglish session using `Aapne kaha:`
- unlocked sessions still falling back to transcript language

**Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest -q tests/test_webhook.py -k 'voice_echo'`
Expected: FAIL

**Step 3: Write minimal implementation**

Base echo prefix selection on locked session language first, then transcript language for unlocked sessions.

**Step 4: Run tests to verify they pass**

Run: `./.venv/bin/pytest -q tests/test_webhook.py -k 'voice_echo'`
Expected: PASS

### Task 4: Validate focused paths and lint

**Files:**
- Modify as needed: `src/services/conversation.py`
- Modify as needed: `src/services/response_generator.py`
- Modify as needed: `src/webhook/handler.py`
- Test: `tests/test_conversation_regressions.py`
- Test: `tests/test_webhook.py`

**Step 1: Run focused pytest coverage**

Run:
- `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'language or application or procedure'`
- `./.venv/bin/pytest -q tests/test_webhook.py -k 'voice or language'`

Expected: PASS

**Step 2: Run lint**

Run: `./.venv/bin/ruff check src/services/conversation.py src/services/response_generator.py src/webhook/handler.py tests/test_conversation_regressions.py tests/test_webhook.py`
Expected: PASS

**Step 3: Commit**

```bash
git add src/services/conversation.py src/services/response_generator.py src/webhook/handler.py tests/test_conversation_regressions.py tests/test_webhook.py docs/plans/2026-03-08-language-normalization-voice-application-fixes.md
git commit -m "fix: normalize response language and voice echo"
```
