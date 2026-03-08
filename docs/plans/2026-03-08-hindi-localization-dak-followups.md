# Hindi Localization And DAK Follow-Ups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix mixed-language scheme follow-up responses and make DAK application/document follow-ups use the richer grounded data already present in the catalog.

**Architecture:** Keep scheme follow-ups deterministic first. Reuse existing structured scheme, document, and rejection data for grounded answers; only use bounded AI translation when the selected reply language is Hindi or Hinglish and the available source text is still English-only. Preserve current FSM state transitions and selected-scheme context.

**Tech Stack:** Python 3.11, pytest, async conversation service, existing AI orchestrator, Telegram text responses

---

## Eval Definition: hindi-localization-dak-followups

### Capability Evals
1. Hindi document guidance translates English-only document metadata instead of leaking raw English snippets.
2. Hindi rejection warnings prefer Hindi rule content over English prevention text.
3. Asking for application steps returns grounded step-by-step guidance when `scheme.application_steps` exists.

### Regression Evals
1. Active scheme context is preserved while moving between details, documents, rejection warnings, and application help.
2. Existing `/language` and translation-follow-up regressions continue to pass.
3. Existing deterministic eligibility / why-this-scheme answers continue to pass.

### Success Metrics
- pass@1 for the new targeted regressions
- pass^1 for the adjacent scheme-context regressions and lint

### Task 1: Add Failing Regressions

**Files:**
- Modify: `tests/test_conversation_regressions.py`

**Step 1: Write the failing tests**

Add regressions for:
- Hindi rejection warnings using Hindi rule content
- Hindi document guidance translating English-only document metadata
- Application-step requests using `scheme.application_steps` instead of only `offline_process`

**Step 2: Run tests to verify they fail**

Run:
```bash
./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'hindi_rejection or hindi_document or application_steps'
```

Expected: FAIL on the current implementation.

### Task 2: Localize Grounded Follow-Up Builders

**Files:**
- Modify: `src/services/conversation.py`
- Modify: `src/services/response_generator.py`

**Step 1: Add a bounded translation helper**

Use the existing AI orchestrator only for faithful translation/rephrasing of already-grounded text. Preserve numbers, URLs, phone numbers, addresses, and ordering. If translation is unavailable, fall back to the original grounded text rather than an error string.

**Step 2: Apply it to mixed-language follow-up builders**

Translate only when needed for:
- document guidance
- application guidance
- any other follow-up builder that still leaks English-only structured fields

**Step 3: Fix deterministic Hindi label leaks**

Replace mixed labels like `rejection warnings` / `application steps` inside Hindi follow-up prompts with localized text.

### Task 3: Use Richer Application Data

**Files:**
- Modify: `src/services/conversation.py`
- Modify: `src/services/response_generator.py`

**Step 1: Expand application guidance inputs**

Pass the selected scheme’s:
- `application_steps`
- `application_url`
- `offline_process`
- `processing_time`
- `helpline`

**Step 2: Prefer step-by-step guidance when available**

Return stepwise application help first; use the shorter offline summary as supporting context instead of the whole answer.

### Task 4: Re-Run Evals And Adjacent Regressions

**Files:**
- No code changes unless failures require a minimal follow-up patch.

**Step 1: Run targeted regressions**

```bash
./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'hindi_rejection or hindi_document or application_steps or language_switch_translation_request_uses_scheme_answer_path or detail_language_change_stays_in_details'
```

Expected: PASS

**Step 2: Run lint**

```bash
./.venv/bin/ruff check src/services/conversation.py src/services/response_generator.py tests/test_conversation_regressions.py
```

Expected: PASS

### Task 5: Summarize Root Cause And Residual Risk

**Files:**
- Optionally update: `agent-handoff.md` after implementation

**Step 1: Capture root causes**

- English-only source fields were being emitted directly with no translation path.
- Hindi rejection warnings used English `prevention_tip` despite Hindi descriptions existing.
- Application-step questions were answered by a generic offline summary builder that ignored `application_steps`, which made repeated user asks feel looped.

**Step 2: Note residual risks**

- Translation quality still depends on LLM availability when no localized source field exists.
- Long official addresses and titles should be preserved verbatim even in Hindi output.
