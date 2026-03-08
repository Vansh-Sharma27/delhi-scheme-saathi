# RGSRY Follow-up Routing And Scheme Matrix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix RGSRY follow-up requests for documents, rejection warnings, and application steps when the LLM repeats the active `selected_scheme_id`, and add regression/data coverage across every active scheme.

**Architecture:** Treat explicit view requests and same-scheme follow-up questions as higher-priority than LLM-provided `selected_scheme_id` echoes. Preserve `resolved_scheme_id` for targeting the current or requested scheme, but stop using it to force a scheme-selection transition when the user is already inside that scheme flow. Add one routing regression for the concrete RGSRY failure shape and one seed-backed matrix test over all active schemes.

**Tech Stack:** Python 3.11, pytest, pytest-asyncio, FastAPI service layer, async session store, JSON seed data.

---

### Task 1: Capture the RGSRY failure as a deterministic regression

**Files:**
- Modify: `tests/test_conversation_regressions.py`
- Test: `tests/test_conversation_regressions.py`

**Step 1: Write the failing test**

Add a regression where:
- session state is `SCHEME_DETAILS`
- active scheme is `SCH-DELHI-005`
- LLM returns `selected_scheme_id="SCH-DELHI-005"` and no useful action
- user asks for `application steps`, `documents`, or `rejection warnings`
- expected result is the dedicated subview, not a replay of scheme details

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k rgsry_followup`

Expected: FAIL because the current logic replays `SCHEME_DETAILS`.

**Step 3: Write minimal implementation**

Adjust `src/services/conversation.py` so explicit subview requests beat same-scheme selection echoes.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k rgsry_followup`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_conversation_regressions.py src/services/conversation.py
git commit -m "fix: prioritize scheme follow-up views over echoed selections"
```

### Task 2: Fix same-scheme question handling in subviews

**Files:**
- Modify: `src/services/conversation.py`
- Test: `tests/test_conversation_regressions.py`

**Step 1: Write the failing test**

Add a regression where:
- state is `APPLICATION_HELP` or `DOCUMENT_GUIDANCE`
- active scheme is already selected
- LLM repeats that same `selected_scheme_id`
- user asks a question like `What is the first application step?`
- expected result uses the answer path, not a card replay

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k same_scheme_followup_question`

Expected: FAIL.

**Step 3: Write minimal implementation**

Teach `_detect_action_override`, `_requested_scheme_view`, and `_should_answer_scheme_question` to distinguish:
- same active scheme context
- actual scheme switching
- explicit subview/question asks

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k same_scheme_followup_question`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_conversation_regressions.py src/services/conversation.py
git commit -m "fix: keep same-scheme followups on the requested answer path"
```

### Task 3: Add all-schemes coverage and data integrity checks

**Files:**
- Modify: `tests/test_conversation_regressions.py`
- Create or Modify: `tests/test_seed_data.py`
- Read: `data/all_schemes.json`

**Step 1: Write the failing test**

Add:
- a seed-backed parameterized routing matrix over every active scheme for `documents`, `rejection warnings`, and `application steps`
- a data-integrity assertion that every active seed scheme has non-empty docs, rejection rules, and application steps

**Step 2: Run test to verify it fails or exposes gaps**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py tests/test_seed_data.py -k scheme_matrix`

Expected: FAIL before the routing fix if same-scheme echoed selections are still winning.

**Step 3: Write minimal implementation**

Use real scheme ids/names from `data/all_schemes.json` for the test matrix and keep production code minimal.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest -q tests/test_conversation_regressions.py tests/test_seed_data.py -k scheme_matrix`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_conversation_regressions.py tests/test_seed_data.py data/all_schemes.json
git commit -m "test: cover scheme followup routing across all active schemes"
```

### Task 4: Validate the final fix bundle

**Files:**
- Modify: `src/services/conversation.py`
- Modify: `tests/test_conversation_regressions.py`
- Modify or Create: `tests/test_seed_data.py`

**Step 1: Run focused regressions**

Run:
- `./.venv/bin/pytest -q tests/test_conversation_regressions.py -k 'rgsry_followup or same_scheme_followup_question or application_request_uses_step_by_step_guidance_when_steps_exist or document_guidance_followup_question_stays_on_answer_path or application_help_followup_question_stays_on_answer_path'`
- `./.venv/bin/pytest -q tests/test_seed_data.py`

Expected: PASS.

**Step 2: Run lint**

Run: `./.venv/bin/ruff check src/services/conversation.py tests/test_conversation_regressions.py tests/test_seed_data.py`

Expected: PASS.

**Step 3: Summarize data conclusion**

Record whether RGSRY was missing any seed data. Current expectation: **no**, the bug is routing-related, not data-related.
