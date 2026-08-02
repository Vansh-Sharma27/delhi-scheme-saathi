# ADR-0004: Split conversation.py into a package with a named-phase pipeline

**Date**: 2026-07-26
**Status**: accepted
**Deciders**: Vansh Sharma

Recorded after the fact. The change shipped in commit `55195ec`; this documents
the reasoning behind it.

## Context

`src/services/conversation.py` had grown to 2,935 lines, and `handle_message`
alone was roughly 870 of them. That single function interleaved intent
detection, language resolution, profile extraction, FSM transitions, scheme
matching and response rendering, sharing mutable local state throughout. Every
concern could reach every other one, so a change anywhere required reading the
whole function, and there was no boundary at which behaviour could be tested
without driving a full turn.

## Decision

Split the module into `src/services/conversation/` — `language.py`,
`intents.py`, `scheme_reference.py`, `turn_policy.py`, `views.py` and
`service.py` — with a strict one-way dependency order in which no module imports
anything later in the list. `handle_message` became a pipeline of named phases
(`_analyze_turn`, `_resolve_language`, `_handle_turn_reset`,
`_apply_profile_updates`, `_decide_next_state`, `_render_state`,
`_finalize_turn`) passing frozen dataclasses (`TurnAnalysis`, `ProfileUpdate`,
`RenderResult`) instead of shared locals.

## Alternatives Considered

### Alternative 1: Leave it as one module
- **Pros**: no churn, no risk of regression, no import reshuffling.
- **Cons**: the cost compounds with every subsequent change.
- **Why not**: this is a stale project that gets picked up after gaps. A 2,900-line module with an 870-line entry point is precisely what makes resuming expensive, and the file was still growing.

### Alternative 2: Extract helper functions, keep one module
- **Pros**: much smaller diff; no changes to import paths or test patch targets.
- **Cons**: does not introduce any boundary — helpers still sit beside their callers with nothing preventing cycles.
- **Why not**: it shortens functions without making the dependency structure legible or enforceable. The layering is the part that pays off later.

### Alternative 3: Split by FSM state, one module per state
- **Pros**: maps directly onto the ten-state machine, which is the system's most visible concept.
- **Cons**: cuts across the real seams. Language detection, intent matching and rendering are each used by most states.
- **Why not**: would duplicate shared logic across state modules or force a shared "common" module that becomes the old problem again.

## Consequences

### Positive
- The longest function is now 79 lines, down from roughly 870, across 96 functions.
- The one-way dependency order means a reader can start at `language.py` and work down without backtracking.
- Phase boundaries are testable individually, and the frozen dataclasses make the data crossing each boundary explicit.
- Logic the webhook handler had duplicated now has one home in `language.py`.

### Negative
- Total line count rose from 2,935 to 3,459. The increase is module headers, imports, dataclass definitions and docstrings — the cost of making the structure explicit rather than implicit.
- `service.py` is still 1,536 lines and remains the natural place for further extraction.
- Tests that patch by string target had to be retargeted, since names are now bound in the module that imports them rather than the module that defines them.

### Risks
- The layering is a convention, not something the tooling enforces; an import that reaches back up the list would compile fine. Reviewing new imports against the documented order is the only guard.
