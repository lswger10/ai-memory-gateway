# Conversation summary separation implementation plan

> Execute inline with TDD; approved design is the user's preceding conversation. No provider calls, remote writes or deployment.

**Goal:** Reuse each actor/room/conversation summary across Profiles while keeping provider cache identities separate and original facts intact.

**Owner:** Gateway. Reuse the release worktree at 67e0653; preserve pre-existing README edits. Tidal d1b27bd remains unchanged.

## Tasks

- [x] RED: existing builder must summarize once across Profile switches, with unchanged facts and distinct provider cache keys. Oversized full summary batches must fail before a provider call, not shrink into repeated paid fragments.
- [x] Separate state: anchored_history.py owns a Profile-independent identity and conversation_summary_state; database.py adds its schema. Adopt valid existing summaries once, preserving text/version/cursor; ambiguous or mismatched provenance fails explicitly. Old rows are rollback evidence, not a second write path.
- [x] Wire execution_context_builder.py and conversation_compression.py to the summary identity, including full/probe/keepalive/manual readers. Preserve original provider namespace construction in model_usage_store.py/model_execution.py. Preserve history repair invalidation and Bedroom deletion for both old and new derived state.
- [x] Verify with synthetic tests in existing modules, including concurrent stale writers, room/actor/session boundaries, restart/store recreation, migration ambiguity, oversized input, failed summaries, and unchanged raw facts. Run full Gateway suite without credentials or live API access; local isolated PostgreSQL only if available (not available; explicitly unverified).
- [x] Review diff, document measured results and migration/rollback limits in README; leave local changes unpushed for the user's review.

## Constraints and decisions

- Keep existing automatic 128-event trigger / 48-event tail. User approved increasing the former 16,000-character summary data bound to accommodate full batches; use a 192,000-character hard bound, including prior summary and event metadata (release verification supersedes the earlier 128,000 choice below). No token/currency ceiling is implied. An oversized target is an explicit no-call error requiring separately bounded catch-up, not a paid loop.
- Keep complete human-led exchanges. Only atomically update a validated replacement summary and cursor; never delete cognitive or Relay facts.
- Summary identity includes actor, room, existing partition ID and execution mode. Bedroom partition remains bedroom:<session>. No Profile, model, tool or Persona version in this identity.
- Existing schema has no historical summary source hash. Adoption can validate coordinates, cursor, bounds and repair invalidation, not reconstruct or prove the old summary's semantic accuracy. Fail ambiguous equal-cursor summaries rather than guessing.
- No new UI/service/framework/dependency. Existing single-replica serialization + revision check remains the paid-call guard; multi-replica paid deduplication is not claimed.

## Progress

- Preflight: existing focused baseline 44 passed. Gateway README is already modified; preserve it. No application DB or provider contacted.
- Initial RED: Profile switch bought a second summary; oversized full batch did not raise (two expected failures). Implemented identity separation and full-batch bound; initial full suite 461 passed / 60 skipped / 2 subtests passed.
- Independent read-only review found the bound is unusable with the actual event envelope: 81 empty human events already serialize to 29,584 characters. Added a real-event-shape test through the builder/store/sync with synthetic Relay events; it was RED on the input-budget check. The subsequent authorized fix is recorded below.
- Requested user decision on changing the single-summary input budget; do not raise it silently or restore small paid chunks. No commit/push/deploy. Actual local PostgreSQL tests remain skipped: no approved isolated DSN or PostgreSQL executable found; no application DATABASE_URL was used.
- Final free run including the release-blocking regression: 1 failed / 461 passed / 60 skipped / 2 subtests passed (40.26s). `git diff --check` passed. Provider namespace uniqueness is asserted across the three Profile switches while summary calls stay at one. Tidal is unchanged. No branch/worktree was created and no temporary diagnostic asset was left; the plan and failing permanent regression record the unresolved decision.

## Authorized budget fix and final verification

- User decision: allow adjustment of the single-summary cap; still free-only validation, no retries and no per-turn partial batches.
- Ruling: select 128,000 characters rather than an unbounded batch. It accommodates the realistic default-batch regression and keeps a hard spend guard; static instructions add overhead, characters are not tokens, and a future large batch can still be rejected. Single-call input ceiling is higher than before, not a promise of a fixed currency cost.
- Final review fix: `test_default_compaction_accepts_real_message_shape_without_next_turn_summary` RED -> GREEN using full event envelopes and synthetic Chinese messages; the next turn reuses the resulting summary and all 133 facts remain. Exact 127,999/128,000/128,001 input boundaries include an existing 4,000-character summary.
- Error-path RED -> GREEN: the manual endpoint returns 413 rather than a misleading state conflict; generation terminates with the existing unavailable event, sanitized diagnostic code and zero provider/fallback attempts instead of an unhandled stream exception.
- Final free suite: Gateway `python -m pytest tests -q -p no:cacheprovider --tb=short` -> 467 passed / 60 skipped / 2 subtests passed (39.39s). Tidal `python -B -m pytest tests/test_gateway_execution_client.py -q -p no:cacheprovider` -> 5 passed. Real PostgreSQL remains unverified; no approved isolated instance was available.
- Scope/finish: keep the existing worktree and all pre-existing README edits. No additional review agent (this fixes the prior independent review finding with RED/GREEN), new dependency, service, feature flag, raw-history write, paid call or remote operation. Keep changes uncommitted/unpushed; no deploy in this authorization.

## Authorized test release verification (2026-10-16)

- User subsequently authorized committing/pushing/deploying this change to Zeabur test, plus synthetic PostgreSQL verification in a temporary isolated schema and cleanup. Laoke must remain on Gemini.
- Live old code still partitions summaries by Profile. Read-only inspection found a more advanced valid Claude summary than the Gemini cursor. The remaining full batch after adoption is about 143,000 characters, so the previously selected 128,000 cap would still block it. Announced the bounded increase to 192,000 before editing; no retry/chunk/fallback added.
- RED: exact increased-boundary and large-realistic-envelope tests failed against 128,000. GREEN: full Gateway suite 468 passed / 60 skipped / 2 subtests passed; Tidal Gateway-client regression 5 passed. Oversize fixtures now exceed the new limit rather than accidentally being allowed-size inputs.
- Real PostgreSQL: eight assertions passed using the exact new store source (SHA256 a55daa7f9f9f886a34c9049ae64202a55a70d59192e66d1c040ba8b9b01d7113) and new table DDL, only synthetic records in an isolated schema. Covered adoption, pool recreation/idempotent DDL, stale revision, wrong room, actor separation, Bedroom cleanup, repair invalidation and factual preservation. Schema removed and absence verified. This does not claim a database-server restart or the skipped full PostgreSQL suite.
- Incident paid diagnostics before release: 2 of 5 authorized requests used. Existing historical summary input returned HTTP 200 with no content/zero usage; synthetic summary input succeeded. No explicit safety signal was returned, so upstream cause remains unproven. No live facts, Persona or bindings changed. Deployment and real-chat restoration are not yet claimed.

## Test rollout outcome

- Code `58f6b38a18944de1c2dea2ffebe7835a502a656c` pushed to the existing test branch and deployed as `6ad1c0ef0b0bc926ae2a941e`. Runtime file hash matches Git (LF-normalized blob SHA256 `3a2fadad7810273e8df64095e0297603f5e1c767c6d2c520b69a9a80e681d0e7`), health 200, additive table present. Existing factual/Memory/Persona/Profile/binding/Pin fingerprints match across deployment.
- After explicit authorization to transmit the existing private summary and pending history to current OFOX Gemini, one manual compression attempt returned Gateway 502 `compression_unavailable`. Provider receipt `6e877dd2-b5b0-4b3c-a3ac-f2bd94b30673` reports succeeded, input 82109 / output 4092, usage received. New shared state successfully adopted cursor 1583, revision 52, summary length 994 and remained there after rejection. Old Gemini cursor was 1271. This proves adoption, not successful new compression.
- The provider receipt does not retain finish reason or summary length; the two remaining application-level rejection paths are truncated output and invalid summary storage length. Do not call it an OFOX HTTP 502, safety refusal, confirmed truncation, or restored chat without further evidence. No retry or model switch; 3/5 authorized actual requests used. The 4096 output cap remains unchanged.
