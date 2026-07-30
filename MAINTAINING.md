# MAINTAINING

Cross-repo runtime contracts for `fifty-agent-sdk`. When you change a row's
contract, sweep its consumers and re-point them in the same change. Seeded by
TD-003 (FR-186 contract→consumer-map obligation).

This file is **maintainer-facing**, not consumer-facing — it records who breaks
when a surface moves, not how to use the SDK. For usage see `README.md`.

## Consumer identities are deliberately not in this file

**This repository is public. Its known consumers are private repositories
belonging to a different organisation.** Naming them here — or citing their
module paths, migration filenames, or the fact that one runs a production
database — would publish a third party's private information into public git
history, irreversibly. So this map identifies consumers **by role only**:

| Label | Role |
|-------|------|
| **Consumer A** | An agent application. Subclasses `MCPClient`, wraps `Tool`s, constructs `Hooks`, and patches an SDK transport by string path in its tests. |
| **Consumer B** | A service with a **production database**. Uses the `sql` extra and hand-authors its own Alembic chain against this SDK's ORM schema. Also implements `Tool`, `AuditSink` and `LLMClient` structurally, and constructs `Hooks`. |

**The exact identities, file paths and line numbers are recorded in the private
brief tracker** (TD-003 and its plan), which is where a maintainer with access
should look before sweeping a row. What is public here is the part that is
useful without them: *which surfaces are externally coupled, how tightly, and
what a change to each one costs.*

This is a deliberate exception to TD-003's AC2, which asked for the consumers to
be named. The disclosure risk was found during review and outranks the citation
precision. The trade is real and worth stating: a sweep from this file alone
tells you **that** you must notify a consumer and **what** to tell them, but not
**which files to grep** — for that, open the tracker.

**Authoring rule (the anti-rot test depends on it).** Every path is backticked.
Every path in this file is a path in **this** repo, starting `src/` or `tests/`,
optionally with a `:line` suffix. `tests/test_maintaining.py` asserts that every
one of them still exists. **Do not add consumer-repo paths** — see above.

**A row requires a real, verified consumer**, even though it is described by
role rather than named. Not "a seam somebody might implement" — an integration
confirmed by grepping an actual checkout. This repo has three standing decisions
of the form *never on speculation* (see `coding_guidelines.md` Decisions), and a
map padded with maybe-consumers is exactly the doc that rots. Seams with no
confirmed consumer go in [Not mapped](#not-mapped), so their absence is a
recorded decision rather than an oversight.

**Row altitude: one row = one integration capability** = the set of symbols a
consumer must adopt together to use one thing. The mechanical test: *if you
renamed any single member of the group, would the same consumer have to change
code (or a migration)?* If yes, one row. A different consumer with a different
reason is a different row. Per-`__all__`-entry is too fine (60 rows nobody
maintains); "the public API" is too coarse (tells a maintainer nothing about who
to notify). This is the grouping the MCP stability pin already uses, not an
invented one.

## Runtime Contracts

| Contract | Surface | Consumers | Re-point action |
|----------|---------|-----------|-----------------|
| **`state/sql.py` physical database schema** — **highest-consequence row** | The table, column, unique-constraint and index NAMES emitted by the `sql` extra's ORM: `agent_sessions` (`src/fifty_agent_sdk/state/sql.py:199`), `agent_messages` (`:280`), `agent_branches` (`:372`), `uq_agent_messages_session_branch_sequence` (`src/fifty_agent_sdk/state/sql.py:286`), `ix_agent_messages_session_branch_sequence` (`:289`), and the `ON DELETE CASCADE` FKs to `agent_sessions.session_id`. Exposed as `fifty_agent_sdk.sql_metadata`. The SDK ships ORM models and **deliberately does not own migrations** — `SqlStateStore` has no `create_all` / bootstrap and expects the tables to pre-exist. | **Consumer B** — hand-authors an Alembic chain against a **production** database, transcribed from this ORM column-for-column, and pins itself to an SDK version in that migration's own docstring. Its migrations drop and recreate the `uq_`/`ix_` pair **by name**. | **Any DDL-name change is a MAJOR break, and this repo catches only some of them.** `tests/state/test_sql.py` asserts against `sql_metadata` directly, not only through the ORM: a **table** rename fails `test_metadata_exposes_both_tables` (`tests/state/test_sql.py:640`) and a **column** rename fails `test_metadata_columns_match_schema` (`:648`), which compares the full column-name set of all three tables. **The constraint and index NAMES were the uncovered gap** — `test_metadata_unique_constraint_on_session_branch_sequence` (`:691`) matches on the constraint's *columns* and never on its `name=`. TD-003 closed that with `test_metadata_pins_constraint_and_index_names`, so a rename now fails here instead of silently desynchronising a production database. On any rename/drop/constraint change: bump MAJOR, say so in `CHANGELOG.md` with the old→new names, and notify Consumer B so a follow-on migration lands before the version bump. Additive nullable columns are safe (their table simply lacks them until they migrate). |
| **MCP public surface** — pinned by `tests/mcp/test_interface_stability.py` | `MCPClient`, `MCPClientConfig`, `MCPToolDef`, `MCPToolErrorHook` (`src/fifty_agent_sdk/mcp/client.py`), `MCPProvider`, `RefreshSummary` (`src/fifty_agent_sdk/tools/mcp_provider.py`) — plus two couplings the pin does not express: the module path `fifty_agent_sdk.mcp.client`, and the fact that `on_tool_error` fires *inside* `MCPClient.invoke`. | **Consumer A** — **subclasses** `MCPClient` and overrides the public `invoke`, constructing it with the public `on_tool_error` hook; patches `fifty_agent_sdk.mcp.client.StreamableHttpTransport` by **string path** in its tests. **Consumer B** — *calls* `MCPClient` / `MCPClientConfig` / `MCPProvider` but does not subclass. | Update the pin in the SAME change that widens the surface, then sweep both consumers. **Subclassing is a tighter contract than calling:** A overriding `invoke` makes the method's name, its `(self, name, args)` shape, and where `on_tool_error` fires relative to it all load-bearing for them, while B would survive an internal re-order. Moving `mcp/client.py` breaks A's string-path test patch with an `AttributeError` naming no SDK file — keep a re-export at the old module path, or major-bump. |
| **Submodule import paths** — no pin | `api_pattern.md` says everything is imported from the package root and "submodule paths work but are not the contract". In practice both consumers import through them in **src-tier** code: `fifty_agent_sdk.tools.protocol`, `fifty_agent_sdk.llm.types`, `fifty_agent_sdk.state.protocol`, `fifty_agent_sdk.streaming`, `fifty_agent_sdk.errors`. Two adjacent cases: `fifty_agent_sdk.observability.hooks` is reached this way only in a consumer **test**, and `fifty_agent_sdk.mcp.client` is never imported by submodule path — it appears only as a `patch()` **string**, which the MCP row covers. | **Consumer A** — three src-tier sites importing from `tools.protocol`. **Consumer B** — sites importing from `state.protocol`, `tools.protocol`, `llm.types` and `errors`; `streaming` is the widest at five src-tier sites. | The stated contract is root-only, so this coupling is **the consumers' risk, not a promise this repo made** — recorded here so the cost of moving a module is visible rather than surprising. **Do not infer a file's import style from the symbols it uses:** the same names are imported by root elsewhere in the same repos, so a sweep of this row must re-grep for `fifty_agent_sdk.` **with a trailing dot**, never for the symbol names. (TD-003's own first draft got this wrong on four files and CI stayed green — see [Known limitations](#known-limitations).) When relocating a module under `src/fifty_agent_sdk/`, prefer leaving a re-export at the old path for one MINOR; otherwise treat it as MAJOR and name the moved paths in `CHANGELOG.md`. |
| **Protocol seams with a confirmed external implementor** — no pin | Structural typing means an external class satisfies `Tool` (`src/fifty_agent_sdk/tools/protocol.py`), `AuditSink` (`src/fifty_agent_sdk/audit/protocol.py`) or `LLMClient` (`src/fifty_agent_sdk/llm/protocol.py`) without importing or inheriting anything — and `Hooks` (`src/fifty_agent_sdk/observability/hooks.py`) is constructed by keyword. Widening a signature or adding a required protocol method breaks the implementor with **no** failure in this repo and no `runtime_checkable` failure there (`runtime_checkable` checks method *presence* only). | `Tool` — both consumers (A wraps tools for caching; B wraps one for gateway resilience). `AuditSink` — Consumer B. `LLMClient` — Consumer B, in a src-tier scripted client (not a test double). `Hooks` — both, each constructing it by keyword. | The only guard is the **consumer's own** `mypy` run, which happens after they upgrade. So: adding a protocol method or widening an existing signature is MAJOR; adding a keyword-only parameter with a default is MINOR and safe. New `Hooks` slots must default to `None` (the existing convention) or every keyword construction site breaks. Name any protocol change in `CHANGELOG.md` — it is the only notice these implementors get. |

## Private symbols

**Any `_`-prefixed name, at any depth in `fifty_agent_sdk`, is internal.** It is
not part of the public API, carries **no semver protection**, and may be
renamed, re-signatured or deleted in a **PATCH** release. Only names in
`fifty_agent_sdk.__all__` (`src/fifty_agent_sdk/__init__.py`) are supported.
This is stated in the shipped package docstring too, so it ships in the wheel
and reaches `help(fifty_agent_sdk)` and IDE hovers.

If you are a consumer and no public surface does what you need, that gap is a
**bug in this SDK** — open an issue rather than reaching in. BR-010 exists
precisely because a consumer papered over a missing public seam in silence.

The effective control is **consumer-side**: a lint rule banning
`from fifty_agent_sdk… import _*` and flake8-`SLF001`-style private-member
access on SDK objects. This repo can recommend that; it cannot enforce it.

### Known private reach-ins

We learn of these **only by accident** — absence from this list is not
evidence. See [Known limitations](#known-limitations).

| Status | Consumer | Private symbol | Observed | Supported replacement |
|--------|----------|----------------|----------|-----------------------|
| **REPLACED — awaiting consumer migration** | Consumer B, in **production** code (not just tests) | `AgentRunner._state` (`src/fifty_agent_sdk/runner.py`) | 2026-07-30 | **`AgentRunner.state`** (`src/fifty_agent_sdk/runner.py`), a read-only property shipped in **1.5.0** by **BR-011**. It returns the *exact instance* passed as `state=` — by identity, so the shared engine **and** the shared per-session lock registry both come with it, which constructing a second store over the same engine would not give. `_state` is unchanged and source-compatible in 1.5.0, so nothing breaks on upgrade, but it still carries no semver protection. **This row flips to CLOSED only on a grep-verified consumer migration**, matching the standard the CLOSED row below was held to — shipping a replacement and a consumer adopting it are different events, and the latter is not observable from this repo (see [Known limitations](#known-limitations)). At that point, also promote the accessor to a Runtime Contracts row and add a runner-side `test_interface_stability.py` pin — which must not land before then, because pins→rows requires a Runtime Contracts row naming a *verified* consumer. That is a policy, not something the path-level test enforces: its row scan reads every table line in this file, so a pin mentioned in this very row would satisfy it mechanically. |
| CLOSED | Consumer A | `fifty_agent_sdk.mcp.client._MCPCallError` (imported) and `MCPClient._unwrap_invoke_result` (overridden) | closed 2026-07-30 | `MCPClient(..., on_tool_error=…)`, shipped in 1.4.0 by BR-010. Verified discharged: both symbols now appear in that consumer **only** inside comments and test docstrings describing the migration away from them. |

## Not mapped

Plugin seams that exist but have **no confirmed external implementor** as of the
provenance date below. Recorded as a decision, not an oversight — a speculative
row is an unverifiable claim, and unverifiable claims are the rot vector. Add a
row the moment one acquires a real consumer.

- `StateStore` (`src/fifty_agent_sdk/state/protocol.py`) — both consumers use the
  shipped `SqlStateStore`; neither implements the protocol.
- `Parser` (`src/fifty_agent_sdk/parser/base.py`) — both use the shipped
  `JsonModeParser`.
- `Transport` (`src/fifty_agent_sdk/mcp/transport.py`) — no external
  implementor; one consumer patches `StreamableHttpTransport` by name, which is
  covered by the MCP row instead.
- `Registry` (`src/fifty_agent_sdk/tools/registry.py`) — a concrete class, not a
  protocol; consumers construct it but do not substitute it.

## Known limitations

**Read this before trusting a row.** There are two distinct rots here and only
one of them is catchable.

- **Surface rot — caught.** A row citing a file or a pin that was renamed or
  deleted fails `tests/test_maintaining.py`. That test asserts **file paths
  only**: every `tests/**/test_interface_stability.py` is named by a table row
  here, and every local path this file names exists. Symbol-level correctness is
  the pin test's job; duplicating it here would be a weaker copy.
- **Consumer rot — NOT caught, and not catchable.** Whether the consumer
  descriptions are still true is cross-repo state. **No test in this repository
  can observe it.** The proof is in this file's own history, twice over: TD-003's
  brief asserted two consumer facts and *both* were stale; then TD-003's *first*
  draft of the submodule-path row described four consumer files that turned out
  to import by **root**, the exact opposite of the claim. CI was green
  throughout both. Only hand re-grepping caught either. **When sweeping a row,
  re-grep the consumer checkouts — do not edit around what is already written**,
  because a wrong description looks identical to a right one from inside this
  repo.
- **The reverse direction is out of scope.** A consumer reaching into a private
  symbol is **undetectable from inside this repo** — the symbol exists in
  whatever version they installed, `mypy --strict` here is happy, and their
  justifying comment lives in their tree. The reach-ins log above is a record of
  accidental discoveries, never a survey.
- **Pins→rows is required; rows→pins is optional.** A row may legitimately have
  no stability pin — the submodule-path row does not, and recording an
  *unpinned* coupling is most of this file's value. Requiring a pin per row would
  push the next author to delete the honest row or fake a pin.
- **Role labels are coarser than names.** Because identities live in the private
  tracker, this file cannot tell you which checkout to grep. That is the cost of
  the redaction, accepted knowingly.
- Consumer descriptions are point-in-time, derived from two checkouts at one
  moment. A third consumer may exist; the SDK is on PyPI. Under-claiming with a
  stated method beats over-claiming.

## Provenance

Every consumer claim above was verified by grep on **2026-07-30** against local
checkouts of the two consumer repositories. Re-verify rather than trust: a map is
a snapshot, and the entire point of TD-003 is that snapshots rot.
