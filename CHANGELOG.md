# Changelog

All notable changes to `fifty-agent-sdk` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-07-30

### Added
- **A public accessor for a Runner's state store.** `AgentRunner` gains a
  read-only `state` property returning the `StateStore` it reads and writes.
  It returns the **exact instance** passed as the `state=` constructor keyword
  — by identity, never a copy and never a wrapper — so a caller sharing it
  shares the store's internal serialization (for example `SqlStateStore`'s
  per-session `asyncio.Lock` registry). Constructing a second store over the
  same engine is **not** equivalent: two stores carry independent lock
  registries, so two writers could interleave on one session. The accessor is
  named for its constructor keyword, making `AgentRunner(loop=…, state=s).state
  is s` a derivable invariant. Reachable from the package root today —
  `AgentRunner` and `StateStore` are both already exported.

  It is **read-only** for correctness, not style: `run()` appends the user
  message, drives the loop, then appends the assistant message, so a store swap
  landing between those appends would split one turn across two backends and
  void the documented transactional-persistence invariants. To use a different
  store, construct another Runner — `__init__` does no I/O. Assignment raises
  `AttributeError`, and `mypy` rejects it statically (there is deliberately no
  raising setter, which would make the assignment type-check as legal). The
  declared return type is the `StateStore` protocol, so a caller needing
  backend-specific API that is not on it (e.g. `SqlStateStore.aclose()`) should
  keep its own concretely-typed reference or narrow with `cast`.

  **This is the supported replacement for reaching into
  `AgentRunner._state`.** That attribute remains private, unsupported, and
  carries no semver protection — it is unchanged and source-compatible in
  1.5.0, so nothing breaks on upgrade, but consumers reading it should migrate
  to `AgentRunner.state`. (BR-011)

### Fixed
- **Branch materialization no longer recurses per lineage hop.** All three
  backends walked a branch's lineage with one Python frame per hop, so a
  pathological *linear* fork chain (~990 deep) raised `RecursionError`. The five
  affected walks — `MemoryStateStore._materialize`,
  `SqlStateStore._materialize_positional` / `._materialized_len`, and
  `RedisStateStore._materialize` / `._materialized_len` (the last two `async`,
  where an `await`ed recursive call consumes a frame just the same) — now walk
  parent pointers into a list and fold root-to-leaf. Lineage depth is bounded
  only by memory.

  The failure bit at **build** time before read time was reachable: `fork` computes the active
  branch's head through the same walk to bound `from_sequence`, so on affected
  versions a chain that deep could not be constructed through the public API at
  all. Realistic branching is wide rather than 1000-deep-linear, which is why
  this went unnoticed.

  **No API and no behaviour change.** No signature moves (all five are private
  helpers), and the fold evaluates the identical expression with the identical
  associativity — the recursive form was already a left fold from the root, so
  each hop's `min` / prefix-slice clamp still sees the already-clamped ancestor
  length. The existing differential fuzz suite passes unmodified. On Redis the
  per-hop `LRANGE`/`LLEN` count and command set are unchanged; only the call
  *order* flips from leaf-to-root to root-to-leaf, and multi-key reads were
  never atomic in either order.

  The brief's optional half — wrapping these paths in `StateStoreError` — was
  considered and **declined**: with the walk iterative the `RecursionError` it
  guarded against is unreachable, and a broad `except Exception` there would
  swallow the protocol-mandated `ValueError` for an unknown explicit `branch_id`
  as well as Redis's deliberate pydantic `ValidationError` pass-through. No
  lineage-cycle guard was added either — `parent_branch_id` is written once, at
  `fork`, to an already-existing branch and never mutated, so the lineage is
  acyclic by construction. (TD-002)

## [1.4.0] - 2026-07-29

### Added
- **A public hook for transforming MCP `isError` text before it reaches the
  model.** `MCPClient` gains a keyword-only `on_tool_error` callback (typed
  `MCPToolErrorHook`, exported from the package root) that receives BOTH the
  SDK's bounded default message (`"MCP tool '<name>' returned isError=True"`)
  AND the server's raw error `content` blocks, and returns the string the model
  will see as `ToolResult.error`. Wire it through plain construction —
  `MCPClient(config, on_tool_error=screen)` then `MCPProvider(client)`; no
  subclassing required. The hook may be sync or `async def` (the return value is
  inspected with `inspect.isawaitable`), and it fires exactly once per per-call
  `isError=True` result — never on success, and never on a transport/protocol/
  session failure, which still raises `MCPError` at the `call_tool` boundary
  before the result is unwrapped (the BR-005 recoverable/fatal split is
  unchanged). It cannot change `is_error` or `output`: a failed tool is never
  reported as a success. On ANY hook failure the original bounded message is
  used — a raising hook is caught and logged `WARNING`
  (`mcp.tool_error_hook_failed`), and a non-`str` or blank/whitespace-only
  return is rejected and logged `WARNING` (`mcp.tool_error_hook_invalid`); those
  logs carry `tool_name`/`error_type`/`returned_type` only, never the exception
  text and never the server content. `asyncio.CancelledError` propagates
  untouched. **Default `None` — the hook-off path is byte-for-byte the 1.3.0
  path**, short-circuiting before any call or allocation.

  **This is the supported replacement for importing
  `fifty_agent_sdk.mcp.client._MCPCallError` or overriding
  `MCPClient._unwrap_invoke_result`.** Those symbols remain private,
  unsupported, and carry no semver protection — they are unchanged and
  source-compatible in 1.4.0, so nothing breaks on upgrade, but consumers doing
  either should migrate to `on_tool_error`. (BR-010)

## [1.3.0] - 2026-07-01

### Added
- **Concurrent multi-call tool dispatch (opt-in).** A single native ReACT
  iteration can now dispatch up to N independent tool calls concurrently under
  a bounded `asyncio.gather`, feeding all observations back to the model in
  one turn. Set `SafetyConfig(max_concurrent_tool_calls=N)` (default `1` —
  serialized, so a caller who enables native multi-call parsing without
  raising the cap sees no pool surprise) alongside `native_tools_enabled=True`.
  A native response carrying >1 `tool_calls` yields a new `MultiAction` parse
  result; the loop mints one distinct `call_id` per call (carried on the new
  `ToolCall.id` field), embeds all N on the replayed assistant turn, dispatches
  them through a `Semaphore`-bounded gather with `return_exceptions=True`, and
  appends observations in CALL order (deterministic regardless of completion
  order). Per-call recoverable failures (`ToolNotFound`/`ToolTimeout`/`is_error`)
  yield a `ToolFailedEvent` for that call only — siblings still complete; fatal
  `AgentSdkError`/`BaseException` re-raise and terminate. The batch counts as
  ONE `max_iterations` unit. **Single-call turns are byte-for-byte unchanged**
  — a 1-entry native response still yields `ThoughtAction` and the gather is
  unreachable; the text/JSON path never produces `MultiAction`.
  (`OpenAICompatibleClient._serialize_message` sources each wire
  `tool_calls[].id` from the entry's `ToolCall.id`, falling back to
  `tool_call_id` for the single-call path.) (BR-006)
- **Native function-calling (opt-in).** The agent loop can now dispatch tools
  from a provider's structured `tool_calls` instead of only from JSON-mode text.
  Set `SafetyConfig(native_tools_enabled=True)` and the OpenAI-compatible
  adapter declares the registry's tools via the `tools`/`tool_choice` request
  params (built from each tool's `ToolSchema`); a provider returning
  `choices[].message.tool_calls` is dispatched through the new concrete
  `NativeToolsParser`, and assistant `tool_calls` turns round-trip on the
  request wire in valid OpenAI envelope shape (`{id, type:"function",
  function:{name, arguments(JSON string)}}`) so multi-turn native conversations
  replay without a 400. `ChatMessage` gains an optional `tool_calls` field; the
  adapter populates it (parsing the provider's JSON-string `arguments`,
  `LLMError(type="MalformedResponse")` on a malformed envelope). The
  `tool_call_id` is minted before the native assistant-turn append and reused
  for the tool reply so provider replay pairs correctly. **Default OFF** — the
  text/JSON parse path (BR-018-hardened) is byte-for-byte unchanged when the
  flag is off; native mode is an explicit per-deployment choice. The
  `NativeToolsParser` Protocol is renamed `NativeToolsParserProtocol`.
  (BR-007, BR-008)

### Changed
- A native response carrying more than one `tool_calls` entry no longer
  truncates to the first call (the BR-007 scope limitation). The
  `NativeToolsParser` now validates EVERY entry's schema and returns a
  `MultiAction` carrying the full list; the `native_tool_calls_truncated`
  DEBUG log is removed. Concurrent/batched dispatch landed in BR-006 (above).

## [1.2.1] - 2026-06-25

### Fixed
- An MCP tool returning `isError=True` is now a **recoverable** tool
  observation instead of a run-terminating error. Previously the MCP adapter
  raised `MCPError` on `isError=True`; that exception escaped the agent loop
  (which only caught `ToolNotFound`/`ToolTimeout`) and propagated out of
  `Runner.run()`, crashing the turn with a raw `MCPError`. Now a per-call
  `isError=True` is surfaced as a `ToolResult(is_error=True)` — the model
  receives it as a `"Tool error: …"` observation and produces a grounded
  reply — converging MCP tool failures onto the exact same recoverable path as
  native `is_error`, `ToolNotFound`, and `ToolTimeout`. Genuinely-fatal
  transport / protocol / session `MCPError`s still terminate the run as before.
  (BR-005)

## [1.2.0] - 2026-06-22

### Added
- First-class conversation **branching** on `StateStore`: `fork`,
  `list_branches`, `switch_branch`, branch-scoped
  `get_messages(..., branch_id=...)`, plus the `BranchInfo` and
  `TRUNK_BRANCH_ID` exports. A session is now a tree of branches with an active
  head; `append` writes to the active branch (the "edit a message / regenerate"
  model). Implemented across all backends (memory, SQL, Redis). The change is
  **data-additive and zero-migration** — existing sessions read as the `trunk`
  branch. **Breaking for custom `StateStore` implementations**: they must add
  the new methods. (BR-004)
- `StateStore.truncate_after(session_id, sequence, *, branch_id=None)` — a
  destructive hard-delete of a branch's tail (messages with `sequence > N`),
  for redaction, retention, and rollback. Only the target branch's own messages
  are removed — a fork's inherited prefix is never touched — and it is
  idempotent / a no-op on an unknown session or branch. (BR-003)

### Fixed
- `Registry.invoke` now enforces timeouts via `asyncio.timeout` instead of
  `asyncio.wait_for`, running the tool coroutine inline in the caller's task.
  This makes `KeyboardInterrupt`/`SystemExit` propagation deterministic across
  Python 3.11–3.13 and fixes a pytest-session abort on the 3.11 CI leg. (BR-002)

### Changed
- `fifty_agent_sdk.__version__` is now derived from installed distribution
  metadata (`importlib.metadata`) rather than a hardcoded string, so it can no
  longer drift from `pyproject.toml`. (TD-001)

## [1.1.1] - 2026-06-22

### Fixed
- `fifty_agent_sdk.__version__` now reports the correct release version. It was
  pinned at `1.0.0` and missed the 1.1.0 bump; all version sources
  (`pyproject.toml` and `__init__.py`) are now in agreement.

## [1.1.0] - 2026-06-22

First public open-source release as a standalone package (`fifty-agent-sdk`),
extracted with its full commit history from the monorepo it was first built in.

### Added
- Standard MCP client over Streamable HTTP via the official `mcp` SDK, exposed
  through `MCPProvider` (full `initialize → tools/list → tools/call` handshake).
  The MCP path is now standard-only.

### Changed
- Import root is now `fifty_agent_sdk` (was `agent_sdk`).
- Distributed and published as `fifty-agent-sdk` on PyPI.

## [1.0.0]

Initial production release: custom ReACT loop, JSON-mode tool calling, a
pluggable LLM client (any OpenAI-compatible endpoint), in-process + MCP tool
sources, pluggable conversation-state storage (memory / SQL / Redis), audit
sinks, observability hooks, and a full-fidelity event stream.
