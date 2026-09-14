<p align="center">
  <img src=".github/banner.png" alt="fifty-agent-sdk — a reusable agent loop for python." width="100%">
</p>

# fifty-agent-sdk

[![PyPI](https://img.shields.io/pypi/v/fifty-agent-sdk)](https://pypi.org/project/fifty-agent-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/fifty-agent-sdk)](https://pypi.org/project/fifty-agent-sdk/)
[![CI](https://github.com/fiftynotai/fifty-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fiftynotai/fifty-agent-sdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

fifty-agent-sdk is a reusable agent loop for python. it implements a custom reACT loop with json-mode tool calls, an mcp client, and pluggable llm, state, and tool backends. it exists because the loop, the parser, the safety checks, and the runner kept getting rewritten per project. this is that loop, factored out once: write the tools, hand them to the runner, let it iterate.

## At a glance

- talks to any openai-compatible chat-completions endpoint by swapping one `base_url`: openai, google distributed cloud, a local oss server.
- llm clients, state stores, and tools are pluggable behind protocols: bring your own, the loop stays the same.
- the run emits a typed event stream the caller consumes, so you watch the react loop step by step.
- an iteration cap and per-tool timeouts bound every run, with a fallback answer on error or cap: a loop that can't end is a loop that doesn't ship.
- zero-infra by default: no db, no redis, until you opt into an extra.

## Installation

```
pip install fifty-agent-sdk
```

Optional extras:

- `pip install 'fifty-agent-sdk[sql]'` — enables SqlStateStore, SqlAuditSink, SQLAlchemy
- `pip install 'fifty-agent-sdk[redis]'` — enables RedisStateStore

Importing `fifty_agent_sdk` pulls neither extra; the extra symbols are re-exported lazily, and first access without the relevant extra installed raises a clear `ImportError`. The `sql` extra installs SQLAlchemy but not a database driver — bring your own async driver (e.g. `aiosqlite` for SQLite, `asyncpg` for PostgreSQL).

Requires Python >=3.11.

## Quickstart

the example builds a tool, hands it to the `AgentRunner`, and consumes the typed event stream the run emits.

```python
import asyncio
from typing import Any

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentLoop,
    AgentRunner,
    JsonModeParser,
    MemoryStateStore,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    tool,
)


@tool()
async def get_weather(city: str) -> dict[str, Any]:
    """Return the current weather for a city."""
    return {"city": city, "temp_c": 21}


async def main() -> None:
    # 1. An LLM client — points at any OpenAI-compatible endpoint.
    #    Pass base_url=... to target GDC or a local OSS server instead of OpenAI.
    llm = OpenAICompatibleClient(api_key="sk-...")

    # 2. A tool registry — register the decorated tool.
    registry = Registry()
    registry.register(get_weather)

    # 3. The ReACT loop — LLM + registry + parser + prompts + safety.
    #    `output_format` shows the model the JSON envelope the parser
    #    expects; without it JsonModeParser raises ParserError on every turn.
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        parser=JsonModeParser(),
        prompts=PromptSections(persona="You are helpful."),
        safety=SafetyConfig(),
        model="gpt-4o",
        output_format=JSON_MODE_OUTPUT_FORMAT,
    )

    # 4. The runner — wraps the loop with conversation-state persistence.
    runner = AgentRunner(
        loop=loop,
        state=MemoryStateStore(),
        system_prompt="You are a helpful weather assistant.",
    )

    # 5. Drive a turn and consume the event stream.
    async for event in runner.run("session-1", "What's the weather in Paris?"):
        print(event)


asyncio.run(main())
```

## Core concepts

### tools

the registry of functions the agent can call. each tool is a side-effecting action exposed to the loop, so the model can do something in the world and not just talk about it.

### llm

the llm client. a protocol plus an openai-compatible adapter, so the loop talks to any chat-completions endpoint by changing one base_url.

### state

the state stores. where conversation state persists between turns, with branching built in: fork a session, switch between branches, truncate back to an earlier point. `MemoryStateStore` needs no infrastructure; `SqlStateStore` and `RedisStateStore` are durable backends behind the extras.

a runner hands back the store it was built with as `runner.state`, so the branching calls above are reachable from a runner you already have:

```python
store = MemoryStateStore()
runner = AgentRunner(loop=..., state=store)

runner.state is store  # True — the exact instance, never a copy or a wrapper
branch = await runner.state.fork(session_id, from_sequence=4)
```

identity is the point rather than convenience: a second store constructed over the same engine carries its own lock registry, so two writers could interleave on one session. sharing `runner.state` shares the serialization too.

it is read-only, for correctness and not for style. `run()` appends the user message, drives the loop, then appends the assistant message — a store swapped in between those appends would split one turn across two backends. assignment raises `AttributeError`, and mypy rejects it statically. to use a different store, construct another runner; `__init__` does no i/o. the declared type is the `StateStore` protocol, so keep your own concretely-typed reference if you need backend-specific api like `SqlStateStore.aclose()`.

`runner.state` is the supported way in. `_state` is private, carries no semver protection, and may be renamed or removed in a patch release.

### streaming

a typed event stream the caller consumes while the loop runs. each step in the run surfaces as an event instead of waiting for a final blob.

### safety

the caps that bound a run: a max-iteration ceiling on react cycles and a per-tool timeout, plus the fallback answer returned when a run errors or hits the cap. a loop that can't end is a loop that doesn't ship.

### audit

the audit sinks and observability hooks. they record what the agent did, so a run can be read back after it finishes.

### mcp

an mcp client over streamable http, adapted into the same registry the in-proc tools live in. a `tools/call` that comes back `isError=True` is a recoverable observation the model can reason about, not a dead run — and `on_tool_error` is the seam for screening that server-controlled text before the model reads it.

```python
def screen(message: str, content: list[dict]) -> str:
    # `message` is the sdk's bounded default; `content` is the server's raw
    # error blocks (read-only). return the string the model should see.
    if any("PII" in str(block) for block in content):  # your own predicate
        return "the upstream tool failed"
    return message


client = MCPClient(MCPClientConfig(base_url=...), auth=..., on_tool_error=screen)
provider = MCPProvider(client)
await provider.attach(registry)
```

the hook may be sync or async, and it only ever fires on a per-call `isError` result — never on success, never on a transport failure (that still raises `MCPError`). if it raises, returns a non-string, or returns a blank string, the sdk falls back to its own bounded message and logs a warning; it can never change `is_error` or `output`.

## Architecture

```
fifty_agent_sdk  —  module graph (from src/fifty_agent_sdk/, ground-truth imports)

src/fifty_agent_sdk/
├─ ▢ audit
├─ errors
├─ ▢ llm
├─ loop
├─ ▢ mcp
├─ ▢ observability
├─ ▢ parser
├─ prompts
├─ ▶ runner
├─ safety
├─ ▢ state
├─ streaming
└─ ▢ tools

depends (→):
   audit → errors
   llm → errors
   loop → errors
   loop → llm
   loop → observability
   loop → parser
   loop → prompts
   loop → safety
   loop → streaming
   loop → tools
   mcp → errors
   observability → llm
   parser → errors
   parser → llm
   runner → audit
   runner → errors
   runner → llm
   runner → loop
   runner → observability
   runner → state
   runner → streaming
   state → errors
   state → llm
   streaming → tools
   tools → errors
   tools → llm
   tools → mcp

legend: ▶ entry   ▢ package   name module   → depends
```

## Highlights

- **branching** — first-class conversation branching on `StateStore`: `fork`, `list_branches`, `switch_branch`, branch-scoped `get_messages(..., branch_id=...)`, plus `BranchInfo` and `TRUNK_BRANCH_ID`. a session is now a tree of branches with an active head, and `append` writes to the active branch (the edit-a-message / regenerate model). implemented across memory, SQL, and Redis backends, data-additive and zero-migration: existing sessions read as the trunk branch. breaking for custom `StateStore` implementations: they must add the new methods.
- **`StateStore.truncate_after(session_id, sequence, *, branch_id=None)`** — a destructive hard-delete of a branch's tail (messages with sequence > N), for redaction, retention, and rollback. only the target branch's own messages are removed (a `fork`'s inherited prefix is never touched), and it is idempotent: a no-op on an unknown session or branch.

editing a turn is a consumer-side fork-then-append, and the original line stays reachable:

```python
# Edit a turn = fork the history before it, switch onto the new branch, then
# append the edited message. `store` is any StateStore; import `ChatMessage`
# from fifty_agent_sdk.
branch = await store.fork(session_id, from_sequence=4)  # keep messages 1..4
await store.switch_branch(session_id, branch)
await store.append(session_id, ChatMessage(role="user", content="...edited..."))
await store.get_messages(session_id, branch_id="trunk")  # original line intact
```

## Links

- [Homepage](https://github.com/fiftynotai/fifty-agent-sdk)
- [Repository](https://github.com/fiftynotai/fifty-agent-sdk)
- [Issues](https://github.com/fiftynotai/fifty-agent-sdk/issues)
- [Changelog](https://github.com/fiftynotai/fifty-agent-sdk/blob/main/CHANGELOG.md)

## License

MIT.
