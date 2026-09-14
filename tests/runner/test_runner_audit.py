"""Integration tests for :class:`fifty_agent_sdk.runner.AgentRunner` audit emission.

Covers BR-011's Runner-integration surface:

* The four audit events (``session_start``, ``tool_invocation``,
  ``final_answer``, ``error``) fire at the right points and in order.
* ``tool_invocation`` correlation carries ``tool_name``, ``call_id``,
  ``args`` and the correct ``outcome``.
* A recoverable tool failure still yields a ``final_answer``.
* An LLM-error run emits ``session_start`` then ``error`` — no
  ``final_answer``.
* ``audit=None`` is a zero-overhead no-op.
* A raising sink never aborts the run; the failure is logged at ``WARNING``.
* ``result_summary`` is bounded with a truncation marker.
"""

from __future__ import annotations

import structlog

from fifty_agent_sdk import (
    ActionEvent,
    AuditEvent,
    AuditSink,
    Registry,
    SafetyConfig,
    ToolResult,
    ToolStartedEvent,
)
from tests.loop.conftest import FakeLLMClient, FakeTool, make_multi_tool_response, make_response
from tests.runner.conftest import collect, final_json, make_runner, tool_json

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class SpyAuditSink:
    """An :class:`AuditSink` that records every event into a list."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class RaisingAuditSink:
    """An :class:`AuditSink` whose ``record`` always raises."""

    def __init__(self) -> None:
        self.calls = 0

    async def record(self, event: AuditEvent) -> None:
        self.calls += 1
        raise RuntimeError("audit backend down")


# ---------------------------------------------------------------------------
# Happy path — single tool
# ---------------------------------------------------------------------------


async def test_single_tool_run_emits_ordered_audit_events() -> None:
    """A 1-tool run emits session_start, tool_invocation, final_answer in order."""
    tool = FakeTool("search", result=ToolResult(output={"x": 1}))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "weather"})),
            make_response(final_json("got it")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, audit=spy)

    await collect(runner.run("s1", "Hi"))

    assert [e.event_type for e in spy.events] == [
        "session_start",
        "tool_invocation",
        "final_answer",
    ]


async def test_tool_invocation_payload_carries_correlation_fields() -> None:
    """The ``tool_invocation`` event carries tool_name, call_id, args, outcome."""
    tool = FakeTool("search", result=ToolResult(output={"x": 1}))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "weather"})),
            make_response(final_json("got it")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, audit=spy)

    await collect(runner.run("s1", "Hi"))

    tool_event = next(e for e in spy.events if e.event_type == "tool_invocation")
    assert tool_event.payload["tool_name"] == "search"
    # Args are reduced to non-content metadata: sorted keys, per-value type
    # and length — never the values themselves.
    assert tool_event.payload["args"] == {"q": {"type": "str", "len": len("weather")}}
    assert tool_event.payload["outcome"] == "ok"
    assert isinstance(tool_event.payload["call_id"], str)
    assert tool_event.payload["call_id"] != ""
    assert "result_summary" in tool_event.payload


async def test_session_start_payload_fields() -> None:
    """The ``session_start`` event carries run_id, is_first_turn, lengths."""
    llm = FakeLLMClient(replies=[make_response(final_json("ok"))])
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, audit=spy)

    await collect(runner.run("s1", "Hello"))

    start = next(e for e in spy.events if e.event_type == "session_start")
    assert start.session_id == "s1"
    assert start.payload["is_first_turn"] is True
    assert start.payload["has_system_prompt"] is False
    assert start.payload["user_message_len"] == len("Hello")
    assert isinstance(start.payload["run_id"], str)


async def test_final_answer_payload_carries_lengths_only() -> None:
    """The ``final_answer`` event carries lengths/counts, never the answer text."""
    llm = FakeLLMClient(replies=[make_response(final_json("the answer text"))])
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, audit=spy)

    await collect(runner.run("s1", "Hi"))

    final = next(e for e in spy.events if e.event_type == "final_answer")
    assert final.payload["final_text_len"] == len("the answer text")
    assert "event_count" in final.payload
    # The actual answer text must not be present anywhere in the payload.
    assert "the answer text" not in str(final.payload)


# ---------------------------------------------------------------------------
# Multi-tool
# ---------------------------------------------------------------------------


async def test_multi_tool_run_emits_one_event_per_tool() -> None:
    """A 2-tool run emits one tool_invocation per tool, correctly ordered."""
    tool_a = FakeTool("alpha", result=ToolResult(output="A-result"))
    tool_b = FakeTool("beta", result=ToolResult(output="B-result"))
    registry = Registry()
    registry.register(tool_a)
    registry.register(tool_b)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("step 1", "alpha", {"n": 1})),
            make_response(tool_json("step 2", "beta", {"n": 2})),
            make_response(final_json("done")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, audit=spy)

    await collect(runner.run("s1", "Hi"))

    assert [e.event_type for e in spy.events] == [
        "session_start",
        "tool_invocation",
        "tool_invocation",
        "final_answer",
    ]
    tool_events = [e for e in spy.events if e.event_type == "tool_invocation"]
    assert tool_events[0].payload["tool_name"] == "alpha"
    assert tool_events[0].payload["args"] == {"n": {"type": "int", "len": None}}
    assert tool_events[1].payload["tool_name"] == "beta"
    assert tool_events[1].payload["args"] == {"n": {"type": "int", "len": None}}


async def test_multi_action_batch_audits_each_call_with_own_args_and_call_id() -> None:
    """A native MultiAction batch emits one tool_invocation per call, each with
    its OWN args and call_id.

    Regression gate for per-call correlation: the loop's MultiAction branch
    emits N ActionEvents, then N ToolStartedEvents, then N terminal events —
    all in call order. The old single-slot correlation gave the FIRST
    terminal event the LAST call's call_id/args and left every other call
    with ``args={}``. With per-call-id correlation each invocation's payload
    carries its own pair.
    """
    registry = Registry()
    registry.register(FakeTool("alpha", result=ToolResult(output="A-result")))
    registry.register(FakeTool("beta", result=ToolResult(output="B-result")))
    llm = FakeLLMClient(
        replies=[
            make_multi_tool_response([("alpha", {"n": 1}), ("beta", {"n": 2})]),
            make_response(final_json("done")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(
        llm=llm,
        registry=registry,
        safety=SafetyConfig(native_tools_enabled=True, max_concurrent_tool_calls=2),
        audit=spy,
    )

    events = await collect(runner.run("s1", "Hi"))

    # Sanity: this run really went through the MultiAction batch shape — two
    # ActionEvents followed by two ToolStartedEvents.
    actions = [e for e in events if isinstance(e, ActionEvent)]
    started = [e for e in events if isinstance(e, ToolStartedEvent)]
    assert len(actions) == len(started) == 2

    call_id_by_tool = {e.tool_name: e.call_id for e in started}
    tool_events = [e for e in spy.events if e.event_type == "tool_invocation"]
    assert len(tool_events) == 2
    by_tool = {e.payload["tool_name"]: e.payload for e in tool_events}
    # Correlation is per call; args in the payload are the non-content
    # metadata summary (see _args_metadata), not the raw values.
    assert by_tool["alpha"]["args"] == {"n": {"type": "int", "len": None}}
    assert by_tool["alpha"]["call_id"] == call_id_by_tool["alpha"]
    assert by_tool["beta"]["args"] == {"n": {"type": "int", "len": None}}
    assert by_tool["beta"]["call_id"] == call_id_by_tool["beta"]


# ---------------------------------------------------------------------------
# Args redaction — payload never carries argument values
# ---------------------------------------------------------------------------


async def test_tool_invocation_args_never_leak_secret_values() -> None:
    """A secret-looking arg VALUE never appears in the tool_invocation payload.

    Mirrors the ``"SECRET" not in json.dumps(...)`` redaction-proof pattern
    of the MCP auth tests: argument values routinely carry credentials and
    PII, so the payload carries only keys, per-value type names and lengths.
    """
    import json

    secret = "SECRET-api-key-DO-NOT-LEAK"
    registry = Registry()
    registry.register(FakeTool("auth_call", result=ToolResult(output="ok")))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("t", "auth_call", {"token": secret, "retries": 3})),
            make_response(final_json("done")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, audit=spy)

    await collect(runner.run("s1", "Hi"))

    tool_event = next(e for e in spy.events if e.event_type == "tool_invocation")
    assert secret not in json.dumps(tool_event.payload, default=str)
    assert tool_event.payload["args"] == {
        "retries": {"type": "int", "len": None},
        "token": {"type": "str", "len": len(secret)},
    }


# ---------------------------------------------------------------------------
# Recoverable tool failure
# ---------------------------------------------------------------------------


async def test_recoverable_tool_failure_marks_outcome_failed() -> None:
    """A ToolResult(is_error=True) yields outcome='failed'; final_answer still fires."""
    tool = FakeTool(
        "broken",
        result=ToolResult(output=None, is_error=True, error="boom"),
    )
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("try it", "broken", {})),
            make_response(final_json("recovered")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, audit=spy)

    await collect(runner.run("s1", "Hi"))

    assert [e.event_type for e in spy.events] == [
        "session_start",
        "tool_invocation",
        "final_answer",
    ]
    tool_event = next(e for e in spy.events if e.event_type == "tool_invocation")
    assert tool_event.payload["outcome"] == "failed"
    assert "boom" in tool_event.payload["result_summary"]


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


async def test_llm_error_run_emits_session_start_then_error() -> None:
    """An LLM-error run audits session_start + error, never final_answer."""
    from fifty_agent_sdk.errors import LLMError

    llm = FakeLLMClient(replies=[LLMError("provider down")])
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, audit=spy)

    await collect(runner.run("s1", "Hi"))

    assert [e.event_type for e in spy.events] == ["session_start", "error"]
    error_event = next(e for e in spy.events if e.event_type == "error")
    assert error_event.payload["error_type"] == "LLMError"
    assert "run_id" in error_event.payload


async def test_fatal_sdk_error_escaping_loop_is_audited() -> None:
    """An MCPError escaping the loop emits an error audit event, terminates
    with ``terminated_by="sdk_error"``, and still re-raises to the caller.

    The registry re-raises non-recoverable :class:`AgentSdkError` subclasses
    untouched, so an MCP transport failure propagates out of the loop as an
    exception rather than as an ErrorEvent. Before this fix such a run exited
    invisibly: no error audit event and ``terminated_by="interrupted"``.
    """
    import pytest

    from fifty_agent_sdk.errors import MCPError

    registry = Registry()
    registry.register(
        FakeTool(
            "mcp_tool",
            raises=MCPError("transport down", context={"server_url": "https://mcp.example.com"}),
        )
    )
    llm = FakeLLMClient(replies=[make_response(tool_json("t", "mcp_tool", {}))])
    spy = SpyAuditSink()
    runner, store = make_runner(llm=llm, registry=registry, audit=spy)

    with structlog.testing.capture_logs() as logs, pytest.raises(MCPError):
        await collect(runner.run("s1", "Hi"))

    # The escaped SDK error is audited like any other error.
    assert [e.event_type for e in spy.events] == ["session_start", "error"]
    error_event = spy.events[-1]
    assert error_event.payload["error_type"] == "MCPError"
    assert "transport down" in error_event.payload["error_message"]

    # The run_completed log attributes the exit to the escaped SDK error.
    completed = [e for e in logs if e.get("event") == "runner.run_completed"]
    assert len(completed) == 1
    assert completed[0]["terminated_by"] == "sdk_error"

    # No assistant message was committed; the durable user message survives.
    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user"]


async def test_state_store_error_on_assistant_persist_is_audited() -> None:
    """A persist_assistant durability failure emits an error audit event."""
    from fifty_agent_sdk import (
        BranchInfo,
        ChatMessage,
        MemoryStateStore,
        StateStore,
        StateStoreError,
    )

    class _FailingAssistantStore:
        """Delegates to memory, but fails the 2nd append (assistant persist)."""

        def __init__(self) -> None:
            self._inner = MemoryStateStore()
            self.append_calls = 0

        async def get_messages(
            self, session_id: str, *, branch_id: str | None = None
        ) -> list[ChatMessage]:
            return await self._inner.get_messages(session_id, branch_id=branch_id)

        async def append(self, session_id: str, message: ChatMessage) -> None:
            self.append_calls += 1
            if self.append_calls == 2:
                raise StateStoreError(
                    "assistant persist down",
                    context={"session_id": session_id},
                )
            await self._inner.append(session_id, message)

        async def delete(self, session_id: str) -> None:
            await self._inner.delete(session_id)

        async def fork(self, session_id: str, from_sequence: int) -> str:
            return await self._inner.fork(session_id, from_sequence)

        async def list_branches(self, session_id: str) -> list[BranchInfo]:
            return await self._inner.list_branches(session_id)

        async def switch_branch(self, session_id: str, branch_id: str) -> None:
            await self._inner.switch_branch(session_id, branch_id)

        async def truncate_after(
            self, session_id: str, sequence: int, *, branch_id: str | None = None
        ) -> None:
            await self._inner.truncate_after(session_id, sequence, branch_id=branch_id)

    failing: StateStore = _FailingAssistantStore()
    llm = FakeLLMClient(replies=[make_response(final_json("answer"))])
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, state=failing, audit=spy)

    import pytest

    with pytest.raises(StateStoreError):
        await collect(runner.run("s1", "Hi"))

    assert [e.event_type for e in spy.events] == ["session_start", "error"]
    error_event = next(e for e in spy.events if e.event_type == "error")
    assert error_event.payload["phase"] == "persist_assistant"
    assert error_event.payload["error_type"] == "StateStoreError"


# ---------------------------------------------------------------------------
# Zero-overhead: audit=None
# ---------------------------------------------------------------------------


async def test_audit_none_does_not_change_behaviour() -> None:
    """With ``audit=None`` the run produces identical events and history."""
    tool = FakeTool("search", result=ToolResult(output={"x": 1}))

    def build_run() -> FakeLLMClient:
        return FakeLLMClient(
            replies=[
                make_response(tool_json("look up", "search", {"q": "x"})),
                make_response(final_json("got it")),
            ]
        )

    # Run once with no audit.
    registry_a = Registry()
    registry_a.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    runner_a, store_a = make_runner(llm=build_run(), registry=registry_a, audit=None)
    events_a = await collect(runner_a.run("s1", "Hi"))

    # Run again with a spy sink — the agent-visible behaviour is unchanged.
    registry_b = Registry()
    registry_b.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    runner_b, store_b = make_runner(llm=build_run(), registry=registry_b, audit=SpyAuditSink())
    events_b = await collect(runner_b.run("s1", "Hi"))
    _ = tool  # silence unused — registries build their own fakes

    assert [type(e) for e in events_a] == [type(e) for e in events_b]
    assert await store_a.get_messages("s1") == await store_b.get_messages("s1")


# ---------------------------------------------------------------------------
# Raising sink is isolated
# ---------------------------------------------------------------------------


async def test_raising_sink_does_not_abort_run() -> None:
    """A sink that raises on every record never breaks the run."""
    llm = FakeLLMClient(replies=[make_response(final_json("hello!"))])
    raising = RaisingAuditSink()
    runner, store = make_runner(llm=llm, audit=raising)

    events = await collect(runner.run("s1", "Hi"))

    # The run completed normally despite the raising sink.
    from fifty_agent_sdk import FinalEvent

    assert isinstance(events[-1], FinalEvent)
    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user", "assistant"]
    # Both session_start and final_answer emission attempts hit the sink.
    assert raising.calls >= 2


async def test_raising_sink_logs_emit_failed_warning() -> None:
    """A raising sink produces an ``audit.emit_failed`` WARNING per attempt."""
    llm = FakeLLMClient(replies=[make_response(final_json("hello!"))])
    runner, _store = make_runner(llm=llm, audit=RaisingAuditSink())

    with structlog.testing.capture_logs() as logs:
        await collect(runner.run("s1", "Hi"))

    failures = [e for e in logs if e.get("event") == "audit.emit_failed"]
    assert len(failures) >= 1
    assert all(f["log_level"] == "warning" for f in failures)
    assert failures[0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# result_summary truncation
# ---------------------------------------------------------------------------


async def test_large_tool_output_is_truncated_in_result_summary() -> None:
    """A very large tool output yields a capped ``result_summary`` with a marker."""
    huge = "x" * 10_000
    tool = FakeTool("big", result=ToolResult(output=huge))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("fetch", "big", {})),
            make_response(final_json("done")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, audit=spy)

    await collect(runner.run("s1", "Hi"))

    tool_event = next(e for e in spy.events if e.event_type == "tool_invocation")
    summary = tool_event.payload["result_summary"]
    assert summary.endswith("…[truncated]")
    # Cap is 500 chars plus the marker.
    assert len(summary) <= 500 + len("…[truncated]")


# ---------------------------------------------------------------------------
# Protocol smoke / multi-turn
# ---------------------------------------------------------------------------


async def test_spy_sink_satisfies_audit_sink_protocol() -> None:
    """The test double is a structurally-valid :class:`AuditSink`."""
    assert isinstance(SpyAuditSink(), AuditSink)


async def test_second_turn_session_start_is_not_first_turn() -> None:
    """On the second run() the session_start event reports is_first_turn=False."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("a")),
            make_response(final_json("b")),
        ]
    )
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, audit=spy)

    await collect(runner.run("s1", "turn one"))
    await collect(runner.run("s1", "turn two"))

    starts = [e for e in spy.events if e.event_type == "session_start"]
    assert len(starts) == 2
    assert starts[0].payload["is_first_turn"] is True
    assert starts[1].payload["is_first_turn"] is False


async def test_iteration_cap_error_is_audited() -> None:
    """Hitting the iteration cap emits an error audit event, no final_answer."""
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=2)
    tool_call = make_response(tool_json("t", "t", {}))
    llm = FakeLLMClient(replies=[tool_call, tool_call])
    spy = SpyAuditSink()
    runner, _store = make_runner(llm=llm, registry=registry, safety=safety, audit=spy)

    await collect(runner.run("s1", "Hi"))

    assert spy.events[0].event_type == "session_start"
    assert spy.events[-1].event_type == "error"
    error_event = spy.events[-1]
    assert error_event.payload["error_type"] == "MaxIterationsExceeded"
    assert all(e.event_type != "final_answer" for e in spy.events)
