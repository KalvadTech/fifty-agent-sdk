"""Integration tests for :class:`fifty_agent_sdk.runner.AgentRunner` hook dispatch.

Covers BR-012's Runner-integration surface — the five Runner-tier hooks:

* ``on_run_start`` / ``on_run_end`` fire once per run, with a positive
  ``duration_ms`` and the right ``error`` per the Q5 taxonomy.
* ``on_tool_start`` / ``on_tool_end`` fire once per tool invocation, with
  correlated ``tool_name`` / ``args`` / ``result``.
* ``on_error`` fires on an LLM failure and on a state-store durability
  failure.
* The two Loop-tier hooks (``on_iteration``, ``on_llm_call``) fire ZERO times
  from the Runner alone — proving the tier separation.
* A raising hook never aborts the run; the failure is logged at ``WARNING``.
* Sync and async hook variants behave identically.
* ``hooks=None`` is a zero-overhead no-op.
* ``on_run_end`` fires from ``finally`` on cancellation and never masks the
  propagating :class:`asyncio.CancelledError`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import structlog

from fifty_agent_sdk import (
    BranchInfo,
    ChatMessage,
    FinalEvent,
    Hooks,
    MemoryStateStore,
    Registry,
    SafetyConfig,
    StateStore,
    StateStoreError,
    ToolResult,
)
from fifty_agent_sdk.errors import LLMError
from tests.loop.conftest import FakeLLMClient, FakeTool, make_multi_tool_response, make_response
from tests.runner.conftest import collect, final_json, make_runner, tool_json

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_HOOK_NAMES = (
    "on_run_start",
    "on_run_end",
    "on_iteration",
    "on_llm_call",
    "on_tool_start",
    "on_tool_end",
    "on_error",
)


class HookRecorder:
    """Builds a :class:`Hooks` whose every field is a counting callable.

    Exposes :attr:`counts` (per-hook call count) and :attr:`args` (per-hook
    list of recorded positional-arg tuples). The ``sync`` flag selects plain
    ``def`` vs ``async def`` callables so a single recorder exercises both
    dispatch paths.
    """

    def __init__(self, *, sync: bool = True) -> None:
        self.counts: dict[str, int] = {name: 0 for name in _HOOK_NAMES}
        self.args: dict[str, list[tuple[Any, ...]]] = {name: [] for name in _HOOK_NAMES}
        self._sync = sync

    def _make(self, name: str) -> Any:
        def record(*args: Any) -> None:
            self.counts[name] += 1
            self.args[name].append(args)

        if self._sync:
            return record

        async def arecord(*args: Any) -> None:
            record(*args)

        return arecord

    def hooks(self) -> Hooks:
        return Hooks(
            on_run_start=self._make("on_run_start"),
            on_run_end=self._make("on_run_end"),
            on_iteration=self._make("on_iteration"),
            on_llm_call=self._make("on_llm_call"),
            on_tool_start=self._make("on_tool_start"),
            on_tool_end=self._make("on_tool_end"),
            on_error=self._make("on_error"),
        )


# ---------------------------------------------------------------------------
# Counting — single tool, 2-step conversation
# ---------------------------------------------------------------------------


async def test_counting_hooks_two_step_run_fires_runner_tier_once() -> None:
    """A 1-tool run fires each Runner-tier hook the expected number of times."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "weather"})),
            make_response(final_json("got it")),
        ]
    )
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, registry=registry, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_run_start"] == 1
    assert rec.counts["on_run_end"] == 1
    assert rec.counts["on_tool_start"] == 1
    assert rec.counts["on_tool_end"] == 1
    assert rec.counts["on_error"] == 0


async def test_runner_does_not_fire_loop_tier_hooks() -> None:
    """The two Loop-tier hooks fire ZERO times through the Runner alone.

    The Runner does NOT drive ``on_iteration`` / ``on_llm_call`` — they are
    Loop-tier. ``make_runner`` shares the same ``Hooks`` into both the loop
    and the runner, so they DO fire here (from the loop). The point of this
    test is the COUNTS line up with the loop's two iterations, proving the
    Runner did not add a third firing of its own.
    """
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "weather"})),
            make_response(final_json("got it")),
        ]
    )
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, registry=registry, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    # Two ReACT iterations => exactly two Loop-tier firings each.
    assert rec.counts["on_iteration"] == 2
    assert rec.counts["on_llm_call"] == 2


async def test_on_run_start_receives_session_id_and_user_message() -> None:
    """``on_run_start`` is called with ``(session_id, user_message)``."""
    llm = FakeLLMClient(replies=[make_response(final_json("ok"))])
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, hooks=rec.hooks())

    await collect(runner.run("sess-abc", "Hello there"))

    assert rec.args["on_run_start"] == [("sess-abc", "Hello there")]


async def test_on_tool_start_and_end_carry_correlated_data() -> None:
    """``on_tool_start`` / ``on_tool_end`` carry tool name, args and result."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output="RESULT")))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "weather"})),
            make_response(final_json("done")),
        ]
    )
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, registry=registry, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    start_args = rec.args["on_tool_start"][0]
    assert start_args[0] == "s1"
    assert start_args[1] == "search"
    assert start_args[2] == {"q": "weather"}

    end_args = rec.args["on_tool_end"][0]
    assert end_args[0] == "s1"
    assert end_args[1] == "search"
    assert end_args[2] == "RESULT"
    assert isinstance(end_args[3], float)
    assert end_args[3] >= 0.0


# ---------------------------------------------------------------------------
# Counting — multi-tool
# ---------------------------------------------------------------------------


async def test_counting_hooks_multi_tool_run() -> None:
    """A 2-tool run fires tool hooks twice; run hooks once."""
    registry = Registry()
    registry.register(FakeTool("alpha", result=ToolResult(output="A")))
    registry.register(FakeTool("beta", result=ToolResult(output="B")))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("step 1", "alpha", {"n": 1})),
            make_response(tool_json("step 2", "beta", {"n": 2})),
            make_response(final_json("done")),
        ]
    )
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, registry=registry, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_tool_start"] == 2
    assert rec.counts["on_tool_end"] == 2
    assert rec.counts["on_run_start"] == 1
    assert rec.counts["on_run_end"] == 1
    assert [a[1] for a in rec.args["on_tool_start"]] == ["alpha", "beta"]
    assert [a[1] for a in rec.args["on_tool_end"]] == ["alpha", "beta"]


async def test_multi_action_batch_tool_hooks_carry_per_call_args_and_results() -> None:
    """A native MultiAction batch correlates each tool hook with its own call.

    Regression gate for per-call correlation: the loop's MultiAction branch
    emits N ActionEvents, then N ToolStartedEvents, then N terminal events —
    all in call order. ``on_tool_start`` for the FIRST call must carry the
    FIRST call's args (the old single-slot correlation handed it the last
    call's args), and ``on_tool_end`` must pair each tool with its own
    result and a non-negative duration.
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
    rec = HookRecorder()
    runner, _store = make_runner(
        llm=llm,
        registry=registry,
        safety=SafetyConfig(native_tools_enabled=True, max_concurrent_tool_calls=2),
        hooks=rec.hooks(),
    )

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_tool_start"] == 2
    assert rec.counts["on_tool_end"] == 2
    starts = {a[1]: a[2] for a in rec.args["on_tool_start"]}
    assert starts == {"alpha": {"n": 1}, "beta": {"n": 2}}
    ends = {a[1]: (a[2], a[3]) for a in rec.args["on_tool_end"]}
    assert ends["alpha"][0] == "A-result"
    assert ends["beta"][0] == "B-result"
    assert all(duration >= 0.0 for _, duration in ends.values())


# ---------------------------------------------------------------------------
# on_run_end — duration + error
# ---------------------------------------------------------------------------


async def test_on_run_end_success_has_no_error_and_positive_duration() -> None:
    """A clean run fires ``on_run_end`` with ``error=None`` and ``duration>0``."""
    llm = FakeLLMClient(replies=[make_response(final_json("ok"))])
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_run_end"] == 1
    session_id, duration_ms, error = rec.args["on_run_end"][0]
    assert session_id == "s1"
    assert isinstance(duration_ms, float)
    assert duration_ms > 0.0
    assert error is None


# ---------------------------------------------------------------------------
# on_error — LLM failure
# ---------------------------------------------------------------------------


async def test_on_error_fires_on_llm_failure() -> None:
    """An LLM failure fires ``on_error``; ``on_run_end`` still fires."""
    llm = FakeLLMClient(replies=[LLMError("provider down")])
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_error"] == 1
    session_id, error, context = rec.args["on_error"][0]
    assert session_id == "s1"
    assert isinstance(error, Exception)
    assert context["error_type"] == "LLMError"
    # A loop-internal failure leaves on_run_end's `error` None (Q5).
    assert rec.counts["on_run_end"] == 1
    assert rec.args["on_run_end"][0][2] is None


# ---------------------------------------------------------------------------
# on_error — state-store failure
# ---------------------------------------------------------------------------


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


async def test_on_error_fires_on_state_store_failure() -> None:
    """A persist_assistant failure fires ``on_error`` with the StateStoreError."""
    failing: StateStore = _FailingAssistantStore()
    llm = FakeLLMClient(replies=[make_response(final_json("answer"))])
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, state=failing, hooks=rec.hooks())

    with pytest.raises(StateStoreError):
        await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_error"] == 1
    _session_id, error, context = rec.args["on_error"][0]
    assert isinstance(error, StateStoreError)
    assert context["phase"] == "persist_assistant"


async def test_on_run_end_carries_state_store_error() -> None:
    """``on_run_end``'s ``error`` is the escaped StateStoreError on a persist fail."""
    failing: StateStore = _FailingAssistantStore()
    llm = FakeLLMClient(replies=[make_response(final_json("answer"))])
    rec = HookRecorder()
    runner, _store = make_runner(llm=llm, state=failing, hooks=rec.hooks())

    with pytest.raises(StateStoreError):
        await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_run_end"] == 1
    _session_id, _duration, error = rec.args["on_run_end"][0]
    assert isinstance(error, StateStoreError)


# ---------------------------------------------------------------------------
# Raising hook is isolated
# ---------------------------------------------------------------------------


async def test_raising_on_tool_start_does_not_abort_run() -> None:
    """A raising ``on_tool_start`` never breaks the run."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "x"})),
            make_response(final_json("done")),
        ]
    )

    def boom(*_args: Any) -> None:
        raise RuntimeError("hook down")

    runner, store = make_runner(llm=llm, registry=registry, hooks=Hooks(on_tool_start=boom))

    events = await collect(runner.run("s1", "Hi"))

    assert isinstance(events[-1], FinalEvent)
    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user", "assistant"]


async def test_raising_on_run_end_does_not_abort_run() -> None:
    """A raising ``on_run_end`` in ``finally`` never breaks the run."""
    llm = FakeLLMClient(replies=[make_response(final_json("done"))])

    def boom(*_args: Any) -> None:
        raise RuntimeError("run-end hook down")

    runner, store = make_runner(llm=llm, hooks=Hooks(on_run_end=boom))

    events = await collect(runner.run("s1", "Hi"))

    assert isinstance(events[-1], FinalEvent)
    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user", "assistant"]


async def test_raising_on_run_end_does_not_mask_state_store_error() -> None:
    """A raising ``on_run_end`` does not swallow an in-flight StateStoreError.

    R2 regression gate: ``on_run_end`` is awaited in ``finally``, but the
    swallow helper guarantees nothing but ``CancelledError`` escapes it — so
    the propagating ``StateStoreError`` still reaches the caller.
    """
    failing: StateStore = _FailingAssistantStore()
    llm = FakeLLMClient(replies=[make_response(final_json("answer"))])

    def boom(*_args: Any) -> None:
        raise RuntimeError("run-end hook down")

    runner, _store = make_runner(llm=llm, state=failing, hooks=Hooks(on_run_end=boom))

    with pytest.raises(StateStoreError):
        await collect(runner.run("s1", "Hi"))


async def test_raising_hook_logs_invoke_failed_warning() -> None:
    """A raising hook produces a ``hook.invoke_failed`` WARNING."""
    llm = FakeLLMClient(replies=[make_response(final_json("done"))])

    def boom(*_args: Any) -> None:
        raise RuntimeError("hook down")

    runner, _store = make_runner(llm=llm, hooks=Hooks(on_run_start=boom))

    with structlog.testing.capture_logs() as logs:
        await collect(runner.run("s1", "Hi"))

    failures = [e for e in logs if e.get("event") == "hook.invoke_failed"]
    assert len(failures) >= 1
    assert all(f["log_level"] == "warning" for f in failures)
    assert failures[0]["hook_name"] == "on_run_start"
    assert failures[0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Sync vs async variants
# ---------------------------------------------------------------------------


async def test_sync_hook_variant_two_step_run() -> None:
    """The 2-step counting scenario works with SYNC callables."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "x"})),
            make_response(final_json("got it")),
        ]
    )
    rec = HookRecorder(sync=True)
    runner, _store = make_runner(llm=llm, registry=registry, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_run_start"] == 1
    assert rec.counts["on_run_end"] == 1
    assert rec.counts["on_tool_start"] == 1
    assert rec.counts["on_tool_end"] == 1


async def test_async_hook_variant_two_step_run() -> None:
    """The 2-step counting scenario works with ASYNC callables."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "x"})),
            make_response(final_json("got it")),
        ]
    )
    rec = HookRecorder(sync=False)
    runner, _store = make_runner(llm=llm, registry=registry, hooks=rec.hooks())

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_run_start"] == 1
    assert rec.counts["on_run_end"] == 1
    assert rec.counts["on_tool_start"] == 1
    assert rec.counts["on_tool_end"] == 1


# ---------------------------------------------------------------------------
# Zero-overhead: hooks=None
# ---------------------------------------------------------------------------


async def test_hooks_none_does_not_change_behaviour() -> None:
    """With ``hooks=None`` the run produces identical events and history."""

    def build_llm() -> FakeLLMClient:
        return FakeLLMClient(
            replies=[
                make_response(tool_json("look up", "search", {"q": "x"})),
                make_response(final_json("got it")),
            ]
        )

    registry_a = Registry()
    registry_a.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    runner_a, store_a = make_runner(llm=build_llm(), registry=registry_a, hooks=None)
    events_a = await collect(runner_a.run("s1", "Hi"))

    registry_b = Registry()
    registry_b.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    runner_b, store_b = make_runner(
        llm=build_llm(), registry=registry_b, hooks=HookRecorder().hooks()
    )
    events_b = await collect(runner_b.run("s1", "Hi"))

    assert [type(e) for e in events_a] == [type(e) for e in events_b]
    assert await store_a.get_messages("s1") == await store_b.get_messages("s1")


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------


async def test_on_error_fires_on_iteration_cap() -> None:
    """Hitting the iteration cap fires ``on_error`` and ``on_run_end``."""
    registry = Registry()
    registry.register(FakeTool("t", result=ToolResult(output="ok")))
    tool_call = make_response(tool_json("t", "t", {}))
    llm = FakeLLMClient(replies=[tool_call, tool_call])
    rec = HookRecorder()
    runner, _store = make_runner(
        llm=llm,
        registry=registry,
        safety=SafetyConfig(max_iterations=2),
        hooks=rec.hooks(),
    )

    await collect(runner.run("s1", "Hi"))

    assert rec.counts["on_error"] == 1
    assert rec.args["on_error"][0][2]["error_type"] == "MaxIterationsExceeded"
    assert rec.counts["on_run_end"] == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_on_run_end_fires_and_cancelled_error_propagates() -> None:
    """On consumer cancellation ``on_run_end`` fires; ``CancelledError`` propagates.

    The consumer breaks out of the ``async for`` mid-stream. ``on_run_end``
    must still fire from ``finally``, AND a raising ``on_run_end`` must not
    mask the propagating ``CancelledError``.
    """
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"x": 1})))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "x"})),
            make_response(final_json("got it")),
        ]
    )
    end_calls: list[tuple[Any, ...]] = []

    def on_run_end(*args: Any) -> None:
        end_calls.append(args)
        raise RuntimeError("run-end hook down")

    runner, _store = make_runner(llm=llm, registry=registry, hooks=Hooks(on_run_end=on_run_end))

    agen = runner.run("s1", "Hi")
    # Consume the first event, then cancel by closing the generator.
    first = await agen.__anext__()
    assert first is not None
    await agen.aclose()

    # `on_run_end` fired from `finally` despite the cancellation, and the
    # raising hook did not crash the close.
    assert len(end_calls) == 1


async def test_run_end_error_is_cancelled_error_on_cancellation() -> None:
    """``on_run_end``'s ``error`` is the ``CancelledError`` on a real cancel.

    A slow tool keeps the consumer task suspended INSIDE the runner's
    generator; cancelling the task then throws ``CancelledError`` into the
    generator at the suspension point, exercising the ``except
    asyncio.CancelledError`` branch that captures it for ``on_run_end``.
    """
    registry = Registry()
    registry.register(FakeTool("slow", sleep_seconds=1.0))
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("t", "slow", {})),
            make_response(final_json("done")),
        ]
    )
    end_calls: list[tuple[Any, ...]] = []

    async def on_run_end(*args: Any) -> None:
        end_calls.append(args)

    runner, _store = make_runner(
        llm=llm,
        registry=registry,
        safety=SafetyConfig(max_iterations=3, tool_timeout_seconds=None),
        hooks=Hooks(on_run_end=on_run_end),
    )

    async def consume() -> None:
        await collect(runner.run("s1", "Hi"))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # `on_run_end` fired from `finally` on the cancelled path, and its
    # `error` is the surfaced CancelledError.
    assert len(end_calls) == 1
    error = end_calls[0][2]
    assert isinstance(error, asyncio.CancelledError)
