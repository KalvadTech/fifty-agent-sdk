"""Integration tests for :class:`fifty_agent_sdk.runner.AgentRunner`.

Covers the full surface from the brief:

* Round-trip across multiple ``run()`` calls preserves history.
* First-turn ``system_prompt`` is persisted exactly once.
* Events forwarded in the right order with no Runner-injected events.
* Transactional persistence under LLM error, parser error, iteration cap,
  consumer cancellation, and state-store failures.
* Recoverable tool failures still trigger success persistence.
* Tool roundtrips are NOT persisted to state.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from fifty_agent_sdk import (
    ActionEvent,
    AgentEvent,
    AgentLoop,
    AgentRunner,
    BranchInfo,
    ChatMessage,
    ErrorEvent,
    FinalEvent,
    JsonModeParser,
    MemoryStateStore,
    ObservationEvent,
    PromptSections,
    Registry,
    SafetyConfig,
    StateStore,
    StateStoreError,
    ThoughtEvent,
    ToolFailedEvent,
    ToolResult,
    ToolStartedEvent,
)
from tests.loop.conftest import FakeLLMClient, FakeTool, make_response
from tests.runner.conftest import (
    FormatAwareFakeLLM,
    collect,
    final_json,
    make_runner,
    tool_json,
)

# ---------------------------------------------------------------------------
# Round-trip across multiple turns
# ---------------------------------------------------------------------------


async def test_two_turn_conversation_preserves_history() -> None:
    """Two successive ``run()`` calls leave exactly user/assistant pairs in history."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("hi back")),
            make_response(final_json("more details")),
        ]
    )
    runner, store = make_runner(llm=llm)

    await collect(runner.run("s1", "Hello"))
    await collect(runner.run("s1", "Tell me more"))

    history = await store.get_messages("s1")
    assert [m.role for m in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history[0].content == "Hello"
    # BR-016: assistant turns persist the raw LLM completion (the JSON
    # envelope), NOT the parsed answer text — so multi-turn sessions
    # feed the same structured shape back to the provider on turn 2+.
    assert history[1].content == final_json("hi back")
    assert history[2].content == "Tell me more"
    assert history[3].content == final_json("more details")


async def test_second_run_sees_prior_history_in_loop_request() -> None:
    """The second LLM call receives the prior user+assistant messages."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("first")),
            make_response(final_json("second")),
        ]
    )
    runner, _store = make_runner(llm=llm)

    await collect(runner.run("s1", "msg1"))
    await collect(runner.run("s1", "msg2"))

    # Second call: messages should be [<loop system>, user1, assistant1, user2].
    second_request = llm.calls[1]
    roles = [m.role for m in second_request.messages]
    assert roles == ["system", "user", "assistant", "user"]
    contents = [m.content for m in second_request.messages]
    assert contents[1] == "msg1"
    # BR-016: the prior assistant turn is the raw JSON envelope, not the
    # parsed answer — that's the whole point of persisting raw_completion.
    assert contents[2] == final_json("first")
    assert contents[3] == "msg2"


async def test_different_sessions_have_isolated_history() -> None:
    """Two sessions on the same runner do not contaminate each other."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("ans1")),
            make_response(final_json("ans2")),
        ]
    )
    runner, store = make_runner(llm=llm)

    await collect(runner.run("alice", "alice-msg"))
    await collect(runner.run("bob", "bob-msg"))

    alice = await store.get_messages("alice")
    bob = await store.get_messages("bob")
    assert [m.content for m in alice] == ["alice-msg", final_json("ans1")]
    assert [m.content for m in bob] == ["bob-msg", final_json("ans2")]


async def test_multi_turn_persists_raw_envelope_so_parser_succeeds() -> None:
    """BR-016 regression: turn 2+ must see the prior turn's raw envelope.

    Locks the live-reproduced GDC Gemini failure: if the runner persisted
    only the parsed ``final_text`` ("hi back"), the provider's
    format-detector would drift on turn 2 and return prose, and
    :class:`fifty_agent_sdk.parser.json_mode.JsonModeParser` would raise a
    :class:`fifty_agent_sdk.errors.ParserError`. The fix persists the raw JSON
    envelope so the provider stays in format and turn 2 parses cleanly.
    """
    llm = FormatAwareFakeLLM(
        json_reply=final_json("hi back"),
        prose_reply="this is prose, no envelope",
    )
    runner, store = make_runner(llm=llm)

    events_1 = await collect(runner.run("s1", "Hello"))
    events_2 = await collect(runner.run("s1", "Tell me more"))

    assert not any(isinstance(e, ErrorEvent) for e in events_1)
    history = await store.get_messages("s1")
    assert history[1].role == "assistant"
    # The raw envelope — NOT the parsed "hi back" — is what hits state.
    assert history[1].content == final_json("hi back")

    # The bug-locking assertion: turn 2 must not error.
    assert not any(isinstance(e, ErrorEvent) for e in events_2)
    finals = [e for e in events_2 if isinstance(e, FinalEvent)]
    assert len(finals) == 1
    assert finals[0].text == "hi back"


# ---------------------------------------------------------------------------
# system_prompt semantics
# ---------------------------------------------------------------------------


async def test_first_turn_system_prompt_persists_exactly_once() -> None:
    """The kickoff system message lands in state on turn 1 and never again."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("a")),
            make_response(final_json("b")),
        ]
    )
    runner, store = make_runner(llm=llm, system_prompt="You are pirate.")

    await collect(runner.run("s1", "Hi"))
    await collect(runner.run("s1", "Again"))

    history = await store.get_messages("s1")
    assert [m.role for m in history] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history[0].content == "You are pirate."
    # Only one system message.
    assert sum(1 for m in history if m.role == "system") == 1


async def test_no_system_prompt_no_system_message_in_state() -> None:
    """When ``system_prompt`` is ``None`` no role="system" message is persisted."""
    llm = FakeLLMClient(replies=[make_response(final_json("ok"))])
    runner, store = make_runner(llm=llm, system_prompt=None)

    await collect(runner.run("s1", "Hi"))

    history = await store.get_messages("s1")
    assert all(m.role != "system" for m in history)


async def test_system_prompt_with_preexisting_non_system_history_does_not_insert() -> None:
    """If the session already has messages, ``system_prompt`` is NOT retroactively inserted."""
    store = MemoryStateStore()
    # Pre-seed a session with non-system content.
    await store.append("s1", ChatMessage(role="user", content="prior"))
    await store.append("s1", ChatMessage(role="assistant", content="prior-ans"))

    llm = FakeLLMClient(replies=[make_response(final_json("new"))])
    runner, _ = make_runner(llm=llm, state=store, system_prompt="You are pirate.")
    await collect(runner.run("s1", "new question"))

    history = await store.get_messages("s1")
    # No retroactive system insertion.
    assert all(m.role != "system" for m in history)
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]


async def test_system_prompt_appears_in_loop_request_after_first_turn() -> None:
    """The persisted system message reaches the loop on subsequent turns."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("a")),
            make_response(final_json("b")),
        ]
    )
    runner, _store = make_runner(llm=llm, system_prompt="ROLE: pirate")

    await collect(runner.run("s1", "msg1"))
    await collect(runner.run("s1", "msg2"))

    # Second LLM call: loop prepends its own "system" message, then the
    # persisted role="system" from runner, then user/assistant/user.
    second = llm.calls[1].messages
    # Expect two system messages: the loop's structured one, then the
    # persisted kickoff.
    assert second[0].role == "system"
    assert second[1].role == "system"
    assert second[1].content == "ROLE: pirate"


# ---------------------------------------------------------------------------
# Event forwarding
# ---------------------------------------------------------------------------


async def test_events_forwarded_in_loop_order() -> None:
    """A multi-step run forwards loop events untouched, in order."""
    tool = FakeTool("search", result=ToolResult(output={"x": 1}))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("look up", "search", {"q": "x"})),
            make_response(final_json("got it")),
        ]
    )
    runner, _store = make_runner(llm=llm, registry=registry)

    events = await collect(runner.run("s1", "Hi"))

    types = [type(e) for e in events]
    assert types == [
        ThoughtEvent,
        ActionEvent,
        ToolStartedEvent,
        ObservationEvent,
        ThoughtEvent,
        FinalEvent,
    ]
    # Sequence numbers should be monotonic and dense from the loop.
    assert [e.sequence for e in events] == list(range(len(events)))


async def test_run_returns_async_iterator() -> None:
    """``run()`` is an :class:`AsyncIterator[AgentEvent]`."""
    llm = FakeLLMClient(replies=[make_response(final_json("ok"))])
    runner, _ = make_runner(llm=llm)
    iterator = runner.run("s1", "Hi")
    assert isinstance(iterator, AsyncIterator)
    await collect(iterator)


async def test_runner_does_not_inject_extra_events() -> None:
    """Runner forwards exactly the loop's event count — never adds or drops."""
    llm = FakeLLMClient(replies=[make_response(final_json("ok"))])
    runner, _ = make_runner(llm=llm)

    events = await collect(runner.run("s1", "Hi"))
    # Loop's happy path emits [ThoughtEvent, FinalEvent]: nothing more.
    assert len(events) == 2
    assert isinstance(events[0], ThoughtEvent)
    assert isinstance(events[1], FinalEvent)


# ---------------------------------------------------------------------------
# Transactional invariants — error paths
# ---------------------------------------------------------------------------


async def test_llm_error_does_not_persist_assistant() -> None:
    """LLM failure inside the loop: user message persists; assistant does not."""
    from fifty_agent_sdk.errors import LLMError

    llm = FakeLLMClient(replies=[LLMError("provider down")])
    runner, store = make_runner(llm=llm)

    events = await collect(runner.run("s1", "Hi"))

    # Loop emits ErrorEvent + fallback FinalEvent.
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert isinstance(events[-1], FinalEvent)

    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user"]
    assert history[0].content == "Hi"


async def test_parser_error_does_not_persist_assistant() -> None:
    """ParserError inside the loop: assistant not persisted.

    BR-018: the loop performs a one-shot parser-error retry by default,
    so two prose replies are needed to exhaust the budget and drive the
    run through to the terminal ParserError path that this test pins.
    """
    llm = FakeLLMClient(
        replies=[
            make_response("not valid json at all"),
            make_response("still not valid json"),
        ]
    )
    runner, store = make_runner(llm=llm)

    events = await collect(runner.run("s1", "Hi"))

    assert any(isinstance(e, ErrorEvent) for e in events)
    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user"]


async def test_iteration_cap_does_not_persist_assistant() -> None:
    """Hitting the iteration cap is treated as an error path."""
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=2)
    tool_call = make_response(tool_json("t", "t", {}))
    llm = FakeLLMClient(replies=[tool_call, tool_call])
    runner, store = make_runner(llm=llm, registry=registry, safety=safety)

    events = await collect(runner.run("s1", "Hi"))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].error_type == "MaxIterationsExceeded"

    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user"]


async def test_system_prompt_persists_even_on_error_path() -> None:
    """First-turn ``system_prompt`` lands BEFORE the user msg; survives loop failure."""
    from fifty_agent_sdk.errors import LLMError

    llm = FakeLLMClient(replies=[LLMError("down")])
    runner, store = make_runner(llm=llm, system_prompt="ROLE: pirate")

    await collect(runner.run("s1", "Hi"))

    history = await store.get_messages("s1")
    # system_prompt + user, no assistant.
    assert [m.role for m in history] == ["system", "user"]
    assert history[0].content == "ROLE: pirate"


# ---------------------------------------------------------------------------
# Recoverable tool failures still mean success persistence
# ---------------------------------------------------------------------------


async def test_recoverable_tool_failure_still_persists_assistant() -> None:
    """A ToolFailedEvent does NOT trigger the error path — the loop recovers."""
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
    runner, store = make_runner(llm=llm, registry=registry)

    events = await collect(runner.run("s1", "Hi"))

    # ToolFailedEvent is NOT an ErrorEvent.
    assert any(isinstance(e, ToolFailedEvent) for e in events)
    assert not any(isinstance(e, ErrorEvent) for e in events)

    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user", "assistant"]
    # BR-016: persisted assistant turn is the raw envelope.
    assert history[1].content == final_json("recovered")


# ---------------------------------------------------------------------------
# Tool roundtrips not persisted to state
# ---------------------------------------------------------------------------


async def test_tool_roundtrips_are_not_persisted_to_state() -> None:
    """Tool messages live in the loop's working list — never in the state store."""
    tool_a = FakeTool("alpha", result=ToolResult(output="A-result"))
    tool_b = FakeTool("beta", result=ToolResult(output="B-result"))
    registry = Registry()
    registry.register(tool_a)
    registry.register(tool_b)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("step 1", "alpha", {})),
            make_response(tool_json("step 2", "beta", {})),
            make_response(final_json("done")),
        ]
    )
    runner, store = make_runner(llm=llm, registry=registry)

    await collect(runner.run("s1", "do it"))

    history = await store.get_messages("s1")
    # State has only the durable conversation: user + final assistant.
    assert [m.role for m in history] == ["user", "assistant"]
    assert all(m.role != "tool" for m in history)
    # No intermediate assistant turns either.
    assistant_msgs = [m for m in history if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    # BR-016: persisted assistant turn is the raw envelope.
    assert assistant_msgs[0].content == final_json("done")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancellation_does_not_persist_assistant() -> None:
    """``aclose()`` mid-stream: user msg persists, assistant does not."""
    tool = FakeTool("slow", sleep_seconds=1.0)
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("t", "slow", {})),
            make_response(final_json("done")),
        ]
    )
    safety = SafetyConfig(max_iterations=3, tool_timeout_seconds=None)
    runner, store = make_runner(llm=llm, registry=registry, safety=safety)

    agen = runner.run("s1", "Hi")
    first_event = await agen.__anext__()
    assert first_event is not None
    await agen.aclose()

    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user"]
    assert history[0].content == "Hi"


async def test_task_cancellation_propagates() -> None:
    """Cancelling the consumer task surfaces CancelledError untouched."""
    tool = FakeTool("slow", sleep_seconds=1.0)
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("t", "slow", {})),
            make_response(final_json("done")),
        ]
    )
    safety = SafetyConfig(max_iterations=3, tool_timeout_seconds=None)
    runner, store = make_runner(llm=llm, registry=registry, safety=safety)

    async def consume() -> list[AgentEvent]:
        return await collect(runner.run("s1", "Hi"))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # User msg already durable; no assistant.
    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user"]


# ---------------------------------------------------------------------------
# State-store failure modes
# ---------------------------------------------------------------------------


class _FailingAppendStore:
    """Test double whose ``append`` always raises ``StateStoreError``."""

    def __init__(self) -> None:
        self.get_calls = 0

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        self.get_calls += 1
        return []

    async def append(self, session_id: str, message: ChatMessage) -> None:
        raise StateStoreError(
            "backend unavailable",
            context={"session_id": session_id, "wrapped": "BackendDown"},
        )

    async def delete(self, session_id: str) -> None:
        return None

    async def fork(self, session_id: str, from_sequence: int) -> str:
        raise NotImplementedError

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        raise NotImplementedError

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        raise NotImplementedError

    async def truncate_after(
        self, session_id: str, sequence: int, *, branch_id: str | None = None
    ) -> None:
        raise NotImplementedError


class _FailingGetStore:
    """Test double whose ``get_messages`` raises ``StateStoreError``."""

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        raise StateStoreError(
            "backend down on read",
            context={"session_id": session_id},
        )

    async def append(self, session_id: str, message: ChatMessage) -> None:
        return None

    async def delete(self, session_id: str) -> None:
        return None

    async def fork(self, session_id: str, from_sequence: int) -> str:
        raise NotImplementedError

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        raise NotImplementedError

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        raise NotImplementedError

    async def truncate_after(
        self, session_id: str, sequence: int, *, branch_id: str | None = None
    ) -> None:
        raise NotImplementedError


async def test_state_store_error_on_load_propagates_before_any_event() -> None:
    """``get_messages`` failure surfaces on the first ``__anext__`` call."""
    failing: StateStore = _FailingGetStore()
    llm = FakeLLMClient(replies=[make_response(final_json("never reached"))])
    runner, _ = make_runner(llm=llm, state=failing)

    agen = runner.run("s1", "Hi")
    with pytest.raises(StateStoreError):
        await agen.__anext__()
    assert len(llm.calls) == 0  # LLM was never called


async def test_state_store_error_on_user_append_propagates_before_any_event() -> None:
    """``append`` failure for the user message surfaces before any event yields."""
    failing = _FailingAppendStore()
    typed_failing: StateStore = failing
    llm = FakeLLMClient(replies=[make_response(final_json("never reached"))])
    runner, _ = make_runner(llm=llm, state=typed_failing)

    agen = runner.run("s1", "Hi")
    with pytest.raises(StateStoreError):
        await agen.__anext__()
    assert failing.get_calls == 1
    assert len(llm.calls) == 0  # LLM never reached


class _FailingNthAppendStore:
    """Test double whose ``append`` raises ``StateStoreError`` on the Nth call.

    All other ``append`` calls (and ``get_messages``) delegate to a real
    :class:`MemoryStateStore`, so the user message remains durable on the
    happy-path appends while the chosen call fails.
    """

    def __init__(self, *, fail_on_call: int) -> None:
        self._inner = MemoryStateStore()
        self._fail_on_call = fail_on_call
        self.append_calls = 0

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        return await self._inner.get_messages(session_id, branch_id=branch_id)

    async def append(self, session_id: str, message: ChatMessage) -> None:
        self.append_calls += 1
        if self.append_calls == self._fail_on_call:
            raise StateStoreError(
                "backend unavailable on assistant persist",
                context={
                    "session_id": session_id,
                    "call_number": self.append_calls,
                },
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


async def test_state_store_error_on_assistant_append_propagates_after_final_event() -> None:
    """Phase-5 failure: FinalEvent IS yielded, then persistence boundary raises.

    The loop produces a clean :class:`FinalEvent`; the Runner attempts to
    append the assistant message and the store raises
    :class:`StateStoreError`. The error must propagate to the caller,
    the user message must remain durable, and no assistant message may
    land in the store.
    """
    # First append = user message (succeeds), second append = assistant
    # message (raises). No system_prompt → no system append in between.
    failing = _FailingNthAppendStore(fail_on_call=2)
    typed_failing: StateStore = failing
    llm = FakeLLMClient(replies=[make_response(final_json("answer"))])
    runner, _ = make_runner(llm=llm, state=typed_failing)

    seen_events: list[AgentEvent] = []
    with pytest.raises(StateStoreError):
        async for event in runner.run("s1", "Hi"):
            seen_events.append(event)

    # FinalEvent was yielded BEFORE the persistence boundary failed.
    assert any(isinstance(e, FinalEvent) for e in seen_events)
    assert isinstance(seen_events[-1], FinalEvent)
    final_event = seen_events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "answer"

    # Exactly two append attempts: user (ok) + assistant (raised).
    assert failing.append_calls == 2

    # State has only the user message — the assistant persist failed at
    # the durability boundary, so no fake assistant turn was committed.
    history = await failing.get_messages("s1")
    assert [m.role for m in history] == ["user"]
    assert history[0].content == "Hi"


# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------


def test_top_level_exports_runner_surface() -> None:
    """Public re-exports for the runner trio."""
    from fifty_agent_sdk import AgentRunner as _AgentRunner
    from fifty_agent_sdk import MemoryStateStore as _MemoryStateStore
    from fifty_agent_sdk import StateStore as _StateStore

    assert _AgentRunner is AgentRunner
    assert _StateStore is StateStore
    assert _MemoryStateStore is MemoryStateStore


# ---------------------------------------------------------------------------
# Sanity wiring: full end-to-end build mirroring the docstring example
# ---------------------------------------------------------------------------


async def test_end_to_end_minimal_build() -> None:
    """A close-to-readme build: registry, loop, memory store, runner — all wire."""
    llm = FakeLLMClient(replies=[make_response(final_json("hello!"))])
    loop = AgentLoop(
        llm=llm,
        registry=Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona="Be terse."),
        safety=SafetyConfig(),
        model="test-model",
    )
    runner = AgentRunner(loop=loop, state=MemoryStateStore())

    events: list[AgentEvent] = []
    async for event in runner.run("s1", "hi"):
        events.append(event)

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.text == "hello!"


# ---------------------------------------------------------------------------
# Public state-store accessor (BR-011)
# ---------------------------------------------------------------------------


def test_runner_state_returns_the_exact_store_instance() -> None:
    """``AgentRunner.state`` returns the store passed as ``state=`` by identity (BR-011).

    The assertion is ``is``, not ``isinstance``: an ``isinstance`` check would
    also pass for a SECOND store constructed over the same engine, which is the
    wrong answer — a second store carries an independent per-session lock
    registry, so two writers could interleave on one session.
    """
    llm = FakeLLMClient(replies=[])
    store = MemoryStateStore()
    runner, returned = make_runner(llm=llm, state=store)

    assert runner.state is store
    assert returned is store


def test_runner_state_is_read_only() -> None:
    """Assigning to ``AgentRunner.state`` raises at runtime (BR-011).

    A mid-run store swap would split one turn's user and assistant appends
    across two backends, voiding the transactional-persistence invariant.

    What this does NOT pin, deliberately stated so the docstring does not
    overclaim: it cannot tell the shipped design (a bare ``@property``, no
    setter) from the design BR-011 rejected (a ``@state.setter`` that raises).
    Both raise ``AttributeError`` here and this test passes either way —
    verified. The difference is visible only to ``mypy``; see the comment at
    the property for why. Nothing mechanical catches a future raising setter —
    that comment is the only thing standing against it.
    """
    llm = FakeLLMClient(replies=[])
    runner, _ = make_runner(llm=llm)

    with pytest.raises(AttributeError):
        runner.state = MemoryStateStore()


async def test_runner_state_observes_writes_made_through_the_runner() -> None:
    """``AgentRunner.state`` is the live store the Runner writes through (BR-011).

    Not a snapshot and not a defensive copy: messages persisted by ``run()``
    are visible through the accessor, and it is still the same object.
    """
    llm = FakeLLMClient(replies=[make_response(final_json("hi there"))])
    store = MemoryStateStore()
    runner, _ = make_runner(llm=llm, state=store)

    await collect(runner.run("s1", "Hello"))

    assert [m.role for m in await runner.state.get_messages("s1")] == [
        "user",
        "assistant",
    ]
    assert runner.state is store


# ---------------------------------------------------------------------------
# Helpers smoke check
# ---------------------------------------------------------------------------


def test_imported_helpers_remain_accessible() -> None:
    """Smoke check that the conftest helpers are importable."""
    assert callable(final_json)
    assert callable(tool_json)
    assert callable(make_runner)
