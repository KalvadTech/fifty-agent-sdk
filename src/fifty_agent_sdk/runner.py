"""AgentRunner — the user-facing orchestrator.

:class:`AgentRunner` is the SDK's top-level entry point for running an
agent across multiple conversational turns. It wraps an
:class:`fifty_agent_sdk.loop.AgentLoop` with conversation-state persistence
around each ``run()`` call:

1. Load prior messages from a :class:`fifty_agent_sdk.state.protocol.StateStore`.
2. Persist any first-turn ``system_prompt`` and the user's new message
   BEFORE driving the loop (durable proof of request).
3. Drive :meth:`AgentLoop.run` and forward every
   :class:`fifty_agent_sdk.streaming.AgentEvent` to the caller.
4. Persist the assistant's final answer ONLY on a clean
   :class:`fifty_agent_sdk.streaming.FinalEvent` with no preceding
   :class:`fifty_agent_sdk.streaming.ErrorEvent`.

System prompt vs. AgentLoop's structured prompt
    :class:`AgentLoop`'s ``prompts: PromptSections`` is the SDK-structured
    prompt rebuilt on every iteration from a tool-registry snapshot. It is
    NEVER persisted — it lives inside the loop's private working list and
    is the model's per-iteration reasoning scaffolding.

    :class:`AgentRunner`'s ``system_prompt: str | None`` is an OPTIONAL
    consumer-supplied kickoff message that, when set, is persisted as
    the first :class:`fifty_agent_sdk.llm.types.ChatMessage` with
    ``role="system"`` on the FIRST turn of a session only. It is the
    high-level instruction the consumer wants to ride alongside the
    SDK's structured prompt (for example, "You are a helpful
    customer-support agent"). Both coexist in the loop's prompt; that
    is intentional and supported by every major LLM provider.

Transactional persistence invariants
    * Every successful ``run()`` appends exactly one user message and
      exactly one assistant message to the state store. If
      ``system_prompt`` was set and the session was empty, exactly one
      :class:`ChatMessage` with ``role="system"`` is also appended,
      BEFORE the user message.
    * On any error during loop execution (LLMError, ParserError,
      iteration cap), the assistant message is NOT persisted; the user
      message remains durable. The fallback final answer is yielded to
      the caller but not committed to history.
    * On consumer cancellation, the assistant message is NOT persisted;
      the user message remains durable.
    * Tool roundtrips (``role="tool"`` messages) are NOT persisted to the
      state store. They live in the loop's private working list and are
      deterministically re-derivable from the assistant's final answer
      on the next turn. Tool-level provenance is instead captured through
      the optional :class:`fifty_agent_sdk.audit.protocol.AuditSink` (see below).

Audit emission
    When an optional :class:`fifty_agent_sdk.audit.protocol.AuditSink` is wired
    in, the Runner emits an :class:`fifty_agent_sdk.audit.protocol.AuditEvent` at
    four points of every ``run()``: session start, each tool invocation
    (args plus a bounded result summary), the final answer, and any error.

    Audit emission is best-effort and isolated from the run: a raising
    sink is caught by :meth:`_emit_audit`, logged at ``WARNING`` under the
    ``fifty_agent_sdk.audit`` logger (event ``audit.emit_failed``), and
    swallowed — a sink outage NEVER aborts a live run.
    :class:`asyncio.CancelledError` is the one exception that is re-raised
    untouched. When ``audit`` is ``None`` (the default) emission is
    zero-overhead: :meth:`_emit_audit` returns before constructing any
    :class:`AuditEvent`.

Observability hooks
    When an optional :class:`fifty_agent_sdk.observability.Hooks` is wired in,
    the Runner fires five of the seven hooks: ``on_run_start`` once at run
    start, ``on_tool_start`` / ``on_tool_end`` per tool invocation,
    ``on_error`` on a loop or durability failure, and ``on_run_end`` once
    from the ``finally`` block on EVERY exit path. The remaining two hooks
    (``on_iteration``, ``on_llm_call``) are Loop-tier — the consumer must
    wire the SAME :class:`Hooks` instance into :class:`fifty_agent_sdk.loop.
    AgentLoop` as well. The Runner does NOT forward ``hooks`` into the loop;
    the two collaborators are wired independently, exactly like ``audit``.

    Hook dispatch is best-effort and isolated, mirroring audit emission: a
    raising hook is caught, logged at ``WARNING`` under the
    ``fifty_agent_sdk.observability`` logger (event ``hook.invoke_failed``), and
    swallowed — including ``on_run_end`` raising inside ``finally``, where
    the swallow guarantee is what makes awaiting a hook there safe.
    :class:`asyncio.CancelledError` is re-raised untouched. When ``hooks``
    is ``None`` (the default) dispatch is zero-overhead.

Logging
    Module-level :mod:`structlog` logger. ``INFO`` on run start and run
    end; ``ERROR`` only when persistence itself fails. Never logs prompt
    or message content — only lengths and counts.

    The ``runner.run_completed`` log carries ``terminated_by`` with one of:

    * ``"final_answer"`` — happy path: a clean :class:`FinalEvent` was
      yielded and the assistant message was persisted.
    * ``"error"`` — loop-internal failure (LLMError, ParserError,
      MaxIterationsExceeded). The loop's fallback FinalEvent is yielded
      but the assistant message is NOT persisted.
    * ``"state_store_error"`` — durability boundary failed. A
      :class:`fifty_agent_sdk.errors.StateStoreError` propagated out of one of
      the load/append calls. A companion ``phase`` field names which
      site failed: ``"load"``, ``"persist_system"``, ``"persist_user"``,
      or ``"persist_assistant"``.
    * ``"cancelled"`` — caller cancelled the consumer task or broke out
      of the ``async for`` via ``aclose()``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Final
from uuid import uuid4

import structlog

from fifty_agent_sdk.audit import AuditEvent, AuditSink
from fifty_agent_sdk.errors import StateStoreError
from fifty_agent_sdk.llm.types import ChatMessage
from fifty_agent_sdk.loop import AgentLoop
from fifty_agent_sdk.observability import Hooks
from fifty_agent_sdk.observability.hooks import invoke_hook
from fifty_agent_sdk.state.protocol import StateStore
from fifty_agent_sdk.streaming import (
    ActionEvent,
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    ObservationEvent,
    ToolFailedEvent,
    ToolStartedEvent,
)

_RESULT_SUMMARY_CAP: Final = 500
"""Character cap for the ``result_summary`` field of a ``tool_invocation``
audit event. A ``repr`` longer than this is truncated with a marker so a
large or binary tool result cannot bloat the audit row or the console log."""

_TRUNCATION_MARKER: Final = "…[truncated]"
"""Suffix appended to a ``result_summary`` that was clipped at the cap."""

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger. INFO at run boundaries; no content payloads."""


def _bounded_repr(value: object) -> str:
    """Return ``repr(value)`` clipped to :data:`_RESULT_SUMMARY_CAP` chars.

    A clipped string carries the :data:`_TRUNCATION_MARKER` suffix so a
    consumer can tell the summary is partial. Used to bound the
    ``result_summary`` of a ``tool_invocation`` audit event.
    """
    text = repr(value)
    if len(text) <= _RESULT_SUMMARY_CAP:
        return text
    return text[:_RESULT_SUMMARY_CAP] + _TRUNCATION_MARKER


class AgentRunner:
    """User-facing orchestrator that drives an :class:`AgentLoop` with state.

    A typical end-to-end agent fits in roughly fifteen lines::

        from fifty_agent_sdk import (
            JSON_MODE_OUTPUT_FORMAT, AgentLoop, AgentRunner, JsonModeParser,
            MemoryStateStore, OpenAICompatibleClient, PromptSections,
            Registry, SafetyConfig,
        )

        llm = OpenAICompatibleClient(...)
        registry = Registry()
        loop = AgentLoop(
            llm=llm, registry=registry, parser=JsonModeParser(),
            prompts=PromptSections(persona="You are helpful."),
            safety=SafetyConfig(), model="gpt-4o",
            output_format=JSON_MODE_OUTPUT_FORMAT,
        )
        runner = AgentRunner(
            loop=loop, state=MemoryStateStore(),
            system_prompt="You are a helpful customer-support agent.",
        )
        async for event in runner.run("session-abc", "Hello"):
            print(event)

    Args:
        loop: The :class:`AgentLoop` instance to drive on each ``run()``
            call. The same loop is reused across turns.
        state: Any :class:`StateStore` implementation. Use
            :class:`MemoryStateStore` for ephemeral in-memory storage;
            BR-009/BR-010 ship durable backends.
        system_prompt: Optional consumer-supplied kickoff persisted as the
            FIRST :class:`ChatMessage` with ``role="system"`` on the first
            turn of each fresh session. When ``None`` (default), no system
            message is persisted; the loop's structured prompt does the
            entire job. See the module docstring for the precise boundary
            between this and ``AgentLoop.prompts``.
        audit: Optional :class:`fifty_agent_sdk.audit.protocol.AuditSink`. When
            set, the Runner emits an
            :class:`fifty_agent_sdk.audit.protocol.AuditEvent` on session start,
            each tool invocation, the final answer, and any error. A
            raising sink never aborts a run — see the module docstring's
            "Audit emission" section. When ``None`` (default), emission is
            zero-overhead.
        hooks: Optional :class:`fifty_agent_sdk.observability.Hooks`. When set,
            the Runner fires the five Runner-tier hooks (``on_run_start``,
            ``on_run_end``, ``on_tool_start``, ``on_tool_end``,
            ``on_error``). The two Loop-tier hooks (``on_iteration``,
            ``on_llm_call``) fire only from :class:`fifty_agent_sdk.loop.
            AgentLoop` — wire the SAME :class:`Hooks` instance into the
            loop as well::

                hooks = Hooks(on_run_start=..., on_iteration=...)
                loop = AgentLoop(..., hooks=hooks)
                runner = AgentRunner(loop=loop, state=..., hooks=hooks)

            A raising hook never aborts a run — see the module docstring's
            "Observability hooks" section. When ``None`` (default),
            dispatch is zero-overhead.

    Invariants:
        * Every ``run()`` either persists exactly one user message and
          exactly one assistant message, OR persists only the user
          message (on error or cancellation).
        * The optional ``system_prompt`` is persisted at most ONCE per
          session — only on the first ``run()`` call for that session.
        * Tool roundtrips are NOT persisted to state; the loop's working
          list carries them. Tool-level provenance is captured through the
          optional ``audit`` sink instead.
        * Audit emission is best-effort and isolated: a raising
          :class:`AuditSink` is caught and logged, never propagated. With
          ``audit=None`` the run behaves identically to a Runner built
          without auditing — no events, no overhead.
        * Observability hook dispatch is best-effort and isolated: a
          raising hook is caught and logged, never propagated. With
          ``hooks=None`` the run behaves identically to a Runner built
          without hooks — no dispatch, no overhead.
        * A Runner-level ``run_id`` is generated per ``run()`` call for
          log correlation and is SEPARATE from the inner :class:`AgentLoop`
          run id. Neither id is exposed on :class:`AgentEvent` values.
        * The :class:`StateStore` passed as ``state`` is retrievable by
          identity through the read-only :attr:`state` property and is
          FIXED for the Runner's lifetime — the store this Runner reads
          and writes never changes after construction.
    """

    def __init__(
        self,
        *,
        loop: AgentLoop,
        state: StateStore,
        system_prompt: str | None = None,
        audit: AuditSink | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self._loop = loop
        self._state = state
        self._system_prompt = system_prompt
        self._audit = audit
        self._hooks = hooks

    # A bare ``@property`` with NO setter is deliberate, and a raising
    # ``@state.setter`` would be worse than nothing: defining a setter makes
    # ``runner.state = x`` type-check as LEGAL, converting a failure ``mypy``
    # catches at the consumer's keyboard into one that only appears at
    # runtime. With no setter, assignment is both a static error and an
    # ``AttributeError``. Do not "improve" this by adding one. (BR-011)
    @property
    def state(self) -> StateStore:
        """The :class:`StateStore` this Runner reads and writes.

        Returns the EXACT instance passed as the ``state`` constructor
        keyword — by identity, never a copy and never a wrapper. That
        identity is the whole point: sharing this object shares the store's
        internal serialization (for example
        :class:`fifty_agent_sdk.state.sql.SqlStateStore`'s per-session
        :class:`asyncio.Lock` registry), which a SECOND store constructed
        over the same engine would NOT share — two such stores carry
        independent lock registries, so two writers could interleave on one
        session. Use this instead of reaching for the private ``_state``
        attribute, which carries no semver protection.

        Read-only, for correctness rather than style. ``run()`` loads
        history, appends the user message, drives the loop, and only then
        appends the assistant message. A settable store would let a swap
        land BETWEEN those appends, splitting one turn's user and assistant
        messages across two backends and silently voiding the
        transactional-persistence invariants documented on this class and
        in the module docstring. The store is fixed at construction because
        the invariant requires it. To run against a different store,
        construct another Runner: ``state`` is a constructor keyword and
        :meth:`__init__` does no I/O, opens nothing, and starts no task, so
        a Runner is free to build.

        The declared type is the :class:`StateStore` protocol, because that
        is all the Runner knows. A caller who needs backend-specific API
        that is NOT on the protocol (for example
        :meth:`SqlStateStore.aclose`) should keep its own concretely-typed
        reference — store/engine lifecycle is caller-owned by design — or
        narrow with :func:`typing.cast` / :func:`isinstance`.

        Returns:
            The :class:`StateStore` this Runner was constructed with, by
            identity.

        (BR-011)
        """
        return self._state

    async def _emit_audit(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit a single :class:`AuditEvent` through the configured sink.

        Best-effort and isolated: short-circuits with zero overhead when no
        sink is configured; otherwise builds an :class:`AuditEvent`
        (``timestamp`` stamped now in UTC, ``user_id=None`` — the Runner has
        no user-id channel, consumers wanting it set it via a wrapping
        sink) and awaits :meth:`AuditSink.record`. A raising sink is caught,
        logged at ``WARNING`` (event ``audit.emit_failed``), and swallowed —
        an audit failure never aborts the run.
        :class:`asyncio.CancelledError` is re-raised untouched so consumer
        cancellation still propagates.

        Args:
            session_id: Opaque session identifier for the event.
            event_type: One of ``"session_start"``, ``"tool_invocation"``,
                ``"final_answer"``, ``"error"``.
            payload: Structured, event-specific detail (lengths/counts and
                tool metadata only — never message or prompt content).
        """
        if self._audit is None:
            return
        event = AuditEvent(
            session_id=session_id,
            user_id=None,
            timestamp=datetime.now(UTC),
            event_type=event_type,
            payload=payload,
        )
        try:
            await self._audit.record(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(
                "audit.emit_failed",
                session_id=session_id,
                event_type=event_type,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def _invoke_hook(self, hook_name: str, *args: Any) -> None:
        """Dispatch a single Runner-tier observability hook.

        Reads the named field off the configured :class:`Hooks` and
        delegates to :func:`fifty_agent_sdk.observability.hooks.invoke_hook`.
        Short-circuits with zero overhead when no :class:`Hooks` is wired.
        A raising hook is logged at ``WARNING`` (event
        ``hook.invoke_failed``) and swallowed by the delegate;
        :class:`asyncio.CancelledError` is re-raised untouched — which is
        what makes awaiting ``on_run_end`` inside ``finally`` safe.

        Args:
            hook_name: Name of the :class:`Hooks` field to fire — one of
                ``"on_run_start"``, ``"on_run_end"``, ``"on_tool_start"``,
                ``"on_tool_end"``, ``"on_error"``.
            *args: Positional arguments forwarded verbatim to the hook.
        """
        if self._hooks is None:
            return
        hook = getattr(self._hooks, hook_name)
        await invoke_hook(hook, hook_name, *args)

    @staticmethod
    def _tool_invocation_payload(
        event: ObservationEvent | ToolFailedEvent,
        args: dict[str, Any],
        pending_call: ToolStartedEvent | None,
    ) -> dict[str, Any]:
        """Build the ``payload`` for a ``tool_invocation`` audit event.

        ``args`` and ``pending_call`` are the per-call state the event loop
        correlated under this call's ``call_id``: ``args`` was claimed FIFO
        from the pending :class:`ActionEvent` queue when the matching
        :class:`ToolStartedEvent` arrived (an :class:`ActionEvent` carries
        no ``call_id``; both the single-call branch and the ``MultiAction``
        batch emit actions and starts in call order), and ``pending_call``
        is the :class:`ToolStartedEvent` popped under the same key. Both
        are already resolved per-call by the caller — this function only
        shapes the payload.

        ``result_summary`` is a bounded ``repr`` of the tool's output (on
        success) or the failure string (on a recoverable failure), capped
        so a large or binary result cannot bloat the audit row.

        Args:
            event: The :class:`ObservationEvent` or :class:`ToolFailedEvent`
                that ended the tool call.
            args: The tool's argument dict, correlated per call; ``{}``
                when no :class:`ActionEvent` could be claimed for the call.
            pending_call: The correlated :class:`ToolStartedEvent`, if seen.

        Returns:
            The structured ``payload`` dict for the audit event.
        """
        if isinstance(event, ObservationEvent):
            outcome = "ok"
            result_summary = _bounded_repr(event.result.output)
        else:
            outcome = "failed"
            result_summary = _bounded_repr(event.error)
        return {
            "tool_name": event.tool_name,
            "call_id": (pending_call.call_id if pending_call is not None else event.call_id),
            "args": args,
            "outcome": outcome,
            "result_summary": result_summary,
        }

    async def run(self, session_id: str, user_message: str) -> AsyncIterator[AgentEvent]:
        """Drive a single conversational turn for ``session_id``.

        Algorithm (full edge-case detail in the module docstring):

        1. Load prior messages from the state store.
        2. If the session is empty AND ``system_prompt`` is set, append the
           system message to state BEFORE the user message.
        3. Append the user message to state BEFORE driving the loop. This
           is the load-bearing transactional property — if the loop
           later fails the user message remains durable.
        4. Drive :meth:`AgentLoop.run` with the loaded-plus-user
           conversation. Forward every event to the caller unchanged.
        5. If the run terminates with a :class:`FinalEvent` and no
           preceding :class:`ErrorEvent`, append the assistant message
           to state. The runner persists the raw LLM completion when the
           loop supplies it (the happy-path :class:`FinalAnswer` branch
           sets :attr:`FinalEvent.raw_completion`), falling back to the
           parsed text on the safety paths. Otherwise skip — the
           fallback final answer is yielded but not committed.

        On consumer cancellation (the consumer breaks out of the
        ``async for`` loop): :class:`asyncio.CancelledError` propagates
        untouched. The user message persisted in step 3 survives; no
        assistant message is persisted.

        On :class:`fifty_agent_sdk.errors.StateStoreError` raised by the state
        store: the error is logged at ``ERROR`` and re-raised. The Runner
        does NOT swallow state-store failures — the caller decides retry
        policy.

        Args:
            session_id: Opaque session identifier.
            user_message: The new user message text.

        Yields:
            :class:`AgentEvent` values forwarded from the inner
            :class:`AgentLoop` in monotonic ``sequence`` order. The
            terminal event is always a :class:`FinalEvent`.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If any state-store
                operation fails. The error is logged before being
                re-raised.
            asyncio.CancelledError: Propagated untouched from the loop
                or from the consumer's cancellation.

        Log events:
            ``runner.run_started`` (INFO): Emitted once after the load
                phase succeeds, before any persistence. Payload:
                ``session_id``, ``run_id``, ``user_message_len``,
                ``is_first_turn``, ``has_system_prompt``,
                ``has_prior_messages``.
            ``runner.run_completed`` (INFO): Emitted from the ``finally``
                block for any run that passed the load gate — every exit
                path, success or failure. Payload: ``session_id``,
                ``run_id``, ``terminated_by``,
                ``assistant_message_persisted``, ``event_count``,
                ``final_event_type``, ``phase``. ``terminated_by`` is one
                of ``"final_answer"``, ``"error"``,
                ``"state_store_error"``, or ``"cancelled"``.
            ``runner.persist_failed`` (ERROR): Emitted at each of the four
                state-store boundaries — load, system-prompt persist, user
                persist, assistant persist — when the underlying
                :class:`fifty_agent_sdk.errors.StateStoreError` is raised.
                Payload: ``phase`` (one of ``"load"``,
                ``"persist_system"``, ``"persist_user"``,
                ``"persist_assistant"``), ``session_id``, ``run_id``,
                ``error_type``, ``error_message``. When the load phase
                fails, ``runner.persist_failed`` with ``phase="load"`` is
                the only log emitted — no ``runner.run_completed`` follows,
                because the run never entered the ``finally`` block.
        """
        run_id = uuid4().hex
        # Monotonic start stamp for the `on_run_end` duration. `perf_counter`
        # (not wall-clock `datetime`) is correct for measuring an elapsed
        # interval. Captured before PHASE 1 so a load failure is still timed
        # — but note a load failure raises before the try/finally below, so
        # `on_run_end` does not fire for it (consistent with the audit layer
        # not emitting a `run_completed` log on a load failure).
        run_start = time.perf_counter()

        # ── PHASE 1: LOAD ──────────────────────────────────────────────
        try:
            history = await self._state.get_messages(session_id)
        except StateStoreError as exc:
            _log.error(
                "runner.persist_failed",
                phase="load",
                session_id=session_id,
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            # No run_completed log here: we never entered the try/finally
            # block below, so there is nothing to summarise. The caller
            # gets the StateStoreError unchanged.
            raise
        is_first_turn = len(history) == 0
        _log.info(
            "runner.run_started",
            session_id=session_id,
            run_id=run_id,
            user_message_len=len(user_message),
            is_first_turn=is_first_turn,
            has_system_prompt=self._system_prompt is not None,
            has_prior_messages=not is_first_turn,
        )
        await self._emit_audit(
            session_id,
            "session_start",
            {
                "run_id": run_id,
                "is_first_turn": is_first_turn,
                "has_system_prompt": self._system_prompt is not None,
                "user_message_len": len(user_message),
            },
        )
        await self._invoke_hook("on_run_start", session_id, user_message)

        # Initial value is "interrupted" — neutral and applies to any
        # unexpected exit path (e.g. an exception escaping the loop that
        # we did not catch explicitly). The dedicated
        # ``except asyncio.CancelledError`` branch upgrades this to
        # ``"cancelled"`` ONLY when we can attribute exit to an actual
        # task/consumer cancellation.
        terminated_by = "interrupted"
        state_store_error_phase: str | None = None
        saw_error = False
        final_text: str | None = None
        raw_final_completion: str | None = None
        event_count = 0
        # `run_error` carries the exception that terminated the run, for the
        # `on_run_end` hook. It is set ONLY by an exception that escaped the
        # run — a `StateStoreError` from a persist site or a surfaced
        # `CancelledError`. Typed `BaseException | None` because
        # `asyncio.CancelledError` is a `BaseException`, not an `Exception`.
        # A loop-internal failure surfaces an `ErrorEvent` (not a Python
        # exception) and is reported via `on_error`; for that path
        # `run_error` stays `None`. See the BR-012 plan's Q5.
        run_error: BaseException | None = None

        try:
            # ── PHASE 2: PERSIST KICKOFF (FIRST TURN ONLY) ────────────
            if is_first_turn and self._system_prompt is not None:
                sys_msg = ChatMessage(role="system", content=self._system_prompt)
                try:
                    await self._state.append(session_id, sys_msg)
                except StateStoreError as exc:
                    state_store_error_phase = "persist_system"
                    run_error = exc
                    _log.error(
                        "runner.persist_failed",
                        phase="persist_system",
                        session_id=session_id,
                        run_id=run_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    # Audit the durability failure BEFORE re-raising. We
                    # cannot do this in `finally` — that block must not
                    # `await` the sink, since a raising sink would mask
                    # the in-flight StateStoreError.
                    await self._emit_audit(
                        session_id,
                        "error",
                        {
                            "run_id": run_id,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "phase": "persist_system",
                        },
                    )
                    await self._invoke_hook(
                        "on_error",
                        session_id,
                        exc,
                        {"phase": "persist_system"},
                    )
                    raise
                # Mirror locally — `history` was a defensive copy.
                history.append(sys_msg)

            # ── PHASE 3: PERSIST USER MESSAGE ─────────────────────────
            user_msg = ChatMessage(role="user", content=user_message)
            try:
                await self._state.append(session_id, user_msg)
            except StateStoreError as exc:
                state_store_error_phase = "persist_user"
                run_error = exc
                _log.error(
                    "runner.persist_failed",
                    phase="persist_user",
                    session_id=session_id,
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                await self._emit_audit(
                    session_id,
                    "error",
                    {
                        "run_id": run_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "phase": "persist_user",
                    },
                )
                await self._invoke_hook(
                    "on_error",
                    session_id,
                    exc,
                    {"phase": "persist_user"},
                )
                raise
            history.append(user_msg)

            # ── PHASE 4: DRIVE THE LOOP ───────────────────────────────
            # AgentLoop adds its OWN structured system message to the
            # head of its working list — that is the per-iteration
            # reasoning scaffold (tool descriptions, output format
            # hints) and is separate from any role="system" message we
            # persisted in phase 2 (the consumer's kickoff). Both
            # coexist in the prompt; that is intentional.
            loop_messages = list(history)  # defensive copy for the loop

            # Per-call correlation for `tool_invocation` audit events and the
            # on_tool_start/on_tool_end hooks, keyed by `call_id`. The
            # MultiAction branch of AgentLoop emits N ActionEvents, then N
            # ToolStartedEvents, then N terminal events — all in call order —
            # so single pending slots would mis-correlate a batch (the first
            # terminal event would inherit the LAST call's pairing).
            # ActionEvents carry no `call_id`, so their `args` are claimed
            # FIFO from `pending_actions` as each ToolStartedEvent arrives
            # (both branches emit actions and starts in call order), then
            # stored under the started call's `call_id`. The paired terminal
            # event pops its entry, leaving no residue. The single-call path
            # is the N=1 case of the same flow and behaves exactly as before.
            # `last_error` holds the most recent ErrorEvent for the error
            # branch below.
            pending_actions: deque[ActionEvent] = deque()
            pending_calls: dict[str, ToolStartedEvent] = {}
            pending_args: dict[str, dict[str, Any]] = {}
            tool_started_at: dict[str, float] = {}
            last_error: ErrorEvent | None = None

            async for event in self._loop.run(loop_messages, session_id=session_id):
                event_count += 1
                if isinstance(event, ErrorEvent):
                    saw_error = True
                    last_error = event
                elif isinstance(event, FinalEvent):
                    final_text = event.text
                    raw_final_completion = event.raw_completion
                elif isinstance(event, ActionEvent):
                    pending_actions.append(event)
                elif isinstance(event, ToolStartedEvent):
                    action = pending_actions.popleft() if pending_actions else None
                    pending_calls[event.call_id] = event
                    pending_args[event.call_id] = action.args if action is not None else {}
                    tool_started_at[event.call_id] = time.perf_counter()
                yield event
                # `on_tool_start` fires once the `ToolStartedEvent` is seen;
                # `args` were correlated under this call's `call_id` above.
                # Fired AFTER yielding so consumer delivery is never blocked.
                if isinstance(event, ToolStartedEvent):
                    await self._invoke_hook(
                        "on_tool_start",
                        session_id,
                        event.tool_name,
                        pending_args[event.call_id],
                    )
                # Emit `tool_invocation` AFTER yielding so consumer event
                # delivery is never blocked on audit latency.
                if isinstance(event, ObservationEvent | ToolFailedEvent):
                    pending_call = pending_calls.pop(event.call_id, None)
                    args = pending_args.pop(event.call_id, {})
                    started_at = tool_started_at.pop(event.call_id, None)
                    await self._emit_audit(
                        session_id,
                        "tool_invocation",
                        self._tool_invocation_payload(event, args, pending_call),
                    )
                    # `on_tool_end` fires beside the audit emission. `result`
                    # is the tool's output on success or the failure string
                    # on a recoverable failure.
                    tool_duration_ms = (
                        (time.perf_counter() - started_at) * 1000 if started_at is not None else 0.0
                    )
                    tool_result = (
                        event.result.output if isinstance(event, ObservationEvent) else event.error
                    )
                    await self._invoke_hook(
                        "on_tool_end",
                        session_id,
                        event.tool_name,
                        tool_result,
                        tool_duration_ms,
                    )

            # ── PHASE 5: PERSIST ASSISTANT (SUCCESS PATH) ─────────────
            if not saw_error and final_text is not None:
                # BR-016: prefer the raw LLM completion (the JSON envelope
                # produced by the parser's source format) so multi-turn
                # sessions persist a faithful assistant turn — the next
                # ``run()`` then sees the same structured shape the
                # provider produced, which is what JSON-mode parsers and
                # provider format detectors rely on. Fall back to the
                # parsed ``final_text`` only when the loop did NOT
                # supply a raw completion (the safety-fallback paths —
                # LLMError, ParserError, iteration cap — emit a
                # :class:`FinalEvent` with ``raw_completion=None`` and
                # ``not saw_error`` blocks those from entering this
                # branch in practice, but the fallback keeps the
                # contract explicit and the local invariant total).
                persist_content = (
                    raw_final_completion if raw_final_completion is not None else final_text
                )
                asst_msg = ChatMessage(role="assistant", content=persist_content)
                try:
                    await self._state.append(session_id, asst_msg)
                except StateStoreError as exc:
                    state_store_error_phase = "persist_assistant"
                    run_error = exc
                    _log.error(
                        "runner.persist_failed",
                        phase="persist_assistant",
                        session_id=session_id,
                        run_id=run_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    await self._emit_audit(
                        session_id,
                        "error",
                        {
                            "run_id": run_id,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "phase": "persist_assistant",
                        },
                    )
                    await self._invoke_hook(
                        "on_error",
                        session_id,
                        exc,
                        {"phase": "persist_assistant"},
                    )
                    raise
                terminated_by = "final_answer"
                await self._emit_audit(
                    session_id,
                    "final_answer",
                    {
                        "run_id": run_id,
                        "final_text_len": len(final_text),
                        "event_count": event_count,
                    },
                )
            else:
                # Error path: an ErrorEvent was emitted; the loop's
                # fallback FinalEvent is yielded but NOT persisted, so
                # the next run() does not see a fake assistant turn.
                terminated_by = "error"
                # Defensive `last_error is not None` fallbacks below: this
                # branch runs only when `saw_error` is True, which is set
                # alongside `last_error` whenever an ErrorEvent is seen — so
                # `last_error` is in practice always non-None here. The
                # "Unknown"/"" fallbacks guard a future refactor that could
                # decouple `saw_error` from `last_error`; do not delete them
                # as dead code.
                await self._emit_audit(
                    session_id,
                    "error",
                    {
                        "run_id": run_id,
                        "error_type": (
                            last_error.error_type if last_error is not None else "Unknown"
                        ),
                        "error_message": (last_error.message if last_error is not None else ""),
                    },
                )
                # `on_error` fires for the loop-internal failure. The loop
                # reports failure via an `ErrorEvent`, not a Python
                # exception, so a lightweight `RuntimeError` is synthesized
                # from `last_error` for the hook's `error: Exception`
                # parameter. `run_error` is NOT set here — the run did not
                # terminate by an escaped exception, so `on_run_end`
                # receives `error=None` (see the BR-012 plan's Q5).
                if last_error is not None:
                    await self._invoke_hook(
                        "on_error",
                        session_id,
                        RuntimeError(last_error.message),
                        {
                            "error_type": last_error.error_type,
                            **dict(last_error.context),
                        },
                    )
        except asyncio.CancelledError as exc:
            # Caller cancelled the consumer task (or broke out via
            # ``aclose()``). Attribute exit to cancellation and let the
            # exception propagate untouched.
            terminated_by = "cancelled"
            run_error = exc
            raise
        finally:
            if state_store_error_phase is not None:
                terminated_by = "state_store_error"
            assistant_persisted = terminated_by == "final_answer"
            _log.info(
                "runner.run_completed",
                session_id=session_id,
                run_id=run_id,
                terminated_by=terminated_by,
                assistant_message_persisted=assistant_persisted,
                event_count=event_count,
                final_event_type="final" if final_text is not None else None,
                phase=state_store_error_phase,
            )
            # `on_run_end` fires on EVERY exit path. `run_error` is
            # non-`None` only for an exception that escaped the run (a
            # `StateStoreError` or the surfaced `CancelledError`); a
            # loop-internal `terminated_by == "error"` keeps it `None`
            # (`on_error` already fired for that). Awaiting a hook in
            # `finally` is safe: `_invoke_hook`/`invoke_hook` swallow every
            # `Exception` and re-raise only `CancelledError`, so a raising
            # `on_run_end` cannot mask an in-flight `StateStoreError`.
            run_duration_ms = (time.perf_counter() - run_start) * 1000
            await self._invoke_hook("on_run_end", session_id, run_duration_ms, run_error)


__all__ = ["AgentRunner"]
