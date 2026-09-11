"""Vendor-neutral observability hooks for the agent SDK.

The :class:`Hooks` dataclass is a container of optional callables that the
SDK invokes at well-defined points of a run. It is the SDK's *only*
observability surface, and it is deliberately vendor-neutral: ``fifty-agent-sdk``
depends on NO APM vendor, exports no tracer, and ships no exporter. The
consumer wires concrete hook implementations in their own startup code and
is free to bridge them to OpenTelemetry, Datadog, Prometheus, structured
logs, or nothing at all.

Two-tier wiring
    The seven hooks fire from two different tiers of the SDK:

    * Five **Runner-tier** hooks (``on_run_start``, ``on_run_end``,
      ``on_tool_start``, ``on_tool_end``, ``on_error``) fire from
      :class:`fifty_agent_sdk.runner.AgentRunner`.
    * Two **Loop-tier** hooks (``on_iteration``, ``on_llm_call``) fire from
      :class:`fifty_agent_sdk.loop.AgentLoop` — the ReACT iteration counter and
      the per-call :class:`~fifty_agent_sdk.llm.types.ChatRequest` /
      :class:`~fifty_agent_sdk.llm.types.ChatResponse` are loop-private and are
      NOT reconstructable from the public event stream.

    Construct ONE :class:`Hooks` instance and pass the SAME instance into
    BOTH :class:`fifty_agent_sdk.loop.AgentLoop` and
    :class:`fifty_agent_sdk.runner.AgentRunner`. The Runner does NOT forward its
    ``hooks`` into the loop — the two collaborators are wired independently
    by the consumer, exactly like ``audit`` is Runner-only. A :class:`Hooks`
    passed to only one of the two simply never fires the other tier's
    hooks — that is a documented, error-free no-op.

Sync or async
    Every hook may be a plain ``def`` or an ``async def`` (or any callable
    returning an awaitable — a ``functools.partial``, a callable object).
    The dispatch helper invokes the hook and inspects the RETURN VALUE with
    :func:`inspect.isawaitable`; an awaitable result is awaited, a plain
    result is not. There is no ``iscoroutinefunction`` guess.

Failure isolation
    A raising hook never breaks a run. :func:`invoke_hook` catches
    :class:`Exception`, logs a ``WARNING`` (event ``hook.invoke_failed``)
    under the fixed ``fifty_agent_sdk.observability`` logger, and swallows it.
    The log line carries the hook name and the exception TYPE only — never
    ``str(exc)`` — because hook arguments are high-value (user message,
    ``ChatRequest``, tool args) and a hook may embed them in its own
    exception message. :class:`asyncio.CancelledError` is the one exception
    re-raised untouched so consumer cancellation still propagates.

Hot-path latency
    Hooks are awaited INLINE. A slow ``on_iteration`` or ``on_llm_call``
    adds latency to every ReACT cycle. Keep hook bodies fast — the
    recommended pattern is enqueue-and-return (drop the work onto a queue
    or background task and return immediately).

Illustrative consumer wiring (NOT shipped — ``fifty-agent-sdk`` has no
OpenTelemetry dependency)::

    # illustrative only — the SDK ships no OpenTelemetry code
    from opentelemetry import trace

    tracer = trace.get_tracer("my-agent")

    def on_llm_call(session_id, request, response, duration_ms):
        with tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.model", request.model)
            span.set_attribute("llm.duration_ms", duration_ms)
            # note: usage is zero-filled in stream mode
            span.set_attribute(
                "llm.completion_tokens", response.usage.completion_tokens
            )

    hooks = Hooks(on_llm_call=on_llm_call)
    loop = AgentLoop(..., hooks=hooks)
    runner = AgentRunner(..., hooks=hooks)  # same instance into both
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import structlog

if TYPE_CHECKING:
    from fifty_agent_sdk.llm.types import ChatRequest, ChatResponse

_log: Final = structlog.get_logger("fifty_agent_sdk.observability")
"""Module-level structured logger.

Bound to the fixed name ``fifty_agent_sdk.observability`` (NOT ``__name__``) so
every hook-failure warning shares one logger name a consumer can route or
filter on — mirroring :mod:`fifty_agent_sdk.audit`'s fixed ``fifty_agent_sdk.audit``
logger.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Hooks:
    """Container of optional observability callables.

    A frozen, keyword-only dataclass — NOT a Pydantic model. Every field is
    an ``Optional[Callable]``; Pydantic adds no real validation value for a
    bare callable and ``Hooks`` does not cross a wire/persistence boundary.
    It is collaborator wiring, the same role :class:`fifty_agent_sdk.safety.
    SafetyConfig` plays for the loop. ``frozen=True`` makes a :class:`Hooks`
    instance safely shareable across concurrent runs; ``kw_only=True`` makes
    consumers name every hook (seven same-typed fields mis-order trivially).

    Every field defaults to ``None``. With all fields ``None`` (the default
    :class:`Hooks`, or ``hooks=None`` on the Runner/Loop) observability is
    zero-overhead: :func:`invoke_hook` short-circuits on the ``None`` check
    before constructing or calling anything.

    Each callable's return type is ``Any`` so a sync hook (``-> None``) and
    an async hook (``-> Awaitable[...]``) both type-check against the same
    field. See the module docstring for the two-tier wiring contract, the
    sync/async dispatch rule, and the failure-isolation guarantee.

    Attributes:
        on_run_start: ``(session_id, user_message) -> Any``. Fires once at
            the start of every :meth:`fifty_agent_sdk.runner.AgentRunner.run`,
            after the run is logged. Runner tier.
        on_run_end: ``(session_id, duration_ms, error) -> Any``. Fires once
            from the ``finally`` block of every
            :meth:`fifty_agent_sdk.runner.AgentRunner.run`, on EVERY exit path.
            ``duration_ms`` is a monotonic wall measurement of the whole
            run. ``error`` is typed ``BaseException | None`` — non-``None``
            ONLY when an exception terminated the run (a
            :class:`~fifty_agent_sdk.errors.StateStoreError`, a fatal
            :class:`~fifty_agent_sdk.errors.AgentSdkError` escaping the loop,
            or a surfaced
            :class:`asyncio.CancelledError` which is a
            :class:`BaseException`, not an :class:`Exception`, hence the
            wider type). A loop-internal failure surfaces an
            :class:`~fifty_agent_sdk.streaming.ErrorEvent` and drives ``on_error``
            instead, with ``on_run_end`` still receiving ``error=None``.
            Runner tier.
        on_iteration: ``(session_id, iteration_n) -> Any``. Fires once per
            ReACT iteration, after the iteration counter is incremented.
            ``session_id`` is ``str | None`` — the loop may run without a
            Runner, in which case it is ``None``. Loop tier.
        on_llm_call: ``(session_id, request, response, duration_ms) -> Any``.
            Fires once per SUCCESSFUL LLM call. An LLM failure surfaces via
            ``on_error`` instead — ``on_llm_call`` never fires with no
            response. In stream mode ``response`` is a :class:`~fifty_agent_sdk.
            llm.types.ChatResponse` synthesized from the accumulated
            completion (a stream has no single response object).
            ``session_id`` is ``str | None`` (see ``on_iteration``). Loop
            tier.
        on_tool_start: ``(session_id, tool_name, args) -> Any``. Fires once
            when a tool invocation begins (on the
            :class:`~fifty_agent_sdk.streaming.ToolStartedEvent`). Runner tier.
        on_tool_end: ``(session_id, tool_name, result, duration_ms) -> Any``.
            Fires once when a tool invocation ends. ``result`` is the tool's
            output on success or the failure string on a recoverable
            failure. ``duration_ms`` is measured by the Runner between the
            tool's start and terminal events. Runner tier.
        on_error: ``(session_id, error, context) -> Any``. Fires on a
            loop-internal failure (a synthesized exception built from the
            :class:`~fifty_agent_sdk.streaming.ErrorEvent`), on a state-store
            durability failure (the caught
            :class:`~fifty_agent_sdk.errors.StateStoreError`), and on a fatal
            :class:`~fifty_agent_sdk.errors.AgentSdkError` escaping the loop
            (the exception itself, before it is re-raised). ``context`` is a
            structured detail dict. Runner tier.
    """

    on_run_start: Callable[[str, str], Any] | None = None
    on_run_end: Callable[[str, float, BaseException | None], Any] | None = None
    on_iteration: Callable[[str | None, int], Any] | None = None
    on_llm_call: Callable[[str | None, ChatRequest, ChatResponse, float], Any] | None = None
    on_tool_start: Callable[[str, str, dict[str, Any]], Any] | None = None
    on_tool_end: Callable[[str, str, Any, float], Any] | None = None
    # `error` is `Exception`, not `BaseException` (cf. `on_run_end`): a
    # cancellation (`asyncio.CancelledError`, a BaseException) routes through
    # `on_run_end`, never `on_error`, so the narrower type is correct here.
    on_error: Callable[[str, Exception, dict[str, Any]], Any] | None = None


async def invoke_hook(
    hook: Callable[..., Any] | None,
    hook_name: str,
    *args: Any,
) -> None:
    """Invoke a single observability hook, isolating any failure.

    The shared dispatch + swallow primitive used by BOTH
    :class:`fifty_agent_sdk.runner.AgentRunner` and
    :class:`fifty_agent_sdk.loop.AgentLoop`, so the sync/async detection and the
    failure-isolation policy live in exactly one place.

    Behaviour:

    * If ``hook is None`` — return immediately. This ``None`` check is the
      zero-overhead guard; with no hook configured nothing is constructed
      or called.
    * Otherwise call ``hook(*args)`` exactly once. If the RETURN VALUE is
      awaitable (:func:`inspect.isawaitable`), await it. A plain ``def``
      hook has already run its body by the time the call returns and yields
      a non-awaitable (typically ``None``) — it is simply not awaited. An
      ``async def`` hook, or any callable returning an awaitable, has its
      result awaited. Inspecting the result — not the function — is correct
      for every callable shape (``async def``, a sync function returning a
      coroutine, a callable class, a :func:`functools.partial`).
    * A raising hook is caught: :class:`asyncio.CancelledError` is re-raised
      untouched (consumer cancellation must propagate); any other
      :class:`Exception` is logged at ``WARNING`` (event
      ``hook.invoke_failed``) under the ``fifty_agent_sdk.observability`` logger
      and swallowed — a hook failure NEVER aborts a run. The log line
      carries the exception TYPE only, never ``str(exc)``: hooks receive
      high-value objects (the raw user message, the full ``ChatRequest``,
      tool args) and a hook may embed them in its exception message.

    Args:
        hook: The callable to invoke, or ``None`` for a no-op.
        hook_name: Stable name of the hook (e.g. ``"on_run_start"``) used in
            the ``hook.invoke_failed`` log line.
        *args: Positional arguments forwarded verbatim to the hook.

    Raises:
        asyncio.CancelledError: Re-raised untouched if the hook raises it.
    """
    if hook is None:
        return
    try:
        result = hook(*args)
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Log the exception TYPE only — never str(exc). Hooks receive
        # high-value objects (on_run_start gets the raw user_message,
        # on_llm_call the full ChatRequest, on_tool_start the tool args),
        # and a hook may embed them in its own exception message
        # (`ValueError(f"unexpected: {request.messages}")`); logging
        # str(exc) would dump that content into a WARNING line. This
        # matches the mcp.client tool-error-hook logging discipline.
        _log.warning(
            "hook.invoke_failed",
            hook_name=hook_name,
            error_type=type(exc).__name__,
        )


__all__ = ["Hooks", "invoke_hook"]
