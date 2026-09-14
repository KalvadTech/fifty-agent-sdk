"""Unit tests for :mod:`fifty_agent_sdk.observability.hooks`.

Covers BR-012's model + dispatch-helper surface:

* :class:`Hooks` constructs with every field defaulting to ``None`` and is
  frozen.
* :func:`invoke_hook` is a no-op when the hook is ``None``.
* :func:`invoke_hook` invokes a SYNC callable once and does not await.
* :func:`invoke_hook` awaits an ASYNC callable once.
* :func:`invoke_hook` awaits the RESULT of a sync callable that returns a
  coroutine — validating ``isawaitable``-on-result, not
  ``iscoroutinefunction``-on-function.
* :func:`invoke_hook` swallows a raising hook and logs a ``hook.invoke_failed``
  WARNING.
* :func:`invoke_hook` re-raises :class:`asyncio.CancelledError` untouched.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
import structlog

from fifty_agent_sdk import Hooks as TopLevelHooks
from fifty_agent_sdk.observability import Hooks
from fifty_agent_sdk.observability.hooks import invoke_hook

# ---------------------------------------------------------------------------
# The Hooks model
# ---------------------------------------------------------------------------


def test_hooks_constructs_with_all_fields_none() -> None:
    """A bare :class:`Hooks` has every one of its seven fields ``None``."""
    hooks = Hooks()
    assert hooks.on_run_start is None
    assert hooks.on_run_end is None
    assert hooks.on_iteration is None
    assert hooks.on_llm_call is None
    assert hooks.on_tool_start is None
    assert hooks.on_tool_end is None
    assert hooks.on_error is None


def test_hooks_is_frozen() -> None:
    """Assigning a field on a constructed :class:`Hooks` raises."""
    hooks = Hooks()

    def _noop() -> None:  # pragma: no cover - never called
        return None

    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.on_run_start = _noop  # type: ignore[misc]


def test_hooks_accepts_a_subset_of_fields() -> None:
    """Only the named fields are set; the rest stay ``None``."""

    def _on_iter(session_id: str | None, n: int) -> None:
        return None

    hooks = Hooks(on_iteration=_on_iter)
    assert hooks.on_iteration is _on_iter
    assert hooks.on_run_start is None


def test_hooks_is_re_exported_at_package_root() -> None:
    """``Hooks`` is an eager re-export from the package root."""
    assert TopLevelHooks is Hooks


# ---------------------------------------------------------------------------
# invoke_hook — None
# ---------------------------------------------------------------------------


async def test_invoke_hook_none_is_a_noop() -> None:
    """``invoke_hook`` with a ``None`` hook returns without error."""
    await invoke_hook(None, "on_run_start", "s1", "hello")


# ---------------------------------------------------------------------------
# invoke_hook — sync
# ---------------------------------------------------------------------------


async def test_invoke_hook_calls_sync_hook_once() -> None:
    """A plain ``def`` hook is invoked exactly once with the passed args."""
    calls: list[tuple[object, ...]] = []

    def hook(*args: object) -> None:
        calls.append(args)

    await invoke_hook(hook, "on_iteration", "s1", 3)

    assert calls == [("s1", 3)]


async def test_invoke_hook_does_not_break_on_sync_hook_returning_none() -> None:
    """A sync hook returning ``None`` is not awaited and raises nothing."""
    flag: list[bool] = []

    def hook() -> None:
        flag.append(True)

    await invoke_hook(hook, "on_run_start")

    assert flag == [True]


# ---------------------------------------------------------------------------
# invoke_hook — async
# ---------------------------------------------------------------------------


async def test_invoke_hook_awaits_async_hook_once() -> None:
    """An ``async def`` hook is awaited exactly once."""
    calls: list[tuple[object, ...]] = []

    async def hook(*args: object) -> None:
        calls.append(args)

    await invoke_hook(hook, "on_run_end", "s1", 12.0, None)

    assert calls == [("s1", 12.0, None)]


async def test_invoke_hook_awaits_result_of_sync_hook_returning_coroutine() -> None:
    """A sync function returning a coroutine has its RESULT awaited.

    This is the discriminating case: ``inspect.iscoroutinefunction`` is
    ``False`` for ``hook`` here, yet the returned coroutine must still be
    awaited. ``invoke_hook`` inspects the return value, not the function.
    """
    awaited: list[bool] = []

    async def _inner() -> None:
        awaited.append(True)

    def hook() -> object:
        # A plain (sync) function whose RETURN VALUE is awaitable.
        return _inner()

    await invoke_hook(hook, "on_iteration")

    assert awaited == [True]


# ---------------------------------------------------------------------------
# invoke_hook — failure isolation
# ---------------------------------------------------------------------------


async def test_invoke_hook_swallows_raising_sync_hook() -> None:
    """A raising sync hook is swallowed — ``invoke_hook`` returns normally."""

    def hook() -> None:
        raise RuntimeError("hook boom")

    # No exception escapes.
    await invoke_hook(hook, "on_run_start")


async def test_invoke_hook_swallows_raising_async_hook() -> None:
    """A raising async hook is swallowed — ``invoke_hook`` returns normally."""

    async def hook() -> None:
        raise ValueError("async hook boom")

    await invoke_hook(hook, "on_llm_call")


async def test_invoke_hook_logs_invoke_failed_warning() -> None:
    """A raising hook produces a ``hook.invoke_failed`` WARNING log line."""

    def hook() -> None:
        raise RuntimeError("hook boom")

    with structlog.testing.capture_logs() as logs:
        await invoke_hook(hook, "on_tool_start")

    failures = [e for e in logs if e.get("event") == "hook.invoke_failed"]
    assert len(failures) == 1
    assert failures[0]["log_level"] == "warning"
    assert failures[0]["hook_name"] == "on_tool_start"
    assert failures[0]["error_type"] == "RuntimeError"
    # Only the exception TYPE is logged — never str(exc) (see below).
    assert "error_message" not in failures[0]


async def test_invoke_hook_log_never_leaks_hook_arguments() -> None:
    """A secret embedded in a hook's exception message never reaches the log.

    Hooks receive high-value objects (``on_run_start`` gets the raw user
    message, ``on_llm_call`` the full ``ChatRequest``, ``on_tool_start`` the
    tool args), so a hook raising ``ValueError(f"unexpected: {args}")`` would
    dump that content into a WARNING line if ``str(exc)`` were logged. The
    failure log carries the exception TYPE only.
    """
    secret = "SECRET-hook-arg-DO-NOT-LEAK"

    def hook(*args: object) -> None:
        raise ValueError(f"unexpected: {args}")

    with structlog.testing.capture_logs() as logs:
        await invoke_hook(hook, "on_tool_start", "s1", "search", {"token": secret})

    failures = [e for e in logs if e.get("event") == "hook.invoke_failed"]
    assert len(failures) == 1
    assert failures[0]["error_type"] == "ValueError"
    assert secret not in str(failures[0])


async def test_invoke_hook_reraises_cancelled_error() -> None:
    """A hook raising :class:`asyncio.CancelledError` propagates untouched."""

    async def hook() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await invoke_hook(hook, "on_run_end")


async def test_invoke_hook_does_not_log_for_cancelled_error() -> None:
    """A :class:`asyncio.CancelledError` is NOT logged as a hook failure."""

    async def hook() -> None:
        raise asyncio.CancelledError

    with structlog.testing.capture_logs() as logs:  # noqa: SIM117
        with pytest.raises(asyncio.CancelledError):
            await invoke_hook(hook, "on_run_end")

    assert [e for e in logs if e.get("event") == "hook.invoke_failed"] == []
