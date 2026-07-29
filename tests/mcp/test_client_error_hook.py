"""Contract suite for the ``MCPClient(on_tool_error=...)`` seam (BR-010).

BR-010 exposes a PUBLIC extension point for transforming a per-call MCP
``isError=True`` message before it reaches the model, replacing the
unsupported private-symbol workaround (importing
:class:`fifty_agent_sdk.mcp.client._MCPCallError` / overriding
``_unwrap_invoke_result``). These tests pin the whole contract:

* the hook receives the SDK's bounded message AND the server's RAW content;
* its returned string replaces the message, ``content`` untouched;
* the default (no hook) path is byte-for-byte unchanged;
* on ANY failure — raise, non-``str`` return, blank return — the ORIGINAL
  bounded message is used, so ``ToolResult.error`` can never become ``None``,
  ``""``, or a non-string via this seam;
* the failure WARNINGs never carry the exception text or the server content
  (the deliberate divergence from
  :func:`fifty_agent_sdk.observability.hooks.invoke_hook`, which logs
  ``str(exc)``);
* :class:`asyncio.CancelledError` propagates untouched;
* the hook is structurally unreachable on the fatal transport path (the
  BR-005 recoverable/fatal split, client tier).

The recoverable scenarios are driven through the in-memory FastMCP oracle
(``boom`` raises ``ToolError("safe message")``); the fatal scenario uses the
strict httpx transport mock. No ``unittest.mock``, no network.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

import httpx
import pytest
import structlog
from mcp.server.fastmcp import FastMCP

from fifty_agent_sdk.errors import MCPError
from fifty_agent_sdk.mcp import MCPClient, MCPClientConfig
from fifty_agent_sdk.mcp.client import _MCPCallError

from .conftest import MCP_URL, make_compat_client, make_strict_http_client

BOUNDED_MESSAGE = "MCP tool 'boom' returned isError=True"
"""The 1.3.0 bounded default for the FastMCP ``boom`` tool — the fallback."""


# --- Test doubles ---------------------------------------------------------


class _RecordingScreen:
    """A callable screen recording every ``(message, content)`` it received."""

    def __init__(self, replacement: Any = "screened") -> None:
        self._replacement = replacement
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def __call__(self, message: str, content: list[dict[str, Any]]) -> Any:
        self.calls.append((message, content))
        return self._replacement


class _RaisingScreen:
    """A screen that always raises, recording that it was reached."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def __call__(self, message: str, content: list[dict[str, Any]]) -> str:
        self.calls.append((message, content))
        raise self._exc


def _hook_logs(logs: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    """Filter captured structlog entries down to one hook event name."""
    return [entry for entry in logs if entry.get("event") == event]


# ---------------------------------------------------------------------------
# The hook receives the raw payload, and its return value is used
# ---------------------------------------------------------------------------


async def test_on_tool_error_receives_bounded_message_and_raw_content(
    fastmcp_server: FastMCP,
) -> None:
    """The hook is called once with the bounded message AND the raw content (BR-010).

    Acceptance criterion 3: ``args[1]`` is the same server content the private
    ``_MCPCallError.content`` carries — the hook sees the raw evidence, not a
    summary.
    """
    screen = _RecordingScreen()
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert len(screen.calls) == 1
    message, content = screen.calls[0]
    assert message == BOUNDED_MESSAGE
    assert isinstance(content, list)
    joined = " ".join(block.get("text", "") for block in content)
    assert "safe message" in joined
    # The hook received the very list the carrier retains.
    assert isinstance(result, _MCPCallError)
    assert result.content == content


async def test_on_tool_error_replacement_becomes_the_call_error_message(
    fastmcp_server: FastMCP,
) -> None:
    """The returned string replaces ``message``; ``content`` is preserved (BR-010)."""
    screen = _RecordingScreen("redacted business failure")
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert isinstance(result, _MCPCallError)
    assert result.message == "redacted business failure"
    assert result.content == screen.calls[0][1]


async def test_on_tool_error_absent_leaves_message_byte_for_byte_unchanged(
    fastmcp_server: FastMCP,
) -> None:
    """Flag-off is byte-for-byte the 1.3.0 path — omitted AND explicit ``None`` (BR-010).

    The mandatory opt-in/flag-off-unchanged test: neither construction shape
    may alter the bounded message or the retained content.
    """
    async with make_compat_client(fastmcp_server) as omitted_client:
        omitted = await omitted_client.invoke("boom", {"x": "y"})
    async with make_compat_client(fastmcp_server, on_tool_error=None) as explicit_client:
        explicit = await explicit_client.invoke("boom", {"x": "y"})

    assert isinstance(omitted, _MCPCallError)
    assert isinstance(explicit, _MCPCallError)
    assert omitted.message == BOUNDED_MESSAGE
    assert explicit.message == BOUNDED_MESSAGE
    assert omitted.content == explicit.content


async def test_on_tool_error_not_called_on_successful_result(
    fastmcp_server: FastMCP,
) -> None:
    """A successful ``tools/call`` returns untouched and never reaches the hook."""
    screen = _RecordingScreen()
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        result = await client.invoke("lookup", {"key": "alpha"})

    assert screen.calls == []
    assert result == {"key": "alpha", "found": True}


# ---------------------------------------------------------------------------
# Failure policy — the ORIGINAL bounded message is always the fallback
# ---------------------------------------------------------------------------


async def test_on_tool_error_raising_hook_falls_back_to_bounded_message(
    fastmcp_server: FastMCP,
) -> None:
    """A raising hook never propagates; the original bounded message is used (BR-010)."""
    screen = _RaisingScreen(RuntimeError("screen exploded"))
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert len(screen.calls) == 1
    assert isinstance(result, _MCPCallError)
    assert result.message == BOUNDED_MESSAGE


async def test_on_tool_error_raising_hook_logs_warning_without_exception_text(
    fastmcp_server: FastMCP,
) -> None:
    """``mcp.tool_error_hook_failed`` carries the type only — never text or content.

    The deliberate divergence from ``invoke_hook`` (which logs ``str(exc)``):
    a hook screening untrusted server content may embed that content in its own
    exception message, so this log line records ``tool_name`` + ``error_type``
    and nothing else (BR-010).
    """
    screen = _RaisingScreen(RuntimeError("SENSITIVE-EXC-TEXT safe message"))
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        with structlog.testing.capture_logs() as logs:
            await client.invoke("boom", {"x": "y"})

    failures = _hook_logs(logs, "mcp.tool_error_hook_failed")
    assert len(failures) == 1, f"expected exactly one hook-failure warning, got {logs}"
    entry = failures[0]
    assert entry["log_level"] == "warning"
    assert entry["tool_name"] == "boom"
    assert entry["error_type"] == "RuntimeError"
    rendered = repr(entry)
    assert "SENSITIVE-EXC-TEXT" not in rendered
    assert "safe message" not in rendered


@pytest.mark.parametrize(
    "returned",
    [None, 42, b"bytes", {"a": 1}, ["x"], object()],
    ids=["none", "int", "bytes", "dict", "list", "object"],
)
async def test_on_tool_error_non_str_return_falls_back_to_bounded_message(
    fastmcp_server: FastMCP,
    returned: Any,
) -> None:
    """A non-``str`` return is rejected and logged ``mcp.tool_error_hook_invalid``."""
    screen = _RecordingScreen(returned)
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        with structlog.testing.capture_logs() as logs:
            result = await client.invoke("boom", {"x": "y"})

    assert isinstance(result, _MCPCallError)
    assert result.message == BOUNDED_MESSAGE
    invalid = _hook_logs(logs, "mcp.tool_error_hook_invalid")
    assert len(invalid) == 1, f"expected exactly one invalid-return warning, got {logs}"
    assert invalid[0]["log_level"] == "warning"
    assert invalid[0]["tool_name"] == "boom"
    assert invalid[0]["returned_type"] == type(returned).__name__
    assert invalid[0]["reason"] == "not_a_string"


@pytest.mark.parametrize(
    "returned",
    ["", "   ", "\n\t"],
    ids=["empty", "spaces", "whitespace"],
)
async def test_on_tool_error_blank_return_falls_back_to_bounded_message(
    fastmcp_server: FastMCP,
    returned: str,
) -> None:
    """A blank/whitespace-only return is rejected — ``error`` can never be empty."""
    screen = _RecordingScreen(returned)
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        with structlog.testing.capture_logs() as logs:
            result = await client.invoke("boom", {"x": "y"})

    assert isinstance(result, _MCPCallError)
    assert result.message == BOUNDED_MESSAGE
    invalid = _hook_logs(logs, "mcp.tool_error_hook_invalid")
    assert len(invalid) == 1, f"expected exactly one invalid-return warning, got {logs}"
    assert invalid[0]["reason"] == "blank_string"
    assert invalid[0]["returned_type"] == "str"


# ---------------------------------------------------------------------------
# Dispatch shapes — inspect the RETURN VALUE, not the function
# ---------------------------------------------------------------------------


async def test_on_tool_error_async_hook_is_awaited(fastmcp_server: FastMCP) -> None:
    """An ``async def`` screen's RESOLVED string is used (the isawaitable branch)."""
    seen: list[str] = []

    async def screen(message: str, content: list[dict[str, Any]]) -> str:
        await asyncio.sleep(0)
        seen.append(message)
        return "async replacement"

    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert seen == [BOUNDED_MESSAGE]
    assert isinstance(result, _MCPCallError)
    assert result.message == "async replacement"


async def test_on_tool_error_sync_hook_result_is_used_directly(
    fastmcp_server: FastMCP,
) -> None:
    """A plain ``def`` screen works and its value is not awaited."""

    def screen(message: str, content: list[dict[str, Any]]) -> str:
        return "sync replacement"

    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert isinstance(result, _MCPCallError)
    assert result.message == "sync replacement"


def _partial_screen(prefix: str, message: str, content: list[dict[str, Any]]) -> str:
    """Module-level target for the :func:`functools.partial` dispatch case."""
    return f"{prefix}:{len(content)}"


class _CallableScreen:
    """A ``__call__``-ing object — a hook shape that is not a function."""

    def __call__(self, message: str, content: list[dict[str, Any]]) -> str:
        return f"callable:{len(content)}"


@pytest.mark.parametrize(
    ("hook", "expected"),
    [
        (functools.partial(_partial_screen, "partial"), "partial:1"),
        (_CallableScreen(), "callable:1"),
    ],
    ids=["partial", "callable_object"],
)
async def test_on_tool_error_accepts_partial_and_callable_object(
    fastmcp_server: FastMCP,
    hook: Any,
    expected: str,
) -> None:
    """Any callable shape works — the SDK inspects the result, not the function."""
    async with make_compat_client(fastmcp_server, on_tool_error=hook) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert isinstance(result, _MCPCallError)
    assert result.message == expected


async def test_on_tool_error_cancelled_error_propagates(fastmcp_server: FastMCP) -> None:
    """``asyncio.CancelledError`` is re-raised untouched, never swallowed (BR-010).

    The one carve-out from the fall-back-to-the-bounded-message policy: consumer
    cancellation must propagate. Pins the arm ORDERING — a ``CancelledError``
    arm placed after ``except Exception`` (or a bare catch) would swallow it
    into the fallback.
    """
    screen = _RaisingScreen(asyncio.CancelledError())
    async with make_compat_client(fastmcp_server, on_tool_error=screen) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.invoke("boom", {"x": "y"})

    assert len(screen.calls) == 1


# ---------------------------------------------------------------------------
# BR-005 fatal half — the hook is structurally unreachable there
# ---------------------------------------------------------------------------


async def test_on_tool_error_never_fires_on_transport_error() -> None:
    """A transport failure still raises MCPError and never reaches the hook.

    BR-005 guardrail extended to the BR-010 seam (client tier): the hook is
    applied strictly inside ``invoke``'s ``isinstance(..., _MCPCallError)``
    branch, which a connection failure never reaches — it is raised as
    :class:`MCPError` at the ``call_tool`` boundary, before the unwrap. A dead
    connection can NEVER be screened into a recoverable observation.
    """

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, _ = make_strict_http_client(boom)
    screen = _RecordingScreen("should never be used")
    client = MCPClient(
        MCPClientConfig(base_url=MCP_URL),
        client=http_client,
        on_tool_error=screen,
    )

    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})

    assert exc.value.context["wrapped"] == "ConnectError"
    assert screen.calls == []


async def test_on_tool_error_does_not_put_content_or_headers_in_mcp_error_context() -> None:
    """The auth-redaction invariant is unchanged with a hook configured (BR-010).

    The seam touches only the per-call error carrier: it never sees, builds, or
    logs a header dict, and it adds no key to any ``MCPError.context``.
    """

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, _ = make_strict_http_client(boom)
    screen = _RecordingScreen("should never be used")
    client = MCPClient(
        MCPClientConfig(base_url=MCP_URL),
        auth={"Authorization": "Bearer s3cret"},
        client=http_client,
        on_tool_error=screen,
    )

    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})

    context = exc.value.context
    assert set(context) <= {"operation", "server_url", "method", "tool_name", "wrapped"}
    rendered = repr(context) + repr(exc.value)
    assert "s3cret" not in rendered
    assert "Authorization" not in rendered
    assert screen.calls == []
