"""Tests for :class:`fifty_agent_sdk.mcp.client.MCPClient` (the ``mcp``-SDK wrapper).

The protocol wire is owned by ``mcp`` and certified by the in-memory
official-client oracle in ``test_client_compat.py``. This file covers the
fifty-agent-sdk-facing contract the wrapper still owns:

- transport-error → :class:`MCPError` translation (connect / timeout / HTTP
  status), exercised through the REAL ``streamable_http_client`` over an
  injected httpx client so the anyio ``ExceptionGroup`` unwrap path runs;
- auth-header redaction (static + callable) and the misbehaved-callable guard;
- the ``aclose`` lifecycle (owned vs injected client, idempotency,
  closed-guard);
- the ``user_agent`` config field set on the owned httpx client;
- the ``MCPError.context`` doc/runtime allow-list parity guard.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from fifty_agent_sdk.errors import MCPError
from fifty_agent_sdk.mcp import MCPClient, MCPClientConfig

from .conftest import MCP_URL, StrictTransportMock, make_compat_client, make_strict_http_client


def _config(**overrides: Any) -> MCPClientConfig:
    fields: dict[str, Any] = {
        "base_url": MCP_URL,
        "connect_timeout_seconds": 1.0,
        "read_timeout_seconds": 1.0,
    }
    fields.update(overrides)
    return MCPClientConfig(**fields)


# ---------------------------------------------------------------------------
# Transport-layer errors via the REAL streamable_http_client + thin httpx mock
# ---------------------------------------------------------------------------


async def test_invoke_maps_connect_error_to_mcp_error() -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, mock = make_strict_http_client(boom)
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert exc.value.context["wrapped"] == "ConnectError"
    assert exc.value.context["server_url"] == MCP_URL
    assert exc.value.context["tool_name"] == "search"
    assert mock.observed_requests, "expected at least one POST to the endpoint"


async def test_invoke_maps_5xx_to_mcp_error() -> None:
    def server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    http_client, _ = make_strict_http_client(server_down)
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert exc.value.context["status_code"] == 503
    assert exc.value.context["wrapped"] == "HTTPStatusError"


async def test_invoke_maps_read_timeout_to_mcp_error() -> None:
    def slow(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    http_client, _ = make_strict_http_client(slow)
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert exc.value.context["wrapped"] == "ReadTimeout"


async def test_discover_maps_connect_error_to_mcp_error() -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, _ = make_strict_http_client(boom)
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert exc.value.context["wrapped"] == "ConnectError"
    assert exc.value.context["method"] == "tools/list"


# ---------------------------------------------------------------------------
# Cancellation leaves inside exception groups (never translated to MCPError)
# ---------------------------------------------------------------------------


def _transport_raising(exc: BaseException) -> Any:
    """A fake Transport whose connect() raises ``exc`` on __aenter__."""

    class _FailingTransport:
        def connect(self) -> AbstractAsyncContextManager[Any]:
            @asynccontextmanager
            async def _cm() -> AsyncIterator[Any]:
                raise exc
                yield  # pragma: no cover — unreachable; keeps this a generator

            return _cm()

    return _FailingTransport()


async def test_cancelled_error_leaf_in_exception_group_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CancelledError leaf in a mixed BaseExceptionGroup is re-raised, not
    translated into MCPError.

    Regression: consumer cancellation racing a real transport error can arrive
    as ``BaseExceptionGroup([ConnectError, CancelledError])`` out of the anyio
    task group; translating the group wholesale surfaced
    ``MCPError("MCP transport error: CancelledError")`` and broke the SDK's
    cancellation contract. The race itself is impractical to stage, so the
    mixed group is constructed directly and driven through ``discover``.
    """
    race = BaseExceptionGroup("race", [httpx.ConnectError("nope"), asyncio.CancelledError()])
    client = MCPClient(_config())
    monkeypatch.setattr(client, "_build_transport", lambda _headers: _transport_raising(race))
    with pytest.raises(asyncio.CancelledError):
        await client.discover()


async def test_cancelled_error_leaf_in_nested_exception_group_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same guarantee when the CancelledError is buried one group deeper
    (``_iter_leaf_exceptions`` flattens recursively before classification)."""
    race = BaseExceptionGroup(
        "outer",
        [
            BaseExceptionGroup("inner", [asyncio.CancelledError()]),
            httpx.ConnectError("nope"),
        ],
    )
    client = MCPClient(_config())
    monkeypatch.setattr(client, "_build_transport", lambda _headers: _transport_raising(race))
    with pytest.raises(asyncio.CancelledError):
        await client.invoke("search", {"q": "x"})


async def test_pure_exception_group_still_translates_to_mcp_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group whose leaves are all plain Exceptions still maps to MCPError."""
    group = ExceptionGroup("transport", [httpx.ConnectError("nope"), httpx.ReadTimeout("slow")])
    client = MCPClient(_config())
    monkeypatch.setattr(client, "_build_transport", lambda _headers: _transport_raising(group))
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert exc.value.context["wrapped"] == "ConnectError"


# ---------------------------------------------------------------------------
# initialize() RuntimeError -> MCPError (uniform-MCPError contract)
# ---------------------------------------------------------------------------


async def test_initialize_runtime_error_surfaces_as_mcp_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``RuntimeError`` from ``initialize()`` must NOT leak.

    ``ClientSession.initialize()`` raises a bare ``RuntimeError`` on an
    unsupported protocol version. It runs inside the transport's anyio task
    group, so it can surface bare OR nested in an ``ExceptionGroup`` — either
    way the wrapper must translate it into :class:`MCPError` (uniform-MCPError
    contract; plan step 5). Here we patch the session's request layer so
    ``initialize()`` sees an unsupported version, driving the REAL
    ``StreamableHttpTransport``.
    """
    import mcp.types as mtypes
    from mcp.client.session import ClientSession

    async def _bad_initialize_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        return mtypes.InitializeResult(
            protocolVersion="1999-01-01",  # unsupported -> RuntimeError
            capabilities=mtypes.ServerCapabilities(),
            serverInfo=mtypes.Implementation(name="x", version="1"),
        )

    monkeypatch.setattr(ClientSession, "send_request", _bad_initialize_request)

    def ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"",
        )

    http_client, _ = make_strict_http_client(ok)
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    # Translated, not a bare RuntimeError.
    assert exc.value.context["operation"] == "discover"
    assert exc.value.context["server_url"] == MCP_URL
    assert exc.value.context["wrapped"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Auth handling + redaction
# ---------------------------------------------------------------------------


async def test_auth_header_redacted_from_error_context() -> None:
    """Forcing a 5xx must NOT leak the auth header into MCPError.context."""

    def server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    http_client, _ = make_strict_http_client(server_down)
    client = MCPClient(
        _config(),
        auth={"Authorization": "Bearer SECRET-DO-NOT-LEAK"},
        client=http_client,
    )
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    serialized = json.dumps(exc.value.context, default=str)
    assert "SECRET-DO-NOT-LEAK" not in serialized
    assert "Bearer" not in serialized
    assert "Authorization" not in serialized


async def test_auth_header_redacted_when_callable_used() -> None:
    """Same redaction guarantee with a callable auth provider."""

    def server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    http_client, _ = make_strict_http_client(server_down)

    async def auth_provider() -> dict[str, str]:
        return {"X-Auth-Token": "ROTATING-SECRET"}

    client = MCPClient(_config(), auth=auth_provider, client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert "ROTATING-SECRET" not in json.dumps(exc.value.context, default=str)


async def test_callable_auth_resolved_per_call() -> None:
    """A callable auth provider is invoked fresh on each call (token rotation)."""
    counter = {"n": 0}

    async def rotating_auth() -> dict[str, str]:
        counter["n"] += 1
        return {"Authorization": f"Bearer token-{counter['n']}"}

    # A connect error short-circuits before any real network — but auth is
    # resolved each call (per-call resolution), so the counter advances.
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, _ = make_strict_http_client(boom)
    client = MCPClient(_config(), auth=rotating_auth, client=http_client)
    with pytest.raises(MCPError):
        await client.discover()
    with pytest.raises(MCPError):
        await client.discover()
    assert counter["n"] == 2


async def test_callable_auth_resolved_exactly_once_per_invoke(
    fastmcp_server: FastMCP,
) -> None:
    """A SUCCESSFUL invoke() must call a side-effecting auth provider once.

    Regression for the double-resolution bug: invoke() previously resolved the
    callable auth provider twice (a fail-fast pre-call AND inside the
    transport build), risking a double-spend for one-time-token providers.
    """
    counter = {"n": 0}

    async def one_time_token() -> dict[str, str]:
        counter["n"] += 1
        return {"Authorization": f"Bearer token-{counter['n']}"}

    async with make_compat_client(fastmcp_server, auth=one_time_token) as client:
        result = await client.invoke("search", {"q": "kittens"})
        assert isinstance(result, list)  # successful call
    assert counter["n"] == 1, f"auth provider invoked {counter['n']}x; expected exactly 1"


async def test_static_auth_header_propagated_to_request() -> None:
    """A static auth header reaches the outbound httpx request."""

    captured: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization", "")
        # Fail fast after capture — we only care about the header.
        raise httpx.ConnectError("captured")

    http_client, _ = make_strict_http_client(capture)
    client = MCPClient(
        _config(),
        auth={"Authorization": "Bearer test-token"},
        client=http_client,
    )
    with pytest.raises(MCPError):
        await client.discover()
    assert captured["authorization"] == "Bearer test-token"


async def test_invalid_auth_callable_return_raises_mcp_error() -> None:
    """A misbehaved auth callable surfaces a clear :class:`MCPError`."""

    def _never_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be invoked")

    http_client, mock = make_strict_http_client(_never_called)

    async def bad_auth() -> Any:
        return "not-a-mapping"

    client = MCPClient(_config(), auth=bad_auth, client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("x", {})
    assert exc.value.context["wrapped"] == "str"
    assert mock.observed_requests == [], "auth must fail before any request"


async def test_raising_auth_callable_raises_mcp_error() -> None:
    """A RAISING auth callable (e.g. token endpoint down) surfaces as MCPError.

    Regression: a bad return TYPE was already validated into MCPError, but an
    auth callable that raised propagated the raw exception — past the
    "returns or raises MCPError" contract and into the Registry's
    non-``AgentSdkError`` branch, which downgrades it to a model-recoverable
    ``ToolResult(is_error=True)`` instead of the fatal failure it is. The
    exception TEXT must not leak into the MCPError (an auth error message may
    carry credential material); only the type name is captured.
    """

    def _never_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be invoked")

    http_client, mock = make_strict_http_client(_never_called)

    async def down_token_endpoint() -> dict[str, str]:
        raise httpx.ConnectError("https://secret-token-endpoint.internal unreachable")

    client = MCPClient(_config(), auth=down_token_endpoint, client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("x", {})
    assert exc.value.context["wrapped"] == "ConnectError"
    assert exc.value.context["server_url"] == MCP_URL
    serialized = json.dumps(exc.value.context, default=str) + exc.value.message
    assert "secret-token-endpoint" not in serialized
    assert mock.observed_requests == [], "auth must fail before any request"


async def test_cancelled_auth_callable_propagates_cancellation() -> None:
    """A CancelledError out of the auth callable is NOT translated into
    MCPError — consumer cancellation must propagate (CancelledError is a
    BaseException on 3.11+, so the ``except Exception`` guard skips it)."""

    def _never_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be invoked")

    http_client, _ = make_strict_http_client(_never_called)

    async def cancelled_auth() -> dict[str, str]:
        raise asyncio.CancelledError

    client = MCPClient(_config(), auth=cancelled_auth, client=http_client)
    with pytest.raises(asyncio.CancelledError):
        await client.invoke("x", {})


# ---------------------------------------------------------------------------
# user_agent config field set on the owned httpx client
# ---------------------------------------------------------------------------


async def test_user_agent_header_set_on_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MCPClientConfig.user_agent`` reaches the wire on the owned-client path.

    No client is injected, so the transport builds its own
    ``httpx.AsyncClient`` with the configured ``User-Agent``. The construction
    is intercepted — the network transport swapped for the strict mock, the
    headers the transport passes left untouched — and the captured request
    must carry the configured value. (The previous version of this test
    injected a client that ALREADY carried the header, so it passed even with
    the ``user_agent`` plumbing deleted.)
    """
    captured: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["user-agent"] = request.headers.get("user-agent", "")
        # Fail fast after capture — we only care about the header.
        raise httpx.ConnectError("captured")

    mock = StrictTransportMock(capture)
    real_async_client = httpx.AsyncClient

    def owned_client_factory(**kwargs: Any) -> httpx.AsyncClient:
        # Keep the timeout/headers the transport passes; swap only the network
        # transport so the request is capturable without real I/O.
        kwargs["transport"] = httpx.MockTransport(mock.handle)
        return real_async_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", owned_client_factory)

    client = MCPClient(_config(user_agent="my-agent/2.0"))  # owned-client path
    with pytest.raises(MCPError):
        await client.discover()
    assert mock.observed_requests, "expected the request to reach the mock transport"
    assert captured["user-agent"] == "my-agent/2.0"


async def test_user_agent_not_applied_to_injected_client() -> None:
    """An injected client's own headers win — ``user_agent`` is NOT applied.

    Documents the injected-client contract: the transport merges only the
    resolved auth headers onto an externally-provided client, so the
    ``User-Agent`` the server sees is the injected client's own (here the
    httpx default), never ``MCPClientConfig.user_agent``.
    """
    captured: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        captured["user-agent"] = request.headers.get("user-agent", "")
        raise httpx.ConnectError("captured")

    http_client, _ = make_strict_http_client(capture)
    client = MCPClient(_config(user_agent="my-agent/2.0"), client=http_client)
    with pytest.raises(MCPError):
        await client.discover()
    assert captured["user-agent"] == http_client.headers["user-agent"]
    assert captured["user-agent"] != "my-agent/2.0"


# ---------------------------------------------------------------------------
# aclose() lifecycle
# ---------------------------------------------------------------------------


async def test_aclose_disposes_owned_client(fastmcp_server: FastMCP) -> None:
    """An owned client closes on aclose; post-close calls raise MCPError."""
    # Use the compat harness to get a working client, then assert the
    # closed-guard maps the "closed" condition into the uniform MCPError
    # contract (TD-007 item 2).
    async with make_compat_client(fastmcp_server) as client:
        await client.discover()  # works
        await client.aclose()
        with pytest.raises(MCPError) as exc:
            await client.discover()
    assert "closed" in exc.value.message.lower()
    assert exc.value.context["operation"] == "discover"


async def test_aclose_is_idempotent(fastmcp_server: FastMCP) -> None:
    """A second ``aclose()`` is a no-op and MUST NOT raise (TD-007 item 2)."""
    async with make_compat_client(fastmcp_server) as client:
        await client.discover()
        await client.aclose()
        await client.aclose()  # second call short-circuits cleanly


async def test_owned_client_aclose_closes_httpx_client() -> None:
    """When MCPClient owns its httpx client, aclose() closes it."""
    client = MCPClient(_config())
    # No client injected -> owns one lazily; aclose with no client is a no-op
    # set-closed (session-per-call builds the client inside the transport).
    await client.aclose()
    with pytest.raises(MCPError) as exc:
        await client.invoke("x", {})
    assert "closed" in exc.value.message.lower()
    assert exc.value.context["operation"] == "invoke"


async def test_caller_provided_client_not_disposed() -> None:
    """The caller-owned http client stays usable after aclose()."""

    def ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    http_client, _ = make_strict_http_client(ok)
    client = MCPClient(_config(), client=http_client)
    await client.aclose()
    # The caller-owned client is still usable.
    response = await http_client.post(MCP_URL, json={})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Docstring-drift regression — MCPError.context allow-list
# ---------------------------------------------------------------------------


async def test_mcperror_context_keys_match_class_docstring_allowlist(
    fastmcp_server: FastMCP,
) -> None:
    """Pin the ``MCPClient`` docstring's ``MCPError.context`` allow-list.

    Derives BOTH sides programmatically — the documented set by parsing
    ``MCPClient.__doc__``, the runtime set by triggering one representative
    :class:`MCPError` per distinct context-key group — and asserts set
    equality in both directions, so a doc edit OR a code edit that drifts the
    two apart fails here instead of rotting silently.
    """
    # --- (a) documented set: parse the brace-delimited allow-list ----------
    doc = MCPClient.__doc__
    assert doc is not None, "MCPClient must carry a class docstring"
    brace_match = re.search(r"allow-list\s*``\{(?P<keys>[^}]*)\}``", doc, re.DOTALL)
    assert brace_match is not None, (
        "could not locate the brace-delimited allow-list after the "
        "'allow-list' keyword in MCPClient.__doc__"
    )
    documented: set[str] = {
        token.strip() for token in brace_match.group("keys").split(",") if token.strip()
    }

    # --- (b) runtime set: trigger one MCPError per context-key group ------
    runtime: set[str] = set()

    def _capture(exc: MCPError) -> None:
        runtime.update(exc.context.keys())

    # Group 1: post-aclose call -> operation, server_url.
    # (A per-call isError=True no longer raises MCPError — it returns a
    # _MCPCallError, BR-005 — so it contributes no MCPError.context keys. The
    # tool_name/method/server_url keys are still exercised by the transport
    # groups below, which carry invoke()'s base_context.)
    async with make_compat_client(fastmcp_server) as compat:
        await compat.aclose()
        with pytest.raises(MCPError) as exc:
            await compat.discover()
        _capture(exc.value)

    # Group 3: transport 5xx -> status_code, wrapped (+ server_url, method).
    def _server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    down_http, _ = make_strict_http_client(_server_down)
    down_client = MCPClient(_config(), client=down_http)
    with pytest.raises(MCPError) as exc:
        await down_client.invoke("search", {"q": "x"})
    _capture(exc.value)

    # Group 4: connect error -> wrapped.
    def _boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    boom_http, _ = make_strict_http_client(_boom)
    boom_client = MCPClient(_config(), client=boom_http)
    with pytest.raises(MCPError) as exc:
        await boom_client.discover()
    _capture(exc.value)

    # Group 5: misbehaved auth callable -> wrapped (+ server_url).
    async def _bad_auth() -> Any:
        return 123

    bad_http, _ = make_strict_http_client(_boom)
    bad_client = MCPClient(_config(), auth=_bad_auth, client=bad_http)
    with pytest.raises(MCPError) as exc:
        await bad_client.invoke("x", {})
    _capture(exc.value)

    # Group 6: JSON-RPC McpError -> error_code, error_data. A 404 from the
    # endpoint surfaces (inside the transport task group) as a real
    # mcp.shared.exceptions.McpError, which the wrapper routes through the
    # protocol mapping. (An unknown *tool* returns isError=True, not an
    # McpError, so it would not exercise this key group.)
    def _not_found(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    nf_http, _ = make_strict_http_client(_not_found)
    nf_client = MCPClient(_config(), client=nf_http)
    with pytest.raises(MCPError) as exc:
        await nf_client.discover()
    _capture(exc.value)
    assert "error_code" in exc.value.context

    # --- assert: both directions, with offending keys named ---------------
    undocumented = runtime - documented
    stale = documented - runtime
    assert undocumented == set(), (
        f"MCPError.context emits key(s) not in the class-docstring "
        f"allow-list: {sorted(undocumented)}"
    )
    assert stale == set(), (
        f"class-docstring allow-list names key(s) the runtime never "
        f"emits (stale doc): {sorted(stale)}"
    )
