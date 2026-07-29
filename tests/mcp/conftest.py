"""Test harnesses for :class:`fifty_agent_sdk.mcp.client.MCPClient`.

The client now wraps the official ``mcp`` SDK's :class:`mcp.ClientSession`,
so the protocol wire / JSON-RPC envelope is owned by ``mcp`` and is validated
by the in-memory **official-client oracle** rather than a hand-rolled
strict-JSON-RPC mock. This module ships two complementary harnesses:

1. :class:`InMemorySessionTransport` + :func:`make_compat_client` — drive our
   ``MCPClient`` mapping/unwrap code through a REAL official
   :class:`mcp.ClientSession` connected to a :class:`FastMCP` server via
   :func:`mcp.shared.memory.create_connected_server_and_client_session`. This
   is the compatibility oracle (learning #760 pt6): the official client/server
   pair certifies handshake + envelope; our wrapper is NOT its own oracle.

2. A thin httpx-level mock (``httpx.MockTransport``) used ONLY for the
   transport-error + redaction paths (connect error / timeout / HTTP status).
   These flow through the real ``streamable_http_client`` over an injected
   ``httpx.AsyncClient``, so the failure surfaces exactly as production would
   (an anyio ``ExceptionGroup`` wrapping the httpx exception). The mock
   asserts request shape (L-356): every request must POST to the canonical
   URL — it is not a permissive blob.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from fifty_agent_sdk.mcp import MCPClient, MCPClientConfig

MCP_URL = "https://mcp.test.local/mcp"
"""Single canonical URL the thin transport-error mock serves."""

ToolHandler = Callable[[dict[str, Any]], Any]
"""``arguments`` -> a value placed into ``structuredContent`` of a result."""


# ---------------------------------------------------------------------------
# In-memory official-client oracle harness
# ---------------------------------------------------------------------------


class InMemorySessionTransport:
    """A test-only :class:`fifty_agent_sdk.mcp.transport.Transport`.

    Yields a pre-built, already-``initialize()``d official
    :class:`mcp.ClientSession` (the one created by
    :func:`create_connected_server_and_client_session`). It satisfies the
    ``Transport`` protocol structurally so the SAME ``MCPClient`` mapping and
    unwrap code runs against a real official client/server pair.
    """

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[ClientSession]:
        yield self._session


@asynccontextmanager
async def make_compat_client(
    server: FastMCP,
    *,
    auth: Any = None,
    on_tool_error: Any = None,
) -> AsyncIterator[MCPClient]:
    """Connect a real official session to ``server`` and wire an ``MCPClient``.

    The returned client's ``_build_transport`` is replaced with one that
    yields the in-memory official session, so ``discover``/``invoke`` exercise
    the production mapping/unwrap path against the official client/server pair.

    ``on_tool_error`` is forwarded verbatim into the ``MCPClient`` constructor
    (BR-010). Defaulting it to ``None`` keeps every existing call site — and the
    behaviour they pin — unchanged.
    """
    config = MCPClientConfig(base_url=MCP_URL)
    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        client = MCPClient(config, auth=auth, on_tool_error=on_tool_error)
        transport = InMemorySessionTransport(session)

        def _build_transport(headers: Any) -> InMemorySessionTransport:
            # discover()/invoke() resolve auth (once) BEFORE calling this, so
            # the misbehaved-callable fail-fast already happened upstream; this
            # override just yields the in-memory official session.
            return transport

        client._build_transport = _build_transport  # type: ignore[method-assign]
        yield client


# ---------------------------------------------------------------------------
# Controllable in-memory server for the MCPProvider consumer regression
# ---------------------------------------------------------------------------
#
# The MCPProvider tests need a MUTABLE catalog (refresh re-discovers a changed
# catalog) and per-tool handlers — FastMCP declares its tools at build time, so
# instead we feed a programmable "server" through the REAL MCPClient mapping and
# unwrap code via a fake session that returns genuine mcp.types objects. This
# keeps MCPProvider exercising the production MCPClient.discover()/invoke()
# contract while letting tests mutate the advertised tools mid-test.


class ControllableServer:
    """A mutable tool catalog + handlers, surfaced as ``mcp.types`` results."""

    def __init__(self) -> None:
        self._catalog: list[dict[str, Any]] = []
        self._handlers: dict[str, ToolHandler] = {}
        self.list_calls: int = 0

    def set_tool_catalog(self, defs: list[dict[str, Any]]) -> None:
        self._catalog = list(defs)

    def register_tool(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def list_tools_result(self) -> ListToolsResult:
        self.list_calls += 1
        tools = [
            Tool(
                name=d["name"],
                description=d.get("description", ""),
                inputSchema=d.get(
                    "inputSchema", {"type": "object", "properties": {}, "required": []}
                ),
            )
            for d in self._catalog
        ]
        return ListToolsResult(tools=tools)

    def call_tool_result(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            # Unknown/unregistered tool -> isError (carries tool_name via the
            # MCPClient unwrap), mirroring the old strict mock's -32602 intent
            # at the MCPError-contract boundary the provider tests assert on.
            return CallToolResult(
                content=[TextContent(type="text", text=f"unknown tool {name!r}")],
                isError=True,
            )
        payload = handler(arguments)
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent=payload if isinstance(payload, dict) else {"result": payload},
            isError=False,
        )


class _ControllableSession:
    """Minimal session satisfying the calls MCPClient makes on a session."""

    def __init__(self, server: ControllableServer) -> None:
        self._server = server

    async def list_tools(self) -> ListToolsResult:
        return self._server.list_tools_result()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        return self._server.call_tool_result(name, arguments or {})


class _ControllableTransport:
    """Yields a :class:`_ControllableSession` — satisfies the Transport seam."""

    def __init__(self, server: ControllableServer) -> None:
        self._server = server

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Any]:
        yield _ControllableSession(self._server)


def make_controllable_client(
    server: ControllableServer,
    *,
    on_tool_error: Any = None,
) -> MCPClient:
    """Wire an ``MCPClient`` whose transport is driven by ``server``.

    The returned client runs the production mapping/unwrap/error-translation
    code, but its catalog/handlers are programmable and mutable mid-test.

    ``on_tool_error`` is forwarded verbatim into the ``MCPClient`` constructor
    (BR-010); the ``None`` default keeps every existing call site unchanged.
    """
    client = MCPClient(MCPClientConfig(base_url=MCP_URL), on_tool_error=on_tool_error)
    transport = _ControllableTransport(server)

    def _build_transport(headers: Any) -> _ControllableTransport:
        return transport

    client._build_transport = _build_transport  # type: ignore[method-assign]
    return client


@pytest.fixture
def controllable_server() -> ControllableServer:
    """A fresh mutable controllable server per test."""
    return ControllableServer()


# ---------------------------------------------------------------------------
# Thin httpx-level transport-error mock (L-356: asserts request shape)
# ---------------------------------------------------------------------------


class StrictTransportMock:
    """Records and shape-asserts every httpx request before delegating.

    Used only by the transport-error / redaction tests. It is NOT a permissive
    blob: every request must be a POST to :data:`MCP_URL`. The wrapped handler
    decides the actual response/exception (connect error, timeout, HTTP
    status). The point is to fail loudly if the wrapper ever stops POSTing to
    the configured endpoint.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self.observed_requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST", (
            f"StrictTransportMock expected POST; got {request.method!r}"
        )
        assert str(request.url) == MCP_URL, (
            f"StrictTransportMock expected URL {MCP_URL!r}; got {str(request.url)!r}"
        )
        self.observed_requests.append(request)
        return self._handler(request)


def make_strict_http_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[httpx.AsyncClient, StrictTransportMock]:
    """Build an injectable httpx client whose transport asserts request shape.

    Returns the client and the :class:`StrictTransportMock` so a test can
    inspect ``observed_requests``.
    """
    mock = StrictTransportMock(handler)
    client = httpx.AsyncClient(transport=httpx.MockTransport(mock.handle))
    return client, mock


@pytest.fixture
def fastmcp_server() -> FastMCP:
    """A fresh FastMCP test server with sanitised tools per test."""
    from ._fastmcp_server import build_test_server

    return build_test_server()
