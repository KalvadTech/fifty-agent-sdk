"""Tests for :class:`fifty_agent_sdk.tools.mcp_provider.MCPProvider`.

These tests pair each scenario from the BR-008 plan §9 with a controllable
in-memory MCP server (:class:`tests.mcp.conftest.ControllableServer`) driven
through the REAL :class:`fifty_agent_sdk.mcp.client.MCPClient` mapping/unwrap code.
The provider tests live in ``tests/tools/`` rather than ``tests/mcp/`` so they
sit alongside the rest of the tool-layer suite — they exercise the bridge
between protocol and registry, not the protocol itself.

This file is the primary "vendored consumers unaffected" regression for
BR-042: it asserts the ``MCPProvider`` behavior is byte-for-byte stable across
the swap to the official ``mcp`` SDK. Only the client-construction helper was
rewired to the new harness; no ``MCPProvider`` assertion changed.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import httpx
import pytest
import structlog

from fifty_agent_sdk.errors import MCPError
from fifty_agent_sdk.mcp import MCPClient, MCPClientConfig, MCPToolDef
from fifty_agent_sdk.tools.mcp_provider import (
    MCPProvider,
    RefreshSummary,
    _MCPToolAdapter,
    _to_tool_schema,
)
from fifty_agent_sdk.tools.protocol import ToolResult, ToolSchema
from fifty_agent_sdk.tools.registry import Registry
from tests.mcp.conftest import (
    MCP_URL,
    ControllableServer,
    make_controllable_client,
    make_strict_http_client,
)


def _tool_def(name: str, *, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Tool {name}",
        "inputSchema": schema
        if schema is not None
        else {"type": "object", "properties": {}, "required": []},
    }


# ---------------------------------------------------------------------------
# P1 — attach() registers one adapter per tool
# ---------------------------------------------------------------------------


async def test_attach_registers_one_adapter_per_tool(
    controllable_server: ControllableServer,
) -> None:
    controllable_server.set_tool_catalog(
        [_tool_def("alpha"), _tool_def("beta"), _tool_def("gamma")]
    )
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)

    names = sorted(t.name for t in registry.list())
    assert names == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# P2 / P3 — schema translation
# ---------------------------------------------------------------------------


def test_schema_translation_identity() -> None:
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
    )
    assert isinstance(schema, ToolSchema)
    assert schema.type == "object"
    assert schema.properties == {"q": {"type": "string"}}
    assert schema.required == ["q"]
    assert schema.additionalProperties is False


def _collect_refs(node: Any) -> list[str]:
    """Recursively collect every ``$ref`` value in a schema structure."""
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.append(value)
            else:
                refs.extend(_collect_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_collect_refs(item))
    return refs


def test_schema_translation_inlines_defs_and_drops_unknown_top_level_keys() -> None:
    """``$defs`` is not just dropped: local ``#/$defs/...`` refs are inlined
    first, so no dangling pointer reaches the LLM-facing schema. Other unknown
    top-level keys (``examples``, …) are still silently dropped — ToolSchema is
    ``extra="forbid"``."""
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {"addr": {"$ref": "#/$defs/Address"}},
            "required": ["addr"],
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
            "examples": [{}],
        }
    )
    assert schema.type == "object"
    assert _collect_refs(schema.properties) == []
    addr = schema.properties["addr"]
    assert addr["type"] == "object"
    assert addr["properties"]["city"] == {"type": "string"}
    assert addr["required"] == ["city"]
    assert schema.required == ["addr"]


def test_schema_translation_inlines_nested_defs_recursively() -> None:
    """A def that itself refs another def must resolve to full depth."""
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {"order": {"$ref": "#/$defs/Order"}},
            "$defs": {
                "Order": {
                    "type": "object",
                    "properties": {"shipping": {"$ref": "#/$defs/Address"}},
                },
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    )
    assert _collect_refs(schema.properties) == []
    shipping = schema.properties["order"]["properties"]["shipping"]
    assert shipping["properties"]["city"] == {"type": "string"}


def test_schema_translation_ref_sibling_keys_are_preserved() -> None:
    """Sibling keys next to a ``$ref`` (JSON Schema 2020-12 conjunctive
    semantics) are merged over the resolved definition, not discarded."""
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {"addr": {"$ref": "#/$defs/Address", "description": "where"}},
            "$defs": {"Address": {"type": "object", "properties": {}}},
        }
    )
    addr = schema.properties["addr"]
    assert addr["description"] == "where"
    assert addr["type"] == "object"


def test_schema_translation_leaves_unresolvable_refs_untouched() -> None:
    """A ref the resolver cannot resolve locally (missing def, external URI)
    passes through — inlining must not invent information the server schema
    did not carry."""
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {
                "x": {"$ref": "#/$defs/Missing"},
                "y": {"$ref": "https://schemas.example.com/ext"},
            },
        }
    )
    assert schema.properties["x"]["$ref"] == "#/$defs/Missing"
    assert schema.properties["y"]["$ref"] == "https://schemas.example.com/ext"


def test_schema_translation_defs_cycle_falls_back_to_empty_schema() -> None:
    """A cyclic server schema (a def that transitively refs itself) has no
    finite inline expansion; untrusted server data takes the same defensive
    fallback as a non-object schema rather than aborting discovery."""
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {"node": {"$ref": "#/$defs/Node"}},
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                }
            },
        }
    )
    assert schema.type == "object"
    assert schema.properties == {}
    assert schema.required == []


def test_schema_translation_falls_back_for_non_object_top_level() -> None:
    schema = _to_tool_schema({"type": "string"})
    assert schema.type == "object"
    assert schema.properties == {}
    assert schema.required == []


# ---------------------------------------------------------------------------
# P4 / P5 / P6 — adapter invoke + error propagation
# ---------------------------------------------------------------------------


async def test_adapter_invoke_returns_tool_result_on_success(
    controllable_server: ControllableServer,
) -> None:
    controllable_server.set_tool_catalog([_tool_def("answer")])
    controllable_server.register_tool("answer", lambda _args: {"answer": 42})
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    adapter = registry.get("answer")
    result = await adapter.invoke({})
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.output == {"answer": 42}


async def test_adapter_maps_is_error_to_tool_result(
    controllable_server: ControllableServer,
) -> None:
    """A per-call ``isError=True`` becomes a recoverable ToolResult, not a raise.

    BR-005: an unregistered tool on the ControllableServer returns a real
    ``CallToolResult(isError=True)``; the adapter maps it to
    ``ToolResult(is_error=True)`` carrying the tool name in ``.error`` and does
    NOT raise.
    """
    # No handler registered -> isError result -> recoverable ToolResult.
    controllable_server.set_tool_catalog([_tool_def("missing")])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    adapter = registry.get("missing")
    result = await adapter.invoke({})
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert result.output is None
    assert result.error is not None
    assert "missing" in result.error


async def test_registry_invoke_maps_is_error_to_tool_result(
    controllable_server: ControllableServer,
) -> None:
    """Registry.invoke() passes the adapter's recoverable ToolResult through.

    BR-005: the per-call ``isError`` path no longer raises an AgentSdkError, so
    the registry returns the ``ToolResult(is_error=True)`` end-to-end instead of
    re-raising.
    """
    controllable_server.set_tool_catalog([_tool_def("missing")])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    result = await registry.invoke("missing", {}, timeout=1.0)
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "missing" in (result.error or "")


async def test_on_tool_error_replacement_reaches_tool_result_error(
    controllable_server: ControllableServer,
) -> None:
    """The screened string is what a consumer reads off ``ToolResult.error`` (BR-010).

    End-to-end through ``MCPProvider`` + ``Registry.invoke``, and — the point of
    the brief — reachable by PLAIN CONSTRUCTION: an ``MCPClient`` built with
    ``on_tool_error=`` handed to ``MCPProvider(client)``. Nothing here subclasses
    ``MCPClient`` or ``_MCPToolAdapter``, and nothing imports a private symbol to
    make the seam work.
    """
    seen: list[tuple[str, list[dict[str, Any]]]] = []

    def screen(message: str, content: list[dict[str, Any]]) -> str:
        seen.append((message, content))
        return "the upstream tool is unavailable"

    controllable_server.set_tool_catalog([_tool_def("missing")])
    client = make_controllable_client(controllable_server, on_tool_error=screen)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)

    result = await registry.invoke("missing", {}, timeout=1.0)
    assert isinstance(result, ToolResult)
    assert result.error == "the upstream tool is unavailable"
    # The hook saw the SDK's bounded default plus the server's raw content.
    assert len(seen) == 1
    assert "missing" in seen[0][0]
    assert isinstance(seen[0][1], list)


async def test_on_tool_error_replacement_still_yields_is_error_true_and_no_output(
    controllable_server: ControllableServer,
) -> None:
    """A hook cannot confabulate success: ``is_error`` and ``output`` are untouchable.

    Learning #886's anti-confabulation guarantee under BR-010 — whatever the
    screen returns, a failed tool is still reported as ``output=None,
    is_error=True``.
    """
    controllable_server.set_tool_catalog([_tool_def("missing")])
    client = make_controllable_client(
        controllable_server,
        on_tool_error=lambda message, content: "everything is fine, nothing failed",
    )
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)

    adapter_result = await registry.get("missing").invoke({})
    registry_result = await registry.invoke("missing", {}, timeout=1.0)
    for result in (adapter_result, registry_result):
        assert isinstance(result, ToolResult)
        assert result.output is None
        assert result.is_error is True
        assert result.error == "everything is fine, nothing failed"


async def test_mcp_transport_error_still_raises() -> None:
    """A transport/connection failure STILL raises MCPError (fatal path).

    BR-005 guardrail (acceptance criterion 3): the recoverable/fatal split is
    structural — a dead connection is raised as :class:`MCPError` at the
    ``call_tool`` boundary, *before* ``_unwrap_invoke_result`` runs, so it can
    NEVER be downgraded to a recoverable ``ToolResult(is_error=True)``. This
    locks the split so a future change cannot silently mask an outage as a
    recoverable per-tool observation. Asserted through BOTH the adapter and the
    registry.
    """

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, _ = make_strict_http_client(boom)
    client = MCPClient(MCPClientConfig(base_url=MCP_URL), client=http_client)
    defn = MCPToolDef(name="search", description="search", input_schema={})
    adapter = _MCPToolAdapter(defn, client)

    # Through the adapter directly.
    with pytest.raises(MCPError) as exc:
        await adapter.invoke({"q": "x"})
    assert exc.value.context["wrapped"] == "ConnectError"

    # And end-to-end through the registry (which re-raises AgentSdkError).
    registry = Registry()
    registry.register(adapter)
    with pytest.raises(MCPError):
        await registry.invoke("search", {"q": "x"}, timeout=1.0)


async def test_mcp_transport_error_still_raises_with_error_hook_configured() -> None:
    """The BR-005 fatal guardrail holds with a BR-010 screen wired in.

    Same ``httpx.ConnectError`` scenario as the test above, but the client is
    constructed with ``on_tool_error=``. The failure must STILL raise
    :class:`MCPError` through BOTH the adapter and the registry, and the screen
    must never be called: the hook is applied strictly inside ``invoke``'s
    per-call-error branch, which a transport failure never reaches. This is the
    guardrail extended to cover the new seam — a screen can never downgrade a
    dead connection into a recoverable observation.
    """
    calls: list[tuple[str, list[dict[str, Any]]]] = []

    def screen(message: str, content: list[dict[str, Any]]) -> str:
        calls.append((message, content))
        return "should never be used"

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client, _ = make_strict_http_client(boom)
    client = MCPClient(
        MCPClientConfig(base_url=MCP_URL),
        client=http_client,
        on_tool_error=screen,
    )
    defn = MCPToolDef(name="search", description="search", input_schema={})
    adapter = _MCPToolAdapter(defn, client)

    with pytest.raises(MCPError) as exc:
        await adapter.invoke({"q": "x"})
    assert exc.value.context["wrapped"] == "ConnectError"

    registry = Registry()
    registry.register(adapter)
    with pytest.raises(MCPError):
        await registry.invoke("search", {"q": "x"}, timeout=1.0)

    assert calls == []


# ---------------------------------------------------------------------------
# P7 / P8 — refresh semantics
# ---------------------------------------------------------------------------


async def test_refresh_adds_new_tools_without_removing_existing(
    controllable_server: ControllableServer,
) -> None:
    controllable_server.set_tool_catalog([_tool_def("A"), _tool_def("B")])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    assert sorted(t.name for t in registry.list()) == ["A", "B"]

    # Catalog mutates: A disappears, C arrives.
    controllable_server.set_tool_catalog([_tool_def("B"), _tool_def("C")])
    summary = await provider.refresh()

    names = sorted(t.name for t in registry.list())
    assert names == ["A", "B", "C"]
    assert isinstance(summary, RefreshSummary)
    assert summary.added == 1  # C is new
    assert summary.refreshed == 1  # B already present


async def test_refresh_updates_existing_tool_definition(
    controllable_server: ControllableServer,
) -> None:
    """Last-write-wins: schema changes propagate to the registry."""
    controllable_server.set_tool_catalog(
        [
            _tool_def(
                "A",
                schema={
                    "type": "object",
                    "properties": {"v1": {"type": "string"}},
                    "required": ["v1"],
                },
            )
        ]
    )
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    assert "v1" in registry.get("A").schema.properties

    controllable_server.set_tool_catalog(
        [
            _tool_def(
                "A",
                schema={
                    "type": "object",
                    "properties": {"v2": {"type": "integer"}},
                    "required": ["v2"],
                },
            )
        ]
    )
    await provider.refresh()
    refreshed = registry.get("A")
    assert "v2" in refreshed.schema.properties
    assert "v1" not in refreshed.schema.properties


async def test_refresh_without_attach_raises_runtime_error(
    controllable_server: ControllableServer,
) -> None:
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    with pytest.raises(RuntimeError, match="prior attach"):
        await provider.refresh()


# ---------------------------------------------------------------------------
# P9 / P10 — periodic refresh
# ---------------------------------------------------------------------------


async def test_periodic_refresh_runs_and_can_be_cancelled(
    controllable_server: ControllableServer,
) -> None:
    controllable_server.set_tool_catalog([_tool_def("A")])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    assert controllable_server.list_calls == 1

    await provider.start_periodic_refresh(interval_seconds=0.02)
    await asyncio.sleep(0.07)
    await provider.aclose()

    # At least one periodic refresh fired (in addition to the attach one).
    assert controllable_server.list_calls >= 2


async def test_periodic_refresh_double_start_raises(
    controllable_server: ControllableServer,
) -> None:
    controllable_server.set_tool_catalog([])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    await provider.start_periodic_refresh(interval_seconds=10.0)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await provider.start_periodic_refresh(interval_seconds=10.0)
    finally:
        await provider.aclose()


async def test_periodic_refresh_non_positive_interval_raises(
    controllable_server: ControllableServer,
) -> None:
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    controllable_server.set_tool_catalog([])
    await provider.attach(registry)
    with pytest.raises(ValueError, match="must be > 0"):
        await provider.start_periodic_refresh(interval_seconds=0)


async def test_periodic_refresh_requires_prior_attach(
    controllable_server: ControllableServer,
) -> None:
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    with pytest.raises(RuntimeError, match="prior attach"):
        await provider.start_periodic_refresh(interval_seconds=1.0)


async def test_periodic_refresh_survives_transient_failure(
    controllable_server: ControllableServer,
) -> None:
    """A discover() failure inside the periodic loop must NOT terminate it."""
    controllable_server.set_tool_catalog([_tool_def("A")])
    state = {"fail_next": False}

    real_list = controllable_server.list_tools_result

    def flaky_list() -> Any:
        if state["fail_next"]:
            state["fail_next"] = False
            raise MCPError("transient", context={"server_url": "x"})
        return real_list()

    controllable_server.list_tools_result = flaky_list  # type: ignore[method-assign]

    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    await provider.start_periodic_refresh(interval_seconds=0.02)
    state["fail_next"] = True
    await asyncio.sleep(0.07)  # spans the failure + at least one success
    await provider.aclose()
    # Provider must still be functional after the failure.
    await provider.refresh()
    assert "A" in [t.name for t in registry.list()]


# ---------------------------------------------------------------------------
# P11 / P12 — async contract + lifecycle
# ---------------------------------------------------------------------------


def test_attach_is_async() -> None:
    assert inspect.iscoroutinefunction(MCPProvider.attach)


def test_refresh_is_async() -> None:
    assert inspect.iscoroutinefunction(MCPProvider.refresh)


async def test_aclose_does_not_close_underlying_client(
    controllable_server: ControllableServer,
) -> None:
    controllable_server.set_tool_catalog([])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    await provider.aclose()
    # The MCPClient remains usable (provider.aclose does not close it).
    await client.discover()


async def test_aclose_is_idempotent_when_no_task_running(
    controllable_server: ControllableServer,
) -> None:
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    await provider.aclose()  # no-op
    await provider.aclose()  # still safe


# ---------------------------------------------------------------------------
# P13 — name-collision warning
# ---------------------------------------------------------------------------


async def test_attach_warns_on_name_collision(
    controllable_server: ControllableServer,
) -> None:
    """Pre-register a tool with the same name; provider must log a warning."""

    class _LocalTool:
        name = "search"
        description = "local"
        schema = ToolSchema()

        async def invoke(self, args: dict[str, Any]) -> ToolResult:
            return ToolResult(output="local")

    registry = Registry()
    registry.register(_LocalTool())  # type: ignore[arg-type]

    controllable_server.set_tool_catalog([_tool_def("search")])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    with structlog.testing.capture_logs() as logs:
        await provider.attach(registry)

    overwrites = [
        entry
        for entry in logs
        if entry.get("event") == "mcp.tool_overwrite" and entry.get("name") == "search"
    ]
    assert len(overwrites) == 1, f"expected exactly one MCPProvider overwrite warning, got {logs}"
    assert overwrites[0]["log_level"] == "warning"


async def test_refresh_does_not_warn_for_first_time_tools(
    controllable_server: ControllableServer,
) -> None:
    """The collision warning fires ONLY for names already in the registry."""
    controllable_server.set_tool_catalog([_tool_def("new")])
    client = make_controllable_client(controllable_server)
    provider = MCPProvider(client)
    registry = Registry()
    with structlog.testing.capture_logs() as logs:
        await provider.attach(registry)
    overwrites = [e for e in logs if e.get("event") == "mcp.tool_overwrite"]
    assert overwrites == []
