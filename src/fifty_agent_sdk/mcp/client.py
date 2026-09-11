"""MCP client wrapping the official :mod:`mcp` Python SDK.

:class:`MCPClient` is a thin wrapper around the official ``mcp`` SDK's
:class:`mcp.ClientSession` over Streamable HTTP. The JSON-RPC wire, envelope
correlation, session-id handling, and protocol handshake are owned by
``mcp`` (pinned ``mcp>=1.27.0,<2.0.0``); this module is responsible only for
the fifty-agent-sdk-facing contract: typed :class:`MCPToolDef`s out of
``tools/list``, the unwrapped tool payload out of ``tools/call``, the uniform
:class:`fifty_agent_sdk.errors.MCPError` translation, and auth-header redaction.

Transport
    The protocol/transport machinery lives behind a
    :class:`fifty_agent_sdk.mcp.transport.Transport`. The default
    :class:`fifty_agent_sdk.mcp.transport.StreamableHttpTransport` establishes the
    Streamable HTTP stream pair and yields an already-``initialize()``d
    session. A future stdio transport implements the same seam without any
    change to :class:`MCPClient` or its consumers.

Session lifecycle — session-per-call
    Each :meth:`discover`/:meth:`invoke` opens a fresh session
    (``initialize`` → call → close). This preserves the original "no
    persistent connection" mental model, keeps :meth:`aclose` trivial, and
    lets callable ``auth`` resolve fresh per call (token rotation). A
    lazy-persistent single-session optimization is a deferred follow-up.

Authentication
    The constructor accepts an ``auth`` argument that is either a static
    ``Mapping[str, str]`` (e.g. ``{"Authorization": "Bearer ..."}``) or an
    async callable returning the same shape on each invocation. Callables
    enable token rotation without re-creating the client. In ``mcp 1.27.0``
    the supported auth/header seam is the injected :class:`httpx.AsyncClient`
    (transport-level ``headers=/auth=`` are deprecated and ignored), so
    resolved auth headers are merged onto the httpx client inside the
    transport. Auth header values are NEVER logged and NEVER appear in
    :class:`fifty_agent_sdk.errors.MCPError` context — header names registered
    through ``auth`` are tracked and redacted via :func:`_redact_headers`
    before any header dict is ever captured.

Retry policy
    ``connect_retries`` is forwarded to the owned httpx transport
    (``httpx.AsyncHTTPTransport(retries=...)``) as a connect-level retry.
    When an external ``client`` is injected this is a documented no-op (the
    injected client's transport governs). Application-level 5xx retry is out
    of scope — those surface as :class:`MCPError`.

Timeouts
    ``connect_timeout_seconds`` and ``read_timeout_seconds`` map to the owned
    httpx :class:`httpx.Timeout` and the session ``read_timeout_seconds``. A
    timeout surfaces as :class:`MCPError` with ``context["wrapped"]`` set to
    the httpx timeout class name.

Error contract
    Every public method either returns successfully or raises
    :class:`MCPError`. No ``mcp`` SDK exceptions (``McpError``) and no
    ``httpx`` exceptions leak — they are unwrapped from anyio
    ``ExceptionGroup``s and translated into :class:`MCPError`. An exception
    raised by a callable ``auth`` provider (e.g. a token endpoint being down)
    is translated the same way, with only the exception TYPE captured in
    ``context`` — never the exception text, which may carry credential
    material. Cancellation is never translated: a
    :class:`asyncio.CancelledError` propagates untouched, including when it
    arrives as a leaf of a mixed ``BaseExceptionGroup`` (cancellation racing
    a real transport error inside the anyio task group). The
    :class:`fifty_agent_sdk.tools.mcp_provider._MCPToolAdapter` deliberately does
    NOT catch :class:`MCPError` so the
    :class:`fifty_agent_sdk.tools.registry.Registry`'s ``AgentSdkError`` branch
    re-raises it untouched.

Error-content transformation seam (BR-010)
    An MCP server's ``isError`` payload is upstream-controlled prose. A
    consumer running against an untrusted or semi-trusted server needs to
    screen, redact, or reshape that text *before* it reaches the model. The
    supported, public seam is the keyword-only ``on_tool_error`` callback on
    :class:`MCPClient` — see :data:`MCPToolErrorHook` and the ``Args:`` entry on
    :class:`MCPClient`. It fires ONLY on a per-call ``isError=True`` outcome
    (never on success, and structurally never on a transport/protocol/session
    failure, which raises :class:`MCPError` at the ``call_tool`` boundary
    *before* the unwrap), and it can change only the bounded ``message``
    string — never ``is_error``, never ``output``, never the recoverable/fatal
    classification. Leaving it unset (the default) is byte-for-byte the
    pre-BR-010 behaviour: the ``None`` check short-circuits before the hook is
    called, inspected, or any new carrier is constructed.

    Reaching into the private :class:`_MCPCallError` / overriding
    :meth:`MCPClient._unwrap_invoke_result` is NOT supported; those symbols
    carry no semver protection. Use ``on_tool_error``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import httpx
import structlog
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel, ConfigDict, Field

from fifty_agent_sdk.errors import MCPError
from fifty_agent_sdk.mcp.transport import StreamableHttpTransport, Transport

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

DEBUG-level log lines include ``method``, ``server_url``, and a duration in
milliseconds. Headers are NEVER logged — the redaction contract is enforced
by funnelling all header dicts through :func:`_redact_headers` before any
log or :class:`MCPError` context capture. The underlying ``mcp`` transport
may log a session id at INFO; a session id is not a secret, but no auth
material ever reaches those lines because auth lives on the httpx client.
"""

_REDACTION_SENTINEL: Final[str] = "<redacted>"
"""Replacement value substituted for any redacted header value."""

# Header names that the SDK treats as sensitive regardless of whether the
# caller declared them via ``auth``. Kept lowercase for case-insensitive
# comparison.
_BUILTIN_SENSITIVE_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-auth-token",
        "cookie",
        "set-cookie",
    }
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class MCPToolDef(BaseModel):
    """A single MCP-advertised tool definition.

    Returned in batches by :meth:`MCPClient.discover`. ``input_schema`` is
    the raw JSON-Schema dict shipped by the MCP server; the
    :class:`fifty_agent_sdk.tools.mcp_provider.MCPProvider` translates it into a
    :class:`fifty_agent_sdk.tools.protocol.ToolSchema` at adapter-construction
    time.

    Attributes:
        name: Tool name. Matches the string the LLM emits and that the
            tool :class:`fifty_agent_sdk.tools.registry.Registry` uses as its
            dict key.
        description: Human-readable explanation rendered into the system
            prompt by the BR-003 prompt builder.
        input_schema: Raw JSON-Schema object (top-level ``"type": "object"``
            per the MCP spec). May be empty for parameter-less tools.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPClientConfig(BaseModel):
    """Connection configuration for :class:`MCPClient`.

    Attributes:
        base_url: Full URL of the MCP server's Streamable HTTP endpoint
            (e.g. ``"https://mcp.example.com/mcp"``). Used verbatim; the
            client does NOT append paths.
        connect_timeout_seconds: TCP connect timeout passed to
            :class:`httpx.Timeout`.
        read_timeout_seconds: Per-call read/write/pool timeout passed to
            :class:`httpx.Timeout` and the :class:`mcp.ClientSession`
            read timeout. The ``Registry``'s ``tool_timeout`` wraps the whole
            adapter call (loop safety budget); this httpx-level timeout is the
            inner budget and SHOULD be smaller than the outer one.
        connect_retries: Number of TCP connect retries for the owned
            :class:`httpx.AsyncHTTPTransport`. A documented no-op when an
            external ``client`` is injected (config-shape stability for
            vendored construction sites — kept, not dropped). Application-level
            5xx retry is out of scope; those responses surface as
            :class:`MCPError`.
        user_agent: Value of the ``User-Agent`` request header set on the
            owned httpx client. A documented no-op when an external ``client``
            is injected (that client's headers win). Kept for config-shape
            stability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    connect_retries: int = 0
    user_agent: str = "fifty-agent-sdk-mcp/0.0.1"


# ---------------------------------------------------------------------------
# Internal redaction helper
# ---------------------------------------------------------------------------


def _redact_headers(
    headers: Mapping[str, str],
    *,
    extra_sensitive_keys: frozenset[str],
) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values replaced.

    Sensitivity is determined case-insensitively against
    :data:`_BUILTIN_SENSITIVE_HEADERS` and ``extra_sensitive_keys`` (the
    header names declared by the caller's ``auth`` mapping).

    The returned dict is suitable for inclusion in :class:`MCPError`
    context or log lines. The sentinel value :data:`_REDACTION_SENTINEL`
    replaces every redacted value verbatim. No public method captures a
    header dict today — every :class:`MCPError` context is built from the
    fixed allow-list documented on :class:`MCPClient` — but any future path
    that surfaces headers MUST funnel through this helper first.
    """
    sensitive = _BUILTIN_SENSITIVE_HEADERS | extra_sensitive_keys
    return {k: (_REDACTION_SENTINEL if k.lower() in sensitive else v) for k, v in headers.items()}


def _iter_leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten an anyio :class:`ExceptionGroup` into its leaf exceptions.

    The ``mcp`` transport runs the stream pair inside an anyio task group, so
    a transport/protocol failure surfaces as an ``ExceptionGroup`` wrapping
    the real cause (``httpx.ConnectError``, ``httpx.ReadTimeout``,
    ``httpx.HTTPStatusError``, or :class:`mcp.shared.exceptions.McpError`).
    A non-group exception returns as a single-element list.
    """
    if isinstance(exc, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for sub in exc.exceptions:
            leaves.extend(_iter_leaf_exceptions(sub))
        return leaves
    return [exc]


# ---------------------------------------------------------------------------
# Per-call error carrier (recoverable `isError` outcome)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MCPCallError:
    """A per-call ``tools/call`` ``isError=True`` outcome (recoverable).

    Returned (NOT raised) by :meth:`MCPClient._unwrap_invoke_result` when a
    ``tools/call`` completes the transport/protocol/session round-trip
    successfully but the server marks the *result* as an error
    (``CallToolResult.isError is True``). The
    :class:`fifty_agent_sdk.tools.mcp_provider._MCPToolAdapter` maps this into a
    recoverable :class:`fifty_agent_sdk.tools.protocol.ToolResult` with
    ``is_error=True`` so the agent loop reasons about it as a tool observation
    rather than terminating the run (BR-005).

    This is deliberately a plain frozen dataclass and NOT an
    :class:`fifty_agent_sdk.errors.AgentSdkError`/:class:`Exception` subclass: if
    it were an exception, the tool
    :class:`fifty_agent_sdk.tools.registry.Registry`'s ``except AgentSdkError``
    branch (registry.py) would re-raise it untouched and defeat the recoverable
    mapping. The recoverable/fatal split is structural — transport, protocol,
    and session failures are raised as :class:`MCPError` *before*
    ``_unwrap_invoke_result`` is reached, so a value of this type can only ever
    represent a per-call result error, never a connection failure.

    Attributes:
        message: A bounded, model-safe description of the failure
            (``"MCP tool '<name>' returned isError=True"``). This is the ONLY
            field surfaced into the model-facing observation.
        content: The server's error content blocks serialised verbatim to plain
            JSON dicts (``[c.model_dump(mode="json") for c in result.content]``).
            Retained for logs/diagnostics ONLY — see the security note on
            :meth:`MCPClient._unwrap_invoke_result`. The adapter MUST NOT splice
            this into the model observation unredacted.
    """

    message: str
    content: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


AuthHeaders = Mapping[str, str]
AuthCallable = Callable[[], Awaitable[AuthHeaders]]
AuthSpec = AuthHeaders | AuthCallable | None

MCPToolErrorHook: TypeAlias = Callable[[str, list[dict[str, Any]]], str | Awaitable[str]]
"""Public type of the :class:`MCPClient` ``on_tool_error`` callback (BR-010).

Called positionally as ``hook(message, content)`` exactly once per per-call
``tools/call`` ``isError=True`` outcome, and never on any other path:

* ``message`` — the SDK's bounded, model-safe default
  (``"MCP tool '<name>' returned isError=True"``).
* ``content`` — the server's raw error content blocks, verbatim, as the SAME
  list object retained on the internal per-call error carrier. It MUST be
  treated as READ-ONLY; the SDK does not copy it (a shallow copy would be a
  misleading half-guarantee since nested dicts stay shared, and a deep copy is
  unwarranted cost on an error path).

The return value replaces ``message`` and is surfaced to the model VERBATIM as
:attr:`fifty_agent_sdk.tools.protocol.ToolResult.error`; the SDK does not
truncate it, so bounding is the consumer's responsibility.

Sync or async: the hook is called once and its RETURN VALUE is inspected with
:func:`inspect.isawaitable` — an awaitable is awaited, a plain value is not.
``async def``, plain ``def``, callable classes and :func:`functools.partial`
all work (the same "inspect the result, not the function" rule as
:func:`fifty_agent_sdk.observability.hooks.invoke_hook`).

See :class:`MCPClient` for the full failure policy.
"""


class MCPClient:
    """MCP client for a server over Streamable HTTP, wrapping the ``mcp`` SDK.

    See the module docstring for design notes. Construction never performs
    network I/O; the first call to :meth:`discover` or :meth:`invoke` opens a
    session.

    Invariant — auth headers MUST NEVER appear in MCPError.context.
        Every :class:`MCPError` raised by this client builds its
        ``context`` dict from the small allow-list ``{server_url, method,
        tool_name, wrapped, status_code, error_code, error_data,
        operation}`` — no header dict is ever captured. (The per-call
        ``isError`` content lives on :class:`_MCPCallError.content`, NOT in any
        :class:`MCPError.context`, since that path no longer raises — BR-005.)
        ``_resolve_auth``
        still tracks declared header names for any future code path that
        *does* need to surface headers (e.g. a debug-mode dump): that path
        MUST funnel through :func:`_redact_headers` first. The next person
        tempted to widen the context shape: do not put headers in. Use
        :func:`_redact_headers` and prove a test exercises the redaction.

    Args:
        config: Connection configuration.
        auth: Either a static header mapping (applied to every request)
            or an async callable invoked on every request (enabling
            token rotation). Header names appearing in this mapping are
            tracked as sensitive and stripped before any header dict
            reaches a log line or :class:`MCPError` context.
        client: An externally-constructed :class:`httpx.AsyncClient`.
            Provided for tests and dependency-injection scenarios; the
            client OWNS the lifecycle of internally-created clients and
            disposes them on :meth:`aclose`, but does NOT close
            externally-provided ones.
        on_tool_error: Optional :data:`MCPToolErrorHook` — the public seam for
            transforming a per-call ``isError=True`` message before it reaches
            the model (BR-010). Called positionally as
            ``on_tool_error(message, content)`` exactly once per ``isError``
            result, with the SDK's bounded default ``message`` AND the server's
            raw ``content`` blocks; its returned ``str`` replaces the message
            the :class:`fifty_agent_sdk.tools.mcp_provider._MCPToolAdapter`
            surfaces as ``ToolResult.error``. May be sync or async. The whole
            wiring is plain construction — ``MCPClient(config,
            on_tool_error=screen)`` then ``MCPProvider(client)``; no subclassing
            and no private import is needed or supported.

            **Fires ONLY on the recoverable path.** A transport, protocol, or
            session failure raises :class:`MCPError` at the ``call_tool``
            boundary, *before* the result is unwrapped, so the hook is
            structurally unreachable there (BR-005). It cannot make a fatal
            error recoverable or vice-versa.

            **What it CANNOT change:** ``is_error`` and ``output``. A replaced
            message still yields ``ToolResult(output=None, is_error=True,
            error=<replacement>)`` — a failed tool is never reported as a
            success.

            **Failure policy — on ANY failure the ORIGINAL bounded message is
            used.** A raising hook is caught, logged at ``WARNING``
            (``mcp.tool_error_hook_failed``), and the original per-call error is
            returned unchanged; the exception never propagates.
            :class:`asyncio.CancelledError` is the one exception re-raised
            untouched (consumer cancellation must propagate). A return that is
            not a ``str``, or a ``str`` that is empty/whitespace-only, is
            rejected and logged at ``WARNING``
            (``mcp.tool_error_hook_invalid``). It is structurally impossible for
            ``ToolResult.error`` to become ``None``, ``""``, or a non-string via
            this seam. This DIVERGES from
            :func:`fifty_agent_sdk.observability.hooks.invoke_hook`, which
            returns ``None`` and merely swallows: this hook must yield a value,
            so swallowing alone is not a policy.

            **Logging discipline.** The failure logs carry ``tool_name``,
            ``error_type`` / ``returned_type`` and a ``reason`` ONLY — never
            ``str(exc)``, never the ``content``, never the message text.

            **Latency.** The hook is awaited INLINE, inside the
            :class:`fifty_agent_sdk.tools.registry.Registry`'s ``tool_timeout``
            budget. Keep it fast — a slow screen eats the tool budget. It only
            ever fires on the error path, so the happy path is untouched.

            **Zero overhead when unset.** The default ``None`` short-circuits
            before the hook is called, inspected, or any new carrier is
            constructed; the hook-off path is byte-for-byte the pre-BR-010 path.
    """

    def __init__(
        self,
        config: MCPClientConfig,
        *,
        auth: AuthSpec = None,
        client: httpx.AsyncClient | None = None,
        on_tool_error: MCPToolErrorHook | None = None,
    ) -> None:
        self._config = config
        self._auth = auth
        # Pre-compute the set of sensitive header names declared by static
        # auth. Callables are resolved per-request, so their declared keys
        # are added to the sensitive set on the fly.
        self._declared_auth_keys: frozenset[str] = (
            frozenset(k.lower() for k in auth) if isinstance(auth, Mapping) else frozenset()
        )
        self._client = client
        self._owns_client = client is None
        self._on_tool_error = on_tool_error
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover(self) -> list[MCPToolDef]:
        """Fetch the remote tool catalog via the ``tools/list`` MCP method.

        Opens a session (``initialize`` → ``list_tools`` → close) and maps each
        :class:`mcp.types.Tool` into a :class:`MCPToolDef`.

        Returns:
            A list of :class:`MCPToolDef`.

        Raises:
            MCPError: For any transport failure, protocol/handshake failure,
                or server-returned JSON-RPC error. Also raised with
                ``message == "MCP client is closed"`` when invoked after
                :meth:`aclose`.
            asyncio.CancelledError: Propagates untouched — including when it
                arrives as a leaf of a mixed anyio ``BaseExceptionGroup``
                (cancellation racing a real transport error). See
                :meth:`_mcp_error_from_transport`.
        """
        if self._closed:
            raise MCPError(
                "MCP client is closed",
                context={
                    "operation": "discover",
                    "server_url": self._config.base_url,
                },
            ) from None
        base_context: dict[str, Any] = {
            "operation": "discover",
            "server_url": self._config.base_url,
            "method": "tools/list",
        }
        log = _log.bind(method="tools/list", server_url=self._config.base_url)
        # Resolve auth ONCE per call (callable providers invoked exactly once).
        headers, _ = await self._resolve_auth()
        try:
            transport = self._build_transport(headers)
            async with transport.connect() as session:
                result = await session.list_tools()
        except McpError as exc:
            raise self._mcp_error_from_protocol(exc, base_context) from exc
        except RuntimeError as exc:
            raise self._mcp_error_from_runtime(exc, base_context) from exc
        except (httpx.HTTPError, BaseExceptionGroup) as exc:
            raise self._mcp_error_from_transport(exc, base_context) from exc
        log.debug("mcp.call_ok")
        return [self._map_tool(tool) for tool in result.tools]

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a remote tool via the ``tools/call`` MCP method.

        Opens a session (``initialize`` → ``call_tool`` → close) and unwraps
        the typed :class:`mcp.types.CallToolResult`.

        Args:
            name: Name of the remote tool (matches a
                :attr:`MCPToolDef.name`).
            args: Argument object passed verbatim as the ``arguments``.

        Returns:
            On success, the unwrapped tool payload. When the server populates
            ``structuredContent`` (a JSON object — e.g. a typed/object return),
            that dict is returned verbatim. Otherwise the content blocks are
            returned as a list of plain JSON dicts
            (``[c.model_dump(mode="json") for c in result.content]``). No
            single-block flattening is performed: a single ``TextContent`` is
            still returned as a one-element list of ``{"type": "text",
            "text": ...}`` dicts. This is the one observable-output shape that
            differs from the BR-008 hand-rolled client; consumers reading
            ``ToolResult.output`` should expect ``dict`` (structured) or
            ``list[dict]`` (content blocks).

            On a per-call ``isError=True`` result, returns a
            :class:`_MCPCallError` (NOT raised — BR-005): the transport,
            protocol, and session round-trip succeeded, so the failure is a
            recoverable per-tool error the
            :class:`fifty_agent_sdk.tools.mcp_provider._MCPToolAdapter` maps to a
            :class:`fifty_agent_sdk.tools.protocol.ToolResult` with
            ``is_error=True``. When an ``on_tool_error`` hook is configured
            (BR-010) its returned string replaces that carrier's ``message``
            (``content`` is preserved unchanged); on any hook failure the
            original bounded message is used. With no hook configured the
            carrier is returned by identity, byte-for-byte as before.

        Raises:
            MCPError: For transport failure, protocol/handshake failure, or
                server-returned JSON-RPC error (all raised at the
                ``call_tool`` boundary, *before* the result is unwrapped). Also
                raised with ``message == "MCP client is closed"`` after
                :meth:`aclose`. A per-call ``isError=True`` result no longer
                raises — it is returned as a :class:`_MCPCallError` (see
                ``Returns``).
            asyncio.CancelledError: Propagates untouched — including when it
                arrives as a leaf of a mixed anyio ``BaseExceptionGroup``
                (cancellation racing a real transport error). See
                :meth:`_mcp_error_from_transport`.
        """
        if self._closed:
            raise MCPError(
                "MCP client is closed",
                context={
                    "operation": "invoke",
                    "server_url": self._config.base_url,
                },
            ) from None
        base_context: dict[str, Any] = {
            "operation": "invoke",
            "server_url": self._config.base_url,
            "method": "tools/call",
            "tool_name": name,
        }
        log = _log.bind(method="tools/call", server_url=self._config.base_url)
        # Resolve auth ONCE per call: this both fails fast on a misbehaved
        # callable (before any session work) AND is the single invocation of a
        # side-effecting/one-time token provider (no double-spend).
        headers, _ = await self._resolve_auth()
        try:
            transport = self._build_transport(headers)
            async with transport.connect() as session:
                result = await session.call_tool(name, arguments=args)
        except McpError as exc:
            raise self._mcp_error_from_protocol(exc, base_context) from exc
        except RuntimeError as exc:
            raise self._mcp_error_from_runtime(exc, base_context) from exc
        except (httpx.HTTPError, BaseExceptionGroup) as exc:
            raise self._mcp_error_from_transport(exc, base_context) from exc
        log.debug("mcp.call_ok")
        unwrapped = self._unwrap_invoke_result(result, tool_name=name)
        # BR-010: the `on_tool_error` seam fires ONLY inside this branch, which
        # is reachable only after `call_tool` returned without raising — i.e.
        # provably a per-call failure. Applying it here (one frame ABOVE the
        # sync `_unwrap_invoke_result`) is what lets the hook be async, and
        # keeps a subclass override of the unwrap composing with the hook
        # rather than racing it. Every fatal path returned/raised above.
        if isinstance(unwrapped, _MCPCallError):
            return await self._apply_tool_error_hook(unwrapped, tool_name=name)
        return unwrapped

    async def aclose(self) -> None:
        """Close the owned :class:`httpx.AsyncClient`, if any.

        Under the session-per-call lifecycle there is no persistent session to
        unwind — each call already opened and closed its own. Externally-
        provided clients are NOT closed; their lifecycle belongs to the
        caller. Idempotent: a second ``aclose()`` is a no-op and does NOT
        raise. After ``aclose()`` returns, calls to :meth:`discover` and
        :meth:`invoke` raise :class:`MCPError` with
        ``message == "MCP client is closed"``.
        """
        if self._closed:
            return
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._closed = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_transport(self, headers: Mapping[str, str]) -> Transport:
        """Build the per-call transport from already-resolved auth headers.

        ``headers`` is the auth mapping resolved ONCE per call by
        :meth:`discover`/:meth:`invoke` (so a side-effecting callable provider
        is invoked exactly once — no double-spend). The transport sets the
        headers on the httpx client (the supported auth seam in ``mcp
        1.27.0``).
        """
        return StreamableHttpTransport(
            base_url=self._config.base_url,
            read_timeout_seconds=self._config.read_timeout_seconds,
            connect_timeout_seconds=self._config.connect_timeout_seconds,
            user_agent=self._config.user_agent,
            connect_retries=self._config.connect_retries,
            headers=headers,
            http_client=self._client,
        )

    async def _resolve_auth(self) -> tuple[Mapping[str, str], frozenset[str]]:
        """Resolve the auth header mapping and the set of declared keys.

        Returns:
            A tuple of ``(headers, declared_keys_lowercase)``. ``headers``
            is the mapping to merge into the outbound request;
            ``declared_keys_lowercase`` lists the header names treated as
            sensitive (used to extend redaction).

        Raises:
            MCPError: When a callable provider returns a non-Mapping value, or
                when the callable itself raises (e.g. a token endpoint being
                down surfacing as :class:`httpx.ConnectError`). A raising
                callable is translated so the module's "returns or raises
                MCPError" contract holds; without it the raw exception would
                reach :class:`fifty_agent_sdk.tools.registry.Registry`'s
                non-``AgentSdkError`` branch and be DOWNGRADED to a
                model-recoverable ``ToolResult(is_error=True)`` instead of
                staying the fatal infrastructure failure it is.
        """
        auth = self._auth
        if auth is None:
            return {}, frozenset()
        if isinstance(auth, Mapping):
            return auth, frozenset(k.lower() for k in auth)
        # auth is a callable
        try:
            resolved = await auth()
        # ``except Exception`` (not bare/BaseException) keeps
        # ``asyncio.CancelledError`` — a BaseException on 3.11+ — propagating
        # untouched: consumer cancellation must never be re-labelled as an
        # auth failure. This is the same classification-ordering rule the
        # ``on_tool_error`` hook documents.
        except Exception as exc:
            # Type name only, never str(exc): an auth error message may echo
            # endpoint material or credentials, and this module's standing
            # rule is that such material never reaches an MCPError context.
            _log.warning(
                "mcp.auth_callable_failed",
                server_url=self._config.base_url,
                error_type=type(exc).__name__,
            )
            raise MCPError(
                f"auth callable raised: {type(exc).__name__}",
                context={
                    "server_url": self._config.base_url,
                    "wrapped": type(exc).__name__,
                },
            ) from exc
        if not isinstance(resolved, Mapping):
            raise MCPError(
                "auth callable must return a Mapping[str, str]",
                context={
                    "server_url": self._config.base_url,
                    "wrapped": type(resolved).__name__,
                },
            )
        return resolved, frozenset(k.lower() for k in resolved)

    def _map_tool(self, tool: Tool) -> MCPToolDef:
        """Translate one :class:`mcp.types.Tool` into a typed :class:`MCPToolDef`.

        ``mcp.types.Tool`` guarantees a present (camelCase) ``inputSchema`` and
        a nullable ``description`` (coerced to ``""``).
        """
        return MCPToolDef(
            name=tool.name,
            description=tool.description or "",
            input_schema=dict(tool.inputSchema),
        )

    def _mcp_error_from_protocol(self, exc: McpError, base_context: dict[str, Any]) -> MCPError:
        """Translate a :class:`mcp.shared.exceptions.McpError` into :class:`MCPError`.

        Maps the JSON-RPC ``error.code``/``error.data`` into the context
        allow-list. Never carries headers.
        """
        _log.debug(
            "mcp.protocol_error",
            method=base_context.get("method"),
            server_url=self._config.base_url,
            error_code=exc.error.code,
        )
        return MCPError(
            str(exc.error.message) or "MCP server returned an error",
            context={
                **base_context,
                "error_code": exc.error.code,
                "error_data": exc.error.data,
            },
        )

    def _mcp_error_from_runtime(self, exc: RuntimeError, base_context: dict[str, Any]) -> MCPError:
        """Translate a session/handshake :class:`RuntimeError` into :class:`MCPError`.

        :meth:`mcp.ClientSession.initialize` raises a bare ``RuntimeError`` on
        an unsupported protocol version, and the session raises one on invalid
        structured content. Either can surface (bare or nested in the
        transport's anyio task group) from ``discover``/``invoke``; this keeps
        the uniform-MCPError contract (no ``mcp``/``RuntimeError`` leaks).
        Never carries headers — context is the redaction-safe allow-list plus
        ``wrapped`` = ``"RuntimeError"``.
        """
        wrapped = type(exc).__name__
        _log.debug(
            "mcp.session_error",
            method=base_context.get("method"),
            server_url=self._config.base_url,
            wrapped=wrapped,
        )
        return MCPError(
            f"MCP session error: {wrapped}",
            context={**base_context, "wrapped": wrapped},
        )

    def _mcp_error_from_transport(
        self, exc: BaseException, base_context: dict[str, Any]
    ) -> MCPError:
        """Translate a transport-layer failure into :class:`MCPError`.

        Unwraps anyio ``ExceptionGroup``s, then classifies the first relevant
        leaf: an :class:`httpx.HTTPStatusError` carries ``status_code``; any
        other ``httpx`` error carries its class name in ``wrapped``; a
        :class:`mcp.shared.exceptions.McpError` nested inside the group is
        routed through the protocol mapping; a nested ``RuntimeError``
        (e.g. an ``initialize`` protocol-version failure raised inside the
        task group) is routed through the session mapping. Never carries
        headers.

        Cancellation — re-raised, never translated:
            A leaf that is a :class:`BaseException` but NOT an
            :class:`Exception` (:class:`asyncio.CancelledError`,
            :class:`KeyboardInterrupt`) is re-raised untouched. Such a leaf
            is reachable when consumer cancellation races a real transport
            error inside the anyio task group and both arrive wrapped in one
            ``BaseExceptionGroup``; translating it would surface
            ``MCPError("MCP transport error: CancelledError")`` and break the
            SDK's cancellation contract. This is the same
            classification-ordering rule the ``on_tool_error`` hook documents
            (and loop.py's classifier follows): cancellation always wins over
            error translation.
        """
        leaves = _iter_leaf_exceptions(exc)
        for leaf in leaves:
            if not isinstance(leaf, Exception):
                raise leaf
        # Prefer a nested protocol error (a 4xx/handshake failure can surface
        # as an McpError inside the transport task group).
        for leaf in leaves:
            if isinstance(leaf, McpError):
                return self._mcp_error_from_protocol(leaf, base_context)
        for leaf in leaves:
            if isinstance(leaf, httpx.HTTPStatusError):
                status = leaf.response.status_code
                _log.debug(
                    "mcp.http_error",
                    method=base_context.get("method"),
                    server_url=self._config.base_url,
                    status_code=status,
                )
                return MCPError(
                    f"MCP server returned HTTP {status}",
                    context={
                        **base_context,
                        "status_code": status,
                        "wrapped": "HTTPStatusError",
                    },
                )
        for leaf in leaves:
            if isinstance(leaf, httpx.HTTPError):
                wrapped = type(leaf).__name__
                _log.debug(
                    "mcp.transport_error",
                    method=base_context.get("method"),
                    server_url=self._config.base_url,
                    wrapped=wrapped,
                )
                return MCPError(
                    f"MCP transport error: {wrapped}",
                    context={**base_context, "wrapped": wrapped},
                )
        # A nested RuntimeError (e.g. an initialize protocol-version failure
        # raised inside the transport task group) routes through the session
        # mapping for a consistent message + context.
        for leaf in leaves:
            if isinstance(leaf, RuntimeError):
                return self._mcp_error_from_runtime(leaf, base_context)
        # No recognised leaf — surface the group's first leaf class name.
        wrapped = type(leaves[0]).__name__ if leaves else type(exc).__name__
        _log.debug(
            "mcp.transport_error",
            method=base_context.get("method"),
            server_url=self._config.base_url,
            wrapped=wrapped,
        )
        return MCPError(
            f"MCP transport error: {wrapped}",
            context={**base_context, "wrapped": wrapped},
        )

    def _unwrap_invoke_result(self, result: CallToolResult, *, tool_name: str) -> Any:
        """Unwrap the typed ``tools/call`` :class:`mcp.types.CallToolResult`.

        Reached ONLY after :meth:`mcp.ClientSession.call_tool` returned a
        ``CallToolResult`` without raising — i.e. the transport, protocol, and
        session are provably healthy. Therefore a ``result.isError`` here is
        always a recoverable *per-call* failure, never a connection failure:
        the latter is raised as :class:`MCPError` at the ``call_tool`` boundary
        (``invoke``) and never enters this function. This call-site structure —
        not field-sniffing — is the discriminator between recoverable and fatal
        MCP errors (BR-005).

        Recoverable error shape (BR-005):
            When ``result.isError`` is True, this RETURNS (does not raise) a
            :class:`_MCPCallError` carrying a bounded model-safe ``message`` and
            the server's error ``content`` blocks. The
            :class:`fifty_agent_sdk.tools.mcp_provider._MCPToolAdapter` translates
            it into a recoverable
            :class:`fifty_agent_sdk.tools.protocol.ToolResult` with
            ``is_error=True``.

        Security note — error-content capture (TD-007 item 3):
            When ``isError`` is True, ``result.content`` is serialised verbatim
            onto :attr:`_MCPCallError.content`. The SDK has no way to know what
            an MCP server places in that field — a non-conformant or
            poorly-implemented server MAY echo request arguments, credentials,
            or other sensitive material into the error content, in which case
            that material surfaces in our error logs. We intentionally preserve
            the value as-is rather than redact it: the SDK does not own the
            schema, redaction would mask real debug info, and the threat model
            here is a misbehaving downstream (which we cannot fix from the
            client side). The adapter keeps this content OFF the model-facing
            observation (only the bounded ``message`` is surfaced to the model);
            consumers running against untrusted MCP servers SHOULD scrub
            :attr:`_MCPCallError.content` before logging.

            The sanctioned scrubbing point is the ``on_tool_error`` hook on
            :class:`MCPClient` (BR-010): it receives BOTH this bounded
            ``message`` and this raw ``content``, and its returned string is what
            the model sees. It is applied by :meth:`_apply_tool_error_hook` one
            frame up, in :meth:`invoke`, *after* this method returns — NOT here,
            so that a subclass override of this (private, unsupported) method
            composes with the hook rather than racing it. Overriding this method
            or importing :class:`_MCPCallError` is NOT a supported extension
            point; ``on_tool_error`` is.

        Success shape:
            Returns ``result.structuredContent`` when the server provides it
            (a JSON object), else the content blocks serialised to plain JSON
            dicts: ``[c.model_dump(mode="json") for c in result.content]``.
        """
        if result.isError:
            return _MCPCallError(
                message=f"MCP tool '{tool_name}' returned isError=True",
                content=[c.model_dump(mode="json") for c in result.content],
            )
        if result.structuredContent is not None:
            return result.structuredContent
        return [c.model_dump(mode="json") for c in result.content]

    async def _apply_tool_error_hook(self, err: _MCPCallError, *, tool_name: str) -> _MCPCallError:
        """Apply the ``on_tool_error`` seam to a per-call ``isError`` outcome (BR-010).

        Called by :meth:`invoke` ONLY inside its ``isinstance(..., _MCPCallError)``
        branch, so it is structurally unreachable on the fatal
        transport/protocol/session path (which raised :class:`MCPError` at the
        ``call_tool`` boundary). The recoverable/fatal split of BR-005 is
        untouched: this method can only ever rewrite one string.

        Contract:

        * ``on_tool_error is None`` (the default) — return ``err`` BY IDENTITY
          before the hook is called, before any :func:`inspect.isawaitable`
          check, and before any new carrier is constructed. The hook-off path
          is byte-for-byte the pre-BR-010 path.
        * Otherwise call ``hook(err.message, err.content)`` exactly once and
          await the result if it is awaitable (inspecting the RESULT, not the
          function, is correct for every callable shape — the same rule as
          :func:`fifty_agent_sdk.observability.hooks.invoke_hook`).
        * On ANY failure the ORIGINAL bounded ``err`` is returned unchanged, so
          ``ToolResult.error`` can never become ``None``, ``""``, or a
          non-string via this seam: a raising hook is caught and logged
          ``mcp.tool_error_hook_failed``; a non-``str`` or blank return is
          rejected and logged ``mcp.tool_error_hook_invalid``.
        * :class:`asyncio.CancelledError` is re-raised untouched.

        Args:
            err: The per-call error carrier produced by
                :meth:`_unwrap_invoke_result`.
            tool_name: Remote tool name, used only for the WARNING log lines.

        Returns:
            ``err`` itself when no hook is configured or the hook failed;
            otherwise a new :class:`_MCPCallError` carrying the hook's string as
            ``message`` and the SAME ``content`` list.

        Raises:
            asyncio.CancelledError: Re-raised untouched if the hook raises it.
        """
        hook = self._on_tool_error
        # Zero-overhead guard: nothing is called, inspected, or allocated when
        # no hook is configured.
        if hook is None:
            return err
        try:
            returned = hook(err.message, err.content)
            result: Any = await returned if inspect.isawaitable(returned) else returned
        # Ordering is load-bearing (the codebase's classification-ordering rule,
        # cf. loop.py): the CancelledError arm MUST precede the Exception arm so
        # consumer cancellation propagates instead of being swallowed into the
        # fallback. CancelledError is a BaseException on 3.11+, so this is
        # defensive documentation of intent — keep it, reviewers look for it.
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Same discipline as `invoke_hook` (which now also logs the
            # exception TYPE only): a hook screening untrusted server content
            # may embed that content in its own exception message
            # (`ValueError(f"bad: {content}")`), and this module's standing
            # rule is that untrusted material never reaches a log line. Log
            # the exception TYPE only — never its text, never `content`,
            # never the message.
            _log.warning(
                "mcp.tool_error_hook_failed",
                tool_name=tool_name,
                error_type=type(exc).__name__,
            )
            return err
        if not isinstance(result, str):
            _log.warning(
                "mcp.tool_error_hook_invalid",
                tool_name=tool_name,
                returned_type=type(result).__name__,
                reason="not_a_string",
            )
            return err
        if not result.strip():
            _log.warning(
                "mcp.tool_error_hook_invalid",
                tool_name=tool_name,
                returned_type=type(result).__name__,
                reason="blank_string",
            )
            return err
        # `content` is carried over unchanged: the hook may only rewrite the
        # model-facing message, never the retained diagnostic evidence.
        return _MCPCallError(message=result, content=err.content)


__all__ = [
    "MCPClient",
    "MCPClientConfig",
    "MCPToolDef",
    "MCPToolErrorHook",
]
