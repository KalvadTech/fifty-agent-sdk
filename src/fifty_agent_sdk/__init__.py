"""fifty-agent-sdk — Production-grade reusable agent loop SDK.

Foundation layer: typed errors, Pydantic v2 message types, the
:class:`LLMClient` protocol, an OpenAI-compatible adapter, and a system
prompt template builder. Higher layers (parser, tool registry, ReACT loop,
state stores) build on these contracts in subsequent briefs.

Optional extra surface
    :class:`SqlStateStore` and :data:`sql_metadata`, plus
    :class:`SqlAuditSink` and :data:`audit_metadata` (all the ``sql``
    extra, ``pip install 'fifty-agent-sdk[sql]'``), and :class:`RedisStateStore`
    (the ``redis`` extra, ``pip install 'fifty-agent-sdk[redis]'``) are
    re-exported lazily. Importing :mod:`fifty_agent_sdk` itself pulls neither
    SQLAlchemy nor redis-py. First access to one of these symbols triggers
    its import and raises a clear :class:`ImportError` if the relevant
    extra is missing.
"""

from __future__ import annotations

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
)
from importlib.metadata import (
    version as _pkg_version,
)
from typing import TYPE_CHECKING, Any

from fifty_agent_sdk.audit import AuditEvent, AuditSink, ConsoleAuditSink
from fifty_agent_sdk.errors import (
    AgentSdkError,
    LLMError,
    MaxIterationsExceeded,
    MCPError,
    ParserError,
    StateStoreError,
    ToolNotFound,
    ToolTimeout,
)
from fifty_agent_sdk.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    LLMClient,
    OpenAICompatibleClient,
    Role,
    ToolCall,
    Usage,
)
from fifty_agent_sdk.loop import AgentLoop
from fifty_agent_sdk.mcp import MCPClient, MCPClientConfig, MCPToolDef, MCPToolErrorHook
from fifty_agent_sdk.observability import Hooks
from fifty_agent_sdk.parser import (
    FinalAnswer,
    JsonModeParser,
    MultiAction,
    NativeToolsParser,
    NativeToolsParserProtocol,
    Parser,
    ParseResult,
    ProseModeParser,
    ThoughtAction,
)
from fifty_agent_sdk.prompts import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    PromptSections,
    json_mode_template,
    prose_mode_template,
    render_system_prompt,
)
from fifty_agent_sdk.runner import AgentRunner
from fifty_agent_sdk.safety import SafetyConfig
from fifty_agent_sdk.state import (
    TRUNK_BRANCH_ID,
    BranchInfo,
    MemoryStateStore,
    StateStore,
)
from fifty_agent_sdk.streaming import (
    ActionEvent,
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    ObservationEvent,
    ThoughtEvent,
    TokenEvent,
    ToolFailedEvent,
    ToolProgressEvent,
    ToolStartedEvent,
)
from fifty_agent_sdk.tools import (
    InProcProvider,
    MCPProvider,
    RefreshSummary,
    Registry,
    Tool,
    ToolResult,
    ToolSchema,
    tool,
)

if TYPE_CHECKING:
    from fifty_agent_sdk.audit.sql import SqlAuditSink, audit_metadata
    from fifty_agent_sdk.state.redis import RedisStateStore
    from fifty_agent_sdk.state.sql import SqlStateStore, sql_metadata

try:
    # Single source of truth: read the version from the installed distribution
    # metadata (generated from pyproject.toml at build/install time) rather than
    # hardcoding it here. Prevents the __version__/pyproject drift fixed in
    # BR-001 from recurring. (TD-001)
    __version__ = _pkg_version("fifty-agent-sdk")
except _PackageNotFoundError:  # source/editable checkout with no installed dist-info
    __version__ = "0.0.0+unknown"

__all__ = [
    "JSON_MODE_OUTPUT_FORMAT",
    "PROSE_MODE_OUTPUT_FORMAT",
    "ActionEvent",
    "AgentEvent",
    "AgentLoop",
    "AgentRunner",
    "AgentSdkError",
    "AuditEvent",
    "AuditSink",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConsoleAuditSink",
    "ErrorEvent",
    "FinalAnswer",
    "FinalEvent",
    "FinishReason",
    "Hooks",
    "InProcProvider",
    "JsonModeParser",
    "LLMClient",
    "LLMError",
    "MCPClient",
    "MCPClientConfig",
    "MCPError",
    "MCPProvider",
    "MCPToolDef",
    "MCPToolErrorHook",
    "MaxIterationsExceeded",
    "MemoryStateStore",
    "MultiAction",
    "NativeToolsParser",
    "NativeToolsParserProtocol",
    "ObservationEvent",
    "OpenAICompatibleClient",
    "ParseResult",
    "Parser",
    "ParserError",
    "PromptSections",
    "ProseModeParser",
    "RedisStateStore",
    "RefreshSummary",
    "Registry",
    "Role",
    "SafetyConfig",
    "SqlAuditSink",
    "SqlStateStore",
    "StateStore",
    "StateStoreError",
    "ThoughtAction",
    "ThoughtEvent",
    "TokenEvent",
    "Tool",
    "ToolCall",
    "ToolFailedEvent",
    "ToolNotFound",
    "ToolProgressEvent",
    "ToolResult",
    "ToolSchema",
    "ToolStartedEvent",
    "ToolTimeout",
    "Usage",
    "audit_metadata",
    "json_mode_template",
    "prose_mode_template",
    "render_system_prompt",
    "sql_metadata",
    "tool",
]


def __getattr__(name: str) -> Any:
    """Lazily import optional-extra surface symbols on first access.

    ``SqlStateStore`` / ``sql_metadata`` and ``SqlAuditSink`` /
    ``audit_metadata`` require the optional ``sql`` extra;
    ``RedisStateStore`` requires the ``redis`` extra. Routing access
    through this hook keeps ``import fifty_agent_sdk`` free of SQLAlchemy and
    redis-py, and surfaces a clear :class:`ImportError` only when the
    symbols are actually used without the extra installed.
    """
    if name in {"SqlStateStore", "sql_metadata"}:
        from fifty_agent_sdk.state import sql

        return getattr(sql, name)
    if name in {"SqlAuditSink", "audit_metadata"}:
        from fifty_agent_sdk.audit import sql as audit_sql

        return getattr(audit_sql, name)
    if name == "RedisStateStore":
        from fifty_agent_sdk.state import redis

        return redis.RedisStateStore
    raise AttributeError(f"module 'fifty_agent_sdk' has no attribute {name!r}")
