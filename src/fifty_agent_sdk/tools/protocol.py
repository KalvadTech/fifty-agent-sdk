"""Tool layer contracts: ``ToolSchema``, ``ToolResult``, ``ToolCall`` (re-exported), ``Tool`` Protocol.

This module defines the boundary the ReACT loop (BR-006) will consume. The
loop never sees a concrete tool — only the :class:`Tool` Protocol, the
:class:`ToolSchema` description of its inputs, and the :class:`ToolResult` of
an invocation.

``ToolCall`` is intentionally NOT redefined here; it is re-exported from
:mod:`fifty_agent_sdk.llm.types` so the parser (BR-005), the loop (BR-006), and the
registry share a single source of truth for the (name, args) tool-invocation
envelope.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# Re-export the canonical ToolCall (single source of truth: fifty_agent_sdk.llm.types).
# The parser (BR-005) materializes a ToolCall from LLM output; the loop (BR-006)
# feeds it straight into Registry.invoke. Duplicating the type would force a
# no-op translation step at the loop boundary.
from fifty_agent_sdk.llm.types import ToolCall


class ToolSchema(BaseModel):
    """JSON-schema-shaped description of a tool's input arguments.

    Mirrors the standard JSON Schema vocabulary so it can be fed directly into
    LLM function-calling envelopes when needed. The :class:`Registry` treats
    it as an opaque blob; only the LLM-facing prompt builder reads its
    contents.

    Attributes:
        type: Always ``"object"`` at the top level. JSON Schema requires a
            top-level object for function-call envelopes.
        properties: Mapping of parameter name to its JSON-Schema description.
            Pydantic's ``model_json_schema()`` populates this when the schema
            is derived from a function signature (see
            :func:`fifty_agent_sdk.tools.inproc_provider.tool`). Local
            ``#/$defs/...`` references are INLINED by the providers before the
            schema is emitted (see
            :mod:`fifty_agent_sdk.tools._schema_refs`) — the dict must be
            self-contained because the loop ships it into provider
            function-calling envelopes that reject unknown top-level keys, so
            a ``$defs`` block cannot ride along.
        required: Names of parameters without defaults. Pydantic's schema
            emitter populates this automatically.
        additionalProperties: Always ``False`` — the schema is the complete
            description of the tool's inputs; unknown keys are a contract
            violation, not a forward-compatibility feature.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolResult(BaseModel):
    """The outcome of a tool invocation.

    A tool's :meth:`Tool.invoke` returns a ``ToolResult`` in BOTH success and
    failure paths — it never raises for "tool failed". This lets the ReACT
    loop treat tool failures as data the LLM can reason about (try different
    args, pick a different tool) rather than as control flow.

    System-level failures (timeouts, unregistered names, SDK errors) are NOT
    surfaced through this type; the registry raises them as exceptions.

    Attributes:
        output: The tool's return value when ``is_error`` is False. ``None``
            when ``is_error`` is True.
        is_error: True if the tool's invocation failed in a recoverable way
            (raised an exception, returned invalid args, etc.). The LLM is
            expected to inspect ``error`` and decide how to recover.
        error: A human-readable description of the failure. Typically
            ``f"{type(e).__name__}: {e}"``. ``None`` on success.
    """

    model_config = ConfigDict(extra="forbid")

    output: Any = None
    is_error: bool = False
    error: str | None = None


@runtime_checkable
class Tool(Protocol):
    """Pluggable tool contract.

    Implementations expose:

    - ``name``: matches the string the LLM emits in :class:`ToolCall.name`.
      The :class:`Registry` uses this as its dict key.
    - ``description``: human-readable explanation rendered into the system
      prompt by the BR-003 prompt builder.
    - ``schema``: a :class:`ToolSchema` describing the input arguments.
    - ``invoke(args)``: an ``async`` method that takes a dict of arguments and
      returns a :class:`ToolResult`.

    Cancellation contract:
        Implementations SHOULD honor :class:`asyncio.CancelledError` so the
        registry's timeout enforcement can clean up resources. Long-running
        tools SHOULD ``await`` periodically so cancellation can propagate.
        Pure-CPU tools wrapping :func:`asyncio.to_thread` MAY continue
        executing after :class:`fifty_agent_sdk.errors.ToolTimeout` raises; this is
        a known limitation of Python threads, not a registry bug.

    Error contract:
        - For "ran but failed" outcomes (bad args, business-rule violation,
          remote service error the tool catches), return
          ``ToolResult(is_error=True, error=...)``.
        - For genuine system failures the loop should see as exceptions
          (unrecoverable infrastructure errors), raise an
          :class:`fifty_agent_sdk.errors.AgentSdkError` subclass. The registry
          propagates these.
    """

    name: str
    description: str
    schema: ToolSchema

    async def invoke(self, args: dict[str, Any]) -> ToolResult: ...


__all__ = ["Tool", "ToolCall", "ToolResult", "ToolSchema"]
