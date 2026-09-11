"""Pydantic v2 data models for the LLM contract.

These models are the *provider-agnostic surface* of the SDK. Every
:class:`fifty_agent_sdk.llm.protocol.LLMClient` implementation accepts and returns
exclusively these types — provider-specific envelopes are translated at the
adapter boundary.

All models set ``extra="forbid"`` so unknown fields raise a validation error
instead of silently passing through. This is intentional: the SDK should
catch typos and provider-drift early rather than letting unknown attributes
ride along to consumers.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
"""Discriminator for the speaker of a :class:`ChatMessage`."""

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "in_progress"]
"""Standard terminal reason for a chat completion — plus the streaming sentinel.

Mirrors the OpenAI chat-completion ``finish_reason`` field. Adapters MUST map
provider-specific values into one of these literals.

The ``"in_progress"`` value is a streaming-only sentinel emitted on
intermediate chunks where the upstream provider has not yet reported a
terminal reason. Consumers can branch on it without misreading an
intermediate delta as a terminal ``"stop"``.
"""


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Who is speaking. One of :data:`Role`.
        content: The textual content of the message. An empty string is
            permitted (for example, an assistant turn that contains only
            tool calls).
        name: Optional name for a function/tool message or a named speaker.
        tool_call_id: Identifier echoed back on a ``role="tool"`` reply so
            the model can match it to the originating tool call.
        tool_calls: Native (provider-structured) tool invocations carried on
            an ``assistant`` turn. Populated by the LLM adapter when the
            upstream provider returns structured ``tool_calls`` (OpenAI
            function-calling); ``None`` on every text-only turn. When the
            loop dispatches a native call, the assistant turn is replayed to
            the provider WITH ``tool_calls`` set so the subsequent
            ``role="tool"`` reply (keyed by ``tool_call_id``) pairs correctly.
            An assistant turn carrying only tool calls may have
            ``content=""``.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolCall(BaseModel):
    """A model-issued tool invocation request.

    The provider-agnostic shape: a tool name plus a dict of arguments. Adapters
    are responsible for translating provider-specific tool-call envelopes into
    this shape and back.

    Attributes:
        name: Name of the tool to invoke. Must match a registered tool.
        args: Arguments to pass to the tool. Defaults to an empty dict.
        id: Optional provider-pairing key (BR-006). The ReACT loop mints a
            per-call id for EVERY call it dispatches and embeds it on the
            assistant turn's ``tool_calls[].id`` so the subsequent
            ``role="tool"`` reply (keyed by ``ChatMessage.tool_call_id``)
            pairs with the correct entry. ``None`` (the default) on a freshly
            adapter-mapped provider response — the single-call native path
            relies on the message-level ``tool_call_id`` instead (see
            :meth:`fifty_agent_sdk.llm.openai_compat.OpenAICompatibleClient.
            _serialize_message`'s fallback). The field is excluded from
            ``model_dump(exclude_none=True)`` output when ``None``, so the
            text/JSON request wire is byte-for-byte unchanged for every
            pre-BR-006 caller.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class Usage(BaseModel):
    """Token accounting for a single LLM call.

    All counts are non-negative integers. When a provider does not return
    usage data (for example, partway through a stream), adapters return zeros
    for the missing fields rather than ``None``.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens emitted in the completion.
        total_tokens: Sum of prompt and completion tokens.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ToolChoiceFunctionName(TypedDict):
    """The ``function`` member of :class:`ToolChoiceFunction`."""

    name: str


class ToolChoiceFunction(TypedDict):
    """The specific-tool form of :attr:`ChatRequest.tool_choice`.

    Mirrors the OpenAI ``ChatCompletionNamedToolChoiceParam`` shape —
    ``{"type": "function", "function": {"name": ...}}`` — and forces the
    model to call the named function instead of choosing freely.
    """

    type: Literal["function"]
    function: ToolChoiceFunctionName


class ChatRequest(BaseModel):
    """Provider-agnostic chat-completion request.

    Attributes:
        messages: Ordered list of conversation messages.
        model: Model identifier. Adapters may override this with a default.
        temperature: Sampling temperature in ``[0.0, 2.0]``. Default ``0.0``.
        max_tokens: Optional cap on completion tokens. Must be ``>= 1`` if set.
        response_format: Optional provider-format hint. Common values are
            ``{"type": "json_object"}`` or ``{"type": "text"}``. Adapters
            pass this through verbatim where supported.
        tools: Optional OpenAI-style tool-declaration envelope for native
            (provider-structured) function-calling. Each entry has the shape
            ``{"type": "function", "function": {"name", "description",
            "parameters": {...JSON Schema...}}}``. When set, the adapter
            declares the tools to the provider via the ``tools`` request param
            so the model may return native ``tool_calls``. ``None`` (the
            default) means NO tools are declared and the request wire is
            byte-for-byte the pre-BR-008 shape.
        tool_choice: Optional steering for native tool-calling when
            :attr:`tools` is set. Accepts a mode string — ``"auto"`` (the
            default the adapter emits when this is ``None``),
            ``"required"``, or ``"none"`` — or a :class:`ToolChoiceFunction`
            object ``{"type": "function", "function": {"name": ...}}`` that
            forces the model to call the named function. Ignored when
            :attr:`tools` is ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | ToolChoiceFunction | None = None


class ChatResponse(BaseModel):
    """Provider-agnostic chat-completion response.

    For non-streaming responses, ``message.content`` holds the full completion.
    For streaming responses, each yielded :class:`ChatResponse` chunk carries
    only the delta in ``message.content`` (consumers accumulate). Intermediate
    chunks have ``finish_reason='in_progress'``; only the final chunk carries a
    real terminal reason (``stop`` / ``length`` / ``tool_calls`` /
    ``content_filter`` / ``error``). Usage figures may be zero on intermediate
    chunks if the provider omits them.

    Attributes:
        message: The assistant's message for this response (or chunk delta).
        usage: Token accounting. Zero-filled when a provider omits counts.
        finish_reason: Why generation stopped. One of :data:`FinishReason`.
    """

    model_config = ConfigDict(extra="forbid")

    message: ChatMessage
    usage: Usage
    finish_reason: FinishReason


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "FinishReason",
    "Role",
    "ToolCall",
    "ToolChoiceFunction",
    "ToolChoiceFunctionName",
    "Usage",
]
