"""OpenAI-compatible :class:`LLMClient` adapter.

Backed by the official ``openai`` Python SDK pointed at any OpenAI-compatible
``/v1/chat/completions`` endpoint — OpenAI itself, Google Distributed Cloud
(GDC), local OSS servers (vLLM, Ollama via the openai-compat layer, etc.).
The provider differences are absorbed by ``base_url``.

All provider SDK exceptions are wrapped into :class:`fifty_agent_sdk.errors.LLMError`
at the public method boundary, in line with the
:class:`fifty_agent_sdk.llm.protocol.LLMClient` contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ToolCall,
    Usage,
)

# OpenAI returns ``"function_call"`` as a legacy finish reason that maps to
# our protocol's ``"tool_calls"``. Anything outside this map is treated as an
# error condition (see :func:`_normalize_finish_reason`).
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "tool_calls",
    "error": "error",
}

_MAX_TOOL_CALL_ARG_EXCERPT: int = 200
"""Maximum length of ``arguments_excerpt`` in the malformed-arguments context."""


class OpenAICompatibleClient:
    """:class:`LLMClient` implementation backed by the ``openai`` Python SDK.

    Works against any OpenAI-compatible ``/v1/chat/completions`` endpoint.
    Provider variation is absorbed by ``base_url`` — the same client class
    drives OpenAI itself, GDC, and local OSS servers.

    Async context management
        The client is an async context manager: ``async with
        OpenAICompatibleClient(...) as client:`` closes the underlying
        connection pool on exit via :meth:`aclose`. Without the context
        manager, call :meth:`aclose` explicitly — an unclosed client leaks
        its ``httpx.AsyncClient`` pool until GC.

    Args:
        api_key: API key passed to the upstream provider. Required even for
            local servers that ignore it; pass any non-empty string.
        base_url: Override the default OpenAI base URL. Use for GDC or a
            self-hosted endpoint. ``None`` means use the SDK default.
        model: Default model identifier used when a :class:`ChatRequest` does
            not set one. ``None`` means callers MUST set ``request.model``.
        timeout: Per-request timeout in seconds. Defaults to ``60.0``.
        max_retries: SDK-level retry count for transient failures (429, 5xx,
            connection errors). Defaults to ``2``. Set to ``0`` to make
            errors surface immediately, which is what tests want.
        http_client: Optional pre-configured ``httpx.AsyncClient``. Useful for
            tests that need to inject a mock transport. When omitted, the
            SDK builds its own client, OWNS it, and closes it on
            :meth:`aclose`. An INJECTED client is NOT closed by
            :meth:`aclose` — its lifecycle belongs to the caller. This
            owned-vs-injected discipline mirrors
            :class:`fifty_agent_sdk.mcp.client.MCPClient`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = AsyncOpenAI(**kwargs)
        self._default_model = model
        # Ownership discipline mirrors MCPClient: an injected http_client is
        # the caller's to close; only the client built here is closed by
        # aclose(). (`AsyncOpenAI.close()` closes the underlying httpx client
        # unconditionally, so it must only be called on the owned path.)
        self._owns_client = http_client is None
        self._closed = False

    async def aclose(self) -> None:
        """Close the underlying ``openai`` client and its connection pool.

        Only an OWNED client is closed: when ``http_client`` was injected at
        construction, its lifecycle belongs to the caller and ``aclose()``
        leaves it open. Idempotent: a second ``aclose()`` is a no-op and does
        NOT raise. The client MUST NOT be used after ``aclose()`` returns.
        """
        if self._closed:
            return
        if self._owns_client:
            await self._client.close()
        self._closed = True

    async def __aenter__(self) -> OpenAICompatibleClient:
        """Enter the async context manager, returning ``self``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context manager, closing via :meth:`aclose`."""
        await self.aclose()

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run a single non-streaming completion.

        Args:
            request: The chat completion request.

        Returns:
            The mapped :class:`ChatResponse`.

        Raises:
            fifty_agent_sdk.errors.LLMError: Wraps any failure of the underlying
                provider call (network, timeout, rate-limit, malformed
                envelope, missing fields).
        """
        model = request.model or self._default_model
        if not model:
            raise LLMError(
                "No model specified: pass `model` to ChatRequest or set a default on the client.",
                context={"request_model": request.model, "client_default": self._default_model},
            )
        body = self._build_body(request, model=model, stream=False)
        try:
            raw = await self._client.chat.completions.create(**body)
        except APITimeoutError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APITimeoutError"},
            ) from e
        except APIConnectionError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APIConnectionError"},
            ) from e
        except RateLimitError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "RateLimitError"},
            ) from e
        except APIError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": type(e).__name__},
            ) from e
        return self._map_response(raw, model=model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Stream a completion as incremental chunks.

        Each yielded :class:`ChatResponse` carries the delta in
        ``message.content`` (not the running accumulation). Intermediate
        chunks emit ``finish_reason="in_progress"``; the terminal chunk
        carries a real terminal reason (``"stop"`` / ``"length"`` /
        ``"tool_calls"`` / ``"content_filter"`` / ``"error"``) mapped from
        the upstream provider's value. Consumers can therefore branch on
        ``finish_reason`` without misreading an intermediate delta as
        terminal.

        Args:
            request: The chat completion request.

        Yields:
            :class:`ChatResponse` chunks containing the latest delta.

        Raises:
            fifty_agent_sdk.errors.LLMError: Wraps any failure of the underlying
                provider call. Errors mid-stream surface from the iterator.
        """
        model = request.model or self._default_model
        if not model:
            raise LLMError(
                "No model specified: pass `model` to ChatRequest or set a default on the client.",
                context={"request_model": request.model, "client_default": self._default_model},
            )
        body = self._build_body(request, model=model, stream=True)
        try:
            stream = await self._client.chat.completions.create(**body)
        except APITimeoutError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APITimeoutError"},
            ) from e
        except APIConnectionError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APIConnectionError"},
            ) from e
        except RateLimitError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "RateLimitError"},
            ) from e
        except APIError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": type(e).__name__},
            ) from e

        try:
            async for raw_chunk in stream:
                yield self._map_chunk(raw_chunk, model=model)
        except APITimeoutError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APITimeoutError", "phase": "stream"},
            ) from e
        except APIConnectionError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APIConnectionError", "phase": "stream"},
            ) from e
        except APIError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": type(e).__name__, "phase": "stream"},
            ) from e
        except LLMError:
            # Allow defensive mapping errors raised from `_map_chunk` to surface unchanged.
            raise
        except Exception as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": type(e).__name__, "phase": "stream"},
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_message(msg: ChatMessage) -> dict[str, Any]:
        """Serialize a single :class:`ChatMessage` for the request wire.

        For every message EXCEPT an assistant turn carrying native
        ``tool_calls``, this is byte-for-byte ``msg.model_dump(exclude_none=True)``
        (the pre-BR-008 behavior). For an assistant turn carrying
        ``tool_calls`` (only set when native function-calling is on), the
        message is translated into the OpenAI envelope:

            {role:"assistant", content, tool_calls: [
                {id, type:"function",
                 function:{name, arguments: json.dumps(args)}}
            ]}

        The ``id`` is sourced PER ENTRY from the entry's own
        :attr:`~fifty_agent_sdk.llm.types.ToolCall.id` (BR-006): the loop
        mints a distinct id per dispatched call so a multi-call assistant
        turn carries N DISTINCT ids, and each subsequent ``role="tool"``
        reply pairs with exactly one entry (a strict OpenAI endpoint returns
        400 otherwise). The fallback
        ``tc.id if tc.id is not None else msg.tool_call_id`` preserves the
        BR-008 single-call wire byte-for-byte: the single-call native path
        sets ``ToolCall(id=None)`` and ``msg.tool_call_id=call_id``, so
        ``tc.id is None`` falls back to ``msg.tool_call_id`` and emits the
        same id BR-008 did. ``arguments`` is emitted as a JSON STRING (not an
        object), per the OpenAI function-calling spec.

        Flag-OFF proof: ``ChatMessage.tool_calls`` defaults to ``None`` and is
        only populated on a native turn (which requires
        ``native_tools_enabled=True`` to even reach the provider), so every
        flag-OFF message takes the ``else`` branch — identical to today.
        """
        if msg.role == "assistant" and msg.tool_calls:
            tool_calls_envelope = [
                {
                    "id": tc.id if tc.id is not None else msg.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    },
                }
                for tc in msg.tool_calls
            ]
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_envelope,
            }
        return msg.model_dump(exclude_none=True)

    @classmethod
    def _build_body(cls, request: ChatRequest, *, model: str, stream: bool) -> dict[str, Any]:
        """Build the kwargs passed to ``client.chat.completions.create``.

        ``temperature`` is omitted entirely when :attr:`ChatRequest.temperature`
        is ``None`` (the opt-out for providers that reject a non-default
        temperature); at the ``0.0`` default it is sent, preserving the
        pre-existing wire shape for every caller that does not opt out.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [cls._serialize_message(m) for m in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            body["response_format"] = request.response_format
        if request.tools is not None:
            body["tools"] = request.tools
            body["tool_choice"] = request.tool_choice or "auto"
        return body

    @staticmethod
    def _normalize_finish_reason(raw: str | None) -> FinishReason:
        """Map an upstream ``finish_reason`` to our :data:`FinishReason` union.

        Unknown or absent values map to ``"error"`` so consumers can detect
        provider drift without crashing.
        """
        if raw is None:
            return "error"
        return _FINISH_REASON_MAP.get(raw, "error")

    @classmethod
    def _map_response(cls, raw: Any, *, model: str) -> ChatResponse:  # noqa: ANN401 - SDK shape is opaque
        """Map a non-streaming SDK response into our :class:`ChatResponse`.

        Defensive: if any required field is missing or has the wrong shape,
        raise :class:`LLMError` rather than letting a ``KeyError`` /
        ``AttributeError`` leak.

        When the upstream provider emits structured ``tool_calls`` (OpenAI
        function-calling), each entry is normalized into the SDK's
        :class:`~fifty_agent_sdk.llm.types.ToolCall` (``{name, args: dict}``)
        by parsing the provider's JSON-string ``arguments``. Parse failure of
        ``arguments`` raises :class:`LLMError` (``type="MalformedResponse"``)
        with the offending ``tool_call_id`` and a bounded
        ``arguments_excerpt`` — a provider sending malformed ``arguments`` is
        an unrecoverable envelope error, not model drift.
        """
        try:
            choices = raw.choices
            if not choices:
                raise LLMError(
                    "Provider returned no choices.",
                    context={"model": model, "type": "MalformedResponse"},
                )
            choice = choices[0]
            sdk_message = choice.message
            content = sdk_message.content if sdk_message.content is not None else ""
            finish_reason = cls._normalize_finish_reason(choice.finish_reason)
            usage = cls._map_usage(raw.usage)
            tool_calls = cls._map_tool_calls(getattr(sdk_message, "tool_calls", None), model=model)
        except LLMError:
            raise
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError(
                f"Malformed provider response: {e}",
                context={"model": model, "type": "MalformedResponse"},
            ) from e
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
            usage=usage,
            finish_reason=finish_reason,
        )

    @classmethod
    def _map_tool_calls(cls, raw: Any, *, model: str) -> list[ToolCall] | None:  # noqa: ANN401 - SDK shape is opaque
        """Normalize upstream OpenAI ``tool_calls`` into SDK :class:`ToolCall`.

        Each upstream entry has the OpenAI shape
        ``{id, type, function: {name, arguments(JSON string)}}``. The SDK
        :class:`ToolCall` is ``{name, args: dict}`` (no ``id`` — see BR-007
        D7: the loop synthesizes the pairing ``call_id`` for history replay).
        The provider's JSON-string ``arguments`` is parsed into ``args``;
        parse failure raises :class:`LLMError` (``type="MalformedResponse"``).

        Returns ``None`` (not ``[]``) when there are no upstream tool calls so
        ``model_dump(exclude_none=True)`` in :meth:`_build_body` stays clean
        on the request wire for every existing caller.

        Args:
            raw: The upstream ``message.tool_calls`` value, or ``None``.
            model: Model identifier for the error context payload.

        Returns:
            A list of SDK :class:`ToolCall` values, or ``None`` when
            ``raw`` is absent/empty.

        Raises:
            fifty_agent_sdk.errors.LLMError: When an entry's ``arguments``
                is not valid JSON.
        """
        if not raw:
            return None
        mapped: list[ToolCall] = []
        for tc in raw:
            function = tc.function
            arguments = function.arguments if function.arguments is not None else ""
            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                raise LLMError(
                    "provider tool_call arguments is not valid JSON",
                    context={
                        "model": model,
                        "type": "MalformedResponse",
                        "tool_call_id": getattr(tc, "id", None),
                        "arguments_excerpt": arguments[:_MAX_TOOL_CALL_ARG_EXCERPT],
                    },
                ) from e
            if not isinstance(args, dict):
                # The OpenAI spec requires `arguments` to be a JSON object;
                # a non-object (e.g. a bare string or number) is malformed.
                raise LLMError(
                    "provider tool_call arguments is not a JSON object",
                    context={
                        "model": model,
                        "type": "MalformedResponse",
                        "tool_call_id": getattr(tc, "id", None),
                        "arguments_excerpt": arguments[:_MAX_TOOL_CALL_ARG_EXCERPT],
                    },
                )
            mapped.append(ToolCall(name=function.name, args=args))
        return mapped

    @classmethod
    def _map_chunk(cls, raw: Any, *, model: str) -> ChatResponse:  # noqa: ANN401 - SDK shape is opaque
        """Map a streaming SDK chunk into a delta-carrying :class:`ChatResponse`.

        ``message.content`` is the chunk's delta. ``finish_reason`` is
        ``"in_progress"`` on intermediate chunks (and on header-only chunks
        with no ``choices``) and only resolves to a terminal value
        (``"stop"``/``"length"``/``"tool_calls"``/``"content_filter"``/``"error"``)
        on the chunk whose upstream ``finish_reason`` is non-``None``. This
        lets consumers branch on ``finish_reason`` without misreading an
        intermediate delta as terminal.
        """
        try:
            choices = raw.choices
            if not choices:
                # Some providers emit a header chunk with only usage and no choices.
                usage = cls._map_usage(getattr(raw, "usage", None))
                return ChatResponse(
                    message=ChatMessage(role="assistant", content=""),
                    usage=usage,
                    finish_reason="in_progress",
                )
            choice = choices[0]
            delta = choice.delta
            content = ""
            # NOTE: native tool_calls are intentionally NOT mapped here. A
            # native tool-decision turn is non-streamed (BR-007 D5), and the
            # loop's native precedence branch only fires when a real
            # ChatResponse carries tool_calls — streamed turns never go native.
            if delta is not None and delta.content is not None:
                content = delta.content
            upstream_finish = choice.finish_reason
            finish_reason: FinishReason
            if upstream_finish is None:
                finish_reason = "in_progress"
            else:
                finish_reason = cls._normalize_finish_reason(upstream_finish)
            usage = cls._map_usage(getattr(raw, "usage", None))
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError(
                f"Malformed provider stream chunk: {e}",
                context={"model": model, "type": "MalformedChunk"},
            ) from e
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            usage=usage,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _map_usage(raw: Any) -> Usage:  # noqa: ANN401 - SDK shape is opaque
        """Map an SDK usage object into our :class:`Usage`.

        Missing or absent usage data resolves to zeros — providers vary in
        whether they emit usage on streaming chunks.
        """
        if raw is None:
            return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
        completion_tokens = getattr(raw, "completion_tokens", 0) or 0
        total_tokens = getattr(raw, "total_tokens", 0) or 0
        return Usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )


__all__ = ["OpenAICompatibleClient"]
