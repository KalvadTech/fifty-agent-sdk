"""Tests for fifty_agent_sdk.llm.openai_compat.OpenAICompatibleClient.

Uses ``pytest-httpx`` to intercept the HTTP calls the openai Python SDK
makes under the hood. No real network is required.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.openai_compat import OpenAICompatibleClient
from fifty_agent_sdk.llm.protocol import LLMClient
from fifty_agent_sdk.llm.types import ChatMessage, ChatRequest, ToolCall, ToolChoiceFunction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://example.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def _canonical_response(
    *,
    content: str = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    total_tokens: int = 8,
    model: str = "gpt-4o",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(
    *,
    delta_role: str | None = None,
    delta_content: str | None = None,
    finish_reason: str | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if delta_role is not None:
        delta["role"] = delta_role
    if delta_content is not None:
        delta["content"] = delta_content
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _make_client(
    *,
    base_url: str = BASE_URL,
    model: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url=base_url,
        model=model,
        timeout=5.0,
        max_retries=0,
        http_client=http_client,
    )


def _basic_request(*, model: str = "gpt-4o", **overrides: Any) -> ChatRequest:
    fields: dict[str, Any] = {
        "messages": [ChatMessage(role="user", content="hi")],
        "model": model,
    }
    fields.update(overrides)
    return ChatRequest(**fields)


# ---------------------------------------------------------------------------
# Happy path: complete()
# ---------------------------------------------------------------------------


async def test_complete_happy_path_maps_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    resp = await client.complete(_basic_request())

    assert resp.message.role == "assistant"
    assert resp.message.content == "hello"
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 8


async def test_complete_targets_configured_base_url(httpx_mock: HTTPXMock) -> None:
    base = "https://gdc.example.com/v1"
    httpx_mock.add_response(
        method="POST",
        url=f"{base}/chat/completions",
        json=_canonical_response(),
    )
    client = _make_client(base_url=base)
    resp = await client.complete(_basic_request())
    assert resp.message.content == "hello"

    # Confirm the SDK hit the custom base_url, not the default OpenAI URL.
    request = httpx_mock.get_request()
    assert request is not None
    assert str(request.url) == f"{base}/chat/completions"


async def test_complete_sends_provider_agnostic_payload(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(
        _basic_request(
            temperature=0.7,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
    )
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["model"] == "gpt-4o"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 128
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_per_request_model_overrides_client_default(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client(model="gpt-3.5-turbo")  # client default
    await client.complete(_basic_request(model="gpt-4o-mini"))  # request override
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["model"] == "gpt-4o-mini"


async def test_client_default_model_used_when_request_omits_it() -> None:
    # The Pydantic model requires `model`, so to exercise the fallback we
    # build a ChatRequest with the default then patch it to be falsy.
    # Easiest: construct via model_construct to bypass validation.
    req = ChatRequest.model_construct(
        messages=[ChatMessage(role="user", content="hi")],
        model="",  # falsy → triggers default lookup
        temperature=0.0,
        max_tokens=None,
        response_format=None,
    )
    client = _make_client(model="default-model")
    # We don't need pytest-httpx here because we'll mock at a higher level
    # by raising. But to keep the test minimal we just verify it raises
    # if NO default and NO request model. Tested separately below.
    # For this test, the request has falsy model and client has a default,
    # so the request body must use the default — verify against mock.
    with pytest.MonkeyPatch.context() as mp:
        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            # Build a minimal SDK-shaped response stand-in.
            from openai.types.chat import ChatCompletion

            return ChatCompletion.model_validate(_canonical_response())

        mp.setattr(client._client.chat.completions, "create", fake_create)
        await client.complete(req)
    assert captured["model"] == "default-model"


async def test_complete_raises_when_no_model_anywhere() -> None:
    client = _make_client(model=None)
    req = ChatRequest.model_construct(
        messages=[ChatMessage(role="user", content="hi")],
        model="",
        temperature=0.0,
        max_tokens=None,
        response_format=None,
    )
    with pytest.raises(LLMError) as exc:
        await client.complete(req)
    assert "No model specified" in exc.value.message


async def test_complete_omits_optional_fields_when_unset(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert "max_tokens" not in body
    assert "response_format" not in body


async def test_build_body_sends_default_temperature(httpx_mock: HTTPXMock) -> None:
    """The ``0.0`` default IS sent — the key is present unless explicitly ``None``."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["temperature"] == 0.0


async def test_build_body_omits_temperature_when_none(httpx_mock: HTTPXMock) -> None:
    """``temperature=None`` removes the key entirely (reasoning-model providers)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(temperature=None))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert "temperature" not in body


@pytest.mark.parametrize(
    "upstream,expected",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
        ("function_call", "tool_calls"),  # legacy → tool_calls
    ],
)
async def test_complete_normalizes_finish_reason(
    httpx_mock: HTTPXMock, upstream: str, expected: str
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        json=_canonical_response(finish_reason=upstream),
    )
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.finish_reason == expected


async def test_complete_handles_null_content(httpx_mock: HTTPXMock) -> None:
    """Some providers send ``content: null`` when only tool_calls are emitted."""
    payload = _canonical_response(finish_reason="tool_calls")
    payload["choices"][0]["message"]["content"] = None
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.message.content == ""
    assert resp.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Native tool_calls mapping (BR-007)
# ---------------------------------------------------------------------------


async def test_complete_maps_native_tool_calls(httpx_mock: HTTPXMock) -> None:
    """OpenAI tool_calls (JSON-string arguments) map to SDK ToolCall list."""
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "x"}'},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())

    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls is not None
    assert len(resp.message.tool_calls) == 1
    mapped = resp.message.tool_calls[0]
    assert isinstance(mapped, ToolCall)
    assert mapped.name == "search"
    # The JSON-string `arguments` is parsed into a dict.
    assert mapped.args == {"q": "x"}


async def test_complete_maps_native_tool_calls_empty_arguments(httpx_mock: HTTPXMock) -> None:
    """An empty-arguments tool call maps to an empty args dict."""
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "ping", "arguments": ""},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())

    assert resp.message.tool_calls is not None
    assert resp.message.tool_calls[0].name == "ping"
    assert resp.message.tool_calls[0].args == {}


async def test_complete_malformed_arguments_raises_llm_error(httpx_mock: HTTPXMock) -> None:
    """Non-JSON `arguments` raises LLMError(MalformedResponse) with the call id."""
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "not json"},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_1"
    assert exc.value.context["model"] == "gpt-4o"
    assert exc.value.context["arguments_excerpt"] == "not json"
    assert exc.value.__cause__ is not None


async def test_complete_oversized_integer_arguments_raise_llm_error(
    httpx_mock: HTTPXMock,
) -> None:
    """BR-013 contains bare ValueError from native tool-call argument decoding."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    arguments = f'{{"n":{digits}}}'
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_big",
                "type": "function",
                "function": {"name": "calculate", "arguments": arguments},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert str(exc.value) == "provider tool_call arguments is not valid JSON"
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_big"
    assert len(str(exc.value.context["arguments_excerpt"])) <= 200
    assert type(exc.value.__cause__) is ValueError


async def test_complete_non_object_arguments_raises_llm_error(httpx_mock: HTTPXMock) -> None:
    """A non-object JSON `arguments` (e.g. a bare array) is rejected."""
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_3",
                "type": "function",
                "function": {"name": "search", "arguments": "[1, 2, 3]"},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_3"


async def test_complete_no_tool_calls_field_is_none(httpx_mock: HTTPXMock) -> None:
    """The default response (no tool_calls) leaves `tool_calls` as None (additive)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.message.tool_calls is None
    # Existing assertions on the additive path are unchanged.
    assert resp.message.content == "hello"
    assert resp.finish_reason == "stop"


async def test_complete_native_tool_call_excluded_from_request_body(
    httpx_mock: HTTPXMock,
) -> None:
    """A None tool_calls on request messages emits no `tool_calls` key on the wire."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    for msg in body["messages"]:
        assert "tool_calls" not in msg


# ---------------------------------------------------------------------------
# BR-008: tools/tool_choice declaration + assistant tool_calls wire envelope
# ---------------------------------------------------------------------------


async def test_build_body_no_tools_when_request_tools_unset(httpx_mock: HTTPXMock) -> None:
    """Flag-OFF proof at the adapter level: no tools/tool_choice on the wire."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert "tools" not in body
    assert "tool_choice" not in body


async def test_build_body_declares_tools_when_set(httpx_mock: HTTPXMock) -> None:
    """A set `request.tools` is emitted verbatim with `tool_choice="auto"` default."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(tools=tools))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


async def test_build_body_tool_choice_override(httpx_mock: HTTPXMock) -> None:
    """A caller-supplied `tool_choice` passes through instead of defaulting to auto."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(
        _basic_request(
            tools=[{"type": "function", "function": {"name": "x"}}], tool_choice="required"
        )
    )
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["tool_choice"] == "required"


async def test_build_body_tool_choice_dict_form_on_wire(httpx_mock: HTTPXMock) -> None:
    """The specific-tool `tool_choice` object is emitted verbatim on the wire."""
    choice: ToolChoiceFunction = {"type": "function", "function": {"name": "search"}}
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(
        _basic_request(
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice=choice,
        )
    )
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["tool_choice"] == {"type": "function", "function": {"name": "search"}}


async def test_assistant_tool_calls_envelope_on_wire(httpx_mock: HTTPXMock) -> None:
    """An assistant turn carrying tool_calls serializes to the OpenAI envelope shape.

    The `tool_calls[].id` is the message's `tool_call_id`, `type` is
    "function", and `function.arguments` is a JSON STRING (not an object) that
    round-trips back to the args dict.
    """
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id="pairing-id-123",
        tool_calls=[ToolCall(name="search", args={"q": "x"})],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(messages=[assistant_msg]))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assistant_wire = body["messages"][0]
    assert assistant_wire["role"] == "assistant"
    assert assistant_wire["content"] == ""
    assert len(assistant_wire["tool_calls"]) == 1
    tc = assistant_wire["tool_calls"][0]
    assert tc["id"] == "pairing-id-123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    # arguments MUST be a JSON string, not an object.
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"q": "x"}


async def test_id_pairing_assistant_id_matches_tool_reply(httpx_mock: HTTPXMock) -> None:
    """The assistant tool_calls id and the tool reply tool_call_id pair on the wire."""
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id="X",
        tool_calls=[ToolCall(name="search", args={"q": "x"})],
    )
    tool_reply = ChatMessage(role="tool", content="result", name="search", tool_call_id="X")
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(messages=[assistant_msg, tool_reply]))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assistant_wire = body["messages"][0]
    tool_wire = body["messages"][1]
    assistant_tc_id = assistant_wire["tool_calls"][0]["id"]
    assert assistant_tc_id == "X"
    assert tool_wire["role"] == "tool"
    assert tool_wire["tool_call_id"] == "X"
    # The pairing key matches across both messages.
    assert assistant_tc_id == tool_wire["tool_call_id"]


async def test_two_turn_native_no_400(httpx_mock: HTTPXMock) -> None:
    """End-to-end: a two-turn native round-trip produces a well-formed wire envelope.

    Turn 1 returns a native tool_call; turn 2 (after the assistant turn + tool
    reply are appended) returns a final answer. The request the mock RECEIVES
    on turn 2 MUST carry the OpenAI assistant envelope (tool_calls[].id
    present) AND a paired role="tool" reply whose tool_call_id matches — the
    shape a strict OpenAI endpoint requires to NOT return 400. This asserts
    the request envelope SHAPE, not merely that no exception was raised.
    """
    tool_call_id = "call_abc"
    turn1 = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "x"}'},
            }
        ],
    )
    turn2 = _canonical_response(content="the answer", finish_reason="stop")
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=turn1)
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=turn2)
    client = _make_client()

    # Turn 1: plain request; provider returns a native tool_call.
    resp1 = await client.complete(_basic_request(messages=[ChatMessage(role="user", content="q")]))
    assert resp1.finish_reason == "tool_calls"
    assert resp1.message.tool_calls is not None
    # Loop would now synthesize the assistant turn (carrying the pairing id)
    # and the tool reply from the response.
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id=tool_call_id,
        tool_calls=resp1.message.tool_calls,
    )
    tool_reply = ChatMessage(
        role="tool", content="search results", name="search", tool_call_id=tool_call_id
    )
    # Turn 2: replay with the assistant + tool turns in history.
    resp2 = await client.complete(
        _basic_request(messages=[ChatMessage(role="user", content="q"), assistant_msg, tool_reply])
    )
    assert resp2.finish_reason == "stop"

    # Assert the SECOND request's envelope shape (the turn that would 400 on a
    # strict endpoint if the pairing were broken).
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    body = json.loads(requests[1].read())
    assistant_wire = next(
        m for m in body["messages"] if m["role"] == "assistant" and "tool_calls" in m
    )
    tool_wire = next(m for m in body["messages"] if m["role"] == "tool")
    assert assistant_wire["tool_calls"][0]["id"] == tool_call_id
    assert assistant_wire["tool_calls"][0]["type"] == "function"
    # arguments is a JSON STRING on the wire.
    assert isinstance(assistant_wire["tool_calls"][0]["function"]["arguments"], str)
    assert json.loads(assistant_wire["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert tool_wire["tool_call_id"] == tool_call_id
    # THE pairing invariant: both sides carry the same id.
    assert assistant_wire["tool_calls"][0]["id"] == tool_wire["tool_call_id"]


# ---------------------------------------------------------------------------
# Error paths: complete()
# ---------------------------------------------------------------------------


async def test_connection_error_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("nope"))
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "APIConnectionError"
    assert exc.value.context["model"] == "gpt-4o"
    assert exc.value.__cause__ is not None


async def test_timeout_error_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.TimeoutException("slow"))
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "APITimeoutError"


async def test_rate_limit_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        status_code=429,
        json={"error": {"message": "rate"}},
    )
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "RateLimitError"


async def test_server_error_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        status_code=500,
        json={"error": {"message": "boom"}},
    )
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    # InternalServerError is an APIError subclass in the openai SDK.
    assert exc.value.context["type"] == "InternalServerError"


async def test_empty_choices_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    payload = _canonical_response()
    payload["choices"] = []
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert "no choices" in exc.value.message.lower()
    assert exc.value.context["type"] == "MalformedResponse"


async def test_missing_usage_field_is_filled_with_zeros(httpx_mock: HTTPXMock) -> None:
    payload = _canonical_response()
    payload["usage"] = None
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.completion_tokens == 0
    assert resp.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_happy_path(httpx_mock: HTTPXMock) -> None:
    body = b"".join(
        [
            _sse(_chunk(delta_role="assistant", delta_content="")),
            _sse(_chunk(delta_content="Hel")),
            _sse(_chunk(delta_content="lo")),
            _sse(_chunk(delta_content=" world")),
            _sse(_chunk(finish_reason="stop")),
            b"data: [DONE]\n\n",
        ]
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    chunks = []
    async for chunk in client.stream(_basic_request()):
        chunks.append(chunk)

    # Each chunk holds the delta only.
    contents = [c.message.content for c in chunks]
    accum = "".join(contents)
    assert accum == "Hello world"
    # Intermediate chunks emit "in_progress"; only the terminal chunk
    # carries the real terminal reason from upstream.
    for intermediate in chunks[:-1]:
        assert intermediate.finish_reason == "in_progress"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].message.content == ""


async def test_stream_intermediate_chunks_use_in_progress(httpx_mock: HTTPXMock) -> None:
    """A naive consumer that breaks on `finish_reason == "stop"` must not exit early."""
    body = b"".join(
        [
            _sse(_chunk(delta_content="a")),
            _sse(_chunk(delta_content="b")),
            _sse(_chunk(delta_content="c")),
            _sse(_chunk(finish_reason="stop")),
            b"data: [DONE]\n\n",
        ]
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    seen = []
    async for chunk in client.stream(_basic_request()):
        seen.append(chunk.finish_reason)
    # Every chunk before the last reports "in_progress"; only the last
    # carries the real "stop".
    assert seen[:-1] == ["in_progress"] * (len(seen) - 1)
    assert seen[-1] == "stop"


@pytest.mark.parametrize(
    "upstream_terminal,expected_terminal",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
    ],
)
async def test_stream_terminal_chunk_preserves_upstream_reason(
    httpx_mock: HTTPXMock,
    upstream_terminal: str,
    expected_terminal: str,
) -> None:
    body = b"".join(
        [
            _sse(_chunk(delta_content="hi")),
            _sse(_chunk(finish_reason=upstream_terminal)),
            b"data: [DONE]\n\n",
        ]
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    chunks = []
    async for chunk in client.stream(_basic_request()):
        chunks.append(chunk)
    assert chunks[-1].finish_reason == expected_terminal


async def test_stream_request_body_marks_stream_true(httpx_mock: HTTPXMock) -> None:
    body = _sse(_chunk(finish_reason="stop")) + b"data: [DONE]\n\n"
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    async for _ in client.stream(_basic_request()):
        pass
    raw = httpx_mock.get_request()
    assert raw is not None
    payload = json.loads(raw.read())
    assert payload["stream"] is True


async def test_stream_connection_error_during_open(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("nope"))
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        async for _ in client.stream(_basic_request()):
            pass
    assert exc.value.context["type"] == "APIConnectionError"


async def test_stream_malformed_chunk_raises_llm_error(httpx_mock: HTTPXMock) -> None:
    """A chunk with a non-dict ``delta`` triggers the defensive mapper."""
    bad = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": "not-an-object", "finish_reason": None}],
    }
    body = _sse(bad) + b"data: [DONE]\n\n"
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    # The openai SDK validates chunks against its own model and may raise
    # before our mapper sees them. Either path must surface as LLMError.
    with pytest.raises(LLMError):
        async for _ in client.stream(_basic_request()):
            pass


# ---------------------------------------------------------------------------
# Lifecycle: aclose() and the async context manager
# ---------------------------------------------------------------------------


async def test_aclose_closes_owned_client() -> None:
    """An owned client (no injected http_client) is closed by aclose()."""
    client = _make_client()
    await client.aclose()
    assert client._client.is_closed()


async def test_aclose_is_idempotent() -> None:
    """A second ``aclose()`` is a no-op and MUST NOT raise."""
    client = _make_client()
    await client.aclose()
    await client.aclose()
    assert client._client.is_closed()


async def test_aclose_does_not_close_injected_http_client() -> None:
    """An injected ``http_client`` stays open after aclose() — the caller owns it."""
    injected = httpx.AsyncClient()
    client = _make_client(http_client=injected)
    await client.aclose()
    assert not injected.is_closed
    await injected.aclose()


async def test_async_context_manager_closes_owned_client_on_exit() -> None:
    """``async with`` returns the client itself and aclose()s it on exit."""
    async with _make_client() as client:
        assert isinstance(client, OpenAICompatibleClient)
        assert not client._client.is_closed()
    assert client._client.is_closed()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_client_satisfies_llmclient_protocol_at_runtime() -> None:
    client = OpenAICompatibleClient(api_key="x", base_url=BASE_URL, model="gpt-4o", max_retries=0)
    assert isinstance(client, LLMClient)
