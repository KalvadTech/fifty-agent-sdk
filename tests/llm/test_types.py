"""Tests for fifty_agent_sdk.llm.types — Pydantic v2 contract surface."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from fifty_agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
    ToolChoiceFunction,
    Usage,
)

# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_chat_message_accepts_every_role(role: str) -> None:
    msg = ChatMessage(role=role, content="hi")  # type: ignore[arg-type]
    assert msg.role == role
    assert msg.content == "hi"
    assert msg.name is None
    assert msg.tool_call_id is None


def test_chat_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="banana", content="hi")  # type: ignore[arg-type]


def test_chat_message_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="hi", unexpected="field")  # type: ignore[call-arg]


def test_chat_message_allows_empty_content() -> None:
    msg = ChatMessage(role="assistant", content="")
    assert msg.content == ""


def test_chat_message_optional_name_and_tool_call_id() -> None:
    msg = ChatMessage(role="tool", content="result", name="search", tool_call_id="call_1")
    assert msg.name == "search"
    assert msg.tool_call_id == "call_1"


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_tool_call_defaults_args_to_empty_dict() -> None:
    call = ToolCall(name="search")
    assert call.args == {}


def test_tool_call_args_can_carry_nested_data() -> None:
    call = ToolCall(name="search", args={"query": "hi", "limit": 5, "deep": {"a": 1}})
    assert call.args == {"query": "hi", "limit": 5, "deep": {"a": 1}}


def test_tool_call_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolCall(name="x", whatever=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_accepts_non_negative_counts() -> None:
    u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert u.prompt_tokens == 10
    assert u.completion_tokens == 5
    assert u.total_tokens == 15


def test_usage_accepts_zero() -> None:
    u = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert u.total_tokens == 0


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_usage_rejects_negative(field: str) -> None:
    payload = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, field: -1}
    with pytest.raises(ValidationError):
        Usage(**payload)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------


def test_chat_request_defaults() -> None:
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], model="gpt-4o")
    assert req.temperature == 0.0
    assert req.max_tokens is None
    assert req.response_format is None


@pytest.mark.parametrize("bad_temp", [-0.01, -1.0, 2.01, 5.0])
def test_chat_request_rejects_out_of_range_temperature(bad_temp: float) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            temperature=bad_temp,
        )


def test_chat_request_accepts_boundary_temperatures() -> None:
    for t in (0.0, 2.0):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            temperature=t,
        )
        assert req.temperature == t


def test_chat_request_accepts_none_temperature() -> None:
    """``None`` is the explicit opt-out: the adapter omits the key from the body."""
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        temperature=None,
    )
    assert req.temperature is None


def test_chat_request_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            max_tokens=0,
        )


def test_chat_request_accepts_one_as_min_max_tokens() -> None:
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        max_tokens=1,
    )
    assert req.max_tokens == 1


def test_chat_request_response_format_passthrough() -> None:
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        response_format={"type": "json_object"},
    )
    assert req.response_format == {"type": "json_object"}


def test_chat_request_tools_and_tool_choice_default_none() -> None:
    """The BR-008 additive fields default to None — flag-OFF wire is unchanged."""
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], model="gpt-4o")
    assert req.tools is None
    assert req.tool_choice is None


def test_chat_request_accepts_tools_and_tool_choice() -> None:
    tools = [{"type": "function", "function": {"name": "search"}}]
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        tools=tools,
        tool_choice="auto",
    )
    assert req.tools == tools
    assert req.tool_choice == "auto"


def test_chat_request_accepts_tool_choice_dict_form() -> None:
    """BR-014 named-tool typing imports and validates across the Python matrix."""
    choice: ToolChoiceFunction = {"type": "function", "function": {"name": "search"}}
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_choice=choice,
    )
    assert req.tool_choice == {"type": "function", "function": {"name": "search"}}


@pytest.mark.parametrize(
    "bad_choice",
    [
        {"type": "function"},  # missing required "function" member
        {"type": "not-a-function", "function": {"name": "search"}},
        {"function": {"name": "search"}},  # missing required "type"
        {"type": "function", "function": {}},  # missing required "name"
        42,
    ],
)
def test_chat_request_rejects_malformed_tool_choice(bad_choice: Any) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            tool_choice=bad_choice,
        )


def test_chat_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            unknown=True,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason",
    ["stop", "length", "tool_calls", "content_filter", "error", "in_progress"],
)
def test_chat_response_accepts_every_finish_reason(finish_reason: str) -> None:
    resp = ChatResponse(
        message=ChatMessage(role="assistant", content="done"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason=finish_reason,  # type: ignore[arg-type]
    )
    assert resp.finish_reason == finish_reason


def test_chat_response_rejects_unknown_finish_reason() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(
            message=ChatMessage(role="assistant", content="done"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="completed",  # type: ignore[arg-type]
        )


def test_chat_response_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(
            message=ChatMessage(role="assistant", content="done"),
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="stop",
            extra_field=1,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Round-trip behavior used by the adapter
# ---------------------------------------------------------------------------


def test_chat_message_model_dump_exclude_none_strips_optional_fields() -> None:
    msg = ChatMessage(role="user", content="hi")
    dumped = msg.model_dump(exclude_none=True)
    assert dumped == {"role": "user", "content": "hi"}


def test_chat_message_model_dump_exclude_none_keeps_set_optionals() -> None:
    msg = ChatMessage(role="tool", content="ok", name="search", tool_call_id="x")
    dumped = msg.model_dump(exclude_none=True)
    assert dumped == {
        "role": "tool",
        "content": "ok",
        "name": "search",
        "tool_call_id": "x",
    }


def test_round_trip_chat_request_via_model_validate() -> None:
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        temperature=0.2,
        max_tokens=64,
        response_format={"type": "text"},
        tools=[{"type": "function", "function": {"name": "search"}}],
        tool_choice="auto",
    )
    same = ChatRequest.model_validate(req.model_dump())
    assert same == req
