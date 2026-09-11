"""Tests for ``fifty_agent_sdk.parser.json_mode.JsonModeParser``."""

from __future__ import annotations

import json

import pytest

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser import (
    FinalAnswer,
    JsonModeParser,
    Parser,
    ThoughtAction,
)
from fifty_agent_sdk.parser.json_mode import _RawEnvelope
from fifty_agent_sdk.prompts import JSON_MODE_OUTPUT_FORMAT


def _parser() -> JsonModeParser:
    return JsonModeParser()


# ---------------------------------------------------------------------- #
# Happy paths                                                            #
# ---------------------------------------------------------------------- #


def test_happy_path_tool_call() -> None:
    completion = json.dumps(
        {
            "thought": "I should search.",
            "action": "tool",
            "tool_name": "search",
            "tool_args": {"q": "x"},
            "answer": None,
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.thought == "I should search."
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_happy_path_final_answer() -> None:
    completion = json.dumps(
        {
            "thought": "Now I know.",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": "hello world",
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.thought == "Now I know."
    assert result.content == "hello world"


def test_tool_args_defaults_to_empty_when_null() -> None:
    completion = json.dumps(
        {
            "thought": "t",
            "action": "tool",
            "tool_name": "noop",
            "tool_args": None,
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {}


def test_tool_args_missing_key_defaults_to_empty() -> None:
    """Pydantic default + None-coalesce means a missing key is fine too."""
    completion = json.dumps(
        {
            "thought": "t",
            "action": "tool",
            "tool_name": "noop",
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {}


# ---------------------------------------------------------------------- #
# Recovery paths                                                         #
# ---------------------------------------------------------------------- #


def test_code_fence_wrapped_json_is_recovered() -> None:
    inner = json.dumps({"thought": "t", "action": "final", "answer": "ok"})
    completion = f"```json\n{inner}\n```"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "ok"


def test_code_fence_without_lang_tag_is_recovered() -> None:
    inner = json.dumps({"thought": "t", "action": "final", "answer": "ok"})
    completion = f"```\n{inner}\n```"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)


def test_extra_prose_around_json_is_recovered() -> None:
    inner = '{"thought":"t","action":"final","answer":"ok"}'
    completion = f"Sure! Here you go: {inner} -- hope that helps"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "ok"


def test_double_fence_takes_first_block() -> None:
    first = json.dumps({"thought": "a", "action": "final", "answer": "first"})
    second = json.dumps({"thought": "b", "action": "final", "answer": "second"})
    completion = f"```json\n{first}\n```\n\n```json\n{second}\n```"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "first"


# ---------------------------------------------------------------------- #
# Failure paths                                                          #
# ---------------------------------------------------------------------- #


def test_malformed_json_raises_parser_error() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("not json at all")
    ctx = excinfo.value.context
    assert ctx["parser"] == "JsonModeParser"
    assert ctx["error_phase"] == "json_decode"
    assert "completion_excerpt" in ctx


def test_recovery_attempt_still_invalid_raises_json_decode() -> None:
    # Has braces so the recovery slice triggers, but contents are not JSON.
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("{ this is not json but has braces }")
    assert excinfo.value.context["error_phase"] == "json_decode"


def test_action_tool_missing_tool_name_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "tool"})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "schema_validation"
    assert ctx["missing"] == "tool_name"


def test_action_tool_empty_tool_name_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "tool", "tool_name": "", "tool_args": {}})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["missing"] == "tool_name"


def test_action_final_missing_answer_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "final"})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "schema_validation"
    assert ctx["missing"] == "answer"


def test_unknown_action_value_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "banana", "answer": "x"})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "schema_validation"


def test_extra_top_level_field_raises_schema_error() -> None:
    completion = json.dumps(
        {
            "thought": "t",
            "action": "final",
            "answer": "a",
            "junk": 1,
        }
    )
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "schema_validation"


def test_empty_completion_raises_with_empty_completion_phase() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_whitespace_only_completion_raises_with_empty_completion_phase() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("   \n\t  ")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_parser_error_context_excerpt_truncated() -> None:
    big = "garbage " * 100  # > 200 chars, no valid JSON
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(big)
    excerpt = excinfo.value.context["completion_excerpt"]
    assert isinstance(excerpt, str)
    assert len(excerpt) <= 200


def test_parser_error_chains_cause_via_raise_from() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("not json")
    assert excinfo.value.__cause__ is not None


def test_deeply_nested_json_raises_parser_error_not_recursion_error() -> None:
    """Pathological nesting trips ``RecursionError`` inside ``json.loads``.

    The Parser contract (parser/base.py) mandates ``ParserError`` on malformed
    input and the loop catches only ``ParserError`` — a raw ``RecursionError``
    would escape the async generator with no ``ErrorEvent``/``FinalEvent``.
    Depth 100k exceeds the recursion threshold of both the C and the pure
    Python ``json`` scanners regardless of ``sys.getrecursionlimit``.
    """
    depth = 100_000
    completion = '{"a":' * depth + "1" + "}" * depth
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "JsonModeParser"
    assert ctx["error_phase"] == "json_decode"
    assert "RecursionError" in str(ctx["cause"])


def test_deeply_nested_json_behind_prose_prefix_raises_parser_error() -> None:
    """Recovery-pass arm: the strict pass fails with JSONDecodeError, then the
    fence-slice recovery candidate is the deeply nested payload and the second
    ``json.loads`` raises ``RecursionError`` — still a ``ParserError``."""
    depth = 100_000
    nested = '{"a":' * depth + "1" + "}" * depth
    completion = f"Sure! Here you go: {nested} -- hope that helps"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "json_decode"
    assert "RecursionError" in str(ctx["cause"])


# ---------------------------------------------------------------------- #
# Protocol / cross-brief contract                                        #
# ---------------------------------------------------------------------- #


def test_parser_protocol_satisfied() -> None:
    assert isinstance(_parser(), Parser)


def test_json_mode_parser_consumes_keys_taught_by_prompt() -> None:
    """Mirror of the prompts-side pin test.

    Every JSON envelope key advertised by JSON_MODE_OUTPUT_FORMAT must be a
    field on the parser's internal validator. Drift on either side breaks
    the cross-brief contract.
    """
    fields = set(_RawEnvelope.model_fields.keys())
    for key in ("thought", "action", "tool_name", "tool_args", "answer"):
        assert key in fields, f"parser missing field for prompt key: {key}"
        assert key in JSON_MODE_OUTPUT_FORMAT
