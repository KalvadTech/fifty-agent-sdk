"""Tests for ``fifty_agent_sdk.parser.json_mode.JsonModeParser``."""

from __future__ import annotations

import json
import sys

import pytest

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser import (
    FinalAnswer,
    JsonModeParser,
    Parser,
    ThoughtAction,
)
from fifty_agent_sdk.parser import json_mode as json_mode_module
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


def test_action_tool_whitespace_only_tool_name_raises() -> None:
    """A whitespace-only name is blank-after-strip and takes the same
    schema_validation path as an empty one (the prose parser strips the
    ``Action:`` header, so the JSON parser must match)."""
    completion = json.dumps(
        {"thought": "t", "action": "tool", "tool_name": "   \t  ", "tool_args": {}}
    )
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "schema_validation"
    assert ctx["missing"] == "tool_name"


def test_action_tool_padded_tool_name_is_stripped() -> None:
    """Surrounding whitespace is stripped so the registry sees the same clean
    name the prose parser would emit."""
    completion = json.dumps(
        {"thought": "t", "action": "tool", "tool_name": "  search  ", "tool_args": {}}
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"


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


def test_strict_recursion_error_is_translated_to_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-017 deterministically pins strict-pass ``RecursionError`` translation."""
    sentinel = RecursionError("deterministic depth failure")
    calls = 0

    def raise_recursion(_payload: str) -> object:
        nonlocal calls
        calls += 1
        raise sentinel

    monkeypatch.setattr(json_mode_module.json, "loads", raise_recursion)
    completion = '{"thought":"t","action":"final","answer":"ok"}'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "JsonModeParser"
    assert ctx["error_phase"] == "json_decode"
    assert "RecursionError" in str(ctx["cause"])
    assert len(str(ctx["completion_excerpt"])) <= 200
    assert excinfo.value.__cause__ is sentinel
    assert calls == 1


def test_recovery_recursion_error_is_translated_to_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-017 deterministically pins recovery-pass ``RecursionError`` translation."""
    first = json.JSONDecodeError("strict failed", "prefix", 0)
    sentinel = RecursionError("deterministic recovery depth failure")
    errors = iter((first, sentinel))
    calls = 0

    def raise_scripted(_payload: str) -> object:
        nonlocal calls
        calls += 1
        raise next(errors)

    monkeypatch.setattr(json_mode_module.json, "loads", raise_scripted)
    completion = 'prefix {"thought":"t","action":"final","answer":"ok"} suffix'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "json_decode"
    assert "RecursionError" in str(ctx["cause"])
    assert len(str(ctx["completion_excerpt"])) <= 200
    assert excinfo.value.__cause__ is sentinel
    assert calls == 2


def test_oversized_integer_strict_decode_is_contained() -> None:
    """BR-013 contains bare ValueError from strict ``json.loads`` decoding."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(digits)
    assert str(excinfo.value) == "could not decode JSON envelope"
    assert excinfo.value.context["error_phase"] == "json_decode"
    assert len(str(excinfo.value.context["completion_excerpt"])) <= 200
    assert type(excinfo.value.__cause__) is ValueError


def test_oversized_integer_recovery_decode_is_contained() -> None:
    """BR-013 contains bare ValueError from the JSON recovery decode."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    completion = f'prefix {{"thought":"t","action":"final","answer":{digits}}} suffix'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode JSON envelope after fence recovery"
    assert excinfo.value.context["error_phase"] == "json_decode"
    assert type(excinfo.value.__cause__) is ValueError


def test_malformed_json_preserves_decode_message_and_json_cause() -> None:
    """BR-013 leaves the common malformed-syntax contract unchanged."""
    completion = "not json"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode JSON envelope"
    assert excinfo.value.context == {
        "parser": "JsonModeParser",
        "error_phase": "json_decode",
        "completion_excerpt": completion,
        "cause": repr(excinfo.value.__cause__),
    }
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


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
