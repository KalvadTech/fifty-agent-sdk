"""Tests for ``fifty_agent_sdk.parser.prose_mode.ProseModeParser``."""

from __future__ import annotations

import pytest

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser import FinalAnswer, Parser, ProseModeParser, ThoughtAction


def _parser() -> ProseModeParser:
    return ProseModeParser()


# ---------------------------------------------------------------------- #
# Happy paths                                                            #
# ---------------------------------------------------------------------- #


def test_happy_path_action() -> None:
    completion = 'Thought: I need to search.\nAction: search\nAction Input: {"q": "x"}'
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.thought == "I need to search."
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_happy_path_final_answer() -> None:
    completion = "Thought: I have the answer.\nFinal Answer: 42"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.thought == "I have the answer."
    assert result.content == "42"


def test_action_input_multiline_json_is_decoded() -> None:
    completion = (
        'Thought: Looking it up.\nAction: search\nAction Input: {\n  "q": "x",\n  "limit": 5\n}'
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {"q": "x", "limit": 5}


def test_action_input_with_code_fences_is_recovered() -> None:
    completion = 'Thought: t\nAction: search\nAction Input: ```json\n{"q": "x"}\n```'
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {"q": "x"}


def test_multiline_final_answer_is_captured() -> None:
    completion = "Thought: T\nFinal Answer: line1\nline2"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "line1\nline2"


# ---------------------------------------------------------------------- #
# Tolerance: case + whitespace                                           #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "completion",
    [
        "THOUGHT: T\nFINAL ANSWER: ok",
        "thought: T\nfinal answer: ok",
        "Thought:   T\n   Final Answer:   ok",
        "  \n\nThought: T\nFinal Answer: ok\n\n  ",
    ],
)
def test_case_variants_are_tolerated_for_final(completion: str) -> None:
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.thought == "T"
    assert result.content == "ok"


@pytest.mark.parametrize(
    "completion",
    [
        'THOUGHT: T\nACTION: search\nACTION INPUT: {"q": "x"}',
        'thought: T\naction: search\naction input: {"q": "x"}',
    ],
)
def test_case_variants_are_tolerated_for_action(completion: str) -> None:
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_whitespace_around_headers_tolerated() -> None:
    completion = (
        '\n\n   Thought:    T   \n   Action:    search   \n   Action Input:    {"q": "x"}   \n\n'
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.thought == "T"
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


# ---------------------------------------------------------------------- #
# Tie-break + special cases                                              #
# ---------------------------------------------------------------------- #


def test_both_action_and_final_answer_present_prefers_action() -> None:
    """Documented tie-break: tool path wins when both shapes are present."""
    completion = 'Thought: T\nAction: search\nAction Input: {"q": "x"}\nFinal Answer: stale'
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"


def test_missing_final_answer_body_is_tolerated_as_empty() -> None:
    """Documented choice: prose parser is tolerant; empty content is fine."""
    completion = "Thought: T\nFinal Answer:"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == ""


# ---------------------------------------------------------------------- #
# Failure paths                                                          #
# ---------------------------------------------------------------------- #


def test_missing_action_input_raises_header_match() -> None:
    completion = "Thought: T\nAction: search"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "header_match"


def test_no_headers_raises_header_match() -> None:
    completion = "random prose with no structure at all"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "ProseModeParser"
    assert ctx["error_phase"] == "header_match"
    assert "completion_excerpt" in ctx


def test_action_input_invalid_json_raises_action_input_decode() -> None:
    completion = "Thought: T\nAction: search\nAction Input: not json"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "action_input_decode"


def test_action_input_non_object_json_raises() -> None:
    """Action Input must decode to an object, not a scalar/list."""
    completion = "Thought: T\nAction: search\nAction Input: [1, 2, 3]"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "action_input_decode"


def test_empty_completion_raises() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_whitespace_only_completion_raises() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("   \n\t  ")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_header_match_completion_excerpt_is_bounded() -> None:
    big = "no headers here " * 100  # > 200 chars, no headers
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(big)
    excerpt = excinfo.value.context["completion_excerpt"]
    assert len(excerpt) <= 200


def test_huge_whitespace_payload_does_not_hang() -> None:
    """ReDoS sanity check: large whitespace-only input fails fast."""
    payload = " " * 10000
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(payload)
    # whitespace-only triggers the empty_completion guard before regex.
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_deeply_nested_action_input_raises_parser_error_not_recursion_error() -> None:
    """Pathological nesting trips ``RecursionError`` inside ``json.loads``; the
    Parser contract mandates ``ParserError`` (see the JSON-mode twin test).
    Depth 100k exceeds the recursion threshold of both the C and the pure
    Python ``json`` scanners regardless of ``sys.getrecursionlimit``."""
    depth = 100_000
    body = "[" * depth + "]" * depth
    completion = f"Thought: T\nAction: search\nAction Input: {body}"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "ProseModeParser"
    assert ctx["error_phase"] == "action_input_decode"
    assert "RecursionError" in str(ctx["cause"])


def test_deeply_nested_fenced_action_input_raises_parser_error() -> None:
    """Recovery-pass arm: the strict pass fails with JSONDecodeError on the
    fence wrapper, the recovered fence body is the deeply nested payload, and
    the second ``json.loads`` raises ``RecursionError`` — still ``ParserError``.
    The payload uses object nesting (not arrays) because the fence-recovery
    slicer keys off ``{``/``}``."""
    depth = 100_000
    body = '{"a":' * depth + "1" + "}" * depth
    completion = f"Thought: T\nAction: search\nAction Input: ```json\n{body}\n```"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "action_input_decode"
    assert "RecursionError" in str(ctx["cause"])


# ---------------------------------------------------------------------- #
# Protocol                                                               #
# ---------------------------------------------------------------------- #


def test_parser_protocol_satisfied() -> None:
    assert isinstance(_parser(), Parser)
