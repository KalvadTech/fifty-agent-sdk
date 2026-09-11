"""Tolerant ReACT prose parser.

Consumes the classic ``Thought / Action / Action Input`` (or
``Thought / Final Answer``) format taught by
:data:`fifty_agent_sdk.prompts.PROSE_MODE_OUTPUT_FORMAT`. Tolerant of whitespace
and header capitalization. Strict in one specific way: if neither pattern
matches the completion as a whole, the parser raises
:class:`fifty_agent_sdk.errors.ParserError` rather than guessing.

Tie-break: when the completion contains BOTH ``Action:`` and ``Final Answer:``
headers, the tool-call path wins. Rationale: the loop terminates only on a
:class:`fifty_agent_sdk.parser.base.FinalAnswer`, so mis-firing a stale tool call
is strictly recoverable (the next iteration will re-parse), while
prematurely terminating on a stray ``Final Answer:`` is not.
"""

from __future__ import annotations

import json
import re
from typing import Final

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ToolCall
from fifty_agent_sdk.parser.base import FinalAnswer, ParseResult, ThoughtAction
from fifty_agent_sdk.parser.json_mode import _strip_code_fences

_MAX_EXCERPT: Final[int] = 200
"""Maximum length of ``completion_excerpt`` in :class:`ParserError` context."""

_TOOL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Thought:\s*(?P<thought>.*?)\s*"
    r"Action:\s*(?P<tool>[^\n]+?)\s*\n\s*"
    r"Action\s+Input:\s*(?P<args>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
"""Matches the tool-call form: ``Thought: ... Action: ... Action Input: ...``.

* Non-greedy quantifiers on ``thought``/``args`` so they don't swallow
  subsequent headers or trailing whitespace.
* ``[^\\n]+?`` on the tool name keeps it single-line.
* Anchored at both ends; ``re.DOTALL`` lets the bodies span newlines.
* Compiled once at module scope to avoid per-call cost and ReDoS surface.
"""

_FINAL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Thought:\s*(?P<thought>.*?)\s*"
    r"Final\s+Answer:\s*(?P<answer>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
"""Matches the final-answer form: ``Thought: ... Final Answer: ...``."""


class ProseModeParser:
    """Classic ReACT format parser; tolerant of whitespace and case variants.

    The parser attempts :data:`_TOOL_RE` first and only falls through to
    :data:`_FINAL_RE` when no tool form matches. See the module docstring
    for the rationale behind the tie-break.

    Failure surfaces:

    * ``error_phase="empty_completion"`` — empty/whitespace-only input.
    * ``error_phase="header_match"`` — neither pattern matched.
    * ``error_phase="action_input_decode"`` — the ``Action Input:`` body
      could not be parsed as JSON, even after the shared fence-stripping
      recovery pass, or it nested deeper than the interpreter recursion
      limit (:class:`RecursionError` from ``json.loads``, translated into
      :class:`ParserError` so the loop's ``ParserError`` contract holds).
    """

    def parse(self, completion: str) -> ParseResult:
        """See :meth:`fifty_agent_sdk.parser.base.Parser.parse`."""
        if not completion or not completion.strip():
            raise ParserError(
                "completion is empty",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "empty_completion",
                    "completion_excerpt": "",
                },
            )

        tool_match = _TOOL_RE.match(completion)
        if tool_match is not None:
            return self._parse_tool(tool_match, completion)

        final_match = _FINAL_RE.match(completion)
        if final_match is not None:
            return self._parse_final(final_match)

        raise ParserError(
            "no recognizable Thought/Action/Final Answer structure",
            context={
                "parser": "ProseModeParser",
                "error_phase": "header_match",
                "completion_excerpt": completion[:_MAX_EXCERPT],
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _parse_tool(self, match: re.Match[str], completion: str) -> ThoughtAction:
        """Build a :class:`ThoughtAction` from a tool-form match."""
        thought = match.group("thought").strip()
        tool_name = match.group("tool").strip()
        args_body = match.group("args").strip()
        args = self._decode_action_input(args_body, completion)
        return ThoughtAction(
            thought=thought,
            tool_call=ToolCall(name=tool_name, args=args),
        )

    def _parse_final(self, match: re.Match[str]) -> FinalAnswer:
        """Build a :class:`FinalAnswer` from a final-form match."""
        thought = match.group("thought").strip()
        answer = match.group("answer").strip()
        return FinalAnswer(thought=thought, content=answer)

    def _decode_action_input(self, body: str, completion: str) -> dict[str, object]:
        """Decode the ``Action Input:`` body as JSON, with one fence retry.

        :class:`RecursionError` from a pathologically nested body is caught at
        BOTH decode attempts and translated into :class:`ParserError`
        (``error_phase="action_input_decode"``) — the same ``ParserError``-only
        contract the JSON-mode parser upholds on its own decode path.
        """
        try:
            decoded = json.loads(body)
        except RecursionError as depth_err:
            raise ParserError(
                "could not decode Action Input JSON: nesting depth exceeds the recursion limit",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "action_input_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(depth_err),
                },
            ) from depth_err
        except json.JSONDecodeError as first_err:
            recovered = _strip_code_fences(body)
            if recovered is None:
                raise ParserError(
                    "could not decode Action Input JSON",
                    context={
                        "parser": "ProseModeParser",
                        "error_phase": "action_input_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(first_err),
                    },
                ) from first_err
            try:
                decoded = json.loads(recovered)
            except RecursionError as depth_err:
                raise ParserError(
                    "could not decode Action Input JSON: nesting depth exceeds the recursion limit",
                    context={
                        "parser": "ProseModeParser",
                        "error_phase": "action_input_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(depth_err),
                    },
                ) from depth_err
            except json.JSONDecodeError as second_err:
                raise ParserError(
                    "could not decode Action Input JSON after fence recovery",
                    context={
                        "parser": "ProseModeParser",
                        "error_phase": "action_input_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(second_err),
                    },
                ) from second_err

        if not isinstance(decoded, dict):
            raise ParserError(
                "Action Input JSON must decode to an object",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "action_input_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "decoded_type": type(decoded).__name__,
                },
            )
        return decoded


__all__ = ["ProseModeParser"]
