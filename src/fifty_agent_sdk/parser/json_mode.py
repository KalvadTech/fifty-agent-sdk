"""Default JSON-mode parser.

Consumes the JSON envelope shape locked in
:data:`fifty_agent_sdk.prompts.JSON_MODE_OUTPUT_FORMAT`:

.. code-block:: json

    {
      "thought":   string,
      "action":    "tool" | "final",
      "tool_name": string | null,
      "tool_args": object | null,
      "answer":    string | null
    }

Parsing is strict: an unknown top-level key or wrong ``action`` value raises
:class:`fifty_agent_sdk.errors.ParserError` with ``error_phase="schema_validation"``.

The parser performs **exactly one** syntactic recovery pass when
``json.loads`` fails on the raw input — it strips Markdown code fences (` ``` `
or ` ```json `) and/or slices between the first ``{`` and the last ``}``, then
re-attempts decoding. Schema-validation failures are *not* retried.

The :func:`_strip_code_fences` helper is shared with
:mod:`fifty_agent_sdk.parser.prose_mode` so the prose parser can recover JSON in
``Action Input:`` bodies the same way.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ToolCall
from fifty_agent_sdk.parser.base import FinalAnswer, ParseResult, ThoughtAction

_MAX_EXCERPT: Final[int] = 200
"""Maximum length of ``completion_excerpt`` in :class:`ParserError` context."""

_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)
"""Captures the body of a Markdown code fence (``json`` lang tag optional).

Non-greedy so two adjacent fenced blocks are not joined; only the first match
is used during recovery.
"""


class _RawEnvelope(BaseModel):
    """Internal validator for the :data:`JSON_MODE_OUTPUT_FORMAT` schema.

    ``extra="forbid"`` rejects unknown top-level keys; ``action`` is a literal
    union so any value besides ``"tool"`` / ``"final"`` raises immediately.

    ``tool_args=None`` is tolerated for either action and treated as ``{}``
    during the envelope→ParseResult conversion. ``tool_name`` / ``answer``
    presence is enforced semantically in
    :meth:`JsonModeParser._to_parse_result`.
    """

    model_config = ConfigDict(extra="forbid")

    thought: str
    action: Literal["tool", "final"]
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    answer: str | None = None


def _strip_code_fences(text: str) -> str | None:
    """Best-effort extraction of a JSON object from arbitrary text.

    Strategy, in order:

    1. If the input contains a Markdown code fence (``\\`\\`\\`json ... \\`\\`\\```
       or bare ``\\`\\`\\``` ... ``\\`\\`\\```), return the first fence body.
    2. Otherwise slice between the first ``{`` and the last ``}`` (inclusive).
    3. If neither a fence body nor a brace pair exists, return ``None``.

    The returned string is NOT validated as JSON — callers must
    :func:`json.loads` it themselves. This helper exists so the JSON-mode and
    prose-mode parsers share a single recovery rule.

    Args:
        text: Arbitrary text that may contain a JSON object.

    Returns:
        The recovered candidate substring, or ``None`` when no candidate
        could be located.
    """
    match = _FENCE_RE.search(text)
    candidate = match.group("body") if match else text
    open_idx = candidate.find("{")
    close_idx = candidate.rfind("}")
    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        return None
    return candidate[open_idx : close_idx + 1]


class JsonModeParser:
    """Strict JSON-envelope parser with one fence-stripping retry pass.

    The default parser when the loop instructs the model to use JSON mode
    (via :func:`fifty_agent_sdk.prompts.json_mode_template`). Consumes the schema
    defined in :data:`fifty_agent_sdk.prompts.JSON_MODE_OUTPUT_FORMAT`.

    Failure surfaces:

    * ``error_phase="empty_completion"`` — empty/whitespace-only input.
    * ``error_phase="json_decode"`` — both the strict and the recovery pass
      failed to produce valid JSON, or the input nested deeper than the
      interpreter recursion limit (``json.loads`` raises
      :class:`RecursionError` there, translated into :class:`ParserError`
      so the loop's ``ParserError`` contract holds).
    * ``error_phase="schema_validation"`` — JSON decoded but the envelope
      does not match the schema (unknown key, wrong ``action`` value,
      missing required field for the chosen action).
    """

    def parse(self, completion: str) -> ParseResult:
        """See :meth:`fifty_agent_sdk.parser.base.Parser.parse`."""
        if not completion or not completion.strip():
            raise ParserError(
                "completion is empty",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "empty_completion",
                    "completion_excerpt": "",
                },
            )
        raw = self._load_json(completion)
        envelope = self._validate(raw, completion)
        return self._to_parse_result(envelope, completion)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _load_json(self, completion: str) -> Any:
        """Strict-then-recover JSON decode. Raises on total failure.

        :class:`RecursionError` from a pathologically nested input is caught
        at BOTH decode attempts and translated into :class:`ParserError`
        (``error_phase="json_decode"``): the :class:`Parser` protocol mandates
        ``ParserError`` on malformed input, and the loop catches only
        ``ParserError`` — a raw ``RecursionError`` would escape the async
        generator with no ``ErrorEvent`` and no terminal ``FinalEvent``.
        """
        try:
            return json.loads(completion.strip())
        except RecursionError as depth_err:
            raise ParserError(
                "could not decode JSON envelope: nesting depth exceeds the recursion limit",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "json_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(depth_err),
                },
            ) from depth_err
        except ValueError as first_err:
            # CPython also raises bare ValueError for implementation limits
            # such as oversized integer literals. Keep it in the established
            # decode phase instead of parsing version-specific error text.
            recovered = _strip_code_fences(completion)
            if recovered is None:
                raise ParserError(
                    "could not decode JSON envelope",
                    context={
                        "parser": "JsonModeParser",
                        "error_phase": "json_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(first_err),
                    },
                ) from first_err
            try:
                return json.loads(recovered)
            except RecursionError as depth_err:
                raise ParserError(
                    "could not decode JSON envelope: nesting depth exceeds the recursion limit",
                    context={
                        "parser": "JsonModeParser",
                        "error_phase": "json_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(depth_err),
                    },
                ) from depth_err
            except ValueError as second_err:
                raise ParserError(
                    "could not decode JSON envelope after fence recovery",
                    context={
                        "parser": "JsonModeParser",
                        "error_phase": "json_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(second_err),
                    },
                ) from second_err

    def _validate(self, raw: Any, completion: str) -> _RawEnvelope:
        """Validate the decoded JSON against :class:`_RawEnvelope`."""
        try:
            return _RawEnvelope.model_validate(raw)
        except ValidationError as exc:
            raise ParserError(
                "JSON envelope did not match required schema",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "schema_validation",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": str(exc),
                },
            ) from exc

    def _to_parse_result(self, env: _RawEnvelope, completion: str) -> ParseResult:
        """Convert a validated envelope into the public :data:`ParseResult`."""
        if env.action == "tool":
            # Strip before the emptiness check (and before emitting the
            # ToolCall): a whitespace-only name must take the same
            # schema_validation path as an empty one, and a padded name
            # must reach the registry in the same stripped form the prose
            # parser produces (prose_mode strips the Action: header).
            tool_name = env.tool_name.strip() if env.tool_name is not None else None
            if not tool_name:
                raise ParserError(
                    "action='tool' requires non-empty tool_name",
                    context={
                        "parser": "JsonModeParser",
                        "error_phase": "schema_validation",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "missing": "tool_name",
                    },
                )
            tool_call = ToolCall(
                name=tool_name,
                args=env.tool_args if env.tool_args is not None else {},
            )
            return ThoughtAction(thought=env.thought, tool_call=tool_call)

        # env.action == "final" — Literal narrows the alternative away.
        if env.answer is None:
            raise ParserError(
                "action='final' requires non-null answer",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "schema_validation",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "missing": "answer",
                },
            )
        return FinalAnswer(thought=env.thought, content=env.answer)


__all__ = ["JsonModeParser"]
