"""Shared ``$ref`` inlining for tool-facing JSON Schemas.

Both tool providers emit a :class:`fifty_agent_sdk.tools.protocol.ToolSchema`,
which carries only ``type`` / ``properties`` / ``required`` /
``additionalProperties`` — the loop copies exactly those four fields into the
system prompt and the native function-calling ``parameters`` envelope, and
provider function-calling APIs reject unknown top-level keys, so a ``$defs``
block cannot be shipped alongside. Pydantic v2 (and some MCP servers) emit
nested models as ``{"$ref": "#/$defs/Foo"}`` plus a top-level ``$defs`` dict;
keeping only ``properties``/``required`` would hand the LLM a dangling
pointer to a definition that does not exist. :func:`inline_local_refs`
resolves those local references before the schema is emitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

_DEFS_PREFIX: Final[str] = "#/$defs/"

_MAX_EXPANSION_DEPTH: Final[int] = 32
"""Cap on ``$ref`` expansion depth.

Guards against pathological reference chains (distinct defs referencing each
other in a long chain, or diamond fan-out) that no cycle-detection set
bounds. Genuine cycles are caught earlier and precisely by the expansion
stack in :func:`inline_local_refs`.
"""


def inline_local_refs(node: Any, defs: Mapping[str, Any]) -> Any:
    """Return a deep-resolved copy of ``node`` with local ``$defs`` refs inlined.

    Every ``{"$ref": "#/$defs/<name>"}`` mapping whose ``<name>`` is present
    in ``defs`` is replaced by the (recursively resolved) definition; sibling
    keys next to a ``$ref`` are preserved by merging them over the resolved
    definition (JSON Schema 2020-12 sibling semantics, which Pydantic v2
    relies on for e.g. a field ``description``). References that cannot be
    resolved locally — a missing ``<name>``, or a non-``#/$defs/`` URI — are
    left untouched: they were dangling before and inlining must not invent
    information the source schema did not carry.

    Args:
        node: The schema fragment to resolve (dicts and lists are walked
            recursively; scalars pass through).
        defs: The top-level ``$defs`` mapping the refs resolve against.

    Returns:
        A new structure; the inputs are never mutated.

    Raises:
        ValueError: On a reference cycle (a def that transitively references
            itself — a recursive model has no finite inline expansion) or when
            expansion exceeds the depth cap. The message names the cycle.
    """
    return _resolve(node, defs, _stack=(), _depth=0)


def _resolve(node: Any, defs: Mapping[str, Any], *, _stack: tuple[str, ...], _depth: int) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(_DEFS_PREFIX):
            name = ref[len(_DEFS_PREFIX) :]
            target = defs.get(name)
            if target is None:
                return {k: _resolve(v, defs, _stack=_stack, _depth=_depth) for k, v in node.items()}
            if name in _stack:
                cycle = " -> ".join([*_stack, name])
                raise ValueError(
                    f"recursive $ref cycle detected ({cycle}); a recursive model has no "
                    "finite inline expansion and cannot be emitted as a $defs-free schema"
                )
            if _depth >= _MAX_EXPANSION_DEPTH:
                raise ValueError(
                    f"$ref expansion exceeded {_MAX_EXPANSION_DEPTH} levels; the schema is "
                    "too deeply chained to inline"
                )
            resolved = _resolve(target, defs, _stack=(*_stack, name), _depth=_depth + 1)
            siblings = {
                k: _resolve(v, defs, _stack=_stack, _depth=_depth)
                for k, v in node.items()
                if k != "$ref"
            }
            if not siblings:
                return resolved
            if isinstance(resolved, dict):
                return {**resolved, **siblings}
            return resolved
        return {k: _resolve(v, defs, _stack=_stack, _depth=_depth) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(item, defs, _stack=_stack, _depth=_depth) for item in node]
    return node


__all__ = ["inline_local_refs"]
