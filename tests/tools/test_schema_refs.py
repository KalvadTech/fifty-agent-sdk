"""Tests for bounded local JSON-Schema reference expansion (BR-016)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from fifty_agent_sdk.tools._schema_refs import _MAX_EXPANDED_NODES, inline_local_refs


def test_inline_local_refs_resolves_shared_refs_without_mutating_inputs() -> None:
    """BR-016 preserves ordinary shared-ref expansion and input immutability."""
    node = {
        "left": {"$ref": "#/$defs/Leaf"},
        "right": {"$ref": "#/$defs/Leaf", "description": "right leaf"},
    }
    defs = {"Leaf": {"type": "object", "properties": {"value": {"type": "string"}}}}
    original_node = deepcopy(node)
    original_defs = deepcopy(defs)

    resolved = inline_local_refs(node, defs)

    assert resolved["left"]["properties"]["value"] == {"type": "string"}
    assert resolved["right"]["description"] == "right leaf"
    assert node == original_node
    assert defs == original_defs


def test_inline_local_refs_leaves_missing_and_external_refs_untouched() -> None:
    """BR-016 does not invent definitions for refs outside the local map."""
    node = {
        "missing": {"$ref": "#/$defs/Missing"},
        "external": {"$ref": "https://example.invalid/schema"},
    }
    assert inline_local_refs(node, {}) == node


def test_inline_local_refs_rejects_cycles_before_budget_exhaustion() -> None:
    """BR-016 preserves the precise recursive-cycle failure."""
    defs = {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}}
    with pytest.raises(ValueError, match=r"recursive \$ref cycle.*Node -> Node"):
        inline_local_refs({"$ref": "#/$defs/Node"}, defs)


def _reference_chain(length: int) -> tuple[dict[str, str], dict[str, Any]]:
    defs: dict[str, Any] = {}
    for index in range(length):
        defs[f"D{index}"] = (
            {"$ref": f"#/$defs/D{index + 1}"} if index + 1 < length else {"type": "string"}
        )
    return {"$ref": "#/$defs/D0"}, defs


def test_inline_local_refs_accepts_depth_boundary() -> None:
    """BR-016 preserves the documented 32-reference depth boundary."""
    node, defs = _reference_chain(32)
    assert inline_local_refs(node, defs) == {"type": "string"}


def test_inline_local_refs_rejects_over_depth_boundary() -> None:
    """BR-016 preserves rejection beyond the 32-reference depth boundary."""
    node, defs = _reference_chain(33)
    with pytest.raises(ValueError, match="exceeded 32 levels"):
        inline_local_refs(node, defs)


def test_inline_local_refs_accepts_exact_node_budget() -> None:
    """BR-016 permits a schema whose resolver visits exactly the node budget."""
    node = [0] * (_MAX_EXPANDED_NODES - 1)
    assert inline_local_refs(node, {}) == node


def test_inline_local_refs_rejects_one_node_over_budget() -> None:
    """BR-016 rejects before visiting beyond the shared work/output budget."""
    node = [0] * _MAX_EXPANDED_NODES
    with pytest.raises(ValueError, match="node budget"):
        inline_local_refs(node, {})


def test_inline_local_refs_bounds_acyclic_exponential_fanout() -> None:
    """BR-016 bounds acyclic diamond fan-out that a depth limit cannot contain."""
    defs: dict[str, Any] = {"D14": {"type": "string"}}
    for index in range(13, -1, -1):
        child = {"$ref": f"#/$defs/D{index + 1}"}
        defs[f"D{index}"] = {"left": child, "right": child}

    with pytest.raises(ValueError, match="node budget"):
        inline_local_refs({"$ref": "#/$defs/D0"}, defs)


def test_inline_local_refs_contains_ordinary_nesting_recursion_error() -> None:
    """BR-016 ordinary dict/list depth cannot escape as raw RecursionError."""
    node: Any = "SCHEMA_SECRET_MUST_NOT_ESCAPE"
    for index in range(1_100):
        node = {"level": index, "child": [node]}

    with pytest.raises(ValueError, match="schema traversal exceeded the safe nesting limit") as exc:
        inline_local_refs(node, {})
    assert isinstance(exc.value.__cause__, RecursionError)
    assert "SCHEMA_SECRET_MUST_NOT_ESCAPE" not in str(exc.value)
