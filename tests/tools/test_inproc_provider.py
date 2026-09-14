"""Tests for fifty_agent_sdk.tools.inproc_provider — @tool decorator and InProcProvider."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel

from fifty_agent_sdk.tools.inproc_provider import InProcProvider, tool
from fifty_agent_sdk.tools.protocol import Tool
from fifty_agent_sdk.tools.registry import Registry

# ---------------------------------------------------------------------------
# @tool — schema generation
# ---------------------------------------------------------------------------


def test_tool_schema_simple_required_and_default() -> None:
    @tool()
    async def greet(name: str, age: int = 0) -> str:
        return f"hi {name} ({age})"

    schema = greet.schema
    assert schema.type == "object"
    assert schema.additionalProperties is False
    # `name` is required; `age` has a default so it is not.
    assert schema.required == ["name"]
    assert "name" in schema.properties
    assert schema.properties["name"]["type"] == "string"
    assert "age" in schema.properties
    assert schema.properties["age"]["type"] == "integer"
    assert schema.properties["age"]["default"] == 0


def test_tool_schema_optional_union() -> None:
    @tool()
    async def maybe(x: int | None = None) -> int | None:
        return x

    schema = maybe.schema
    assert schema.required == []
    prop = schema.properties["x"]
    # Pydantic v2 emits anyOf for `int | None`.
    assert "anyOf" in prop
    types = {entry.get("type") for entry in prop["anyOf"]}
    assert {"integer", "null"} <= types


def test_tool_schema_list_of_strings() -> None:
    @tool()
    async def list_items(items: list[str]) -> int:
        return len(items)

    prop = list_items.schema.properties["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"


def test_tool_schema_literal_becomes_enum() -> None:
    @tool()
    async def pick(mode: Literal["a", "b"]) -> str:
        return mode

    prop = pick.schema.properties["mode"]
    assert prop["enum"] == ["a", "b"]


def test_tool_schema_dict_annotation() -> None:
    @tool()
    async def with_dict(payload: dict[str, int]) -> int:
        return sum(payload.values())

    prop = with_dict.schema.properties["payload"]
    assert prop["type"] == "object"
    assert prop["additionalProperties"]["type"] == "integer"


# ---------------------------------------------------------------------------
# @tool — name and description resolution
# ---------------------------------------------------------------------------


def test_tool_default_name_is_function_name() -> None:
    @tool()
    async def my_fn(x: int) -> int:
        return x

    assert my_fn.name == "my_fn"


def test_tool_explicit_name_overrides_function_name() -> None:
    @tool(name="custom")
    async def my_fn(x: int) -> int:
        return x

    assert my_fn.name == "custom"


def test_tool_description_falls_back_to_docstring() -> None:
    @tool()
    async def documented(x: int) -> int:
        """A short docstring."""
        return x

    assert documented.description == "A short docstring."


def test_tool_explicit_description_overrides_docstring() -> None:
    @tool(description="Explicit doc")
    async def documented(x: int) -> int:
        """Ignored."""
        return x

    assert documented.description == "Explicit doc"


def test_tool_description_falls_back_to_name_when_no_doc() -> None:
    @tool()
    async def undocumented(x: int) -> int:
        return x

    assert undocumented.description == "undocumented"


# ---------------------------------------------------------------------------
# @tool — rejected signatures
# ---------------------------------------------------------------------------


def test_tool_rejects_sync_function() -> None:
    def sync_fn(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="async"):
        tool()(sync_fn)  # type: ignore[arg-type]


def test_tool_rejects_var_positional() -> None:
    async def variadic(*args: int) -> int:
        return sum(args)

    with pytest.raises(TypeError, match=r"\*args/\*\*kwargs"):
        tool()(variadic)


def test_tool_rejects_var_keyword() -> None:
    async def variadic(**kwargs: int) -> int:
        return sum(kwargs.values())

    with pytest.raises(TypeError, match=r"\*args/\*\*kwargs"):
        tool()(variadic)


def test_tool_rejects_unannotated_parameter() -> None:
    async def missing_hint(x) -> int:  # type: ignore[no-untyped-def]
        return x  # type: ignore[no-any-return]

    with pytest.raises(TypeError, match="no type annotation"):
        tool()(missing_hint)


def test_tool_rejects_positional_only() -> None:
    # Use exec to define a function with positional-only params (PEP 570).
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102
        "async def pos_only(x: int, /) -> int:\n    return x\n",
        namespace,
    )
    pos_only = namespace["pos_only"]
    with pytest.raises(TypeError, match="positional-only"):
        tool()(pos_only)


# ---------------------------------------------------------------------------
# @tool — runtime behavior
# ---------------------------------------------------------------------------


def test_decorated_callable_satisfies_tool_protocol() -> None:
    @tool()
    async def fn(x: int) -> int:
        return x + 1

    assert isinstance(fn, Tool)


async def test_decorated_callable_invokes_via_registry() -> None:
    @tool()
    async def add_one(x: int) -> int:
        return x + 1

    r = Registry()
    r.register(add_one)
    result = await r.invoke("add_one", {"x": 41}, timeout=1.0)
    assert result.is_error is False
    assert result.output == 42


async def test_decorated_callable_returns_error_on_bad_args() -> None:
    @tool()
    async def add_one(x: int) -> int:
        return x + 1

    r = Registry()
    r.register(add_one)
    # `unknown` is not a declared param; extra="forbid" rejects it.
    result = await r.invoke("add_one", {"unknown": 1}, timeout=1.0)
    assert result.is_error is True
    assert result.error is not None
    assert result.error.startswith("ValidationError:")


async def test_decorated_callable_returns_error_on_missing_required_arg() -> None:
    @tool()
    async def needs_x(x: int) -> int:
        return x

    r = Registry()
    r.register(needs_x)
    result = await r.invoke("needs_x", {}, timeout=1.0)
    assert result.is_error is True
    assert result.error is not None
    assert result.error.startswith("ValidationError:")


async def test_decorated_callable_handles_optional_default() -> None:
    @tool()
    async def maybe(x: int | None = None) -> str:
        return f"got {x}"

    r = Registry()
    r.register(maybe)
    result = await r.invoke("maybe", {}, timeout=1.0)
    assert result.is_error is False
    assert result.output == "got None"


# ---------------------------------------------------------------------------
# InProcProvider
# ---------------------------------------------------------------------------


def _build_three_tools() -> list[Tool]:
    @tool()
    async def t1(x: int) -> int:
        return x

    @tool()
    async def t2(y: str) -> str:
        return y

    @tool()
    async def t3(z: bool = False) -> bool:
        return not z

    return [t1, t2, t3]


def test_inproc_provider_register_into_adds_all_tools() -> None:
    provider = InProcProvider(_build_three_tools())
    registry = Registry()
    provider.register_into(registry)
    names = {t.name for t in registry.list()}
    assert names == {"t1", "t2", "t3"}


def test_inproc_provider_tools_snapshot_is_independent() -> None:
    tools = _build_three_tools()
    provider = InProcProvider(tools)
    snapshot = provider.tools()
    snapshot.clear()
    assert len(provider.tools()) == 3


async def test_inproc_provider_end_to_end_invocation() -> None:
    @tool(description="Adds two ints.")
    async def add(a: int, b: int) -> int:
        return a + b

    provider = InProcProvider([add])
    registry = Registry()
    provider.register_into(registry)
    result = await registry.invoke("add", {"a": 2, "b": 3}, timeout=1.0)
    assert result.is_error is False
    assert result.output == 5


# ---------------------------------------------------------------------------
# @tool — failure inside function body bubbles through the registry classifier
# ---------------------------------------------------------------------------


async def test_decorated_callable_runtime_error_wraps_via_registry() -> None:
    @tool()
    async def fails(x: int) -> int:
        raise RuntimeError(f"sad: {x}")

    r = Registry()
    r.register(fails)
    result = await r.invoke("fails", {"x": 7}, timeout=1.0)
    assert result.is_error is True
    assert result.error == "RuntimeError: sad: 7"


# ---------------------------------------------------------------------------
# @tool — nested BaseModel boundary
# ---------------------------------------------------------------------------


class _Address(BaseModel):
    """Defined at module scope so ``typing.get_type_hints`` can resolve it
    inside ``_build_model_from_signature`` (PEP 563 / ``from __future__
    import annotations`` evaluates annotations in the global namespace)."""

    street: str
    city: str


async def test_decorated_callable_preserves_nested_basemodel_instance() -> None:
    """A function annotated with a Pydantic model parameter must receive an
    instance of that model, not the underlying dict produced by ``model_dump``.

    Regression test for the ``model_dump`` -> ``getattr`` fix in ``_DecoratedTool.invoke``.
    """

    @tool()
    async def describe(address: _Address) -> dict[str, Any]:
        # The function body asserts the parameter type itself so the bug
        # surfaces inside the tool, not just in the returned payload.
        assert isinstance(address, _Address), f"expected _Address, got {type(address).__name__}"
        return {
            "type": type(address).__name__,
            "street": address.street,
            "city": address.city,
        }

    r = Registry()
    r.register(describe)
    result = await r.invoke(
        "describe",
        {"address": {"street": "1 Infinite Loop", "city": "Cupertino"}},
        timeout=1.0,
    )
    assert result.is_error is False, result.error
    assert result.output == {
        "type": "_Address",
        "street": "1 Infinite Loop",
        "city": "Cupertino",
    }


# ---------------------------------------------------------------------------
# @tool — nested BaseModel schemas: $defs inlined, no dangling $ref
# ---------------------------------------------------------------------------


def _collect_refs(node: Any) -> list[str]:
    """Recursively collect every ``$ref`` value in a schema structure."""
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.append(value)
            else:
                refs.extend(_collect_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_collect_refs(item))
    return refs


def test_tool_schema_nested_basemodel_refs_are_inlined() -> None:
    """A nested BaseModel parameter must not emit a dangling ``$ref``.

    Pydantic v2 emits nested models as ``{"$ref": "#/$defs/Foo"}`` plus a
    top-level ``$defs``; ``ToolSchema`` has no ``$defs`` slot and the loop
    ships only the four function-calling fields, so the refs are inlined.
    """

    @tool()
    async def lookup(address: _Address) -> str:
        return address.city

    schema = lookup.schema
    assert _collect_refs(schema.properties) == []
    prop = schema.properties["address"]
    assert prop["type"] == "object"
    assert prop["properties"]["street"]["type"] == "string"
    assert prop["properties"]["city"]["type"] == "string"
    assert schema.required == ["address"]


class _Order(BaseModel):
    """Doubly-nested model: its schema refs ``_Address``, which is itself a
    def — inlining must recurse through defs, not just expand one level."""

    shipping: _Address
    qty: int


def test_tool_schema_doubly_nested_basemodel_refs_are_inlined() -> None:
    @tool()
    async def place(order: _Order) -> str:
        return order.shipping.city

    schema = place.schema
    assert _collect_refs(schema.properties) == []
    shipping = schema.properties["order"]["properties"]["shipping"]
    assert shipping["properties"]["city"]["type"] == "string"


class _Node(BaseModel):
    """Recursive model: ``children`` refs ``_Node`` itself, so the def has no
    finite inline expansion."""

    name: str
    children: list[_Node] = []


def test_tool_schema_recursive_model_raises_clear_error() -> None:
    """A recursive model cannot be inlined; decoration must fail loudly rather
    than emit a dangling ``$ref``."""
    with pytest.raises(ValueError, match=r"recursive \$ref cycle"):

        @tool()
        async def walk(node: _Node) -> int:
            return 0
