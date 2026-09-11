"""The ``@tool`` decorator and :class:`InProcProvider` helper.

This module turns plain Python ``async`` callables into :class:`Tool`
instances. The decorator auto-derives a JSON Schema from the callable's
signature via a dynamically-built Pydantic model, so writing a tool boils
down to writing an annotated ``async def``.

Rich types are supported via Pydantic's JSON Schema emitter:
``int | None``, ``list[str]``, ``dict[str, int]``, ``Literal[...]``,
``Enum``, nested ``BaseModel`` — anything Pydantic can validate it can also
emit a schema for. Nested-model ``#/$defs/...`` references are INLINED into
the emitted :class:`fifty_agent_sdk.tools.protocol.ToolSchema` (see
:mod:`fifty_agent_sdk.tools._schema_refs`) so the schema the LLM receives is
self-contained; a recursive model has no finite inline expansion and is
rejected at decoration time with a :class:`ValueError`.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable, Iterable
from inspect import Parameter
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic_core import PydanticUndefined

from fifty_agent_sdk.tools._schema_refs import inline_local_refs
from fifty_agent_sdk.tools.protocol import Tool, ToolResult, ToolSchema
from fifty_agent_sdk.tools.registry import Registry

ToolFn = Callable[..., Awaitable[Any]]
"""An ``async`` callable suitable for ``@tool`` decoration."""


class _DecoratedTool:
    """Generated :class:`Tool` implementation produced by the ``@tool`` decorator.

    Validates ``args`` against the generated input model on every invocation,
    then calls the underlying function with ``**kwargs``. Validation errors
    are returned as ``ToolResult(is_error=True)`` — they reflect the LLM
    passing bad arguments, not a system fault.
    """

    def __init__(
        self,
        fn: ToolFn,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        schema: ToolSchema,
    ) -> None:
        self._fn = fn
        self.name = name
        self.description = description
        self.schema = schema
        self._input_model = input_model

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Validate ``args`` and call the underlying function.

        Bad LLM args produce ``ToolResult(is_error=True)`` rather than
        raising — the LLM should see argument failures as data and recover.
        Genuine system errors raised by the function body propagate to the
        :class:`Registry.invoke` classifier.
        """
        try:
            validated = self._input_model.model_validate(args)
        except ValidationError as e:
            return ToolResult(
                output=None,
                is_error=True,
                error=f"ValidationError: {e}",
            )
        # Use attribute access rather than ``model_dump()`` so nested
        # ``BaseModel`` parameters reach the function as their original
        # instances rather than being recursively serialized back into dicts.
        kwargs = {k: getattr(validated, k) for k in self._input_model.model_fields}
        output = await self._fn(**kwargs)
        return ToolResult(output=output, is_error=False, error=None)


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[ToolFn], Tool]:
    """Decorator: convert an ``async`` function into a :class:`Tool`.

    Usage::

        @tool(description="Look up a customer by ID")
        async def get_customer(customer_id: str) -> dict[str, Any]:
            ...

    The decorated function MUST be ``async def``. All parameters MUST be
    type-annotated. ``*args``, ``**kwargs``, and positional-only parameters
    are rejected at decoration time with a clear error message.

    Args:
        name: Override the tool name. Defaults to ``fn.__name__``.
        description: Tool description shown to the LLM. Defaults to
            ``inspect.getdoc(fn)`` (stripped) if available; otherwise falls
            back to the resolved tool name. Providing an explicit description
            is strongly recommended for LLM clarity.

    Returns:
        A decorator that wraps the function in an object satisfying the
        :class:`Tool` protocol.

    Raises:
        TypeError: At decoration time if ``fn`` is not ``async``, uses
            ``*args``/``**kwargs``, has a positional-only parameter, or has
            any unannotated parameter. The error message names the offending
            parameter.
        ValueError: At decoration time if a parameter's model is recursive
            (a ``$ref`` cycle): recursive models have no finite inline
            expansion and cannot be emitted as a ``$defs``-free schema.
    """

    def decorator(fn: ToolFn) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@tool requires an async function; '{fn.__name__}' is not async")
        resolved_name = name or fn.__name__
        resolved_description = description or (inspect.getdoc(fn) or "").strip() or resolved_name
        input_model = _build_model_from_signature(fn)
        schema = _schema_from_model(input_model)
        decorated = _DecoratedTool(
            fn,
            name=resolved_name,
            description=resolved_description,
            input_model=input_model,
            schema=schema,
        )
        # _DecoratedTool structurally satisfies the Tool Protocol; cast so
        # mypy --strict sees the return type as Tool.
        return cast(Tool, decorated)

    return decorator


class InProcProvider:
    """Convenience helper: bulk-register decorated callables into a :class:`Registry`.

    Not part of the :class:`Tool` contract — purely ergonomic sugar.
    Equivalent to looping ``registry.register(t)`` over the provided tools,
    but groups related tools so they can be wired into a registry in one
    call.
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: list[Tool] = list(tools)

    def register_into(self, registry: Registry) -> None:
        """Register every tool managed by this provider into ``registry``."""
        for t in self._tools:
            registry.register(t)

    def tools(self) -> list[Tool]:
        """Return a snapshot of the tools this provider manages.

        Mutating the returned list does not affect the provider.
        """
        return list(self._tools)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_model_from_signature(fn: ToolFn) -> type[BaseModel]:
    """Build a Pydantic model whose fields mirror ``fn``'s parameters.

    The model has ``extra="forbid"`` so unknown keys raise a
    :class:`pydantic.ValidationError`, which :meth:`_DecoratedTool.invoke`
    translates into ``ToolResult(is_error=True)``.

    Raises:
        TypeError: If the function uses ``*args``/``**kwargs``, a
            positional-only parameter, or has any unannotated parameter.
    """
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise TypeError(f"@tool callable '{fn.__name__}' cannot use *args/**kwargs")
        if param.kind == Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"@tool callable '{fn.__name__}' cannot use positional-only parameter '{pname}'"
            )
        if pname not in hints:
            raise TypeError(
                f"@tool callable '{fn.__name__}' parameter '{pname}' has no type annotation"
            )
        annotation = hints[pname]
        default = param.default if param.default is not Parameter.empty else PydanticUndefined
        fields[pname] = (annotation, Field(default=default))
    model_name = f"{_camel(fn.__name__)}Args"
    return cast(
        type[BaseModel],
        create_model(
            model_name,
            __config__=ConfigDict(extra="forbid"),
            **fields,
        ),
    )


def _schema_from_model(model: type[BaseModel]) -> ToolSchema:
    """Translate the model's JSON Schema into a :class:`ToolSchema`.

    Pydantic emits ``{"type": "object", "properties": {...}, "required": [...]}``
    for a plain model; these pass through. ``additionalProperties`` is forced
    to ``False`` to mirror the model's ``extra="forbid"`` config.

    Nested models come out of Pydantic v2 as ``{"$ref": "#/$defs/Foo"}`` plus
    a top-level ``$defs`` dict. ``ToolSchema`` has no ``$defs`` slot and the
    loop ships only the four function-calling fields (providers reject unknown
    top-level keys), so the refs are INLINED via
    :func:`fifty_agent_sdk.tools._schema_refs.inline_local_refs` — shipping the
    ``$ref`` without its definition would hand the LLM a dangling pointer.
    Decoration raises :class:`ValueError` when a recursive model has no finite
    expansion, a reference chain exceeds the depth cap, or expansion exhausts
    the shared node budget. Cycle failures name the cycle; limit failures carry
    bounded content-free messages.
    """
    raw = model.model_json_schema()
    defs_raw = raw.get("$defs", {})
    defs: dict[str, Any] = dict(defs_raw) if isinstance(defs_raw, dict) else {}
    properties_raw = raw.get("properties", {})
    required_raw = raw.get("required", [])
    properties: dict[str, Any] = (
        inline_local_refs(properties_raw, defs) if isinstance(properties_raw, dict) else {}
    )
    required: list[str] = list(required_raw) if isinstance(required_raw, list) else []
    return ToolSchema(
        type=str(raw.get("type", "object")),
        properties=properties,
        required=required,
        additionalProperties=False,
    )


def _camel(snake: str) -> str:
    """Convert ``snake_case`` into ``CamelCase`` for generated model names."""
    return "".join(part.title() for part in snake.split("_") if part)


__all__ = ["InProcProvider", "tool"]
