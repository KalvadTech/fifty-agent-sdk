"""Redis implementation of :class:`StateStore` built on ``redis.asyncio``.

:class:`RedisStateStore` is the SDK's durable-but-ephemeral conversation-state
backend: it persists across process restarts (unlike
:class:`fifty_agent_sdk.state.memory.MemoryStateStore`) yet is a natural fit for a
TTL-bounded "hot session" cache rather than a system-of-record. For a true
system-of-record use the SQL backend (:class:`fifty_agent_sdk.state.sql.SqlStateStore`).

Extras requirement
    This module requires the optional ``redis`` extra::

        pip install 'fifty-agent-sdk[redis]'

    Importing :mod:`fifty_agent_sdk` itself does NOT pull redis-py. The Redis
    surface is re-exported lazily from :mod:`fifty_agent_sdk.state` and the
    package root via module-level ``__getattr__``; first access triggers
    this module's import, and a missing dependency surfaces as a clear
    :class:`ImportError` referencing the extras line above.

Key layout (BR-004 branching)
    The trunk branch reuses the bare session key, so pre-BR-004 single-list
    data IS the trunk with zero migration::

        <key_prefix><session_id>              # trunk's own message list
        <key_prefix><session_id>:branch:<id>  # a fork's own message list
        <key_prefix><session_id>:branches     # hash: branch_id -> metadata JSON
        <key_prefix><session_id>:active       # string: the active head branch id

    where ``key_prefix`` defaults to ``"fifty_agent_sdk:state:"``. Each message
    list holds JSON-encoded :class:`ChatMessage` payloads (``RPUSH`` to append,
    ``LRANGE 0 -1`` to read in append order). A branch's materialized history
    is ``parent_history[:fork_point] + own`` (see
    :meth:`RedisStateStore._materialize`); a message's materialized sequence is
    ``anchor + index``, so no sequence column is stored. Existing single-list
    sessions read as the trunk and gain the extra keys only when first forked.

TTL semantics
    When ``ttl_seconds`` is a positive integer, every state mutation re-issues
    ``EXPIRE`` across ALL of the session's keys (trunk, every fork list, the
    registry, and the active pointer) so the whole session's expiry window
    slides forward together — a "hot session stays alive" cache, and a fork's
    parent line never expires out from under it. ``append``, ``fork``,
    ``switch_branch``, and ``truncate_after`` share one optimistic transaction:
    registry, active pointer, and message lists are watched, the mutation and
    every ``EXPIRE`` execute together, and conflicts retry from a fresh snapshot
    up to a bounded limit. When ``ttl_seconds`` is ``None`` no ``EXPIRE`` is
    ever issued and the session is durable until :meth:`delete`. A non-positive
    ``ttl_seconds`` is rejected with
    :class:`ValueError` at construction: ``EXPIRE`` with ``0`` deletes a key
    immediately, so accepting it would wipe every session key on the first
    append. :meth:`get_messages` NEVER sets or refreshes a TTL —
    reading a session does not keep it alive.

Atomicity
    Every state mutation and its optional whole-session TTL refresh execute in
    one ``MULTI``/``EXEC`` transaction. Optimistic ``WATCH`` covers the branch
    registry, active pointer, and all enumerated message lists; sustained
    contention fails after three attempts through the normal
    :class:`StateStoreError` wrapping contract.

Error wrapping contract
    Every public method wraps :class:`redis.exceptions.RedisError` (the
    redis-py base exception class) into
    :class:`fifty_agent_sdk.errors.StateStoreError` with:

    * ``message``: ``"RedisStateStore.<operation> failed for session_id=<id>"``
    * ``context["session_id"]``: the input session id (echoed for log
      correlation)
    * ``context["wrapped"]``: the underlying exception's class name
      (e.g., ``"ConnectionError"``, ``"TimeoutError"``) — read by
      the Runner's ``runner.persist_failed`` ERROR log per TD-004
    * ``context["operation"]``: ``"get_messages"``, ``"append"``, or
      ``"delete"``
    * ``__cause__``: the original exception, via ``raise ... from exc``

    :class:`asyncio.CancelledError` propagates untouched (it is not a
    :class:`~redis.exceptions.RedisError`). Pydantic ``ValidationError`` on
    read — which would indicate a corrupt list member — is not wrapped
    either: it is a corruption signal, not a backend failure (same stance
    as the SQL backend).

Connection ownership
    The constructor accepts a connection URL string and the store creates
    and owns the underlying connection pool. :meth:`aclose` releases it;
    callers should invoke it in a ``finally`` block when done with the
    store.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Generic, TypeVar, cast

import structlog

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError, WatchError
except ImportError as exc:  # pragma: no cover - exercised via importlib in tests
    raise ImportError(
        "fifty_agent_sdk.state.redis requires redis-py. Install with: pip install 'fifty-agent-sdk[redis]'"
    ) from exc

from fifty_agent_sdk.errors import StateStoreError
from fifty_agent_sdk.llm.types import ChatMessage
from fifty_agent_sdk.state.protocol import TRUNK_BRANCH_ID, BranchInfo

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

Successful operations log at ``DEBUG`` with the session id and a small
shape summary (message count, TTL applied). Failures are NOT logged
here — the wrapped :class:`StateStoreError` carries everything the
Runner's ``runner.persist_failed`` ERROR log needs.
"""

_DEFAULT_KEY_PREFIX: Final = "fifty_agent_sdk:state:"
"""Default namespace prefix for session keys.

Prepended to each ``session_id`` to form the Redis key. A prefix keeps the
SDK's keys grouped under one namespace so they are easy to spot, scope, or
flush without disturbing co-tenant data in a shared Redis instance.
"""

_RESERVED_KEY_INFIXES: Final = (":branch:", ":branches", ":active")
"""Substrings the Redis backend reserves for its per-branch key layout (BR-004).

A ``session_id`` containing one of these would collide with another session's
auxiliary keys (registry hash / active pointer / fork lists), so the Redis
backend rejects it with :class:`ValueError`. SDK-generated UUID session ids
never contain them; hierarchical / tenant-derived ids must avoid them. Memory
and SQL have no such constraint — they do not derive structured keys.
"""

_MAX_MUTATION_RETRIES: Final[int] = 3
"""Maximum optimistic-lock attempts for one session mutation (BR-015)."""

_T = TypeVar("_T")


@dataclass(frozen=True)
class _SessionSnapshot:
    """Consistent session metadata captured under Redis ``WATCH``."""

    exists: bool
    active: str
    registry: dict[str, dict[str, Any]]
    branch_map: dict[str, tuple[str | None, int | None]]
    keys: tuple[str, ...]


@dataclass(frozen=True)
class _MutationPlan(Generic[_T]):
    """Commands and result prepared from one watched session snapshot."""

    apply: Callable[[Any], None] | None
    result: _T
    created_keys: tuple[str, ...] = ()


def _now() -> datetime:
    """Current timezone-aware UTC time (branch creation stamp)."""
    return datetime.now(UTC)


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (stored in the branch registry)."""
    return _now().isoformat()


def _wrap_state_store_error(
    exc: RedisError,
    *,
    session_id: str,
    operation: str,
) -> StateStoreError:
    """Build the SDK's standard wrap of a Redis backend failure.

    Returns the :class:`StateStoreError`; the caller writes
    ``raise _wrap_state_store_error(...) from exc`` so the ``__cause__``
    chain (the BR-009 error-wrapping contract, shared by BR-010) is
    preserved at every call site.
    """
    return StateStoreError(
        f"RedisStateStore.{operation} failed for session_id={session_id}",
        context={
            "session_id": session_id,
            "wrapped": type(exc).__name__,
            "operation": operation,
        },
    )


class RedisStateStore:
    """Redis-backed implementation of :class:`StateStore`.

    Satisfies :class:`fifty_agent_sdk.state.protocol.StateStore` structurally
    (no explicit inheritance needed thanks to ``@runtime_checkable``).

    Storage model:
        One Redis list per session, keyed ``<key_prefix><session_id>``,
        whose members are JSON-encoded :class:`ChatMessage` payloads.
        ``RPUSH`` appends; ``LRANGE 0 -1`` reads the whole list in append
        order. See the module docstring's "Key layout" section.

    TTL model:
        With a positive ``ttl_seconds`` every successful mutation slides every
        extant session key's expiry window forward together — a hot-session
        cache. With ``ttl_seconds=None`` no expiry command is emitted.
        :meth:`get_messages` never touches the TTL.

    Atomicity:
        Mutations and their whole-session TTL refresh run in one watched
        ``MULTI``/``EXEC`` transaction with bounded conflict retry.

    Connection ownership:
        The store owns the connection pool created from the URL.
        :meth:`aclose` releases it; call it in a ``finally`` block.

    Example:
        Construct from a URL and wire the store into an
        :class:`AgentRunner`, releasing the connection pool on exit::

            from fifty_agent_sdk import (
                JSON_MODE_OUTPUT_FORMAT, AgentLoop, AgentRunner,
                JsonModeParser, PromptSections, Registry, RedisStateStore,
                SafetyConfig,
            )
            from fifty_agent_sdk.llm import OpenAICompatibleClient

            state = RedisStateStore(
                "redis://localhost:6379/0", ttl_seconds=3600
            )
            try:
                runner = AgentRunner(
                    loop=AgentLoop(
                        llm=OpenAICompatibleClient(...),
                        registry=Registry(),
                        parser=JsonModeParser(),
                        prompts=PromptSections(persona="You are helpful."),
                        safety=SafetyConfig(),
                        model="gpt-4o",
                        output_format=JSON_MODE_OUTPUT_FORMAT,
                    ),
                    state=state,
                    system_prompt="You are a helpful customer-support agent.",
                )
                async for event in runner.run("session-abc", "Hello"):
                    print(event)
            finally:
                await state.aclose()

    Failure mode:
        Every public method wraps :class:`redis.exceptions.RedisError`
        into :class:`fifty_agent_sdk.errors.StateStoreError` with
        ``context["wrapped"]`` carrying the underlying class name. See
        the module docstring for the full contract.
    """

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str = _DEFAULT_KEY_PREFIX,
        ttl_seconds: int | None = None,
    ) -> None:
        """Construct a :class:`RedisStateStore`.

        Args:
            url: A redis-py connection URL (e.g.,
                ``"redis://localhost:6379/0"`` or
                ``"rediss://user:pass@host:6379/1"``). The store creates
                and owns the connection pool built from this URL.
            key_prefix: Namespace prepended to each ``session_id`` to form
                the Redis key. Defaults to :data:`_DEFAULT_KEY_PREFIX`
                (``"fifty_agent_sdk:state:"``).
            ttl_seconds: Per-session time-to-live, in seconds. When positive,
                every mutation re-issues ``EXPIRE`` across all session keys so
                the window slides forward as one unit. When ``None`` (the
                default), no ``EXPIRE`` is issued.

        Raises:
            ValueError: If ``ttl_seconds`` is not positive. ``EXPIRE``
                with a non-positive value deletes a key immediately, so
                accepting ``0`` or less would wipe every session key on
                the first append.
        """
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be a positive integer or None, got {ttl_seconds!r}")
        # ``decode_responses=True`` makes list members come back as ``str``
        # (rather than ``bytes``), ready to hand straight to
        # ``ChatMessage.model_validate_json``.
        self._client: aioredis.Redis = aioredis.from_url(url, decode_responses=True)
        self._key_prefix: str = key_prefix
        self._ttl_seconds: int | None = ttl_seconds

    def _key(self, session_id: str) -> str:
        """Return the Redis key for ``session_id``.

        The ``session_id`` is concatenated onto the configured prefix. It must
        not contain one of :data:`_RESERVED_KEY_INFIXES` (which would collide
        with another session's auxiliary keys); such ids are rejected with
        :class:`ValueError` — the only validation the Redis backend imposes on
        the otherwise-opaque id.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The fully-qualified Redis key (``<key_prefix><session_id>``).

        Raises:
            ValueError: If ``session_id`` contains a reserved key infix.
        """
        for infix in _RESERVED_KEY_INFIXES:
            if infix in session_id:
                raise ValueError(
                    f"session_id {session_id!r} contains reserved Redis key infix "
                    f"{infix!r} (reserved: {_RESERVED_KEY_INFIXES})"
                )
        return f"{self._key_prefix}{session_id}"

    async def aclose(self) -> None:
        """Release the underlying connection pool.

        Safe to call multiple times — redis-py tolerates repeated
        ``aclose`` calls. Callers should invoke this in a ``finally``
        block to ensure clean connection-pool teardown.
        """
        await self._client.aclose()

    # --- Branching key layout & helpers (BR-004) ----------------------------

    def _msgs_key(self, session_id: str, branch_id: str) -> str:
        """Redis key for a branch's OWN (non-inherited) message list.

        The trunk reuses the bare session key (``<prefix><session_id>``) so
        pre-BR-004 single-list data IS the trunk with zero migration; forks
        get a ``:branch:<branch_id>`` suffix.
        """
        if branch_id == TRUNK_BRANCH_ID:
            return self._key(session_id)
        return f"{self._key(session_id)}:branch:{branch_id}"

    def _branches_key(self, session_id: str) -> str:
        """Redis key for the branch-registry hash (``branch_id`` -> metadata JSON)."""
        return f"{self._key(session_id)}:branches"

    def _active_key(self, session_id: str) -> str:
        """Redis key for the active-head pointer (a string holding a branch id)."""
        return f"{self._key(session_id)}:active"

    async def _get_active(self, session_id: str) -> str:
        """Return the active branch id, defaulting to the trunk."""
        active = await cast("Any", self._client.get(self._active_key(session_id)))
        return str(active) if active is not None else TRUNK_BRANCH_ID

    async def _session_exists(self, session_id: str) -> bool:
        """True if any key backs this session (trunk list, registry, or active)."""
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.exists(self._msgs_key(session_id, TRUNK_BRANCH_ID))
            pipe.exists(self._branches_key(session_id))
            pipe.exists(self._active_key(session_id))
            results = await pipe.execute()
        return any(int(r) for r in results)

    async def _load_registry(self, session_id: str) -> dict[str, dict[str, Any]]:
        """Load the branch-registry hash, always including a (possibly
        synthesized) trunk entry so the lineage is resolvable."""
        raw = await cast("Any", self._client.hgetall(self._branches_key(session_id)))
        registry: dict[str, dict[str, Any]] = {bid: json.loads(meta) for bid, meta in raw.items()}
        if TRUNK_BRANCH_ID not in registry:
            registry[TRUNK_BRANCH_ID] = {
                "parent_branch_id": None,
                "forked_from_sequence": None,
                "created_at": None,
            }
        return registry

    @staticmethod
    def _branch_map(
        registry: dict[str, dict[str, Any]],
    ) -> dict[str, tuple[str | None, int | None]]:
        """Project the registry to a ``branch_id -> (parent, anchor)`` map."""
        return {
            bid: (meta["parent_branch_id"], meta["forked_from_sequence"])
            for bid, meta in registry.items()
        }

    async def _materialize(
        self,
        session_id: str,
        branch_id: str,
        branch_map: dict[str, tuple[str | None, int | None]],
    ) -> list[ChatMessage]:
        """Materialize a branch's full history: ``parent_history[:fork] + own``.

        Mirrors :meth:`MemoryStateStore._materialize`; each branch's own list
        is one ``LRANGE`` and the lineage is walked to the trunk.

        The walk is **iterative** (TD-002) — an ``await``ed recursive call
        consumes a Python frame just as a synchronous one does, so a deep
        linear fork chain used to raise :class:`RecursionError` here. The
        per-hop ``LRANGE`` count and the command set are unchanged; only the
        *order* flips, from leaf-to-root to root-to-leaf. Multi-key reads were
        never atomic in either order and no ordering was ever documented, so no
        contract moves. The per-hop calls are deliberately NOT pipelined.
        """
        # Phase 1: walk parent pointers leaf -> root. Reads only branch_map,
        # so this phase issues zero commands. No cycle guard, deliberately:
        # parent_branch_id is written once, in fork(), to the already-existing
        # active branch and never mutated, so the lineage is acyclic by
        # construction. Honest asymmetry (TD-002): a hand-edited registry hash
        # holding a cycle used to raise RecursionError and would now spin
        # forever; that is unreachable through any public API.
        chain: list[tuple[str, int | None]] = []
        bid: str | None = branch_id
        while bid is not None:
            parent, anchor = branch_map[bid]  # KeyError on a missing branch: preserved
            chain.append((bid, anchor))
            bid = parent
        chain.reverse()  # root .. leaf

        # Phase 2: fold root -> leaf, one LRANGE per hop. The base case and the
        # step mirror the recursive form literally, and stay separate, so the
        # two can be diffed line for line.
        root_id = chain[0][0]
        raw = await cast("Any", self._client.lrange(self._msgs_key(session_id, root_id), 0, -1))
        # `chain` is non-empty: branch_id is never None, so the walk ran once.
        history = [ChatMessage.model_validate_json(item) for item in raw]  # base case: the trunk
        for cur_id, cur_anchor in chain[1:]:  # step
            raw = await cast("Any", self._client.lrange(self._msgs_key(session_id, cur_id), 0, -1))
            own = [ChatMessage.model_validate_json(item) for item in raw]
            history = history[: cur_anchor or 0] + own
        return history

    async def _materialized_len(
        self,
        session_id: str,
        branch_id: str,
        branch_map: dict[str, tuple[str | None, int | None]],
    ) -> int:
        """Length of ``branch_id``'s materialized history.

        ``min(anchor, len(parent_history)) + own_len`` — the length analogue of
        :meth:`_materialize`. ``min`` is load-bearing when an ancestor was
        truncated below this branch's fork point. Used for ``fork`` bounds and
        :class:`BranchInfo.head_sequence`.

        **Computed iteratively** (TD-002). The recursive form was already a left
        fold from the root — each hop depends only on its parent — so the fold
        below runs root -> leaf and every hop's ``min`` still sees the
        already-``min``-clamped ancestor length. Same expression, same
        associativity. As in :meth:`_materialize` the per-hop ``LLEN`` count is
        unchanged and only the call *order* flips to root -> leaf; the calls are
        deliberately NOT pipelined.
        """
        # Walk parent pointers leaf -> root (zero commands), then fold
        # root -> leaf. No cycle guard, deliberately — see :meth:`_materialize`
        # for the reasoning and the honest asymmetry it accepts (TD-002).
        chain: list[tuple[str, int | None]] = []
        bid: str | None = branch_id
        while bid is not None:
            parent, anchor = branch_map[bid]  # KeyError on a missing branch: preserved
            chain.append((bid, anchor))
            bid = parent
        chain.reverse()  # root .. leaf

        # Base case and step mirror the recursive form literally, and stay
        # separate, so the two can be diffed line for line.
        root_id = chain[0][0]
        # `chain` is non-empty: branch_id is never None, so the walk ran once.
        length = int(await cast("Any", self._client.llen(self._msgs_key(session_id, root_id))))
        for cur_id, cur_anchor in chain[1:]:  # step
            own = int(await cast("Any", self._client.llen(self._msgs_key(session_id, cur_id))))
            length = min(cur_anchor or 0, length) + own
        return length

    async def _session_keys(self, session_id: str) -> list[str]:
        """Every Redis key backing a session (for TTL refresh and delete)."""
        fork_ids = await cast("Any", self._client.hkeys(self._branches_key(session_id)))
        keys = [
            self._msgs_key(session_id, TRUNK_BRANCH_ID),
            self._branches_key(session_id),
            self._active_key(session_id),
        ]
        keys.extend(self._msgs_key(session_id, fid) for fid in fork_ids if fid != TRUNK_BRANCH_ID)
        return keys

    async def _mutation_snapshot(self, pipe: Any, session_id: str) -> _SessionSnapshot:  # noqa: ANN401
        """Read and watch every key needed by a session mutation.

        The registry is watched before it is enumerated. Any concurrent fork
        changes that enumeration and therefore invalidates ``EXEC``. The active
        pointer and all known message lists are watched as well, making routing,
        validation, mutation, and the subsequent whole-session TTL refresh one
        optimistic transaction.
        """
        trunk_key = self._msgs_key(session_id, TRUNK_BRANCH_ID)
        branches_key = self._branches_key(session_id)
        active_key = self._active_key(session_id)
        base_keys = (trunk_key, branches_key, active_key)
        await pipe.watch(*base_keys)

        raw_registry = await pipe.hgetall(branches_key)
        raw_active = await pipe.get(active_key)
        exists = bool(await pipe.exists(*base_keys))
        registry: dict[str, dict[str, Any]] = {
            str(branch_id): json.loads(meta) for branch_id, meta in raw_registry.items()
        }
        if TRUNK_BRANCH_ID not in registry:
            registry[TRUNK_BRANCH_ID] = {
                "parent_branch_id": None,
                "forked_from_sequence": None,
                "created_at": None,
            }
        active = str(raw_active) if raw_active is not None else TRUNK_BRANCH_ID
        if active not in registry:
            active = TRUNK_BRANCH_ID

        fork_keys = tuple(
            self._msgs_key(session_id, branch_id)
            for branch_id in registry
            if branch_id != TRUNK_BRANCH_ID
        )
        if fork_keys:
            await pipe.watch(*fork_keys)
        keys = tuple(dict.fromkeys((*base_keys, *fork_keys)))
        return _SessionSnapshot(
            exists=exists,
            active=active,
            registry=registry,
            branch_map=self._branch_map(registry),
            keys=keys,
        )

    async def _mutate_session(
        self,
        session_id: str,
        prepare: Callable[[Any, _SessionSnapshot], Awaitable[_MutationPlan[_T]]],
    ) -> _T:
        """Apply one session mutation and slide every session key's TTL atomically.

        A ``WatchError`` retries from a fresh registry/active/key snapshot. The
        retry count is deliberately bounded so sustained contention surfaces
        through the existing ``RedisError`` -> ``StateStoreError`` translation.
        With ``ttl_seconds=None`` the same mutation transaction runs but queues
        no expiry commands.
        """
        last_error: WatchError | None = None
        for _attempt in range(_MAX_MUTATION_RETRIES):
            try:
                pipeline = cast("Any", self._client.pipeline(transaction=True))
                async with pipeline as pipe:
                    snapshot = await self._mutation_snapshot(pipe, session_id)
                    plan = await prepare(pipe, snapshot)
                    if plan.apply is None:
                        await pipe.unwatch()
                        return plan.result

                    pipe.multi()
                    plan.apply(pipe)
                    ttl = self._ttl_seconds
                    if ttl is not None:
                        ttl_keys = dict.fromkeys((*snapshot.keys, *plan.created_keys))
                        for key in ttl_keys:
                            pipe.expire(key, ttl)
                    await pipe.execute()
                    return plan.result
            except WatchError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    @staticmethod
    def _trunk_meta() -> str:
        """Serialize the lazy trunk-registry entry queued by every write."""
        return json.dumps(
            {"parent_branch_id": None, "forked_from_sequence": None, "created_at": _now_iso()}
        )

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        """Return the persisted messages for ``session_id``.

        Returns a freshly-constructed list of :class:`ChatMessage`
        instances in append order. An unknown session yields ``[]``:
        ``LRANGE`` on a missing key returns an empty list, satisfying the
        empty-session invariant with no special case. Reading does NOT
        refresh the session's TTL.

        Args:
            session_id: Opaque session identifier.
            branch_id: Which branch to read (BR-004). ``None`` reads the
                active branch; pre-M4 only the trunk exists.

        Returns:
            A list of :class:`ChatMessage` values in append order,
            possibly empty.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                redis-py exception class name.
        """
        try:
            registry = await self._load_registry(session_id)
            branch_map = self._branch_map(registry)
            if branch_id is not None:
                # An explicit branch request on an unknown session, or for a
                # non-existent branch, is a programmer error.
                if not await self._session_exists(session_id):
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for unknown session {session_id!r}"
                    )
                if branch_id not in branch_map:
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for session {session_id!r}"
                    )
                target = branch_id
            else:
                target = await self._get_active(session_id)
                if target not in branch_map:
                    target = TRUNK_BRANCH_ID
            # A fresh list per call satisfies the defensive-copy invariant.
            # ValidationError from a corrupt member is intentionally NOT caught:
            # a malformed payload is a corruption signal, not a backend failure.
            messages = await self._materialize(session_id, target, branch_map)
            _log.debug(
                "redis_state_store.get_messages",
                session_id=session_id,
                branch_id=target,
                count=len(messages),
            )
            return messages
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="get_messages"
            ) from exc

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's **active** branch.

        Issues ``RPUSH`` to the active branch's list inside a ``MULTI``/``EXEC``
        transaction so the write is atomic with respect to concurrent reads on
        that list. When ``ttl_seconds`` is set, the same transaction re-issues
        ``EXPIRE`` across ALL of the session's keys (trunk, every fork list,
        the registry, and the active pointer) so the whole session's expiry
        window slides forward together — a fork's parent line never expires out
        from under it.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                redis-py exception class name.
        """
        payload = message.model_dump_json()
        try:
            active = TRUNK_BRANCH_ID

            async def prepare(_pipe: Any, snapshot: _SessionSnapshot) -> _MutationPlan[None]:
                nonlocal active
                active = snapshot.active
                key = self._msgs_key(session_id, active)
                trunk_meta = self._trunk_meta()

                def apply(pipe: Any) -> None:
                    pipe.hsetnx(self._branches_key(session_id), TRUNK_BRANCH_ID, trunk_meta)
                    pipe.rpush(key, payload)

                return _MutationPlan(apply=apply, result=None, created_keys=(key,))

            await self._mutate_session(session_id, prepare)
            _log.debug(
                "redis_state_store.append",
                session_id=session_id,
                branch_id=active,
                ttl=self._ttl_seconds,
            )
        except RedisError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="append") from exc

    async def delete(self, session_id: str) -> None:
        """Remove all persisted state for ``session_id``.

        Idempotent — ``DEL`` on a missing key returns ``0`` and is a
        silent no-op, satisfying the idempotent-delete invariant with no
        special case.

        Args:
            session_id: Opaque session identifier.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                redis-py exception class name.
        """
        try:
            # Delete every key backing the session (trunk list, all fork
            # lists, the registry, and the active pointer). DEL ignores
            # missing keys, so this stays an idempotent no-op on an unknown
            # session. ``cast`` pins the awaitable arm for mypy --strict.
            keys = await self._session_keys(session_id)
            removed = int(await cast("Any", self._client.delete(*keys)))
            _log.debug(
                "redis_state_store.delete",
                session_id=session_id,
                existed=removed >= 1,
            )
        except RedisError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="delete") from exc

    async def fork(self, session_id: str, from_sequence: int) -> str:
        """Fork the active branch at ``from_sequence`` into a new branch.

        Records a new entry in the branch-registry hash whose parent is the
        active branch. The new branch's own list is created lazily on its
        first :meth:`append`. Does NOT change the active head. The registry
        write and optional whole-session TTL refresh are one transaction.

        Raises:
            ValueError: If the session is unknown, or ``from_sequence`` is
                outside ``0..head`` of the active branch.
            fifty_agent_sdk.errors.StateStoreError: On backend failure.
        """
        new_id = uuid.uuid4().hex
        try:

            async def prepare(_pipe: Any, snapshot: _SessionSnapshot) -> _MutationPlan[str]:
                if not snapshot.exists:
                    raise ValueError(f"cannot fork unknown session {session_id!r}")
                head = await self._materialized_len(
                    session_id, snapshot.active, snapshot.branch_map
                )
                if not 0 <= from_sequence <= head:
                    raise ValueError(
                        f"from_sequence={from_sequence} out of range 0..{head} "
                        f"for active branch {snapshot.active!r} of session {session_id!r}"
                    )
                meta = json.dumps(
                    {
                        "parent_branch_id": snapshot.active,
                        "forked_from_sequence": from_sequence,
                        "created_at": _now_iso(),
                    }
                )
                trunk_meta = self._trunk_meta()

                def apply(pipe: Any) -> None:
                    pipe.hsetnx(self._branches_key(session_id), TRUNK_BRANCH_ID, trunk_meta)
                    pipe.hset(self._branches_key(session_id), new_id, meta)

                return _MutationPlan(apply=apply, result=new_id)

            await self._mutate_session(session_id, prepare)
            _log.debug("redis_state_store.fork", session_id=session_id, branch_id=new_id)
            return new_id
        except RedisError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="fork") from exc

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        """Enumerate all branches of ``session_id`` (trunk first, then by age).

        An unknown session yields ``[]``. A pre-BR-004 session reports a single
        synthesized trunk (its ``created_at`` is approximated as "now", since
        Redis stores no per-key creation time).

        Raises:
            fifty_agent_sdk.errors.StateStoreError: On backend failure.
        """
        try:
            if not await self._session_exists(session_id):
                return []
            registry = await self._load_registry(session_id)
            branch_map = self._branch_map(registry)
            active = await self._get_active(session_id)
            infos: list[BranchInfo] = []
            for bid, meta in registry.items():
                created_raw = meta.get("created_at")
                created = datetime.fromisoformat(created_raw) if created_raw else _now()
                infos.append(
                    BranchInfo(
                        branch_id=bid,
                        parent_branch_id=meta["parent_branch_id"],
                        forked_from_sequence=meta["forked_from_sequence"],
                        head_sequence=await self._materialized_len(session_id, bid, branch_map),
                        created_at=created,
                        is_active=(bid == active),
                    )
                )
            infos.sort(
                key=lambda b: (
                    0 if b.branch_id == TRUNK_BRANCH_ID else 1,
                    b.created_at,
                    b.branch_id,
                )
            )
            return infos
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="list_branches"
            ) from exc

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        """Set the session's active head to ``branch_id``.

        The pointer write and optional whole-session TTL refresh are one
        optimistic transaction, so the pointer cannot outlive its messages.

        Raises:
            ValueError: If ``branch_id`` does not exist for this session.
            fifty_agent_sdk.errors.StateStoreError: On backend failure.
        """
        try:

            async def prepare(_pipe: Any, snapshot: _SessionSnapshot) -> _MutationPlan[None]:
                if not snapshot.exists or branch_id not in snapshot.registry:
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for session {session_id!r}"
                    )
                trunk_meta = self._trunk_meta()

                def apply(pipe: Any) -> None:
                    pipe.hsetnx(self._branches_key(session_id), TRUNK_BRANCH_ID, trunk_meta)
                    pipe.set(self._active_key(session_id), branch_id)

                return _MutationPlan(
                    apply=apply,
                    result=None,
                    created_keys=(self._active_key(session_id),),
                )

            await self._mutate_session(session_id, prepare)
            _log.debug(
                "redis_state_store.switch_branch", session_id=session_id, branch_id=branch_id
            )
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="switch_branch"
            ) from exc

    async def truncate_after(
        self, session_id: str, sequence: int, *, branch_id: str | None = None
    ) -> None:
        """Destructively trim the target branch's own list to messages with
        ``sequence <= N`` (``LTRIM``).

        Only the target branch's own list is trimmed; a fork's inherited prefix
        (held under ancestor keys) is never touched. Idempotent; a no-op on an
        unknown session or branch. With a TTL set, the session's expiry window
        is refreshed across all of its keys.
        """
        try:
            target = branch_id or TRUNK_BRANCH_ID

            async def prepare(_pipe: Any, snapshot: _SessionSnapshot) -> _MutationPlan[None]:
                nonlocal target
                if not snapshot.exists:
                    return _MutationPlan(apply=None, result=None)
                target = branch_id if branch_id is not None else snapshot.active
                if target not in snapshot.registry:
                    return _MutationPlan(apply=None, result=None)
                anchor = snapshot.registry[target]["forked_from_sequence"] or 0
                # Own message at index i has materialized sequence anchor + 1 + i;
                # keep the first ``N - anchor`` (those with sequence <= N).
                keep = sequence - anchor
                key = self._msgs_key(session_id, target)
                trunk_meta = self._trunk_meta()

                def apply(pipe: Any) -> None:
                    pipe.hsetnx(self._branches_key(session_id), TRUNK_BRANCH_ID, trunk_meta)
                    if keep <= 0:
                        pipe.ltrim(key, 1, 0)  # start > end empties the list
                    else:
                        pipe.ltrim(key, 0, keep - 1)

                return _MutationPlan(apply=apply, result=None)

            await self._mutate_session(session_id, prepare)
            _log.debug(
                "redis_state_store.truncate_after",
                session_id=session_id,
                branch_id=target,
                sequence=sequence,
            )
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="truncate_after"
            ) from exc


__all__ = ["RedisStateStore"]
