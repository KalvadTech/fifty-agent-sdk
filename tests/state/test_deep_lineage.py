"""Deep-lineage regression tests (TD-002): materialization must not recurse.

Every backend used to walk a branch's lineage with one Python frame per hop, so
a pathological *linear* fork chain (~990 deep) raised :class:`RecursionError`.
The failure bit at **build** time before read time was reachable: ``fork`` computes the active
branch's head via the same walk to bound ``from_sequence``, so on unfixed code
the chain could not even be constructed through the public API. Every test in
this module fails with :class:`RecursionError` before the fix.

Depth tiers, and why they differ per backend
    Memory is the declared reference implementation and its public build path is
    affordable, so it carries the literal end-to-end reproduction: depth 2000
    built entirely through ``fork``/``switch_branch``. SQL and Redis seed the
    lineage in ONE bulk write and then drive the ordinary public
    ``get_messages`` / ``fork`` / ``truncate_after``, because their public build
    path is O(depth**2) in backend round-trips (``fork`` re-loads the whole
    branch map, and Redis's also re-walks the lineage per hop) — a cost entirely
    in the construction path, not in the property under test. Seeding buys depth
    past the recursion limit in milliseconds instead of tens of seconds.

What these tests deliberately do NOT pin
    They never call ``list_branches``. It is O(branches x depth) on every
    backend by design — hops scale with branches x depth — and the recursion property it
    would exercise is already covered by ``fork``'s head computation. Lineage
    behaviour of ``list_branches`` on a chain is covered at depth 40 by
    ``test_branching_differential.py::test_deep_linear_chain_agrees_across_backends``.
    They also do not pin the *absence* of a lineage-cycle guard (TD-002 declines
    one, deliberately); a cycle is unreachable through any public API, so no test
    can construct the case.

    **Most importantly, they are blind to fold DIRECTION.** Every chain here is
    homogeneous — each anchor is 1, the trunk holds one message, and no branch
    has own messages — so every branch's materialized length is 1, and folding
    leaf-to-root yields the identical answer. That homogeneity is what makes
    depth 1500 cheap, so it is not a defect to fix here; but it means a change
    that reverses the fold, or drops the ``min`` / prefix-slice clamp, passes
    this module untouched. *What* is computed is pinned by
    ``test_branching_differential.py`` (verified: all six such mutants fail
    there, and dropping ``min`` also fails one test here). These tests pin only
    that depth no longer consumes the interpreter stack. The two files are
    complements — **neither covers the other**, so if you are changing the fold,
    the differential suite is your guard, not this one.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import fakeredis
import pytest
import pytest_asyncio
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from fifty_agent_sdk import (
    TRUNK_BRANCH_ID,
    ChatMessage,
    MemoryStateStore,
    RedisStateStore,
    SqlStateStore,
    StateStore,
    sql_metadata,
)
from fifty_agent_sdk.state.sql import AgentBranch, AgentMessage, AgentSession

_MEMORY_DEPTH = 2000
"""Depth of the Memory end-to-end chain — comfortably past the ~990 limit."""

_SEEDED_DEPTH = 1500
"""Depth of the seeded SQL/Redis chains.

Chosen for cost, not for headroom. Any depth meaningfully past the ~990
default recursion limit proves the same property — that the lineage walk no
longer scales with the interpreter stack — so the extra 1000 hops bought no
additional evidence and cost roughly half a second of a ~13s suite, on three
CI Pythons. 1500 is ~50% past the limit, which is the margin worth paying for.
"""

_SID = "s"

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """In-memory aiosqlite engine with the schema created (StaticPool shape as
    in ``test_sql.py`` / ``test_branching_differential.py``)."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(sql_metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def redis_store() -> AsyncIterator[RedisStateStore]:
    """A :class:`RedisStateStore` backed by ``fakeredis`` (no network)."""
    rds = RedisStateStore("redis://localhost:6379/0")
    rds._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield rds
    finally:
        await rds.aclose()


def _M(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _texts(messages: list[ChatMessage]) -> list[str]:
    return [m.content for m in messages]


def _chain_ids(depth: int) -> list[str]:
    """Branch ids of a linear chain, in creation order (excluding the trunk)."""
    return [f"b{i}" for i in range(1, depth + 1)]


async def _build_memory_chain(store: MemoryStateStore, depth: int) -> str:
    """Build a depth-``depth`` linear chain through the PUBLIC API and return
    the leaf id. This loop IS the TD-002 reproduction: on unfixed code ``fork``
    raises :class:`RecursionError` once the lineage passes ~990 hops."""
    await store.append(_SID, _M("seed"))
    leaf = TRUNK_BRANCH_ID
    for _ in range(depth):
        leaf = await store.fork(_SID, 1)
        await store.switch_branch(_SID, leaf)
    return leaf


async def _seed_sql_chain(eng: AsyncEngine, depth: int) -> str:
    """Bulk-seed ``trunk -> b1 -> ... -> b<depth>`` (each anchored at 1, no own
    messages) plus one trunk message, and return the leaf id.

    One session row (active head = the leaf), one message row, and one
    executemany over the branch rows — so the lineage exists without paying the
    O(depth**2) public build path. ``created_at`` comes from the server default
    on every branch row, which is why these tests must not call
    ``list_branches`` (its ordering would be ambiguous).
    """
    ids = _chain_ids(depth)
    async with eng.begin() as conn:
        await conn.execute(insert(AgentSession).values(session_id=_SID, active_branch_id=ids[-1]))
        await conn.execute(
            insert(AgentMessage).values(
                session_id=_SID,
                branch_id=TRUNK_BRANCH_ID,
                sequence=1,
                role="user",
                content="seed",
            )
        )
        rows: list[dict[str, object]] = [
            {
                "session_id": _SID,
                "branch_id": TRUNK_BRANCH_ID,
                "parent_branch_id": None,
                "forked_from_sequence": None,
            }
        ]
        rows += [
            {
                "session_id": _SID,
                "branch_id": bid,
                "parent_branch_id": parent,
                "forked_from_sequence": 1,
            }
            for parent, bid in zip([TRUNK_BRANCH_ID, *ids[:-1]], ids, strict=True)
        ]
        await conn.execute(insert(AgentBranch), rows)
    return ids[-1]


async def _seed_redis_chain(store: RedisStateStore, depth: int) -> str:
    """Redis counterpart of :func:`_seed_sql_chain`: one ``HSET`` of the whole
    registry, one ``RPUSH``, one ``SET`` of the active pointer.

    Keys are derived via the store's own ``_msgs_key`` / ``_branches_key`` /
    ``_active_key`` (as the differential fixture already reaches for
    ``_client``) so a key-layout change breaks this seed loudly instead of
    silently seeding keys nothing reads.
    """
    ids = _chain_ids(depth)
    mapping: dict[str, str] = {
        TRUNK_BRANCH_ID: json.dumps(
            {"parent_branch_id": None, "forked_from_sequence": None, "created_at": None}
        )
    }
    for parent, bid in zip([TRUNK_BRANCH_ID, *ids[:-1]], ids, strict=True):
        mapping[bid] = json.dumps(
            {"parent_branch_id": parent, "forked_from_sequence": 1, "created_at": None}
        )
    client = store._client
    await client.hset(store._branches_key(_SID), mapping=mapping)
    await client.rpush(store._msgs_key(_SID, TRUNK_BRANCH_ID), _M("seed").model_dump_json())
    await client.set(store._active_key(_SID), ids[-1])
    return ids[-1]


# ---------------------------------------------------------------------------
# Memory — the end-to-end reproduction
# ---------------------------------------------------------------------------


async def test_memory_deep_linear_chain_materializes_via_public_api() -> None:
    """A depth-2000 linear fork chain can be BUILT and read through the public
    API (TD-002: ``fork``'s head computation used to raise RecursionError)."""
    store = MemoryStateStore()
    leaf = await _build_memory_chain(store, _MEMORY_DEPTH)

    messages = await store.get_messages(_SID)
    assert _texts(messages) == ["seed"]
    assert len(messages) == 1
    assert _texts(await store.get_messages(_SID, branch_id=leaf)) == ["seed"]


# ---------------------------------------------------------------------------
# SQL — seeded lineage, driven through the public API
# ---------------------------------------------------------------------------


async def test_sql_deep_linear_chain_materializes(engine: AsyncEngine) -> None:
    """Reading the leaf of a deep seeded chain exercises
    ``_materialize_positional`` at depth (TD-002)."""
    store = SqlStateStore(engine)
    leaf = await _seed_sql_chain(engine, _SEEDED_DEPTH)

    assert _texts(await store.get_messages(_SID)) == ["seed"]  # leaf is the active head
    assert _texts(await store.get_messages(_SID, branch_id=leaf)) == ["seed"]


async def test_sql_deep_linear_chain_head_bounds_fork(engine: AsyncEngine) -> None:
    """``fork`` bounds on a deep seeded chain exercise ``_materialized_len`` —
    including its per-hop ``min`` — at depth (TD-002)."""
    store = SqlStateStore(engine)
    await _seed_sql_chain(engine, _SEEDED_DEPTH)

    assert await store.fork(_SID, 1)  # head is 1: in range
    with pytest.raises(ValueError):
        await store.fork(_SID, 2)  # 2 > head 1


# ---------------------------------------------------------------------------
# Redis — seeded lineage, driven through the public API
# ---------------------------------------------------------------------------


async def test_redis_deep_linear_chain_materializes(redis_store: RedisStateStore) -> None:
    """Reading the leaf of a deep seeded chain exercises the (awaited, and so
    equally stack-consuming) ``_materialize`` at depth (TD-002)."""
    leaf = await _seed_redis_chain(redis_store, _SEEDED_DEPTH)

    assert _texts(await redis_store.get_messages(_SID)) == ["seed"]
    assert _texts(await redis_store.get_messages(_SID, branch_id=leaf)) == ["seed"]


async def test_redis_deep_linear_chain_head_bounds_fork(redis_store: RedisStateStore) -> None:
    """``fork`` bounds on a deep seeded chain exercise ``_materialized_len`` —
    including its per-hop ``min`` — at depth (TD-002)."""
    await _seed_redis_chain(redis_store, _SEEDED_DEPTH)

    assert await redis_store.fork(_SID, 1)  # head is 1: in range
    with pytest.raises(ValueError):
        await redis_store.fork(_SID, 2)  # 2 > head 1


# ---------------------------------------------------------------------------
# All three — clamping composed with depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["memory", "sql", "redis"])
async def test_deep_linear_chain_truncated_root_collapses_all_descendants(
    backend: str, engine: AsyncEngine, redis_store: RedisStateStore
) -> None:
    """Truncating the ROOT of a deep chain to 0 collapses every descendant:
    the leaf reads ``[]`` and its head is 0, so ``fork(1)`` is rejected.

    Composes the two properties that produced the BR-003 x BR-004 bugs — the
    inherited-prefix clamp (``min`` / slice) and thousands of lineage hops —
    which no existing test covered together (TD-002).
    """
    store: StateStore
    if backend == "memory":
        store = MemoryStateStore()
        leaf = await _build_memory_chain(store, _MEMORY_DEPTH)
    elif backend == "sql":
        store = SqlStateStore(engine)
        leaf = await _seed_sql_chain(engine, _SEEDED_DEPTH)
    else:
        store = redis_store
        leaf = await _seed_redis_chain(redis_store, _SEEDED_DEPTH)

    assert _texts(await store.get_messages(_SID, branch_id=leaf)) == ["seed"]

    await store.truncate_after(_SID, 0, branch_id=TRUNK_BRANCH_ID)

    assert await store.get_messages(_SID, branch_id=leaf) == []
    assert await store.get_messages(_SID) == []  # the leaf is the active head
    with pytest.raises(ValueError):
        await store.fork(_SID, 1)  # materialized head is now 0
