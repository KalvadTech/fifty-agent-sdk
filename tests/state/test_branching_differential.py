"""Differential (fuzz) tests: Memory vs SQL vs Redis must behave identically.

MemoryStateStore is the declared reference implementation (protocol.py); SQL and
Redis must match it operation-for-operation. This module drives all three
backends through the SAME randomized sequences of branching operations and
asserts their observable results never diverge. It also pins three explicit
regressions for bugs found in pre-release review (the BR-003 × BR-004
interaction): SQL by-position materialization after an ancestor truncation,
SQL/Redis materialized-length head after an ancestor truncation, and the Redis
trunk-only-truncated-to-empty session-existence bug.

It also carries a *linear-chain* agreement case (TD-002): the randomized
sequences fork from whatever branch is active, which produces wide, shallow
trees, so a 40-deep chain is built explicitly. Depth past Python's recursion
limit is pinned separately in ``test_deep_lineage.py``; the case here pins that
making the lineage walk iterative did not change WHAT any backend computes.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator

import fakeredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
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


@pytest_asyncio.fixture
async def stores() -> AsyncIterator[list[tuple[str, StateStore]]]:
    """Memory, SQL (in-memory aiosqlite), and Redis (fakeredis) side by side.

    Memory is first so it serves as the reference in comparisons.
    """
    mem = MemoryStateStore()
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(sql_metadata.create_all)
    sql = SqlStateStore(eng)
    rds = RedisStateStore("redis://localhost:6379/0")
    rds._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield [("mem", mem), ("sql", sql), ("redis", rds)]
    finally:
        await sql.aclose()
        await eng.dispose()
        await rds.aclose()


def _texts(messages: list[ChatMessage]) -> list[str]:
    return [m.content for m in messages]


def _M(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


async def _read_or_err(store: StateStore, sid: str, branch_id: str) -> object:
    """A branch read rendered as a comparable value: the message texts, or
    ``"ERR:<Type>"`` if the backend raises. Comparing this across backends
    asserts BOTH content and error-consistency (e.g. an explicit branch read on
    a not-yet-created session must raise on all three, not raise on some)."""
    try:
        return _texts(await store.get_messages(sid, branch_id=branch_id))
    except ValueError as exc:
        return f"ERR:{type(exc).__name__}"


async def _assert_agree(stores: list[tuple[str, StateStore]], bids: dict[str, list[str]]) -> None:
    """Assert all backends agree on the active read, every branch read (value or
    raised error), and every branch's head_sequence."""
    sid = "s"
    ref_name, ref_store = stores[0]

    # active-branch read (branch_id=None never raises for an unknown session)
    ref_active = _texts(await ref_store.get_messages(sid))
    for name, store in stores[1:]:
        got = _texts(await store.get_messages(sid))
        assert got == ref_active, f"active read: {name}={got} {ref_name}={ref_active}"

    # each branch by creation index — compare the value OR the raised error type
    for i in range(len(bids[ref_name])):
        ref_branch = await _read_or_err(ref_store, sid, bids[ref_name][i])
        for name, store in stores[1:]:
            got = await _read_or_err(store, sid, bids[name][i])
            assert got == ref_branch, f"branch[{i}]: {name}={got} {ref_name}={ref_branch}"

    # head_sequence per branch index (None if the branch/session is absent)
    def heads_of(lb: dict[str, int], name: str) -> list[int | None]:
        return [lb.get(bids[name][i]) for i in range(len(bids[name]))]

    ref_heads = heads_of(
        {b.branch_id: b.head_sequence for b in await ref_store.list_branches(sid)}, ref_name
    )
    for name, store in stores[1:]:
        lb = {b.branch_id: b.head_sequence for b in await store.list_branches(sid)}
        assert heads_of(lb, name) == ref_heads, (
            f"heads: {name}={heads_of(lb, name)} {ref_name}={ref_heads}"
        )


@pytest.mark.parametrize("seed", range(10))
async def test_backends_agree_under_random_ops(
    stores: list[tuple[str, StateStore]], seed: int
) -> None:
    rng = random.Random(seed)
    sid = "s"
    bids: dict[str, list[str]] = {name: [TRUNK_BRANCH_ID] for name, _ in stores}
    counter = 0

    for step in range(60):
        op = rng.choice(["append", "append", "append", "fork", "switch", "truncate", "truncate"])
        if op == "append":
            msg = _M(f"m{counter}")
            counter += 1
            for _, store in stores:
                await store.append(sid, msg)
        elif op == "fork":
            f = rng.randint(0, 8)
            results: dict[str, str] = {}
            errored: set[str] = set()
            for name, store in stores:
                try:
                    results[name] = await store.fork(sid, f)
                except ValueError:
                    errored.add(name)
            # fork validation must be all-or-none across backends
            assert errored == set() or errored == {n for n, _ in stores}, (
                f"fork({f}) consistency seed={seed} step={step}: errored={errored}"
            )
            if not errored:
                for name, _ in stores:
                    bids[name].append(results[name])
        elif op == "switch":
            i = rng.randint(0, len(bids[stores[0][0]]) - 1)
            errored = set()
            for name, store in stores:
                try:
                    await store.switch_branch(sid, bids[name][i])
                except ValueError:
                    errored.add(name)
            # switch on a not-yet-created session must raise on all or none
            assert errored == set() or errored == {n for n, _ in stores}, (
                f"switch consistency seed={seed} step={step}: errored={errored}"
            )
        else:  # truncate the active branch
            n = rng.randint(0, 8)
            for _, store in stores:
                await store.truncate_after(sid, n)

        await _assert_agree(stores, bids)


async def test_deep_linear_chain_agrees_across_backends(
    stores: list[tuple[str, StateStore]],
) -> None:
    """A 40-deep LINEAR fork chain reads identically on all three backends —
    active read, every branch read, and every head_sequence (TD-002).

    Depth 40 is deliberate: this case pins cross-backend *agreement* on a
    lineage shape the randomized sequences barely produce, not depth itself.
    Depth past Python's recursion limit is pinned in ``test_deep_lineage.py``,
    which the O(depth**2) public build loop used here could not afford.
    """
    sid = "s"
    bids: dict[str, list[str]] = {name: [TRUNK_BRANCH_ID] for name, _ in stores}
    for _, store in stores:
        await store.append(sid, _M("seed"))
    for _ in range(40):
        for name, store in stores:
            new_id = await store.fork(sid, 1)
            await store.switch_branch(sid, new_id)
            bids[name].append(new_id)

    await _assert_agree(stores, bids)


# ---------------------------------------------------------------------------
# Explicit regressions (pre-release review findings)
# ---------------------------------------------------------------------------


async def test_regression_inherited_prefix_after_ancestor_truncate(
    stores: list[tuple[str, StateStore]],
) -> None:
    """A fork forked ABOVE an ancestor's later-truncated length must still
    inherit by POSITION (SQL previously dropped the message by sequence)."""
    sid = "s"
    for name, store in stores:
        await store.append(sid, _M("a"))
        await store.append(sid, _M("b"))
        b1 = await store.fork(sid, 2)
        await store.truncate_after(sid, 1)  # active=trunk -> [a]
        await store.switch_branch(sid, b1)
        await store.append(sid, _M("p"))
        b2 = await store.fork(sid, 2)
        assert _texts(await store.get_messages(sid, branch_id=b2)) == ["a", "p"], name


async def test_regression_head_after_ancestor_truncate(
    stores: list[tuple[str, StateStore]],
) -> None:
    """After truncating an ancestor below a child's fork point, the child's
    head_sequence and fork bounds must reflect the SHORTENED materialized
    length (SQL/Redis previously used the stale anchor + own count)."""
    sid = "s"
    for name, store in stores:
        for c in ("a", "b", "c"):
            await store.append(sid, _M(c))
        b = await store.fork(sid, 3)
        await store.switch_branch(sid, b)
        await store.append(sid, _M("d"))
        await store.truncate_after(sid, 1, branch_id=TRUNK_BRANCH_ID)  # truncate ANCESTOR
        assert _texts(await store.get_messages(sid, branch_id=b)) == ["a", "d"], name
        head = next(x.head_sequence for x in await store.list_branches(sid) if x.branch_id == b)
        assert head == 2, f"{name} head_sequence={head}"
        with pytest.raises(ValueError):
            await store.fork(sid, 4)  # 4 > materialized head 2 -> must reject


async def test_regression_truncate_trunk_only_to_empty_keeps_session(
    stores: list[tuple[str, StateStore]],
) -> None:
    """Truncating a trunk-only session's only branch to empty must leave the
    session existing (Redis previously deleted it with the trunk list)."""
    sid = "s"
    for name, store in stores:
        await store.append(sid, _M("a"))
        await store.append(sid, _M("b"))
        await store.truncate_after(sid, 0)
        assert await store.get_messages(sid) == [], name
        branches = await store.list_branches(sid)
        assert len(branches) == 1 and branches[0].branch_id == TRUNK_BRANCH_ID, name
        assert branches[0].head_sequence == 0, name
