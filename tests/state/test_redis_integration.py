"""Integration tests for :class:`RedisStateStore` against a real Redis.

Skipped at collection time unless ``REDIS_TEST_URL`` is set in the
environment. The URL must point at a Redis instance the test process can
both read from and write to; the suite scopes its keys under a dedicated
``fifty_agent_sdk:test:`` prefix and deletes every session it touches in
fixture teardown so it does not disturb co-tenant data.

These tests pin behaviours that ``fakeredis`` cannot fully validate:

* Round-trip against a real Redis server.
* A real ``EXPIRE`` is actually set (``TTL`` returns a positive value).
* A short-TTL key actually disappears once the TTL elapses (the one
  deliberately slow test — it sleeps for the TTL window).
* ``DEL`` against a real server removes the key.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest

REDIS_TEST_URL = os.environ.get("REDIS_TEST_URL")

if not REDIS_TEST_URL:
    pytest.skip(
        "REDIS_TEST_URL not set — skipping Redis integration tests",
        allow_module_level=True,
    )

# Imports deferred to below the skip so a missing redis-py package does
# not break collection when the marker is inactive.
import pytest_asyncio  # noqa: E402

from fifty_agent_sdk import ChatMessage  # noqa: E402
from fifty_agent_sdk.state.redis import RedisStateStore  # noqa: E402

pytestmark = pytest.mark.redis

_TEST_KEY_PREFIX = "fifty_agent_sdk:test:"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store() -> AsyncIterator[RedisStateStore]:
    """A real :class:`RedisStateStore` scoped under a dedicated test prefix.

    Teardown deletes the session ids exercised by the suite and then
    releases the connection pool, so the suite leaves the target Redis
    instance clean even when it shares a database with other data.
    """
    assert REDIS_TEST_URL is not None  # narrowed by the module-level skip
    s = RedisStateStore(REDIS_TEST_URL, key_prefix=_TEST_KEY_PREFIX, ttl_seconds=3600)
    try:
        yield s
    finally:
        for session_id in ("s1", "s2", "expiring", "branched"):
            await s.delete(session_id)
        await s.aclose()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def test_redis_round_trip(store: RedisStateStore) -> None:
    """Append + get round-trips identically to the fakeredis implementation."""
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b", tool_call_id="t1"),
        ChatMessage(role="tool", content="r", name="search", tool_call_id="t1"),
    ]
    for m in msgs:
        await store.append("s1", m)

    got = await store.get_messages("s1")
    assert got == msgs


async def test_redis_ttl_is_set(store: RedisStateStore) -> None:
    """An append with ``ttl_seconds`` set leaves a positive TTL on the key."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    ttl = await store._client.ttl(f"{_TEST_KEY_PREFIX}s1")
    assert 0 < ttl <= 3600


async def test_redis_branch_mutation_synchronizes_all_session_pttls(
    store: RedisStateStore,
) -> None:
    """BR-015 real Redis applies one transaction's TTL to every session key."""
    await store.append("branched", ChatMessage(role="user", content="a"))
    branch = await store.fork("branched", from_sequence=1)
    await store.switch_branch("branched", branch)
    await store.append("branched", ChatMessage(role="user", content="b"))

    keys = await store._client.keys(f"{_TEST_KEY_PREFIX}branched*")
    for key in keys:
        await store._client.pexpire(key, 10_000)
    await store.switch_branch("branched", "trunk")

    pttls = [int(await store._client.pttl(key)) for key in keys]
    assert min(pttls) > 3_500_000
    assert max(pttls) - min(pttls) <= 100


async def test_redis_delete_removes_key(store: RedisStateStore) -> None:
    """``delete`` removes the session key from the real server."""
    await store.append("s2", ChatMessage(role="user", content="a"))
    assert await store.get_messages("s2") != []
    await store.delete("s2")
    assert await store.get_messages("s2") == []


async def test_redis_no_ttl_when_ttl_seconds_is_none() -> None:
    """With ``ttl_seconds=None`` a real Redis key exists with no expiry (TTL == -1).

    The shared :func:`store` fixture pins ``ttl_seconds=3600``; this test
    builds its own no-expiry store so the ``ttl_seconds=None`` path is
    exercised against a real server. Mirrors the fakeredis-covered
    ``test_no_ttl_when_ttl_seconds_is_none`` in ``test_redis.py``.
    """
    assert REDIS_TEST_URL is not None  # narrowed by the module-level skip
    s = RedisStateStore(REDIS_TEST_URL, key_prefix=_TEST_KEY_PREFIX, ttl_seconds=None)
    try:
        await s.append("s1", ChatMessage(role="user", content="a"))
        # Redis returns -1 for a key that exists but has no associated expiry.
        ttl = await s._client.ttl(f"{_TEST_KEY_PREFIX}s1")
        assert ttl == -1
    finally:
        await s.delete("s1")
        await s.aclose()


async def test_redis_short_ttl_key_expires() -> None:
    """A short-TTL session disappears once the TTL window elapses.

    This is the one deliberately slow test in the suite — it sleeps just
    over the TTL window to prove real Redis eviction actually fires
    (something fakeredis cannot validate without simulated time).
    """
    assert REDIS_TEST_URL is not None
    s = RedisStateStore(REDIS_TEST_URL, key_prefix=_TEST_KEY_PREFIX, ttl_seconds=1)
    try:
        await s.append("expiring", ChatMessage(role="user", content="a"))
        assert await s.get_messages("expiring") != []
        # Sleep just past the 1s TTL so Redis evicts the key.
        await asyncio.sleep(1.5)
        assert await s.get_messages("expiring") == []
    finally:
        await s.delete("expiring")
        await s.aclose()
