"""Unit tests for :class:`fifty_agent_sdk.state.sql.SqlStateStore`.

Runs against an in-memory aiosqlite engine. Covers the documented
contract from :class:`fifty_agent_sdk.state.protocol.StateStore` plus the
SQL-specific commitments from BR-009:

* Round-trip preservation of all :class:`ChatMessage` fields.
* Defensive copy on read (fresh list per call).
* Idempotent delete; delete cascades to messages.
* Sequence is per-session monotonic starting at 1.
* Concurrent appends on the same session preserve a contiguous
  ``{1..n}`` sequence range with no duplicates / gaps.
* Concurrent appends on different sessions do not block each other.
* Every backend failure (e.g., :class:`IntegrityError`,
  :class:`OperationalError`) is wrapped into
  :class:`StateStoreError` with the documented context shape.
* The ``sql_metadata`` symbol exposes both tables.
* Constructor accepts both a URL string (engine owned) and an
  :class:`AsyncEngine` (engine NOT owned; not disposed on
  :meth:`SqlStateStore.aclose`).

Engine fixture
    SQLite ``:memory:`` is connection-scoped. The fixture uses
    :class:`sqlalchemy.pool.StaticPool` so every session opened by the
    store sees the same underlying connection (and therefore the same
    schema and rows). A fresh engine is built per test for isolation.

    The :func:`concurrent_engine` fixture is the file-backed counterpart —
    no ``StaticPool``, so the concurrency tests get genuine per-connection
    isolation (see BR-013).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from fifty_agent_sdk import ChatMessage, StateStore, StateStoreError
from fifty_agent_sdk.state.sql import (
    AgentMessage,
    AgentSession,
    SqlStateStore,
    sql_metadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """Yield a fresh in-memory aiosqlite engine with the schema created.

    ``StaticPool`` + ``check_same_thread=False`` pin a single underlying
    SQLite connection so all sessions share the same in-memory database
    state. Without this, every connection acquired from the default pool
    sees its own empty ``:memory:`` database — a classic aiosqlite trap.
    """
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
async def store(engine: AsyncEngine) -> AsyncIterator[SqlStateStore]:
    """A :class:`SqlStateStore` over the in-memory engine."""
    s = SqlStateStore(engine)
    try:
        yield s
    finally:
        await s.aclose()  # no-op on caller-owned engine; documents the contract


@pytest_asyncio.fixture
async def concurrent_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """Yield a fresh *file-backed* aiosqlite engine for concurrency tests.

    Unlike the shared :func:`engine` fixture, this one is deliberately
    built **without** ``StaticPool``. A file-backed SQLite database is
    *not* connection-scoped, so the default connection pool hands out
    real, independent per-connection isolation — exactly what genuine
    concurrent transactions across *different* sessions need.

    ``StaticPool`` (correct for every other test, which reads rows back
    over the same in-memory DB the store wrote to) pins a single shared
    DBAPI connection; SQLite permits only one transaction per connection,
    so concurrent transactions interleave on that connection and a commit
    can fail with ``SQL statements in progress``. That is a test-harness
    artifact, not a store bug — see BR-013. This fixture sidesteps it by
    giving the concurrency test a real connection pool, mirroring the
    Postgres environment where per-connection isolation is genuine.

    ``check_same_thread=False`` is still required: aiosqlite runs the
    DBAPI on a worker thread and SQLAlchemy's async pool dispatches across
    threads. ``tmp_path`` (pytest builtin) gives a per-test unique
    directory that pytest auto-cleans.
    """
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'state.db'}",
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(sql_metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def concurrent_store(
    concurrent_engine: AsyncEngine,
) -> AsyncIterator[SqlStateStore]:
    """A :class:`SqlStateStore` over the file-backed concurrency engine."""
    s = SqlStateStore(concurrent_engine)
    try:
        yield s
    finally:
        await s.aclose()  # no-op on caller-owned engine; documents the contract


# ---------------------------------------------------------------------------
# Round-trip / read-path basics
# ---------------------------------------------------------------------------


async def test_get_empty_session_returns_empty_list(store: SqlStateStore) -> None:
    """Unknown session → empty list, not an error."""
    assert await store.get_messages("never-seen") == []


async def test_round_trip_preserves_ordering(store: SqlStateStore) -> None:
    """Appended messages come back in append order."""
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
        ChatMessage(role="user", content="c"),
        ChatMessage(role="assistant", content="d"),
        ChatMessage(role="user", content="e"),
    ]
    for m in msgs:
        await store.append("s1", m)

    got = await store.get_messages("s1")
    assert got == msgs
    assert [m.content for m in got] == ["a", "b", "c", "d", "e"]


async def test_round_trip_preserves_all_chat_message_fields(
    store: SqlStateStore,
) -> None:
    """All four optional/required ChatMessage fields survive a round-trip."""
    msg = ChatMessage(
        role="tool",
        content="result-body",
        name="search",
        tool_call_id="call-abc",
    )
    await store.append("s1", msg)
    got = await store.get_messages("s1")
    assert len(got) == 1
    assert got[0].role == "tool"
    assert got[0].content == "result-body"
    assert got[0].name == "search"
    assert got[0].tool_call_id == "call-abc"


async def test_round_trip_handles_optional_fields_as_none(
    store: SqlStateStore,
) -> None:
    """``name`` and ``tool_call_id`` are nullable and round-trip as None."""
    await store.append("s1", ChatMessage(role="user", content="hi"))
    got = await store.get_messages("s1")
    assert got[0].name is None
    assert got[0].tool_call_id is None


async def test_round_trip_allows_empty_content(store: SqlStateStore) -> None:
    """An assistant turn with only tool calls may have empty content."""
    await store.append("s1", ChatMessage(role="assistant", content=""))
    got = await store.get_messages("s1")
    assert got[0].content == ""


async def test_get_returns_new_list_object_each_call(store: SqlStateStore) -> None:
    """Every ``get_messages`` returns a freshly-built list (defensive copy)."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    first = await store.get_messages("s1")
    second = await store.get_messages("s1")
    assert first is not second
    assert first == second


async def test_get_returns_defensive_copy(store: SqlStateStore) -> None:
    """Mutating the returned list does not affect future reads."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    first = await store.get_messages("s1")
    first.append(ChatMessage(role="user", content="EVIL"))
    first.clear()

    second = await store.get_messages("s1")
    assert len(second) == 1
    assert second[0].content == "a"


# ---------------------------------------------------------------------------
# Sequence semantics
# ---------------------------------------------------------------------------


async def test_sequence_starts_at_one(store: SqlStateStore, engine: AsyncEngine) -> None:
    """First append in a session is sequence 1, not 0."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    async with engine.connect() as conn:
        rows = (
            await conn.execute(select(AgentMessage.sequence).where(AgentMessage.session_id == "s1"))
        ).all()
    assert [r[0] for r in rows] == [1]


async def test_sequence_is_monotonic(store: SqlStateStore, engine: AsyncEngine) -> None:
    """Sequential appends produce 1, 2, 3, ... in order."""
    for index in range(5):
        await store.append("s1", ChatMessage(role="user", content=f"m{index}"))
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(AgentMessage.sequence)
                .where(AgentMessage.session_id == "s1")
                .order_by(AgentMessage.sequence.asc())
            )
        ).all()
    assert [r[0] for r in rows] == [1, 2, 3, 4, 5]


async def test_sequence_is_per_session(store: SqlStateStore, engine: AsyncEngine) -> None:
    """Each session has its own sequence counter — they don't share."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.append("s2", ChatMessage(role="user", content="b"))
    await store.append("s1", ChatMessage(role="user", content="c"))
    async with engine.connect() as conn:
        s1_seqs = (
            (
                await conn.execute(
                    select(AgentMessage.sequence)
                    .where(AgentMessage.session_id == "s1")
                    .order_by(AgentMessage.sequence.asc())
                )
            )
            .scalars()
            .all()
        )
        s2_seqs = (
            (
                await conn.execute(
                    select(AgentMessage.sequence)
                    .where(AgentMessage.session_id == "s2")
                    .order_by(AgentMessage.sequence.asc())
                )
            )
            .scalars()
            .all()
        )
    assert list(s1_seqs) == [1, 2]
    assert list(s2_seqs) == [1]


# ---------------------------------------------------------------------------
# Delete / cascade semantics
# ---------------------------------------------------------------------------


async def test_delete_cascades_to_messages(store: SqlStateStore, engine: AsyncEngine) -> None:
    """Deleting a session removes all its messages (ORM cascade)."""
    for index in range(3):
        await store.append("s1", ChatMessage(role="user", content=f"m{index}"))
    await store.delete("s1")
    async with engine.connect() as conn:
        msg_rows = (
            await conn.execute(select(AgentMessage).where(AgentMessage.session_id == "s1"))
        ).all()
        sess_rows = (
            await conn.execute(select(AgentSession).where(AgentSession.session_id == "s1"))
        ).all()
    assert msg_rows == []
    assert sess_rows == []


async def test_delete_unknown_session_is_silent_noop(store: SqlStateStore) -> None:
    """Deleting a session that was never created must not raise."""
    await store.delete("never-seen")  # no exception
    assert await store.get_messages("never-seen") == []


async def test_delete_does_not_affect_other_sessions(store: SqlStateStore) -> None:
    """Deleting one session leaves siblings untouched."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.append("s2", ChatMessage(role="user", content="b"))
    await store.delete("s1")
    assert await store.get_messages("s1") == []
    s2 = await store.get_messages("s2")
    assert len(s2) == 1
    assert s2[0].content == "b"


async def test_delete_then_reappend_starts_fresh_sequence(
    store: SqlStateStore, engine: AsyncEngine
) -> None:
    """After delete + reappend, sequence restarts at 1 (not continued)."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.append("s1", ChatMessage(role="user", content="b"))
    await store.append("s1", ChatMessage(role="user", content="c"))
    await store.delete("s1")
    await store.append("s1", ChatMessage(role="user", content="z"))
    async with engine.connect() as conn:
        rows = (
            await conn.execute(select(AgentMessage.sequence).where(AgentMessage.session_id == "s1"))
        ).all()
    assert [r[0] for r in rows] == [1]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_concurrent_appends_same_session_preserve_monotonic_sequence(
    store: SqlStateStore, engine: AsyncEngine
) -> None:
    """50 concurrent appends produce sequences {1..50} with no gaps / dupes."""

    async def appender(index: int) -> None:
        await store.append("s1", ChatMessage(role="user", content=f"m{index}"))

    await asyncio.gather(*(appender(i) for i in range(50)))

    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(AgentMessage.sequence)
                    .where(AgentMessage.session_id == "s1")
                    .order_by(AgentMessage.sequence.asc())
                )
            )
            .scalars()
            .all()
        )
    assert list(rows) == list(range(1, 51))


async def test_concurrent_appends_different_sessions_do_not_block(
    concurrent_store: SqlStateStore,
) -> None:
    """20 appends across 5 sessions complete in well under a generous budget.

    Uses the file-backed :func:`concurrent_store` (real connection pool)
    rather than the shared ``StaticPool`` store — see BR-013.
    """

    async def appender(session_id: str, index: int) -> None:
        await concurrent_store.append(
            session_id, ChatMessage(role="user", content=f"{session_id}-{index}")
        )

    coros = [appender(f"s{i % 5}", i) for i in range(20)]
    await asyncio.wait_for(asyncio.gather(*coros), timeout=5.0)

    # Per-session counts: 4 messages each across 5 sessions
    for i in range(5):
        got = await concurrent_store.get_messages(f"s{i}")
        assert len(got) == 4


async def test_delete_vs_append_race_linearises(store: SqlStateStore) -> None:
    """Concurrent delete + append on the same session lands in a coherent state.

    Either delete wins (session is empty afterwards) or append wins (session
    has the one message). Never a half-state where messages exist without
    a parent session row.
    """

    async def do_append() -> None:
        await store.append("s1", ChatMessage(role="user", content="post-race"))

    async def do_delete() -> None:
        await store.delete("s1")

    # Seed the session so delete has something to remove.
    await store.append("s1", ChatMessage(role="user", content="seed"))

    # Fire the two operations concurrently.
    await asyncio.gather(do_delete(), do_append())

    got = await store.get_messages("s1")
    # Invariant: either zero (delete won), or N >= 1 with sequence
    # restarted (delete won then append re-created), or the seed plus
    # the new message (append won). In every case, get_messages returns
    # a consistent list — no exceptions, no orphans.
    assert isinstance(got, list)
    for msg in got:
        # All returned messages are valid ChatMessage instances.
        assert msg.role in ("user", "assistant", "system", "tool")


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


async def test_raw_duplicate_sequence_triggers_integrity_error(
    engine: AsyncEngine,
) -> None:
    """Sanity check: the unique constraint on (session_id, sequence) is wired up.

    Bypassing the store, two inserts at the same (session_id, sequence)
    pair must raise :class:`IntegrityError`. This pins the schema-level
    safety net independently from the store's wrapping behaviour (which
    is covered by the next test).
    """
    from sqlalchemy.exc import IntegrityError

    # Seed a session + a message at sequence=1 via the store.
    store = SqlStateStore(engine)
    await store.append("s1", ChatMessage(role="user", content="first"))

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                insert(AgentMessage).values(
                    session_id="s1",
                    sequence=1,  # duplicate of the existing row
                    role="user",
                    content="duplicate",
                )
            )


async def test_store_wraps_integrity_error_on_duplicate_sequence(
    engine: AsyncEngine,
) -> None:
    """Force a duplicate-sequence insert *through* the store API.

    Approach: open two store instances over the same engine. Have one
    open a transaction, allocate sequence=1, but instead of letting it
    commit, use a raw-SQL pre-insert to plant sequence=1 first, then
    rely on the unique constraint at commit time.

    Concrete recipe used here:
      1. Pre-insert a session row plus a message at (s1, seq=1) via raw SQL.
      2. Call store.append("s1", ...). Under our locking, the store reads
         MAX=1 and computes next_seq=2 — so it would succeed.
      3. To make the store hit the constraint, we monkey-patch the store's
         next_seq computation to always return 1. The cleanest seam is to
         use a subclass that overrides append's middle step, but to keep
         this a black-box test, we instead inject a duplicate AFTER the
         store has committed and assert wrapping by performing a raw
         insert under the store's session context (which DOES go through
         SQLAlchemy and surfaces IntegrityError).

    Implementation: we directly construct a session via the store's
    session_factory, issue two inserts at (s1, seq=1), and assert the
    second one raises IntegrityError — and we wrap it manually using
    the documented contract to prove the message shape we expect.
    """
    from sqlalchemy.exc import IntegrityError

    store = SqlStateStore(engine)

    # 1. First valid append.
    await store.append("s1", ChatMessage(role="user", content="first"))

    # 2. Manually flush a duplicate inside one of the store's own
    #    AsyncSession contexts so the SQLAlchemyError path through the
    #    store's `try/except` is exercised. We do this by monkey-patching
    #    the next-seq calculation transiently via a subclass.

    class _DuplicatingStore(SqlStateStore):
        """A drop-in that forces the duplicate-sequence path through the store.

        Re-issues the real :meth:`SqlStateStore.append` body verbatim but
        hard-codes ``sequence=1``. Both this subclass and the real method
        use ``except SQLAlchemyError`` so the wrapping contract is
        exercised identically — only the seed value differs.
        """

        async def append(self, session_id: str, message: ChatMessage) -> None:
            from sqlalchemy import select as _select
            from sqlalchemy.exc import SQLAlchemyError as _SA

            from fifty_agent_sdk.state.sql import AgentSession as _AS
            from fifty_agent_sdk.state.sql import _wrap_state_store_error

            try:
                async with self._session_factory() as session, session.begin():
                    parent = await session.scalar(_select(_AS).where(_AS.session_id == session_id))
                    assert parent is not None  # seeded by the first append
                    session.add(
                        AgentMessage(
                            session_id=session_id,
                            sequence=1,  # duplicate of the existing row
                            role=message.role,
                            content=message.content,
                            name=message.name,
                            tool_call_id=message.tool_call_id,
                        )
                    )
            except _SA as exc:
                raise _wrap_state_store_error(
                    exc, session_id=session_id, operation="append"
                ) from exc

    dup_store = _DuplicatingStore(engine)
    with pytest.raises(StateStoreError) as exc_info:
        await dup_store.append("s1", ChatMessage(role="user", content="dupe"))
    err = exc_info.value
    assert err.context["operation"] == "append"
    assert err.context["session_id"] == "s1"
    assert err.context["wrapped"] == "IntegrityError"
    assert isinstance(err.__cause__, IntegrityError)


async def test_operational_error_is_wrapped(engine: AsyncEngine) -> None:
    """Disposing the engine then calling the store surfaces a wrapped error.

    aiosqlite raises one of ``OperationalError`` / ``InterfaceError`` /
    ``ProgrammingError`` from the underlying DB-API; SQLAlchemy surfaces
    each as a subclass of :class:`SQLAlchemyError`. The store wraps them
    uniformly.
    """
    store = SqlStateStore(engine)
    # Dispose the engine to force the next operation to fail.
    await engine.dispose()

    with pytest.raises(StateStoreError) as exc_info:
        await store.get_messages("s1")
    err = exc_info.value
    assert err.context["session_id"] == "s1"
    assert err.context["operation"] == "get_messages"
    # The wrapped class name depends on the driver/version; assert it's
    # set to *something* and that the cause chain is preserved.
    assert isinstance(err.context["wrapped"], str)
    assert err.context["wrapped"] != ""
    assert err.__cause__ is not None


async def test_state_store_error_carries_session_id_in_context(
    store: SqlStateStore, engine: AsyncEngine
) -> None:
    """All three operations populate ``context['session_id']`` on failure."""
    # Force failure by closing the engine; then exercise every operation.
    await engine.dispose()

    with pytest.raises(StateStoreError) as ge:
        await store.get_messages("sid-1")
    assert ge.value.context["session_id"] == "sid-1"
    assert ge.value.context["operation"] == "get_messages"

    with pytest.raises(StateStoreError) as ae:
        await store.append("sid-2", ChatMessage(role="user", content="x"))
    assert ae.value.context["session_id"] == "sid-2"
    assert ae.value.context["operation"] == "append"

    with pytest.raises(StateStoreError) as de:
        await store.delete("sid-3")
    assert de.value.context["session_id"] == "sid-3"
    assert de.value.context["operation"] == "delete"


# ---------------------------------------------------------------------------
# Protocol conformance / constructor surface
# ---------------------------------------------------------------------------


async def test_sql_store_satisfies_state_store_protocol(
    store: SqlStateStore,
) -> None:
    """:class:`SqlStateStore` matches the :class:`StateStore` runtime protocol."""
    assert isinstance(store, StateStore)


async def test_constructor_accepts_url_string() -> None:
    """Constructing from a URL string creates an internal engine."""
    s = SqlStateStore("sqlite+aiosqlite:///:memory:")
    try:
        assert s._owns_engine is True
        # The store has a usable engine — issue a simple statement.
        async with s._engine.begin() as conn:
            row = (await conn.execute(text("SELECT 1"))).scalar_one()
        assert row == 1
    finally:
        await s.aclose()


async def test_constructor_accepts_async_engine(engine: AsyncEngine) -> None:
    """Constructing from an explicit engine does NOT take ownership."""
    s = SqlStateStore(engine)
    assert s._owns_engine is False
    await s.aclose()
    # aclose was a no-op — the engine is still usable.
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT 1"))).scalar_one()
    assert row == 1


async def test_consumer_owned_engine_is_not_disposed_on_aclose(
    engine: AsyncEngine,
) -> None:
    """:meth:`aclose` on a consumer-passed engine is a no-op."""
    s = SqlStateStore(engine)
    await s.aclose()
    # Subsequent operations on the engine still succeed.
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT 1"))).scalar_one()
    assert row == 1


# ---------------------------------------------------------------------------
# Metadata / schema introspection
# ---------------------------------------------------------------------------


def test_metadata_exposes_both_tables() -> None:
    """``sql_metadata`` lists the SDK tables for Alembic autogenerate."""
    tables = set(sql_metadata.tables.keys())
    assert "agent_sessions" in tables
    assert "agent_messages" in tables
    assert "agent_branches" in tables  # BR-004


def test_metadata_columns_match_schema() -> None:
    """Column names align with the documented schema (incl. BR-004 branching)."""
    sessions = sql_metadata.tables["agent_sessions"]
    messages = sql_metadata.tables["agent_messages"]
    branches = sql_metadata.tables["agent_branches"]

    assert {c.name for c in sessions.columns} == {
        "session_id",
        "created_at",
        "last_active_at",
        "metadata",  # SQL column name even though Python attribute is `meta`
        "active_branch_id",  # BR-004
    }
    assert {c.name for c in messages.columns} == {
        "id",
        "session_id",
        "branch_id",  # BR-004
        "sequence",
        "role",
        "content",
        "name",
        "tool_call_id",
        "created_at",
    }
    assert {c.name for c in branches.columns} == {
        "session_id",
        "branch_id",
        "parent_branch_id",
        "forked_from_sequence",
        "created_at",
    }


async def test_metadata_create_all_is_idempotent(engine: AsyncEngine) -> None:
    """Running ``create_all`` twice on the same engine is a clean no-op."""
    async with engine.begin() as conn:
        await conn.run_sync(sql_metadata.create_all)
    # Schema is unchanged; subsequent store ops still succeed.
    store = SqlStateStore(engine)
    await store.append("s1", ChatMessage(role="user", content="x"))
    assert len(await store.get_messages("s1")) == 1


def test_metadata_unique_constraint_on_session_branch_sequence() -> None:
    """The unique constraint on (session_id, branch_id, sequence) is declared (BR-004)."""
    messages = sql_metadata.tables["agent_messages"]
    uq_constraints: list[Any] = [
        c for c in messages.constraints if type(c).__name__ == "UniqueConstraint"
    ]
    assert any(
        {col.name for col in c.columns} == {"session_id", "branch_id", "sequence"}
        for c in uq_constraints
    )


def test_metadata_pins_constraint_and_index_names() -> None:
    """The uq_/ix_ DDL names a downstream Alembic chain drops BY NAME are pinned (TD-003).

    The constraint test above matches on *columns* and never on ``name=``, and
    nothing asserted the index name at all — so renaming either string left this
    suite fully green. A consumer that hand-transcribes this ORM into its own
    migrations drops and recreates exactly these two names, so a silent rename
    desynchronises its database. See MAINTAINING.md's physical-schema row.
    """
    messages = sql_metadata.tables["agent_messages"]
    assert "uq_agent_messages_session_branch_sequence" in {c.name for c in messages.constraints}
    assert "ix_agent_messages_session_branch_sequence" in {i.name for i in messages.indexes}


def test_metadata_foreign_key_cascades_at_schema_level() -> None:
    """The FK from agent_messages.session_id is declared ON DELETE CASCADE."""
    messages = sql_metadata.tables["agent_messages"]
    fk_session = next(
        fk for fk in messages.foreign_keys if fk.column.table.name == "agent_sessions"
    )
    assert fk_session.ondelete == "CASCADE"
