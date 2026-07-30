"""SQLAlchemy 2.0 asyncio implementation of :class:`StateStore`.

:class:`SqlStateStore` is the SDK's durable conversation-state backend
when callers need persistence across process restarts. It targets the
SQLAlchemy 2.0 asyncio API and is tested against SQLite (via
``aiosqlite``) by default and Postgres (via ``asyncpg`` or ``psycopg``)
behind a marker.

Extras requirement
    This module requires the optional ``sql`` extra::

        pip install 'fifty-agent-sdk[sql]'

    Importing :mod:`fifty_agent_sdk` itself does NOT pull SQLAlchemy. The SQL
    surface is re-exported lazily from :mod:`fifty_agent_sdk.state` and the
    package root via module-level ``__getattr__``; first access triggers
    this module's import, and a missing dependency surfaces as a clear
    :class:`ImportError` referencing the extras line above.

Schema lifecycle
    The SDK ships the table definitions as declarative ORM models but
    does NOT own migrations. Consumers register :data:`sql_metadata`
    with their own Alembic environment::

        # alembic/env.py
        from fifty_agent_sdk import sql_metadata
        target_metadata = sql_metadata

    Then::

        alembic revision --autogenerate -m "fifty_agent_sdk schema"

    Review the generated revision before applying — the SDK takes no
    position on naming conventions, schemas, or tablespaces in your
    deployment.

Dialect parity
    The schema is engineered to work on SQLite and Postgres without
    conditional code paths in the store itself. Dialect drift is
    confined to:

    * :class:`AgentSession.meta` — ``JSON().with_variant(JSONB(),
      "postgresql")`` so the column becomes ``JSONB`` on Postgres
      (indexable, queryable with ``->>``) and ``TEXT``-with-JSON on
      SQLite.
    * Timestamp columns are ``DateTime(timezone=True)`` so Postgres
      stores ``timestamptz``; SQLAlchemy normalises SQLite's ISO-string
      representation on read.

    SQLite-only note: ``ON DELETE CASCADE`` at the schema level
    requires ``PRAGMA foreign_keys = ON`` per connection. The store's
    :meth:`delete` uses the ORM cascade path which fires regardless of
    that pragma, so the SDK's own delete is correct on either dialect.
    Consumers issuing raw-SQL ``DELETE FROM agent_sessions`` against a
    SQLite engine must enable the pragma themselves.

Sequence semantics
    :meth:`append` allocates a per-(session, branch) ``sequence`` value
    (``COALESCE(MAX(sequence), anchor) + 1`` for the active branch) inside a
    transaction that acquires a ``SELECT ... FOR UPDATE`` lock on the parent
    ``agent_sessions`` row. On Postgres this serialises concurrent appenders
    for the same session; on SQLite ``with_for_update`` is a documented no-op
    and the database's global writer serialisation provides the same
    guarantee. A ``UniqueConstraint(session_id, branch_id, sequence)`` is the
    schema-level safety net — a hypothetical race that reached the INSERT step
    would surface as :class:`sqlalchemy.exc.IntegrityError`, wrapped into
    :class:`fifty_agent_sdk.errors.StateStoreError` with
    ``context["wrapped"] == "IntegrityError"`` for deterministic caller-side
    handling.

Branching (BR-004) & migration
    Branching adds the ``agent_branches`` table, ``agent_messages.branch_id``
    (``server_default 'trunk'``), and ``agent_sessions.active_branch_id``
    (``server_default 'trunk'``). A message's ``sequence`` is its position in
    the OWNING branch's *materialized* history; :meth:`get_messages` walks the
    lineage and reads it as a union of disjoint, increasing sequence ranges
    (one ``ORDER BY sequence`` yields append order). The schema is **additive
    and zero-loss**: when consumers run ``alembic revision --autogenerate``
    against :data:`sql_metadata`, existing ``agent_messages`` rows take
    ``branch_id = 'trunk'`` and sessions take ``active_branch_id = 'trunk'``
    via the server defaults. The autogenerate diff will NOT backfill the
    ``agent_branches`` table for pre-existing sessions — but no backfill is
    required: the store synthesizes an implicit trunk on read for any session
    that has no branch row, and materializes the trunk row lazily on the next
    :meth:`append` / :meth:`fork`. Review the generated revision before
    applying.

Error wrapping contract
    Every public method wraps :class:`sqlalchemy.exc.SQLAlchemyError`
    (the SQLAlchemy base class) into
    :class:`fifty_agent_sdk.errors.StateStoreError` with:

    * ``message``: ``"SqlStateStore.<operation> failed for session_id=<id>"``
    * ``context["session_id"]``: the input session id (echoed for log
      correlation)
    * ``context["wrapped"]``: the underlying exception's class name
      (e.g., ``"IntegrityError"``, ``"OperationalError"``) — read by
      the Runner's ``runner.persist_failed`` ERROR log per TD-004
    * ``context["operation"]``: ``"get_messages"``, ``"append"``, or
      ``"delete"``
    * ``__cause__``: the original exception, via ``raise ... from exc``

    :class:`asyncio.CancelledError` propagates untouched. Pydantic
    ``ValidationError`` on read (which would indicate corrupt rows) is
    not wrapped either — it is a corruption signal, not a backend
    failure.

Engine ownership
    The constructor accepts EITHER an :class:`AsyncEngine` or a
    connection URL string. When a URL is provided, the store creates
    its own engine and tracks ``self._owns_engine = True``;
    :meth:`aclose` disposes it. When an externally-built engine is
    passed, the store DOES NOT dispose it on close — the caller owns
    its lifecycle.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Final

import structlog

try:
    from sqlalchemy import (
        BigInteger,
        DateTime,
        ForeignKey,
        Index,
        Integer,
        String,
        Text,
        UniqueConstraint,
        delete,
        func,
        select,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import (
        DeclarativeBase,
        Mapped,
        mapped_column,
        relationship,
    )
    from sqlalchemy.types import JSON
except ImportError as exc:  # pragma: no cover - exercised via importlib in tests
    raise ImportError(
        "fifty_agent_sdk.state.sql requires SQLAlchemy. Install with: pip install 'fifty-agent-sdk[sql]'"
    ) from exc

from fifty_agent_sdk.errors import StateStoreError
from fifty_agent_sdk.llm.types import ChatMessage
from fifty_agent_sdk.state.protocol import TRUNK_BRANCH_ID, BranchInfo

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

Successful operations log at ``DEBUG`` with the session id and a small
shape summary (rows affected, sequence assigned). Failures are NOT
logged here — the wrapped :class:`StateStoreError` carries everything
the Runner's ``runner.persist_failed`` ERROR log needs.
"""


class Base(DeclarativeBase):
    """Declarative base shared by the SDK's ORM models.

    Module-private on purpose: consumers register :data:`sql_metadata`
    (the underlying ``Base.metadata``) with their Alembic environment
    rather than subclassing this base. Keeping ``Base`` out of
    ``__all__`` prevents consumer tables from accidentally merging into
    the SDK's autogenerate run.
    """


class AgentSession(Base):
    """Per-session metadata row.

    A parent row exists for every active conversation; it is created
    lazily on the first :meth:`SqlStateStore.append` and removed by
    :meth:`SqlStateStore.delete` (which cascades to messages).

    The ``meta`` Python attribute maps to a SQL column literally named
    ``metadata``; we cannot name the attribute ``metadata`` directly
    because :class:`DeclarativeBase` reserves that name for the
    metadata registry.
    """

    __tablename__ = "agent_sessions"

    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    """Opaque caller-provided id (the SDK does no validation)."""

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Insertion timestamp (server-side default; timezone-aware)."""

    last_active_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    """Bumped on every :meth:`SqlStateStore.append`."""

    meta: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=True,
        default=None,
    )
    """Forward-compatibility column for per-session tags / annotations.

    Not exposed at the store API in BR-009. ``JSONB`` on Postgres,
    JSON-encoded ``TEXT`` on SQLite. The Python attribute is ``meta`` to
    avoid shadowing :attr:`DeclarativeBase.metadata`.
    """

    active_branch_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=TRUNK_BRANCH_ID,
    )
    """The session's active head branch (BR-004).

    :meth:`SqlStateStore.append` writes to this branch and a default
    :meth:`SqlStateStore.get_messages` reads it. Existing rows migrated from
    before BR-004 take the trunk via the server default — zero data loss.
    """

    messages: Mapped[list[AgentMessage]] = relationship(
        "AgentMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=False,
        order_by="AgentMessage.sequence",
    )
    """Per-session messages, ordered by sequence ascending.

    ``cascade="all, delete-orphan"`` makes
    :meth:`SqlStateStore.delete` cascade through the ORM regardless of
    schema-level FK enforcement (SQLite needs an explicit ``PRAGMA
    foreign_keys = ON`` for the FK-level cascade to fire). With
    ``passive_deletes=False``, SQLAlchemy issues an explicit DELETE for
    each child row from Python, which is correct on every dialect.
    """

    branches: Mapped[list[AgentBranch]] = relationship(
        "AgentBranch",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=False,
    )
    """Per-session branch rows (BR-004); cascade-deleted with the session."""


class AgentMessage(Base):
    """One persisted :class:`ChatMessage` row.

    Round-trips all four :class:`ChatMessage` fields (``role``,
    ``content``, ``name``, ``tool_call_id``) plus a per-session
    ``sequence`` and the surrogate ``id`` primary key. The
    ``sequence`` is the durable ordering key; ``id`` is only used for
    row identity.
    """

    __tablename__ = "agent_messages"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "branch_id",
            "sequence",
            name="uq_agent_messages_session_branch_sequence",
        ),
        Index(
            "ix_agent_messages_session_branch_sequence",
            "session_id",
            "branch_id",
            "sequence",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer().with_variant(BigInteger(), "postgresql"),
        primary_key=True,
        autoincrement=True,
    )
    """Surrogate row id; ``BigInteger`` on Postgres, ``Integer`` on SQLite."""

    session_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("agent_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    """Foreign key to :attr:`AgentSession.session_id`."""

    branch_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=TRUNK_BRANCH_ID,
    )
    """Which branch this message belongs to (BR-004).

    Existing rows migrated from before BR-004 take the trunk via the server
    default. The ``(session_id, branch_id, sequence)`` unique constraint is
    the database-level ordering safety net.
    """

    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    """Per-(session, branch) materialized order key (``>= 1``).

    A message's ``sequence`` is its position in the OWNING branch's
    materialized history: a branch forked at sequence ``N`` allocates its own
    messages starting at ``N + 1``. Allocated inside
    :meth:`SqlStateStore.append` as ``COALESCE(MAX(sequence), anchor) + 1``
    for the active branch under a ``SELECT ... FOR UPDATE`` lock on the
    parent session row.
    """

    role: Mapped[str] = mapped_column(String(32), nullable=False)
    """One of ``"system"`` / ``"user"`` / ``"assistant"`` / ``"tool"``."""

    content: Mapped[str] = mapped_column(Text, nullable=False)
    """Arbitrary text; an empty string is permitted (see
    :class:`ChatMessage`)."""

    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """Optional named speaker / tool name."""

    tool_call_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """Optional id echoed back on ``role="tool"`` replies."""

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Insertion timestamp (server-side default; timezone-aware)."""

    session: Mapped[AgentSession] = relationship(
        "AgentSession",
        back_populates="messages",
    )
    """Parent session (the inverse side of :attr:`AgentSession.messages`)."""


class AgentBranch(Base):
    """One branch of a session's conversation tree (BR-004).

    The trunk branch (``branch_id == "trunk"``) is created idempotently on a
    session's first :meth:`SqlStateStore.append`; :meth:`SqlStateStore.fork`
    adds a row recording its parent and the sequence it diverged at. Reads of
    a branch walk this lineage to materialize the full history.

    Pre-BR-004 sessions have no branch rows; the store synthesizes an implicit
    trunk on read, so the feature is additive with zero data loss.
    """

    __tablename__ = "agent_branches"

    session_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("agent_sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    """Foreign key to :attr:`AgentSession.session_id` (half of the PK)."""

    branch_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    """Branch identifier (``"trunk"`` for the root; an opaque id for forks)."""

    parent_branch_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """The branch this one forked from, or ``NULL`` for the trunk."""

    forked_from_sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """The parent's materialized sequence this branch diverged at, ``NULL``
    for the trunk."""

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Branch creation timestamp (server-side default; timezone-aware)."""

    session: Mapped[AgentSession] = relationship(
        "AgentSession",
        back_populates="branches",
    )
    """Parent session (inverse of :attr:`AgentSession.branches`)."""


sql_metadata = Base.metadata
"""The :class:`sqlalchemy.MetaData` object describing the SDK's schema.

Consumers register this with their Alembic environment::

    # alembic/env.py
    from fifty_agent_sdk import sql_metadata
    target_metadata = sql_metadata

The SDK does NOT ship migration scripts; consumers run
``alembic revision --autogenerate`` against this object and review the
output before applying.
"""


def _wrap_state_store_error(
    exc: SQLAlchemyError,
    *,
    session_id: str,
    operation: str,
) -> StateStoreError:
    """Build the SDK's standard wrap of a SQLAlchemy backend failure.

    Returns the :class:`StateStoreError`; the caller writes
    ``raise _wrap_state_store_error(...) from exc`` so the
    ``__cause__`` chain (the BR-009 error-wrapping contract) is
    preserved at every call site.
    """
    return StateStoreError(
        f"SqlStateStore.{operation} failed for session_id={session_id}",
        context={
            "session_id": session_id,
            "wrapped": type(exc).__name__,
            "operation": operation,
        },
    )


class SqlStateStore:
    """SQLAlchemy-backed implementation of :class:`StateStore`.

    Satisfies :class:`fifty_agent_sdk.state.protocol.StateStore` structurally
    (no explicit inheritance needed thanks to ``@runtime_checkable``).

    Concurrency model:
        Two layers of serialisation cover same-session writes:

        * **In-process:** a per-``session_id`` :class:`asyncio.Lock`
          serialises :meth:`append` and :meth:`delete` calls on the
          same store instance. Different sessions never block each
          other. Mirrors the locking shape of
          :class:`MemoryStateStore`.
        * **Cross-process / DB-level:** every transaction acquires
          ``SELECT ... FOR UPDATE`` on the parent
          :class:`AgentSession` row. On Postgres this serialises
          concurrent writers from any process; on SQLite the DB
          serialises writers globally and ``with_for_update`` is a
          documented no-op.

        Parent-row creation is idempotent via ``INSERT ... ON
        CONFLICT DO NOTHING``, so the cold-start race (two appenders
        on a never-seen ``session_id`` both attempting to create the
        parent row) cannot produce a primary-key violation.

    Engine ownership:
        * Construct with a URL string → the store creates and owns the
          engine. :meth:`aclose` disposes it.
        * Construct with an :class:`AsyncEngine` → the caller owns the
          engine. :meth:`aclose` is a no-op on the engine; the caller
          must dispose it themselves.

    Example:
        Construct from a URL (the store owns and disposes the engine)
        and wire it into an :class:`AgentRunner`::

            from fifty_agent_sdk import (
                JSON_MODE_OUTPUT_FORMAT, AgentLoop, AgentRunner,
                JsonModeParser, PromptSections, Registry, SafetyConfig,
                SqlStateStore,
            )
            from fifty_agent_sdk.llm import OpenAICompatibleClient

            state = SqlStateStore("sqlite+aiosqlite:///./agent.db")
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

        Schema must be created out-of-band (the SDK does not own
        migrations); register :data:`sql_metadata` with your Alembic
        environment first. See the module docstring's
        "Schema lifecycle" section.

    Failure mode:
        Every public method wraps :class:`sqlalchemy.exc.SQLAlchemyError`
        into :class:`fifty_agent_sdk.errors.StateStoreError` with
        ``context["wrapped"]`` carrying the underlying class name. See
        the module docstring for the full contract.
    """

    def __init__(self, engine_or_url: AsyncEngine | str) -> None:
        """Construct a :class:`SqlStateStore`.

        Args:
            engine_or_url: Either a fully-configured
                :class:`sqlalchemy.ext.asyncio.AsyncEngine` (caller-owned)
                or a connection URL string (e.g.,
                ``"sqlite+aiosqlite:///:memory:"``,
                ``"postgresql+asyncpg://..."``) — in which case the
                store creates and owns the engine.
        """
        if isinstance(engine_or_url, str):
            self._engine: AsyncEngine = create_async_engine(engine_or_url)
            self._owns_engine: bool = True
        else:
            self._engine = engine_or_url
            self._owns_engine = False
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        # Per-session asyncio locks serialise concurrent appends/deletes
        # within this process. The DB-level row lock
        # (`SELECT ... FOR UPDATE`) is still the cross-process correctness
        # mechanism on Postgres; the in-process lock is an additional
        # guarantee that mirrors :class:`MemoryStateStore` and avoids
        # relying on `with_for_update` being honoured (which it is not on
        # SQLite). Different sessions never block each other.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._lock_registry_lock: asyncio.Lock = asyncio.Lock()

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session :class:`asyncio.Lock`, creating it lazily.

        Same double-checked-locking pattern as
        :class:`MemoryStateStore` — the hot path is lock-free; only
        first-time creation takes the registry lock.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The :class:`asyncio.Lock` associated with this session.
        """
        existing = self._session_locks.get(session_id)
        if existing is not None:
            return existing
        async with self._lock_registry_lock:
            existing = self._session_locks.get(session_id)
            if existing is not None:
                return existing
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
            return lock

    async def aclose(self) -> None:
        """Dispose the underlying engine if (and only if) the store owns it.

        Safe to call multiple times. When the engine was passed in by
        the caller, this is a no-op — the caller's engine lifecycle is
        not the store's responsibility.
        """
        if self._owns_engine:
            await self._engine.dispose()

    async def _upsert_parent(self, session: AsyncSession, session_id: str) -> None:
        """Idempotently ensure an :class:`AgentSession` row exists.

        Two concurrent appends on a never-seen ``session_id`` would
        both see a NULL parent on the initial SELECT and race to insert
        — the slower one would fail the primary-key constraint. We use
        the dialect-aware ``INSERT ... ON CONFLICT DO NOTHING`` so both
        appenders converge on a single row without an exception. The
        subsequent ``SELECT ... FOR UPDATE`` in :meth:`append` then
        serialises the rest of the work.

        SQLite ``>=3.24`` and Postgres both support ``ON CONFLICT``;
        the SDK targets SQLAlchemy 2.0 which requires recent SQLite,
        so no fallback path is needed.

        Args:
            session: Active :class:`AsyncSession` (inside a transaction).
            session_id: Opaque session identifier.
        """
        dialect = session.bind.dialect.name if session.bind is not None else ""
        stmt: Any
        if dialect == "postgresql":
            stmt = pg_insert(AgentSession).values(session_id=session_id)
            stmt = stmt.on_conflict_do_nothing(index_elements=["session_id"])
        else:
            # SQLite (and any other dialect we run against in tests).
            stmt = sqlite_insert(AgentSession).values(session_id=session_id)
            stmt = stmt.on_conflict_do_nothing(index_elements=["session_id"])
        await session.execute(stmt)

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        """Return the persisted messages for ``session_id``.

        Returns a freshly-constructed list of :class:`ChatMessage`
        instances ordered by ``sequence`` ascending. An unknown session
        yields ``[]`` (not an error).

        Args:
            session_id: Opaque session identifier.
            branch_id: Which branch to read (BR-004). ``None`` reads the
                active branch; pre-M3 only the trunk exists.

        Returns:
            A list of :class:`ChatMessage` values in append order.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name.
        """
        try:
            async with self._session_factory() as session:
                branch_map, active = await self._load_branch_map(session, session_id)
                if not branch_map:
                    # Unknown session: an explicit branch request is a
                    # programmer error; the default read is an empty result.
                    if branch_id is not None:
                        raise ValueError(
                            f"branch_id={branch_id!r} does not exist for "
                            f"unknown session {session_id!r}"
                        )
                    return []
                target = branch_id if branch_id is not None else active
                if target not in branch_map:
                    raise ValueError(
                        f"branch_id={target!r} does not exist for session {session_id!r}"
                    )
                # Assemble the branch's history BY POSITION (mirrors
                # MemoryStateStore): materialize(parent)[:fork_point] + own.
                # Position-based, NOT sequence-based: truncate_after frees
                # sequence numbers that later appends reuse, so a fork point is
                # a COUNT of inherited messages, not a sequence value. One query
                # fetches all of the session's rows; assembly happens in Python.
                rows = await session.scalars(
                    select(AgentMessage)
                    .where(AgentMessage.session_id == session_id)
                    .order_by(AgentMessage.branch_id.asc(), AgentMessage.sequence.asc())
                )
                by_branch: dict[str, list[ChatMessage]] = {}
                for row in rows:
                    by_branch.setdefault(row.branch_id, []).append(
                        ChatMessage(
                            role=row.role,  # type: ignore[arg-type]
                            content=row.content,
                            name=row.name,
                            tool_call_id=row.tool_call_id,
                        )
                    )
                messages = self._materialize_positional(branch_map, by_branch, target)
                _log.debug(
                    "sql_state_store.get_messages",
                    session_id=session_id,
                    branch_id=target,
                    count=len(messages),
                )
                return messages
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="get_messages"
            ) from exc

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's **active** branch.

        Allocates a per-branch ``sequence`` value transactionally: opens a
        transaction, takes a ``SELECT ... FOR UPDATE`` lock on the parent
        :class:`AgentSession` row (creating it if absent), ensures the trunk
        branch row exists, resolves the active branch, computes
        ``COALESCE(MAX(sequence), anchor) + 1`` for that branch under the held
        lock, and inserts the new :class:`AgentMessage` row. The
        ``(session_id, branch_id, sequence)`` unique constraint is the
        database-level safety net.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name (e.g.,
                ``"IntegrityError"`` if a hypothetical race reached the
                unique-constraint check).
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                # 1. Idempotently ensure the parent row exists (ON CONFLICT DO
                #    NOTHING) so the cold-start race on a never-seen session_id
                #    cannot raise a primary-key violation.
                await self._upsert_parent(session, session_id)

                # 2. Lock the parent row. with_for_update() is a no-op on
                #    SQLite (writers serialised globally) and SELECT ... FOR
                #    UPDATE on Postgres — serialising appenders for this id.
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                assert parent is not None  # guaranteed by the upsert above
                parent.last_active_at = func.now()

                # 3. Resolve the active branch and ensure its row exists. The
                #    trunk is created lazily here on first append; forks have
                #    already created theirs in fork().
                active: str = parent.active_branch_id or TRUNK_BRANCH_ID
                await self._ensure_branch_row(session, session_id, active)

                # 4. Compute the next sequence for THIS branch under the held
                #    lock. COALESCE(MAX(sequence), anchor) + 1 starts a freshly
                #    forked branch (no own messages) at anchor + 1, continuing
                #    the parent's materialized numbering.
                anchor = await session.scalar(
                    select(AgentBranch.forked_from_sequence).where(
                        AgentBranch.session_id == session_id,
                        AgentBranch.branch_id == active,
                    )
                )
                anchor_val = anchor or 0
                result = await session.execute(
                    select(func.coalesce(func.max(AgentMessage.sequence), anchor_val) + 1).where(
                        AgentMessage.session_id == session_id,
                        AgentMessage.branch_id == active,
                    )
                )
                next_seq: int = int(result.scalar_one())

                # 5. Insert the message on the active branch.
                session.add(
                    AgentMessage(
                        session_id=session_id,
                        branch_id=active,
                        sequence=next_seq,
                        role=message.role,
                        content=message.content,
                        name=message.name,
                        tool_call_id=message.tool_call_id,
                    )
                )
                # commit on session.begin() context exit
            _log.debug(
                "sql_state_store.append",
                session_id=session_id,
                branch_id=active,
                sequence=next_seq,
            )
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="append") from exc

    async def delete(self, session_id: str) -> None:
        """Remove the session row and all its messages.

        Idempotent — deleting an unknown session is a silent no-op (no
        rows affected, no error). Cascade is handled via the ORM
        relationship (``cascade="all, delete-orphan"``) so it works on
        both SQLite (where schema-level CASCADE requires
        ``PRAGMA foreign_keys = ON``) and Postgres.

        Args:
            session_id: Opaque session identifier.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name.
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                if parent is None:
                    _log.debug(
                        "sql_state_store.delete",
                        session_id=session_id,
                        existed=False,
                    )
                    return
                # ORM-level delete fires the cascade regardless of dialect.
                await session.delete(parent)
            _log.debug(
                "sql_state_store.delete",
                session_id=session_id,
                existed=True,
            )
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="delete") from exc

    # --- Branching (BR-004) -------------------------------------------------

    @staticmethod
    def _materialize_positional(
        branch_map: dict[str, tuple[str | None, int | None]],
        by_branch: dict[str, list[ChatMessage]],
        branch_id: str,
    ) -> list[ChatMessage]:
        """Assemble a branch's materialized history by POSITION.

        The SQL mirror of :meth:`MemoryStateStore._materialize`: a branch's
        history is its parent's history truncated at the fork point (a count of
        inherited messages), followed by its own messages. ``by_branch`` maps
        each branch id to its own messages in append order.

        That definition is recursive; the computation is **iterative**
        (TD-002), so lineage depth is bounded only by memory and not by
        Python's recursion limit.
        """
        # Walk parent pointers leaf -> root (pure pointer work over
        # branch_map), then fold root -> leaf. No cycle guard, deliberately:
        # parent_branch_id is written once, in fork(), to the already-existing
        # active branch and never mutated, so the lineage is acyclic by
        # construction. Honest asymmetry (TD-002): an externally corrupted row
        # introducing a cycle used to raise RecursionError and would now spin
        # forever; that is unreachable through any public API.
        chain: list[tuple[str, int | None]] = []
        bid: str | None = branch_id
        while bid is not None:
            parent, anchor = branch_map[bid]  # KeyError on a missing branch: preserved
            chain.append((bid, anchor))
            bid = parent
        chain.reverse()  # root .. leaf

        # Base case and step mirror the recursive form literally, and stay
        # separate, so the two can be diffed line for line.
        root_id, _ = chain[0]
        # `chain` is non-empty: branch_id is never None, so the walk ran once.
        history = list(by_branch.get(root_id, []))  # base case: the trunk
        for cur_id, cur_anchor in chain[1:]:  # step
            history = history[: cur_anchor or 0] + list(by_branch.get(cur_id, []))
        return history

    @staticmethod
    def _materialized_len(
        branch_map: dict[str, tuple[str | None, int | None]],
        own_counts: dict[str, int],
        branch_id: str,
    ) -> int:
        """Return the length of ``branch_id``'s materialized history.

        ``min(anchor, len(parent_history)) + own_count`` — the length analogue
        of :meth:`_materialize_positional`. ``min`` is load-bearing: if an
        ancestor was truncated below this branch's fork point, the inherited
        portion is bounded by the ancestor's (shortened) length, matching the
        Memory reference. This is the head used for ``fork`` bounds and
        :class:`BranchInfo.head_sequence`.

        **Computed iteratively** (TD-002). The recursive form was already a
        left fold from the root — each hop depends only on its parent — so the
        fold below runs root -> leaf and every hop's ``min`` still sees the
        already-``min``-clamped ancestor length. Same expression, same
        associativity, no Python frame per lineage hop.
        """
        # Walk parent pointers leaf -> root, then fold root -> leaf. No cycle
        # guard, deliberately — see _materialize_positional for the reasoning
        # and the honest asymmetry it accepts (TD-002).
        chain: list[tuple[str, int | None]] = []
        bid: str | None = branch_id
        while bid is not None:
            parent, anchor = branch_map[bid]  # KeyError on a missing branch: preserved
            chain.append((bid, anchor))
            bid = parent
        chain.reverse()  # root .. leaf

        # Base case and step mirror the recursive form literally, and stay
        # separate, so the two can be diffed line for line.
        root_id, _ = chain[0]
        # `chain` is non-empty: branch_id is never None, so the walk ran once.
        length = own_counts.get(root_id, 0)  # base case: the trunk
        for cur_id, cur_anchor in chain[1:]:  # step
            length = min(cur_anchor or 0, length) + own_counts.get(cur_id, 0)
        return length

    async def _own_counts(self, session: AsyncSession, session_id: str) -> dict[str, int]:
        """Return ``{branch_id: own_message_count}`` for the session (one query)."""
        result = await session.execute(
            select(AgentMessage.branch_id, func.count())
            .where(AgentMessage.session_id == session_id)
            .group_by(AgentMessage.branch_id)
        )
        return {row[0]: int(row[1]) for row in result.all()}

    async def _load_branch_map(
        self, session: AsyncSession, session_id: str
    ) -> tuple[dict[str, tuple[str | None, int | None]], str]:
        """Load the branch lineage map and active head for ``session_id``.

        Returns ``({}, TRUNK_BRANCH_ID)`` for an unknown session. A pre-BR-004
        session (message rows but no branch records) synthesizes an implicit
        trunk so reads stay correct with zero migration.
        """
        parent = await session.get(AgentSession, session_id)
        if parent is None:
            return {}, TRUNK_BRANCH_ID
        rows = await session.scalars(
            select(AgentBranch).where(AgentBranch.session_id == session_id)
        )
        branch_map: dict[str, tuple[str | None, int | None]] = {
            row.branch_id: (row.parent_branch_id, row.forked_from_sequence) for row in rows
        }
        if not branch_map:
            branch_map = {TRUNK_BRANCH_ID: (None, None)}
        active: str = parent.active_branch_id or TRUNK_BRANCH_ID
        return branch_map, active

    async def _ensure_branch_row(
        self, session: AsyncSession, session_id: str, branch_id: str
    ) -> None:
        """Idempotently create the TRUNK branch row (lazy, on first write).

        Forks create their own rows in :meth:`fork`; only the trunk is
        materialized lazily. ``ON CONFLICT DO NOTHING`` keeps it race-safe on
        both dialects. A non-trunk ``branch_id`` is a no-op — its row already
        exists.
        """
        if branch_id != TRUNK_BRANCH_ID:
            return
        dialect = session.bind.dialect.name if session.bind is not None else ""
        values = {
            "session_id": session_id,
            "branch_id": TRUNK_BRANCH_ID,
            "parent_branch_id": None,
            "forked_from_sequence": None,
        }
        stmt: Any
        if dialect == "postgresql":
            stmt = pg_insert(AgentBranch).values(**values)
        else:
            stmt = sqlite_insert(AgentBranch).values(**values)
        stmt = stmt.on_conflict_do_nothing(index_elements=["session_id", "branch_id"])
        await session.execute(stmt)

    async def fork(self, session_id: str, from_sequence: int) -> str:
        """Fork the active branch at ``from_sequence`` into a new branch.

        Records a new :class:`AgentBranch` row whose parent is the active
        branch; subsequent appends to it continue the materialized sequence
        from ``from_sequence``. Does NOT change the active head.

        Raises:
            ValueError: If the session is unknown, or ``from_sequence`` is
                outside ``0..head`` of the active branch.
            fifty_agent_sdk.errors.StateStoreError: On backend failure.
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                if parent is None:
                    raise ValueError(f"cannot fork unknown session {session_id!r}")
                # Materialize the trunk row so the new branch's lineage is
                # always resolvable, even for a pre-BR-004 session.
                await self._ensure_branch_row(session, session_id, TRUNK_BRANCH_ID)
                branch_map, active = await self._load_branch_map(session, session_id)
                own_counts = await self._own_counts(session, session_id)
                head = self._materialized_len(branch_map, own_counts, active)
                if not 0 <= from_sequence <= head:
                    raise ValueError(
                        f"from_sequence={from_sequence} out of range 0..{head} "
                        f"for active branch {active!r} of session {session_id!r}"
                    )
                new_id = uuid.uuid4().hex
                session.add(
                    AgentBranch(
                        session_id=session_id,
                        branch_id=new_id,
                        parent_branch_id=active,
                        forked_from_sequence=from_sequence,
                    )
                )
            _log.debug("sql_state_store.fork", session_id=session_id, branch_id=new_id)
            return new_id
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="fork") from exc

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        """Enumerate all branches of ``session_id`` (trunk first, then by age).

        An unknown session yields ``[]``. A pre-BR-004 session reports a single
        synthesized trunk.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: On backend failure.
        """
        try:
            async with self._session_factory() as session:
                parent = await session.get(AgentSession, session_id)
                if parent is None:
                    return []
                active = parent.active_branch_id or TRUNK_BRANCH_ID
                rows = list(
                    await session.scalars(
                        select(AgentBranch).where(AgentBranch.session_id == session_id)
                    )
                )
                branch_map: dict[str, tuple[str | None, int | None]] = {
                    row.branch_id: (row.parent_branch_id, row.forked_from_sequence) for row in rows
                }
                if not branch_map:
                    branch_map = {TRUNK_BRANCH_ID: (None, None)}
                own_counts = await self._own_counts(session, session_id)
                if not rows:
                    # Pre-BR-004 session: synthesize the implicit trunk.
                    return [
                        BranchInfo(
                            branch_id=TRUNK_BRANCH_ID,
                            parent_branch_id=None,
                            forked_from_sequence=None,
                            head_sequence=self._materialized_len(
                                branch_map, own_counts, TRUNK_BRANCH_ID
                            ),
                            created_at=parent.created_at,
                            is_active=(active == TRUNK_BRANCH_ID),
                        )
                    ]
                infos = [
                    BranchInfo(
                        branch_id=row.branch_id,
                        parent_branch_id=row.parent_branch_id,
                        forked_from_sequence=row.forked_from_sequence,
                        head_sequence=self._materialized_len(branch_map, own_counts, row.branch_id),
                        created_at=row.created_at,
                        is_active=(row.branch_id == active),
                    )
                    for row in rows
                ]
                infos.sort(
                    key=lambda b: (
                        0 if b.branch_id == TRUNK_BRANCH_ID else 1,
                        b.created_at,
                        b.branch_id,
                    )
                )
                return infos
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="list_branches"
            ) from exc

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        """Set the session's active head to ``branch_id``.

        Raises:
            ValueError: If ``branch_id`` does not exist for this session.
            fifty_agent_sdk.errors.StateStoreError: On backend failure.
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                if parent is None:
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for session {session_id!r}"
                    )
                branch_map, _active = await self._load_branch_map(session, session_id)
                if branch_id not in branch_map:
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for session {session_id!r}"
                    )
                parent.active_branch_id = branch_id
            _log.debug("sql_state_store.switch_branch", session_id=session_id, branch_id=branch_id)
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="switch_branch"
            ) from exc

    async def truncate_after(
        self, session_id: str, sequence: int, *, branch_id: str | None = None
    ) -> None:
        """Destructively delete the target branch's own messages with
        ``sequence > N`` under a ``SELECT ... FOR UPDATE`` lock.

        The ``branch_id`` filter means only rows physically on the target
        branch are removed — a fork's inherited prefix (rows owned by ancestor
        branches) is never touched. Idempotent; a no-op on an unknown session
        or branch (the DELETE simply matches no rows).
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                if parent is None:
                    return
                target = (
                    branch_id
                    if branch_id is not None
                    else (parent.active_branch_id or TRUNK_BRANCH_ID)
                )
                await session.execute(
                    delete(AgentMessage).where(
                        AgentMessage.session_id == session_id,
                        AgentMessage.branch_id == target,
                        AgentMessage.sequence > sequence,
                    )
                )
            _log.debug(
                "sql_state_store.truncate_after",
                session_id=session_id,
                branch_id=target,
                sequence=sequence,
            )
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="truncate_after"
            ) from exc


__all__ = [
    "AgentBranch",
    "AgentMessage",
    "AgentSession",
    "SqlStateStore",
    "sql_metadata",
]
