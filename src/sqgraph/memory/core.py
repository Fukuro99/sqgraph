"""Temporal knowledge-graph long-term memory on top of :class:`sqgraph.GraphDB`.

Data model (all nodes carry ``group_id``; times are stored as UTC ``YYYY-MM-DDTHH:MM:SSZ``)::

    (:Episode {id, group_id, source, channel, author, content, occurred_at, ingested_at, source_ref})
    (:Entity  {id, group_id, name, type, summary, aliases, created_at, updated_at, ...})   # vector + trigram FTS

    (subject:Entity)-[:FACT {id, group_id, fact, text_norm, relation, subject_id, object_id, valid_at,
                             invalid_at, created_at, expired_at, channels, episode_ids, confidence,
                             modality, superseded_by, supersedes}]->(object:Entity)      # vector + trigram FTS
    (Episode)-[:MENTIONS]->(Entity)

A fact is one ``FACT`` relationship (like a Graphiti edge); the verb is the ``relation`` property. A fact
without an object (an attribute such as a status or an absence) is a self-loop ``(s)-[:FACT]->(s)`` with
``object_id`` absent. ``subject_id`` / ``object_id`` are the denormalised entity ids.

Bi-temporal: ``valid_at`` / ``invalid_at`` are event time (when the fact is true in the world),
``created_at`` / ``expired_at`` are system time. Contradicted facts are never deleted: the old fact gets
``invalid_at`` and ``superseded_by`` (id of the newer fact), the newer one lists the ids in ``supersedes``.

Access control: every Episode and Fact records its source ``channel``(s). All read methods take
``channels`` (allowed channels, ``None`` = unrestricted). See :meth:`Memory.search` and the README.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..db import GraphDB
from ..errors import SqgraphError
from ..schema import RELATIONSHIP
from ..types import Node, Relationship
from . import prompts
from .filelock import FileLock
from .providers import Embedder, LLMProvider
from .util import (
    cosine,
    fts_or_query,
    iso_or_none,
    local_with_weekday,
    nfkc,
    new_id,
    norm_key,
    now_iso,
    parse_time,
    person_forms,
    query_terms,
    resolve_tz,
    short,
    to_iso,
    to_local,
    clean_message,
)

ENTITY_INDEX = "entity_emb"
FACT_INDEX = "fact_emb"
ENTITY_FT = "entity_ft"
FACT_FT = "fact_ft"
FACT_ID = "fact_id"  # relationship property index on FACT.id
FACT_TYPE = "FACT"
EPISODE_FT = "episode_ft"
RRF_K = 60

# Version of the memory data layout, kept in the ``_meta`` table (key ``memory_schema_version``). Databases
# created before the key existed count as version 1. To change the layout: bump the constant and register a
# migration ``{from_version: fn(memory)}`` in ``_MIGRATIONS`` (run in order, each in its own transaction).
MEMORY_SCHEMA_VERSION = 1
META_SCHEMA_KEY = "memory_schema_version"
_MIGRATIONS: dict[int, Callable[["Memory"], None]] = {}

# Identity of the embedder that produced the stored vectors, kept in ``_meta``. Vectors of different models are not
# comparable even when their dimensions match, so opening a database with another model is refused (``reembed``).
META_EMBED_MODEL = "embed_model"
META_EMBED_DIMS = "embed_dimensions"
META_EMBED_ENDPOINT = "embed_endpoint"


# --------------------------------------------------------------------------- result types


@dataclass
class EpisodeRef:
    """A source episode, for citation."""

    id: str
    author: str | None
    channel: str
    occurred_at: str | None  # local ISO
    source_ref: str | None
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FactHit:
    """A fact returned by :meth:`Memory.search` / :meth:`Memory.get_entity`."""

    id: str
    fact: str
    relation: str
    subject: dict[str, Any]  # {"id", "name"}
    object: dict[str, Any] | None
    valid_at: str | None  # local ISO
    invalid_at: str | None
    created_at: str | None
    expired_at: str | None
    status: str  # valid | scheduled | ended  (relative to as_of)
    confidence: float
    modality: str
    score: float = 0.0
    episodes: list[EpisodeRef] = field(default_factory=list)
    superseded_by: str | None = None
    supersedes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["episodes"] = [e.to_dict() for e in self.episodes]
        d["score"] = round(self.score, 5)
        return d


@dataclass
class EpisodeResult:
    episode_id: str
    skipped: bool = False  # already ingested (same source_ref)
    entities: list[str] = field(default_factory=list)
    new_entities: list[str] = field(default_factory=list)
    facts_created: list[str] = field(default_factory=list)
    facts_duplicate: list[str] = field(default_factory=list)
    facts_invalidated: list[str] = field(default_factory=list)
    llm_calls: int = 0
    usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ForgetResult:
    """What a ``forget_*`` / :meth:`Memory.purge` call deleted or changed (ids, not counts).

    ``facts_restored`` are facts that had been superseded by a deleted fact and are valid again;
    ``entities_updated`` are surviving entities whose aliases / summaries / embedding were recomputed.
    With ``dry_run`` nothing was written.
    """

    episodes_deleted: list[str] = field(default_factory=list)
    facts_deleted: list[str] = field(default_factory=list)
    facts_restored: list[str] = field(default_factory=list)
    entities_deleted: list[str] = field(default_factory=list)
    entities_updated: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, list) else v for k, v in self.__dict__.items()}

    @property
    def empty(self) -> bool:
        return not (self.episodes_deleted or self.facts_deleted or self.entities_deleted)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[FactHit]
    entities: list[dict[str, Any]]
    as_of: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "as_of": self.as_of,
            "facts": [h.to_dict() for h in self.hits],
            "entities": self.entities,
        }


@dataclass
class _WorkEntity:
    """An entity of the episode being ingested."""

    idx: int
    name: str
    type: str
    description: str
    node: Node | None = None  # resolved existing entity
    is_author: bool = False
    forms: list[str] = field(default_factory=list)  # surface forms seen in this episode
    match_keys: set[str] = field(default_factory=set)  # normalised keys used for exact entity matching
    new_summary: str | None = None  # LLM-updated summary line for the episode's channel
    chan_summary: str | None = None  # final summary line to store for the channel (None = unchanged)
    embedding: list[float] | None = None
    qvec: list[float] | None = None


# --------------------------------------------------------------------------- helpers


def _rrf(rank_lists: Sequence[tuple[float, list[str]]]) -> dict[str, float]:
    """Reciprocal-rank fusion of weighted ranked id lists."""
    scores: dict[str, float] = defaultdict(float)
    for weight, ids in rank_lists:
        for rank, i in enumerate(ids):
            scores[i] += weight / (RRF_K + rank + 1)
    return scores


def _types_compatible(a: str, b: str) -> bool:
    return (a == "Person") == (b == "Person")


def _as_int(x: Any) -> int | None:
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def _channel_set(values: Iterable[str] | None) -> set[str] | None:
    return None if values is None else {str(v) for v in values}


def _fact_status(p: dict[str, Any], at: str) -> str:
    ia, va = p.get("invalid_at"), p.get("valid_at")
    if ia is not None and ia <= at:
        return "ended"
    if va is not None and va > at:
        return "scheduled"
    return "valid"


# --------------------------------------------------------------------------- Memory


class Memory:
    """Long-term memory service: ingestion (LLM-assisted) and hybrid temporal retrieval.

    Parameters
    ----------
    db:
        A :class:`GraphDB` or a path (opened here; closed by :meth:`close`).
    llm, embedder:
        Providers (see :mod:`sqgraph.memory.providers`). ``llm`` is only needed for ingestion and
        :meth:`answer`; ``embedder`` for ingestion and vector search (without it search falls back to
        fulltext + entity matching).
    default_group_id:
        Namespace used when a method is called without ``group_id``.
    tz:
        Timezone (name or ``tzinfo``) for naive timestamps and for the times shown to the LLM and in
        results. Default Asia/Tokyo (JST).
    context_window:
        Number of preceding messages of the same channel given to the LLM as context.
    lock_timeout:
        Seconds a write operation waits for the cross-process write lock (``<db path>.lock``) before raising
        :class:`TimeoutError`. Every write holds that lock for its whole duration (including LLM calls), so
        several processes (agents) can safely share one database file.

    allow_embedder_change:
        Skip the embedding identity / dimension check (for :meth:`reembed`, which replaces every vector).
        Otherwise a database built with another embedding model or dimension is refused (``SqgraphError``).

    A database opened here has ``PRAGMA secure_delete=ON``: forgotten data is overwritten, not left in free pages.
    """

    def __init__(
        self,
        db: GraphDB | str | os.PathLike[str],
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        *,
        default_group_id: str | None = None,
        tz: str | tzinfo | None = None,
        context_window: int = 5,
        min_content_chars: int = 5,
        read_only: bool = False,
        clock: Callable[[], datetime] | None = None,
        lock_timeout: float = 300.0,
        allow_embedder_change: bool = False,
    ) -> None:
        self._allow_embedder_change = allow_embedder_change
        self._owns_db = not isinstance(db, GraphDB)
        self.db = db if isinstance(db, GraphDB) else GraphDB(db, read_only=read_only, secure_delete=True)
        self.llm = llm
        self.embedder = embedder
        self.default_group_id = default_group_id
        self.tz = resolve_tz(tz)
        self.context_window = context_window
        self.min_content_chars = min_content_chars
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._write_lock = threading.RLock()
        self.lock_timeout = lock_timeout
        self._lock_depth = 0
        self._file_lock: FileLock | None = None
        self._llm_calls = 0
        try:
            if not self.db.read_only:
                self.db.set_secure_delete(True)
                if self.db.path != ":memory:":
                    self._file_lock = FileLock(self.db.path + ".lock", lock_timeout)
            self._ensure_schema()
        except BaseException:
            self.close()  # do not leak a connection opened here
            raise

    # ------------------------------------------------------------------ lifecycle / schema

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _write_guard(self) -> Iterator[None]:
        """Exclusive write access: the in-process lock, then the cross-process file lock (outermost entry only)."""
        with self._write_lock:
            if self._lock_depth == 0 and self._file_lock is not None:
                self._file_lock.acquire(self.lock_timeout)
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._file_lock is not None:
                    self._file_lock.release()

    @property
    def schema_version(self) -> int:
        """Version of the memory data layout stored in the database (see ``MEMORY_SCHEMA_VERSION``)."""
        stored = self._stored_version()
        return 1 if stored is None else stored

    def _stored_version(self) -> int | None:
        raw = self.db.get_meta(META_SCHEMA_KEY)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            raise SqgraphError(f"invalid {META_SCHEMA_KEY} in the database: {raw!r}") from None

    @property
    def usage(self) -> Any:
        """The LLM provider's :class:`UsageTracker` if it has one."""
        return getattr(self.llm, "tracker", None)

    def _dims(self) -> int:
        if self.embedder is not None:
            return self.embedder.dimensions
        for ix in self.db.vector_indexes():
            if ix.name == FACT_INDEX:
                return ix.dimensions
        return 1536

    def _ensure_schema(self) -> None:
        with self._write_guard():  # concurrent first opens must not race on index creation / migration
            self._ensure_schema_locked()

    def _ensure_schema_locked(self) -> None:
        db = self.db
        self._reject_old_format()
        stored = self._stored_version()
        if stored is not None and stored > MEMORY_SCHEMA_VERSION:
            raise SqgraphError(
                f"this memory database has schema version {stored} (created by a newer sqgraph; this one "
                f"supports up to {MEMORY_SCHEMA_VERSION}): upgrade sqgraph"
            )
        if db.read_only:
            names = {i["name"] for i in db.list_indexes()}
            missing = {FACT_INDEX, FACT_FT, FACT_ID, ENTITY_FT} - names
            if missing:
                raise SqgraphError(f"not a memory database (missing indexes: {sorted(missing)})")
        else:
            dims = self._dims()
            db.create_vector_index(ENTITY_INDEX, "Entity", "embedding", dimensions=dims, if_not_exists=True)
            db.create_relationship_vector_index(FACT_INDEX, FACT_TYPE, "embedding", dimensions=dims, if_not_exists=True)
            db.create_fulltext_index(ENTITY_FT, "Entity", ["search_text"], if_not_exists=True, tokenizer="trigram")
            db.create_relationship_fulltext_index(FACT_FT, FACT_TYPE, ["text_norm"], if_not_exists=True, tokenizer="trigram")
            db.create_relationship_property_index(FACT_TYPE, "id", name=FACT_ID)
            db.create_fulltext_index(EPISODE_FT, "Episode", ["text_norm"], if_not_exists=True, tokenizer="trigram")
            db.create_property_index("Memory", "id", name="mem_id")
            db.create_property_index("Memory", "source_ref", name="mem_source_ref")
            self._migrate(stored)
        if self.embedder is not None and not self._allow_embedder_change:
            self._check_embedder()

    # ------------------------------------------------------------------ embedding identity

    @staticmethod
    def embedder_identity(embedder: Embedder) -> dict[str, Any] | None:
        """``{"model", "dimensions", "endpoint"}`` of an embedder, or ``None`` if it does not name its model
        (such embedders are only checked for their dimensions)."""
        model = getattr(embedder, "model", None)
        if not model:
            return None
        return {
            "model": str(model),
            "dimensions": int(embedder.dimensions),
            "endpoint": getattr(embedder, "endpoint", None) or None,
        }

    def embedding_identity(self) -> dict[str, Any] | None:
        """The identity recorded in the database (``model`` / ``dimensions`` / ``endpoint``), or ``None`` if none yet."""
        model = self.db.get_meta(META_EMBED_MODEL)
        if model is None:
            return None
        dims = self.db.get_meta(META_EMBED_DIMS)
        return {
            "model": model,
            "dimensions": int(dims) if dims and dims.isdigit() else None,
            "endpoint": self.db.get_meta(META_EMBED_ENDPOINT),
        }

    def _store_embedding_identity(self, ident: dict[str, Any]) -> None:
        self.db.set_meta(META_EMBED_MODEL, ident["model"])
        self.db.set_meta(META_EMBED_DIMS, str(ident["dimensions"]))
        if ident.get("endpoint"):
            self.db.set_meta(META_EMBED_ENDPOINT, ident["endpoint"])

    def _check_embedder(self) -> None:
        """Refuse an embedder whose model / dimensions differ from the stored vectors; adopt its identity if none is
        recorded yet; record a changed endpoint (same model and dimensions). Read-only opens only check."""
        emb = self.embedder
        assert emb is not None
        db = self.db
        ident = self.embedder_identity(emb)
        stored = self.embedding_identity()
        ix = next((i for i in db.vector_indexes() if i.name == FACT_INDEX), None)
        hint = "run `sqgraph-memory reembed` to recompute every embedding with the current embedder"
        if ident is not None and stored is not None:
            if stored["model"] != ident["model"] or stored["dimensions"] not in (None, ident["dimensions"]):
                was = f"model {stored['model']!r} ({stored['dimensions']} dimensions, endpoint {stored['endpoint'] or 'unknown'})"
                now = f"model {ident['model']!r} ({ident['dimensions']} dimensions, endpoint {ident['endpoint'] or 'unknown'})"
                raise SqgraphError(
                    f"embedding model mismatch: this database was built with {was}, but the configured embedder is "
                    f"{now}. Vectors from different models are not comparable, even when the dimensions match: "
                    f"{hint} (or switch back to the original model)"
                )
        if ix and ix.dimensions != emb.dimensions:
            raise SqgraphError(
                f"embedding dimension mismatch: database has {ix.dimensions}, embedder produces {emb.dimensions}; {hint}"
            )
        if ident is not None and not db.read_only and (stored is None or stored["endpoint"] != ident["endpoint"]):
            with db.transaction():
                self._store_embedding_identity(ident)

    def _migrate(self, stored: int | None) -> None:
        """Bring the stored memory schema version up to date (``None``: no version recorded yet)."""
        version = stored if stored is not None else 1  # databases from before the key existed are version 1
        if stored is None and not self.db.query("MATCH (e:Episode) RETURN e.id AS id LIMIT 1"):
            version = MEMORY_SCHEMA_VERSION  # nothing stored yet: starts at the current layout
        while version < MEMORY_SCHEMA_VERSION:
            step = _MIGRATIONS.get(version)
            if step is None:
                raise SqgraphError(f"no migration from memory schema version {version} to {version + 1}")
            with self.db.transaction():
                step(self)
                self.db.set_meta(META_SCHEMA_KEY, str(version + 1))
            version += 1
        if stored != version:
            self.db.set_meta(META_SCHEMA_KEY, str(version))

    def _reject_old_format(self) -> None:
        """Databases from before edge-based facts (Fact nodes, node-level fact indexes) are not migrated."""
        db = self.db
        old = any(ix.name == FACT_INDEX and ix.entity_type != RELATIONSHIP for ix in db.vector_indexes())
        old = old or any(
            i["name"] == FACT_FT and i["entityType"] != RELATIONSHIP.upper() for i in db.list_indexes()
        )
        old = old or bool(db.query("MATCH (f:Fact) RETURN f.id AS id LIMIT 1"))
        if old:
            raise SqgraphError(
                "this memory database uses the old format (facts as :Fact nodes); facts are now stored as "
                "[:FACT] relationships and old databases are not migrated: recreate the database"
            )

    # ------------------------------------------------------------------ small helpers

    def _group(self, group_id: str | None) -> str:
        g = group_id or self.default_group_id
        if not g:
            raise ValueError("group_id is required (no default_group_id configured)")
        return g

    def _now(self) -> str:
        return to_iso(self._clock())

    def _local(self, iso: str | None) -> str | None:
        return to_local(iso, self.tz)

    def _need_llm(self) -> LLMProvider:
        if self.llm is None:
            raise RuntimeError("this operation needs an LLM provider (Memory(..., llm=...))")
        return self.llm

    def _need_embedder(self) -> Embedder:
        if self.embedder is None:
            raise RuntimeError("this operation needs an embedder (Memory(..., embedder=...))")
        return self.embedder

    def _llm_json(self, task: str, system: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        self._llm_calls += 1
        return self._need_llm().complete_json(
            system, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), schema, task=task
        )

    def _one(self, cypher: str, params: dict[str, Any]) -> Node | None:
        rows = self.db.query(cypher, params)
        return next(iter(rows[0].values())) if rows else None

    def _entity_node(self, group_id: str, entity_id: str) -> Node | None:
        n = self._one("MATCH (e:Entity {id:$id}) RETURN e", {"id": entity_id})
        return n if n is not None and n.get("group_id") == group_id else None

    def _fact_edge(self, fact_id: str) -> Relationship | None:
        """A fact by id (uses the relationship property index on ``FACT.id``)."""
        rows = self.db.query("MATCH ()-[f:FACT {id:$id}]->() RETURN f", {"id": fact_id})
        return rows[0]["f"] if rows else None

    def _facts_by_id(self, ids: Iterable[str], group: str | None = None) -> dict[str, Relationship]:
        out: dict[str, Relationship] = {}
        for i in dict.fromkeys(ids):
            f = self._fact_edge(i)
            if f is not None and (group is None or f.get("group_id") == group):
                out[i] = f
        return out

    def _nodes(self, label: str, ids: Iterable[str], group_id: str | None = None) -> dict[str, Node]:
        ids = list(dict.fromkeys(ids))
        out: dict[str, Node] = {}
        for i in range(0, len(ids), 200):
            rows = self.db.query(f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n", {"ids": ids[i : i + 200]})
            for r in rows:
                n = r["n"]
                if group_id is None or n.get("group_id") == group_id:
                    out[n["id"]] = n
        return out

    def _embed_query(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        return self.embedder.embed([text])[0]

    # ================================================================== INGESTION

    def add_episode(
        self,
        content: str,
        *,
        group_id: str | None = None,
        source: str = "api",
        channel: str = "default",
        author: str | None = None,
        occurred_at: str | datetime | None = None,
        source_ref: str | None = None,
    ) -> EpisodeResult:
        """Ingest one message / note: extract entities and facts, resolve, deduplicate, write.

        Typically 2-4 LLM calls (entity extraction; entity resolution only for ambiguous names; fact
        extraction; contradiction check only when related facts exist) and 2 batched embedding calls.
        Everything is written in a single transaction. Re-ingesting the same ``(source, source_ref)``
        is a no-op (``skipped=True``).
        """
        if self.db.read_only:
            raise SqgraphError("memory database is read-only")
        group = self._group(group_id)
        with self._write_guard():
            calls0 = self._llm_calls
            usd0 = self.usage.snapshot() if self.usage else None
            if source_ref:
                dup = self._one(
                    "MATCH (e:Episode {group_id:$g, source:$s, source_ref:$r}) RETURN e",
                    {"g": group, "s": source, "r": source_ref},
                )
                if dup is not None:
                    return EpisodeResult(dup["id"], skipped=True)
            occurred = parse_time(occurred_at, self.tz) if occurred_at else self._clock()
            if occurred is None:
                raise ValueError(f"cannot parse occurred_at: {occurred_at!r}")
            occurred_iso = to_iso(occurred)
            text = clean_message(content)
            result = self._ingest(group, source, channel, author, occurred_iso, text, source_ref)
            result.llm_calls = self._llm_calls - calls0
            if self.usage is not None and usd0 is not None:
                result.usd = self.usage.diff_usd(usd0)
            return result

    def add_episodes_bulk(
        self,
        items: Iterable[dict[str, Any]],
        *,
        group_id: str | None = None,
        source: str = "bulk",
        channel: str = "default",
        progress: Callable[[int, int, EpisodeResult], None] | None = None,
    ) -> list[EpisodeResult]:
        """Ingest many episodes in chronological order (sorted by ``occurred_at``).

        Each item is a dict with ``content`` and optional ``occurred_at``, ``author``, ``channel``,
        ``source``, ``source_ref``. Sequential on purpose: later messages see earlier facts.
        """
        rows = list(items)

        def key(it: dict[str, Any]) -> str:
            dt = parse_time(it.get("occurred_at"), self.tz)
            return to_iso(dt) if dt else ""

        rows.sort(key=key)
        out: list[EpisodeResult] = []
        for i, it in enumerate(rows):
            r = self.add_episode(
                it["content"],
                group_id=group_id,
                source=it.get("source", source),
                channel=it.get("channel", channel),
                author=it.get("author"),
                occurred_at=it.get("occurred_at"),
                source_ref=it.get("source_ref"),
            )
            out.append(r)
            if progress:
                progress(i + 1, len(rows), r)
        if not self.db.read_only:
            with self._write_guard():
                self.db.analyze()
        return out

    # -- pipeline ---------------------------------------------------------------------------

    def _recent_context(self, group: str, channel: str, before_iso: str) -> list[dict[str, str]]:
        if self.context_window <= 0:
            return []
        rows = self.db.query(
            "MATCH (e:Episode {group_id:$g, channel:$c}) WHERE e.occurred_at <= $t "
            "RETURN e.author AS author, e.content AS content, e.occurred_at AS at "
            "ORDER BY e.occurred_at DESC LIMIT $n",
            {"g": group, "c": channel, "t": before_iso, "n": self.context_window},
        )
        return [
            {"at": local_with_weekday(r["at"], self.tz)[:16], "author": r["author"] or "?", "text": short(r["content"], 240)}
            for r in reversed(rows)
        ]

    def _ingest(
        self,
        group: str,
        source: str,
        channel: str,
        author: str | None,
        occurred_iso: str,
        text: str,
        source_ref: str | None,
    ) -> EpisodeResult:
        ingested = self._now()
        episode_id = new_id("ep")
        ctx = self._recent_context(group, channel, occurred_iso)
        author_name, author_aliases = person_forms(author) if author else (None, [])
        trivial = len(text.strip()) < self.min_content_chars or self.llm is None or self.embedder is None

        work: list[_WorkEntity] = []
        facts_out: list[dict[str, Any]] = []
        if not trivial:
            work, facts_out = self._understand(
                group, channel, author, author_name, author_aliases, occurred_iso, text, ctx
            )
        elif author_name:
            work = [self._author_entity(author_name, author_aliases, author)]
            self._resolve_entities(group, work, self.embedder)
        return self._commit(
            group, source, channel, author, occurred_iso, ingested, text, source_ref, episode_id, work, facts_out
        )

    def _payload_aliases(self, w: _WorkEntity, channel: str) -> list[str]:
        """Aliases shown to the fact-extraction LLM: forms seen in this message + forms known in this channel."""
        known = self._visible_forms(w.node.properties, {channel})[1] if w.node is not None else []
        seen = [f for f in w.forms if f != w.name and norm_key(f) in w.match_keys]
        return list(dict.fromkeys([*seen, *known]))[:4]

    @staticmethod
    def _author_entity(name: str, aliases: list[str], raw: str | None) -> _WorkEntity:
        w = _WorkEntity(0, name, "Person", "", is_author=True)
        w.forms = [name, *aliases]
        # match on the canonical name and on its '/'-separated parts, never on the raw (suffixed) form
        w.match_keys = {norm_key(name)} | {norm_key(a) for a in aliases if a != raw}
        w.match_keys.discard("")
        return w

    def _exact_matches(self, group: str, keys: set[str], typ: str) -> list[Node]:
        keys = {k for k in keys if k}
        if not keys:
            return []
        rows = self.db.query(
            "MATCH (e:Entity {group_id:$g}) WHERE any(k IN $keys WHERE k IN e.alias_keys) RETURN e",
            {"g": group, "keys": sorted(keys)},
        )
        return [r["e"] for r in rows if _types_compatible(r["e"].get("type", "Other"), typ)]

    def _understand(
        self,
        group: str,
        channel: str,
        author: str | None,
        author_name: str | None,
        author_aliases: list[str],
        occurred_iso: str,
        text: str,
        ctx: list[dict[str, str]],
    ) -> tuple[list[_WorkEntity], list[dict[str, Any]]]:
        embedder = self._need_embedder()
        when = local_with_weekday(occurred_iso, self.tz)
        # 1. entities ------------------------------------------------------------------
        ex = self._llm_json(
            "extract_entities",
            prompts.EXTRACT_ENTITIES_SYSTEM,
            {"author": author_name or "unknown", "occurred_at": when, "context": ctx, "message": text},
            prompts.EXTRACT_ENTITIES_SCHEMA,
        )
        work: list[_WorkEntity] = []
        if author_name:
            work.append(self._author_entity(author_name, author_aliases, author))
        for e in ex.get("entities", []):
            name = (e.get("name") or "").strip()
            if not name:
                continue
            existing = next((w for w in work if norm_key(name) in w.match_keys), None)
            if existing is not None:
                existing.forms.append(name)
                continue
            work.append(
                _WorkEntity(
                    len(work),
                    name,
                    e.get("type") or "Other",
                    (e.get("description") or "").strip(),
                    forms=[name],
                    match_keys={norm_key(name)} - {""},
                )
            )
        for i, w in enumerate(work):
            w.idx = i

        # 2. resolution -----------------------------------------------------------------
        self._resolve_entities(group, work, embedder)

        # 3. facts ----------------------------------------------------------------------
        # ACL: text generated by the LLM must only depend on information visible in this channel
        # (names as written here, this channel's summary entry), never on other channels' data.
        facts: list[dict[str, Any]] = []
        fx: dict[str, Any] = {}
        if work and ex.get("has_facts", True):
            fx = self._llm_json(
                "extract_facts",
                prompts.EXTRACT_FACTS_SYSTEM,
                {
                    "author": author_name or "unknown",
                    "occurred_at": when,
                    "context": ctx,
                    "entities": [
                        {
                            "idx": w.idx,
                            "name": w.name,
                            "type": w.node["type"] if w.node else w.type,
                            "aliases": self._payload_aliases(w, channel),
                            "summary": short(self._channel_summary(w.node, channel) or w.description, 160),
                        }
                        for w in work
                    ],
                    "message": text,
                },
                prompts.EXTRACT_FACTS_SCHEMA,
            )
        by_idx = {w.idx: w for w in work}
        for s in fx.get("entity_summaries", []):
            i = _as_int(s.get("idx"))
            if i in by_idx and (s.get("summary") or "").strip():
                by_idx[i].new_summary = short(s["summary"], 300)
        for f in fx.get("facts", []):
            si, oi = _as_int(f.get("subject")), _as_int(f.get("object"))
            fact_text = (f.get("fact") or "").strip()
            if si not in by_idx or not fact_text or (oi is not None and oi not in by_idx):
                continue
            va = iso_or_none(f.get("valid_at"), self.tz) or occurred_iso
            ia = iso_or_none(f.get("invalid_at"), self.tz)
            if ia is not None and ia <= va:
                ia = None
            modality = f.get("modality") if f.get("modality") in ("definite", "planned", "tentative") else "definite"
            rel = "_".join((f.get("relation") or "RELATES_TO").upper().replace("-", "_").split()) or "RELATES_TO"
            facts.append(
                {
                    "subject": si,
                    "object": oi,
                    "fact": fact_text,
                    "relation": rel,
                    "valid_at": va,
                    "invalid_at": ia,
                    "modality": modality,
                    "confidence": {"definite": 1.0, "planned": 0.7, "tentative": 0.5}[modality],
                }
            )
        # de-duplicate identical sentences within the message
        seen: set[str] = set()
        facts = [f for f in facts if not (f["fact"] in seen or seen.add(f["fact"]))]

        # 4. embeddings (entities needing one + new facts), one batched call ------------------
        texts: list[str] = []
        slots: list[tuple[str, Any]] = []
        self._finalize_summaries(work, channel)
        for w in work:
            if w.node is None or w.chan_summary is not None:
                texts.append(self._entity_embed_text(w, channel))
                slots.append(("entity", w))
        for f in facts:
            texts.append(f["fact"])
            slots.append(("fact", f))
        if texts:
            vecs = embedder.embed(texts)
            for (kind, obj), v in zip(slots, vecs):
                if kind == "entity":
                    obj.embedding = v
                else:
                    obj["embedding"] = v

        # 5. contradiction / duplicate check ----------------------------------------------
        if facts:
            self._dedupe(group, channel, work, facts)
        return work, facts

    # -- per-channel entity summaries ---------------------------------------------------------
    # An entity keeps one summary line per source channel (``summaries``: "channel<TAB>text"). The LLM only
    # ever sees / rewrites the line of the channel being ingested, so no text crosses channels.

    @staticmethod
    def _entries(p: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for e in p.get("summaries", []):
            ch, _, t = e.partition("\t")
            out[ch] = t
        return out

    def _channel_summary(self, node: Node | None, channel: str) -> str:
        return self._entries(node.properties).get(channel, "") if node is not None else ""

    def _finalize_summaries(self, work: list[_WorkEntity], channel: str) -> None:
        """Decide the new summary line of ``channel`` for every entity (``chan_summary``; None = unchanged)."""
        for w in work:
            if w.new_summary:
                w.chan_summary = w.new_summary
            elif w.node is None:
                w.chan_summary = w.description
            elif not self._channel_summary(w.node, channel) and w.description:
                w.chan_summary = w.description
            else:
                w.chan_summary = None

    def _merged_entries(self, w: _WorkEntity, channel: str) -> dict[str, str]:
        entries = self._entries(w.node.properties) if w.node is not None else {}
        if w.chan_summary:
            entries[channel] = w.chan_summary
        return entries

    def _entity_embed_text(self, w: _WorkEntity, channel: str) -> str:
        name = w.node["name"] if w.node is not None else w.name
        return short(f"{name}: {' / '.join(self._merged_entries(w, channel).values())}", 400)

    def _resolve_entities(self, group: str, work: list[_WorkEntity], embedder: Embedder | None) -> None:
        """Fill ``node`` of every work entity that matches an existing entity (LLM only when ambiguous)."""
        if not work:
            return
        all_keys = set().union(*(w.match_keys for w in work))
        pool: dict[str, Node] = {}
        if all_keys:
            for r in self.db.query(
                "MATCH (e:Entity {group_id:$g}) WHERE any(k IN $keys WHERE k IN e.alias_keys) RETURN e",
                {"g": group, "keys": sorted(all_keys)},
            ):
                pool[r["e"]["id"]] = r["e"]
        ambiguous: list[_WorkEntity] = []
        for w in work:
            exact = [
                n
                for n in pool.values()
                if set(n.get("alias_keys", [])) & w.match_keys and _types_compatible(n.get("type", "Other"), w.type)
            ]
            if w.is_author:  # a person's display name is authoritative: no LLM needed
                if exact:
                    w.node = exact[0]
            elif len(exact) == 1:
                w.node = exact[0]
            else:
                ambiguous.append(w)
        # semantic candidates for the rest
        if ambiguous and embedder is not None:
            qvecs = embedder.embed([short(f"{w.name}: {w.description}", 300) for w in ambiguous])
            payload_entities: list[dict[str, Any]] = []
            cand_ids: dict[int, set[str]] = {}
            for w, qv in zip(ambiguous, qvecs):
                w.qvec = qv
                cands: dict[str, Node] = {}
                for n in self._exact_matches(group, w.match_keys, w.type):
                    cands[n["id"]] = n
                for n, score in self.db.vector_search(ENTITY_INDEX, qv, k=8):
                    if n.get("group_id") == group and score >= 0.72 and _types_compatible(n.get("type", "Other"), w.type):
                        cands.setdefault(n["id"], n)
                q = fts_or_query([nfkc(w.name).casefold()])
                if q:
                    for n, _ in self.db.fulltext_search(ENTITY_FT, q, k=5):
                        if n.get("group_id") == group and _types_compatible(n.get("type", "Other"), w.type):
                            cands.setdefault(n["id"], n)
                if not cands:
                    continue
                top = list(cands.values())[:5]
                cand_ids[w.idx] = {n["id"] for n in top}
                payload_entities.append(
                    {
                        "idx": w.idx,
                        "name": w.name,
                        "type": w.type,
                        "description": w.description,
                        "candidates": [
                            {
                                "id": n["id"],
                                "name": n["name"],
                                "type": n.get("type"),
                                "aliases": n.get("aliases", [])[:5],
                                "summary": short(n.get("summary", ""), 100),
                            }
                            for n in top
                        ],
                    }
                )
            if payload_entities:
                res = self._llm_json(
                    "resolve_entities",
                    prompts.RESOLVE_ENTITIES_SYSTEM,
                    {"entities": payload_entities},
                    prompts.RESOLVE_ENTITIES_SCHEMA,
                )
                by_idx = {w.idx: w for w in work}
                nodes = self._nodes("Entity", {i for s in cand_ids.values() for i in s}, group)
                for r in res.get("resolutions", []):
                    i, m = _as_int(r.get("idx")), r.get("match")
                    if i in by_idx and m in cand_ids.get(i, ()) and m in nodes:
                        by_idx[i].node = nodes[m]
        # entities that resolved to the same existing node are merged into the first one
        first: dict[str, _WorkEntity] = {}
        for w in list(work):
            if w.node is None:
                continue
            nid = w.node["id"]
            if nid in first:
                first[nid].forms += w.forms
                work.remove(w)
            else:
                first[nid] = w

    def _dedupe(self, group: str, channel: str, work: list[_WorkEntity], facts: list[dict[str, Any]]) -> None:
        """Classify each new fact as new / duplicate / contradicting (fills ``duplicate_of`` / ``contradicts``)."""
        by_idx = {w.idx: w for w in work}
        ent_ids = {w.node["id"] for w in work if w.node}
        for f in facts:
            f["duplicate_of"] = None
            f["contradicts"] = []
        if not ent_ids:
            return
        # existing valid facts around the involved entities + vector neighbours
        around = [f for eid in sorted(ent_ids) for f in self._facts_of_entity(eid, group) if f.get("invalid_at") is None]
        around.sort(key=lambda f: f.get("valid_at", ""), reverse=True)
        pool: dict[str, Relationship] = {f["id"]: f for f in around[:60]}
        cand_for: list[list[str]] = []
        for f in facts:
            si = by_idx[f["subject"]]
            oi = by_idx.get(f["object"]) if f["object"] is not None else None
            sids = {x.node["id"] for x in (si, oi) if x is not None and x.node}
            related = [n for n in pool.values() if n.get("subject_id") in sids or n.get("object_id") in sids]
            ids = [n["id"] for n in related[:12]]
            for n, score in self.db.vector_search(FACT_INDEX, f["embedding"], k=6):
                if n.get("group_id") == group and n.get("invalid_at") is None and score >= 0.7 and n["id"] not in ids:
                    ids.append(n["id"])
                    pool.setdefault(n["id"], n)
            cand_for.append(ids)
        all_ids = list(dict.fromkeys(i for ids in cand_for for i in ids))
        if not all_ids:
            return
        handle = {fid: f"h{i + 1}" for i, fid in enumerate(all_ids)}
        rev = {h: fid for fid, h in handle.items()}
        names = self._entity_names(group, {p for fid in all_ids for p in (pool[fid].get("subject_id"), pool[fid].get("object_id")) if p})

        def ename(w: _WorkEntity | None) -> str | None:
            return None if w is None else (w.node["name"] if w.node else w.name)

        payload = {
            "new_facts": [
                {
                    "idx": i,
                    "fact": f["fact"],
                    "relation": f["relation"],
                    "subject": ename(by_idx[f["subject"]]),
                    "object": ename(by_idx.get(f["object"])) if f["object"] is not None else None,
                    "valid_at": self._local(f["valid_at"]),
                    "invalid_at": self._local(f["invalid_at"]),
                    "candidates": [handle[x] for x in cand_for[i]],
                }
                for i, f in enumerate(facts)
                if cand_for[i]
            ],
            "existing_facts": [
                {
                    "h": handle[fid],
                    "fact": pool[fid]["fact"],
                    "relation": pool[fid].get("relation"),
                    "subject": names.get(pool[fid].get("subject_id")),
                    "object": names.get(pool[fid].get("object_id")),
                    "valid_at": self._local(pool[fid].get("valid_at")),
                }
                for fid in all_ids
            ],
        }
        if not payload["new_facts"]:
            return
        res = self._llm_json("dedupe_facts", prompts.DEDUPE_FACTS_SYSTEM, payload, prompts.DEDUPE_FACTS_SCHEMA)
        for r in res.get("results", []):
            i = _as_int(r.get("idx"))
            if i is None or not 0 <= i < len(facts):
                continue
            allowed = {handle[x] for x in cand_for[i]}
            dup = r.get("duplicate_of")
            if dup in allowed:
                # ACL: a fact's text may only be shared between episodes of the SAME channel; the same statement
                # made in another channel becomes its own fact (never attach it to a fact worded from elsewhere).
                if channel in pool[rev[dup]].get("channels", []):
                    facts[i]["duplicate_of"] = rev[dup]
                continue
            facts[i]["contradicts"] = [rev[h] for h in r.get("contradicts", []) if h in allowed]

    def _entity_names(self, group: str, ids: Iterable[str]) -> dict[str, str]:
        return {i: n["name"] for i, n in self._nodes("Entity", ids, group).items()}

    # -- write --------------------------------------------------------------------------------

    def _commit(
        self,
        group: str,
        source: str,
        channel: str,
        author: str | None,
        occurred_iso: str,
        ingested: str,
        text: str,
        source_ref: str | None,
        episode_id: str,
        work: list[_WorkEntity],
        facts: list[dict[str, Any]],
    ) -> EpisodeResult:
        db = self.db
        res = EpisodeResult(episode_id)
        self._finalize_summaries(work, channel)
        missing = [w for w in work if w.embedding is None and (w.node is None or w.chan_summary is not None)]
        if missing and self.embedder is not None:
            for w, v in zip(missing, self.embedder.embed([self._entity_embed_text(w, channel) for w in missing])):
                w.embedding = v
        with db.transaction():
            ep_props: dict[str, Any] = {
                "id": episode_id,
                "group_id": group,
                "source": source,
                "channel": channel,
                "content": text,
                "text_norm": nfkc(text).casefold(),
                "occurred_at": occurred_iso,
                "ingested_at": ingested,
            }
            if author:
                ep_props["author"] = author
            if source_ref:
                ep_props["source_ref"] = source_ref
            ep = db.create_node("Episode", ep_props)

            # entities ---------------------------------------------------------------------
            node_of: dict[int, Node] = {}
            for w in work:
                node_of[w.idx] = self._write_entity(group, channel, ingested, w, res)
                db.create_edge(ep.id, node_of[w.idx].id, "MENTIONS", {"role": "author"} if w.is_author else {})
            res.entities = [n["id"] for n in node_of.values()]

            # facts ------------------------------------------------------------------------
            for f in facts:
                self._write_fact(group, channel, ingested, episode_id, node_of, f, res)
        return res

    def _write_entity(self, group: str, channel: str, now: str, w: _WorkEntity, res: EpisodeResult) -> Node:
        db = self.db
        forms = [x for x in dict.fromkeys(w.forms) if x]
        if w.node is None:
            name = w.name
            aliases = [x for x in forms if x != name]
            props: dict[str, Any] = {
                "id": new_id("ent"),
                "group_id": group,
                "name": name,
                "type": w.type,
                "summary": w.chan_summary or "",
                "summaries": [f"{channel}\t{w.chan_summary}"] if w.chan_summary else [],
                "summary_channels": [channel] if w.chan_summary else [],
                "aliases": aliases,
                "alias_keys": sorted({k for k in (norm_key(x) for x in [name, *aliases]) if k}),
                "alias_src": [f"{channel}\t{x}" for x in [name, *aliases]],
                "channels": [channel],
                "created_at": now,
                "updated_at": now,
            }
            props["search_text"] = nfkc("\n".join([name, *aliases])).casefold()
            if w.embedding is not None:
                props["embedding"] = w.embedding
            node = db.create_node("Entity", props)
            res.new_entities.append(props["id"])
            return node
        node = w.node
        p = node.properties
        name = p["name"]
        aliases = list(p.get("aliases", []))
        for x in forms:
            if x != name and x not in aliases:
                aliases.append(x)
        alias_src = list(p.get("alias_src", []))
        for x in forms:
            if f"{channel}\t{x}" not in alias_src:
                alias_src.append(f"{channel}\t{x}")
        channels = list(p.get("channels", []))
        if channel not in channels:
            channels.append(channel)
        upd: dict[str, Any] = {
            "aliases": aliases,
            "alias_keys": sorted({k for k in (norm_key(x) for x in [name, *aliases]) if k}),
            "alias_src": alias_src,
            "search_text": nfkc("\n".join([name, *aliases])).casefold(),
            "channels": channels,
            "updated_at": now,
        }
        if w.chan_summary:
            entries = self._merged_entries(w, channel)
            upd["summaries"] = [f"{ch}\t{t}" for ch, t in entries.items()]
            upd["summary"] = " / ".join(entries.values())
            upd["summary_channels"] = list(entries)
            if w.embedding is not None:
                upd["embedding"] = w.embedding
        return db.update_node(node.id, upd)

    def _write_fact(
        self,
        group: str,
        channel: str,
        now: str,
        episode_id: str,
        node_of: dict[int, Node],
        f: dict[str, Any],
        res: EpisodeResult,
    ) -> None:
        db = self.db
        if f.get("duplicate_of"):
            old = self._fact_edge(f["duplicate_of"])
            if old is not None:
                p = old.properties
                eps = list(p.get("episode_ids", []))
                chs = list(p.get("channels", []))
                if episode_id not in eps:
                    eps.append(episode_id)
                if channel not in chs:
                    chs.append(channel)
                upd: dict[str, Any] = {"episode_ids": eps, "channels": chs}
                if f["confidence"] > p.get("confidence", 0):
                    upd["confidence"] = f["confidence"]
                db.update_edge(old.id, upd)
                res.facts_duplicate.append(p["id"])
                return
        subj, obj = node_of[f["subject"]], node_of.get(f["object"]) if f["object"] is not None else None
        va, ia = f["valid_at"], f["invalid_at"]
        expired: str | None = None
        olds: list[Relationship] = []
        for oid in f.get("contradicts", []):
            o = self._fact_edge(oid)
            if o is not None and o.get("invalid_at") is None:
                olds.append(o)
        # the new fact may itself be historical (older than an existing one): it ends where the newer begins
        newer = [o for o in olds if o.get("valid_at", "") > va]
        for o in newer:
            if ia is None or o["valid_at"] < ia:
                ia = o["valid_at"]
            expired = now
        older = [o for o in olds if o.id not in {n.id for n in newer}]
        fid = new_id("fact")
        props: dict[str, Any] = {
            "id": fid,
            "group_id": group,
            "fact": f["fact"],
            "text_norm": nfkc(f["fact"]).casefold(),
            "relation": f["relation"],
            "subject_id": subj["id"],
            "valid_at": va,
            "created_at": now,
            "channels": [channel],
            "episode_ids": [episode_id],
            "confidence": f["confidence"],
            "modality": f["modality"],
            "embedding": f["embedding"],
        }
        if obj is not None:
            props["object_id"] = obj["id"]
        if ia is not None:
            props["invalid_at"] = ia
        if expired is not None:
            props["expired_at"] = expired
        if older:
            props["supersedes"] = [o["id"] for o in older]
        if newer:  # several newer facts: the one where this fact ends
            props["superseded_by"] = min(newer, key=lambda o: o["valid_at"])["id"]
        # an attribute fact (no object) is a self-loop on its subject
        db.create_edge(subj.id, (obj if obj is not None else subj).id, FACT_TYPE, props)
        res.facts_created.append(fid)
        for o in newer:  # existing fact is newer: it supersedes the new (historical) one
            db.update_edge(o.id, {"supersedes": [*o.get("supersedes", []), fid]})
        for o in older:
            self._invalidate(o, va, now, fid)
            res.facts_invalidated.append(o["id"])

    def _invalidate(self, old: Relationship, at: str, now: str, by_id: str) -> None:
        """Close ``old`` at valid time ``at`` (never deletes) and mark it ``superseded_by`` the new fact."""
        at = max(at, old.get("valid_at", at))
        self.db.update_edge(old.id, {"invalid_at": at, "expired_at": now, "superseded_by": by_id})

    # ================================================================== RETRIEVAL

    # -- visibility -----------------------------------------------------------------------------

    @staticmethod
    def _fact_visible(p: dict[str, Any], allowed: set[str] | None) -> bool:
        return allowed is None or bool(set(p.get("channels", ())) & allowed)

    @staticmethod
    def _entity_visible(p: dict[str, Any], allowed: set[str] | None) -> bool:
        return allowed is None or bool(set(p.get("channels", ())) & allowed)

    def _visible_forms(self, p: dict[str, Any], allowed: set[str] | None) -> tuple[str, list[str]]:
        """(display name, aliases) restricted to surface forms seen in allowed channels."""
        name, aliases = p["name"], list(p.get("aliases", []))
        if allowed is None:
            return name, aliases
        seen: set[str] = set()
        for entry in p.get("alias_src", []):
            ch, _, form = entry.partition("\t")
            if ch in allowed:
                seen.add(form)
        vis_aliases = [a for a in aliases if a in seen]
        if name in seen or not seen:
            return name, vis_aliases
        first = next(iter(vis_aliases), None) or sorted(seen)[0]
        return first, [a for a in vis_aliases if a != first]

    def _visible_norm_forms(self, n: Node, allowed: set[str] | None) -> list[str]:
        """NFKC/case-folded name + aliases the caller may see (used for matching, so hidden forms never match)."""
        name, aliases = self._visible_forms(n.properties, allowed)
        return [nfkc(f).casefold() for f in [name, *aliases]]

    def _entity_view(self, n: Node, allowed: set[str] | None, group: str) -> dict[str, Any]:
        """Entity as shown to a caller restricted to ``allowed`` channels.

        An entity stores one summary line per source channel (``summaries``); a restricted caller gets only the
        lines of allowed channels. With no visible line the summary is derived at query time from the visible,
        currently valid facts (``summary_source="derived"``).
        """
        p = n.properties
        name, aliases = self._visible_forms(p, allowed)
        entries = self._entries(p)
        if allowed is None:
            summary, source = p.get("summary", ""), "stored"
        elif entries:
            summary, source = " / ".join(t for ch, t in entries.items() if ch in allowed), "stored"
        elif set(p.get("summary_channels", [])) <= allowed:  # legacy single-summary entity
            summary, source = p.get("summary", ""), "stored"
        else:
            summary, source = "", "stored"
        if not summary:  # e.g. every summary line was dropped by a forget: derive one from the valid facts
            summary, source = self._derived_summary(p["id"], allowed, group), "derived"
        return {
            "id": p["id"],
            "name": name,
            "type": p.get("type"),
            "aliases": aliases,
            "summary": summary,
            "summary_source": source,
            "created_at": self._local(p.get("created_at")),
            "updated_at": self._local(p.get("updated_at")),
        }

    def _derived_summary(self, entity_id: str, allowed: set[str] | None, group: str, n: int = 4) -> str:
        now = self._now()
        facts = [
            f
            for f in self._facts_of_entity(entity_id, group)
            if self._fact_visible(f.properties, allowed) and _fact_status(f.properties, now) == "valid"
        ]
        facts.sort(key=lambda f: f["valid_at"], reverse=True)
        return " / ".join(f["fact"] for f in facts[:n])

    def _facts_of_entity(self, entity_id: str, group: str) -> list[Relationship]:
        """Facts touching an entity in either direction (attribute self-loops match both, deduplicated by id)."""
        out: dict[str, Relationship] = {}
        for cy in (
            "MATCH (e:Entity {id:$id})-[f:FACT]->() RETURN f",
            "MATCH ()-[f:FACT]->(e:Entity {id:$id}) RETURN f",
        ):
            for r in self.db.query(cy, {"id": entity_id}):
                if r["f"].get("group_id") == group:
                    out[r["f"]["id"]] = r["f"]
        return list(out.values())

    # -- fact hits ---------------------------------------------------------------------------------

    def _episode_refs(self, group: str, ids: Iterable[str], allowed: set[str] | None) -> dict[str, EpisodeRef]:
        out: dict[str, EpisodeRef] = {}
        for i, n in self._nodes("Episode", ids, group).items():
            if allowed is not None and n.get("channel") not in allowed:
                continue
            out[i] = EpisodeRef(
                id=i,
                author=n.get("author"),
                channel=n.get("channel", ""),
                occurred_at=self._local(n.get("occurred_at")),
                source_ref=n.get("source_ref"),
                excerpt=short(n.get("content", ""), 200),
            )
        return out

    def _to_hits(
        self,
        group: str,
        facts: list[tuple[Relationship, float]],
        allowed: set[str] | None,
        at: str,
        max_episodes: int = 3,
    ) -> list[FactHit]:
        if not facts:
            return []
        ent_ids = {x for f, _ in facts for x in (f.get("subject_id"), f.get("object_id")) if x}
        ents = self._nodes("Entity", ent_ids, group)
        eps = self._episode_refs(group, [e for f, _ in facts for e in f.get("episode_ids", [])], allowed)
        # superseded_by / supersedes links, only towards facts the caller may see
        link_ids: list[str] = []
        for f, _ in facts:
            link_ids += [f["superseded_by"]] if f.get("superseded_by") else []
            link_ids += f.get("supersedes", [])
        linked = self._facts_by_id(link_ids, group)

        def link_visible(i: str | None) -> bool:
            return i in linked and self._fact_visible(linked[i].properties, allowed)

        out: list[FactHit] = []
        for f, score in facts:

            def ref(eid: str | None) -> dict[str, Any] | None:
                if not eid or eid not in ents:
                    return None
                return {"id": eid, "name": self._visible_forms(ents[eid].properties, allowed)[0]}

            out.append(
                FactHit(
                    id=f["id"],
                    fact=f["fact"],
                    relation=f.get("relation", ""),
                    subject=ref(f.get("subject_id")) or {"id": f.get("subject_id"), "name": "?"},
                    object=ref(f.get("object_id")),
                    valid_at=self._local(f.get("valid_at")),
                    invalid_at=self._local(f.get("invalid_at")),
                    created_at=self._local(f.get("created_at")),
                    expired_at=self._local(f.get("expired_at")),
                    status=_fact_status(f.properties, at),
                    confidence=float(f.get("confidence", 1.0)),
                    modality=f.get("modality", "definite"),
                    score=score,
                    episodes=[eps[e] for e in f.get("episode_ids", []) if e in eps][:max_episodes],
                    superseded_by=f.get("superseded_by") if link_visible(f.get("superseded_by")) else None,
                    supersedes=[i for i in f.get("supersedes", []) if link_visible(i)],
                )
            )
        return out

    # -- search ------------------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        group_id: str | None = None,
        channels: Sequence[str] | None = None,
        as_of: str | datetime | None = None,
        center_entity: str | None = None,
        limit: int = 10,
        include_invalid: bool = False,
        include_future: bool = False,
    ) -> list[FactHit]:
        """Hybrid temporal fact search.

        Candidates come from (1) vector search over fact sentences, (2) trigram fulltext over facts,
        (3) entity matching (name / alias / vector) followed by 1-2 hop graph expansion over facts;
        they are fused with reciprocal-rank fusion and optionally re-ranked by graph distance to
        ``center_entity``.

        * ``channels``: allowed source channels (``None`` = all, ``[]`` = nothing). A fact is returned only
          if at least one of the episodes asserting it comes from an allowed channel; episode citations
          and entity names/summaries in the result are likewise restricted.
        * ``as_of``: point in time (default now). Only facts with ``valid_at <= as_of`` and
          (``invalid_at`` is null or ``> as_of``) are returned. ``include_invalid`` also returns facts that
          had already ended (``status="ended"``), ``include_future`` facts not yet started
          (``status="scheduled"``).
        """
        hits, _ = self._search(
            query, group_id, channels, as_of, center_entity, limit, include_invalid, include_future
        )
        return hits

    def _search(
        self,
        query: str,
        group_id: str | None,
        channels: Sequence[str] | None,
        as_of: str | datetime | None,
        center_entity: str | None,
        limit: int,
        include_invalid: bool,
        include_future: bool,
    ) -> tuple[list[FactHit], list[Node]]:
        group = self._group(group_id)
        allowed = _channel_set(channels)
        if allowed is not None and not allowed:
            return [], []
        at = self._as_of(as_of)
        qvec = self._embed_query(query)
        terms = query_terms(query)

        def ok(f: Relationship) -> bool:
            if f.get("group_id") != group or not self._fact_visible(f.properties, allowed):
                return False
            st = _fact_status(f.properties, at)
            return st == "valid" or (st == "ended" and include_invalid) or (st == "scheduled" and include_future)

        pool: dict[str, Relationship] = {}
        lists: list[tuple[float, list[str]]] = []
        # (1) vector over facts
        if qvec is not None:
            k = max(60, limit * 6)
            ids = []
            for n, _ in self.db.vector_search(FACT_INDEX, qvec, k=k):
                if ok(n):
                    pool[n["id"]] = n
                    ids.append(n["id"])
            lists.append((1.0, ids))
        # (2) fulltext over facts
        q = fts_or_query(terms)
        if q:
            ids = []
            for n, _ in self.db.fulltext_search(FACT_FT, q, k=max(40, limit * 4)):
                if ok(n):
                    pool[n["id"]] = n
                    ids.append(n["id"])
            lists.append((0.8, ids))
        # (3) entity match + graph expansion
        matched = self._match_entities(group, allowed, query, terms, qvec, center_entity, limit=5)
        if matched:
            hop1_ids, hop1_facts, neighbour_ids = self._expand(group, [m.get("id") for m in matched], ok)
            self._rank_by_similarity(hop1_facts, qvec)
            pool.update({f["id"]: f for f in hop1_facts})
            lists.append((0.9, [f["id"] for f in hop1_facts]))
            if neighbour_ids:
                _, hop2_facts, _ = self._expand(group, neighbour_ids[:8], ok, skip=set(hop1_ids))
                self._rank_by_similarity(hop2_facts, qvec)
                pool.update({f["id"]: f for f in hop2_facts})
                lists.append((0.5, [f["id"] for f in hop2_facts]))
        scores = _rrf(lists)
        if center_entity:
            center = self._resolve_entity_ref(group, allowed, center_entity)
            if center is not None:
                dist = self._distances(group, center["id"], max_depth=3)
                for fid, s in list(scores.items()):
                    f = pool[fid]
                    d = min((dist.get(x, 99) for x in (f.get("subject_id"), f.get("object_id")) if x), default=99)
                    if d < 99:
                        scores[fid] = s + 0.02 / (1 + d)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        hits = self._to_hits(group, [(pool[i], s) for i, s in ranked], allowed, at)
        return hits, matched

    def _as_of(self, as_of: str | datetime | None) -> str:
        if as_of is None:
            return self._now()
        dt = parse_time(as_of, self.tz)
        if dt is None:
            raise ValueError(f"cannot parse as_of: {as_of!r}")
        return to_iso(dt)

    def _rank_by_similarity(self, facts: list[Relationship], qvec: list[float] | None) -> None:
        if qvec is None or not facts:
            facts.sort(key=lambda f: f.get("valid_at", ""), reverse=True)
            return
        ix = next(i for i in self.db.vector_indexes() if i.name == FACT_INDEX)
        sims: dict[str, float] = {}
        for f in facts:
            v = self.db._vec_read(ix, f.id)
            sims[f["id"]] = cosine(qvec, v) if v else -1.0
        facts.sort(key=lambda f: -sims[f["id"]])

    def _expand(
        self, group: str, entity_ids: Sequence[str | None], ok: Callable[[Relationship], bool], skip: set[str] | None = None
    ) -> tuple[list[str], list[Relationship], list[str]]:
        """Facts touching the given entities (passing ``ok``) and the ids of neighbouring entities."""
        facts: dict[str, Relationship] = {}
        neigh: Counter[str] = Counter()
        base = {e for e in entity_ids if e}
        for eid in base:
            for f in self._facts_of_entity(eid, group):
                if skip and f["id"] in skip:
                    continue
                if not ok(f):
                    continue
                facts[f["id"]] = f
                for x in (f.get("subject_id"), f.get("object_id")):
                    if x and x not in base:
                        neigh[x] += 1
        return list(facts), list(facts.values()), [k for k, _ in neigh.most_common()]

    def _distances(self, group: str, start: str, max_depth: int = 3) -> dict[str, int]:
        """BFS distance (in FACT hops, both directions; self-loops add nothing) from an entity.

        Validity is deliberately not filtered: the distance only re-ranks candidates, and ended facts still
        connect entities for ``as_of`` / ``include_invalid`` searches.
        """
        dist = {start: 0}
        frontier = [start]
        for d in range(1, max_depth + 1):
            nxt: list[str] = []
            for eid in frontier:
                for r in self.db.query(
                    "MATCH (a:Entity {id:$id})-[:FACT]-(b:Entity) RETURN DISTINCT b.id AS id", {"id": eid}
                ):
                    if r["id"] not in dist:
                        dist[r["id"]] = d
                        nxt.append(r["id"])
            frontier = nxt
            if not frontier:
                break
        return dist

    # -- entities ----------------------------------------------------------------------------------

    def _match_entities(
        self,
        group: str,
        allowed: set[str] | None,
        query: str,
        terms: list[str],
        qvec: list[float] | None,
        center: str | None,
        limit: int = 5,
    ) -> list[Node]:
        """Entities the query is about: name/alias mention, trigram FTS and vector similarity."""
        scored: dict[str, float] = defaultdict(float)
        nodes: dict[str, Node] = {}
        qn = nfkc(query).casefold()
        if terms:
            for r in self.db.query(
                "MATCH (e:Entity {group_id:$g}) WHERE any(t IN $ts WHERE e.search_text CONTAINS t) RETURN e",
                {"g": group, "ts": terms},
            ):
                e = r["e"]
                if not self._entity_visible(e.properties, allowed):
                    continue
                forms = self._visible_norm_forms(e, allowed)
                text = " | ".join(forms)
                score = max((len(t) for t in terms if t in text), default=0)
                if not score:  # matched only through a surface form the caller may not see
                    continue
                if any(len(f) >= 3 and f in qn for f in forms):  # the query literally names this entity
                    score += 10
                elif any(f in terms for f in forms):
                    score += 5
                scored[e["id"]] += float(score)
                nodes[e["id"]] = e
        if qvec is not None:
            for e, s in self.db.vector_search(ENTITY_INDEX, qvec, k=6):
                # an entity embedding encodes its summary: only trust it when every source channel is allowed
                if (
                    e.get("group_id") == group
                    and self._entity_visible(e.properties, allowed)
                    and (allowed is None or set(e.get("summary_channels", [])) <= allowed)
                    and s >= 0.66
                ):
                    scored[e["id"]] += 2.0 * s
                    nodes.setdefault(e["id"], e)
        if center:
            c = self._resolve_entity_ref(group, allowed, center)
            if c is not None:
                scored[c["id"]] += 20.0
                nodes.setdefault(c["id"], c)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
        return [nodes[i] for i, _ in ranked]

    def _resolve_entity_ref(self, group: str, allowed: set[str] | None, ref: str) -> Node | None:
        """Entity by id, exact name/alias, or best fuzzy match; ``None`` if not found / not visible."""
        n = self._entity_node(group, ref)
        if n is not None:
            return n if self._entity_visible(n.properties, allowed) else None
        key = norm_key(ref)
        if key:
            rows = self.db.query(
                "MATCH (e:Entity {group_id:$g}) WHERE $k IN e.alias_keys RETURN e", {"g": group, "k": key}
            )
            for r in rows:
                e = r["e"]
                if self._entity_visible(e.properties, allowed):
                    name, aliases = self._visible_forms(e.properties, allowed)
                    if key in {norm_key(f) for f in [name, *aliases]}:
                        return e
        found = self._entity_candidates(group, allowed, ref, limit=1)
        return found[0] if found else None

    def _entity_candidates(self, group: str, allowed: set[str] | None, query: str, limit: int) -> list[Node]:
        terms = query_terms(query) or [nfkc(query).casefold().strip()]
        qvec = self._embed_query(query)
        matched = self._match_entities(group, allowed, query, terms, qvec, None, limit=limit)
        if matched:
            return matched
        q = fts_or_query([nfkc(query).casefold()])
        out = []
        if q:
            needle = nfkc(query).casefold().strip()
            for n, _ in self.db.fulltext_search(ENTITY_FT, q, k=limit * 3):
                if (
                    n.get("group_id") == group
                    and self._entity_visible(n.properties, allowed)
                    and any(needle in f for f in self._visible_norm_forms(n, allowed))
                ):
                    out.append(n)
        return out[:limit]

    def search_entities(
        self,
        query: str,
        *,
        group_id: str | None = None,
        channels: Sequence[str] | None = None,
        limit: int = 10,
        types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find entities by name / alias / description (channel-restricted views, see :meth:`get_entity`)."""
        group = self._group(group_id)
        allowed = _channel_set(channels)
        if allowed is not None and not allowed:
            return []
        found = self._entity_candidates(group, allowed, query, limit=max(limit * 2, 10))
        views = [self._entity_view(n, allowed, group) for n in found if not types or n.get("type") in types]
        return views[:limit]

    def get_entity(
        self,
        name_or_id: str,
        *,
        group_id: str | None = None,
        channels: Sequence[str] | None = None,
        as_of: str | datetime | None = None,
        max_facts: int = 100,
    ) -> dict[str, Any] | None:
        """Entity profile with its full fact timeline (including superseded facts) and related entities.

        Facts are sorted by ``valid_at``; each carries ``status`` relative to ``as_of`` (default now):
        ``valid`` / ``scheduled`` / ``ended``. Returns ``None`` if unknown or not visible.
        """
        group = self._group(group_id)
        allowed = _channel_set(channels)
        if allowed is not None and not allowed:
            return None
        node = self._resolve_entity_ref(group, allowed, name_or_id)
        if node is None:
            return None
        at = self._as_of(as_of)
        facts = [f for f in self._facts_of_entity(node["id"], group) if self._fact_visible(f.properties, allowed)]
        facts.sort(key=lambda f: (f.get("valid_at", ""), f.get("created_at", "")))
        hits = self._to_hits(group, [(f, 0.0) for f in facts[-max_facts:]], allowed, at)
        related: Counter[str] = Counter()
        for h in hits:
            for side in (h.subject, h.object):
                if side and side["id"] != node["id"]:
                    related[side["id"]] += 1
        rel_nodes = self._nodes("Entity", related, group)
        return {
            "entity": self._entity_view(node, allowed, group),
            "as_of": self._local(at),
            "timeline": [h.to_dict() for h in hits],
            "related_entities": [
                {"id": i, "name": self._visible_forms(rel_nodes[i].properties, allowed)[0], "facts": c}
                for i, c in related.most_common(15)
                if i in rel_nodes
            ],
        }

    def get_episodes(
        self,
        *,
        group_id: str | None = None,
        channels: Sequence[str] | None = None,
        entity: str | None = None,
        channel: str | None = None,
        query: str | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Raw source episodes (newest first), optionally about an entity / matching a text / in a time window."""
        group = self._group(group_id)
        allowed = _channel_set(channels)
        if allowed is not None and not allowed:
            return []
        s = iso_or_none(since, self.tz)
        u = iso_or_none(until, self.tz)
        nodes: list[Node]
        if entity:
            ent = self._resolve_entity_ref(group, allowed, entity)
            if ent is None:
                return []
            nodes = [
                r["e"]
                for r in self.db.query("MATCH (e:Episode)-[:MENTIONS]->(x:Entity {id:$id}) RETURN e", {"id": ent["id"]})
            ]
        elif query:
            q = fts_or_query(query_terms(query))
            nodes = [n for n, _ in self.db.fulltext_search(EPISODE_FT, q, k=limit * 5)] if q else []
        else:
            nodes = [r["e"] for r in self.db.query("MATCH (e:Episode {group_id:$g}) RETURN e", {"g": group})]
        out = []
        for n in nodes:
            if n.get("group_id") != group or (allowed is not None and n.get("channel") not in allowed):
                continue
            if channel and n.get("channel") != channel:
                continue
            at = n.get("occurred_at", "")
            if (s and at < s) or (u and at >= u):
                continue
            out.append(n)
        out.sort(key=lambda n: n.get("occurred_at", ""), reverse=True)
        return [
            {
                "id": n["id"],
                "author": n.get("author"),
                "channel": n.get("channel"),
                "source": n.get("source"),
                "occurred_at": self._local(n.get("occurred_at")),
                "source_ref": n.get("source_ref"),
                "content": n.get("content"),
            }
            for n in out[:limit]
        ]

    # -- answer ------------------------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        group_id: str | None = None,
        channels: Sequence[str] | None = None,
        as_of: str | datetime | None = None,
        center_entity: str | None = None,
        limit: int = 20,
    ) -> Answer:
        """Retrieve facts for ``question`` and let the LLM answer in Japanese, citing sources and validity.

        Retrieval includes ended facts (history) and, when ``as_of`` is not given, planned facts that have
        not started yet; each is labelled with its ``status`` so the model can tell them apart.
        """
        llm = self._need_llm()
        group = self._group(group_id)
        allowed = _channel_set(channels)
        hits, ents = self._search(
            question, group, channels, as_of, center_entity, limit, include_invalid=True, include_future=as_of is None
        )
        if not hits:
            return Answer(question, "該当する記憶が見つかりませんでした。", [], [], self._local(self._as_of(as_of)) if as_of else None)
        payload = {
            "now": local_with_weekday(self._now(), self.tz),
            "as_of": self._local(self._as_of(as_of)) if as_of is not None else None,
            "question": question,
            "entities": [
                {k: v for k, v in self._entity_view(e, allowed, group).items() if k in ("name", "type", "summary")}
                for e in ents[:4]
            ],
            "facts": [
                {
                    "id": f"F{i + 1}",
                    "fact": h.fact,
                    "status": h.status,
                    "valid_at": h.valid_at[:16] if h.valid_at else None,
                    "invalid_at": h.invalid_at[:16] if h.invalid_at else None,
                    "modality": h.modality,
                    "sources": [
                        {"author": e.author, "at": (e.occurred_at or "")[:16], "channel": e.channel} for e in h.episodes[:2]
                    ],
                }
                for i, h in enumerate(hits)
            ],
        }
        self._llm_calls += 1
        text = llm.complete_text(
            prompts.ANSWER_SYSTEM, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), task="answer"
        )
        views = [self._entity_view(e, allowed, group) for e in ents[:4]]
        return Answer(question, text, hits, views, payload["as_of"])

    # ================================================================== FORGET

    # Deleting is physical: rows are removed (``PRAGMA secure_delete`` overwrites them), the fulltext indexes are
    # merged and the WAL is truncated, so forgotten text does not linger in the file. Facts derived from deleted
    # episodes go with them (a fact asserted by several episodes only loses the deleted ones as sources); a
    # deleted fact gives the facts it superseded back their validity.

    def _need_writable(self) -> None:
        if self.db.read_only:
            raise SqgraphError("memory database is read-only")

    def forget_episode(
        self,
        episode_id: str | None = None,
        *,
        source: str | None = None,
        source_ref: str | None = None,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> ForgetResult:
        """Delete one source episode (by ``episode_id``, or by ``source_ref`` / ``(source, source_ref)``).

        Facts asserted only by it are deleted too (facts with other sources keep those and lose this one),
        superseded facts are restored and entities left without mentions / facts are removed. An unknown
        episode yields an empty result.
        """
        if episode_id is None and source_ref is None:
            raise ValueError("give episode_id or source_ref (optionally with source)")
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            ids: set[str] = set()
            if episode_id is not None:
                n = self._one("MATCH (e:Episode {id:$id}) RETURN e", {"id": episode_id})
                if n is not None and n.get("group_id") == group:
                    ids.add(n["id"])
            else:
                cy = "MATCH (e:Episode {group_id:$g, source_ref:$r}) RETURN e"
                params = {"g": group, "r": source_ref}
                if source is not None:
                    cy, params = "MATCH (e:Episode {group_id:$g, source_ref:$r, source:$s}) RETURN e", {**params, "s": source}
                ids = {r["e"]["id"] for r in self.db.query(cy, params)}
            return self._forget(group, episode_ids=ids, dry_run=dry_run)

    def forget_fact(self, fact_id: str, *, group_id: str | None = None, dry_run: bool = False) -> ForgetResult:
        """Delete one fact (its episodes stay). Facts it superseded become valid again."""
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            return self._forget(group, fact_ids={fact_id}, dry_run=dry_run)

    def forget_entity(
        self,
        entity: str,
        *,
        with_episodes: bool = True,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> ForgetResult:
        """Delete an entity (``entity``: id, or exact name / alias), every fact it is part of and, with
        ``with_episodes`` (default), every episode that mentions it.

        Only exact names count (no fuzzy matching, since this deletes); a name shared by several entities
        raises ``ValueError`` listing their ids. An unknown entity yields an empty result.
        """
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            nodes = self._entities_by_ref(group, entity)
            if len(nodes) > 1:
                options = ", ".join(f"{n['id']} ({n.get('type')} {n['name']!r})" for n in nodes)
                raise ValueError(f"{entity!r} is ambiguous; use an entity id: {options}")
            return self._forget(
                group, entity_ids={n["id"] for n in nodes}, entity_episodes=with_episodes, dry_run=dry_run
            )

    def forget_range(
        self,
        *,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        channel: str | None = None,
        source: str | None = None,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> ForgetResult:
        """Delete the episodes with ``since <= occurred_at < until`` (naive times are in ``tz``), optionally
        only those of a ``channel`` / ``source``. At least one filter is required."""
        if since is None and until is None and channel is None and source is None:
            raise ValueError("forget_range needs at least one of since / until / channel / source")
        bounds: dict[str, str | None] = {}
        for label, value in (("since", since), ("until", until)):
            dt = parse_time(value, self.tz) if value is not None else None
            if value is not None and dt is None:
                raise ValueError(f"cannot parse {label}: {value!r}")
            bounds[label] = to_iso(dt) if dt else None
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            ids = set()
            for r in self.db.query("MATCH (e:Episode {group_id:$g}) RETURN e", {"g": group}):
                n = r["e"]
                at = n.get("occurred_at", "")
                if bounds["since"] is not None and at < bounds["since"]:
                    continue
                if bounds["until"] is not None and at >= bounds["until"]:
                    continue
                if channel is not None and n.get("channel") != channel:
                    continue
                if source is not None and n.get("source") != source:
                    continue
                ids.add(n["id"])
            return self._forget(group, episode_ids=ids, dry_run=dry_run)

    def purge(self, *, group_id: str | None = None, dry_run: bool = False) -> ForgetResult:
        """Delete every episode, entity and fact of the group."""
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            return self._forget(group, everything=True, dry_run=dry_run)

    def _entities_by_ref(self, group: str, ref: str) -> list[Node]:
        """Entities whose id, or whose name / alias (normalised), equals ``ref`` (no fuzzy matching)."""
        n = self._entity_node(group, ref)
        if n is not None:
            return [n]
        key = norm_key(ref)
        if not key:
            return []
        rows = self.db.query("MATCH (e:Entity {group_id:$g}) WHERE $k IN e.alias_keys RETURN e", {"g": group, "k": key})
        return [r["e"] for r in rows]

    def _forget(
        self,
        group: str,
        *,
        episode_ids: Iterable[str] = (),
        fact_ids: Iterable[str] = (),
        entity_ids: Iterable[str] = (),
        entity_episodes: bool = False,
        everything: bool = False,
        dry_run: bool = False,
    ) -> ForgetResult:
        """Shared cascade of all forget operations; the whole change is planned in memory, then applied in one
        transaction (so ``dry_run`` reports exactly what a real run would do)."""
        db = self.db
        g = {"g": group}
        eps: dict[str, Node] = {r["e"]["id"]: r["e"] for r in db.query("MATCH (e:Episode {group_id:$g}) RETURN e", g)}
        ents: dict[str, Node] = {r["x"]["id"]: r["x"] for r in db.query("MATCH (x:Entity {group_id:$g}) RETURN x", g)}
        facts: dict[str, Relationship] = {
            r["f"]["id"]: r["f"] for r in db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f", g)
        }
        ent_eps: dict[str, set[str]] = defaultdict(set)  # entity -> episodes mentioning it
        ep_ents: dict[str, set[str]] = defaultdict(set)
        for r in db.query("MATCH (e:Episode {group_id:$g})-[:MENTIONS]->(x:Entity) RETURN e.id AS e, x.id AS x", g):
            ent_eps[r["x"]].add(r["e"])
            ep_ents[r["e"]].add(r["x"])
        ent_facts: dict[str, set[str]] = defaultdict(set)  # entity -> facts it is an endpoint of
        for fid, f in facts.items():
            for x in (f.get("subject_id"), f.get("object_id")):
                if x:
                    ent_facts[x].add(fid)

        # -- what goes ------------------------------------------------------------------------------
        if everything:
            E, N, F = set(eps), set(ents), set(facts)
        else:
            E = {i for i in episode_ids if i in eps}
            N = {i for i in entity_ids if i in ents}
            if entity_episodes:
                E |= {e for x in N for e in ent_eps[x] if e in eps}
            F = {i for i in fact_ids if i in facts} | {fid for x in N for fid in ent_facts[x]}
        changes: dict[str, dict[str, Any]] = defaultdict(dict)  # surviving fact id -> property updates (None = remove)

        def getp(fid: str, key: str, default: Any = None) -> Any:
            if key in changes.get(fid, ()):
                v = changes[fid][key]
                return default if v is None else v
            return facts[fid].get(key, default)

        # facts lose their deleted sources; facts left without any source go too
        for fid, f in facts.items():
            sources = f.get("episode_ids", [])
            if fid in F or E.isdisjoint(sources):
                continue
            rest = [x for x in sources if x not in E]
            if not rest:
                F.add(fid)
                continue
            upd: dict[str, Any] = {"episode_ids": rest}
            chans = list(dict.fromkeys(eps[x].get("channel") for x in rest if x in eps and eps[x].get("channel")))
            if chans:
                upd["channels"] = chans
            changes[fid].update(upd)

        # -- supersede links ------------------------------------------------------------------------
        restored: list[str] = []
        superseded: dict[str, list[str]] = defaultdict(list)  # fact -> facts it superseded
        for fid, f in facts.items():
            if f.get("superseded_by"):
                superseded[f["superseded_by"]].append(fid)

        def surviving_superseder(fid: str) -> str | None:
            """The fact that (transitively, through deleted facts) superseded ``fid``, if it survives."""
            seen: set[str] = set()
            nxt = facts[fid].get("superseded_by")
            while nxt in F and nxt not in seen:
                seen.add(nxt)
                nxt = facts[nxt].get("superseded_by")
            return nxt if nxt in facts and nxt not in F else None

        for fid, f in facts.items():  # deleted ids vanish from the lists of the surviving facts
            if fid not in F and any(x in F for x in f.get("supersedes", [])):
                changes[fid]["supersedes"] = [x for x in f["supersedes"] if x not in F] or None
        for did in sorted(F):
            d = facts[did]
            nxt = surviving_superseder(did)
            for oid in superseded.get(did, []):
                if oid in F:
                    continue
                o = facts[oid]
                if nxt is not None:
                    # chain old -> deleted -> newer: the old fact now ends where the newer one begins
                    changes[oid].update(
                        {"invalid_at": max(facts[nxt].get("valid_at", ""), o.get("valid_at", "")), "superseded_by": nxt}
                    )
                    changes[nxt]["supersedes"] = list(dict.fromkeys([*(getp(nxt, "supersedes") or []), oid]))
                    continue
                upd = {"expired_at": None, "superseded_by": None}
                if o.get("invalid_at") == max(d.get("valid_at", ""), o.get("valid_at", "")):
                    upd["invalid_at"] = None  # set by the supersession (a stated end time is kept)
                changes[oid].update(upd)
                restored.append(oid)

        # -- entities -------------------------------------------------------------------------------
        touched: set[str] = {x for e in E for x in ep_ents[e]}
        for fid in F:
            touched |= {x for x in (facts[fid].get("subject_id"), facts[fid].get("object_id")) if x}
        touched = (touched - N) & set(ents)
        deleted_ents = set(N)
        updates: dict[str, dict[str, Any]] = {}
        embed_texts: dict[str, str] = {}
        now = self._now()
        for x in sorted(touched):
            remaining_eps = ent_eps[x] - E
            remaining_facts = ent_facts[x] - F
            if not remaining_eps and not remaining_facts:
                deleted_ents.add(x)
                continue
            upd, text = self._recompute_entity(
                ents[x], now, [eps[e] for e in remaining_eps if e in eps], [e for e in ent_eps[x] if e in E and e in eps],
                [facts[f] for f in remaining_facts], [facts[f] for f in ent_facts[x] & F], getp, eps
            )
            updates[x] = upd
            embed_texts[x] = text

        result = ForgetResult(
            episodes_deleted=sorted(E),
            facts_deleted=sorted(F),
            facts_restored=sorted(set(restored)),
            entities_deleted=sorted(deleted_ents),
            entities_updated=sorted(updates),
            dry_run=dry_run,
        )
        if dry_run or result.empty:
            return result

        # -- apply ----------------------------------------------------------------------------------
        vectors: list[list[float]] | None = None
        if self.embedder is not None and embed_texts:
            try:
                vectors = self.embedder.embed(list(embed_texts.values()))
            except Exception:  # forgetting must not depend on the embedding API being reachable
                vectors = None
        if vectors is not None:
            for x, v in zip(embed_texts, vectors):
                updates[x]["embedding"] = v
        else:
            for upd in updates.values():
                upd["embedding"] = None  # a stale vector could still encode forgotten text
        with db.transaction():
            for fid, ch in changes.items():
                if fid not in F and ch:
                    db.update_edge(facts[fid].id, ch)
            for fid in F:
                db.delete_edge(facts[fid].id)
            for e in E:
                db.delete_node(eps[e].id, detach=True)
            for x in deleted_ents:
                db.delete_node(ents[x].id, detach=True)
            for x, upd in updates.items():
                db.update_node(ents[x].id, upd)
        db.optimize_fulltext([ENTITY_FT, FACT_FT, EPISODE_FT])
        db.checkpoint(truncate=True)
        return result

    def _recompute_entity(
        self,
        node: Node,
        now: str,
        remaining_eps: list[Node],
        deleted_eps: list[str],
        remaining_facts: list[Relationship],
        deleted_facts: list[Relationship],
        getp: Callable[..., Any],
        eps: dict[str, Node],
    ) -> tuple[dict[str, Any], str]:
        """Property updates of a surviving entity after forgetting, and the text its embedding is built from.

        Dropped: the per-channel summary lines of the channels of the deleted episodes / facts (they may state
        what was forgotten) and the aliases that no remaining mentioning episode uses any more.
        """
        p = node.properties
        name = p["name"]
        drop = {eps[e].get("channel") for e in deleted_eps} | {c for f in deleted_facts for c in f.get("channels", [])}
        entries = {c: t for c, t in self._entries(p).items() if c not in drop}
        upd: dict[str, Any] = {"updated_at": now}
        if self._entries(p):
            upd.update(
                summaries=[f"{c}\t{t}" for c, t in entries.items()],
                summary=" / ".join(entries.values()),
                summary_channels=list(entries),
            )
        elif p.get("summary") and set(p.get("summary_channels", [])) & drop:  # legacy single summary
            upd.update(summary="", summary_channels=[])
        texts = [
            (e.get("text_norm") or nfkc(e.get("content", "")).casefold(), nfkc(e.get("author") or "").casefold(), e.get("channel"))
            for e in remaining_eps
        ]
        mention_channels = {c for _, _, c in texts}

        def seen_in(form: str) -> set[str]:
            f = nfkc(form).casefold().strip()
            return {c for t, a, c in texts if f and (f in t or f in a)}

        aliases = [a for a in p.get("aliases", []) if seen_in(a)]
        forms = {name: mention_channels, **{a: seen_in(a) for a in aliases}}
        chans = mention_channels | {c for f in remaining_facts for c in f.get("channels", [])}
        old_channels = list(p.get("channels", []))
        upd.update(
            aliases=aliases,
            alias_keys=sorted({k for k in (norm_key(x) for x in [name, *aliases]) if k}),
            alias_src=[
                e
                for e in p.get("alias_src", [])
                if e.partition("\t")[2] in forms and e.partition("\t")[0] in forms[e.partition("\t")[2]]
            ],
            search_text=nfkc("\n".join([name, *aliases])).casefold(),
            channels=[c for c in old_channels if c in chans] + sorted(chans - set(old_channels)),
        )
        if entries:
            body = " / ".join(entries.values())
        else:  # the same fallback as a restricted view: the currently valid facts
            valid = []
            for f in remaining_facts:
                eff = {k: getp(f["id"], k) for k in ("valid_at", "invalid_at")}
                if _fact_status(eff, now) == "valid":
                    valid.append((eff["valid_at"] or "", f["fact"]))
            body = " / ".join(t for _, t in sorted(valid, reverse=True)[:4])
        return upd, short(f"{name}: {body}", 400)

    # -- re-embedding ----------------------------------------------------------------------------------

    def reembed(
        self,
        group_id: str | None = None,
        *,
        batch_size: int = 64,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Recompute every entity and fact embedding with the current embedder and record its identity.

        ``group_id=None`` (default) covers all groups; with a group only that group's vectors are replaced, which is
        refused when the model / dimensions change while other groups exist (their vectors would be left behind).
        Texts are the ones ingestion embeds: an entity's ``name: summaries`` (else its currently valid facts), a
        fact's sentence. Nothing is written until every text has been embedded, then the new vectors (and, when the
        dimension changed, freshly created ``entity_emb`` / ``fact_emb`` indexes) are applied in one transaction, so
        a failing embedding server leaves the database untouched. Holds the write lock for the whole run; open the
        :class:`Memory` with ``allow_embedder_change=True`` when the identity differs. ``progress(kind, done, total)``
        is called after every batch (``kind``: ``entities`` / ``facts``). Returns the counts and the old / new identity.
        """
        emb = self._need_embedder()
        self._need_writable()
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        ident = self.embedder_identity(emb)
        with self._write_guard():
            db = self.db
            flt = "{group_id:$g}" if group_id else ""
            params: dict[str, Any] = {"g": group_id} if group_id else {}
            previous = self.embedding_identity()
            indexes = {i.name: i for i in db.vector_indexes()}
            new_dims = int(emb.dimensions)
            recreate = any(indexes[n].dimensions != new_dims for n in (ENTITY_INDEX, FACT_INDEX) if n in indexes)
            if group_id:
                others = db.query("MATCH (e:Entity) WHERE e.group_id <> $g RETURN count(e) AS c", params)[0]["c"]
                new_model = previous is not None and ident is not None and previous["model"] != ident["model"]
                if others and (recreate or new_model):
                    raise SqgraphError(
                        f"cannot re-embed only group {group_id!r}: the model or dimension changes and {others} entities "
                        "of other groups would keep vectors of the old model; re-embed all groups (no group_id)"
                    )
            entities = [r["e"] for r in db.query(f"MATCH (e:Entity {flt}) RETURN e", params)]
            facts = [r["f"] for r in db.query(f"MATCH ()-[f:FACT {flt}]->() RETURN f", params)]
            now = self._now()
            by_entity: dict[str, list[Relationship]] = defaultdict(list)
            for f in facts:
                for x in {f.get("subject_id"), f.get("object_id")} - {None}:
                    by_entity[x].append(f)

            def batched(kind: str, texts: list[str]) -> list[list[float]]:
                out: list[list[float]] = []
                for i in range(0, len(texts), batch_size):
                    out += emb.embed(texts[i : i + batch_size])
                    if progress is not None:
                        progress(kind, len(out), len(texts))
                for v in out:
                    if len(v) != new_dims:
                        raise SqgraphError(f"the embedder returned a {len(v)}-dimensional vector, expected {new_dims}")
                return out

            e_vecs = batched("entities", [self._reembed_entity_text(n, by_entity.get(n["id"], ()), now) for n in entities])
            f_vecs = batched("facts", [f["fact"] for f in facts])
            with db.transaction():
                if recreate:
                    for name in (ENTITY_INDEX, FACT_INDEX):
                        db.drop_index(name, if_exists=True)
                    db.create_vector_index(ENTITY_INDEX, "Entity", "embedding", dimensions=new_dims)
                    db.create_relationship_vector_index(FACT_INDEX, FACT_TYPE, "embedding", dimensions=new_dims)
                for n, v in zip(entities, e_vecs):
                    db.set_node_vector(n.id, "embedding", v)
                for f, v in zip(facts, f_vecs):
                    db.set_edge_vector(f.id, "embedding", v)
                if ident is not None:
                    self._store_embedding_identity(ident)
        return {
            "group_id": group_id,
            "entities": len(entities),
            "facts": len(facts),
            "model": ident["model"] if ident else None,
            "dimensions": new_dims,
            "endpoint": ident["endpoint"] if ident else None,
            "previous": previous,
            "indexes_recreated": recreate,
        }

    def _reembed_entity_text(self, node: Node, facts: Iterable[Relationship], now: str) -> str:
        """The text an entity's embedding is built from (the same as at ingestion / after forgetting)."""
        p = node.properties
        entries = self._entries(p)
        if entries:
            body = " / ".join(entries.values())
        elif p.get("summary"):
            body = p["summary"]
        else:  # no summary: the currently valid facts, newest first
            valid = [(f.get("valid_at") or "", f["fact"]) for f in facts if _fact_status(f.properties, now) == "valid"]
            body = " / ".join(t for _, t in sorted(valid, reverse=True)[:4])
        return short(f"{p['name']}: {body}", 400)

    # -- misc ----------------------------------------------------------------------------------------

    def stats(self, group_id: str | None = None) -> dict[str, Any]:
        """Node counts by label, fact counts (``FACT`` edges) and edge counts by type (optionally for one group)."""
        g = {"g": group_id} if group_id else {}
        flt = "{group_id:$g}" if group_id else ""
        nodes = {
            lb: self.db.query(f"MATCH (n:{lb} {flt}) RETURN count(n) AS c", g)[0]["c"]
            for lb in ("Episode", "Entity")
        }
        facts = self.db.query(f"MATCH ()-[f:FACT {flt}]->() RETURN f.invalid_at IS NULL AS valid, count(f) AS c", g)
        rel_counts = {
            r["t"]: r["c"]
            for r in self.db.query("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC")
        }
        return {
            "nodes": nodes,
            "facts_valid": sum(r["c"] for r in facts if r["valid"]),
            "facts_invalidated": sum(r["c"] for r in facts if not r["valid"]),
            "relationships": rel_counts,
        }
