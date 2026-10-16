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
``channels`` (allowed channels, ``None`` = unrestricted). See :meth:`Memory.search` and ``docs/memory.md``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import zlib
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
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

log = logging.getLogger("sqgraph.memory.core")

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

# --- relevance gate of :meth:`Memory.search` -----------------------------------------------------------------------
# Cosine similarity between the query and a fact below which a candidate has no semantic evidence. Embedding models
# differ a lot in how similarities are scaled (text-embedding-3-small: related ~0.3-0.7, unrelated ~0.0-0.3; nomic-embed
# without task prefixes: related ~0.6-0.8, unrelated ~0.45-0.7), so the default depends on the model name (substring
# match on the case-folded model id). Values marked "measured" come from a calibration on ~40 Japanese personal facts
# and ~30 queries (see the commit notes); "estimate" ones are typical ranges, not measured here.
MIN_SIMILARITY_BY_MODEL: tuple[tuple[str, float], ...] = (
    ("text-embedding-3-small", 0.34),  # measured
    ("text-embedding-3-large", 0.32),  # estimate (3-large is scaled like 3-small, slightly lower)
    ("text-embedding-ada-002", 0.78),  # estimate (ada-002 cosines rarely fall below ~0.7)
    ("nomic-embed-text", 0.72),  # measured (LM Studio id text-embedding-nomic-embed-text-v1.5, Ollama nomic-embed-text)
    ("qwen3-embedding", 0.46),  # measured on 0.6B without a query instruction (LM Studio id text-embedding-qwen3-embedding-0.6b)
    ("bge-m3", 0.45),  # estimate
)
MIN_SIMILARITY_UNKNOWN_MODEL = 0.30  # an embedder with an unrecognised model name: low, so it only trims the clearly unrelated
MIN_SIMILARITY_NO_MODEL = 0.25  # embedders that do not name a model (FakeEmbedder: hashed character trigrams)
EXPAND_SLACK = 0.05  # facts reached through the graph (weakly) need only ``min_similarity - EXPAND_SLACK``
RELATIVE_MARGIN = 0.08  # vector-only candidates must be within this of the best candidate's similarity

NO_MATCH_TEXT = "No matching memories found."  # ``Answer.text`` when nothing matched (the LLM is not called)
# prepare_episode (agent extraction) is more permissive than search: a missed fact may be a missed contradiction
PREPARE_ENTITY_SLACK = 0.05  # candidate entities (vector match): min_similarity - this
PREPARE_FACT_SLACK = 0.10  # related facts: min_similarity - this ...
PREPARE_MIN_RELATED = 6  # ... but the best this-many facts around the message are always offered
HUB_MIN_FACTS = 5  # an entity with >= this many facts that make up >= half of all facts is a "hub" (e.g. the user)
# Words that stand for "the user" in personal memory: they occur in nearly every fact, so they are no evidence
PERSONA_TERMS = frozenset(
    {"ユーザー", "ユーザ", "user", "users", "the user", "me", "my", "私", "わたし", "僕", "ぼく", "俺", "おれ", "自分", "あなた", "本人"}
)


# --- relation vocabulary ------------------------------------------------------------------------------------------
# Passive relations are stored in their active direction (subject and object swapped), so one kind of link has one
# name: "repo OWNED_BY Bob" is written "Bob OWNS repo". Only relations whose active form is unambiguous are listed.
RELATION_INVERSES: dict[str, str] = {
    "OWNED_BY": "OWNS",
    "MANAGED_BY": "MANAGES",
    "MAINTAINED_BY": "MAINTAINS",
    "OPERATED_BY": "OPERATES",
    "DEVELOPED_BY": "DEVELOPS",
    "USED_BY": "USES",
    "PREFERRED_BY": "PREFERS",
    "REQUIRED_BY": "REQUIRES",
    "LED_BY": "LEADS",
    "EMPLOYED_BY": "EMPLOYS",
    "HOSTED_BY": "HOSTS",
    "FUNDED_BY": "FUNDS",
    "SUPPORTED_BY": "SUPPORTS",
    "CREATED_BY": "CREATED",
    "BUILT_BY": "BUILT",
    "WRITTEN_BY": "WROTE",
    "REVIEWED_BY": "REVIEWED",
    "REQUESTED_BY": "REQUESTED",
    "APPROVED_BY": "APPROVED",
    "DECIDED_BY": "DECIDED",
}
KNOWN_RELATIONS_IN_CONTEXT = 40  # relations offered to the extracting agent (most used first)
STALE_PLAN_DAYS = 14  # review(): a planned / tentative fact this long past its start is worth confirming


def normalize_relation(raw: str | None) -> str:
    """``UPPER_SNAKE_CASE`` form of a relation name (``RELATES_TO`` when empty)."""
    s = re.sub(r"\W+", "_", nfkc(raw or "").strip()).strip("_").upper()
    s = re.sub(r"_+", "_", s)
    return s or "RELATES_TO"


def canonical_relation(raw: str | None, has_object: bool) -> tuple[str, bool]:
    """``(relation, swap)``: the stored name of a relation and whether subject and object must be swapped for it
    (a passive relation with an object is turned around, see ``RELATION_INVERSES``)."""
    rel = normalize_relation(raw)
    if has_object and rel in RELATION_INVERSES:
        return RELATION_INVERSES[rel], True
    return rel, False


NO_HITS_HINT = (
    "nothing passed the relevance gate (a fact needs a keyword hit, a named entity or similarity >= min_similarity): "
    "try other keywords or names, or lower min_similarity"
)


def default_min_similarity(model: str | None) -> float:
    """Default relevance threshold (cosine) for an embedding ``model`` name (``None``: an embedder without a model name)."""
    if not model:
        return MIN_SIMILARITY_NO_MODEL
    m = str(model).casefold()
    for key, value in MIN_SIMILARITY_BY_MODEL:
        if key in m:
            return value
    return MIN_SIMILARITY_UNKNOWN_MODEL


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
    similarity: float | None = None  # cosine similarity of the fact to the query (None: no vector evidence)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["episodes"] = [e.to_dict() for e in self.episodes]
        d["score"] = round(self.score, 5)
        d["similarity"] = None if self.similarity is None else round(self.similarity, 4)
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
    warnings: list[str] = field(default_factory=list)  # agent extraction: items dropped / merged by the server

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["warnings"] = list(self.warnings)
        return d


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
    if ia is not None and (ia <= at or (va is not None and ia <= va)):  # ia <= va: cancelled before it began
        return "ended"
    if va is not None and va > at:
        return "scheduled"
    return "valid"


# --------------------------------------------------------------------------- agent extraction tokens

TOKEN_PREFIX = "sqm1"  # format version 1 of the prepare_episode token
_TOKEN_MAX_BYTES = 8 * 1024 * 1024
_FACT_ID_RE = re.compile(r"^fact_[0-9a-f]{12}$")
_MODALITIES = ("definite", "planned", "tentative")
_DEFAULT_CONFIDENCE = {"definite": 1.0, "planned": 0.7, "tentative": 0.5}


def _encode_token(payload: dict[str, Any]) -> str:
    """``sqm1.<urlsafe base64 of zlib(JSON)>.<blake2b checksum>``: self-contained, the checksum only catches copy errors."""
    raw = zlib.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)
    body = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{TOKEN_PREFIX}.{body}.{hashlib.blake2b(raw, digest_size=4).hexdigest()}"


def _decode_token(token: Any) -> dict[str, Any]:
    bad = "invalid token (it must be passed back exactly as returned by prepare / add_memory)"
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"{bad}: empty")
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        raise ValueError(f"{bad}: unknown format")
    try:
        raw = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
    except Exception:
        raise ValueError(f"{bad}: not base64") from None
    if hashlib.blake2b(raw, digest_size=4).hexdigest() != parts[2]:
        raise ValueError(f"{bad}: checksum mismatch (truncated or altered)")
    try:
        d = zlib.decompressobj().decompress(raw, _TOKEN_MAX_BYTES)
        data = json.loads(d.decode("utf-8"))
    except Exception:
        raise ValueError(f"{bad}: corrupt payload") from None
    if not isinstance(data, dict) or data.get("v") != 1:
        raise ValueError(f"{bad}: unsupported version {data.get('v') if isinstance(data, dict) else None!r}")
    try:
        out = {
            "group_id": str(data["g"]),
            "source": str(data["s"]),
            "channel": str(data["c"]),
            "author": None if data.get("a") is None else str(data["a"]),
            "occurred_at": str(data["t"]),
            "source_ref": None if data.get("r") is None else str(data["r"]),
            "content": str(data["x"]),
        }
    except KeyError as exc:
        raise ValueError(f"{bad}: missing field {exc.args[0]!r}") from None
    if parse_time(out["occurred_at"], timezone.utc) is None:
        raise ValueError(f"{bad}: bad time")
    return out


def _content_key(text: str) -> str:
    """NFKC / case-folded text without separators: the form entity ``alias_keys`` are searched in."""
    return re.sub(r"[\s_\-・/／.]+", "", nfkc(text).casefold())


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

    min_similarity:
        Cosine similarity a fact needs to count as semantically related to a query (the relevance gate of
        :meth:`search`). ``None`` = a default for the embedding model of the database (:func:`default_min_similarity`).
        ``expand_slack`` lowers that bar for facts reached through the entity graph; ``relative_margin`` drops
        vector-only candidates that are further than this below the best candidate.

    answer_prompt:
        System prompt of :meth:`answer`; ``None`` = :data:`sqgraph.memory.prompts.ANSWER_SYSTEM` (answers in English).
        The user message it receives is a JSON object with ``now`` (local time with weekday), ``as_of`` (or null),
        ``question``, ``entities`` (name, type, summary; up to 4) and ``facts`` (each with ``id`` "F1".., ``fact``,
        ``status`` valid|ended|scheduled, ``valid_at``, ``invalid_at``, ``modality`` and ``sources``
        [{author, at, channel}]). E.g. ``Memory(..., answer_prompt=ANSWER_SYSTEM.replace("in English", "in Japanese"))``.

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
        min_similarity: float | None = None,
        expand_slack: float = EXPAND_SLACK,
        relative_margin: float = RELATIVE_MARGIN,
        answer_prompt: str | None = None,
    ) -> None:
        if answer_prompt is not None and not answer_prompt.strip():
            raise ValueError("answer_prompt must not be empty (use None for the default prompt)")
        self.answer_prompt: str = prompts.ANSWER_SYSTEM if answer_prompt is None else answer_prompt
        self._allow_embedder_change = allow_embedder_change
        self.min_similarity = self._check_similarity(min_similarity)
        self.expand_slack = float(expand_slack)
        self.relative_margin = float(relative_margin)
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

    def _embed_query_or_none(self, text: str) -> list[float] | None:
        """Like :meth:`_embed_query`, but an embedder that fails (endpoint down) means "no vector evidence"."""
        try:
            return self._embed_query(text)
        except Exception as exc:  # noqa: BLE001 - any provider error: degrade to fulltext + entity search
            log.warning("query embedding failed (%s: %s): searching without vectors", type(exc).__name__, str(exc)[:200])
            return None

    # -- relevance gate helpers ----------------------------------------------------------------------

    @staticmethod
    def _check_similarity(value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"min_similarity must be a number between -1 and 1 (cosine similarity), got {value!r}")
        return float(value)

    def effective_min_similarity(self, override: float | None = None) -> float:
        """The relevance threshold in force: the per-call ``override``, else the constructor value, else the default of the
        embedding model recorded in the database (or of the configured embedder)."""
        v = self._check_similarity(override)
        if v is None:
            v = self.min_similarity
        if v is not None:
            return v
        if self.embedder is None:
            return MIN_SIMILARITY_UNKNOWN_MODEL
        if not getattr(self.embedder, "model", None):
            return MIN_SIMILARITY_NO_MODEL
        ident = self.embedding_identity()
        return default_min_similarity((ident or {}).get("model") or getattr(self.embedder, "model", None))

    def _fact_total(self, group: str) -> int:
        rows = self.db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN count(f) AS n", {"g": group})
        return int(rows[0]["n"]) if rows else 0

    def _generic_terms(self, group: str, terms: Sequence[str], total: int) -> set[str]:
        """Query terms that carry no evidence: persona words ("ユーザー", "user", ...) and terms occurring in at least half
        of all facts (>= 3 facts). In a personal memory nearly every fact mentions the user, so a fulltext hit on such a
        term says nothing about relevance."""
        generic = {t for t in terms if t in PERSONA_TERMS}
        if total >= 4:
            for t in [t for t in terms if t not in generic and len(t) >= 3][:20]:
                rows = self.db.query(
                    "MATCH ()-[f:FACT {group_id:$g}]->() WHERE f.text_norm CONTAINS $t RETURN count(f) AS n", {"g": group, "t": t}
                )
                df = int(rows[0]["n"]) if rows else 0
                if df >= 3 and df * 2 >= total:
                    generic.add(t)
        return generic

    def _fact_sims(self, facts: Iterable[Relationship], qvec: list[float] | None) -> dict[str, float | None]:
        """Cosine similarity of the stored fact vectors to ``qvec`` (``None`` for a fact without a vector / no ``qvec``)."""
        facts = list(facts)
        if qvec is None or not facts:
            return {f["id"]: None for f in facts}
        ix = next(i for i in self.db.vector_indexes() if i.name == FACT_INDEX)
        out: dict[str, float | None] = {}
        for f in facts:
            v = self.db._vec_read(ix, f.id)
            out[f["id"]] = cosine(qvec, v) if v else None
        return out

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

    # -- agent extraction (two phases, no server-side LLM) -------------------------------------------------
    # prepare_episode(): read-only; returns the context + instructions for the calling agent (the LLM) and a
    # stateless token. commit_episode(): the agent's extraction is validated and written with the same
    # structures / write path (_commit, _write_fact, _invalidate) as the OpenAI pipeline, so supersession
    # semantics are identical.

    @staticmethod
    def read_token(token: str) -> dict[str, Any]:
        """Decode a :meth:`prepare_episode` token (``group_id``, ``source``, ``channel``, ``author``,
        ``occurred_at`` (UTC ISO), ``source_ref``, ``content``). ``ValueError`` if it is malformed or altered."""
        return _decode_token(token)

    def _episode_by_ref(self, group: str, source: str, source_ref: str) -> Node | None:
        return self._one(
            "MATCH (e:Episode {group_id:$g, source:$s, source_ref:$r}) RETURN e",
            {"g": group, "s": source, "r": source_ref},
        )

    def prepare_episode(
        self,
        content: str,
        *,
        group_id: str | None = None,
        source: str = "api",
        channel: str = "default",
        author: str | None = None,
        occurred_at: str | datetime | None = None,
        source_ref: str | None = None,
        max_entities: int = 20,
        max_facts: int = 30,
    ) -> dict[str, Any]:
        """Phase 1 of agent extraction: no LLM, no write, no write lock.

        Returns ``{"already_stored": True, "episode_id"}`` if ``(source, source_ref)`` was ingested, else
        ``{"already_stored": False, "token", "instructions", "schema", "context", "next"}``. ``context`` holds the
        message, the author ref, the preceding episodes of the channel, the existing entities the message may be
        about (name / alias matches, fulltext, vector search) and the currently valid facts around them. Only
        information visible in ``channel`` is returned (the same ACL as the pipeline's LLM input). The calling
        agent writes the extraction JSON (``schema``) and passes it with ``token`` to :meth:`commit_episode`.
        """
        group = self._group(group_id)
        if source_ref:
            dup = self._episode_by_ref(group, source, source_ref)
            if dup is not None:
                return {"already_stored": True, "episode_id": dup["id"]}
        occurred = parse_time(occurred_at, self.tz) if occurred_at else self._clock()
        if occurred is None:
            raise ValueError(f"cannot parse occurred_at: {occurred_at!r}")
        occurred_iso = to_iso(occurred)
        text = clean_message(content or "")
        if not text:
            raise ValueError("content must not be empty")
        allowed = {channel}
        qvec = self._embed_query(short(text, 1500))
        author_name, author_aliases = person_forms(author) if author else (None, [])

        author_node: Node | None = None
        if author_name:
            aw = self._author_entity(author_name, author_aliases, author)
            self._resolve_entities(group, [aw], None)
            author_node = aw.node
        author_visible = author_node is not None and self._entity_visible(author_node.properties, allowed)

        cands = self._agent_candidate_nodes(
            group, channel, text, qvec, author_node if author_visible else None, max(0, max_entities)
        )
        around = list(cands)
        if author_visible and author_node is not None and author_node["id"] not in {n["id"] for n in cands}:
            around.append(author_node)
        related = self._agent_related_facts(group, channel, text, qvec, occurred_iso, around, max(0, max_facts))
        views = [self._entity_view(n, allowed, group) for n in cands]
        token = _encode_token(
            {"v": 1, "g": group, "s": source, "c": channel, "a": author, "t": occurred_iso, "r": source_ref, "x": text}
        )
        return {
            "already_stored": False,
            "token": token,
            "instructions": prompts.AGENT_EXTRACTION_INSTRUCTIONS,
            "schema": prompts.AGENT_EXTRACTION_SCHEMA,
            "context": {
                "message": {
                    "content": text,
                    "author": author_name,
                    "occurred_at": local_with_weekday(occurred_iso, self.tz),
                    "channel": channel,
                },
                "author_ref": (
                    {"ref": "author", "name": author_name, "existing_id": author_node["id"] if author_visible else None}
                    if author_name
                    else None
                ),
                "recent": self._recent_context(group, channel, occurred_iso),
                "candidate_entities": [
                    {
                        "id": v["id"],
                        "name": v["name"],
                        "type": v["type"],
                        "aliases": v["aliases"][:6],
                        "summary": short(v["summary"], 120),
                    }
                    for v in views
                ],
                "related_facts": related,
                "known_relations": self._known_relations(group, channel),
            },
            "next": (
                "Write the extraction JSON (see `schema`, `instructions`) and call commit_memory(token, extraction) "
                "(CLI: sqgraph-memory commit --token ... < extraction.json) with this exact token."
            ),
        }

    def _agent_candidate_nodes(
        self, group: str, channel: str, text: str, qvec: list[float] | None, author_node: Node | None, limit: int
    ) -> list[Node]:
        """Existing entities (visible in ``channel``) the message may be about, best first (at most ``limit``)."""
        allowed = {channel}
        scored: dict[str, float] = defaultdict(float)
        nodes: dict[str, Node] = {}

        def add(n: Node, s: float) -> None:
            if n.get("group_id") == group and self._entity_visible(n.properties, allowed):
                scored[n["id"]] = max(scored[n["id"]], s)
                nodes.setdefault(n["id"], n)

        if author_node is not None:
            add(author_node, 1000.0)
        # (a) a name / alias occurring in the message (substring of the separator-free text)
        ckey = _content_key(text)
        for r in self.db.query("MATCH (e:Entity {group_id:$g}) WHERE $c IN e.channels RETURN e", {"g": group, "c": channel}):
            e = r["e"]
            name, aliases = self._visible_forms(e.properties, allowed)
            best = max(
                (
                    len(k)
                    for k in (norm_key(f) for f in [name, *aliases])
                    if k and k in ckey and (len(k) >= 3 or not k.isascii())
                ),
                default=0,
            )
            if best:
                add(e, 100.0 + best)
        # (b) trigram fulltext / term containment / vector similarity of the whole message
        terms = query_terms(text)[:60]
        q = fts_or_query(terms)
        if q:
            for rank, (n, _) in enumerate(self.db.fulltext_search(ENTITY_FT, q, k=max(limit, 5) * 2)):
                add(n, 50.0 - rank * 0.1)
        # entity vectors: more permissive than search (a missed candidate may hide a contradiction), but not "any entity"
        floor = self.effective_min_similarity() - PREPARE_ENTITY_SLACK
        for rank, n in enumerate(
            self._match_entities(group, allowed, text, terms, qvec, None, limit=max(limit, 5), min_similarity=floor)
        ):
            add(n, 40.0 - rank * 0.1)
        if qvec is not None:
            for n, s in self.db.vector_search(ENTITY_INDEX, qvec, k=max(limit, 5)):
                # an entity embedding encodes its summaries: only trust it when every source channel is visible here
                if (2.0 * s - 1.0) >= floor and set(n.get("summary_channels", [])) <= allowed:
                    add(n, 20.0 * s)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
        return [nodes[i] for i, _ in ranked]

    def _agent_related_facts(
        self,
        group: str,
        channel: str,
        text: str,
        qvec: list[float] | None,
        occurred_iso: str,
        entities: list[Node],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Facts valid / scheduled at ``occurred_iso`` and visible in ``channel``: those touching ``entities`` (when
        they have some evidence of relevance, see below) plus vector / fulltext matches of the message, fused with RRF."""
        if limit <= 0:
            return []
        allowed = {channel}

        def ok(f: Relationship) -> bool:
            return (
                f.get("group_id") == group
                and self._fact_visible(f.properties, allowed)
                and _fact_status(f.properties, occurred_iso) in ("valid", "scheduled")
            )

        # Not every fact touching the author / a hub entity (the user) is related to the message: those need evidence
        # (similarity above a permissive floor, or a fulltext hit on an informative term). Facts of an entity the
        # message names stay. Deliberately more permissive than search: a missed fact may be a missed contradiction.
        floor = self.effective_min_similarity() - PREPARE_FACT_SLACK
        total = self._fact_total(group)
        terms = query_terms(text)[:60]
        generic = self._generic_terms(group, terms, total)
        ckey = _content_key(text)
        pool: dict[str, Relationship] = {}
        touching: dict[str, Relationship] = {}
        named: set[str] = set()
        for n in entities:
            allf = self._facts_of_entity(n["id"], group)
            nm, als = self._visible_forms(n.properties, allowed)
            forms = [norm_key(f) for f in [nm, *als]]
            literal = any(k and k in ckey and k not in PERSONA_TERMS and (len(k) >= 3 or not k.isascii()) for k in forms)
            hub = len(allf) >= HUB_MIN_FACTS and len(allf) * 2 >= total
            for f in allf:
                if ok(f):
                    touching[f["id"]] = f
                    if literal and not hub:
                        named.add(f["id"])
        sims: dict[str, float | None] = self._fact_sims(touching.values(), qvec)
        pool.update(touching)
        vec_ids: list[str] = []
        if qvec is not None:
            for f, sc in self.db.vector_search(FACT_INDEX, qvec, k=max(40, limit * 3)):
                if ok(f):
                    sims[f["id"]] = 2.0 * sc - 1.0
                    if len(vec_ids) < PREPARE_MIN_RELATED or sims[f["id"]] >= floor:
                        pool[f["id"]] = f
                        vec_ids.append(f["id"])
        ft_ids: list[str] = []
        q = fts_or_query([t for t in terms if t not in generic])
        if q:
            for f, _ in self.db.fulltext_search(FACT_FT, q, k=max(40, limit * 3)):
                if ok(f):
                    pool[f["id"]] = f
                    ft_ids.append(f["id"])
        ftset = set(ft_ids)
        if qvec is None:
            ranked_touch = sorted(touching.values(), key=lambda f: f.get("valid_at", ""), reverse=True)
        else:
            ranked_touch = sorted(touching.values(), key=lambda f: -(sims[f["id"]] if sims.get(f["id"]) is not None else -2.0))  # type: ignore[operator]
        around = [
            f
            for i, f in enumerate(ranked_touch)
            if f["id"] in named
            or f["id"] in ftset
            or sims.get(f["id"]) is None
            or sims[f["id"]] >= floor  # type: ignore[operator]
            or i < PREPARE_MIN_RELATED
        ]
        lists: list[tuple[float, list[str]]] = [(1.0, [f["id"] for f in around[: limit * 2]])]
        if vec_ids:
            lists.append((0.8, vec_ids))
        if ft_ids:
            lists.append((0.8, ft_ids))
        ranked = sorted(_rrf(lists).items(), key=lambda kv: -kv[1])[:limit]
        facts = [pool[i] for i, _ in ranked]
        ents = self._nodes("Entity", {x for f in facts for x in (f.get("subject_id"), f.get("object_id")) if x}, group)

        def ref(eid: str | None) -> dict[str, Any] | None:
            if not eid or eid not in ents:
                return None
            return {"id": eid, "name": self._visible_forms(ents[eid].properties, allowed)[0]}

        return [
            {
                "id": f["id"],
                "fact": f["fact"],
                "relation": f.get("relation", ""),
                "subject": ref(f.get("subject_id")) or {"id": f.get("subject_id"), "name": "?"},
                "object": ref(f.get("object_id")),
                "valid_at": self._local(f.get("valid_at")),
                "invalid_at": self._local(f.get("invalid_at")),
                "status": _fact_status(f.properties, occurred_iso),
            }
            for f in facts
        ]

    def commit_episode(self, token: str, extraction: dict[str, Any] | str) -> EpisodeResult:
        """Phase 2 of agent extraction: validate the agent's ``extraction`` and write the episode, entities and facts.

        Runs under the write guard (in-process lock + cross-process file lock). The token (see
        :meth:`prepare_episode`) carries group, source, channel, author, time and text, so nothing is kept on the
        server between the phases. ``ValueError`` lists every problem of a malformed ``extraction``. The database may
        have changed since ``prepare_episode``: stale ``duplicate_of`` / ``contradicts`` ids are dropped with a
        message in ``EpisodeResult.warnings``, and entities marked new (``existing_id`` null) are still merged into an
        existing entity with the same name / alias. Needs no LLM; embeddings use the configured embedder (none: no vectors).
        """
        if self.db.read_only:
            raise SqgraphError("memory database is read-only")
        tok = _decode_token(token)
        group, source, channel = tok["group_id"], tok["source"], tok["channel"]
        with self._write_guard():
            if tok["source_ref"]:
                dup = self._episode_by_ref(group, source, tok["source_ref"])
                if dup is not None:
                    return EpisodeResult(dup["id"], skipped=True)
            work, facts, warnings = self._extraction_to_work(group, tok, extraction)
            self._finalize_summaries(work, channel)
            if self.embedder is not None:  # one batched call for the entities needing a vector and the new facts
                slots: list[tuple[str, Any]] = [("entity", w) for w in work if w.node is None or w.chan_summary is not None]
                slots += [("fact", f) for f in facts]
                texts = [self._entity_embed_text(o, channel) if k == "entity" else o["fact"] for k, o in slots]
                if texts:
                    for (k, o), v in zip(slots, self.embedder.embed(texts)):
                        if k == "entity":
                            o.embedding = v
                        else:
                            o["embedding"] = v
            res = self._commit(
                group,
                source,
                channel,
                tok["author"],
                tok["occurred_at"],
                self._now(),
                tok["content"],
                tok["source_ref"],
                new_id("ep"),
                work,
                facts,
            )
            res.warnings = warnings
            return res

    def _extraction_to_work(
        self, group: str, tok: dict[str, Any], extraction: Any
    ) -> tuple[list[_WorkEntity], list[dict[str, Any]], list[str]]:
        """Validate an agent extraction and build what ``_understand`` produces: ``_WorkEntity`` list + fact dicts."""
        if isinstance(extraction, str):
            try:
                extraction = json.loads(extraction)
            except ValueError as exc:
                raise ValueError(f"extraction is not valid JSON: {exc}") from None
        if not isinstance(extraction, dict):
            raise ValueError('extraction must be a JSON object like {"entities": [...], "facts": [...]}')
        errors: list[str] = []
        warnings: list[str] = []
        channel, occurred_iso = tok["channel"], tok["occurred_at"]
        ents_in, facts_in = extraction.get("entities") or [], extraction.get("facts") or []
        if not isinstance(ents_in, list):
            errors.append("entities must be an array")
            ents_in = []
        if not isinstance(facts_in, list):
            errors.append("facts must be an array")
            facts_in = []

        # -- entities ---------------------------------------------------------------------------
        work: list[_WorkEntity] = []
        refs: dict[str, _WorkEntity] = {}
        author_name, author_aliases = person_forms(tok["author"]) if tok["author"] else (None, [])
        if author_name:
            aw = self._author_entity(author_name, author_aliases, tok["author"])
            self._resolve_entities(group, [aw], None)
            work.append(aw)
            refs["author"] = aw

        def opt_str(v: Any, where: str) -> str | None:
            if v is None:
                return None
            if not isinstance(v, str):
                errors.append(f"{where} must be a string or null")
                return None
            return v.strip() or None

        for i, e in enumerate(ents_in):
            where = f"entities[{i}]"
            if not isinstance(e, dict):
                errors.append(f"{where} must be an object")
                continue
            ref, name = opt_str(e.get("ref"), f"{where}.ref"), opt_str(e.get("name"), f"{where}.name")
            if ref is None:
                errors.append(f"{where}.ref is required")
            if name is None:
                errors.append(f"{where}.name is required")
            existing_id = opt_str(e.get("existing_id"), f"{where}.existing_id")
            typ = opt_str(e.get("type"), f"{where}.type")
            summary = opt_str(e.get("summary"), f"{where}.summary")
            raw_aliases = e.get("aliases")
            if raw_aliases is not None and not (isinstance(raw_aliases, list) and all(isinstance(a, str) for a in raw_aliases)):
                errors.append(f"{where}.aliases must be an array of strings")
                raw_aliases = None
            if ref is None or name is None:
                continue
            if ref == "author":
                if author_name:
                    warnings.append(f'{where}: the author is added automatically; ignored (reference it as "author")')
                else:
                    errors.append(f'{where}.ref "author" is reserved (this episode has no author)')
                continue
            if ref in refs:
                errors.append(f"{where}.ref {ref!r} is used twice")
                continue
            if typ is not None and typ not in prompts.ENTITY_TYPES:
                warnings.append(f"{where}: unknown type {typ!r} stored as 'Other'")
                typ = "Other"
            forms = list(dict.fromkeys([name, *[a.strip() for a in (raw_aliases or []) if a.strip()]]))
            keys = {k for k in (norm_key(f) for f in forms) if k}
            node: Node | None = None
            if existing_id:
                node = self._one("MATCH (e:Entity {id:$id}) RETURN e", {"id": existing_id})
                if node is None:
                    errors.append(f"{where}.existing_id {existing_id!r} does not exist (use an id from candidate_entities, or null)")
                    continue
                if node.get("group_id") != group:
                    errors.append(f"{where}.existing_id {existing_id!r} belongs to another group")
                    continue
            else:  # safety net: the agent may have missed a candidate; the deterministic exact / alias match decides
                found = (
                    self._exact_matches(group, keys, typ)
                    if typ
                    else list({n["id"]: n for n in (*self._exact_matches(group, keys, "Person"), *self._exact_matches(group, keys, "Other"))}.values())
                )
                if len(found) == 1:
                    node = found[0]
                    warnings.append(f"{where}: {name!r} matched the existing entity {node['name']!r} ({node['id']}) by name: merged")
                elif len(found) > 1:
                    warnings.append(f"{where}: {name!r} matches several existing entities; stored as a new entity (use existing_id)")
            prev = next(
                (
                    w
                    for w in work
                    if (node is not None and w.node is not None and w.node["id"] == node["id"]) or w.match_keys & keys
                ),
                None,
            )
            if prev is not None:  # the same thing declared twice (or the author itself): merge into the first
                prev.forms += forms
                prev.match_keys |= keys
                if summary and prev.new_summary is None:
                    prev.new_summary = short(summary, 300)
                refs[ref] = prev
                continue
            w = _WorkEntity(
                len(work),
                node["name"] if node is not None else name,
                node.get("type", "Other") if node is not None else (typ or "Other"),
                "",
                node=node,
                forms=forms,
                match_keys=keys,
            )
            if summary:
                w.new_summary = short(summary, 300)
            work.append(w)
            refs[ref] = w

        # -- facts ------------------------------------------------------------------------------
        facts: list[dict[str, Any]] = []

        def entity_ref(v: Any, where: str, nullable: bool) -> _WorkEntity | None:
            if v is None or v == "":
                if not nullable:
                    errors.append(f"{where} is required")
                return None
            if not isinstance(v, str) or v not in refs:
                hint = ' ("author" is the message author)' if author_name else ""
                errors.append(f"{where} {v!r} is not a declared entity ref{hint}")
                return None
            return refs[v]

        def fact_ids(v: Any, where: str) -> list[str]:
            if v is None:
                return []
            vals = [v] if isinstance(v, str) else v
            if not isinstance(vals, list) or not all(isinstance(x, str) for x in vals):
                errors.append(f"{where} must be an array of fact ids")
                return []
            return [x.strip() for x in vals if x.strip()]

        def lookup_fact(fid: str, where: str) -> Relationship | None:
            """The existing fact, or ``None`` (a malformed / foreign-group id is an error, a vanished one a warning)."""
            if not _FACT_ID_RE.match(fid):
                errors.append(f"{where}: {fid!r} is not a fact id (copy ids from context.related_facts)")
                return None
            edge = self._fact_edge(fid)
            if edge is None:
                warnings.append(f"{where}: fact {fid} no longer exists; ignored")
                return None
            if edge.get("group_id") != group:
                errors.append(f"{where}: fact {fid} belongs to another group")
                return None
            return edge

        seen: set[str] = set()
        for j, f in enumerate(facts_in):
            where = f"facts[{j}]"
            if not isinstance(f, dict):
                errors.append(f"{where} must be an object")
                continue
            n_err = len(errors)
            subj = entity_ref(f.get("subject"), f"{where}.subject", False)
            obj = entity_ref(f.get("object"), f"{where}.object", True)
            fact_text = opt_str(f.get("fact"), f"{where}.fact")
            if fact_text is None:
                errors.append(f"{where}.fact is required")
            rel = normalize_relation(opt_str(f.get("relation"), f"{where}.relation"))
            times: list[str | None] = []
            for key in ("valid_at", "invalid_at"):
                raw = f.get(key)
                if raw is None or raw == "":
                    times.append(None)
                    continue
                dt = parse_time(raw, self.tz) if isinstance(raw, (str, datetime)) else None
                if dt is None:
                    errors.append(f"{where}.{key} {raw!r} cannot be parsed (use ISO 8601, e.g. 2026-09-17 or 2026-09-14T09:00)")
                    times.append(None)
                else:
                    times.append(to_iso(dt))
            va, ia = times[0] or occurred_iso, times[1]
            if ia is not None and ia <= va:
                warnings.append(f"{where}: invalid_at is not after valid_at; ignored")
                ia = None
            modality = f.get("modality")
            if modality is None or modality == "":
                modality = "definite"
            elif modality not in _MODALITIES:
                errors.append(f"{where}.modality {modality!r} must be one of {', '.join(_MODALITIES)}")
                modality = "definite"
            conf = f.get("confidence")
            if conf is None:
                conf = _DEFAULT_CONFIDENCE[modality]
            elif isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                errors.append(f"{where}.confidence {conf!r} must be a number between 0 and 1")
                conf = _DEFAULT_CONFIDENCE[modality]
            dup_ids = fact_ids(f.get("duplicate_of"), f"{where}.duplicate_of")
            if len(dup_ids) > 1:
                errors.append(f"{where}.duplicate_of must be a single fact id")
            duplicate_of: str | None = None
            if dup_ids:
                edge = lookup_fact(dup_ids[0], f"{where}.duplicate_of")
                if edge is not None:
                    if channel not in edge.get("channels", []):
                        warnings.append(
                            f"{where}.duplicate_of: fact {edge['id']} belongs to another channel; stored as a separate fact"
                        )
                    elif _fact_status(edge.properties, occurred_iso) == "ended":
                        warnings.append(f"{where}.duplicate_of: fact {edge['id']} has already ended; stored as a new fact")
                    else:
                        duplicate_of = edge["id"]
            contradicts: list[str] = []
            for cid in dict.fromkeys(fact_ids(f.get("contradicts"), f"{where}.contradicts")):
                edge = lookup_fact(cid, f"{where}.contradicts")
                if edge is None:
                    continue
                if edge.get("invalid_at") is not None:
                    warnings.append(f"{where}.contradicts: fact {cid} is no longer valid (already ended); left unchanged")
                else:
                    contradicts.append(cid)
            if duplicate_of and contradicts:
                warnings.append(f"{where}: a duplicate contradicts nothing; contradicts ignored")
                contradicts = []
            if len(errors) > n_err or subj is None or fact_text is None:
                continue
            if fact_text in seen:  # identical sentences within one message
                continue
            seen.add(fact_text)
            facts.append(
                {
                    "subject": subj.idx,
                    "object": obj.idx if obj is not None else None,
                    "fact": fact_text,
                    "relation": rel,
                    "valid_at": va,
                    "invalid_at": ia,
                    "modality": modality,
                    "confidence": float(conf),
                    "duplicate_of": duplicate_of,
                    "contradicts": contradicts,
                }
            )
        if errors:
            raise ValueError(
                f"invalid extraction ({len(errors)} problem{'s' if len(errors) != 1 else ''}):\n- " + "\n- ".join(errors)
            )
        return work, facts, warnings

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
                    "known_relations": self._known_relations(group, channel),
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
            rel = normalize_relation(f.get("relation"))
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
        relation, swap = canonical_relation(f["relation"], obj is not None and obj["id"] != subj["id"])
        if swap:
            subj, obj = obj, subj
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
            "relation": relation,
            "subject_id": subj["id"],
            "valid_at": va,
            "created_at": now,
            "channels": [channel],
            "episode_ids": [episode_id],
            "confidence": f["confidence"],
            "modality": f["modality"],
        }
        if f.get("embedding") is not None:
            props["embedding"] = f["embedding"]
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
        sims: dict[str, float | None] | None = None,
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
                    similarity=(sims or {}).get(f["id"]),
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
        min_similarity: float | None = None,
    ) -> list[FactHit]:
        """Hybrid temporal fact search.

        Candidates come from (1) vector search over fact sentences, (2) trigram fulltext over facts,
        (3) entity matching (name / alias / vector) followed by 1-2 hop graph expansion over facts.
        A relevance gate then keeps only candidates with evidence (see below) and the survivors are fused with
        reciprocal-rank fusion (``score``) and optionally re-ranked by graph distance to ``center_entity``. A query
        nothing relates to returns ``[]`` (or very few facts) instead of every fact in the memory.

        Relevance gate. For every candidate the cosine ``similarity`` to the query is computed from the stored fact
        vector (``None`` without an embedder / vector); ``min_similarity`` (default: per embedding model, see
        :func:`default_min_similarity`; per-call override) is its threshold. A candidate is kept if

        * (a) it is a fulltext hit on an informative query term (persona words such as "ユーザー" and terms in half of
          all facts do not count), or
        * (b) ``similarity >= min_similarity``, or
        * (c) it is a fact of an entity the query names (an entity name / alias occurs in the query, or ``center_entity``;
          an entity that is a "hub" of the memory, e.g. the user, only counts as a weak match), or
        * (d) it is a fact of a weakly matched entity (fulltext fragment / entity vector similarity), or one hop
          further from a named entity, and ``similarity >= min_similarity - expand_slack``. Two hops from a weak
          match only count through (a) / (b).

        Candidates that survive only through (b) / (d) are also dropped when ``similarity`` is more than
        ``relative_margin`` below the best candidate. Without a query vector (no embedder, endpoint down, fact without
        a vector) only (a), (c) and the facts of weakly matched entities remain.

        * ``channels``: allowed source channels (``None`` = all, ``[]`` = nothing). A fact is returned only
          if at least one of the episodes asserting it comes from an allowed channel; episode citations
          and entity names/summaries in the result are likewise restricted.
        * ``as_of``: point in time (default now). Only facts with ``valid_at <= as_of`` and
          (``invalid_at`` is null or ``> as_of``) are returned. ``include_invalid`` also returns facts that
          had already ended (``status="ended"``), ``include_future`` facts not yet started
          (``status="scheduled"``).
        """
        hits, _ = self._search(
            query, group_id, channels, as_of, center_entity, limit, include_invalid, include_future, min_similarity
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
        min_similarity: float | None = None,
    ) -> tuple[list[FactHit], list[Node]]:
        group = self._group(group_id)
        allowed = _channel_set(channels)
        if allowed is not None and not allowed:
            return [], []
        at = self._as_of(as_of)
        thr = self.effective_min_similarity(min_similarity)
        qvec = self._embed_query_or_none(query)
        terms = query_terms(query)
        total = self._fact_total(group)
        generic = self._generic_terms(group, terms, total)

        def ok(f: Relationship) -> bool:
            if f.get("group_id") != group or not self._fact_visible(f.properties, allowed):
                return False
            st = _fact_status(f.properties, at)
            return st == "valid" or (st == "ended" and include_invalid) or (st == "scheduled" and include_future)

        pool: dict[str, Relationship] = {}
        sims: dict[str, float | None] = {}  # cosine similarity to the query (None: no vector evidence)
        ft_hits: set[str] = set()  # fulltext hits on informative terms
        # how a fact was reached through entities: 1 strong hop-1, 2 weak hop-1, 3 strong hop-2, 4 weak hop-2
        tier: dict[str, int] = {}
        lists: list[tuple[float, list[str]]] = []
        # (1) vector over facts
        if qvec is not None:
            k = max(60, limit * 6)
            ids = []
            for n, s in self.db.vector_search(FACT_INDEX, qvec, k=k):
                if ok(n):
                    pool[n["id"]] = n
                    ids.append(n["id"])
                    sims[n["id"]] = 2.0 * s - 1.0  # cosine index: score = (1 + cos) / 2
            lists.append((1.0, ids))
        # (2) fulltext over facts (informative terms only)
        q = fts_or_query([t for t in terms if t not in generic])
        if q:
            ids = []
            for n, _ in self.db.fulltext_search(FACT_FT, q, k=max(40, limit * 4)):
                if ok(n):
                    pool[n["id"]] = n
                    ids.append(n["id"])
                    ft_hits.add(n["id"])
            lists.append((0.8, ids))
        # (3) entity match + graph expansion
        matched = self._match_entities_ex(
            group, allowed, query, terms, qvec, center_entity, limit=5, min_similarity=thr, hub_total=total
        )
        hop1_ids: list[str] = []
        hop2_ids: list[str] = []
        if matched:
            strong = [n["id"] for n, st in matched if st == "strong"]
            weak = [n["id"] for n, st in matched if st != "strong"]
            _, facts1s, neigh_s = self._expand(group, strong, ok)
            _, facts1w, neigh_w = self._expand(group, weak, ok)
            hop1 = {f["id"]: f for f in [*facts1w, *facts1s]}
            for f in facts1w:
                tier[f["id"]] = 2
            for f in facts1s:
                tier[f["id"]] = 1
            skip = set(hop1)
            _, facts2s, _ = self._expand(group, neigh_s[:8], ok, skip=skip)
            _, facts2w, _ = self._expand(group, neigh_w[:8], ok, skip=skip)
            hop2 = {f["id"]: f for f in [*facts2w, *facts2s]}
            for f in facts2w:
                tier[f["id"]] = 4
            for f in facts2s:
                tier[f["id"]] = 3
            pool.update(hop1)
            pool.update(hop2)
            hop1_ids, hop2_ids = list(hop1), list(hop2)
        # similarity of every candidate to the query
        sims.update(self._fact_sims([f for i, f in pool.items() if i not in sims], qvec))

        def ordered(ids: list[str]) -> list[str]:
            if qvec is None:
                return sorted(ids, key=lambda i: pool[i].get("valid_at", ""), reverse=True)
            return sorted(ids, key=lambda i: -(sims[i] if sims.get(i) is not None else -2.0))

        if hop1_ids:
            lists.append((0.9, ordered(hop1_ids)))
        if hop2_ids:
            lists.append((0.5, ordered(hop2_ids)))

        # relevance gate: keep a candidate only with evidence (see search())
        known = [v for v in sims.values() if v is not None]
        rel_floor = max(known) - self.relative_margin if known else None
        slack = self.expand_slack
        survivors: set[str] = set()
        for fid in pool:
            if fid in ft_hits or tier.get(fid) == 1:  # (a) fulltext, (c) fact of a named entity
                survivors.add(fid)
                continue
            sv = sims.get(fid)
            if sv is None:  # no vector evidence: the facts of a (fulltext-)matched entity are all there is
                if tier.get(fid) == 2:
                    survivors.add(fid)
                continue
            if sv >= thr or (tier.get(fid) in (2, 3) and sv >= thr - slack):  # (b) / (d)
                if rel_floor is None or sv >= rel_floor:
                    survivors.add(fid)
        scores = _rrf([(w, [i for i in ids if i in survivors]) for w, ids in lists])
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
        hits = self._to_hits(group, [(pool[i], s) for i, s in ranked], allowed, at, sims=sims)
        return hits, [n for n, _ in matched]

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
        *,
        min_similarity: float | None = None,
    ) -> list[Node]:
        """Entities the query is about: name/alias mention, trigram FTS and vector similarity."""
        return [
            n
            for n, _ in self._match_entities_ex(
                group, allowed, query, terms, qvec, center, limit=limit, min_similarity=min_similarity
            )
        ]

    def _match_entities_ex(
        self,
        group: str,
        allowed: set[str] | None,
        query: str,
        terms: list[str],
        qvec: list[float] | None,
        center: str | None,
        limit: int = 5,
        *,
        min_similarity: float | None = None,
        hub_total: int | None = None,
    ) -> list[tuple[Node, str]]:
        """Like :meth:`_match_entities`, with the strength of each match, best first: ``"strong"`` (an entity name /
        alias literally occurs in the query, or it is the ``center``) or ``"weak"`` (only a trigram-fulltext fragment
        or the similarity of the entity vector).

        ``min_similarity``: cosine threshold of the entity vector (``None``: the legacy ``(1 + cos) / 2 >= 0.66``).
        ``hub_total`` (number of facts of the group): a literal mention of a persona word ("ユーザー", "user") or of a
        hub entity (>= ``HUB_MIN_FACTS`` facts and half of all facts, typically the user) counts only as ``"weak"``,
        since all its facts would otherwise come back for any question that says "the user".
        """
        scored: dict[str, float] = defaultdict(float)
        nodes: dict[str, Node] = {}
        strong: set[str] = set()
        persona: set[str] = set()
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
                    strong.add(e["id"])
                elif any(f in terms for f in forms):
                    score += 5
                    strong.add(e["id"])
                if any(f in PERSONA_TERMS for f in forms):
                    persona.add(e["id"])
                scored[e["id"]] += float(score)
                nodes[e["id"]] = e
        if qvec is not None:
            for e, s in self.db.vector_search(ENTITY_INDEX, qvec, k=6):
                # an entity embedding encodes its summary: only trust it when every source channel is allowed
                if (
                    e.get("group_id") == group
                    and self._entity_visible(e.properties, allowed)
                    and (allowed is None or set(e.get("summary_channels", [])) <= allowed)
                    and ((2.0 * s - 1.0) >= min_similarity if min_similarity is not None else s >= 0.66)
                ):
                    scored[e["id"]] += 2.0 * s
                    nodes.setdefault(e["id"], e)
        center_id = None
        if center:
            c = self._resolve_entity_ref(group, allowed, center)
            if c is not None:
                center_id = c["id"]
                scored[c["id"]] += 20.0
                nodes.setdefault(c["id"], c)
                strong.add(c["id"])
        if hub_total is not None:
            for eid in list(strong):
                if eid == center_id:
                    continue
                hub = eid in persona
                if not hub and hub_total >= HUB_MIN_FACTS:
                    rows = self.db.query(
                        "MATCH (e:Entity {id:$id})-[f:FACT]-() RETURN count(DISTINCT f.id) AS n", {"id": eid}
                    )
                    n = int(rows[0]["n"]) if rows else 0
                    hub = n >= HUB_MIN_FACTS and n * 2 >= hub_total
                if hub:
                    strong.discard(eid)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
        return [(nodes[i], "strong" if i in strong else "weak") for i, _ in ranked]

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
        min_similarity: float | None = None,
    ) -> Answer:
        """Retrieve facts for ``question`` and let the LLM answer, citing sources and validity.

        The LLM is given :attr:`answer_prompt` as its system prompt. When nothing matches, the LLM is not
        called and ``text`` is :data:`NO_MATCH_TEXT`; callers can check ``not answer.hits`` to show their own text.

        Retrieval includes ended facts (history) and, when ``as_of`` is not given, planned facts that have
        not started yet; each is labelled with its ``status`` so the model can tell them apart.
        """
        llm = self._need_llm()
        group = self._group(group_id)
        allowed = _channel_set(channels)
        hits, ents = self._search(
            question,
            group,
            channels,
            as_of,
            center_entity,
            limit,
            include_invalid=True,
            include_future=as_of is None,
            min_similarity=min_similarity,
        )
        if not hits:
            return Answer(question, NO_MATCH_TEXT, [], [], self._local(self._as_of(as_of)) if as_of else None)
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
            self.answer_prompt, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), task="answer"
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

    # ================================================================== CURATION
    # Corrections that keep history (unlike forget): two entities that are the same thing become one, a fact that is
    # no longer true is closed, the relation vocabulary is cleaned up. ``review`` lists candidates for these; the
    # caller (an agent, with the user's confirmation) decides.

    def _fact_vector(self, edge: Relationship) -> list[float] | None:
        ix = next((i for i in self.db.vector_indexes() if i.name == FACT_INDEX), None)
        return self.db._vec_read(ix, edge.id) if ix is not None else None

    def _replace_fact_edge(self, edge: Relationship, src: int, dst: int, upd: dict[str, Any]) -> None:
        """Re-create a fact between other nodes (relationships cannot change endpoints), keeping id and vector."""
        props = {k: v for k, v in edge.properties.items() if v is not None}
        for k, v in upd.items():
            if v is None:
                props.pop(k, None)
            else:
                props[k] = v
        vec = self._fact_vector(edge)
        if vec:
            props["embedding"] = vec
        self.db.delete_edge(edge.id)
        self.db.create_edge(src, dst, FACT_TYPE, props)

    def merge_entities(
        self,
        keep: str,
        merge: str,
        *,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Merge entity ``merge`` into ``keep`` (each: id, or exact name / alias): one entity remains.

        ``keep`` gets the other's name and aliases as aliases, its channels and per-channel summaries; every fact and
        mention of ``merge`` moves to ``keep``. A fact that becomes an exact duplicate of one ``keep`` already has (same
        subject, object, relation and sentence, both still valid or both ended) is folded into it (sources united,
        supersede links redirected). A fact between the two becomes an attribute of ``keep`` (listed in ``warnings``).
        Nothing is deleted from the history. Returns what moved; with ``dry_run`` nothing is written.
        """
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            k_node = self._single_entity(group, keep, "keep")
            m_node = self._single_entity(group, merge, "merge")
            if k_node["id"] == m_node["id"]:
                raise ValueError(f"{keep!r} and {merge!r} are the same entity ({k_node['id']})")
            kid, mid = k_node["id"], m_node["id"]
            warnings: list[str] = []
            if not _types_compatible(k_node.get("type", "Other"), m_node.get("type", "Other")):
                warnings.append(
                    f"types differ ({k_node.get('type')} vs {m_node.get('type')}); the merged entity keeps {k_node.get('type')}"
                )
            keep_facts = {f["id"]: f for f in self._facts_of_entity(kid, group)}
            moving = [f for f in self._facts_of_entity(mid, group) if f["id"] not in keep_facts]
            between = [f for f in self._facts_of_entity(mid, group) if f["id"] in keep_facts]

            def remap(x: str | None) -> str | None:
                return kid if x == mid else x

            def key(subj: str | None, obj: str | None, f: Relationship) -> tuple[Any, ...]:
                return (subj, obj, f.get("relation"), f.get("text_norm") or nfkc(f["fact"]).casefold(), f.get("invalid_at") is None)

            existing = {key(f.get("subject_id"), f.get("object_id"), f): fid for fid, f in keep_facts.items()}
            plan_move: list[tuple[Relationship, dict[str, Any]]] = []
            plan_fold: list[tuple[Relationship, str]] = []  # (fact to drop, survivor id)
            for f in moving:
                subj, obj = remap(f.get("subject_id")), remap(f.get("object_id"))
                k = key(subj, obj, f)
                if k in existing:
                    plan_fold.append((f, existing[k]))
                    continue
                existing[k] = f["id"]
                plan_move.append((f, {"subject_id": subj, "object_id": obj}))
            for f in between:  # a fact linking the two: now about one entity
                warnings.append(f"fact {f['id']} linked the two entities and is now an attribute of {k_node['name']!r}: {f['fact']}")
                plan_move.append((f, {"subject_id": kid, "object_id": None}))
            mentions = [
                (r["e"], r["role"])
                for r in self.db.query(
                    "MATCH (e:Episode)-[m:MENTIONS]->(x:Entity {id:$id}) RETURN e, m.role AS role", {"id": mid}
                )
            ]
            already = {r["id"] for r in self.db.query("MATCH (e:Episode)-[:MENTIONS]->(x:Entity {id:$id}) RETURN e.id AS id", {"id": kid})}
            new_mentions = [(e, role) for e, role in mentions if e["id"] not in already]
            kp, mp = k_node.properties, m_node.properties
            aliases = list(kp.get("aliases", []))
            added: list[str] = []
            for a in [mp["name"], *mp.get("aliases", [])]:
                if a and a != kp["name"] and a not in aliases:
                    aliases.append(a)
                    added.append(a)
            entries = dict(self._entries(kp))
            for ch, text in self._entries(mp).items():
                entries[ch] = short(f"{entries[ch]} / {text}", 300) if ch in entries and text not in entries[ch] else entries.get(ch, text)
            legacy = not self._entries(kp) and not self._entries(mp)
            upd: dict[str, Any] = {
                "aliases": aliases,
                "alias_keys": sorted({k for k in (norm_key(x) for x in [kp["name"], *aliases]) if k}),
                "alias_src": list(dict.fromkeys([*kp.get("alias_src", []), *mp.get("alias_src", [])])),
                "search_text": nfkc("\n".join([kp["name"], *aliases])).casefold(),
                "channels": list(dict.fromkeys([*kp.get("channels", []), *mp.get("channels", [])])),
                "created_at": min(kp.get("created_at") or "9999", mp.get("created_at") or "9999"),
                "updated_at": self._now(),
            }
            if legacy:
                parts = [s for s in (kp.get("summary"), mp.get("summary")) if s]
                upd["summary"] = short(" / ".join(dict.fromkeys(parts)), 300)
            else:
                upd.update(
                    summaries=[f"{c}\t{t}" for c, t in entries.items()],
                    summary=" / ".join(entries.values()),
                    summary_channels=list(entries),
                )
            result: dict[str, Any] = {
                "kept": {"id": kid, "name": kp["name"], "type": kp.get("type")},
                "merged": {"id": mid, "name": mp["name"], "type": mp.get("type")},
                "facts_moved": sorted(f["id"] for f, _ in plan_move),
                "facts_folded": [{"fact_id": f["id"], "into": s} for f, s in plan_fold],
                "mentions_moved": len(new_mentions),
                "aliases_added": added,
                "warnings": warnings,
                "dry_run": dry_run,
            }
            if dry_run:
                return result
            # new vector of the kept entity (outside the transaction; a failing embedder keeps the old vector)
            vec: list[float] | None = None
            if self.embedder is not None:
                after = Node(k_node.id, k_node.labels, {**kp, **upd})
                facts_after = [*keep_facts.values(), *(f for f, _ in plan_move)]
                try:
                    vec = self.embedder.embed([self._reembed_entity_text(after, facts_after, self._now())])[0]
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"could not re-embed the merged entity ({type(exc).__name__}); its old vector is kept")
            if vec is not None:
                upd["embedding"] = vec
            nodes = self._nodes("Entity", {x for f, u in plan_move for x in (u["subject_id"], u["object_id"]) if x}, group)
            folds = {s: [f for f, s2 in plan_fold if s2 == s] for s in {s for _, s in plan_fold}}
            with self.db.transaction():
                for f, u in plan_move:
                    src = nodes[u["subject_id"]].id
                    dst = nodes[u["object_id"]].id if u["object_id"] else src
                    self._replace_fact_edge(f, src, dst, u)
                for survivor_id, dropped in folds.items():
                    self._fold_facts(group, survivor_id, dropped)
                for e, role in new_mentions:
                    self.db.create_edge(e.id, k_node.id, "MENTIONS", {"role": role} if role else {})
                self.db.delete_node(m_node.id, detach=True)
                self.db.update_node(k_node.id, upd)
            return result

    def _single_entity(self, group: str, ref: str, label: str) -> Node:
        nodes = self._entities_by_ref(group, ref)
        if not nodes:
            raise ValueError(f"{label}: no entity with id / exact name {ref!r} (find it with search_entities)")
        if len(nodes) > 1:
            options = ", ".join(f"{n['id']} ({n.get('type')} {n['name']!r})" for n in nodes)
            raise ValueError(f"{label}: {ref!r} is ambiguous; use an entity id: {options}")
        return nodes[0]

    def _fold_facts(self, group: str, survivor_id: str, dropped: list[Relationship]) -> None:
        """Delete ``dropped`` duplicates of the fact ``survivor_id``: their sources and links move to the survivor."""
        survivor = self._fact_edge(survivor_id)
        assert survivor is not None
        sp = survivor.properties
        gone = {f["id"] for f in dropped}
        eps = list(sp.get("episode_ids", []))
        chans = list(sp.get("channels", []))
        sup = list(sp.get("supersedes", []))
        conf = float(sp.get("confidence", 0))
        for f in dropped:
            eps += [e for e in f.get("episode_ids", []) if e not in eps]
            chans += [c for c in f.get("channels", []) if c not in chans]
            sup += [s for s in f.get("supersedes", []) if s not in sup and s != survivor_id]
            conf = max(conf, float(f.get("confidence", 0)))
        upd: dict[str, Any] = {"episode_ids": eps, "channels": chans, "confidence": conf, "supersedes": sup or None}
        self.db.update_edge(survivor.id, upd)
        for f in dropped:
            self.db.delete_edge(f.id)
        # links pointing at a dropped fact now point at the survivor
        for r in self.db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f", {"g": group}):
            f = r["f"]
            if f["id"] == survivor_id:
                continue
            fu: dict[str, Any] = {}
            if f.get("superseded_by") in gone:
                fu["superseded_by"] = survivor_id
            if any(s in gone for s in f.get("supersedes", [])):
                fu["supersedes"] = list(dict.fromkeys(survivor_id if s in gone else s for s in f["supersedes"]))
            if fu:
                self.db.update_edge(f.id, fu)

    def invalidate_fact(
        self,
        fact_id: str,
        *,
        at: str | datetime | None = None,
        reason: str | None = None,
        group_id: str | None = None,
    ) -> FactHit:
        """Close a fact that is no longer true (a manual correction; nothing is deleted).

        ``invalid_at`` becomes ``at`` (default now; a plan that has not started yet is cancelled at its start, so it
        never becomes valid). ``reason`` is stored as ``invalid_reason``. A fact that already ended is refused.
        """
        self._need_writable()
        group = self._group(group_id)
        with self._write_guard():
            f = self._fact_edge(fact_id)
            if f is None or f.get("group_id") != group:
                raise ValueError(f"no fact {fact_id!r} in group {group!r}")
            now = self._now()
            va = f.get("valid_at") or now
            if at is None:
                end = max(now, va)
            else:
                dt = parse_time(at, self.tz)
                if dt is None:
                    raise ValueError(f"cannot parse at: {at!r}")
                end = to_iso(dt)
                if end < va:
                    raise ValueError(f"at ({self._local(end)}) is before the fact became valid ({self._local(va)})")
            if f.get("invalid_at") is not None and f["invalid_at"] <= max(now, end):
                raise ValueError(f"fact {fact_id} already ended at {self._local(f['invalid_at'])}")
            upd: dict[str, Any] = {"invalid_at": end, "expired_at": now, "ended_by": "manual"}
            if reason and reason.strip():
                upd["invalid_reason"] = short(reason, 300)
            edge = self.db.update_edge(f.id, upd)
        return self._to_hits(group, [(edge, 0.0)], None, now)[0]

    def _known_relations(self, group: str, channel: str | None = None, limit: int = KNOWN_RELATIONS_IN_CONTEXT) -> list[str]:
        """Relation names in use (facts visible in ``channel``; ``None`` = all), most used first."""
        counts: Counter[str] = Counter()
        for r in self.db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f.relation AS r, f.channels AS c", {"g": group}):
            if r["r"] and (channel is None or channel in (r["c"] or [])):
                counts[r["r"]] += 1
        return [rel for rel, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]

    def relation_stats(self, group_id: str | None = None) -> list[dict[str, Any]]:
        """Every relation name with its fact counts (``facts``, ``valid``) and the newest example sentence."""
        group = self._group(group_id)
        stats: dict[str, dict[str, Any]] = {}
        now = self._now()
        for r in self.db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f", {"g": group}):
            f = r["f"]
            s = stats.setdefault(f.get("relation", ""), {"relation": f.get("relation", ""), "facts": 0, "valid": 0, "_at": "", "example": ""})
            s["facts"] += 1
            if _fact_status(f.properties, now) != "ended":
                s["valid"] += 1
            if (f.get("valid_at") or "") >= s["_at"]:
                s["_at"], s["example"] = f.get("valid_at") or "", short(f["fact"], 120)
        out = sorted(stats.values(), key=lambda s: (-s["facts"], s["relation"]))
        for s in out:
            del s["_at"]
        return out

    def rename_relation(
        self,
        old: str,
        new: str,
        *,
        swap: bool = False,
        fact_ids: Sequence[str] | None = None,
        group_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Rename relation ``old`` to ``new`` on every fact (or only ``fact_ids``); ``swap`` also turns the facts around
        (subject <-> object; attribute facts have no object and are only renamed). A passive ``new`` with a known active
        form (``OWNED_BY``) is stored as that form with the direction swapped, like at ingestion."""
        if not dry_run:
            self._need_writable()
        group = self._group(group_id)
        old_n = normalize_relation(old)
        with self._write_guard():
            facts = [r["f"] for r in self.db.query("MATCH ()-[f:FACT {group_id:$g}]->() WHERE f.relation = $r RETURN f", {"g": group, "r": old_n})]
            if fact_ids is not None:
                wanted = set(fact_ids)
                missing = wanted - {f["id"] for f in facts}
                if missing:
                    raise ValueError(f"not facts with relation {old_n}: {', '.join(sorted(missing))}")
                facts = [f for f in facts if f["id"] in wanted]
            renamed: list[str] = []
            swapped: list[str] = []
            plan: list[tuple[Relationship, str, bool]] = []
            for f in facts:
                has_obj = bool(f.get("object_id")) and f.get("object_id") != f.get("subject_id")
                rel, inv = canonical_relation(new, has_obj)
                turn = has_obj and (swap != inv)
                plan.append((f, rel, turn))
                renamed.append(f["id"])
                if turn:
                    swapped.append(f["id"])
            if not dry_run and plan:
                ents = self._nodes("Entity", {x for f, _, t in plan if t for x in (f["subject_id"], f["object_id"])}, group)
                with self.db.transaction():
                    for f, rel, turn in plan:
                        if turn:
                            s, o = f["object_id"], f["subject_id"]
                            self._replace_fact_edge(f, ents[s].id, ents[o].id, {"relation": rel, "subject_id": s, "object_id": o})
                        else:
                            self.db.update_edge(f.id, {"relation": rel})
        return {
            "relation": old_n,
            "to": RELATION_INVERSES.get(normalize_relation(new), normalize_relation(new)),
            "renamed": renamed,
            "swapped": swapped,
            "dry_run": dry_run,
        }

    def review(self, group_id: str | None = None, *, limit: int = 20) -> dict[str, Any]:
        """Housekeeping report for an agent (read-only): candidates to clean up, never changed here.

        * ``duplicate_entities``: pairs that may be the same thing (shared alias, one name containing the other, very
          similar vectors), with the reasons; fix with :meth:`merge_entities`.
        * ``relations``: the vocabulary with counts; ``relation_suggestions``: passive names to turn around, names that
          differ only in spelling, and names used once (candidates for :meth:`rename_relation`).
        * ``stale_plans``: planned / tentative facts whose start passed more than ``STALE_PLAN_DAYS`` ago without a
          confirmation, to confirm or close (:meth:`invalidate_fact`).
        """
        group = self._group(group_id)
        rels = self.relation_stats(group)
        suggestions: list[dict[str, Any]] = []
        names = {r["relation"] for r in rels}
        for r in rels:
            rel = r["relation"]
            if rel in RELATION_INVERSES:
                suggestions.append({"relation": rel, "suggest": RELATION_INVERSES[rel], "swap": True, "why": "passive form: store the active direction"})
        by_squash: dict[str, list[str]] = defaultdict(list)
        for n in names:
            by_squash[re.sub(r"[_\W]+", "", n).casefold().rstrip("s")].append(n)
        for variants in by_squash.values():
            if len(variants) > 1:
                top = max(variants, key=lambda v: next(r["facts"] for r in rels if r["relation"] == v))
                for v in variants:
                    if v != top:
                        suggestions.append({"relation": v, "suggest": top, "swap": False, "why": "spelling variant"})
        singletons = [r["relation"] for r in rels if r["facts"] == 1]
        return {
            "group_id": group,
            "duplicate_entities": self._duplicate_entity_candidates(group, limit),
            "relations": rels[:60],
            "relation_suggestions": suggestions,
            "relations_used_once": singletons,
            "stale_plans": self._stale_plans(group, limit),
            "next": (
                "Show the user the candidates that look real and ask before changing anything. Then: merge_entities(keep, merge) "
                "for duplicates, rename_relation(old, new[, swap]) to unify relation names (a relation used once is often a "
                "synonym of a common one: compare the examples), invalidate_fact(fact_id) for plans that did not happen."
            ),
        }

    def _duplicate_entity_candidates(self, group: str, limit: int) -> list[dict[str, Any]]:
        ents = [r["e"] for r in self.db.query("MATCH (e:Entity {group_id:$g}) RETURN e", {"g": group})]
        by_id = {n["id"]: n for n in ents}
        reasons: dict[tuple[str, str], dict[str, Any]] = {}

        def note(a: str, b: str, why: str, sim: float | None = None) -> None:
            if a == b or not _types_compatible(by_id[a].get("type", "Other"), by_id[b].get("type", "Other")):
                return
            k = (a, b) if a < b else (b, a)
            r = reasons.setdefault(k, {"reasons": [], "similarity": None})
            if why not in r["reasons"]:
                r["reasons"].append(why)
            if sim is not None:
                r["similarity"] = sim
        keys: dict[str, list[str]] = defaultdict(list)
        for n in ents:
            for k in n.get("alias_keys", []) or [norm_key(n["name"])]:
                keys[k].append(n["id"])
        for ids in keys.values():
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    note(a, b, "shared name / alias")
        names = [(norm_key(n["name"]), n["id"]) for n in ents]
        for ka, a in names:
            if len(ka) < 3:
                continue
            for kb, b in names:
                if a != b and len(kb) > len(ka) and ka in kb:
                    note(a, b, "one name contains the other")
        ix = next((i for i in self.db.vector_indexes() if i.name == ENTITY_INDEX), None)
        if ix is not None and len(ents) > 1:
            thr = min(0.95, self.effective_min_similarity() + 0.3)
            for n in ents:
                v = self.db._vec_read(ix, n.id)
                if not v:
                    continue
                for m, s in self.db.vector_search(ENTITY_INDEX, v, k=6):
                    if m.get("group_id") != group or m["id"] == n["id"]:
                        continue
                    sim = 2.0 * s - 1.0
                    if sim >= thr:
                        note(n["id"], m["id"], "very similar description", round(sim, 4))
        fact_counts: Counter[str] = Counter()
        for r in self.db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f.subject_id AS s, f.object_id AS o", {"g": group}):
            for x in {r["s"], r["o"]} - {None}:
                fact_counts[x] += 1

        def view(i: str) -> dict[str, Any]:
            n = by_id[i]
            return {"id": i, "name": n["name"], "type": n.get("type"), "aliases": n.get("aliases", [])[:5], "summary": short(n.get("summary", ""), 100), "facts": fact_counts[i]}

        ranked = sorted(reasons.items(), key=lambda kv: (-len(kv[1]["reasons"]), -(kv[1]["similarity"] or 0)))
        return [{"a": view(a), "b": view(b), **r} for (a, b), r in ranked[:limit]]

    def _stale_plans(self, group: str, limit: int) -> list[dict[str, Any]]:
        now = self._now()
        cutoff = to_iso(self._clock() - timedelta(days=STALE_PLAN_DAYS))
        rows = self.db.query(
            "MATCH ()-[f:FACT {group_id:$g}]->() WHERE f.modality IN ['planned', 'tentative'] AND f.invalid_at IS NULL "
            "AND f.valid_at < $cutoff RETURN f ORDER BY f.valid_at",
            {"g": group, "cutoff": cutoff},
        )
        return [h.to_dict() for h in self._to_hits(group, [(r["f"], 0.0) for r in rows[:limit]], None, now, max_episodes=1)]

    # ================================================================== UPCOMING

    def upcoming(
        self,
        days: float = 7,
        *,
        group_id: str | None = None,
        channels: Sequence[str] | None = None,
        past_days: float = 0,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Facts that start or end within the next ``days`` (and the last ``past_days``), soonest first.

        Each item is ``{"event": "starts" | "ends", "at": local time, "fact": FactHit dict}``. "ends" covers stated
        ends (an absence until 9/17), not ends caused by a newer contradicting fact; a plan cancelled before it began
        is not listed.
        """
        group = self._group(group_id)
        allowed = _channel_set(channels)
        if allowed is not None and not allowed:
            return []
        now_dt = self._clock()
        lo = to_iso(now_dt - timedelta(days=past_days))
        hi = to_iso(now_dt + timedelta(days=days))
        rows = self.db.query(
            "MATCH ()-[f:FACT {group_id:$g}]->() WHERE (f.valid_at > $lo AND f.valid_at <= $hi) "
            "OR (f.invalid_at > $lo AND f.invalid_at <= $hi) RETURN f",
            {"g": group, "lo": lo, "hi": hi},
        )
        events: list[tuple[str, str, Relationship]] = []
        for r in rows:
            f = r["f"]
            if not self._fact_visible(f.properties, allowed):
                continue
            va, ia = f.get("valid_at"), f.get("invalid_at")
            if va and lo < va <= hi and (ia is None or ia > va):
                events.append((va, "starts", f))
            if ia and lo < ia <= hi and not f.get("superseded_by") and f.get("ended_by") != "manual" and (va is None or ia > va):
                events.append((ia, "ends", f))
        events.sort(key=lambda e: e[0])
        events = events[:limit]
        hits = {h.id: h for h in self._to_hits(group, [(f, 0.0) for _, _, f in events], allowed, to_iso(now_dt), max_episodes=1)}
        return [{"event": kind, "at": self._local(at), "fact": hits[f["id"]].to_dict()} for at, kind, f in events]

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
