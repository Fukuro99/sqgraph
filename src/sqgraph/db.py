"""The :class:`GraphDB` class: storage, Python API and index management."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import struct
import threading
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Any, Iterable, Iterator, Sequence

import sqlite_vec

from . import functions
from .errors import DimensionMismatchError, SqgraphError
from .ftsquery import fallback_fts5, lucene_to_fts5
from .schema import (
    DDL,
    ENTITY_TYPE_TABLES,
    NODE,
    RELATIONSHIP,
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    fts_table,
    is_identifier,
    vec_table,
)
from .types import Node, Relationship

_TOKENIZERS = {"unicode61", "porter", "ascii", "trigram"}


@dataclass(frozen=True)
class VectorIndex:
    """Metadata of a vector index.

    For a relationship index (``entity_type == "relationship"``) ``label`` holds the relationship type.
    """

    name: str
    label: str
    property: str
    dimensions: int
    similarity: str
    entity_type: str = NODE  # 'node' | 'relationship'

    @property
    def table(self) -> str:
        return vec_table(self.name)

    @property
    def key_column(self) -> str:
        """Column of the vec0 table holding the entity id."""
        return "edge_id" if self.entity_type == RELATIONSHIP else "node_id"


@dataclass(frozen=True)
class FulltextIndex:
    """Metadata of a fulltext index.

    For a relationship index (``entity_type == "relationship"``) ``labels`` holds the relationship types.
    """

    name: str
    labels: tuple[str, ...]
    properties: tuple[str, ...]
    tokenizer: str = "unicode61"
    entity_type: str = NODE  # 'node' | 'relationship'

    @property
    def table(self) -> str:
        return fts_table(self.name)

    @property
    def key_column(self) -> str:
        """Column of the FTS5 table repeating the entity id (the rowid is the id as well)."""
        return "edge_id" if self.entity_type == RELATIONSHIP else "node_id"

    def column(self, i: int) -> str:
        return f"c{i}"


def normalize_value(value: Any) -> Any:
    """Convert ``value`` to something JSON-serializable (numpy arrays/scalars, tuples, ...)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_value(v) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return normalize_value(value.tolist())
    raise SqgraphError(f"unsupported property value type: {type(value).__name__}")


def normalize_props(props: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize a property dict; ``None`` values are dropped."""
    out: dict[str, Any] = {}
    for k, v in (props or {}).items():
        if not isinstance(k, str) or not k:
            raise SqgraphError(f"property keys must be non-empty strings, got {k!r}")
        v = normalize_value(v)
        if v is not None:
            out[k] = v
    return out


def dumps(obj: Any) -> str:
    """Compact JSON encoding used for stored properties."""
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise SqgraphError(f"property value is not JSON-serializable: {exc}") from exc


def vector_to_list(vector: Any) -> list[float]:
    """Accept list/tuple/numpy array of numbers and return ``list[float]``."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Iterable):
        raise SqgraphError("a vector must be a sequence of numbers")
    try:
        return [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        raise SqgraphError(f"a vector must be a sequence of numbers: {exc}") from exc


def _labels_list(labels: Sequence[str] | str | None) -> list[str]:
    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [labels]
    out: list[str] = []
    for label in labels:
        if not isinstance(label, str) or not label:
            raise SqgraphError(f"labels must be non-empty strings, got {label!r}")
        if label not in out:
            out.append(label)
    return out


class GraphDB:
    """An embedded property graph stored in a single SQLite file (or in memory).

    Parameters
    ----------
    path:
        File path or ``":memory:"``.
    read_only:
        Open the database read-only (write queries are rejected).
    max_var_length:
        Upper bound used for unbounded variable-length patterns such as ``-[*]->``.
    secure_delete:
        Run ``PRAGMA secure_delete=ON`` (writable databases): deleted content is overwritten with zeros
        instead of lingering in free pages. See also :meth:`checkpoint` and :meth:`optimize_fulltext`.
    """

    def __init__(
        self,
        path: str | FsPath = ":memory:",
        read_only: bool = False,
        max_var_length: int = 10,
        secure_delete: bool = False,
    ) -> None:
        self.path = str(path)
        self.read_only = read_only
        self.max_var_length = max_var_length
        self._lock = threading.RLock()
        self._depth = 0
        self._vec: dict[str, VectorIndex] = {}
        self._fts: dict[str, FulltextIndex] = {}
        self._executor: Any = None
        memory = self.path == ":memory:"
        if read_only:
            if memory:
                raise SqgraphError("an in-memory database cannot be opened read-only")
            p = FsPath(self.path)
            if not p.exists():
                raise SqgraphError(f"database file not found: {self.path}")
            uri = p.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
        else:
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn = conn
        conn.row_factory = None
        try:
            if sqlite3.sqlite_version_info < (3, 38, 0):
                raise SqgraphError(
                    f"SQLite >= 3.38 is required (found {sqlite3.sqlite_version})"
                )
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            functions.register(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            if not memory and not read_only:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA analysis_limit=2000")
            if secure_delete and not read_only:
                conn.execute("PRAGMA secure_delete=ON")
            if read_only:
                self._check_schema()
            else:
                conn.executescript(DDL)
                conn.execute(
                    "INSERT OR IGNORE INTO _meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                self._migrate_schema()
            self._load_indexes()
        except BaseException:
            conn.close()
            raise

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the database connection (running ``PRAGMA optimize`` on writable databases)."""
        if not self.read_only and self._depth == 0:
            try:
                self._conn.execute("PRAGMA optimize")
            except sqlite3.Error:
                pass
        self._conn.close()

    def analyze(self) -> None:
        """Refresh SQLite's query-planner statistics (worth calling after bulk loads).

        Without statistics SQLite may pick poor join orders for selective anchors such as
        ``MATCH (n:Label {prop: $v})-[:REL]->(m)``.
        """
        self._require_writable()
        with self._atomic():
            self._conn.execute("ANALYZE")

    def set_secure_delete(self, on: bool = True) -> None:
        """Switch ``PRAGMA secure_delete`` on this connection (no-op on a read-only database)."""
        if not self.read_only:
            with self._lock:
                self._conn.execute(f"PRAGMA secure_delete={'ON' if on else 'OFF'}")

    def checkpoint(self, truncate: bool = True) -> None:
        """Run ``PRAGMA wal_checkpoint`` (``TRUNCATE`` by default, else ``PASSIVE``).

        Moves the write-ahead log into the database file; ``TRUNCATE`` also empties the ``-wal`` file, so
        deleted data no longer survives in it. A no-op for in-memory and read-only databases. It cannot
        truncate while another connection is reading; it then does as much as it can.
        """
        if self.read_only or self.path == ":memory:":
            return
        with self._lock:
            if self._depth:
                raise SqgraphError("cannot checkpoint inside a transaction")
            self._conn.execute(f"PRAGMA wal_checkpoint({'TRUNCATE' if truncate else 'PASSIVE'})").fetchall()

    def optimize_fulltext(self, names: Iterable[str] | None = None) -> None:
        """Merge the segments of fulltext indexes (FTS5 ``optimize``), dropping the remains of deleted rows.

        ``names`` selects the indexes (default: all).
        """
        self._require_writable()
        if names is None:
            indexes = list(self._fts.values())
        else:
            indexes = [self._get_fulltext_index(n) for n in names]
        with self._atomic():
            for ix in indexes:
                self._conn.execute(f"INSERT INTO {ix.table}({ix.table}) VALUES('optimize')")

    def get_meta(self, key: str) -> str | None:
        """Value stored under ``key`` in the ``_meta`` table, or ``None``."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM _meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` in the ``_meta`` table (applications may keep their own keys)."""
        self._require_writable()
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise SqgraphError("meta key and value must be strings (key non-empty)")
        with self._atomic():
            self._conn.execute(
                "INSERT INTO _meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def __enter__(self) -> "GraphDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"GraphDB({self.path!r}, read_only={self.read_only})"

    def _migrate_schema(self) -> None:
        """Upgrade databases created by older versions (tracked in ``_meta.schema_version``)."""
        conn = self._conn
        version = int(conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()[0])
        if version < 2:
            # idx_edges_type(type) is covered by idx_edges_src(src, type, dst) and lured SQLite's
            # planner into scanning by type on un-analyzed databases.
            conn.execute("DROP INDEX IF EXISTS idx_edges_type")
        if version < 3:
            # index metadata gained the kind of entity (node | relationship) an index covers
            for table in ENTITY_TYPE_TABLES:
                if not self._has_column(table, "entity_type"):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'node'")
        if version < SCHEMA_VERSION:
            conn.execute("UPDATE _meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))

    def _has_column(self, table: str, column: str) -> bool:
        return any(r[1] == column for r in self._conn.execute(f"PRAGMA table_info({table})"))

    def _check_schema(self) -> None:
        names = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master")}
        missing = [t for t in REQUIRED_TABLES if t not in names]
        if missing:
            raise SqgraphError(f"not a sqgraph database (missing tables: {', '.join(missing)})")

    def _entity_type_sql(self, table: str) -> str:
        """Select expression for ``entity_type`` (a read-only open cannot migrate an older file: all nodes)."""
        return "entity_type" if self._has_column(table, "entity_type") else f"'{NODE}'"

    def _load_indexes(self) -> None:
        self._vec = {
            r[0]: VectorIndex(*r)
            for r in self._conn.execute(
                "SELECT name, label, property, dimensions, similarity, "
                f"{self._entity_type_sql('_vector_indexes')} FROM _vector_indexes"
            )
        }
        self._fts = {
            r[0]: FulltextIndex(r[0], tuple(json.loads(r[1])), tuple(json.loads(r[2])), r[3], r[4])
            for r in self._conn.execute(
                "SELECT name, labels, properties, tokenizer, "
                f"{self._entity_type_sql('_fulltext_indexes')} FROM _fulltext_indexes"
            )
        }

    # ------------------------------------------------------------------ transactions

    @contextmanager
    def _atomic(self, write: bool = True) -> Iterator[None]:
        with self._lock:
            depth = self._depth
            conn = self._conn
            if depth == 0:
                conn.execute("BEGIN IMMEDIATE" if (write and not self.read_only) else "BEGIN")
            else:
                conn.execute(f"SAVEPOINT sqg_sp{depth}")
            self._depth += 1
            # the index registries mirror tables that a rollback will undo: restore them too
            snapshot = (dict(self._vec), dict(self._fts))
            try:
                yield
            except BaseException:
                self._vec, self._fts = snapshot
                self._depth -= 1
                if depth == 0:
                    conn.execute("ROLLBACK")
                else:
                    conn.execute(f"ROLLBACK TO sqg_sp{depth}")
                    conn.execute(f"RELEASE sqg_sp{depth}")
                raise
            else:
                self._depth -= 1
                if depth == 0:
                    conn.execute("COMMIT")
                else:
                    conn.execute(f"RELEASE sqg_sp{depth}")

    def transaction(self) -> AbstractContextManager[None]:
        """Context manager grouping several operations into one atomic unit.

        Nested use is supported through SAVEPOINTs; an exception rolls back the
        (innermost) block.
        """
        return self._atomic()

    def _require_writable(self) -> None:
        if self.read_only:
            raise SqgraphError("database is opened read-only")

    # ------------------------------------------------------------------ loading

    def _load_nodes(self, ids: Iterable[int]) -> dict[int, Node]:
        ids = list(dict.fromkeys(ids))
        out: dict[int, Node] = {}
        for i in range(0, len(ids), 5000):
            chunk = ids[i : i + 5000]
            rows = self._conn.execute(
                "SELECT n.id, n.props, (SELECT json_group_array(label) FROM "
                "(SELECT label FROM node_labels WHERE node_id = n.id ORDER BY label)) "
                "FROM nodes n WHERE n.id IN (SELECT value FROM json_each(?))",
                (json.dumps(chunk),),
            ).fetchall()
            for nid, props, labels in rows:
                out[nid] = Node(nid, tuple(json.loads(labels)), json.loads(props))
        return out

    def _load_edges(self, ids: Iterable[int]) -> dict[int, Relationship]:
        ids = list(dict.fromkeys(ids))
        out: dict[int, Relationship] = {}
        for i in range(0, len(ids), 5000):
            chunk = ids[i : i + 5000]
            rows = self._conn.execute(
                "SELECT id, type, src, dst, props FROM edges "
                "WHERE id IN (SELECT value FROM json_each(?))",
                (json.dumps(chunk),),
            ).fetchall()
            for eid, typ, src, dst, props in rows:
                out[eid] = Relationship(eid, typ, src, dst, json.loads(props))
        return out

    def _node_labels(self, node_id: int) -> list[str]:
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT label FROM node_labels WHERE node_id = ? ORDER BY label", (node_id,)
            )
        ]

    def _node_exists(self, node_id: int) -> bool:
        return self._conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is not None

    # ------------------------------------------------------------------ index helpers

    def _vector_props(self, labels: Iterable[str]) -> dict[str, list[VectorIndex]]:
        """Vector-indexed property names -> indexes, for a node with ``labels``."""
        if not self._vec:
            return {}
        ls = set(labels)
        out: dict[str, list[VectorIndex]] = {}
        for ix in self._vec.values():
            if ix.entity_type == NODE and ix.label in ls:
                out.setdefault(ix.property, []).append(ix)
        return out

    def _edge_vector_props(self, type_: str) -> dict[str, list[VectorIndex]]:
        """Vector-indexed property names -> indexes, for a relationship of type ``type_``."""
        out: dict[str, list[VectorIndex]] = {}
        for ix in self._vec.values():
            if ix.entity_type == RELATIONSHIP and ix.label == type_:
                out.setdefault(ix.property, []).append(ix)
        return out

    def _has_indexes(self, entity_type: str) -> bool:
        """True if any vector or fulltext index covers ``entity_type`` ('node' | 'relationship')."""
        return any(ix.entity_type == entity_type for ix in (*self._vec.values(), *self._fts.values()))

    def _vec_blob(self, ix: VectorIndex, vector: Any) -> bytes:
        vec = vector_to_list(vector)
        if len(vec) != ix.dimensions:
            raise DimensionMismatchError(
                f"vector has {len(vec)} dimensions but index '{ix.name}' expects {ix.dimensions}"
            )
        return struct.pack(f"{len(vec)}f", *vec)

    def _vec_upsert(self, ix: VectorIndex, entity_id: int, vector: Any) -> None:
        blob = self._vec_blob(ix, vector)
        self._conn.execute(f"DELETE FROM {ix.table} WHERE {ix.key_column} = ?", (entity_id,))
        self._conn.execute(
            f"INSERT INTO {ix.table}({ix.key_column}, embedding) VALUES (?, ?)", (entity_id, blob)
        )

    def _vec_delete(
        self, entity_id: int, indexes: Iterable[VectorIndex] | None = None, entity_type: str = NODE
    ) -> None:
        """Drop the embeddings of a node/relationship (from ``indexes``, default: all of its kind)."""
        if indexes is None:
            indexes = [ix for ix in self._vec.values() if ix.entity_type == entity_type]
        for ix in indexes:
            self._conn.execute(f"DELETE FROM {ix.table} WHERE {ix.key_column} = ?", (entity_id,))

    @staticmethod
    def _fts_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return " ".join(str(GraphDB._fts_text(v) or "") for v in value)
        return json.dumps(value, ensure_ascii=False)

    def _fts_row(self, ix: FulltextIndex, entity_id: int, labels: Iterable[str], props: dict[str, Any]) -> None:
        """(Re)write the FTS row of a node/relationship (``labels``: its labels, or ``[type]``) for index ``ix``."""
        self._conn.execute(f"DELETE FROM {ix.table} WHERE rowid = ?", (entity_id,))
        if not set(labels) & set(ix.labels):
            return
        values = [self._fts_text(props.get(p)) for p in ix.properties]
        if all(v is None for v in values):
            return
        cols = ", ".join(ix.column(i) for i in range(len(ix.properties)))
        marks = ", ".join("?" for _ in ix.properties)
        self._conn.execute(
            f"INSERT INTO {ix.table}(rowid, {cols}, {ix.key_column}) VALUES (?, {marks}, ?)",
            (entity_id, *values, entity_id),
        )

    def _sync_fts(self, node_id: int, labels: Iterable[str] | None = None, props: dict[str, Any] | None = None) -> None:
        """Bring all fulltext indexes in line with the node's current state."""
        fts = [ix for ix in self._fts.values() if ix.entity_type == NODE]
        if not fts:
            return
        if labels is None:
            labels = self._node_labels(node_id)
        if props is None:
            row = self._conn.execute("SELECT props FROM nodes WHERE id = ?", (node_id,)).fetchone()
            props = json.loads(row[0]) if row else {}
        labels = list(labels)
        for ix in fts:
            self._fts_row(ix, node_id, labels, props)

    def _sync_edge_fts(self, edge_id: int, type_: str, props: dict[str, Any]) -> None:
        """Bring all relationship fulltext indexes in line with the relationship's current state."""
        for ix in self._fts.values():
            if ix.entity_type == RELATIONSHIP:
                self._fts_row(ix, edge_id, [type_], props)

    def _drop_edge_index_rows(self, edge_ids: Iterable[int]) -> None:
        """Remove relationships from all relationship vector and fulltext indexes."""
        ids = [(i,) for i in edge_ids]
        for ix in self._vec.values():
            if ix.entity_type == RELATIONSHIP:
                self._conn.executemany(f"DELETE FROM {ix.table} WHERE {ix.key_column} = ?", ids)
        for fx in self._fts.values():
            if fx.entity_type == RELATIONSHIP:
                self._conn.executemany(f"DELETE FROM {fx.table} WHERE rowid = ?", ids)

    # ------------------------------------------------------------------ low-level write primitives
    # (shared by the Python API and the Cypher executor; callers hold a transaction)

    def _w_create_node(self, labels: Sequence[str], props: dict[str, Any]) -> int:
        labels = _labels_list(labels)
        props = normalize_props(props)
        vec_props = self._vector_props(labels)
        vec_values = {k: props.pop(k) for k in list(props) if k in vec_props}
        cur = self._conn.execute("INSERT INTO nodes(props) VALUES (?)", (dumps(props),))
        nid = cur.lastrowid
        assert nid is not None
        if labels:
            self._conn.executemany(
                "INSERT INTO node_labels(node_id, label) VALUES (?, ?)", [(nid, lb) for lb in labels]
            )
        for k, v in vec_values.items():
            for ix in vec_props[k]:
                self._vec_upsert(ix, nid, v)
        self._sync_fts(nid, labels, props)
        return nid

    def _w_set_node_props(self, node_id: int, updates: dict[str, Any], replace: bool = False) -> None:
        row = self._conn.execute("SELECT props FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise SqgraphError(f"node {node_id} does not exist")
        labels = self._node_labels(node_id) if (self._vec or self._fts) else []
        vec_props = self._vector_props(labels)
        props = {} if replace else json.loads(row[0])
        for k, v in updates.items():
            if not isinstance(k, str) or not k:
                raise SqgraphError(f"property keys must be non-empty strings, got {k!r}")
            v = normalize_value(v)
            if k in vec_props:
                props.pop(k, None)
                if v is None:
                    self._vec_delete(node_id, vec_props[k])
                else:
                    for ix in vec_props[k]:
                        self._vec_upsert(ix, node_id, v)
            elif v is None:
                props.pop(k, None)
            else:
                props[k] = v
        if replace:
            for k, ixs in vec_props.items():
                if k not in updates:
                    self._vec_delete(node_id, ixs)
        self._conn.execute("UPDATE nodes SET props = ? WHERE id = ?", (dumps(props), node_id))
        self._sync_fts(node_id, labels, props)

    def _vec_read(self, ix: VectorIndex, node_id: int) -> list[float] | None:
        row = self._conn.execute(
            f"SELECT embedding FROM {ix.table} WHERE {ix.key_column} = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        blob = row[0]
        return list(struct.unpack(f"{len(blob) // 4}f", blob))

    def _w_add_labels(self, node_id: int, labels: Sequence[str]) -> None:
        labels = _labels_list(labels)
        row = self._conn.execute("SELECT props FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise SqgraphError(f"node {node_id} does not exist")
        before = set(self._node_labels(node_id)) if self._vec else set()
        self._conn.executemany(
            "INSERT OR IGNORE INTO node_labels(node_id, label) VALUES (?, ?)",
            [(node_id, lb) for lb in labels],
        )
        if self._vec:
            # embeddings stored as plain properties move into the vector tables of newly covering indexes
            new = [lb for lb in labels if lb not in before]
            props = json.loads(row[0])
            moved = False
            for prop, ixs in self._vector_props(new).items():
                value = props.get(prop)
                if isinstance(value, list):
                    for ix in ixs:
                        self._vec_upsert(ix, node_id, value)  # validates the dimensions
                    del props[prop]
                    moved = True
            if moved:
                self._conn.execute("UPDATE nodes SET props = ? WHERE id = ?", (dumps(props), node_id))
        self._sync_fts(node_id)

    def _w_remove_labels(self, node_id: int, labels: Sequence[str]) -> None:
        labels = _labels_list(labels)
        self._conn.executemany(
            "DELETE FROM node_labels WHERE node_id = ? AND label = ?", [(node_id, lb) for lb in labels]
        )
        remaining = self._node_labels(node_id)
        still_covered = set(self._vector_props(remaining)) if self._vec else set()
        restore: dict[str, list[float]] = {}
        for ix in self._vec.values():
            if ix.entity_type == NODE and ix.label in labels and ix.label not in remaining:
                if ix.property not in still_covered and ix.property not in restore:
                    vec = self._vec_read(ix, node_id)
                    if vec is not None:
                        restore[ix.property] = vec
                self._vec_delete(node_id, [ix])
        if restore:  # no index covers the property any more: keep the embedding as a plain property
            props = json.loads(self._conn.execute("SELECT props FROM nodes WHERE id = ?", (node_id,)).fetchone()[0])
            props.update(restore)
            self._conn.execute("UPDATE nodes SET props = ? WHERE id = ?", (dumps(props), node_id))
        self._sync_fts(node_id, remaining)

    def _w_delete_node(self, node_id: int, detach: bool) -> bool:
        if not self._node_exists(node_id):
            return False
        n_edges = self._conn.execute(
            "SELECT count(*) FROM edges WHERE src = ? OR dst = ?", (node_id, node_id)
        ).fetchone()[0]
        if n_edges:
            if not detach:
                raise SqgraphError(
                    f"cannot delete node {node_id}: it still has {n_edges} relationship(s) "
                    "(use detach=True / DETACH DELETE)"
                )
            if self._has_indexes(RELATIONSHIP):
                self._drop_edge_index_rows(
                    r[0]
                    for r in self._conn.execute("SELECT id FROM edges WHERE src = ? OR dst = ?", (node_id, node_id))
                )
            self._conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (node_id, node_id))
        self._vec_delete(node_id)
        for ix in self._fts.values():
            if ix.entity_type == NODE:
                self._conn.execute(f"DELETE FROM {ix.table} WHERE rowid = ?", (node_id,))
        self._conn.execute("DELETE FROM node_labels WHERE node_id = ?", (node_id,))
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        return True

    def _w_create_edge(self, src: int, dst: int, type_: str, props: dict[str, Any] | None) -> int:
        if not isinstance(type_, str) or not type_:
            raise SqgraphError("relationship type must be a non-empty string")
        props = normalize_props(props)
        for nid in (src, dst):
            if not self._node_exists(nid):
                raise SqgraphError(f"node {nid} does not exist")
        vec_props = self._edge_vector_props(type_) if self._vec else {}
        vec_values = {k: props.pop(k) for k in list(props) if k in vec_props}
        cur = self._conn.execute(
            "INSERT INTO edges(src, dst, type, props) VALUES (?, ?, ?, ?)",
            (src, dst, type_, dumps(props)),
        )
        eid = cur.lastrowid
        assert eid is not None
        for k, v in vec_values.items():
            for ix in vec_props[k]:
                self._vec_upsert(ix, eid, v)
        self._sync_edge_fts(eid, type_, props)
        return eid

    def _w_set_edge_props(self, edge_id: int, updates: dict[str, Any], replace: bool = False) -> None:
        row = self._conn.execute("SELECT type, props FROM edges WHERE id = ?", (edge_id,)).fetchone()
        if row is None:
            raise SqgraphError(f"relationship {edge_id} does not exist")
        type_ = row[0]
        vec_props = self._edge_vector_props(type_) if self._vec else {}
        props = {} if replace else json.loads(row[1])
        for k, v in updates.items():
            v = normalize_value(v)
            if k in vec_props:
                props.pop(k, None)
                if v is None:
                    self._vec_delete(edge_id, vec_props[k])
                else:
                    for ix in vec_props[k]:
                        self._vec_upsert(ix, edge_id, v)
            elif v is None:
                props.pop(k, None)
            else:
                props[k] = v
        if replace:
            for k, ixs in vec_props.items():
                if k not in updates:
                    self._vec_delete(edge_id, ixs)
        self._conn.execute("UPDATE edges SET props = ? WHERE id = ?", (dumps(props), edge_id))
        self._sync_edge_fts(edge_id, type_, props)

    def _w_delete_edge(self, edge_id: int) -> bool:
        if self._has_indexes(RELATIONSHIP):
            self._drop_edge_index_rows([edge_id])
        return self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,)).rowcount > 0

    # ------------------------------------------------------------------ public API: nodes

    def create_node(self, labels: Sequence[str] | str | None = None, props: dict[str, Any] | None = None) -> Node:
        """Create a node with ``labels`` (list or single string) and properties."""
        self._require_writable()
        with self._atomic():
            nid = self._w_create_node(_labels_list(labels), props or {})
            return self._load_nodes([nid])[nid]

    def get_node(self, id: int) -> Node | None:
        """Return the node with this id, or ``None``."""
        with self._atomic(write=False):
            return self._load_nodes([id]).get(id)

    def update_node(self, id: int, props: dict[str, Any], replace: bool = False) -> Node:
        """Merge ``props`` into the node's properties (``None`` removes a key).

        With ``replace=True`` the existing properties are discarded first.
        """
        self._require_writable()
        with self._atomic():
            self._w_set_node_props(id, props, replace)
            return self._load_nodes([id])[id]

    def add_labels(self, id: int, labels: Sequence[str] | str) -> Node:
        """Add labels to a node."""
        self._require_writable()
        with self._atomic():
            self._w_add_labels(id, _labels_list(labels))
            return self._load_nodes([id])[id]

    def remove_labels(self, id: int, labels: Sequence[str] | str) -> Node:
        """Remove labels from a node."""
        self._require_writable()
        with self._atomic():
            if not self._node_exists(id):
                raise SqgraphError(f"node {id} does not exist")
            self._w_remove_labels(id, _labels_list(labels))
            return self._load_nodes([id])[id]

    def delete_node(self, id: int, detach: bool = False) -> bool:
        """Delete a node. Raises :class:`SqgraphError` if it has relationships and ``detach`` is false.

        Returns ``False`` if the node did not exist.
        """
        self._require_writable()
        with self._atomic():
            return self._w_delete_node(id, detach)

    # ------------------------------------------------------------------ public API: edges

    def create_edge(self, src: int, dst: int, type: str, props: dict[str, Any] | None = None) -> Relationship:
        """Create a relationship ``(src)-[:type]->(dst)``."""
        self._require_writable()
        with self._atomic():
            eid = self._w_create_edge(src, dst, type, props)
            return self._load_edges([eid])[eid]

    def get_edge(self, id: int) -> Relationship | None:
        """Return the relationship with this id, or ``None``."""
        with self._atomic(write=False):
            return self._load_edges([id]).get(id)

    def update_edge(self, id: int, props: dict[str, Any], replace: bool = False) -> Relationship:
        """Merge (or replace) relationship properties."""
        self._require_writable()
        with self._atomic():
            self._w_set_edge_props(id, props, replace)
            return self._load_edges([id])[id]

    def delete_edge(self, id: int) -> bool:
        """Delete a relationship; returns ``False`` if it did not exist."""
        self._require_writable()
        with self._atomic():
            return self._w_delete_edge(id)

    # ------------------------------------------------------------------ bulk helpers

    def create_nodes(self, items: Iterable[tuple[Sequence[str] | str | None, dict[str, Any] | None]]) -> list[int]:
        """Bulk-create nodes from ``(labels, props)`` pairs; returns the new ids in order."""
        self._require_writable()
        prepared = [(_labels_list(lb), normalize_props(pr)) for lb, pr in items]
        if not prepared:
            return []
        with self._atomic():
            if self._has_indexes(NODE):
                return [self._w_create_node(lb, pr) for lb, pr in prepared]
            start = self._conn.execute("SELECT coalesce(max(id), 0) FROM nodes").fetchone()[0]
            ids = [start + 1 + i for i in range(len(prepared))]
            self._conn.executemany(
                "INSERT INTO nodes(id, props) VALUES (?, ?)",
                [(i, dumps(pr)) for i, (_, pr) in zip(ids, prepared)],
            )
            self._conn.executemany(
                "INSERT INTO node_labels(node_id, label) VALUES (?, ?)",
                [(i, lb) for i, (labels, _) in zip(ids, prepared) for lb in labels],
            )
            return ids

    def create_edges(self, items: Iterable[tuple[int, int, str] | tuple[int, int, str, dict[str, Any] | None]]) -> list[int]:
        """Bulk-create relationships from ``(src, dst, type[, props])``; returns the new ids."""
        self._require_writable()
        prepared: list[tuple[int, int, str, dict[str, Any]]] = []
        for it in items:
            src, dst, typ = it[0], it[1], it[2]
            props = it[3] if len(it) > 3 else None  # type: ignore[misc]
            if not isinstance(typ, str) or not typ:
                raise SqgraphError("relationship type must be a non-empty string")
            prepared.append((src, dst, typ, normalize_props(props)))
        if not prepared:
            return []
        with self._atomic():
            if self._has_indexes(RELATIONSHIP):
                return [self._w_create_edge(*p) for p in prepared]
            start = self._conn.execute("SELECT coalesce(max(id), 0) FROM edges").fetchone()[0]
            ids = [start + 1 + i for i in range(len(prepared))]
            try:
                self._conn.executemany(
                    "INSERT INTO edges(id, src, dst, type, props) VALUES (?, ?, ?, ?, ?)",
                    [(i, s, d, t, dumps(pr)) for i, (s, d, t, pr) in zip(ids, prepared)],
                )
            except sqlite3.IntegrityError as exc:
                raise SqgraphError(f"cannot create relationships: {exc} (does every endpoint exist?)") from exc
            return ids

    # ------------------------------------------------------------------ traversal

    def neighbors(
        self,
        node_id: int,
        types: Sequence[str] | str | None = None,
        direction: str = "out",
        depth: int = 1,
        labels: Sequence[str] | str | None = None,
    ) -> list[Node]:
        """Distinct nodes reachable from ``node_id`` in 1..``depth`` hops (start node excluded).

        ``direction`` is ``"out"``, ``"in"`` or ``"both"``; ``types`` restricts relationship
        types; ``labels`` keeps only nodes having at least one of the given labels.
        Cycle-safe (recursive CTE over ``(node, depth)``). Result is ordered by id.
        """
        if direction not in ("out", "in", "both"):
            raise SqgraphError("direction must be 'out', 'in' or 'both'")
        if not isinstance(depth, int) or depth < 1:
            raise SqgraphError("depth must be an integer >= 1")
        type_list = _labels_list(types)
        label_list = _labels_list(labels)
        params: list[Any] = [node_id]
        if direction == "out":
            adj, nxt = "e.src = r.id", "e.dst"
        elif direction == "in":
            adj, nxt = "e.dst = r.id", "e.src"
        else:
            adj, nxt = "(e.src = r.id OR e.dst = r.id)", "CASE WHEN e.src = r.id THEN e.dst ELSE e.src END"
        type_sql = ""
        if type_list:
            type_sql = f" AND e.type IN ({', '.join('?' for _ in type_list)})"
        sql = (
            "WITH RECURSIVE r(id, d) AS ("
            "SELECT ?, 0 UNION "
            f"SELECT {nxt}, r.d + 1 FROM r JOIN edges e ON {adj} WHERE r.d < ?{type_sql}"
            ") SELECT DISTINCT r.id FROM r WHERE r.id <> ?"
        )
        params += [depth, *type_list, node_id]
        if label_list:
            sql += (
                " AND EXISTS (SELECT 1 FROM node_labels l WHERE l.node_id = r.id AND l.label IN "
                f"({', '.join('?' for _ in label_list)}))"
            )
            params += label_list
        sql += " ORDER BY r.id"
        with self._atomic(write=False):
            ids = [r[0] for r in self._conn.execute(sql, params)]
            nodes = self._load_nodes(ids)
            return [nodes[i] for i in ids if i in nodes]

    # ------------------------------------------------------------------ property indexes

    def create_property_index(
        self, label: str, property: str, if_not_exists: bool = True, name: str | None = None
    ) -> str:
        """Create an expression index on ``json_extract(props, '$.property')``.

        Returns the index name: ``name`` if given (an identifier), otherwise a generated,
        collision-free ``prop_<label>_<property>_<hash>``. The index covers all nodes (labels are
        recorded for bookkeeping only); it is used by equality lookups such as
        ``MATCH (n:L {prop: $v})``.
        """
        return self._create_property_index(NODE, label, property, if_not_exists, name)

    def create_relationship_property_index(
        self, type: str, property: str, if_not_exists: bool = True, name: str | None = None
    ) -> str:
        """Create an expression index on ``json_extract(props, '$.property')`` of the ``edges`` table.

        The relationship counterpart of :meth:`create_property_index`: the generated name is
        ``rprop_<type>_<property>_<hash>``. The index covers all relationships (the type is recorded
        for bookkeeping only, as SQLite cannot use a partial index for a bound type); it is used by
        ``MATCH ()-[r:T {prop: $v}]->()`` and ``WHERE r.prop = $v``.
        """
        return self._create_property_index(RELATIONSHIP, type, property, if_not_exists, name)

    def _create_property_index(
        self, entity_type: str, label: str, property: str, if_not_exists: bool, name: str | None
    ) -> str:
        self._require_writable()
        rel = entity_type == RELATIONSHIP
        if not is_identifier(property):
            raise SqgraphError("property indexes require a simple identifier property name")
        if not isinstance(label, str) or not label:
            raise SqgraphError("relationship type must be a non-empty string" if rel else "label must be a non-empty string")
        if name is None:
            digest = hashlib.sha1((f"rel\0{label}\0{property}" if rel else f"{label}\0{property}").encode()).hexdigest()[:8]
            safe = re.sub(r"[^A-Za-z0-9_]", "_", label)
            name = f"{'rprop' if rel else 'prop'}_{safe}_{property}_{digest}"
        elif not is_identifier(name):
            raise SqgraphError(f"invalid index name {name!r}: use [A-Za-z_][A-Za-z0-9_]*")
        physical = f"idx_p_{name}"
        with self._atomic():
            if name in self._vec or name in self._fts:
                raise SqgraphError(f"an index named '{name}' already exists")
            exists = self._conn.execute(
                "SELECT label, property, entity_type FROM _property_indexes WHERE name = ?", (name,)
            ).fetchone()
            if exists:
                if not if_not_exists:
                    raise SqgraphError(f"index '{name}' already exists")
                if tuple(exists) != (label, property, entity_type):
                    raise SqgraphError(
                        f"index '{name}' already exists for a different "
                        f"{'relationship type' if exists[2] == RELATIONSHIP else 'label'}/property"
                    )
                return name
            table = "edges" if rel else "nodes"
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {physical} ON {table}(json_extract(props, '$.{property}'))"
            )
            self._conn.execute(
                "INSERT INTO _property_indexes(name, label, property, entity_type) VALUES (?, ?, ?, ?)",
                (name, label, property, entity_type),
            )
            self._conn.execute("ANALYZE")
        return name

    # ------------------------------------------------------------------ vector indexes

    def create_vector_index(
        self,
        name: str,
        label: str,
        property: str = "embedding",
        dimensions: int | None = None,
        similarity: str = "cosine",
        if_not_exists: bool = False,
    ) -> VectorIndex | None:
        """Create a vector index (backed by a sqlite-vec ``vec0`` table).

        Embeddings for nodes labelled ``label`` in property ``property`` live only in the
        vector table. Any list already stored under that property in the node JSON is migrated.
        """
        return self._create_vector_index(NODE, name, label, property, dimensions, similarity, if_not_exists)

    def create_relationship_vector_index(
        self,
        name: str,
        type: str,
        property: str = "embedding",
        dimensions: int | None = None,
        similarity: str = "cosine",
        if_not_exists: bool = False,
    ) -> VectorIndex | None:
        """Create a vector index over relationships of type ``type`` (vec0 table keyed by ``edge_id``).

        Like node indexes, the embedding in ``property`` lives only in the vector table (so
        ``RETURN r.embedding`` is null); a list already stored under that property in the
        relationship JSON is migrated.
        """
        return self._create_vector_index(RELATIONSHIP, name, type, property, dimensions, similarity, if_not_exists)

    def _create_vector_index(
        self,
        entity_type: str,
        name: str,
        label: str,
        property: str,
        dimensions: int | None,
        similarity: str,
        if_not_exists: bool,
    ) -> VectorIndex | None:
        self._require_writable()
        rel = entity_type == RELATIONSHIP
        if not is_identifier(name):
            raise SqgraphError(f"invalid index name {name!r}: use [A-Za-z_][A-Za-z0-9_]*")
        if not isinstance(label, str) or not label or not isinstance(property, str) or not property:
            raise SqgraphError(
                "type and property must be non-empty strings" if rel else "label and property must be non-empty strings"
            )
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
            raise SqgraphError("dimensions must be a positive integer")
        sim = {"cosine": "cosine", "euclidean": "euclidean", "l2": "euclidean"}.get(str(similarity).lower())
        if sim is None:
            raise SqgraphError("similarity must be 'cosine' or 'euclidean'")
        with self._atomic():
            if name in self._vec or self._conn.execute(
                "SELECT 1 FROM _vector_indexes WHERE name = ?", (name,)
            ).fetchone():
                if if_not_exists:
                    old = self._vec.get(name)
                    if old is not None and old.entity_type != entity_type:
                        raise SqgraphError(f"vector index '{name}' already exists for {old.entity_type}s")
                    return old
                raise SqgraphError(f"vector index '{name}' already exists")
            metric = "cosine" if sim == "cosine" else "l2"
            ix = VectorIndex(name, label, property, dimensions, sim, entity_type)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE {ix.table} USING vec0("
                f"{ix.key_column} INTEGER PRIMARY KEY, embedding float[{int(dimensions)}] distance_metric={metric})"
            )
            self._conn.execute(
                "INSERT INTO _vector_indexes(name, label, property, dimensions, similarity, entity_type) "
                "VALUES (?,?,?,?,?,?)",
                (name, label, property, dimensions, sim, entity_type),
            )
            self._vec[name] = ix
            # migrate embeddings that were stored as plain properties before the index existed
            path = self._json_path(property)
            if rel:
                rows = self._conn.execute(
                    "SELECT id, json_extract(props, ?) FROM edges WHERE type = ? AND json_type(props, ?) = 'array'",
                    (path, label, path),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT n.id, json_extract(n.props, ?) FROM nodes n "
                    "JOIN node_labels l ON l.node_id = n.id AND l.label = ? "
                    "WHERE json_type(n.props, ?) = 'array'",
                    (path, label, path),
                ).fetchall()
            for eid, emb in rows:
                self._vec_upsert(ix, eid, json.loads(emb))
                self._conn.execute(
                    f"UPDATE {'edges' if rel else 'nodes'} SET props = json_remove(props, ?) WHERE id = ?", (path, eid)
                )
            return ix

    @staticmethod
    def _json_path(key: str) -> str:
        return '$."' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def vector_indexes(self) -> list[VectorIndex]:
        """All vector indexes (node and relationship)."""
        return list(self._vec.values())

    def _get_vector_index(self, name: str) -> VectorIndex:
        try:
            return self._vec[name]
        except KeyError:
            raise SqgraphError(f"no such vector index: {name!r}") from None

    def set_node_vector(self, node_id: int, property: str, vector: Any) -> None:
        """Store an embedding for a node in every vector index matching its labels and ``property``."""
        self._require_writable()
        with self._atomic():
            self._w_set_node_vector(node_id, property, vector)

    def _w_set_node_vector(self, node_id: int, property: str, vector: Any) -> None:
        if not self._node_exists(node_id):
            raise SqgraphError(f"node {node_id} does not exist")
        matches = self._vector_props(self._node_labels(node_id)).get(property)
        if not matches:
            raise SqgraphError(
                f"no vector index covers property {property!r} for the labels of node {node_id}"
            )
        for ix in matches:
            self._vec_upsert(ix, node_id, vector)

    def set_edge_vector(self, edge_id: int, property: str, vector: Any) -> None:
        """Store an embedding for a relationship in every vector index matching its type and ``property``."""
        self._require_writable()
        with self._atomic():
            self._w_set_edge_vector(edge_id, property, vector)

    def _w_set_edge_vector(self, edge_id: int, property: str, vector: Any) -> None:
        row = self._conn.execute("SELECT type FROM edges WHERE id = ?", (edge_id,)).fetchone()
        if row is None:
            raise SqgraphError(f"relationship {edge_id} does not exist")
        matches = self._edge_vector_props(row[0]).get(property)
        if not matches:
            raise SqgraphError(
                f"no vector index covers property {property!r} for the type of relationship {edge_id}"
            )
        for ix in matches:
            self._vec_upsert(ix, edge_id, vector)

    def vector_search(self, index_name: str, vector: Any, k: int = 10) -> list[tuple[Node | Relationship, float]]:
        """k-nearest-neighbour search; returns ``(entity, score)`` sorted by score descending.

        ``entity`` is a :class:`Node` for a node index and a :class:`Relationship` for a
        relationship index. Score follows Neo4j: cosine ``(1 + cos_sim) / 2``, euclidean
        ``1 / (1 + d^2)``.
        """
        ix = self._get_vector_index(index_name)
        if not isinstance(k, int) or k < 1:
            raise SqgraphError("k must be a positive integer")
        blob = self._vec_blob(ix, vector)
        with self._atomic(write=False):
            rows = self._conn.execute(
                f"SELECT {ix.key_column}, distance FROM {ix.table} WHERE embedding MATCH ? AND k = ?",
                (blob, k),
            ).fetchall()
            entities = self._load_entities(ix.entity_type, (r[0] for r in rows))
        out = [(entities[eid], vector_score(ix.similarity, dist)) for eid, dist in rows if eid in entities]
        out.sort(key=lambda t: -t[1])
        return out

    def _load_entities(self, entity_type: str, ids: Iterable[int]) -> dict[int, Any]:
        return self._load_edges(ids) if entity_type == RELATIONSHIP else self._load_nodes(ids)

    # ------------------------------------------------------------------ fulltext indexes

    def create_fulltext_index(
        self,
        name: str,
        labels: Sequence[str] | str,
        properties: Sequence[str] | str,
        if_not_exists: bool = False,
        tokenizer: str = "unicode61",
    ) -> FulltextIndex | None:
        """Create an FTS5-backed fulltext index over ``properties`` of nodes with any of ``labels``.

        ``tokenizer`` may be ``unicode61`` (default), ``porter``, ``ascii`` or ``trigram``
        (substring search; useful for Japanese text).
        """
        return self._create_fulltext_index(NODE, name, labels, properties, if_not_exists, tokenizer)

    def create_relationship_fulltext_index(
        self,
        name: str,
        types: Sequence[str] | str,
        properties: Sequence[str] | str,
        tokenizer: str = "unicode61",
        if_not_exists: bool = False,
    ) -> FulltextIndex | None:
        """Create an FTS5-backed fulltext index over ``properties`` of relationships of any of ``types``.

        Existing relationships of those types are indexed. ``tokenizer`` is as for
        :meth:`create_fulltext_index` (``trigram`` allowed).
        """
        return self._create_fulltext_index(RELATIONSHIP, name, types, properties, if_not_exists, tokenizer)

    def _create_fulltext_index(
        self,
        entity_type: str,
        name: str,
        labels: Sequence[str] | str,
        properties: Sequence[str] | str,
        if_not_exists: bool,
        tokenizer: str,
    ) -> FulltextIndex | None:
        self._require_writable()
        rel = entity_type == RELATIONSHIP
        if not is_identifier(name):
            raise SqgraphError(f"invalid index name {name!r}: use [A-Za-z_][A-Za-z0-9_]*")
        label_list = _labels_list(labels)
        prop_list = _labels_list(properties)
        if not label_list or not prop_list:
            raise SqgraphError(
                f"a fulltext index needs at least one {'type' if rel else 'label'} and one property"
            )
        if tokenizer not in _TOKENIZERS:
            raise SqgraphError(f"unsupported tokenizer {tokenizer!r}; choose from {sorted(_TOKENIZERS)}")
        with self._atomic():
            if name in self._fts or self._conn.execute(
                "SELECT 1 FROM _fulltext_indexes WHERE name = ?", (name,)
            ).fetchone():
                if if_not_exists:
                    old = self._fts.get(name)
                    if old is not None and old.entity_type != entity_type:
                        raise SqgraphError(f"fulltext index '{name}' already exists for {old.entity_type}s")
                    return old
                raise SqgraphError(f"fulltext index '{name}' already exists")
            ix = FulltextIndex(name, tuple(label_list), tuple(prop_list), tokenizer, entity_type)
            cols = ", ".join(ix.column(i) for i in range(len(prop_list)))
            self._conn.execute(
                f"CREATE VIRTUAL TABLE {ix.table} USING fts5({cols}, {ix.key_column} UNINDEXED, tokenize='{tokenizer}')"
            )
            self._conn.execute(
                "INSERT INTO _fulltext_indexes(name, labels, properties, tokenizer, entity_type) VALUES (?,?,?,?,?)",
                (name, json.dumps(label_list), json.dumps(prop_list), tokenizer, entity_type),
            )
            self._fts[name] = ix
            if rel:
                rows = self._conn.execute(
                    "SELECT id, type, props FROM edges WHERE type IN (SELECT value FROM json_each(?))",
                    (json.dumps(label_list),),
                ).fetchall()
                for eid, typ, props in rows:
                    self._fts_row(ix, eid, [typ], json.loads(props))
            else:
                rows = self._conn.execute(
                    "SELECT DISTINCT n.id, n.props FROM nodes n JOIN node_labels l ON l.node_id = n.id "
                    f"WHERE l.label IN (SELECT value FROM json_each(?))",
                    (json.dumps(label_list),),
                ).fetchall()
                for nid, props in rows:
                    self._fts_row(ix, nid, label_list, json.loads(props))
            return ix

    def fulltext_indexes(self) -> list[FulltextIndex]:
        """All fulltext indexes (node and relationship)."""
        return list(self._fts.values())

    def _get_fulltext_index(self, name: str) -> FulltextIndex:
        try:
            return self._fts[name]
        except KeyError:
            raise SqgraphError(f"no such fulltext index: {name!r}") from None

    def fts_match_expression(self, ix: FulltextIndex, query: str) -> str:
        """Translate a Lucene-style query string into an FTS5 MATCH expression."""
        columns = {p: ix.column(i) for i, p in enumerate(ix.properties)}
        return lucene_to_fts5(query, columns)

    def fulltext_search(self, index_name: str, query: str, k: int = 10) -> list[tuple[Node | Relationship, float]]:
        """BM25 search; returns ``(entity, score)`` with ``score = -bm25`` (higher is better).

        ``entity`` is a :class:`Node` for a node index and a :class:`Relationship` for a
        relationship index.
        """
        ix = self._get_fulltext_index(index_name)
        if not isinstance(k, int) or k < 1:
            raise SqgraphError("k must be a positive integer")
        rows = self._fts_query(ix, query, k)
        with self._atomic(write=False):
            entities = self._load_entities(ix.entity_type, (r[0] for r in rows))
        return [(entities[eid], score) for eid, score in rows if eid in entities]

    def _fts_query(self, ix: FulltextIndex, query: str, limit: int | None) -> list[tuple[int, float]]:
        match = self.fts_match_expression(ix, query)
        if not match:
            return []
        sql = (
            f"SELECT rowid, -bm25({ix.table}) AS score FROM {ix.table} "
            f"WHERE {ix.table} MATCH ? ORDER BY score DESC"
        )
        params: list[Any] = [match]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        try:
            return [(r[0], r[1]) for r in self._conn.execute(sql, params)]
        except sqlite3.OperationalError:
            params[0] = fallback_fts5(query)
            if not params[0]:
                return []
            try:
                return [(r[0], r[1]) for r in self._conn.execute(sql, params)]
            except sqlite3.OperationalError as exc:
                raise SqgraphError(f"invalid fulltext query {query!r}: {exc}") from exc

    # ------------------------------------------------------------------ index management

    def drop_index(self, name: str, if_exists: bool = False) -> bool:
        """Drop a vector, fulltext or property index by name."""
        self._require_writable()
        with self._atomic():
            if name in self._vec:
                ix = self._vec.pop(name)
                self._conn.execute(f"DROP TABLE {ix.table}")
                self._conn.execute("DELETE FROM _vector_indexes WHERE name = ?", (name,))
                return True
            if name in self._fts:
                fx = self._fts.pop(name)
                self._conn.execute(f"DROP TABLE {fx.table}")
                self._conn.execute("DELETE FROM _fulltext_indexes WHERE name = ?", (name,))
                return True
            row = self._conn.execute("SELECT 1 FROM _property_indexes WHERE name = ?", (name,)).fetchone()
            if row:
                self._conn.execute(f"DROP INDEX IF EXISTS idx_p_{name}")
                if name.startswith("idx_prop_"):  # created by an earlier version under this exact name
                    self._conn.execute(f"DROP INDEX IF EXISTS {name}")
                self._conn.execute("DELETE FROM _property_indexes WHERE name = ?", (name,))
                return True
            if if_exists:
                return False
            raise SqgraphError(f"no such index: {name!r}")

    def list_indexes(self) -> list[dict[str, Any]]:
        """Describe all indexes (``SHOW INDEXES`` equivalent)."""
        out: list[dict[str, Any]] = []
        for ix in self._vec.values():
            out.append(
                {
                    "name": ix.name,
                    "type": "VECTOR",
                    "entityType": ix.entity_type.upper(),
                    "labelsOrTypes": [ix.label],
                    "properties": [ix.property],
                    "options": {"vector.dimensions": ix.dimensions, "vector.similarity_function": ix.similarity},
                }
            )
        for fx in self._fts.values():
            out.append(
                {
                    "name": fx.name,
                    "type": "FULLTEXT",
                    "entityType": fx.entity_type.upper(),
                    "labelsOrTypes": list(fx.labels),
                    "properties": list(fx.properties),
                    "options": {"tokenizer": fx.tokenizer},
                }
            )
        for name, label, prop, entity_type in self._conn.execute(
            f"SELECT name, label, property, {self._entity_type_sql('_property_indexes')} FROM _property_indexes"
        ):
            out.append(
                {
                    "name": name,
                    "type": "RANGE",
                    "entityType": entity_type.upper(),
                    "labelsOrTypes": [label],
                    "properties": [prop],
                    "options": {},
                }
            )
        return out

    # ------------------------------------------------------------------ Cypher

    @property
    def _cypher(self) -> Any:
        if self._executor is None:
            from .cypher.executor import Executor

            self._executor = Executor(self)
        return self._executor

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a Cypher query and return the rows as dicts keyed by RETURN column name.

        The whole call is one transaction: any error rolls back all of its writes.
        """
        return self._cypher.run(cypher, params or {})

    def explain(self, cypher: str, params: dict[str, Any] | None = None) -> str:
        """Return the SQL generated for the read parts of ``cypher`` (for debugging)."""
        return self._cypher.explain(cypher, params or {})

    # ------------------------------------------------------------------ misc

    def stats(self) -> dict[str, int]:
        """Node, relationship and label counts."""
        with self._atomic(write=False):
            c = self._conn
            return {
                "nodes": c.execute("SELECT count(*) FROM nodes").fetchone()[0],
                "relationships": c.execute("SELECT count(*) FROM edges").fetchone()[0],
                "labels": c.execute("SELECT count(DISTINCT label) FROM node_labels").fetchone()[0],
            }


def vector_score(similarity: str, distance: float) -> float:
    """Convert a sqlite-vec distance to the Neo4j similarity score for ``similarity``."""
    if similarity == "cosine":
        return 1.0 - distance / 2.0
    return 1.0 / (1.0 + distance * distance)
