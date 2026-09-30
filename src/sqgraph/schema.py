"""SQL DDL for the sqgraph storage schema."""

from __future__ import annotations

import re

# 2: dropped the redundant idx_edges_type (covered by idx_edges_src)
# 3: index metadata tables gained entity_type ('node' | 'relationship')
SCHEMA_VERSION = 3

NODE = "node"
RELATIONSHIP = "relationship"

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

DDL = """
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id    INTEGER PRIMARY KEY,
    props TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS node_labels (
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    label   TEXT NOT NULL,
    PRIMARY KEY (label, node_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_node_labels_node ON node_labels(node_id);

CREATE TABLE IF NOT EXISTS edges (
    id    INTEGER PRIMARY KEY,
    src   INTEGER NOT NULL REFERENCES nodes(id),
    dst   INTEGER NOT NULL REFERENCES nodes(id),
    type  TEXT NOT NULL,
    props TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, type, dst);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, type, src);

CREATE TABLE IF NOT EXISTS _vector_indexes (
    name       TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    property   TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    similarity TEXT NOT NULL CHECK (similarity IN ('cosine', 'euclidean')),
    entity_type TEXT NOT NULL DEFAULT 'node'
);

CREATE TABLE IF NOT EXISTS _fulltext_indexes (
    name       TEXT PRIMARY KEY,
    labels     TEXT NOT NULL,
    properties TEXT NOT NULL,
    tokenizer  TEXT NOT NULL DEFAULT 'unicode61',
    entity_type TEXT NOT NULL DEFAULT 'node'
);

CREATE TABLE IF NOT EXISTS _property_indexes (
    name     TEXT PRIMARY KEY,
    label    TEXT NOT NULL,
    property TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'node'
);
"""

# index metadata tables that carry the entity_type column (added by schema version 3)
ENTITY_TYPE_TABLES = ("_vector_indexes", "_fulltext_indexes", "_property_indexes")

REQUIRED_TABLES = ("_meta", "nodes", "node_labels", "edges", "_vector_indexes", "_fulltext_indexes")


def is_identifier(name: str) -> bool:
    """True if ``name`` is a safe SQL identifier fragment (``[A-Za-z_][A-Za-z0-9_]*``)."""
    return isinstance(name, str) and IDENT_RE.fullmatch(name) is not None


def vec_table(index_name: str) -> str:
    """Name of the sqlite-vec table backing a vector index (validated identifier)."""
    if not is_identifier(index_name):
        raise ValueError(f"invalid index name: {index_name!r}")
    return f"vec_{index_name}"


def fts_table(index_name: str) -> str:
    """Name of the FTS5 table backing a fulltext index (validated identifier)."""
    if not is_identifier(index_name):
        raise ValueError(f"invalid index name: {index_name!r}")
    return f"fts_{index_name}"
