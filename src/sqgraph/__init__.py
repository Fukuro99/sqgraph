"""sqgraph: an embedded property graph database on SQLite + sqlite-vec, built for GraphRAG."""

from .db import FulltextIndex, GraphDB, VectorIndex
from .errors import (
    CypherError,
    CypherRuntimeError,
    CypherSyntaxError,
    CypherUnsupportedError,
    DimensionMismatchError,
    SqgraphError,
)
from .types import Node, Path, Relationship

__all__ = [
    "GraphDB",
    "Node",
    "Relationship",
    "Path",
    "VectorIndex",
    "FulltextIndex",
    "SqgraphError",
    "CypherError",
    "CypherSyntaxError",
    "CypherUnsupportedError",
    "CypherRuntimeError",
    "DimensionMismatchError",
]

__version__ = "0.1.0"
