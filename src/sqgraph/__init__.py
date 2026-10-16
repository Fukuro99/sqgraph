"""sqgraph: an embedded property graph database on SQLite + sqlite-vec, built for GraphRAG."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

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

try:  # single source of truth: the version in pyproject.toml, as recorded in the installed distribution metadata
    __version__ = _dist_version("sqgraph")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0+unknown"
