"""Exception hierarchy for sqgraph."""

from __future__ import annotations


class SqgraphError(Exception):
    """Base class for all sqgraph errors."""


class DimensionMismatchError(SqgraphError, ValueError):
    """A vector's length does not match the dimensions of its index."""


class CypherError(SqgraphError):
    """Base class for errors raised while handling a Cypher query."""

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        query: str | None = None,
    ) -> None:
        self.message = message
        self.line = line
        self.column = column
        self.query = query
        text = message
        if line is not None and column is not None:
            text = f"{message} (line {line}, column {column})"
        super().__init__(text)


class CypherSyntaxError(CypherError):
    """The query text is not valid (or refers to an undefined variable)."""


class CypherUnsupportedError(CypherError):
    """The query uses valid Cypher that sqgraph does not implement."""


class CypherRuntimeError(CypherError):
    """The query failed while executing (constraint, type or state problem)."""
