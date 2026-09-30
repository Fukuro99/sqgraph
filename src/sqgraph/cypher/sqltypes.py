"""Static types and compiled-expression containers used by the Cypher compiler.

SQL representation ("rep") of each kind:

* ``node`` / ``rel``  - integer id (NULL when absent)
* ``bool``            - 0 / 1
* ``int`` / ``float`` / ``str`` / ``scalar`` - native SQLite value (``scalar`` = unknown primitive)
* ``list`` / ``map`` / ``path`` - JSON text (elements encoded per the element type)
* ``any``             - JSON text of one arbitrary JSON value (SQL NULL for null)
* ``null``            - NULL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class T:
    """A static Cypher type."""

    kind: str
    elem: Optional["T"] = None  # element type for lists
    fields: Optional[tuple[tuple[str, "T"], ...]] = None  # static fields for maps

    def __repr__(self) -> str:
        if self.kind == "list":
            return f"list<{self.elem!r}>"
        return self.kind


NODE = T("node")
REL = T("rel")
PATH = T("path")
BOOL = T("bool")
INT = T("int")
FLOAT = T("float")
STR = T("str")
ANY = T("any")
NULL = T("null")
SCALAR = T("scalar")
MAP = T("map")
RAWPATH = T("rawpath")  # internal: comma-prefixed relationship id list of a variable-length hop
LIST_ANY = T("list", ANY)

ENTITY_KINDS = ("node", "rel")
JSON_KINDS = ("list", "map", "path", "any")
NATIVE_KINDS = ("int", "float", "str", "scalar", "bool", "node", "rel")


def list_of(t: T) -> T:
    return T("list", t)


def map_of(fields: dict[str, T] | None = None) -> T:
    return T("map", None, tuple(fields.items()) if fields is not None else None)


def contains_entity(t: T) -> bool:
    """True if values of type ``t`` embed node/relationship ids that need loading."""
    if t.kind in ENTITY_KINDS or t.kind == "path":
        return True
    if t.kind == "list":
        return t.elem is not None and contains_entity(t.elem)
    if t.kind == "map" and t.fields:
        return any(contains_entity(ft) for _, ft in t.fields)
    return False


@dataclass
class Ex:
    """A compiled expression: SQL text in the canonical representation of ``type``.

    ``nat`` (only meaningful for ``any``) is an alternative SQL fragment yielding the
    native SQLite scalar, used to avoid needless JSON round trips (e.g. property access).
    """

    sql: str
    type: T
    nat: Optional[str] = None
    jt: Optional[str] = None  # SQL giving json_type() of an ``any`` value (avoids re-parsing)


@dataclass
class Var:
    """A variable in scope: how to reference it in SQL and what it holds."""

    sql: str
    type: T
    props: Optional[str] = None  # SQL for its props JSON when a table alias is at hand
    cte: Optional[str] = None  # CTE the value comes from (if it is a plain CTE column)


@dataclass
class Frame:
    """The binding table produced by the previous clause (a CTE)."""

    cte: str
    cols: dict[str, tuple[str, T]]  # variable name -> (column, type)
