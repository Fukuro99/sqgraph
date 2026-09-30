"""AST dataclasses for the supported openCypher subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Base:
    """Common base: ``pos`` is the character offset in the query text (for error messages)."""

    pos: int = field(default=0, kw_only=True)


# --------------------------------------------------------------------- expressions


@dataclass
class Literal(Base):
    value: Any  # None | bool | int | float | str


@dataclass
class Param(Base):
    name: str


@dataclass
class Variable(Base):
    name: str


@dataclass
class Property(Base):
    base: "Expr"
    key: str


@dataclass
class Index(Base):
    base: "Expr"
    index: "Expr"


@dataclass
class Slice(Base):
    base: "Expr"
    start: Optional["Expr"]
    end: Optional["Expr"]


@dataclass
class FunctionCall(Base):
    name: str
    args: list["Expr"]
    distinct: bool = False
    star: bool = False


@dataclass
class UnaryOp(Base):
    op: str  # NOT, NEG
    operand: "Expr"


@dataclass
class BinaryOp(Base):
    op: str  # + - * / % ^ = <> < > <= >= AND OR XOR IN STARTS ENDS CONTAINS =~
    left: "Expr"
    right: "Expr"


@dataclass
class IsNull(Base):
    operand: "Expr"
    negated: bool = False


@dataclass
class LabelCheck(Base):
    operand: "Expr"
    labels: list[str]


@dataclass
class Case(Base):
    subject: Optional["Expr"]
    whens: list[tuple["Expr", "Expr"]]
    default: Optional["Expr"]


@dataclass
class ListLiteral(Base):
    items: list["Expr"]


@dataclass
class MapLiteral(Base):
    items: list[tuple[str, "Expr"]]


@dataclass
class ListComprehension(Base):
    var: str
    source: "Expr"
    where: Optional["Expr"]
    projection: Optional["Expr"]


@dataclass
class Quantifier(Base):
    kind: str  # all, any, none, single
    var: str
    source: "Expr"
    where: "Expr"


@dataclass
class MapProjection(Base):
    base: "Expr"
    items: list[tuple[str, Any]]  # ("prop", name) | ("all", None) | ("kv", (key, expr)) | ("var", name)


@dataclass
class ExistsSubquery(Base):
    patterns: list["PatternPart"]
    where: Optional["Expr"]


Expr = Union[
    Literal,
    Param,
    Variable,
    Property,
    Index,
    Slice,
    FunctionCall,
    UnaryOp,
    BinaryOp,
    IsNull,
    LabelCheck,
    Case,
    ListLiteral,
    MapLiteral,
    ListComprehension,
    Quantifier,
    MapProjection,
    ExistsSubquery,
]


# --------------------------------------------------------------------- patterns


@dataclass
class NodePattern(Base):
    var: Optional[str]
    labels: list[str]
    props: Optional[Union[MapLiteral, Param]]


@dataclass
class RelPattern(Base):
    var: Optional[str]
    types: list[str]
    direction: str  # "out" | "in" | "both"
    varlen: bool
    min_hops: int
    max_hops: Optional[int]  # None = unbounded
    props: Optional[Union[MapLiteral, Param]]


@dataclass
class PatternPart(Base):
    var: Optional[str]
    elements: list[Union[NodePattern, RelPattern]]  # node, rel, node, rel, node ...


# --------------------------------------------------------------------- clauses


@dataclass
class ProjectionItem(Base):
    expr: "Expr"
    alias: Optional[str]
    text: str  # source text of the expression (Neo4j-style column name)


@dataclass
class SortItem(Base):
    expr: "Expr"
    descending: bool = False


@dataclass
class Projection(Base):
    items: list[ProjectionItem]
    star: bool = False
    distinct: bool = False
    order_by: list[SortItem] = field(default_factory=list)
    skip: Optional["Expr"] = None
    limit: Optional["Expr"] = None
    where: Optional["Expr"] = None


@dataclass
class Match(Base):
    patterns: list[PatternPart]
    where: Optional["Expr"]
    optional: bool = False


@dataclass
class Unwind(Base):
    expr: "Expr"
    var: str


@dataclass
class With(Base):
    projection: Projection


@dataclass
class Return(Base):
    projection: Projection


@dataclass
class Create(Base):
    patterns: list[PatternPart]


@dataclass
class SetItem(Base):
    kind: str  # "prop" | "replace" | "merge" | "labels"
    var: str
    key: Optional[str] = None
    expr: Optional["Expr"] = None
    labels: list[str] = field(default_factory=list)


@dataclass
class Merge(Base):
    pattern: PatternPart
    on_create: list[SetItem] = field(default_factory=list)
    on_match: list[SetItem] = field(default_factory=list)


@dataclass
class SetClause(Base):
    items: list[SetItem]


@dataclass
class RemoveItem(Base):
    kind: str  # "prop" | "labels"
    var: str
    key: Optional[str] = None
    labels: list[str] = field(default_factory=list)


@dataclass
class Remove(Base):
    items: list[RemoveItem]


@dataclass
class Delete(Base):
    exprs: list["Expr"]
    detach: bool = False


@dataclass
class YieldItem(Base):
    name: str
    alias: Optional[str]


@dataclass
class Call(Base):
    name: str  # dotted procedure name
    args: list["Expr"]
    yields: Optional[list[YieldItem]]  # None: no YIELD clause
    where: Optional["Expr"] = None


Clause = Union[Match, Unwind, With, Return, Create, Merge, SetClause, Remove, Delete, Call]


@dataclass
class SingleQuery(Base):
    clauses: list[Clause]


@dataclass
class Query(Base):
    parts: list[SingleQuery]
    union_all: bool = False  # True if the queries are joined with UNION ALL


# --------------------------------------------------------------------- schema commands


@dataclass
class CreateVectorIndex(Base):
    name: Optional[str]
    if_not_exists: bool
    label: str  # the relationship type when entity == "relationship"
    property: str
    options: Optional[MapLiteral]
    entity: str = "node"  # "node" | "relationship"


@dataclass
class CreateFulltextIndex(Base):
    name: Optional[str]
    if_not_exists: bool
    labels: list[str]  # relationship types when entity == "relationship"
    properties: list[str]
    options: Optional[MapLiteral]
    entity: str = "node"  # "node" | "relationship"


@dataclass
class CreatePropertyIndex(Base):
    name: Optional[str]
    if_not_exists: bool
    label: str  # the relationship type when entity == "relationship"
    properties: list[str]
    entity: str = "node"  # "node" | "relationship"


@dataclass
class DropIndex(Base):
    name: str
    if_exists: bool


@dataclass
class ShowIndexes(Base):
    pass


Statement = Union[
    Query, CreateVectorIndex, CreateFulltextIndex, CreatePropertyIndex, DropIndex, ShowIndexes
]
