"""Query execution: read segments run as SQL, write clauses run in Python over the rows."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from .. import functions
from ..db import GraphDB, normalize_value
from ..errors import (
    CypherRuntimeError,
    CypherSyntaxError,
    CypherUnsupportedError,
    SqgraphError,
)
from ..types import Node, Path, Relationship
from . import ast as A
from .compiler import Compiler, Segment
from .parser import parse
from .sqltypes import ANY, INT, NODE, REL, T, contains_entity, list_of

_WRITE_PROCS = {
    "db.create.setnodevectorproperty",
    "db.create.setvectorproperty",
    "db.index.vector.createnodeindex",
    "db.index.fulltext.createnodeindex",
    "db.create.setrelationshipvectorproperty",
}


def _is_write(cl: A.Clause) -> bool:
    if isinstance(cl, (A.Create, A.Merge, A.SetClause, A.Remove, A.Delete)):
        return True
    return isinstance(cl, A.Call) and cl.name.lower() in _WRITE_PROCS


# ---------------------------------------------------------------- value plumbing


def json_cell(v: Any, t: T) -> str:
    """JSON text of one internal value, for injecting rows back into SQL."""
    if v is None:
        return "null"
    if t.kind in ("list", "map", "path", "any"):
        return v if isinstance(v, str) else json.dumps(v)
    if isinstance(v, float):
        return "null" if (math.isnan(v) or math.isinf(v)) else repr(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return json.dumps(v)


def rows_json(cols: list[tuple[str, T]], rows: list[list[Any]]) -> str:
    types = [t for _, t in cols]
    return "[" + ",".join("[" + ",".join(json_cell(v, t) for v, t in zip(r, types)) + "]" for r in rows) + "]"


@dataclass
class _NodeRef:
    id: int


@dataclass
class _RelRef:
    id: int


@dataclass
class _PathRef:
    nodes: list[_NodeRef]
    rels: list[_RelRef]


class Decoder:
    """Turns raw SQL values into Python results (Node/Relationship/Path objects, bools, dicts)."""

    def __init__(self, db: GraphDB) -> None:
        self.db = db
        self.nodes: set[int] = set()
        self.rels: set[int] = set()

    def prep(self, v: Any, t: T) -> Any:
        if v is None:
            return None
        k = t.kind
        if k == "node":
            self.nodes.add(v)
            return _NodeRef(v)
        if k == "rel":
            self.rels.add(v)
            return _RelRef(v)
        if k == "bool":
            return bool(v)
        if k in ("int", "float", "str", "scalar", "null", "rawpath"):
            return v
        if not isinstance(v, str):
            return v
        return self.prep_json(json.loads(v), t)

    def prep_json(self, obj: Any, t: T) -> Any:
        if obj is None:
            return None
        k = t.kind
        if k == "node":
            self.nodes.add(obj)
            return _NodeRef(obj)
        if k == "rel":
            self.rels.add(obj)
            return _RelRef(obj)
        if k == "bool":
            return bool(obj)
        if k == "list":
            et = t.elem or ANY
            return [self.prep_json(x, et) for x in obj]
        if k == "map" and t.fields:
            ft = dict(t.fields)
            return {key: self.prep_json(val, ft.get(key, ANY)) for key, val in obj.items()}
        if k == "path":
            for i in obj["n"]:
                self.nodes.add(i)
            for i in obj["r"]:
                self.rels.add(i)
            return _PathRef([_NodeRef(i) for i in obj["n"]], [_RelRef(i) for i in obj["r"]])
        return obj

    def load(self) -> tuple[dict[int, Node], dict[int, Relationship]]:
        return self.db._load_nodes(self.nodes), self.db._load_edges(self.rels)

    def resolve(self, obj: Any, nodes: dict[int, Node], rels: dict[int, Relationship]) -> Any:
        if isinstance(obj, _NodeRef):
            n = nodes.get(obj.id)
            if n is None:
                raise CypherRuntimeError(f"Node {obj.id} does not exist (it was deleted in this query)")
            return n
        if isinstance(obj, _RelRef):
            r = rels.get(obj.id)
            if r is None:
                raise CypherRuntimeError(f"Relationship {obj.id} does not exist (it was deleted in this query)")
            return r
        if isinstance(obj, _PathRef):
            return Path(
                [self.resolve(n, nodes, rels) for n in obj.nodes],
                [self.resolve(r, nodes, rels) for r in obj.rels],
            )
        if isinstance(obj, list):
            return [self.resolve(x, nodes, rels) for x in obj]
        if isinstance(obj, dict):
            return {k: self.resolve(v, nodes, rels) for k, v in obj.items()}
        return obj


@dataclass
class EntityVal:
    """A node/relationship id produced by an evaluated expression."""

    kind: str
    id: int


def plain_decode(v: Any, t: T) -> Any:
    """Decode an evaluated value for use as a property/map/id (entities stay as ids)."""
    if v is None:
        return None
    k = t.kind
    if k in ("node", "rel"):
        return EntityVal(k, v)
    if k == "bool":
        return bool(v)
    if k in ("list", "map", "path", "any") and isinstance(v, str):
        obj = json.loads(v)
        if contains_entity(t):
            return _EntityContainer(obj, t)
        return obj
    return v


@dataclass
class _EntityContainer:
    obj: Any
    type: T


def const_eval(e: A.Expr, params: dict[str, Any]) -> Any:
    """Evaluate a literal/parameter/map/list expression at plan time."""
    if isinstance(e, A.Literal):
        return e.value
    if isinstance(e, A.Param):
        if e.name not in params:
            raise CypherRuntimeError(f"Missing parameter: ${e.name}")
        return params[e.name]
    if isinstance(e, A.MapLiteral):
        return {k: const_eval(v, params) for k, v in e.items}
    if isinstance(e, A.ListLiteral):
        return [const_eval(v, params) for v in e.items]
    if isinstance(e, A.UnaryOp) and e.op == "NEG":
        return -const_eval(e.operand, params)
    raise CypherUnsupportedError("only literals and parameters are allowed here")


def _pattern_vars(patterns: list[A.PatternPart]) -> list[tuple[str, T]]:
    out: list[tuple[str, T]] = []
    for part in patterns:
        for el in part.elements:
            if el.var:
                out.append((el.var, NODE if isinstance(el, A.NodePattern) else REL))
    return out


class Executor:
    """Runs parsed Cypher against a :class:`GraphDB`."""

    def __init__(self, db: GraphDB) -> None:
        self.db = db

    # ------------------------------------------------------------ entry points

    def run(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        stmt = parse(cypher)
        try:
            if isinstance(stmt, A.Query):
                return self._run_query(stmt, cypher, params)
            return self._run_schema(stmt, cypher, params)
        except sqlite3.Error as exc:
            msg = functions.last_error[0] if functions.last_error and "user-defined function" in str(exc) else str(exc)
            functions.last_error.clear()
            raise CypherRuntimeError(f"query failed: {msg}", query=cypher) from exc

    def explain(self, cypher: str, params: dict[str, Any]) -> str:
        stmt = parse(cypher)
        if not isinstance(stmt, A.Query):
            return "-- schema command (executed in Python)"
        out: list[str] = []
        if len(stmt.parts) > 1:
            comp = Compiler(self.db, params, cypher)
            seg = comp.compile_union([p.clauses for p in stmt.parts], stmt.union_all)
            return seg.sql
        cols: list[tuple[str, T]] = []
        injected = False
        for i, (kind, payload) in enumerate(self._split(stmt.parts[0].clauses)):
            if kind == "read":
                comp = Compiler(self.db, params, cypher)
                seg = comp.compile_segment(payload, cols, injected)
                out.append(f"-- read segment {i}" + (" (input rows bound as :rows)" if injected else ""))
                out.append(seg.sql)
                cols = seg.columns
            else:
                out.append(f"-- write step {i}: {type(payload).__name__} (executed in Python)")
                cols = self._cols_after_write(payload, cols)
            injected = True
        return "\n".join(out)

    # ------------------------------------------------------------ query driver

    @staticmethod
    def _split(clauses: list[A.Clause]) -> list[tuple[str, Any]]:
        steps: list[tuple[str, Any]] = []
        cur: list[A.Clause] = []
        for cl in clauses:
            if _is_write(cl):
                if cur:
                    steps.append(("read", cur))
                    cur = []
                steps.append(("write", cl))
            else:
                cur.append(cl)
        if cur:
            steps.append(("read", cur))
        return steps

    def _validate(self, sq: A.SingleQuery, text: str) -> None:
        cls = sq.clauses
        for i, cl in enumerate(cls):
            if isinstance(cl, A.Return) and i != len(cls) - 1:
                raise self._syn("RETURN can only be used at the end of the query", cl, text)
        last = cls[-1]
        if isinstance(last, (A.Match, A.Unwind, A.With)):
            raise self._syn(
                f"Query cannot conclude with {type(last).__name__.upper()} "
                "(must be a RETURN clause, an update clause or a procedure call)",
                last,
                text,
            )

    @staticmethod
    def _syn(msg: str, node: Any, text: str) -> CypherSyntaxError:
        pos = getattr(node, "pos", 0)
        return CypherSyntaxError(
            msg, line=text.count("\n", 0, pos) + 1, column=pos - (text.rfind("\n", 0, pos) + 1) + 1, query=text
        )

    def _run_query(self, q: A.Query, text: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        for sq in q.parts:
            self._validate(sq, text)
        has_write = any(_is_write(cl) for sq in q.parts for cl in sq.clauses)
        if has_write and self.db.read_only:
            raise CypherRuntimeError("Cannot execute a write query on a read-only database", query=text)
        if len(q.parts) > 1:
            if has_write:
                raise CypherUnsupportedError("write clauses inside UNION queries are not supported", query=text)
            comp = Compiler(self.db, params, text)
            seg = comp.compile_union([p.clauses for p in q.parts], q.union_all)
            with self.db._atomic(write=False):
                return self._run_final(seg, None, None)
        steps = self._split(q.parts[0].clauses)
        with self.db._atomic(write=has_write):
            return self._run_steps(steps, text, params)

    def _run_steps(self, steps: list[tuple[str, Any]], text: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        cols: list[tuple[str, T]] = []
        rows: list[list[Any]] | None = None
        for kind, payload in steps:
            if kind == "read":
                comp = Compiler(self.db, params, text)
                seg = comp.compile_segment(payload, cols, rows is not None, whole_query=len(steps) == 1)
                if seg.final:
                    return self._run_final(seg, cols if rows is not None else None, rows)
                binds = dict(seg.binds)
                if rows is not None:
                    binds["rows"] = rows_json(cols, rows)
                raw = self.db._conn.execute(seg.sql, binds).fetchall()
                cols = seg.columns
                rows = [list(r) if seg.columns else [] for r in raw]
            else:
                if rows is None:
                    rows = [[]]
                w = _Writer(self, text, params)
                cols, rows = w.run(payload, cols, rows)
        return []

    def _run_final(
        self, seg: Segment, cols: list[tuple[str, T]] | None, rows: list[list[Any]] | None
    ) -> list[dict[str, Any]]:
        binds = dict(seg.binds)
        if rows is not None and cols is not None:
            binds["rows"] = rows_json(cols, rows)
        raw = self.db._conn.execute(seg.sql, binds).fetchall()
        return self.decode_rows(seg.columns, raw)

    def decode_rows(self, columns: list[tuple[str, T]], raw: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        dec = Decoder(self.db)
        types = [t for _, t in columns]
        names = [n for n, _ in columns]
        prepped = [[dec.prep(v, t) for v, t in zip(r, types)] for r in raw]
        if dec.nodes or dec.rels:
            nodes, rels = dec.load()
            needs = [contains_entity(t) for t in types]
            prepped = [
                [dec.resolve(v, nodes, rels) if need else v for v, need in zip(r, needs)] for r in prepped
            ]
        return [dict(zip(names, r)) for r in prepped]

    # ------------------------------------------------------------ helper for writers

    def _cols_after_write(self, cl: A.Clause, cols: list[tuple[str, T]]) -> list[tuple[str, T]]:
        have = {n for n, _ in cols}
        if isinstance(cl, A.Create):
            return cols + [(n, t) for n, t in _pattern_vars(cl.patterns) if n not in have and not have.add(n)]  # type: ignore[func-returns-value]
        if isinstance(cl, A.Merge):
            return cols + [(n, t) for n, t in _pattern_vars([cl.pattern]) if n not in have and not have.add(n)]  # type: ignore[func-returns-value]
        if isinstance(cl, A.Call) and cl.yields:
            return cols + [(y.alias or y.name, NODE) for y in cl.yields]
        return cols

    # ------------------------------------------------------------ schema commands

    def _run_schema(self, stmt: A.Statement, text: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        db = self.db
        if db.read_only and not isinstance(stmt, A.ShowIndexes):
            raise CypherRuntimeError("Cannot execute a schema command on a read-only database", query=text)
        try:
            if isinstance(stmt, A.ShowIndexes):
                return [
                    {k: v for k, v in ix.items() if k != "options"} for ix in db.list_indexes()
                ]
            if isinstance(stmt, A.CreateVectorIndex):
                opts = const_eval(stmt.options, params) if stmt.options else {}
                cfg = opts.get("indexConfig", {}) if isinstance(opts, dict) else {}
                dims = cfg.get("vector.dimensions")
                if dims is None:
                    raise CypherRuntimeError("vector indexes require OPTIONS {indexConfig: {`vector.dimensions`: N}}")
                sim = cfg.get("vector.similarity_function", "cosine")
                if stmt.entity == "relationship":
                    name = stmt.name or f"vector_rel_{stmt.label}_{stmt.property}"
                    db.create_relationship_vector_index(
                        name, stmt.label, stmt.property, dims, sim, if_not_exists=stmt.if_not_exists
                    )
                    return []
                name = stmt.name or f"vector_{stmt.label}_{stmt.property}"
                db.create_vector_index(name, stmt.label, stmt.property, dims, sim, if_not_exists=stmt.if_not_exists)
                return []
            if isinstance(stmt, A.CreateFulltextIndex):
                opts = const_eval(stmt.options, params) if stmt.options else {}
                cfg = opts.get("indexConfig", {}) if isinstance(opts, dict) else {}
                tok = cfg.get("fulltext.tokenizer", "unicode61")
                if stmt.entity == "relationship":
                    name = stmt.name or "fulltext_rel_" + "_".join(stmt.labels) + "_" + "_".join(stmt.properties)
                    db.create_relationship_fulltext_index(
                        name, stmt.labels, stmt.properties, tokenizer=tok, if_not_exists=stmt.if_not_exists
                    )
                    return []
                name = stmt.name or "fulltext_" + "_".join(stmt.labels) + "_" + "_".join(stmt.properties)
                db.create_fulltext_index(
                    name, stmt.labels, stmt.properties, if_not_exists=stmt.if_not_exists, tokenizer=tok
                )
                return []
            if isinstance(stmt, A.CreatePropertyIndex):
                if len(stmt.properties) != 1:
                    raise CypherUnsupportedError("composite property indexes are not supported", query=text)
                create = (
                    db.create_relationship_property_index
                    if stmt.entity == "relationship"
                    else db.create_property_index
                )
                create(stmt.label, stmt.properties[0], if_not_exists=stmt.if_not_exists, name=stmt.name)
                return []
            if isinstance(stmt, A.DropIndex):
                db.drop_index(stmt.name, if_exists=stmt.if_exists)
                return []
        except CypherRuntimeError:
            raise
        except SqgraphError as exc:
            if isinstance(exc, CypherUnsupportedError):
                raise
            raise CypherRuntimeError(str(exc), query=text) from exc
        raise CypherUnsupportedError("unsupported statement", query=text)  # pragma: no cover


# ---------------------------------------------------------------- write execution


class Evaluator:
    """Evaluates expressions over injected rows via SQL (compiled once, runnable per batch)."""

    def __init__(self, ex: Executor, cols: list[tuple[str, T]], exprs: list[A.Expr], text: str, params: dict[str, Any]):
        self.ex = ex
        comp = Compiler(ex.db, params, text)
        comp.begin(cols, True)
        scope = comp.frame_scope(comp.frame)
        exs = [comp.expr(e, scope) for e in exprs]
        self.types = [e.type for e in exs]
        self.reads_graph = comp.reads_graph
        sel = ", ".join(e.sql for e in exs)
        cte = comp.frame.cte
        self.sql = comp.assemble(f"SELECT {cte}._i{', ' + sel if sel else ''} FROM {cte} ORDER BY {cte}._i")
        self.binds = comp.binds
        self.cols = cols

    def run(self, rows: list[list[Any]]) -> list[list[Any]]:
        if not rows:
            return []
        binds = dict(self.binds)
        binds["rows"] = rows_json(self.cols, rows)
        out = self.ex.db._conn.execute(self.sql, binds).fetchall()
        return [[plain_decode(v, t) for v, t in zip(r[1:], self.types)] for r in out]


def check_prop_value(v: Any, where: str = "property") -> Any:
    """Validate a value about to be stored as a property (primitives or lists thereof)."""
    if isinstance(v, (EntityVal, _EntityContainer)):
        raise CypherRuntimeError(f"Property values can only be of primitive types or arrays thereof (got a node/relationship for {where})")
    if isinstance(v, dict):
        raise CypherRuntimeError(f"Property values can only be of primitive types or arrays thereof (got a map for {where})")
    if isinstance(v, list):
        for x in v:
            if isinstance(x, (dict, list, EntityVal, _EntityContainer)):
                raise CypherRuntimeError(
                    f"Property values can only be of primitive types or arrays thereof (nested collection in {where})"
                )
    return v


class _Writer:
    """Executes one write clause over the current rows."""

    def __init__(self, ex: Executor, text: str, params: dict[str, Any]) -> None:
        self.ex = ex
        self.db = ex.db
        self.text = text
        self.params = params

    def err(self, msg: str, node: Any = None) -> CypherRuntimeError:
        pos = getattr(node, "pos", None)
        if pos is None:
            return CypherRuntimeError(msg, query=self.text)
        return CypherRuntimeError(
            msg,
            line=self.text.count("\n", 0, pos) + 1,
            column=pos - (self.text.rfind("\n", 0, pos) + 1) + 1,
            query=self.text,
        )

    def syn(self, msg: str, node: Any = None) -> CypherSyntaxError:
        pos = getattr(node, "pos", None)
        if pos is None:
            return CypherSyntaxError(msg, query=self.text)
        nl = chr(10)
        return CypherSyntaxError(
            msg,
            line=self.text.count(nl, 0, pos) + 1,
            column=pos - (self.text.rfind(nl, 0, pos) + 1) + 1,
            query=self.text,
        )

    def evaluator(self, cols: list[tuple[str, T]], exprs: list[A.Expr]) -> Evaluator:
        return Evaluator(self.ex, cols, exprs, self.text, self.params)

    def run(self, cl: A.Clause, cols: list[tuple[str, T]], rows: list[list[Any]]) -> tuple[list[tuple[str, T]], list[list[Any]]]:
        if isinstance(cl, A.Create):
            return self.create(cl, cols, rows)
        if isinstance(cl, A.Merge):
            return self.merge(cl, cols, rows)
        if isinstance(cl, A.SetClause):
            self.apply_set(cl.items, cols, rows)
            return cols, rows
        if isinstance(cl, A.Remove):
            self.remove(cl, cols, rows)
            return cols, rows
        if isinstance(cl, A.Delete):
            self.delete(cl, cols, rows)
            return cols, rows
        if isinstance(cl, A.Call):
            return self.call(cl, cols, rows)
        raise CypherUnsupportedError(f"clause {type(cl).__name__} is not supported")  # pragma: no cover

    # ---------------------------------------------------------- pattern creation

    def _prop_plan(self, patterns: list[A.PatternPart]) -> tuple[list[A.Expr], dict[int, Any]]:
        """Collect property expressions of all pattern elements; map element id -> plan."""
        exprs: list[A.Expr] = []
        plan: dict[int, Any] = {}
        for part in patterns:
            for el in part.elements:
                props = el.props
                if props is None:
                    continue
                if isinstance(props, A.Param):
                    val = self.params.get(props.name)
                    if not isinstance(val, dict):
                        raise self.err(f"Parameter ${props.name} must be a map", props)
                    plan[id(el)] = ("const", val)
                else:
                    keys = []
                    for k, e in props.items:
                        keys.append((k, len(exprs)))
                        exprs.append(e)
                    plan[id(el)] = ("exprs", keys)
        return exprs, plan

    def _props_for(self, el: Any, plan: dict[int, Any], vals: list[Any], merge: bool) -> dict[str, Any]:
        p = plan.get(id(el))
        out: dict[str, Any] = {}
        if p is None:
            return out
        if p[0] == "const":
            items = list(p[1].items())
        else:
            items = [(k, vals[i]) for k, i in p[1]]
        for k, v in items:
            if v is None:
                if merge:
                    raise self.err(f"Cannot merge using null property value for `{k}`", el)
                continue
            out[k] = check_prop_value(v, k)
        return out

    def _create_pattern(
        self,
        part: A.PatternPart,
        binding: dict[str, Any],
        plan: dict[int, Any],
        vals: list[Any],
        merge: bool,
    ) -> None:
        """Create the elements of one path; ``binding`` maps var name -> id and is extended."""
        db = self.db
        ids: list[int] = []
        elems = part.elements
        for el in elems:
            if isinstance(el, A.NodePattern):
                if el.var is not None and el.var in binding:
                    nid = binding[el.var]
                    if nid is None:
                        raise self.err(f"Failed to create a relationship: `{el.var}` is null", el)
                    if (el.labels or el.props is not None) and not merge:
                        raise self.err(
                            f"Can't create node `{el.var}` with labels or properties here. "
                            "The variable is already declared in this context",
                            el,
                        )
                else:
                    nid = db._w_create_node(el.labels, self._props_for(el, plan, vals, merge))
                    if el.var is not None:
                        binding[el.var] = nid
                ids.append(nid)
        # relationships
        ni = 0
        for j in range(1, len(elems), 2):
            rel = elems[j]
            assert isinstance(rel, A.RelPattern)
            if rel.varlen:
                raise self.err("Variable length relationships cannot be used in CREATE/MERGE", rel)
            if len(rel.types) != 1:
                raise self.err("Exactly one relationship type must be specified for CREATE/MERGE", rel)
            if rel.direction == "both" and not merge:
                raise self.err("Only directed relationships are supported in CREATE", rel)
            a, b = ids[ni], ids[ni + 1]
            ni += 1
            if rel.direction == "in":
                a, b = b, a
            eid = db._w_create_edge(a, b, rel.types[0], self._props_for(rel, plan, vals, merge))
            if rel.var is not None:
                binding[rel.var] = eid

    def create(self, cl: A.Create, cols: list[tuple[str, T]], rows: list[list[Any]]):
        for part in cl.patterns:
            if part.var is not None:
                raise CypherUnsupportedError("named paths in CREATE are not supported", query=self.text)
        exprs, plan = self._prop_plan(cl.patterns)
        vals = self.evaluator(cols, exprs).run(rows) if exprs else [[] for _ in rows]
        colidx = {n: i for i, (n, _) in enumerate(cols)}
        for n, t in cols:
            pass
        new_vars = [(n, t) for n, t in _pattern_vars(cl.patterns) if n not in colidx]
        seen: set[str] = set()
        new_vars = [(n, t) for n, t in new_vars if not (n in seen or seen.add(n))]
        out_rows = []
        for row, v in zip(rows, vals):
            binding = {n: row[i] for n, i in colidx.items()}
            self._check_bound_types(cl.patterns, cols)
            for part in cl.patterns:
                self._create_pattern(part, binding, plan, v, merge=False)
            out_rows.append(row + [binding[n] for n, _ in new_vars])
        return cols + new_vars, out_rows

    def _check_bound_types(self, patterns: list[A.PatternPart], cols: list[tuple[str, T]]) -> None:
        types = dict(cols)
        for part in patterns:
            for el in part.elements:
                if el.var in types:
                    want = "node" if isinstance(el, A.NodePattern) else "rel"
                    if types[el.var].kind != want:
                        raise self.err(f"Variable `{el.var}` already declared as {types[el.var].kind}", el)

    # ---------------------------------------------------------- MERGE

    def merge(self, cl: A.Merge, cols: list[tuple[str, T]], rows: list[list[Any]]):
        part = cl.pattern
        if part.var is not None:
            raise CypherUnsupportedError("named paths in MERGE are not supported", query=self.text)
        for el in part.elements:
            if isinstance(el, A.RelPattern) and el.varlen:
                raise self.err("Variable length relationships cannot be used in MERGE", el)
        self._check_bound_types([part], cols)
        exprs, plan = self._prop_plan([part])
        vals = self.evaluator(cols, exprs).run(rows) if exprs else [[] for _ in rows]
        comp = Compiler(self.db, self.params, self.text)
        probe = comp.compile_probe(cols, A.Match([part], None, optional=False))
        colidx = {n: i for i, (n, _) in enumerate(cols)}
        seen: set[str] = set()
        new_vars = [
            (n, t)
            for n, t in _pattern_vars([part])
            if n not in colidx and not (n in seen or seen.add(n))  # type: ignore[func-returns-value]
        ]
        out_cols = cols + new_vars
        set_create = self._prepare_set(cl.on_create, out_cols) if cl.on_create else None
        set_match = self._prepare_set(cl.on_match, out_cols) if cl.on_match else None
        out_rows: list[list[Any]] = []
        probe_names = [n for n, _ in probe.columns]
        for row, v in zip(rows, vals):
            binds = dict(probe.binds)
            binds["rows"] = rows_json(cols, [row])
            found = self.db._conn.execute(probe.sql, binds).fetchall()
            if found:
                for f in found:
                    got = dict(zip(probe_names, f))
                    new_row = row + [got.get(n) for n, _ in new_vars]
                    out_rows.append(new_row)
                    if set_match:
                        set_match(out_cols, [new_row])
            else:
                binding = {n: row[i] for n, i in colidx.items()}
                self._create_pattern(part, binding, plan, v, merge=True)
                new_row = row + [binding.get(n) for n, _ in new_vars]
                out_rows.append(new_row)
                if set_create:
                    set_create(out_cols, [new_row])
        return out_cols, out_rows

    # ---------------------------------------------------------- SET / REMOVE

    def _prepare_set(self, items: list[A.SetItem], cols: list[tuple[str, T]]):
        """Compile SET items once; returns a callable applying them to rows."""
        exprs = [it.expr for it in items if it.expr is not None]
        ev = self.evaluator(cols, exprs) if exprs else None

        def apply(cols_: list[tuple[str, T]], rows: list[list[Any]]) -> None:
            self._apply_set_items(items, cols_, rows, ev)

        return apply

    def apply_set(self, items: list[A.SetItem], cols: list[tuple[str, T]], rows: list[list[Any]]) -> None:
        self._prepare_set(items, cols)(cols, rows)

    def _apply_set_items(
        self, items: list[A.SetItem], cols: list[tuple[str, T]], rows: list[list[Any]], ev: Evaluator | None
    ) -> None:
        colidx = {n: i for i, (n, _) in enumerate(cols)}
        types = dict(cols)
        for it in items:
            if it.var not in colidx:
                raise CypherSyntaxError(f"Variable `{it.var}` not defined", query=self.text)
            if types[it.var].kind not in ("node", "rel"):
                raise self.syn(f"Type mismatch: expected Node or Relationship but was {types[it.var].kind}", it)
        batch = None
        if ev is not None and not ev.reads_graph:
            batch = ev.run(rows)
        for ri, row in enumerate(rows):
            vals = batch[ri] if batch is not None else (ev.run([row])[0] if ev is not None else [])
            vi = 0
            for it in items:
                val = None
                if it.expr is not None:
                    val = vals[vi]
                    vi += 1
                self._apply_one(it, row[colidx[it.var]], types[it.var].kind, val)

    def _apply_one(self, it: A.SetItem, ent: Any, kind: str, val: Any) -> None:
        if ent is None:
            return  # SET on a null (OPTIONAL MATCH) is a no-op
        db = self.db
        if it.kind == "labels":
            if kind != "node":
                raise self.err("Labels can only be set on nodes", it)
            db._w_add_labels(ent, it.labels)
            return
        if it.kind == "prop":
            val = check_prop_value(val, it.key or "")
            if kind == "node":
                db._w_set_node_props(ent, {it.key: val})
            else:
                db._w_set_edge_props(ent, {it.key: val})
            return
        # replace / merge: value must be a map (or a node/relationship whose properties are copied)
        if isinstance(val, EntityVal):
            src = (db._load_nodes([val.id]) if val.kind == "node" else db._load_edges([val.id])).get(val.id)
            if src is None:
                raise self.err("entity used as property map no longer exists", it)
            mapping = dict(src.properties)
        elif isinstance(val, dict):
            mapping = val
        elif val is None:
            if it.kind == "merge":
                return
            mapping = {}
        else:
            raise self.err("Type mismatch: expected a Map, Node or Relationship on the right of SET", it)
        clean = {k: check_prop_value(v, k) for k, v in mapping.items()}
        replace = it.kind == "replace"
        if kind == "node":
            db._w_set_node_props(ent, clean, replace=replace)
        else:
            db._w_set_edge_props(ent, clean, replace=replace)

    def remove(self, cl: A.Remove, cols: list[tuple[str, T]], rows: list[list[Any]]) -> None:
        colidx = {n: i for i, (n, _) in enumerate(cols)}
        types = dict(cols)
        for it in cl.items:
            if it.var not in colidx:
                raise CypherSyntaxError(f"Variable `{it.var}` not defined", query=self.text)
            if types[it.var].kind not in ("node", "rel"):
                raise self.syn(f"Type mismatch: expected Node or Relationship but was {types[it.var].kind}", it)
        db = self.db
        for row in rows:
            for it in cl.items:
                ent = row[colidx[it.var]]
                if ent is None:
                    continue
                kind = types[it.var].kind
                if it.kind == "labels":
                    if kind != "node":
                        raise self.err("Labels can only be removed from nodes", it)
                    db._w_remove_labels(ent, it.labels)
                elif kind == "node":
                    db._w_set_node_props(ent, {it.key: None})
                else:
                    db._w_set_edge_props(ent, {it.key: None})

    # ---------------------------------------------------------- DELETE

    def delete(self, cl: A.Delete, cols: list[tuple[str, T]], rows: list[list[Any]]) -> None:
        ev = self.evaluator(cols, cl.exprs)
        vals = ev.run(rows)
        node_ids: dict[int, None] = {}
        rel_ids: dict[int, None] = {}

        def collect(v: Any, t: T, node: Any) -> None:
            if v is None:
                return
            if isinstance(v, EntityVal):
                (node_ids if v.kind == "node" else rel_ids)[v.id] = None
            elif isinstance(v, _EntityContainer):
                obj, tt = v.obj, v.type
                if tt.kind == "path":
                    for i in obj["n"]:
                        node_ids[i] = None
                    for i in obj["r"]:
                        rel_ids[i] = None
                elif tt.kind == "list" and tt.elem is not None and tt.elem.kind in ("node", "rel"):
                    for i in obj:
                        if i is not None:
                            (node_ids if tt.elem.kind == "node" else rel_ids)[i] = None
                else:
                    raise self.err("DELETE expects nodes, relationships or paths", node)
            else:
                raise self.err("Type mismatch: DELETE expects a node, relationship or path", node)

        for row_vals in vals:
            for v, t, e in zip(row_vals, ev.types, cl.exprs):
                collect(v, t, e)
        db = self.db
        for rid in rel_ids:
            db._w_delete_edge(rid)
        for nid in node_ids:
            try:
                db._w_delete_node(nid, cl.detach)
            except SqgraphError as exc:
                raise self.err(
                    f"Cannot delete node<{nid}>, because it still has relationships. "
                    "To delete this node, you must first delete its relationships, or use DETACH DELETE"
                ) from exc

    # ---------------------------------------------------------- CALL (write procedures)

    def create_index_proc(self, cl: A.Call, name: str, cols: list[tuple[str, T]], rows: list[list[Any]]):
        """``db.index.vector.createNodeIndex`` / ``db.index.fulltext.createNodeIndex`` (Neo4j 4.x/5.x procedures)."""
        if cl.yields:
            raise self.err(f"{cl.name} does not return any results to YIELD", cl)
        ev = self.evaluator(cols, cl.args)
        vals = ev.run(rows[:1]) if rows else []
        for v in vals:
            try:
                if name == "db.index.vector.createnodeindex":
                    if len(v) not in (4, 5):
                        raise self.err(f"{cl.name}(name, label, property, dimensions[, similarity]) takes 4 or 5 arguments", cl)
                    idx, label, prop, dims = v[:4]
                    sim = v[4] if len(v) == 5 and v[4] is not None else "cosine"
                    self.db.create_vector_index(idx, label, prop, dims, sim)
                else:
                    if len(v) < 3:
                        raise self.err(f"{cl.name}(name, labels, properties) takes 3 arguments", cl)
                    idx, labels, props = v[:3]
                    self.db.create_fulltext_index(idx, labels, props)
            except CypherRuntimeError:
                raise
            except SqgraphError as exc:
                raise self.err(str(exc), cl) from exc
        return cols, rows

    def call(self, cl: A.Call, cols: list[tuple[str, T]], rows: list[list[Any]]):
        name = cl.name.lower()
        if name in ("db.index.vector.createnodeindex", "db.index.fulltext.createnodeindex"):
            return self.create_index_proc(cl, name, cols, rows)
        rel = name == "db.create.setrelationshipvectorproperty"
        if len(cl.args) != 3:
            raise self.err(
                f"{cl.name}({'relationship' if rel else 'node'}, propertyName, vector) takes 3 arguments", cl
            )
        ev = self.evaluator(cols, cl.args)
        for row_vals in ev.run(rows):
            node, prop, vec = row_vals
            if node is None:
                continue
            if not isinstance(node, EntityVal) or node.kind != ("rel" if rel else "node"):
                raise self.err(f"{cl.name}: first argument must be a {'relationship' if rel else 'node'}", cl)
            if not isinstance(prop, str):
                raise self.err(f"{cl.name}: property name must be a string", cl)
            if isinstance(vec, _EntityContainer) or not isinstance(vec, list):
                raise self.err(f"{cl.name}: the vector must be a list of numbers", cl)
            if rel:
                self.db._w_set_edge_vector(node.id, prop, vec)
            else:
                self.db._w_set_node_vector(node.id, prop, vec)
        if name in ("db.create.setnodevectorproperty", "db.create.setrelationshipvectorproperty"):
            if cl.yields:
                raise self.err(f"{cl.name} does not return any results to YIELD", cl)
            return cols, rows
        if cl.yields is None:
            return cols, rows
        new_cols = list(cols)
        out_rows = [list(r) for r in rows]
        idx = {n: i for i, (n, _) in enumerate(cols)}
        # setVectorProperty yields the node it was called with
        for y in cl.yields:
            if y.name != "node":
                raise self.err(f"{cl.name} has no output `{y.name}`", y)
            var = y.alias or y.name
            new_cols.append((var, NODE))
            node_expr_vals = ev.run(rows)
            for r, nv in zip(out_rows, node_expr_vals):
                r.append(nv[0].id if isinstance(nv[0], EntityVal) else None)
        return new_cols, out_rows
