"""Compile Cypher read clauses to SQL.

Each clause becomes a CTE over the binding table of the previous clause
(``Frame``).  Bound node/relationship variables are integer id columns;
properties are read with ``json_extract`` / ``->`` on the ``props`` JSON;
variable-length patterns are recursive CTEs; ``UNWIND`` uses ``json_each`` and
``collect`` uses ``json_group_array``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..errors import CypherRuntimeError, DimensionMismatchError
from ..ftsquery import fallback_fts5
from . import ast as A
from .exprs import (
    ExprCompiler,
    elem_type,
    free_vars,
    implicit_grouping_violation,
    json_text,
    native,
    split_conjuncts,
)
from .sqltypes import (
    ANY,
    FLOAT,
    INT,
    JSON_KINDS,
    NODE,
    PATH,
    RAWPATH,
    REL,
    SCALAR,
    STR,
    Ex,
    Frame,
    T,
    Var,
    contains_entity,
    list_of,
)

HIDDEN = "$"
ORD = HIDDEN + "ord"  # implicit ordinal of a procedure's rows (its natural, best-first order)

_NATIVE_SCALARS = ("int", "float", "str", "scalar")


def _unify_union(a: T, b: T) -> T | None:
    """Common type of one UNION column across two branches (None: not representable)."""
    if a == b:
        return a
    if a.kind == "null":
        return b
    if b.kind == "null":
        return a
    if contains_entity(a) or contains_entity(b):
        return None
    if a.kind in _NATIVE_SCALARS and b.kind in _NATIVE_SCALARS:
        return SCALAR  # native SQLite values either way: no re-encoding needed
    return ANY


@dataclass
class Segment:
    """A compiled read segment: SQL, its bind parameters and result columns."""

    sql: str
    binds: dict[str, Any]
    columns: list[tuple[str, T]]
    final: bool = False  # ends with RETURN (result rows are user-visible)
    reads_graph: bool = False


@dataclass
class Slot:
    """A node position in a pattern while it is being compiled."""

    name: str
    ref: str | None
    cte_bound: bool
    np: A.NodePattern
    pending: bool = False  # label/property constraints not applied yet (slot created lazily)


@dataclass
class Stage:
    """The SELECT being assembled for a MATCH clause."""

    from_items: list[str]
    scope: dict[str, Var]
    conds: list[str] = field(default_factory=list)
    rels: list[tuple[str, str]] = field(default_factory=list)  # (hidden var name, 'fixed'|'vl')


def _visible(names: Any) -> list[str]:
    return [n for n in names if not n.startswith(HIDDEN)]


def _carried(names: Any) -> list[str]:
    return [n for n in names if not n.startswith(HIDDEN) or n == ORD]


class Compiler(ExprCompiler):
    """Compiles clauses of one query (or one UNION query) into a single SQL statement."""

    def __init__(self, db: Any, params: dict[str, Any], text: str = "") -> None:
        self.db = db
        self.params = params
        self.text = text
        self.max_len: int = int(db.max_var_length)
        self.binds: dict[str, Any] = {}
        self.ctes: list[str] = []
        self._n = 0
        self.reads_graph = False
        self.saw_agg = False
        self.frame: Frame = Frame("c0", {})

    # ------------------------------------------------------------ plumbing

    def bind(self, value: Any) -> str:
        name = f"p{self._n}"
        self._n += 1
        if isinstance(value, bool):
            value = int(value)
        self.binds[name] = value
        return f":{name}"

    def uid(self) -> int:
        self._n += 1
        return self._n

    def add_cte(self, name: str, sql: str, columns: str = "", materialized: bool = False) -> None:
        mat = "MATERIALIZED " if materialized else ""
        self.ctes.append(f"{name}{columns} AS {mat}({sql})")

    def assemble(self, final_select: str) -> str:
        if not self.ctes:
            return final_select
        return f"WITH RECURSIVE {', '.join(self.ctes)} {final_select}"

    @staticmethod
    def frame_scope(frame: Frame) -> dict[str, Var]:
        return {n: Var(f"{frame.cte}.{c}", t, None, frame.cte) for n, (c, t) in frame.cols.items()}

    # ------------------------------------------------------------ segments

    def begin(self, base_cols: list[tuple[str, T]] | None, injected: bool) -> None:
        """Create the base CTE: one unit row, or the rows injected as JSON in ``:rows``."""
        name = f"c{self.uid()}"
        cols: dict[str, tuple[str, T]] = {}
        if not injected:
            self.add_cte(name, "SELECT 0 AS _i")
        else:
            sel = ["j.key AS _i"]
            for i, (vname, t) in enumerate(base_cols or []):
                col = f"v{self.uid()}"
                path = f"'$[{i}]'"
                if t.kind == "any":
                    expr = f"CASE json_type(j.value, {path}) WHEN 'null' THEN NULL ELSE (j.value -> {path}) END"
                else:
                    expr = f"json_extract(j.value, {path})"
                sel.append(f"{expr} AS {col}")
                cols[vname] = (col, t)
            self.add_cte(name, f"SELECT {', '.join(sel)} FROM json_each(:rows) j")
        self.frame = Frame(name, cols)

    def compile_segment(
        self,
        clauses: list[A.Clause],
        base_cols: list[tuple[str, T]] | None = None,
        injected: bool = False,
        whole_query: bool = False,
    ) -> Segment:
        self.begin(base_cols, injected)
        final_sql: str | None = None
        columns: list[tuple[str, T]] = []
        for cl in clauses:
            if isinstance(cl, A.Match):
                self.c_match(cl)
            elif isinstance(cl, A.Unwind):
                self.c_unwind(cl)
            elif isinstance(cl, A.With):
                self.c_with(cl)
            elif isinstance(cl, A.Call):
                self.c_call(cl, len(clauses) == 1)
            elif isinstance(cl, A.Return):
                final_sql, columns = self.c_return(cl)
            else:  # pragma: no cover - the executor splits write clauses off
                raise self.unsupported(f"clause {type(cl).__name__} cannot be compiled to SQL", cl)
        final = final_sql is not None
        if final_sql is None:
            final_sql, columns = self.c_carry()
            # a standalone CALL returns its yielded columns as the result
            final = whole_query and len(clauses) == 1 and isinstance(clauses[0], A.Call)
        return Segment(self.assemble(final_sql), dict(self.binds), columns, final, self.reads_graph)

    def compile_probe(self, base_cols: list[tuple[str, T]], match: A.Match) -> Segment:
        """Compile a MATCH over injected rows returning only the newly bound variables."""
        self.begin(base_cols, True)
        old = set(self.frame.cols)
        self.c_match(match)
        new = [n for n in self.frame.cols if n not in old]
        sql, columns = self.c_carry(new)
        return Segment(self.assemble(sql), dict(self.binds), columns, False, self.reads_graph)

    def compile_union(self, queries: list[list[A.Clause]], union_all: bool) -> Segment:
        branches: list[tuple[str, list[tuple[str, T]]]] = []
        columns: list[tuple[str, T]] = []
        for i, clauses in enumerate(queries):
            self.begin(None, False)
            final_sql: str | None = None
            cols: list[tuple[str, T]] = []
            for cl in clauses:
                if isinstance(cl, A.Match):
                    self.c_match(cl)
                elif isinstance(cl, A.Unwind):
                    self.c_unwind(cl)
                elif isinstance(cl, A.With):
                    self.c_with(cl)
                elif isinstance(cl, A.Call):
                    self.c_call(cl, False)
                elif isinstance(cl, A.Return):
                    final_sql, cols = self.c_return(cl)
                else:
                    raise self.unsupported("write clauses inside UNION queries are not supported", cl)
            if final_sql is None:
                raise self.err("each query of a UNION must end with RETURN", clauses[-1])
            if i == 0:
                columns = cols
            elif [n for n, _ in cols] != [n for n, _ in columns]:
                raise self.err("All sub queries in an UNION must have the same return column names", clauses[-1])
            branches.append((final_sql, cols))
        # one column type for all branches; a branch whose representation differs is re-encoded
        types = [t for _, t in branches[0][1]]
        for _, cols in branches[1:]:
            for j, (name, t) in enumerate(cols):
                u = _unify_union(types[j], t)
                if u is None:
                    raise self.unsupported(
                        f"UNION column `{name}` mixes nodes/relationships/paths with other types", queries[-1][-1]
                    )
                types[j] = u
        columns = [(n, t) for (n, _), t in zip(columns, types)]
        selects: list[str] = []
        for final_sql, cols in branches:
            conv = [
                f"{json_text(Ex(f'a{j}', t))} AS a{j}" if types[j].kind == "any" and t.kind not in JSON_KINDS
                else f"a{j}"
                for j, (_, t) in enumerate(cols)
            ]
            if all(c == f"a{j}" for j, c in enumerate(conv)):
                selects.append(f"SELECT * FROM ({final_sql})")
            else:
                selects.append(f"SELECT {', '.join(conv)} FROM ({final_sql})")
        op = " UNION ALL " if union_all else " UNION "
        return Segment(self.assemble(op.join(selects)), dict(self.binds), columns, True, self.reads_graph)

    def c_carry(self, only: list[str] | None = None) -> tuple[str, list[tuple[str, T]]]:
        """Final SELECT handing the visible variables to the executor."""
        f = self.frame
        names = _visible(f.cols) if only is None else only
        if not names:
            return f"SELECT 1 AS _u FROM {f.cte}", []
        cols = ", ".join(f"{f.cte}.{f.cols[n][0]}" for n in names)
        return f"SELECT {cols} FROM {f.cte}", [(n, f.cols[n][1]) for n in names]

    # ------------------------------------------------------------ MATCH

    def c_match(self, cl: A.Match) -> None:
        frame = self.frame
        base = frame.cte
        scope = self.frame_scope(frame)
        if cl.optional:
            a = f"c{self.uid()}"
            self.add_cte(a, f"SELECT {frame.cte}.*, ROW_NUMBER() OVER () AS _r FROM {frame.cte}", materialized=True)
            base = a
            scope = self.frame_scope(Frame(a, frame.cols))
            scope[HIDDEN + "r"] = Var(f"{a}._r", INT, cte=a)
        st = Stage([base], scope)
        pending = split_conjuncts(cl.where)
        for part in cl.patterns:
            self.pattern_part(st, part, pending)
        self.attach(st, pending, final=True)
        where = " AND ".join(st.conds) or "1"
        from_sql = ", ".join(st.from_items)
        new_cte = f"c{self.uid()}"
        if not cl.optional:
            names = _carried(st.scope)
            items, cols = [], {}
            for n in names:
                col = f"v{self.uid()}"
                items.append(f"{st.scope[n].sql} AS {col}")
                cols[n] = (col, st.scope[n].type)
            if not items:
                items = ["1 AS _u"]
            self.add_cte(new_cte, f"SELECT {', '.join(items)} FROM {from_sql} WHERE {where}")
            self.frame = Frame(new_cte, cols)
            return
        new_names = [n for n in _visible(st.scope) if n not in frame.cols]
        m_items = [f"{st.scope[HIDDEN + 'r'].sql} AS _r"]
        new_cols: dict[str, tuple[str, T]] = {}
        for n in new_names:
            col = f"v{self.uid()}"
            m_items.append(f"{st.scope[n].sql} AS {col}")
            new_cols[n] = (col, st.scope[n].type)
        m = f"c{self.uid()}"
        self.add_cte(m, f"SELECT {', '.join(m_items)} FROM {from_sql} WHERE {where}")
        out_items = [f"{base}.{c} AS {c}" for c, _ in frame.cols.values()]
        out_items += [f"{m}.{c} AS {c}" for c, _ in new_cols.values()]
        if not out_items:
            out_items = ["1 AS _u"]
        self.add_cte(new_cte, f"SELECT {', '.join(out_items)} FROM {base} LEFT JOIN {m} ON {m}._r = {base}._r")
        self.frame = Frame(new_cte, {**frame.cols, **new_cols})

    def attach(self, st: Stage, pending: list[A.Expr], final: bool) -> None:
        """Move WHERE conjuncts whose variables are all bound into the current stage."""
        rest: list[A.Expr] = []
        for c in pending:
            fv = free_vars(c)
            has_exists = "$exists" in fv
            fv.discard("$exists")
            if (has_exists and not final) or not fv <= set(st.scope):
                rest.append(c)
                continue
            st.conds.append(self.cond(c, st.scope))
        pending[:] = rest
        if final and pending:
            self.cond(pending[0], st.scope)  # raises "Variable not defined"
            raise self.err("could not resolve WHERE clause", pending[0])  # pragma: no cover

    def flush(self, st: Stage) -> None:
        """Materialize the stage as a CTE so later hops can seed from its columns."""
        cte = f"c{self.uid()}"
        items, new_scope = [], {}
        for n, v in st.scope.items():
            col = f"v{self.uid()}"
            items.append(f"{v.sql} AS {col}")
            new_scope[n] = Var(f"{cte}.{col}", v.type, None, cte)
        where = " AND ".join(st.conds) or "1"
        self.add_cte(cte, f"SELECT {', '.join(items) or '1 AS _u'} FROM {', '.join(st.from_items)} WHERE {where}")
        st.from_items = [cte]
        st.conds = []
        st.scope = new_scope

    def refresh(self, st: Stage, *slots: Slot) -> None:
        for s in slots:
            if s.name in st.scope and s.ref is not None:
                v = st.scope[s.name]
                s.ref = v.sql
                s.cte_bound = v.cte is not None

    # -- pattern parts

    def pattern_part(self, st: Stage, part: A.PatternPart, pending: list[A.Expr]) -> None:
        elems = part.elements
        reverse_first = False
        if (
            len(elems) >= 3
            and isinstance(elems[1], A.RelPattern)
            and elems[1].varlen
            and elems[1].var is None
            and part.var is None
            and (elems[0].var is None or elems[0].var not in st.scope)
            and self.selectivity(elems[2], st, pending) > self.selectivity(elems[0], st, pending)
        ):
            reverse_first = True  # seed the recursion from the more selective (right) end
        left = self.node_slot(st, elems[0], lazy=reverse_first)
        segs: list[tuple[str, ...]] = [("n", left.name)]
        if len(elems) == 1:
            self.materialize(st, left)
        for i in range(1, len(elems), 2):
            rel = elems[i]
            assert isinstance(rel, A.RelPattern)
            right_np = elems[i + 1]
            assert isinstance(right_np, A.NodePattern)
            if rel.varlen:
                left, seg = self.varlen_hop(
                    st, part, left, rel, right_np, pending, force_reverse=reverse_first and i == 1
                )
            else:
                left, seg = self.fixed_hop(st, left, rel, right_np)
            segs.extend(seg)
            self.attach(st, pending, final=False)
        if part.var:
            if part.var in st.scope:
                raise self.err(f"Variable `{part.var}` already declared", part)
            nodes, rels = "''", "''"
            for s in segs:
                if s[0] == "n":
                    nodes += f" || ',' || {st.scope[s[1]].sql}"
                elif s[0] == "r":
                    rels += f" || ',' || {st.scope[s[1]].sql}"
                else:
                    rels += f" || {st.scope[s[1]].sql}"
                    nodes += f" || {st.scope[s[2]].sql}"
            path = f"('{{\"n\":[' || substr({nodes}, 2) || '],\"r\":[' || substr({rels}, 2) || ']}}')"
            st.scope[part.var] = Var(path, PATH)

    # -- node handling

    def prop_items(self, props: A.MapLiteral | A.Param | None, scope: dict[str, Var]) -> list[tuple[str, Ex]]:
        if props is None:
            return []
        if isinstance(props, A.Param):
            if props.name not in self.params:
                raise self.rt(f"Missing parameter: ${props.name}", props)
            val = self.params[props.name]
            if not isinstance(val, dict):
                raise self.rt(f"Parameter ${props.name} must be a map", props)
            return [(str(k), self.const_ex(v, props)) for k, v in val.items()]
        return [(k, self.expr(v, scope)) for k, v in props.items]

    def prop_eq(self, kind: str, ref: str, alias: str | None, key: str, ex: Ex, node: Any) -> str:
        """``entity.key = value`` with Neo4j type semantics, keeping the plain equality index-friendly."""
        pe = self.prop_of(Ex(ref, NODE if kind == "node" else REL), self.jpath(key), alias, node, key)
        return self._compare("=", pe, ex, node).sql

    def prop_conds(self, st: Stage, kind: str, ref: str, alias: str | None, props: Any) -> None:
        items = self.prop_items(props, st.scope)
        if not items:
            return
        for key, ex in items:
            st.conds.append(self.prop_eq(kind, ref, alias, key, ex, props))

    def label_conds(self, st: Stage, ref: str, labels: list[str]) -> None:
        for lb in labels:
            st.conds.append(
                f"EXISTS (SELECT 1 FROM node_labels WHERE node_id = {ref} AND label = {self.bind(lb)})"
            )

    def node_alias(self, st: Stage, np: A.NodePattern) -> tuple[str, str]:
        """Add ``nodes`` (plus a label join for the first label) to FROM; returns (id ref, props ref)."""
        k = self.uid()
        n = f"n{k}"
        if np.labels:
            lt = f"l{k}"
            st.from_items.append(f"node_labels {lt}")
            st.from_items.append(f"nodes {n}")
            st.conds.append(f"{lt}.label = {self.bind(np.labels[0])}")
            st.conds.append(f"{n}.id = {lt}.node_id")
            self.label_conds(st, f"{n}.id", np.labels[1:])
        else:
            st.from_items.append(f"nodes {n}")
        self.prop_conds(st, "node", f"{n}.id", f"{n}.props", np.props)
        return f"{n}.id", f"{n}.props"

    def selectivity(self, np: A.NodePattern, st: Stage, pending: list[A.Expr]) -> int:
        """Rough score of how few nodes ``np`` matches: bound variable > id()/property equality > label."""
        if np.var is not None and np.var in st.scope and st.scope[np.var].cte is not None:
            return 100
        score = 10 * bool(np.labels)
        if np.props is not None:
            score += 40
        if np.var is not None:
            for c in pending:
                if isinstance(c, A.BinaryOp) and c.op in ("=", "IN"):
                    for side, other in ((c.left, c.right), (c.right, c.left)):
                        if isinstance(side, A.FunctionCall) and side.name.lower() == "id" and len(side.args) == 1:
                            arg = side.args[0]
                            if isinstance(arg, A.Variable) and arg.name == np.var and np.var not in free_vars(other):
                                score += 60
                        elif (
                            isinstance(side, A.Property)
                            and isinstance(side.base, A.Variable)
                            and side.base.name == np.var
                            and np.var not in free_vars(other)
                            and c.op == "="
                        ):
                            score += 40
        return score

    def apply_pending(self, st: Stage, slot: Slot, props_alias: str | None = None) -> None:
        if slot.pending:
            slot.pending = False
            assert slot.ref is not None
            self.label_conds(st, slot.ref, slot.np.labels)
            self.prop_conds(st, "node", slot.ref, props_alias, slot.np.props)

    def node_slot(self, st: Stage, np: A.NodePattern, lazy: bool = False) -> Slot:
        name = np.var
        if name is not None and name in st.scope:
            v = st.scope[name]
            if v.type.kind != "node":
                raise self.err(f"Variable `{name}` already declared as {v.type.kind}", np)
            self.label_conds(st, v.sql, np.labels)
            self.prop_conds(st, "node", v.sql, v.props, np.props)
            return Slot(name, v.sql, v.cte is not None, np)
        name = name or f"{HIDDEN}n{self.uid()}"
        if lazy:
            return Slot(name, None, False, np, pending=bool(np.labels or np.props))
        if np.labels or np.props:
            ref, props = self.node_alias(st, np)
            st.scope[name] = Var(ref, NODE, props=props)
            return Slot(name, ref, False, np)
        return Slot(name, None, False, np)

    def resolve(self, st: Stage, slot: Slot, sql: str) -> None:
        slot.ref = sql
        st.scope[slot.name] = Var(sql, NODE)
        self.apply_pending(st, slot)

    def materialize(self, st: Stage, slot: Slot) -> None:
        if slot.ref is None:
            k = self.uid()
            st.from_items.append(f"nodes n{k}")
            slot.ref = f"n{k}.id"
            st.scope[slot.name] = Var(slot.ref, NODE, props=f"n{k}.props")
            self.apply_pending(st, slot, f"n{k}.props")

    def bind_end(self, st: Stage, slot: Slot, sql: str) -> None:
        if slot.ref is None:
            self.resolve(st, slot, sql)
        else:
            st.conds.append(f"({slot.ref} = {sql})")

    # -- relationship uniqueness

    def register_rel(self, st: Stage, name: str, kind: str) -> None:
        sql = st.scope[name].sql
        for pn, pk in st.rels:
            ps = st.scope[pn].sql
            if pk == "fixed" and kind == "fixed":
                st.conds.append(f"({ps} <> {sql})")
            elif pk == "fixed" and kind == "vl":
                st.conds.append(f"(instr({sql} || ',', ',' || {ps} || ',') = 0)")
            elif pk == "vl" and kind == "fixed":
                st.conds.append(f"(instr({ps} || ',', ',' || {sql} || ',') = 0)")
            else:
                st.conds.append(
                    f"(NOT EXISTS (SELECT 1 FROM json_each('[' || substr({ps}, 2) || ']') a, "
                    f"json_each('[' || substr({sql}, 2) || ']') b WHERE a.value = b.value))"
                )
        st.rels.append((name, kind))

    def rel_filters(self, scope: dict[str, Var], alias: str, rel: A.RelPattern) -> list[str]:
        conds: list[str] = []
        if rel.types:
            if len(rel.types) == 1:
                conds.append(f"{alias}.type = {self.bind(rel.types[0])}")
            else:
                conds.append(f"{alias}.type IN ({', '.join(self.bind(t) for t in rel.types)})")
        for key, ex in self.prop_items(rel.props, scope):
            conds.append(self.prop_eq("rel", f"{alias}.id", f"{alias}.props", key, ex, rel))
        if rel.props is not None:
            self.reads_graph = True
        return conds

    # -- fixed-length hop

    def fixed_hop(
        self, st: Stage, left: Slot, rel: A.RelPattern, right_np: A.NodePattern
    ) -> tuple[Slot, list[tuple[str, ...]]]:
        k = self.uid()
        e = f"e{k}"
        st.from_items.append(f"edges {e}")
        if rel.var is not None and rel.var in st.scope:
            v = st.scope[rel.var]
            if v.type.kind != "rel":
                raise self.err(f"Variable `{rel.var}` already declared as {v.type.kind}", rel)
            st.conds.append(f"({e}.id = {v.sql})")
            relname = rel.var
        else:
            relname = rel.var or f"{HIDDEN}rel{k}"
            st.scope[relname] = Var(f"{e}.id", REL, props=f"{e}.props")
        st.conds.extend(self.rel_filters(st.scope, e, rel))
        self.register_rel(st, relname, "fixed")
        if rel.direction == "both":
            self.materialize(st, left)
            lr = left.ref
            right = self.node_slot(st, right_np)
            if right.ref is None:
                st.conds.append(f"({e}.src = {lr} OR {e}.dst = {lr})")
                self.resolve(st, right, f"(CASE WHEN {e}.src = {lr} THEN {e}.dst ELSE {e}.src END)")
            else:
                rr = right.ref
                st.conds.append(
                    f"(({e}.src = {lr} AND {e}.dst = {rr}) OR ({e}.dst = {lr} AND {e}.src = {rr}))"
                )
        else:
            lcol, rcol = (f"{e}.src", f"{e}.dst") if rel.direction == "out" else (f"{e}.dst", f"{e}.src")
            self.bind_end(st, left, lcol)
            right = self.node_slot(st, right_np)
            self.bind_end(st, right, rcol)
        return right, [("r", relname), ("n", right.name)]

    # -- variable-length hop

    def varlen_hop(
        self,
        st: Stage,
        part: A.PatternPart,
        left: Slot,
        rel: A.RelPattern,
        right_np: A.NodePattern,
        pending: list[A.Expr],
        force_reverse: bool = False,
    ) -> tuple[Slot, list[tuple[str, ...]]]:
        k = self.uid()
        vl = f"vl{k}"
        need_path = part.var is not None
        direction = rel.direction
        rv = right_np.var
        reverse = force_reverse or (
            rel.var is None
            and not need_path
            and rv is not None
            and rv in st.scope
            and st.scope[rv].cte is not None
            and not left.cte_bound
        )
        if reverse:
            origin = self.node_slot(st, right_np)
            far = left
            direction = {"out": "in", "in": "out", "both": "both"}[direction]
        else:
            origin = left
            far = None  # created after the recursion is set up
        if origin.ref is None:
            self.materialize(st, origin)
        if st.scope[origin.name].cte is None:
            self.attach(st, pending, final=False)
            self.flush(st)
            self.refresh(st, origin, left)
        ov = st.scope[origin.name]
        assert ov.cte is not None
        seeds = f"SELECT DISTINCT {ov.sql} AS id FROM {ov.cte}"
        max_h = rel.max_hops if rel.max_hops is not None else max(self.max_len, rel.min_hops)
        # INDEXED BY pins the adjacency index so that no statistics (or stale ones) can make the
        # planner walk the whole edge table in the recursive step
        if direction == "out":
            adj, nxt, hint = "e.src = r.x", "e.dst", " INDEXED BY idx_edges_src"
        elif direction == "in":
            adj, nxt, hint = "e.dst = r.x", "e.src", " INDEXED BY idx_edges_dst"
        else:
            adj, nxt, hint = "(e.src = r.x OR e.dst = r.x)", "CASE WHEN e.src = r.x THEN e.dst ELSE e.src END", ""
        extra = "".join(f" AND {c}" for c in self.rel_filters({}, "e", rel))
        np_cols = ", np" if need_path else ""
        np_seed = ", ''" if need_path else ""
        np_step = f", r.np || ',' || {nxt}" if need_path else ""
        self.add_cte(
            vl,
            f"SELECT s.id, s.id, 0, ''{np_seed} FROM ({seeds}) s "
            f"UNION ALL SELECT r.o, {nxt}, r.d + 1, r.rp || ',' || e.id{np_step} "
            f"FROM {vl} r JOIN edges e{hint} ON {adj} "
            f"WHERE r.d < {int(max_h)} AND instr(r.rp || ',', ',' || e.id || ',') = 0{extra}",
            columns=f"(o, x, d, rp{np_cols})",
        )
        st.from_items.append(vl)
        st.conds.append(f"({vl}.o = {ov.sql})")
        st.conds.append(f"({vl}.d >= {int(rel.min_hops)})")
        if rel.var is not None:
            if rel.var in st.scope:
                raise self.unsupported("re-using a bound variable-length relationship variable", rel)
            st.scope[rel.var] = Var(f"('[' || substr({vl}.rp, 2) || ']')", list_of(REL))
        rpname = f"{HIDDEN}vl{k}"
        st.scope[rpname] = Var(f"{vl}.rp", RAWPATH)
        self.register_rel(st, rpname, "vl")
        npname = f"{HIDDEN}vln{k}"
        if need_path:
            st.scope[npname] = Var(f"{vl}.np", RAWPATH)
        if reverse:
            self.refresh(st, far)  # type: ignore[arg-type]
            self.bind_end(st, far, f"{vl}.x")  # type: ignore[arg-type]
            result = origin
        else:
            far = self.node_slot(st, right_np)
            self.bind_end(st, far, f"{vl}.x")
            result = far
        return result, [("vl", rpname, npname)]

    # ------------------------------------------------------------ EXISTS { ... }

    def exists_subquery(self, n: A.ExistsSubquery, scope: dict[str, Var]) -> str:
        st = Stage([], dict(scope))
        pending = split_conjuncts(n.where)
        self.reads_graph = True
        for part in n.patterns:
            if part.var is not None:
                raise self.unsupported("named paths inside EXISTS are not supported", part)
            for e in part.elements:
                if isinstance(e, A.RelPattern) and e.varlen:
                    raise self.unsupported("variable-length patterns inside EXISTS are not supported", e)
            self.pattern_part(st, part, pending)
        self.attach(st, pending, final=True)
        from_sql = ", ".join(st.from_items) or "(SELECT 1)"
        return f"EXISTS (SELECT 1 FROM {from_sql} WHERE {' AND '.join(st.conds) or '1'})"

    # ------------------------------------------------------------ UNWIND

    def c_unwind(self, cl: A.Unwind) -> None:
        frame = self.frame
        if cl.var in frame.cols:
            raise self.err(f"Variable `{cl.var}` already declared", cl)
        src = self.expr(cl.expr, self.frame_scope(frame))
        if src.type.kind not in ("list", "any", "null"):
            raise self.err(f"Type mismatch: expected List but was {src.type.kind}", cl)
        et = elem_type(src.type)
        alias = f"je{self.uid()}"
        col = f"v{self.uid()}"
        cte = f"c{self.uid()}"
        self.add_cte(
            cte,
            f"SELECT {frame.cte}.*, {self.je_elem(alias, et)} AS {col} "
            f"FROM {frame.cte}, json_each({src.sql}) {alias}",
        )
        self.frame = Frame(cte, {**frame.cols, cl.var: (col, et)})

    # ------------------------------------------------------------ WITH / RETURN

    def c_with(self, cl: A.With) -> None:
        proj = cl.projection
        select, names = self.project(proj, final=False)
        cte = f"c{self.uid()}"
        cols = {n: (f"a{i}", t) for i, (n, t) in enumerate(names)}
        self.add_cte(cte, select)
        self.frame = Frame(cte, cols)
        if proj.where is not None:
            scope = self.frame_scope(self.frame)
            self.saw_agg = False
            cond = self.cond(proj.where, scope)
            f2 = f"c{self.uid()}"
            self.add_cte(f2, f"SELECT * FROM {cte} WHERE {cond}")
            self.frame = Frame(f2, cols)

    def c_return(self, cl: A.Return) -> tuple[str, list[tuple[str, T]]]:
        select, names = self.project(cl.projection, final=True)
        return select, names

    def _limit_sql(self, proj: A.Projection) -> str:
        def const_int(e: A.Expr, what: str) -> str:
            ex = self.expr(e, {})
            if ex.type.kind not in ("int", "any", "scalar"):
                raise self.err(f"{what} must be an integer", e)
            if isinstance(e, A.Literal) and (not isinstance(e.value, int) or e.value < 0):
                raise self.err(f"{what} must be a non-negative integer", e)
            if isinstance(e, A.Param):
                v = self.params.get(e.name)
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise self.rt(f"{what} must be a non-negative integer", e)
            return f"CAST({native(ex)} AS INTEGER)"

        sql = ""
        if proj.limit is not None:
            sql += f" LIMIT {const_int(proj.limit, 'LIMIT')}"
        elif proj.skip is not None:
            sql += " LIMIT -1"
        if proj.skip is not None:
            sql += f" OFFSET {const_int(proj.skip, 'SKIP')}"
        return sql

    def project(self, proj: A.Projection, final: bool) -> tuple[str, list[tuple[str, T]]]:
        """Build the SELECT for a WITH/RETURN projection; returns (sql, [(name, type)]).

        Result columns are named ``a0, a1, ...`` in the SELECT.
        """
        frame = self.frame
        scope = self.frame_scope(frame)
        items: list[tuple[str, Ex, bool]] = []
        asts: list[A.ProjectionItem | None] = []  # None for items coming from `*`
        if proj.star:
            names = _visible(frame.cols)
            if not names and not proj.items:
                raise self.err("RETURN * is not allowed when there are no variables in scope", proj)
            for n in names:
                v = scope[n]
                items.append((n, Ex(v.sql, v.type), False))
                asts.append(None)
        for it in proj.items:
            self.saw_agg = False
            ex = self.expr(it.expr, scope)
            agg = self.saw_agg
            self.saw_agg = False
            if it.alias:
                name = it.alias
            elif final:
                name = it.text
            elif isinstance(it.expr, A.Variable):
                name = it.expr.name
            else:
                raise self.err("Expression in WITH must be aliased (use AS)", it)
            items.append((name, ex, agg))
            asts.append(it)
        seen: set[str] = set()
        for n, _, _ in items:
            if n in seen:
                raise self.err(f"Multiple result columns with the same name `{n}` are not supported", proj)
            seen.add(n)
        has_agg = any(a for _, _, a in items)
        if has_agg:
            key_vars: set[str] = set()
            key_exprs: list[Any] = []
            for (n, _, agg), it in zip(items, asts):
                if not agg:
                    if it is None:
                        key_vars.add(n)
                    else:
                        key_exprs.append(it.expr)
                        if isinstance(it.expr, A.Variable):
                            key_vars.add(it.expr.name)
            for (n, _, agg), it in zip(items, asts):
                if agg and it is not None:
                    bad = implicit_grouping_violation(it.expr, key_exprs, key_vars)
                    if bad:
                        raise self.err(
                            f"Aggregation column contains implicit grouping expressions: `{bad}` must also be "
                            "returned as a separate (grouping) column",
                            it,
                        )
        select_items = [f"{ex.sql} AS a{i}" for i, (_, ex, _) in enumerate(items)]
        group = ""
        if has_agg:
            keys = [str(i + 1) for i, (_, _, a) in enumerate(items) if not a]
            if keys:
                group = f" GROUP BY {', '.join(keys)}"
        order = ""
        ord_var = scope.get(ORD)
        if ord_var is not None and not proj.order_by and not has_agg and not proj.distinct:
            # no explicit ORDER BY: keep the procedure's natural order (so LIMIT keeps the best rows)
            if final or proj.skip is not None or proj.limit is not None:
                order = f" ORDER BY {ord_var.sql} ASC"
            if not final:
                select_items.append(f"{ord_var.sql} AS a{len(items)}")
                items.append((ORD, Ex(ord_var.sql, INT), False))
        if proj.order_by:
            oscope = dict(scope)
            for n, ex, _ in items:
                oscope[n] = Var(ex.sql, ex.type)
            parts = []
            for si in proj.order_by:
                self.saw_agg = False
                ex = self.expr(si.expr, oscope)
                if self.saw_agg and not has_agg:
                    raise self.err("Aggregation in ORDER BY requires an aggregating projection", si)
                self.saw_agg = False
                if ex.type.kind in ("path",):
                    raise self.unsupported("ORDER BY on a path is not supported", si)
                parts.append(f"{native(ex)} {'DESC NULLS FIRST' if si.descending else 'ASC NULLS LAST'}")
            order = f" ORDER BY {', '.join(parts)}"
        distinct = "DISTINCT " if proj.distinct else ""
        sql = (
            f"SELECT {distinct}{', '.join(select_items) or '1 AS _u'} FROM {frame.cte}"
            f"{group}{order}{self._limit_sql(proj)}"
        )
        return sql, [(n, ex.type) for n, ex, _ in items]

    # ------------------------------------------------------------ CALL (read procedures)

    def const_value(self, e: A.Expr, what: str) -> Any:
        if isinstance(e, A.Literal):
            return e.value
        if isinstance(e, A.Param):
            if e.name not in self.params:
                raise self.rt(f"Missing parameter: ${e.name}", e)
            return self.params[e.name]
        if isinstance(e, A.MapLiteral):
            return {k: self.const_value(v, what) for k, v in e.items}
        if isinstance(e, A.ListLiteral):
            return [self.const_value(v, what) for v in e.items]
        raise self.unsupported(f"{what} must be a literal or a parameter", e)

    def c_call(self, cl: A.Call, standalone: bool) -> None:
        name = cl.name.lower()
        frame = self.frame
        scope = self.frame_scope(frame)
        k = self.uid()
        from_items: list[str] = []
        conds: list[str] = []
        order = ""
        ord_sql: str | None = None
        if name in ("db.index.vector.querynodes", "db.index.vector.queryrelationships"):
            outputs = self.p_vector(cl, scope, k, from_items, conds, rel=name.endswith("relationships"))
            order = f" ORDER BY v{k}.distance"
            ord_sql = f"ROW_NUMBER() OVER (ORDER BY v{k}.distance)"
        elif name in ("db.index.fulltext.querynodes", "db.index.fulltext.queryrelationships"):
            outputs = self.p_fulltext(cl, scope, k, from_items, conds, rel=name.endswith("relationships"))
            order = f" ORDER BY f{k}.ord"
            ord_sql = f"f{k}.ord"
        elif name == "db.labels":
            self.no_args(cl)
            from_items.append(f"(SELECT DISTINCT label FROM node_labels) p{k}")
            outputs = [("label", f"p{k}.label", STR)]
        elif name == "db.relationshiptypes":
            self.no_args(cl)
            from_items.append(f"(SELECT DISTINCT type FROM edges) p{k}")
            outputs = [("relationshipType", f"p{k}.type", STR)]
        elif name == "db.propertykeys":
            self.no_args(cl)
            from_items.append(
                f"(SELECT DISTINCT key FROM (SELECT j.key FROM nodes n, json_each(n.props) j UNION ALL "
                f"SELECT j.key FROM edges e, json_each(e.props) j)) p{k}"
            )
            outputs = [("propertyKey", f"p{k}.key", STR)]
        elif name in (
            "db.create.setnodevectorproperty",
            "db.create.setvectorproperty",
            "db.create.setrelationshipvectorproperty",
            "db.index.vector.createnodeindex",
            "db.index.fulltext.createnodeindex",
        ):
            raise self.err(f"{cl.name} must be executed by the write executor", cl)  # pragma: no cover
        else:
            raise self.unsupported(f"Procedure {cl.name} is not supported", cl)
        if cl.yields is None:
            if not standalone:
                raise self.err(
                    "Procedure call inside a query does not support naming results implicitly (use YIELD)", cl
                )
            chosen = [(o[0], o) for o in outputs]
        else:
            by_name = {o[0]: o for o in outputs}
            chosen = []
            for y in cl.yields:
                if y.name not in by_name:
                    raise self.err(
                        f"Procedure {cl.name} has no output `{y.name}` (available: {', '.join(by_name)})", y
                    )
                chosen.append((y.alias or y.name, by_name[y.name]))
        new_cols: dict[str, tuple[str, T]] = {}
        items = []
        for var, (_, sql, t) in chosen:
            if var in frame.cols or var in new_cols:
                raise self.err(f"Variable `{var}` already declared", cl)
            col = f"v{self.uid()}"
            items.append(f"{sql} AS {col}")
            new_cols[var] = (col, t)
        if ord_sql is not None and not _visible(frame.cols) and ORD not in frame.cols:
            col = f"v{self.uid()}"  # single input row: remember the procedure's row order
            items.append(f"{ord_sql} AS {col}")
            new_cols[ORD] = (col, INT)
        cte = f"c{self.uid()}"
        where = " AND ".join(conds) or "1"
        base_items = [frame.cte] + from_items
        self.add_cte(
            cte,
            f"SELECT {frame.cte}.*, {', '.join(items)} FROM {', '.join(base_items)} WHERE {where}{order}",
        )
        self.frame = Frame(cte, {**frame.cols, **new_cols})
        if cl.where is not None:
            scope2 = self.frame_scope(self.frame)
            cond = self.cond(cl.where, scope2)
            f2 = f"c{self.uid()}"
            self.add_cte(f2, f"SELECT * FROM {cte} WHERE {cond}")
            self.frame = Frame(f2, self.frame.cols)

    def no_args(self, cl: A.Call) -> None:
        if cl.args:
            raise self.err(f"{cl.name}() takes no arguments", cl)

    def index_kind(self, ix: Any, idx: str, what: str, rel: bool, cl: A.Call) -> None:
        """Reject a node index used by a ``queryRelationships`` procedure and vice versa."""
        if (ix.entity_type == "relationship") != rel:
            have, want = ("relationship", "queryNodes") if not rel else ("node", "queryRelationships")
            raise self.rt(f"Index `{idx}` is a {have} {what} index; use {cl.name.rsplit('.', 1)[0]}.{want}", cl)

    def p_vector(
        self, cl: A.Call, scope: dict[str, Var], k: int, from_items: list[str], conds: list[str], rel: bool = False
    ) -> list[tuple[str, str, T]]:
        if len(cl.args) != 3:
            raise self.err(
                f"db.index.vector.{'queryRelationships' if rel else 'queryNodes'}(indexName, k, vector) takes 3 arguments",
                cl,
            )
        idx = self.const_value(cl.args[0], "the index name")
        kk = self.const_value(cl.args[1], "k")
        if not isinstance(idx, str):
            raise self.rt("index name must be a string", cl)
        if not isinstance(kk, int) or isinstance(kk, bool) or kk < 1:
            raise self.rt("k must be a positive integer", cl)
        ix = self.db._vec.get(idx)
        if ix is None:
            raise self.rt(f"There is no such vector schema index: {idx}", cl)
        self.index_kind(ix, idx, "vector", rel, cl)
        vec = self.expr(cl.args[2], scope)
        if vec.type.kind not in ("list", "any"):
            raise self.err(f"Type mismatch: the query vector must be a List, got {vec.type.kind}", cl.args[2])
        try:
            const = self.const_value(cl.args[2], "the query vector")
        except Exception:
            const = None
        if isinstance(const, list) and len(const) != ix.dimensions:
            raise DimensionMismatchError(
                f"query vector has {len(const)} dimensions but index '{idx}' expects {ix.dimensions}"
            )
        from_items.append(f"{ix.table} v{k}")
        conds.append(f"v{k}.embedding MATCH vec_f32({vec.sql})")
        conds.append(f"v{k}.k = {self.bind(kk)}")
        if ix.similarity == "cosine":
            score = f"(1.0 - v{k}.distance / 2.0)"
        else:
            score = f"(1.0 / (1.0 + v{k}.distance * v{k}.distance))"
        if rel:
            return [("relationship", f"v{k}.{ix.key_column}", REL), ("score", score, FLOAT)]
        return [("node", f"v{k}.{ix.key_column}", NODE), ("score", score, FLOAT)]

    def p_fulltext(
        self, cl: A.Call, scope: dict[str, Var], k: int, from_items: list[str], conds: list[str], rel: bool = False
    ) -> list[tuple[str, str, T]]:
        if len(cl.args) not in (2, 3):
            raise self.err(
                f"db.index.fulltext.{'queryRelationships' if rel else 'queryNodes'}"
                "(indexName, queryString[, options]) takes 2 or 3 arguments",
                cl,
            )
        idx = self.const_value(cl.args[0], "the index name")
        query = self.const_value(cl.args[1], "the query string")
        opts = self.const_value(cl.args[2], "the options") if len(cl.args) == 3 else {}
        if not isinstance(idx, str) or not isinstance(query, str):
            raise self.rt("index name and query must be strings", cl)
        ix = self.db._fts.get(idx)
        if ix is None:
            raise self.rt(f"There is no such fulltext schema index: {idx}", cl)
        self.index_kind(ix, idx, "fulltext", rel, cl)
        limit = ""
        if isinstance(opts, dict) and opts.get("limit") is not None:
            lim = opts["limit"]
            if not isinstance(lim, int) or isinstance(lim, bool) or lim < 0:
                raise self.rt("options.limit must be a non-negative integer", cl)
            limit = f" LIMIT {lim}"
        match = self.db.fts_match_expression(ix, query)
        if match:
            try:
                self.db._conn.execute(f"SELECT 1 FROM {ix.table} WHERE {ix.table} MATCH ? LIMIT 0", (match,))
            except sqlite3.OperationalError:
                match = fallback_fts5(query)
        if not match:
            conds.append("0")
            match = '""'
        # bm25() only works while the FTS table is scanned directly, so the subquery must never be
        # flattened into the caller: materialize it (and its ranking) as CTEs
        self.add_cte(
            f"ftr{k}",
            f"SELECT rowid AS {ix.key_column}, -bm25({ix.table}) AS score FROM {ix.table} "
            f"WHERE {ix.table} MATCH {self.bind(match)} ORDER BY score DESC{limit}",
            materialized=True,
        )
        self.add_cte(
            f"f{k}",
            f"SELECT {ix.key_column}, score, ROW_NUMBER() OVER (ORDER BY score DESC) AS ord FROM ftr{k}",
            materialized=True,
        )
        from_items.append(f"f{k}")
        if rel:
            return [("relationship", f"f{k}.{ix.key_column}", REL), ("score", f"f{k}.score", FLOAT)]
        return [("node", f"f{k}.{ix.key_column}", NODE), ("score", f"f{k}.score", FLOAT)]
