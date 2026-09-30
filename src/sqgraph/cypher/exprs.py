"""Expression compilation: Cypher expression AST -> SQL fragments (:class:`Ex`)."""

from __future__ import annotations

import json
import re
from typing import Any

from ..db import normalize_value
from ..errors import CypherRuntimeError, CypherSyntaxError, CypherUnsupportedError
from ..types import Node, Relationship
from . import ast as A
from .sqltypes import (
    ANY,
    BOOL,
    FLOAT,
    INT,
    LIST_ANY,
    MAP,
    NODE,
    NULL,
    PATH,
    REL,
    SCALAR,
    STR,
    Ex,
    T,
    Var,
    list_of,
    map_of,
)

_SAFE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WS = "char(32, 9, 10, 13)"
AGGREGATES = {"count", "sum", "avg", "min", "max", "collect"}
_NUMERIC = ("int", "float")


# ---------------------------------------------------------------- representation helpers


def native(ex: Ex) -> str:
    """SQL yielding the native SQLite scalar of ``ex`` (unwraps ``any`` JSON)."""
    if ex.type.kind == "any":
        return ex.nat if ex.nat is not None else f"json_extract({ex.sql}, '$')"
    return ex.sql


def enc(ex: Ex) -> str:
    """SQL usable as an element inside json_array / json_object / json_group_array."""
    k = ex.type.kind
    if k == "bool":
        return f"json(CASE WHEN {ex.sql} IS NULL THEN NULL WHEN {ex.sql} THEN 'true' ELSE 'false' END)"
    if k in ("list", "map", "path", "any"):
        return f"json({ex.sql})"
    return ex.sql


def json_text(ex: Ex) -> str:
    """SQL yielding the JSON text of the value (NULL for null); node/rel become their ids."""
    k = ex.type.kind
    if k in ("list", "map", "path", "any"):
        return ex.sql
    if k == "bool":
        return f"CASE WHEN {ex.sql} IS NULL THEN NULL WHEN {ex.sql} THEN 'true' ELSE 'false' END"
    if k in ("node", "rel", "int", "float"):
        return f"CAST({ex.sql} AS TEXT)"
    if k in ("str", "scalar"):
        return f"CASE WHEN {ex.sql} IS NULL THEN NULL ELSE json_quote({ex.sql}) END"
    return "NULL"


def extract(json_sql: str, path_sql: str, t: T) -> str:
    """SQL extracting the element at ``path_sql`` from JSON text, in the rep of ``t``."""
    if t.kind == "any":
        return f"nullif(({json_sql} -> {path_sql}), 'null')"
    return f"json_extract({json_sql}, {path_sql})"


def elem_type(t: T) -> T:
    if t.kind == "list" and t.elem is not None:
        return t.elem
    return ANY


def is_num(t: T) -> bool:
    return t.kind in _NUMERIC


_STATIC_CLS = {
    "int": "n", "float": "n", "str": "s", "bool": "b", "list": "l", "map": "m", "node": "e", "rel": "e2", "path": "p",
}
_GUARD_TYPES = {"n": ("integer", "real"), "s": ("text",), "b": ("true", "false"), "l": ("array",), "m": ("object",)}
_CLS_CODE = {"n": 1, "s": 2, "b": 3, "l": 4, "m": 5, "e": 1, "e2": 1}


def static_cls(t: T) -> str | None:
    """Comparison class of a statically typed value (None: dynamic ``any`` / ``scalar``, or null)."""
    return _STATIC_CLS.get(t.kind)


def jtype_sql(ex: Ex) -> str:
    """SQL yielding the JSON/storage type name of a dynamically typed value."""
    if ex.type.kind == "scalar":
        return f"typeof({ex.sql})"
    return ex.jt or f"json_type({ex.sql})"


def cls_code_sql(ex: Ex) -> str:
    """SQL yielding a comparison-class code (1 number, 2 string, 3 boolean, 4 list, 5 map; NULL for null)."""
    c = static_cls(ex.type)
    if c is not None:
        return str(_CLS_CODE[c])
    return (
        f"(CASE {jtype_sql(ex)} WHEN 'integer' THEN 1 WHEN 'real' THEN 1 WHEN 'text' THEN 2 "
        "WHEN 'true' THEN 3 WHEN 'false' THEN 3 WHEN 'array' THEN 4 WHEN 'object' THEN 5 END)"
    )


def guard_sql(dyn: Ex, cls: str) -> str:
    """SQL true iff the dynamic value ``dyn`` belongs to comparison class ``cls``."""
    types = _GUARD_TYPES.get(cls)
    if types is None or (dyn.type.kind == "scalar" and cls in ("b", "l", "m")):
        return "(0)"
    return f"({jtype_sql(dyn)} IN ({', '.join(repr(t) for t in types)}))"


def _json_kind_of_py(values: list[Any]) -> T:
    kinds = set()
    for v in values:
        if v is None:
            return ANY  # a null element makes `x IN list` three-valued: no static element type
        if isinstance(v, bool):
            kinds.add("bool")
        elif isinstance(v, int):
            kinds.add("int")
        elif isinstance(v, float):
            kinds.add("float")
        elif isinstance(v, str):
            kinds.add("str")
        else:
            kinds.add("other")
    if len(kinds) == 1:
        k = next(iter(kinds))
        return {"bool": BOOL, "int": INT, "float": FLOAT, "str": STR}.get(k, ANY)
    if kinds == {"int", "float"}:
        return FLOAT
    return ANY


class ExprCompiler:
    """Mixin providing ``expr()``; concrete state is supplied by :class:`Compiler`."""

    # supplied by Compiler
    params: dict[str, Any]
    text: str
    binds: dict[str, Any]
    reads_graph: bool
    saw_agg: bool

    def bind(self, value: Any) -> str: ...
    def uid(self) -> int: ...

    # ---------------------------------------------------------- errors

    def _loc(self, node: Any) -> dict[str, Any]:
        pos = getattr(node, "pos", None)
        if pos is None or not self.text:
            return {"query": self.text}
        line = self.text.count("\n", 0, pos) + 1
        col = pos - (self.text.rfind("\n", 0, pos) + 1) + 1
        return {"line": line, "column": col, "query": self.text}

    def err(self, msg: str, node: Any = None) -> CypherSyntaxError:
        return CypherSyntaxError(msg, **self._loc(node))

    def unsupported(self, msg: str, node: Any = None) -> CypherUnsupportedError:
        return CypherUnsupportedError(msg, **self._loc(node))

    def rt(self, msg: str, node: Any = None) -> CypherRuntimeError:
        return CypherRuntimeError(msg, **self._loc(node))

    # ---------------------------------------------------------- constants & coercions

    def const_ex(self, v: Any, node: Any = None) -> Ex:
        """Wrap a Python value as a compiled constant (bound as SQL parameter)."""
        if v is None:
            return Ex("NULL", NULL)
        if isinstance(v, bool):
            return Ex(self.bind(int(v)), BOOL)
        if isinstance(v, int):
            if not -(2**63) <= v < 2**63:
                raise self.rt("integer literal out of range", node)
            return Ex(self.bind(v), INT)
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                return Ex("NULL", NULL)
            return Ex(self.bind(v), FLOAT)
        if isinstance(v, str):
            return Ex(self.bind(v), STR)
        if isinstance(v, Node):
            return Ex(self.bind(v.id), NODE)
        if isinstance(v, Relationship):
            return Ex(self.bind(v.id), REL)
        try:
            nv = normalize_value(v)
            text = json.dumps(nv, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            raise self.rt(f"unsupported parameter value of type {type(v).__name__}: {exc}", node) from None
        if isinstance(nv, list):
            return Ex(self.bind(text), list_of(_json_kind_of_py(nv)))
        return Ex(self.bind(text), MAP)

    def to_any(self, ex: Ex, node: Any = None) -> str:
        """SQL converting ``ex`` to the ``any`` representation (JSON text)."""
        k = ex.type.kind
        if k == "any":
            return ex.sql
        if k in ("node", "rel", "path") and k != "path":
            raise self.unsupported("nodes and relationships cannot be mixed with other types here", node)
        if k == "null":
            return "NULL"
        return json_text(ex)

    def unify(self, exs: list[Ex], node: Any = None) -> tuple[T, list[str]]:
        """Common type of several expressions and their SQL coerced to it."""
        kinds = {e.type.kind for e in exs if e.type.kind != "null"}
        if not kinds:
            return NULL, [e.sql for e in exs]
        types = {e.type for e in exs if e.type.kind != "null"}
        if len(types) == 1:
            return next(iter(types)), [e.sql for e in exs]
        if kinds == {"list"}:
            elems = [e.type.elem or ANY for e in exs if e.type.kind == "list"]
            if any(el.kind in ("node", "rel", "path") for el in elems):
                if len(set(elems)) == 1:
                    return list_of(elems[0]), [e.sql for e in exs]
                raise self.unsupported("cannot mix lists of different entity types", node)
            return LIST_ANY, [e.sql for e in exs]
        if kinds <= set(_NUMERIC):
            return FLOAT, [e.sql for e in exs]
        if kinds <= {"int", "float", "str", "scalar"}:
            return SCALAR, [native(e) for e in exs]
        if kinds & {"node", "rel", "path"}:
            raise self.err("Type mismatch: cannot combine nodes/relationships/paths with other types", node)
        return ANY, [self.to_any(e, node) for e in exs]

    # ---------------------------------------------------------- dispatch

    def expr(self, node: A.Expr, scope: dict[str, Var]) -> Ex:
        m = getattr(self, "e_" + type(node).__name__, None)
        if m is None:
            raise self.unsupported(f"expression {type(node).__name__} is not supported", node)
        return m(node, scope)

    def cond(self, node: A.Expr, scope: dict[str, Var]) -> str:
        """Compile a predicate to a SQL boolean expression."""
        prev, self.saw_agg = self.saw_agg, False
        ex = self.expr(node, scope)
        if self.saw_agg:
            raise self.err("Invalid use of an aggregating function in a predicate (WHERE)", node)
        self.saw_agg = prev
        if ex.type.kind in ("list", "map", "node", "rel", "path"):
            raise self.err(f"Type mismatch: expected Boolean but was {ex.type.kind}", node)
        return f"({native(ex)})"

    # ---------------------------------------------------------- atoms

    def e_Literal(self, n: A.Literal, scope: dict[str, Var]) -> Ex:
        return self.const_ex(n.value, n)

    def e_Param(self, n: A.Param, scope: dict[str, Var]) -> Ex:
        if n.name not in self.params:
            raise self.rt(f"Missing parameter: ${n.name}", n)
        return self.const_ex(self.params[n.name], n)

    def e_Variable(self, n: A.Variable, scope: dict[str, Var]) -> Ex:
        v = scope.get(n.name)
        if v is None:
            raise self.err(f"Variable `{n.name}` not defined", n)
        return Ex(v.sql, v.type)

    # ---------------------------------------------------------- property access

    def jpath(self, key: str) -> str:
        """SQL literal/parameter for the JSON path of property ``key``."""
        if _SAFE_KEY.fullmatch(key):
            return f"'$.{key}'"
        return self.bind('$."' + key.replace("\\", "\\\\").replace('"', '\\"') + '"')

    @staticmethod
    def dyn_key_path(key_sql: str) -> str:
        return "('$.\"' || replace(replace(" + key_sql + ", '\\', '\\\\'), '\"', '\\\"') || '\"')"

    @staticmethod
    def props_of(kind: str, ref: str, alias: str | None) -> str:
        if alias:
            return alias
        table = "nodes" if kind == "node" else "edges"
        return f"(SELECT props FROM {table} WHERE id = {ref})"

    def prop_of(self, b: Ex, path_sql: str, alias: str | None, node: Any, key: str | None = None) -> Ex:
        kind = b.type.kind
        if kind in ("node", "rel"):
            self.reads_graph = True
            props = self.props_of(kind, b.sql, alias)
            return Ex(
                f"({props} -> {path_sql})", ANY, nat=f"json_extract({props}, {path_sql})",
                jt=f"json_type({props}, {path_sql})",
            )
        if kind == "map":
            if key is not None and b.type.fields is not None:
                for fk, ft in b.type.fields:
                    if fk == key:
                        e = extract(b.sql, path_sql, ft)
                        return Ex(e, ft)
                return Ex("NULL", ANY)
            return Ex(
                f"nullif(({b.sql} -> {path_sql}), 'null')", ANY, nat=f"json_extract({b.sql}, {path_sql})"
            )
        if kind == "any":
            return Ex(
                f"nullif(({b.sql} -> {path_sql}), 'null')", ANY, nat=f"json_extract({b.sql}, {path_sql})"
            )
        if kind == "null":
            return Ex("NULL", ANY)
        raise self.err(f"Type mismatch: expected Node, Relationship or Map but was {kind}", node)

    def e_Property(self, n: A.Property, scope: dict[str, Var]) -> Ex:
        alias = None
        if isinstance(n.base, A.Variable):
            v = scope.get(n.base.name)
            if v is None:
                raise self.err(f"Variable `{n.base.name}` not defined", n.base)
            b = Ex(v.sql, v.type)
            alias = v.props
        else:
            b = self.expr(n.base, scope)
        return self.prop_of(b, self.jpath(n.key), alias, n, n.key)

    def e_Index(self, n: A.Index, scope: dict[str, Var]) -> Ex:
        b = self.expr(n.base, scope)
        i = self.expr(n.index, scope)
        kind = b.type.kind
        if kind == "null":
            return Ex("NULL", ANY)
        if kind in ("node", "rel"):
            if i.type.kind not in ("str", "any", "scalar", "null"):
                raise self.err("Type mismatch: property key must be a string", n)
            return self.prop_of(b, self.dyn_key_path(native(i)), None, n)
        if kind == "map" or (kind == "any" and i.type.kind == "str"):
            if i.type.kind not in ("str", "any", "scalar", "null"):
                raise self.err("Type mismatch: map key must be a string", n)
            return self.prop_of(b, self.dyn_key_path(native(i)), None, n)
        if kind in ("list", "any"):
            et = elem_type(b.type)
            if isinstance(n.index, A.Literal) and isinstance(n.index.value, int) and not isinstance(n.index.value, bool):
                k = n.index.value
                path = f"'$[{k}]'" if k >= 0 else f"'$[#{k}]'"
            else:
                idx = native(i)
                path = f"CASE WHEN {idx} < 0 THEN '$[#' || {idx} || ']' ELSE '$[' || {idx} || ']' END"
            return Ex(extract(b.sql, path, et), et)
        raise self.err(f"Type mismatch: cannot index a value of type {kind}", n)

    def e_Slice(self, n: A.Slice, scope: dict[str, Var]) -> Ex:
        b = self.expr(n.base, scope)
        if b.type.kind not in ("list", "any", "null"):
            raise self.err(f"Type mismatch: cannot slice a value of type {b.type.kind}", n)
        lo = native(self.expr(n.start, scope)) if n.start is not None else "NULL"
        hi = native(self.expr(n.end, scope)) if n.end is not None else "NULL"
        t = b.type if b.type.kind == "list" else LIST_ANY
        return Ex(f"sqg_jslice({b.sql}, {lo}, {hi})", t)

    def e_MapProjection(self, n: A.MapProjection, scope: dict[str, Var]) -> Ex:
        alias = None
        if isinstance(n.base, A.Variable):
            v = scope.get(n.base.name)
            if v is None:
                raise self.err(f"Variable `{n.base.name}` not defined", n.base)
            b = Ex(v.sql, v.type)
            alias = v.props
        else:
            b = self.expr(n.base, scope)
        kind = b.type.kind
        if kind == "null":
            return Ex("NULL", NULL)
        if kind not in ("node", "rel", "map", "any"):
            raise self.err(f"Type mismatch: expected Node, Relationship or Map but was {kind}", n)
        props = self._props_json(b, n) if alias is None else alias
        acc = "'{}'"
        for what, arg in n.items:
            if what == "all":
                acc = f"sqg_mapproj({acc}, {props})"
                continue
            if what == "prop":
                key = arg
                ex = self.prop_of(b, self.jpath(key), alias, n, key)
            elif what == "var":
                key = arg
                ex = self.e_Variable(A.Variable(key, pos=n.pos), scope)
            else:
                key, e = arg
                ex = self.expr(e, scope)
            if ex.type.kind in ("node", "rel", "path"):
                raise self.unsupported("nodes/relationships inside a map projection are not supported", n)
            acc = f"sqg_mapproj({acc}, json_object({self.bind(key)}, {enc(ex)}))"
        return Ex(f"(CASE WHEN {b.sql} IS NULL THEN NULL ELSE {acc} END)", MAP)

    # ---------------------------------------------------------- operators

    def e_UnaryOp(self, n: A.UnaryOp, scope: dict[str, Var]) -> Ex:
        x = self.expr(n.operand, scope)
        if n.op == "NOT":
            return Ex(f"(NOT {native(x)})", BOOL)
        t = x.type if is_num(x.type) else SCALAR
        return Ex(f"(-{native(x)})", t)

    def e_IsNull(self, n: A.IsNull, scope: dict[str, Var]) -> Ex:
        x = self.expr(n.operand, scope)
        return Ex(f"({x.sql} IS {'NOT ' if n.negated else ''}NULL)", BOOL)

    def e_LabelCheck(self, n: A.LabelCheck, scope: dict[str, Var]) -> Ex:
        x = self.expr(n.operand, scope)
        if x.type.kind == "null":
            return Ex("NULL", BOOL)
        if x.type.kind != "node":
            raise self.err(f"Type mismatch: expected Node but was {x.type.kind}", n)
        self.reads_graph = True
        parts = [
            f"EXISTS (SELECT 1 FROM node_labels WHERE node_id = {x.sql} AND label = {self.bind(lb)})"
            for lb in n.labels
        ]
        return Ex(f"(CASE WHEN {x.sql} IS NULL THEN NULL ELSE ({' AND '.join(parts)}) END)", BOOL)

    def e_BinaryOp(self, n: A.BinaryOp, scope: dict[str, Var]) -> Ex:
        op = n.op
        if op in ("AND", "OR"):
            a, b = self.expr(n.left, scope), self.expr(n.right, scope)
            return Ex(f"({native(a)} {op} {native(b)})", BOOL)
        if op == "XOR":
            a, b = self.expr(n.left, scope), self.expr(n.right, scope)
            return Ex(f"(({native(a)}) <> ({native(b)}))", BOOL)
        if op == "IN":
            return self._in(n, scope)
        a, b = self.expr(n.left, scope), self.expr(n.right, scope)
        if op in ("=", "<>", "<", ">", "<=", ">="):
            for side in (a, b):
                if side.type.kind == "path":
                    raise self.unsupported("comparing paths is not supported", n)
            return self._compare(op, a, b, n)
        if op == "STARTS":
            x, y = native(a), native(b)
            return Ex(f"(substr({x}, 1, length({y})) = {y})", BOOL)
        if op == "ENDS":
            x, y = native(a), native(b)
            return Ex(f"(substr({x}, length({x}) - length({y}) + 1) = {y})", BOOL)
        if op == "CONTAINS":
            return Ex(f"(instr({native(a)}, {native(b)}) > 0)", BOOL)
        if op == "=~":
            return Ex(f"sqg_matches({native(a)}, {native(b)})", BOOL)
        return self._arith(op, a, b, n)

    def _compare(self, op: str, a: Ex, b: Ex, node: Any = None) -> Ex:
        """Comparison with Neo4j semantics: equality across types is false, ordering across types is null."""
        if "null" in (a.type.kind, b.type.kind):
            return Ex("NULL", BOOL)
        ca, cb = static_cls(a.type), static_cls(b.type)
        na, nb = native(a), native(b)
        plain = f"({na} {op} {nb})"
        if ca and cb:
            if ca == cb:
                return Ex(plain, BOOL)
            mism = {"=": "0", "<>": "1"}.get(op, "NULL")
            return Ex(f"(CASE WHEN {a.sql} IS NULL OR {b.sql} IS NULL THEN NULL ELSE {mism} END)", BOOL)
        if ca is None and cb is None:
            ja, jb = cls_code_sql(a), cls_code_sql(b)
            eq = f"({na} = {nb} AND {ja} = {jb})"
            if op == "=":
                return Ex(eq, BOOL)
            if op == "<>":
                return Ex(f"(NOT {eq})", BOOL)
            return Ex(f"(CASE WHEN {ja} = {jb} THEN {plain} END)", BOOL)
        dyn, cls = (a, cb) if ca is None else (b, ca)
        g = guard_sql(dyn, cls)  # type: ignore[arg-type]
        if op == "=":
            return Ex(f"({na} = {nb} AND {g})", BOOL)
        if op == "<>":
            return Ex(f"(NOT ({na} = {nb} AND {g}))", BOOL)
        return Ex(f"(CASE WHEN {g} THEN {plain} END)", BOOL)

    def _in(self, n: A.BinaryOp, scope: dict[str, Var]) -> Ex:
        x = self.expr(n.left, scope)
        cx = static_cls(x.type)
        nx = native(x)
        null_or_false = f"(CASE WHEN {x.sql} IS NULL THEN NULL ELSE 0 END)"
        if (
            isinstance(n.right, A.ListLiteral)
            and len(n.right.items) <= 200
            and all(isinstance(i, A.Literal) for i in n.right.items)
        ):
            vals = [i.value for i in n.right.items]  # type: ignore[union-attr]
            if not vals:
                return Ex("(0)", BOOL)
            classes = {
                ("b" if isinstance(v, bool) else "n" if isinstance(v, (int, float)) else "s" if isinstance(v, str) else "?")
                for v in vals
                if v is not None
            }
            if None not in vals and len(classes) == 1 and "?" not in classes:
                (c,) = classes
                binds = ", ".join(self.const_ex(v).sql for v in vals)
                if cx == c or (cx in ("e", "e2") and c == "n"):
                    return Ex(f"({nx} IN ({binds}))", BOOL)
                if cx is None:
                    return Ex(f"({nx} IN ({binds}) AND {guard_sql(x, c)})", BOOL)
                return Ex(null_or_false, BOOL)
        lst = self.expr(n.right, scope)
        if lst.type.kind == "null":
            return Ex("NULL", BOOL)
        if lst.type.kind not in ("list", "any"):
            raise self.err(f"Type mismatch: expected List but was {lst.type.kind}", n)
        ecls = static_cls(lst.type.elem) if lst.type.kind == "list" and lst.type.elem is not None else None
        if ecls in ("n", "s", "b", "e", "e2"):  # statically typed, null-free element type
            members = f"SELECT value FROM json_each({lst.sql})"
            if cx == ecls:
                return Ex(f"({nx} IN ({members}))", BOOL)
            if cx is None and ecls in ("n", "s", "b"):
                return Ex(f"({nx} IN ({members}) AND {guard_sql(x, ecls)})", BOOL)
            if cx is not None:
                return Ex(null_or_false, BOOL)
        je_code = (
            "(CASE j.type WHEN 'integer' THEN 1 WHEN 'real' THEN 1 WHEN 'text' THEN 2 "
            "WHEN 'true' THEN 3 WHEN 'false' THEN 3 WHEN 'array' THEN 4 WHEN 'object' THEN 5 END)"
        )
        return Ex(
            f"(CASE WHEN {lst.sql} IS NULL THEN NULL "
            f"WHEN {x.sql} IS NULL THEN (CASE WHEN json_array_length({lst.sql}) = 0 THEN 0 END) "
            f"WHEN EXISTS (SELECT 1 FROM json_each({lst.sql}) j WHERE j.value = {nx} AND {je_code} = {cls_code_sql(x)}) THEN 1 "
            f"WHEN EXISTS (SELECT 1 FROM json_each({lst.sql}) j WHERE j.type = 'null') THEN NULL "
            "ELSE 0 END)",
            BOOL,
        )

    def _arith(self, op: str, a: Ex, b: Ex, node: Any) -> Ex:
        ka, kb = a.type.kind, b.type.kind
        if "null" in (ka, kb):
            return Ex("NULL", NULL)
        for x in (a, b):
            if x.type.kind in ("node", "rel", "path", "bool"):
                raise self.err(f"Type mismatch: cannot apply '{op}' to {x.type.kind}", node)
        na, nb = native(a), native(b)
        if op == "+":
            if ka == "list" or kb == "list":
                if ka == "list" and kb == "list":
                    et = a.type.elem if a.type.elem == b.type.elem else ANY
                    return Ex(f"sqg_jconcat({a.sql}, {b.sql})", list_of(et or ANY))
                if ka == "list":
                    return Ex(f"sqg_jappend({a.sql}, {json_text(b)})", a.type)
                return Ex(f"sqg_jprepend({json_text(a)}, {b.sql})", b.type)
            if ka == "map" or kb == "map":
                raise self.err("Type mismatch: cannot add maps", node)
            if is_num(a.type) and is_num(b.type):
                t = INT if ka == kb == "int" else FLOAT
                return Ex(f"({na} + {nb})", t)
            if ka == "str" and kb == "str":
                return Ex(f"({na} || {nb})", STR)
            if ka == "str" and is_num(b.type):
                return Ex(f"({na} || CAST({nb} AS TEXT))", STR)
            if kb == "str" and is_num(a.type):
                return Ex(f"(CAST({na} AS TEXT) || {nb})", STR)
            return Ex(f"sqg_add({na}, {nb})", SCALAR)
        if ka in ("str", "list", "map") or kb in ("str", "list", "map"):
            raise self.err(f"Type mismatch: cannot apply '{op}' to {ka} and {kb}", node)
        both_int = ka == kb == "int"
        t = INT if both_int else (FLOAT if (is_num(a.type) and is_num(b.type)) else SCALAR)
        if op == "-":
            return Ex(f"({na} - {nb})", t)
        if op == "*":
            return Ex(f"({na} * {nb})", t)
        if op == "/":
            if both_int:
                return Ex(f"(CASE WHEN {nb} = 0 THEN sqg_divzero() ELSE {na} / {nb} END)", INT)
            return Ex(f"({na} / {nb})", t)
        if op == "%":
            if both_int:
                return Ex(f"(CASE WHEN {nb} = 0 THEN sqg_divzero() ELSE {na} % {nb} END)", INT)
            return Ex(f"sqg_mod({na}, {nb})", t if t != INT else FLOAT)
        if op == "^":
            return Ex(f"sqg_pow({na}, {nb})", FLOAT)
        raise self.unsupported(f"operator {op} is not supported", node)

    # ---------------------------------------------------------- CASE / literals

    def e_Case(self, n: A.Case, scope: dict[str, Var]) -> Ex:
        branches: list[tuple[str, Ex]] = []
        subject = self.expr(n.subject, scope) if n.subject is not None else None
        for w, r in n.whens:
            we = self.expr(w, scope)
            cond = self._compare("=", subject, we, n).sql if subject is not None else native(we)
            branches.append((cond, self.expr(r, scope)))
        default = self.expr(n.default, scope) if n.default is not None else Ex("NULL", NULL)
        t, sqls = self.unify([e for _, e in branches] + [default], n)
        parts = " ".join(f"WHEN {c} THEN {s}" for (c, _), s in zip(branches, sqls))
        return Ex(f"(CASE {parts} ELSE {sqls[-1]} END)", t)

    def e_ListLiteral(self, n: A.ListLiteral, scope: dict[str, Var]) -> Ex:
        if all(isinstance(i, A.Literal) for i in n.items):
            vals = [i.value for i in n.items]  # type: ignore[union-attr]
            text = json.dumps(vals, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            return Ex(self.bind(text), list_of(_json_kind_of_py(vals)))
        exs = [self.expr(i, scope) for i in n.items]
        kinds = {e.type for e in exs if e.type.kind != "null"}
        if len(kinds) == 1:
            et = next(iter(kinds))
        elif not kinds:
            et = ANY
        else:
            if any(t.kind in ("node", "rel", "path") for t in kinds):
                raise self.unsupported("lists mixing nodes/relationships with other types are not supported", n)
            et = FLOAT if {t.kind for t in kinds} <= set(_NUMERIC) else ANY
        return Ex(f"json_array({', '.join(enc(e) for e in exs)})", list_of(et))

    def e_MapLiteral(self, n: A.MapLiteral, scope: dict[str, Var]) -> Ex:
        fields: dict[str, T] = {}
        parts: list[str] = []
        for k, v in n.items:
            ex = self.expr(v, scope)
            fields[k] = ex.type
            parts.append(f"{self.bind(k)}, {enc(ex)}")
        return Ex(f"json_object({', '.join(parts)})", map_of(fields))

    # ---------------------------------------------------------- iteration constructs

    @staticmethod
    def je_elem(alias: str, t: T) -> str:
        if t.kind == "any":
            return (
                f"(CASE {alias}.type WHEN 'null' THEN NULL WHEN 'text' THEN json_quote({alias}.value) "
                f"WHEN 'true' THEN 'true' WHEN 'false' THEN 'false' "
                f"WHEN 'array' THEN {alias}.value WHEN 'object' THEN {alias}.value "
                f"ELSE CAST({alias}.value AS TEXT) END)"
            )
        return f"{alias}.value"

    def _iter(self, var: str, source: A.Expr, scope: dict[str, Var], node: Any) -> tuple[Ex, str, T, dict[str, Var]]:
        src = self.expr(source, scope)
        if src.type.kind not in ("list", "any", "null"):
            raise self.err(f"Type mismatch: expected List but was {src.type.kind}", node)
        et = elem_type(src.type)
        alias = f"je{self.uid()}"
        child = dict(scope)
        child[var] = Var(self.je_elem(alias, et), et)
        return src, alias, et, child

    def e_ListComprehension(self, n: A.ListComprehension, scope: dict[str, Var]) -> Ex:
        src, alias, et, child = self._iter(n.var, n.source, scope, n)
        where = f" WHERE {self.cond(n.where, child)}" if n.where is not None else ""
        if n.projection is not None:
            proj = self.expr(n.projection, child)
            rt = proj.type
        else:
            proj = Ex(child[n.var].sql, et)
            rt = et
        return Ex(
            f"(SELECT json_group_array({enc(proj)}) FROM json_each({src.sql}) {alias}{where})",
            list_of(rt),
        )

    def e_Quantifier(self, n: A.Quantifier, scope: dict[str, Var]) -> Ex:
        """all/any/none/single with Neo4j's three-valued logic (a null predicate is "unknown")."""
        src, alias, et, child = self._iter(n.var, n.source, scope, n)
        pred = self.cond(n.where, child)
        counts = (
            "SELECT count(CASE WHEN p THEN 1 END) AS t, count(CASE WHEN NOT p THEN 1 END) AS f, "
            f"count(CASE WHEN p IS NULL THEN 1 END) AS u FROM (SELECT {pred} AS p FROM json_each({src.sql}) {alias})"
        )
        verdict = {
            "all": "CASE WHEN f > 0 THEN 0 WHEN u > 0 THEN NULL ELSE 1 END",
            "any": "CASE WHEN t > 0 THEN 1 WHEN u > 0 THEN NULL ELSE 0 END",
            "none": "CASE WHEN t > 0 THEN 0 WHEN u > 0 THEN NULL ELSE 1 END",
            "single": "CASE WHEN t > 1 THEN 0 WHEN u > 0 THEN NULL WHEN t = 1 THEN 1 ELSE 0 END",
        }[n.kind]
        return Ex(
            f"(CASE WHEN {src.sql} IS NULL THEN NULL ELSE (SELECT {verdict} FROM ({counts})) END)", BOOL
        )

    # ---------------------------------------------------------- function calls

    def _args(self, n: A.FunctionCall, scope: dict[str, Var], lo: int, hi: int | None = None) -> list[Ex]:
        hi = lo if hi is None else hi
        if not (lo <= len(n.args) <= hi):
            exp = str(lo) if lo == hi else f"{lo} to {hi}"
            raise self.err(f"Function {n.name}() expects {exp} argument(s), got {len(n.args)}", n)
        return [self.expr(a, scope) for a in n.args]

    def e_FunctionCall(self, n: A.FunctionCall, scope: dict[str, Var]) -> Ex:
        lname = n.name.lower()
        if lname in AGGREGATES:
            return self._aggregate(lname, n, scope)
        if n.star or n.distinct:
            raise self.err(f"{n.name}(*)/DISTINCT is only valid for aggregate functions", n)
        m = getattr(self, "fn_" + lname, None)
        if m is None:
            raise self.unsupported(f"Function {n.name}() is not supported", n)
        return m(n, scope)

    # -- aggregates
    def _aggregate(self, name: str, n: A.FunctionCall, scope: dict[str, Var]) -> Ex:
        self.saw_agg = True
        if name == "count":
            if n.star:
                return Ex("count(*)", INT)
            (x,) = self._args(n, scope, 1)
            d = "DISTINCT " if n.distinct else ""
            # any-typed values are counted by their JSON text so that true and 1 stay distinct
            key = x.sql if (d and x.type.kind == "any") else native(x)
            return Ex(f"count({d}{key})", INT)
        if n.star:
            raise self.err(f"{n.name}(*) is not valid", n)
        (x,) = self._args(n, scope, 1)
        d = "DISTINCT " if n.distinct else ""
        if name == "collect":
            return Ex(
                f"json_group_array({d}{enc(x)}) FILTER (WHERE {native(x)} IS NOT NULL)",
                list_of(x.type if x.type.kind != "null" else ANY),
            )
        if x.type.kind in ("node", "rel", "path", "list", "map", "bool") and name != "min" and name != "max":
            raise self.err(f"Type mismatch: {n.name}() expects numbers, got {x.type.kind}", n)
        if name == "sum":
            t = x.type if is_num(x.type) else SCALAR
            return Ex(f"coalesce(sum({d}{native(x)}), 0)", t)
        if name == "avg":
            return Ex(f"avg({d}{native(x)})", FLOAT)
        t = x.type if x.type.kind in ("int", "float", "str", "node", "rel") else SCALAR
        return Ex(f"{name}({d}{native(x)})", t)

    # -- entity functions
    def fn_id(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "null":
            return Ex("NULL", NULL)
        if x.type.kind not in ("node", "rel"):
            raise self.err(f"Type mismatch: id() expects Node or Relationship, got {x.type.kind}", n)
        return Ex(x.sql, INT)

    def fn_elementid(self, n, scope):
        e = self.fn_id(n, scope)
        return Ex(f"CAST({e.sql} AS TEXT)", STR) if e.sql != "NULL" else e

    def fn_labels(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "null":
            return Ex("NULL", NULL)
        if x.type.kind != "node":
            raise self.err(f"Type mismatch: labels() expects Node, got {x.type.kind}", n)
        self.reads_graph = True
        return Ex(
            f"(CASE WHEN {x.sql} IS NULL THEN NULL ELSE (SELECT json_group_array(label) FROM "
            f"(SELECT label FROM node_labels WHERE node_id = {x.sql} ORDER BY label)) END)",
            list_of(STR),
        )

    def fn_type(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "null":
            return Ex("NULL", NULL)
        if x.type.kind != "rel":
            raise self.err(f"Type mismatch: type() expects Relationship, got {x.type.kind}", n)
        self.reads_graph = True
        return Ex(f"(SELECT type FROM edges WHERE id = {x.sql})", STR)

    def _props_json(self, x: Ex, n: Any) -> str:
        if x.type.kind in ("node", "rel"):
            self.reads_graph = True
            return self.props_of(x.type.kind, x.sql, None)
        if x.type.kind in ("map", "any"):
            return x.sql
        raise self.err(f"Type mismatch: expected Node, Relationship or Map but was {x.type.kind}", n)

    def fn_properties(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "null":
            return Ex("NULL", NULL)
        return Ex(self._props_json(x, n), MAP)

    def fn_keys(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "null":
            return Ex("NULL", NULL)
        p = self._props_json(x, n)
        return Ex(
            f"(CASE WHEN {p} IS NULL THEN NULL ELSE (SELECT json_group_array(key) FROM json_each({p})) END)",
            list_of(STR),
        )

    def fn_startnode(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind != "rel":
            raise self.err("startNode() expects a Relationship", n)
        self.reads_graph = True
        return Ex(f"(SELECT src FROM edges WHERE id = {x.sql})", NODE)

    def fn_endnode(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind != "rel":
            raise self.err("endNode() expects a Relationship", n)
        self.reads_graph = True
        return Ex(f"(SELECT dst FROM edges WHERE id = {x.sql})", NODE)

    def fn_nodes(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind != "path":
            raise self.err("nodes() expects a Path", n)
        return Ex(f"json_extract({x.sql}, '$.n')", list_of(NODE))

    def fn_relationships(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind != "path":
            raise self.err("relationships() expects a Path", n)
        return Ex(f"json_extract({x.sql}, '$.r')", list_of(REL))

    fn_rels = fn_relationships

    # -- generic scalar functions
    def fn_coalesce(self, n, scope):
        if not n.args:
            raise self.err("coalesce() expects at least one argument", n)
        exs = [self.expr(a, scope) for a in n.args]
        t, sqls = self.unify(exs, n)
        return Ex(f"coalesce({', '.join(sqls)})", t)

    def fn_exists(self, n, scope):
        (x,) = self._args(n, scope, 1)
        return Ex(f"({x.sql} IS NOT NULL)", BOOL)

    def fn_size(self, n, scope):
        (x,) = self._args(n, scope, 1)
        k = x.type.kind
        if k == "null":
            return Ex("NULL", NULL)
        if k == "list":
            return Ex(f"json_array_length({x.sql})", INT)
        if k == "str":
            return Ex(f"length({x.sql})", INT)
        if k in ("any", "scalar"):
            v = x.sql
            if k == "scalar":
                return Ex(f"length({v})", INT)
            return Ex(
                f"(CASE json_type({v}) WHEN 'array' THEN json_array_length({v}) ELSE length({native(x)}) END)",
                INT,
            )
        raise self.err(f"Type mismatch: size() expects a List or String, got {k}", n)

    fn_char_length = fn_size
    fn_character_length = fn_size

    def fn_length(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "path":
            return Ex(f"json_array_length(json_extract({x.sql}, '$.r'))", INT)
        return self.fn_size(n, scope)

    def fn_isempty(self, n, scope):
        e = self.fn_size(n, scope)
        return Ex(f"({e.sql} = 0)", BOOL)

    def fn_head(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind not in ("list", "any", "null"):
            raise self.err("head() expects a List", n)
        et = elem_type(x.type)
        return Ex(extract(x.sql, "'$[0]'", et), et)

    def fn_last(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind not in ("list", "any", "null"):
            raise self.err("last() expects a List", n)
        et = elem_type(x.type)
        return Ex(extract(x.sql, "'$[#-1]'", et), et)

    def fn_tail(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind not in ("list", "any", "null"):
            raise self.err("tail() expects a List", n)
        return Ex(f"sqg_jtail({x.sql})", x.type if x.type.kind == "list" else LIST_ANY)

    def fn_reverse(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind == "str":
            return Ex(f"sqg_reverse({x.sql})", STR)
        if x.type.kind == "list":
            return Ex(f"sqg_jreverse({x.sql})", x.type)
        if x.type.kind == "null":
            return Ex("NULL", NULL)
        raise self.err("reverse() expects a String or List", n)

    def fn_range(self, n, scope):
        args = self._args(n, scope, 2, 3)
        a, b = native(args[0]), native(args[1])
        s = native(args[2]) if len(args) == 3 else "1"
        return Ex(
            f"(CASE WHEN ({a}) IS NULL OR ({b}) IS NULL OR ({s}) IS NULL THEN NULL ELSE "
            f"(WITH RECURSIVE r(x, b, s) AS (SELECT ({a}), ({b}), ({s}) WHERE ({s}) <> 0 AND "
            f"(CASE WHEN ({s}) > 0 THEN ({a}) <= ({b}) ELSE ({a}) >= ({b}) END) "
            f"UNION ALL SELECT x + s, b, s FROM r WHERE CASE WHEN s > 0 THEN x + s <= b ELSE x + s >= b END) "
            f"SELECT json_group_array(x) FROM r) END)",
            list_of(INT),
        )

    def fn_timestamp(self, n, scope):
        self._args(n, scope, 0)
        return Ex("CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)", INT)

    def fn_rand(self, n, scope):
        self._args(n, scope, 0)
        return Ex("sqg_rand()", FLOAT)

    # -- strings
    def _str_fn(self, n, scope, sql_fmt: str, nargs: int = 1, rt: T = STR) -> Ex:
        args = self._args(n, scope, nargs)
        for a in args:
            if a.type.kind in ("node", "rel", "path", "list", "map", "bool"):
                raise self.err(f"Type mismatch: {n.name}() expects a String, got {a.type.kind}", n)
        return Ex(sql_fmt.format(*[native(a) for a in args]), rt)

    def fn_tolower(self, n, scope):
        return self._str_fn(n, scope, "sqg_lower({0})")

    def fn_toupper(self, n, scope):
        return self._str_fn(n, scope, "sqg_upper({0})")

    def fn_trim(self, n, scope):
        return self._str_fn(n, scope, f"trim({{0}}, {_WS})")

    def fn_ltrim(self, n, scope):
        return self._str_fn(n, scope, f"ltrim({{0}}, {_WS})")

    def fn_rtrim(self, n, scope):
        return self._str_fn(n, scope, f"rtrim({{0}}, {_WS})")

    def fn_replace(self, n, scope):
        return self._str_fn(n, scope, "replace({0}, {1}, {2})", 3)

    def fn_split(self, n, scope):
        return self._str_fn(n, scope, "sqg_split({0}, {1})", 2, list_of(STR))

    def fn_left(self, n, scope):
        return self._str_fn(n, scope, "(CASE WHEN {1} <= 0 THEN '' ELSE substr({0}, 1, {1}) END)", 2)

    def fn_right(self, n, scope):
        return self._str_fn(n, scope, "(CASE WHEN {1} <= 0 THEN '' ELSE substr({0}, -({1})) END)", 2)

    def fn_substring(self, n, scope):
        args = self._args(n, scope, 2, 3)
        s, st = native(args[0]), native(args[1])
        if len(args) == 3:
            return Ex(f"substr({s}, ({st}) + 1, {native(args[2])})", STR)
        return Ex(f"substr({s}, ({st}) + 1)", STR)

    # -- conversions
    def fn_tostring(self, n, scope):
        (x,) = self._args(n, scope, 1)
        k = x.type.kind
        if k == "null":
            return Ex("NULL", NULL)
        if k == "str":
            return Ex(x.sql, STR)
        if k == "bool":
            return Ex(f"(CASE WHEN {x.sql} IS NULL THEN NULL WHEN {x.sql} THEN 'true' ELSE 'false' END)", STR)
        if k in ("int", "float", "scalar"):
            return Ex(f"sqg_tostr({x.sql})", STR)
        if k == "any":
            return Ex(
                f"(CASE json_type({x.sql}) WHEN 'true' THEN 'true' WHEN 'false' THEN 'false' "
                f"WHEN 'array' THEN {x.sql} WHEN 'object' THEN {x.sql} ELSE sqg_tostr({native(x)}) END)",
                STR,
            )
        if k in ("list", "map"):
            return Ex(x.sql, STR)
        raise self.err(f"Type mismatch: toString() does not accept {k}", n)

    def fn_tointeger(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind in ("node", "rel", "path", "list", "map"):
            raise self.err(f"Type mismatch: toInteger() does not accept {x.type.kind}", n)
        return Ex(f"sqg_toint({native(x)})", INT)

    fn_tointegerornull = fn_tointeger

    def fn_tofloat(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind in ("node", "rel", "path", "list", "map"):
            raise self.err(f"Type mismatch: toFloat() does not accept {x.type.kind}", n)
        return Ex(f"sqg_tofloat({native(x)})", FLOAT)

    fn_tofloatornull = fn_tofloat

    def fn_toboolean(self, n, scope):
        (x,) = self._args(n, scope, 1)
        k = x.type.kind
        if k == "bool":
            return x
        if k == "str":
            return Ex(f"(CASE lower({x.sql}) WHEN 'true' THEN 1 WHEN 'false' THEN 0 END)", BOOL)
        if k == "any":
            return Ex(
                f"(CASE json_type({x.sql}) WHEN 'true' THEN 1 WHEN 'false' THEN 0 "
                f"WHEN 'text' THEN (CASE lower({native(x)}) WHEN 'true' THEN 1 WHEN 'false' THEN 0 END) END)",
                BOOL,
            )
        if k == "null":
            return Ex("NULL", NULL)
        raise self.err(f"Type mismatch: toBoolean() does not accept {k}", n)

    # -- math
    def _num1(self, n, scope, sql_fmt: str, rt: T) -> Ex:
        (x,) = self._args(n, scope, 1)
        if x.type.kind in ("node", "rel", "path", "list", "map", "str", "bool"):
            raise self.err(f"Type mismatch: {n.name}() expects a number, got {x.type.kind}", n)
        return Ex(sql_fmt.format(native(x)), rt)

    def fn_abs(self, n, scope):
        (x,) = self._args(n, scope, 1)
        if x.type.kind in ("node", "rel", "path", "list", "map", "str", "bool"):
            raise self.err(f"Type mismatch: abs() expects a number, got {x.type.kind}", n)
        return Ex(f"abs({native(x)})", x.type if is_num(x.type) else SCALAR)

    def fn_round(self, n, scope):
        args = self._args(n, scope, 1, 2)
        if len(args) == 2:
            return Ex(f"sqg_round({native(args[0])}, {native(args[1])})", FLOAT)
        return Ex(f"sqg_round({native(args[0])})", FLOAT)

    def fn_ceil(self, n, scope):
        return self._num1(n, scope, "sqg_ceil({0})", FLOAT)

    def fn_floor(self, n, scope):
        return self._num1(n, scope, "sqg_floor({0})", FLOAT)

    def fn_sqrt(self, n, scope):
        return self._num1(n, scope, "sqg_sqrt({0})", FLOAT)

    def fn_sign(self, n, scope):
        return self._num1(n, scope, "sqg_sign({0})", INT)

    def fn_exp(self, n, scope):
        return self._num1(n, scope, "sqg_exp({0})", FLOAT)

    def fn_log(self, n, scope):
        return self._num1(n, scope, "sqg_log({0})", FLOAT)

    def fn_log10(self, n, scope):
        return self._num1(n, scope, "sqg_log10({0})", FLOAT)

    def fn_sin(self, n, scope):
        return self._num1(n, scope, "sqg_sin({0})", FLOAT)

    def fn_cos(self, n, scope):
        return self._num1(n, scope, "sqg_cos({0})", FLOAT)

    def fn_pi(self, n, scope):
        self._args(n, scope, 0)
        return Ex("3.141592653589793", FLOAT)

    def fn_e(self, n, scope):
        self._args(n, scope, 0)
        return Ex("2.718281828459045", FLOAT)

    # ---------------------------------------------------------- EXISTS { ... }

    def e_ExistsSubquery(self, n: A.ExistsSubquery, scope: dict[str, Var]) -> Ex:
        return Ex(self.exists_subquery(n, scope), BOOL)

    def exists_subquery(self, n: A.ExistsSubquery, scope: dict[str, Var]) -> str:  # implemented by Compiler
        raise NotImplementedError


def free_vars(node: Any, bound: frozenset[str] = frozenset(), skip_aggregates: bool = False) -> set[str]:
    """Variable names referenced by an expression AST (excluding comprehension-local ones).

    With ``skip_aggregates`` the arguments of aggregate function calls are ignored.
    """
    out: set[str] = set()

    def walk(x: Any, b: frozenset[str]) -> None:
        if skip_aggregates and isinstance(x, A.FunctionCall) and x.name.lower() in AGGREGATES:
            return
        if isinstance(x, A.Variable):
            if x.name not in b:
                out.add(x.name)
        elif isinstance(x, (A.ListComprehension, A.Quantifier)):
            walk(x.source, b)
            inner = b | {x.var}
            for child in (getattr(x, "where", None), getattr(x, "projection", None)):
                if child is not None:
                    walk(child, inner)
        elif isinstance(x, A.ExistsSubquery):
            out.add("$exists")
        elif isinstance(x, A.MapProjection):
            walk(x.base, b)
            for what, arg in x.items:
                if what == "var" and arg not in b:
                    out.add(arg)
                elif what == "kv":
                    walk(arg[1], b)
        elif isinstance(x, list):
            for i in x:
                walk(i, b)
        elif isinstance(x, tuple):
            for i in x:
                walk(i, b)
        elif hasattr(x, "__dataclass_fields__"):
            for f in x.__dataclass_fields__:
                if f == "pos":
                    continue
                walk(getattr(x, f), b)

    walk(node, bound)
    return out


def split_conjuncts(node: A.Expr | None) -> list[A.Expr]:
    """Split ``a AND b AND c`` into ``[a, b, c]``."""
    if node is None:
        return []
    if isinstance(node, A.BinaryOp) and node.op == "AND":
        return split_conjuncts(node.left) + split_conjuncts(node.right)
    return [node]


def _has_aggregate(x: Any) -> bool:
    if isinstance(x, A.FunctionCall) and x.name.lower() in AGGREGATES:
        return True
    return any(_has_aggregate(c) for c in _children(x))


def _children(x: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(x, (list, tuple)):
        for i in x:
            out.extend(_children(i) if isinstance(i, (list, tuple)) else ([i] if isinstance(i, A.Base) else []))
        return out
    for f in getattr(x, "__dataclass_fields__", {}):
        if f == "pos":
            continue
        v = getattr(x, f)
        if isinstance(v, A.Base):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(_children(v))
    return out


def ast_equal(a: Any, b: Any) -> bool:
    """Structural equality of two expression ASTs, ignoring source positions."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(ast_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "__dataclass_fields__"):
        return all(ast_equal(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__ if f != "pos")
    return bool(a == b)


def implicit_grouping_violation(expr: Any, key_exprs: list[Any], key_vars: set[str]) -> str | None:
    """Return the offending variable if an aggregating expression uses non-grouped variables.

    Sub-expressions outside aggregate calls must not reference variables unless the variable is
    itself a grouping key, or the sub-expression equals a grouping key expression.
    """

    def rec(x: Any, bound: frozenset[str]) -> str | None:
        if isinstance(x, A.FunctionCall) and x.name.lower() in AGGREGATES:
            return None
        fv = free_vars(x, bound) - {"$exists"}
        if not fv:
            return None
        if not _has_aggregate(x) and (fv <= key_vars or any(ast_equal(x, k) for k in key_exprs)):
            return None
        if isinstance(x, (A.ListComprehension, A.Quantifier)):
            r = rec(x.source, bound)
            if r:
                return r
            inner = bound | {x.var}
            for c in (getattr(x, "where", None), getattr(x, "projection", None)):
                if c is not None:
                    r = rec(c, inner)
                    if r:
                        return r
            return None
        if isinstance(x, A.Variable):
            return x.name
        kids = _children(x)
        for c in kids:
            r = rec(c, bound)
            if r:
                return r
        return sorted(fv)[0] if not kids else None

    return rec(expr, frozenset())
