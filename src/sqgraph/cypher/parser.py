"""Cypher text -> AST (via lark)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
    VisitError,
)

from ..errors import CypherSyntaxError, CypherUnsupportedError
from . import ast as A

_GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")

_UNSUPPORTED_WORDS = {
    "FOREACH": "FOREACH is not supported; use UNWIND with CREATE/MERGE/SET instead",
    "LOAD": "LOAD CSV is not supported",
    "USING": "query hints (USING INDEX / USING SCAN) are not supported",
    "CONSTRAINT": "constraints are not supported",
    "TERMINATE": "TERMINATE is not supported",
}

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "\\": "\\", "'": "'", '"': '"', "`": "`"}


def _line_col(text: str, pos: int) -> tuple[int, int]:
    line = text.count("\n", 0, pos) + 1
    col = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return line, col


def unescape_string(raw: str) -> str:
    """Decode the body of a quoted Cypher string literal (without the quotes)."""
    if "\\" not in raw:
        return raw
    out: list[str] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            out.append("\\")
            break
        e = raw[i]
        if e in _ESCAPES:
            out.append(_ESCAPES[e])
            i += 1
        elif e in "uU":
            n = 4 if e == "u" else 8
            hexpart = raw[i + 1 : i + 1 + n]
            if len(hexpart) != n or not re.fullmatch(r"[0-9a-fA-F]+", hexpart):
                raise ValueError(f"invalid unicode escape \\{e}{hexpart}")
            out.append(chr(int(hexpart, 16)))
            i += 1 + n
        else:
            raise ValueError(f"invalid escape sequence \\{e}")
    return "".join(out)


def _unbacktick(s: str) -> str:
    if s.startswith("`"):
        return s[1:-1].replace("``", "`")
    return str(s)


@v_args(meta=True)
class _Builder(Transformer):
    """Builds AST dataclasses from the lark parse tree."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    # -- helpers
    def _p(self, node: Any, meta: Any) -> Any:
        try:
            node.pos = meta.start_pos
        except AttributeError:
            pass
        return node

    # -- names & literals
    def name(self, meta, ch):
        return _unbacktick(str(ch[0]))

    def int_lit(self, meta, ch):
        s = str(ch[0])
        return self._p(A.Literal(int(s, 16) if s[:2] in ("0x", "0X") else int(s)), meta)

    def float_lit(self, meta, ch):
        return self._p(A.Literal(float(ch[0])), meta)

    def string_lit(self, meta, ch):
        raw = str(ch[0])[1:-1]
        try:
            return self._p(A.Literal(unescape_string(raw)), meta)
        except ValueError as exc:
            line, col = _line_col(self.text, meta.start_pos)
            raise CypherSyntaxError(str(exc), line=line, column=col, query=self.text) from None

    def true_lit(self, meta, ch):
        return self._p(A.Literal(True), meta)

    def false_lit(self, meta, ch):
        return self._p(A.Literal(False), meta)

    def null_lit(self, meta, ch):
        return self._p(A.Literal(None), meta)

    def param(self, meta, ch):
        return self._p(A.Param(str(ch[0])[1:]), meta)

    def variable(self, meta, ch):
        return self._p(A.Variable(ch[0]), meta)

    # -- expressions
    def expression(self, meta, ch):
        node = ch[0]
        node.span = (meta.start_pos, meta.end_pos)
        return node

    def _bin(self, op, meta, ch):
        return self._p(A.BinaryOp(op, ch[0], ch[1]), meta)

    def or_op(self, meta, ch):
        return self._bin("OR", meta, ch)

    def xor_op(self, meta, ch):
        return self._bin("XOR", meta, ch)

    def and_op(self, meta, ch):
        return self._bin("AND", meta, ch)

    def not_op(self, meta, ch):
        return self._p(A.UnaryOp("NOT", ch[0]), meta)

    def comp_op(self, meta, ch):
        s = str(ch[0])
        return "<>" if s == "!=" else s

    def compare(self, meta, ch):
        operands = ch[0::2]
        ops = ch[1::2]
        result = None
        for i, op in enumerate(ops):
            cmp_ = self._p(A.BinaryOp(op, operands[i], operands[i + 1]), meta)
            result = cmp_ if result is None else self._p(A.BinaryOp("AND", result, cmp_), meta)
        return result

    def is_null(self, meta, ch):
        return self._p(A.IsNull(ch[0], False), meta)

    def is_not_null(self, meta, ch):
        return self._p(A.IsNull(ch[0], True), meta)

    def in_op(self, meta, ch):
        return self._bin("IN", meta, ch)

    def starts_with(self, meta, ch):
        return self._bin("STARTS", meta, ch)

    def ends_with(self, meta, ch):
        return self._bin("ENDS", meta, ch)

    def contains(self, meta, ch):
        return self._bin("CONTAINS", meta, ch)

    def regex_op(self, meta, ch):
        return self._bin("=~", meta, ch)

    def add(self, meta, ch):
        return self._bin("+", meta, ch)

    def sub(self, meta, ch):
        return self._bin("-", meta, ch)

    def mul(self, meta, ch):
        return self._bin("*", meta, [ch[0], ch[2]])

    def div(self, meta, ch):
        return self._bin("/", meta, ch)

    def mod(self, meta, ch):
        return self._bin("%", meta, ch)

    def pow(self, meta, ch):
        return self._bin("^", meta, ch)

    def neg(self, meta, ch):
        inner = ch[0]
        if isinstance(inner, A.Literal) and isinstance(inner.value, (int, float)) and not isinstance(inner.value, bool):
            return self._p(A.Literal(-inner.value), meta)
        return self._p(A.UnaryOp("NEG", inner), meta)

    def pos(self, meta, ch):
        return ch[0]

    def property(self, meta, ch):
        return self._p(A.Property(ch[0], ch[1]), meta)

    def index(self, meta, ch):
        return self._p(A.Index(ch[0], ch[1]), meta)

    def slice(self, meta, ch):
        return self._p(A.Slice(ch[0], ch[1], ch[2]), meta)

    def label_check(self, meta, ch):
        return self._p(A.LabelCheck(ch[0], ch[1]), meta)

    def call_noargs(self, meta, ch):
        return self._p(A.FunctionCall(str(ch[0]), []), meta)

    def call_star(self, meta, ch):
        return self._p(A.FunctionCall(str(ch[0]), [], star=True), meta)

    def call_args_fn(self, meta, ch):
        return self._p(A.FunctionCall(str(ch[0]), ch[2], distinct=ch[1] is not None), meta)

    def quantifier_or_call(self, meta, ch):
        fname, var, source, where = str(ch[0]), ch[1], ch[2], ch[3]
        if fname.lower() in ("all", "any", "none", "single"):
            if where is None:
                line, col = _line_col(self.text, meta.start_pos)
                raise CypherSyntaxError(
                    f"{fname}() requires a WHERE predicate", line=line, column=col, query=self.text
                )
            return self._p(A.Quantifier(fname.lower(), var, source, where), meta)
        if where is not None:
            line, col = _line_col(self.text, meta.start_pos)
            raise CypherSyntaxError(
                f"unexpected WHERE in call to {fname}()", line=line, column=col, query=self.text
            )
        arg = self._p(A.BinaryOp("IN", self._p(A.Variable(var), meta), source), meta)
        return self._p(A.FunctionCall(fname, [arg]), meta)

    def call_args(self, meta, ch):
        return list(ch)

    def case_when(self, meta, ch):
        return (ch[0], ch[1])

    def case_expr(self, meta, ch):
        subject, default = ch[0], ch[-1]
        whens = [c for c in ch[1:-1]]
        return self._p(A.Case(subject, whens, default), meta)

    def list_literal(self, meta, ch):
        return self._p(A.ListLiteral(ch[0] or []), meta)

    def list_comprehension(self, meta, ch):
        return self._p(A.ListComprehension(ch[0], ch[1], ch[2], ch[3]), meta)

    def map_item(self, meta, ch):
        return (ch[0], ch[1])

    def map_literal(self, meta, ch):
        return self._p(A.MapLiteral([c for c in ch if c is not None]), meta)

    def pi_prop(self, meta, ch):
        return ("prop", ch[0])

    def pi_all(self, meta, ch):
        return ("all", None)

    def pi_kv(self, meta, ch):
        return ("kv", (ch[0], ch[1]))

    def pi_var(self, meta, ch):
        return ("var", ch[0])

    def map_proj_items(self, meta, ch):
        return list(ch)

    def map_projection(self, meta, ch):
        return self._p(A.MapProjection(ch[0], ch[1] or []), meta)

    def exists_body(self, meta, ch):
        return (ch[0], ch[1])

    def exists_subquery(self, meta, ch):
        patterns, where = ch[0]
        return self._p(A.ExistsSubquery(patterns, where), meta)

    def exists_fn(self, meta, ch):
        return self._p(A.FunctionCall("exists", [ch[0]]), meta)

    # -- patterns
    def node_labels(self, meta, ch):
        return list(ch)

    def pattern_props(self, meta, ch):
        return ch[0]

    def node_pattern(self, meta, ch):
        var, labels, props = ch
        return self._p(A.NodePattern(var, labels or [], props), meta)

    def rel_types(self, meta, ch):
        return list(ch)

    def range_any(self, meta, ch):
        return (1, None)

    def range_exact(self, meta, ch):
        n = int(str(ch[1]), 0) if str(ch[1]).lower().startswith("0x") else int(ch[1])
        return (n, n)

    def range_both(self, meta, ch):
        return (int(ch[1]), int(ch[2]))

    def range_min(self, meta, ch):
        return (int(ch[1]), None)

    def range_max(self, meta, ch):
        return (1, int(ch[1]) if len(ch) == 2 else int(ch[-1]))

    def rel_detail(self, meta, ch):
        return tuple(ch)  # (var, types, range, props)

    def _rel(self, direction, meta, ch):
        detail = ch[0]
        var = types = rng = props = None
        if detail is not None:
            var, types, rng, props = detail
        if rng is not None:
            lo, hi = rng
            if hi is not None and hi < lo:
                line, col = _line_col(self.text, meta.start_pos)
                raise CypherSyntaxError(
                    f"invalid variable-length range *{lo}..{hi}", line=line, column=col, query=self.text
                )
            return self._p(A.RelPattern(var, types or [], direction, True, lo, hi, props), meta)
        return self._p(A.RelPattern(var, types or [], direction, False, 1, 1, props), meta)

    def rel_both(self, meta, ch):
        return self._rel("both", meta, ch)

    def rel_out(self, meta, ch):
        return self._rel("out", meta, ch)

    def rel_in(self, meta, ch):
        return self._rel("in", meta, ch)

    def pattern_chain(self, meta, ch):
        return list(ch)

    def pattern_part(self, meta, ch):
        return self._p(A.PatternPart(ch[0], ch[1]), meta)

    def pattern_list(self, meta, ch):
        return list(ch)

    # -- clauses
    def where_clause(self, meta, ch):
        return ch[0]

    def match_clause(self, meta, ch):
        return self._p(A.Match(ch[1], ch[2], optional=ch[0] is not None), meta)

    def unwind_clause(self, meta, ch):
        return self._p(A.Unwind(ch[0], ch[1]), meta)

    def projection_item(self, meta, ch):
        expr, alias = ch
        span = getattr(expr, "span", None)
        text = self.text[span[0] : span[1]].strip() if span else ""
        return self._p(A.ProjectionItem(expr, alias, text), meta)

    def star_items(self, meta, ch):
        return ("*", list(ch[1:]))

    def plain_items(self, meta, ch):
        return ("items", list(ch))

    def sort_item(self, meta, ch):
        d = ch[1]
        return self._p(A.SortItem(ch[0], d is not None and str(d).upper().startswith("DESC")), meta)

    def order_by(self, meta, ch):
        return ("order", list(ch))

    def skip_clause(self, meta, ch):
        return ("skip", ch[0])

    def limit_clause(self, meta, ch):
        return ("limit", ch[0])

    def _projection(self, meta, distinct, items, rest, where=None):
        kind, lst = items
        proj = A.Projection(lst, star=(kind == "*"), distinct=distinct is not None, where=where)
        for r in rest:
            if r is None:
                continue
            tag, val = r
            if tag == "order":
                proj.order_by = val
            elif tag == "skip":
                proj.skip = val
            elif tag == "limit":
                proj.limit = val
        return self._p(proj, meta)

    def with_clause(self, meta, ch):
        distinct, items, order, skip, limit, where = ch
        return self._p(A.With(self._projection(meta, distinct, items, [order, skip, limit], where)), meta)

    def return_clause(self, meta, ch):
        distinct, items, order, skip, limit = ch
        return self._p(A.Return(self._projection(meta, distinct, items, [order, skip, limit])), meta)

    def create_clause(self, meta, ch):
        return self._p(A.Create(ch[0]), meta)

    def set_prop(self, meta, ch):
        return self._p(A.SetItem("prop", ch[0], key=ch[1], expr=ch[2]), meta)

    def set_replace(self, meta, ch):
        return self._p(A.SetItem("replace", ch[0], expr=ch[1]), meta)

    def set_merge(self, meta, ch):
        return self._p(A.SetItem("merge", ch[0], expr=ch[1]), meta)

    def set_labels(self, meta, ch):
        return self._p(A.SetItem("labels", ch[0], labels=ch[1]), meta)

    def set_items(self, meta, ch):
        return list(ch)

    def set_clause(self, meta, ch):
        return self._p(A.SetClause(ch[0]), meta)

    def on_create(self, meta, ch):
        return ("create", ch[0])

    def on_match(self, meta, ch):
        return ("match", ch[0])

    def merge_clause(self, meta, ch):
        m = A.Merge(ch[0])
        for kind, items in ch[1:]:
            (m.on_create if kind == "create" else m.on_match).extend(items)
        return self._p(m, meta)

    def remove_prop(self, meta, ch):
        return self._p(A.RemoveItem("prop", ch[0], key=ch[1]), meta)

    def remove_labels(self, meta, ch):
        return self._p(A.RemoveItem("labels", ch[0], labels=ch[1]), meta)

    def remove_clause(self, meta, ch):
        return self._p(A.Remove(list(ch)), meta)

    def delete_clause(self, meta, ch):
        return self._p(A.Delete(list(ch[1:]), detach=ch[0] is not None), meta)

    def proc_name(self, meta, ch):
        return ".".join(ch)

    def yield_item(self, meta, ch):
        return self._p(A.YieldItem(ch[0], ch[1]), meta)

    def yield_clause(self, meta, ch):
        items = [c for c in ch if isinstance(c, A.YieldItem)]
        where = ch[-1] if not isinstance(ch[-1], A.YieldItem) else None
        return (items, where)

    def call_clause(self, meta, ch):
        name, args, y = ch
        yields, where = (y if y is not None else (None, None))
        return self._p(A.Call(name, args or [], yields, where), meta)

    def single_query(self, meta, ch):
        return self._p(A.SingleQuery(list(ch)), meta)

    def union_all(self, meta, ch):
        return "all"

    def union_distinct(self, meta, ch):
        return "distinct"

    def query(self, meta, ch):
        parts = [c for c in ch if isinstance(c, A.SingleQuery)]
        ops = {c for c in ch if isinstance(c, str)}
        if len(ops) > 1:
            line, col = _line_col(self.text, meta.start_pos)
            raise CypherSyntaxError(
                "cannot mix UNION and UNION ALL in one query", line=line, column=col, query=self.text
            )
        return self._p(A.Query(parts, union_all="all" in ops), meta)

    # -- schema commands
    def if_not_exists(self, meta, ch):
        return True

    def if_exists(self, meta, ch):
        return True

    def index_prop(self, meta, ch):
        return (ch[0], ch[1])

    def index_props(self, meta, ch):
        return list(ch)

    def ft_labels(self, meta, ch):
        return list(ch)

    def node_target(self, meta, ch):
        var, labels = ch
        return ("node", var, labels)

    def rel_target(self, meta, ch):
        var, types, _end = ch
        return ("relationship", var, types)

    def rel_end(self, meta, ch):
        return None

    def create_vector_index(self, meta, ch):
        name, ine, (entity, _var, label), (_v, prop), opts = ch
        return self._p(A.CreateVectorIndex(name, bool(ine), label, prop, opts, entity), meta)

    def create_fulltext_index(self, meta, ch):
        name, ine, (entity, _var, labels), props, opts = ch
        return self._p(A.CreateFulltextIndex(name, bool(ine), labels, [p for _, p in props], opts, entity), meta)

    def create_property_index(self, meta, ch):
        _kind, name, ine, (entity, _var, label), props = ch
        if isinstance(props, tuple):  # ON n.prop (without parentheses)
            props = [props]
        return self._p(A.CreatePropertyIndex(name, bool(ine), label, [p for _, p in props], entity), meta)

    def drop_index(self, meta, ch):
        return self._p(A.DropIndex(ch[0], bool(ch[1])), meta)

    def show_indexes(self, meta, ch):
        return self._p(A.ShowIndexes(), meta)

    def start(self, meta, ch):
        return ch[0]


@lru_cache(maxsize=1)
def _lark() -> Lark:
    return Lark(
        _GRAMMAR_PATH.read_text(encoding="utf-8"),
        parser="lalr",
        lexer="contextual",
        propagate_positions=True,
        maybe_placeholders=True,
        start="start",
    )


def _describe_token(tok: Token) -> str:
    v = str(tok.value)
    return "end of input" if tok.type == "$END" else repr(v)


def _friendly_expected(expected: set[str]) -> str:
    names = {
        "LPAR": "'('",
        "RPAR": "')'",
        "LSQB": "'['",
        "RSQB": "']'",
        "LBRACE": "'{'",
        "RBRACE": "'}'",
        "COMMA": "','",
        "COLON": "':'",
        "DOT": "'.'",
        "NAME": "an identifier",
        "BACKTICK": "an identifier",
        "INT": "a number",
        "FLOAT": "a number",
        "STRING": "a string",
        "PARAM": "a parameter",
        "$END": "end of input",
        "MINUS": "'-'",
        "EQUAL": "'='",
        "EQ": "'='",
        "STAR": "'*'",
        "PLUS": "'+'",
    }
    out: list[str] = []
    for e in sorted(expected):
        n = names.get(e, e if e.isupper() and len(e) > 1 else e)
        if n not in out:
            out.append(n)
    if len(out) > 10:
        out = out[:10] + ["..."]
    return ", ".join(out)


def _check_unsupported(text: str, pos: int | None) -> None:
    stripped = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", lambda m: " " * len(m.group(0)), text)
    checks = [
        (r"\b(shortestPath|allShortestPaths)\s*\(", "shortestPath()/allShortestPaths() are not supported"),
        (r"\bCALL\s*\{", "CALL { ... } subqueries are not supported"),
        (r"\breduce\s*\(", "reduce() is not supported; use UNWIND with aggregation instead"),
        (
            r"\[\s*\([^\]]*\)\s*<?-[^\]]*\|",
            "pattern comprehensions are not supported; use MATCH/OPTIONAL MATCH with collect()",
        ),
        (r"\bFOREACH\b", _UNSUPPORTED_WORDS["FOREACH"]),
        (r"\bLOAD\s+CSV\b", _UNSUPPORTED_WORDS["LOAD"]),
        (r"\bUSING\s+(INDEX|SCAN|JOIN)\b", _UNSUPPORTED_WORDS["USING"]),
        (r"\bCREATE\s+CONSTRAINT\b|\bDROP\s+CONSTRAINT\b", _UNSUPPORTED_WORDS["CONSTRAINT"]),
    ]
    for pat, msg in checks:
        m = re.search(pat, stripped, re.IGNORECASE)
        if m:
            line, col = _line_col(text, m.start())
            raise CypherUnsupportedError(msg, line=line, column=col, query=text)


def _syntax_error(text: str, exc: UnexpectedInput) -> CypherSyntaxError:
    pos = getattr(exc, "pos_in_stream", None)
    if isinstance(exc, UnexpectedToken):
        pos = exc.token.start_pos if exc.token.type != "$END" else len(text)
        found = _describe_token(exc.token)
        msg = f"Unexpected {found}"
        expected = getattr(exc, "expected", None) or getattr(exc, "accepts", None)
        if expected:
            msg += f"; expected {_friendly_expected(set(expected))}"
    elif isinstance(exc, UnexpectedEOF):
        pos = len(text)
        msg = "Unexpected end of input"
        if exc.expected:
            msg += f"; expected {_friendly_expected(set(exc.expected))}"
    elif isinstance(exc, UnexpectedCharacters):
        ch = text[exc.pos_in_stream] if exc.pos_in_stream is not None and exc.pos_in_stream < len(text) else "?"
        msg = f"Invalid input {ch!r}"
        pos = exc.pos_in_stream
    else:
        msg = str(exc)
    pos = pos or 0
    hint = ""
    window = text[max(0, pos - 1) : pos + 12]
    if re.search(r"\)\s*<?-{1,2}\s*[\[>(]|<-|->", window) and re.search(r"\bWHERE\b|\bAND\b|\bOR\b|\bNOT\b", text[:pos], re.I):
        hint = " (pattern predicates in expressions are not supported; use EXISTS { MATCH ... })"
    line, col = _line_col(text, pos)
    return CypherSyntaxError(msg + hint, line=line, column=col, query=text)


@lru_cache(maxsize=512)
def parse(text: str) -> A.Statement:
    """Parse a Cypher statement. Raises CypherSyntaxError / CypherUnsupportedError."""
    if not text or not text.strip():
        raise CypherSyntaxError("empty query", query=text)
    src = text.rstrip()
    while src.endswith(";"):
        src = src[:-1].rstrip()
    try:
        tree = _lark().parse(src)
    except UnexpectedInput as exc:
        _check_unsupported(src, None)
        raise _syntax_error(src, exc) from None
    try:
        return _Builder(src).transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, (CypherSyntaxError, CypherUnsupportedError)):
            raise exc.orig_exc from None
        raise
