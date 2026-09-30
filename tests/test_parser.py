from __future__ import annotations

import pytest

from sqgraph import CypherSyntaxError, CypherUnsupportedError
from sqgraph.cypher import ast as A
from sqgraph.cypher.parser import parse, unescape_string


def single(cy):
    return parse(cy).parts[0].clauses


def test_match_pattern_structure():
    (m, r) = single("MATCH (a:P:Q {name: 'x'})<-[r:K|L*1..3 {w: 1}]-(b) WHERE a.x > 1 RETURN a")
    assert isinstance(m, A.Match) and not m.optional
    node_a, rel, node_b = m.patterns[0].elements
    assert (node_a.var, node_a.labels) == ("a", ["P", "Q"])
    assert isinstance(node_a.props, A.MapLiteral)
    assert (rel.var, rel.types, rel.direction, rel.varlen, rel.min_hops, rel.max_hops) == ("r", ["K", "L"], "in", True, 1, 3)
    assert node_b.var == "b" and node_b.labels == []
    assert isinstance(m.where, A.BinaryOp) and m.where.op == ">"


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("-[*]->", (True, 1, None)),
        ("-[*2]->", (True, 2, 2)),
        ("-[*1..3]->", (True, 1, 3)),
        ("-[*..3]->", (True, 1, 3)),
        ("-[*2..]->", (True, 2, None)),
        ("-[*0..]->", (True, 0, None)),
        ("-[:T*..3]->", (True, 1, 3)),
        ("-[:T]->", (False, 1, 1)),
        ("-->", (False, 1, 1)),
    ],
)
def test_relationship_ranges(pattern, expected):
    rel = single(f"MATCH (a){pattern}(b) RETURN a")[0].patterns[0].elements[1]
    assert (rel.varlen, rel.min_hops, rel.max_hops) == expected


def test_directions_shorthand():
    dirs = []
    for arrow in ("-->", "<--", "--", "-[]->", "<-[]-", "-[]-"):
        rel = single(f"MATCH (a){arrow}(b) RETURN a")[0].patterns[0].elements[1]
        dirs.append(rel.direction)
    assert dirs == ["out", "in", "both", "out", "in", "both"]


def test_invalid_range_rejected():
    with pytest.raises(CypherSyntaxError):
        parse("MATCH (a)-[*3..1]->(b) RETURN a")


def test_operator_precedence():
    e = single("RETURN 1 + 2 * 3 ^ 2 AS x")[0].projection.items[0].expr
    assert (e.op, e.right.op, e.right.right.op) == ("+", "*", "^")
    e = single("RETURN NOT a AND b OR c XOR d AS x")[0].projection.items[0].expr
    assert e.op == "OR"
    e = single("RETURN a OR b AND c AS x")[0].projection.items[0].expr
    assert e.op == "OR" and e.right.op == "AND"
    e = single("RETURN -2 ^ 2 AS x")[0].projection.items[0].expr
    assert isinstance(e, A.UnaryOp) and e.op == "NEG" and e.operand.op == "^"  # unary minus applies to the power


def test_chained_comparison_becomes_conjunction():
    e = single("RETURN 1 < x <= 5 AS r")[0].projection.items[0].expr
    assert e.op == "AND" and e.left.op == "<" and e.right.op == "<="


def test_literals_and_unescape():
    items = single("RETURN 1, 0x10, 1.5, 2e3, 'a', \"b\", true, FALSE, null, [1, 2], {a: 1}")[0].projection.items
    values = [getattr(i.expr, "value", "n/a") for i in items]
    assert values[:9] == [1, 16, 1.5, 2000.0, "a", "b", True, False, None]
    assert isinstance(items[9].expr, A.ListLiteral) and isinstance(items[10].expr, A.MapLiteral)
    assert unescape_string(r"a\nb\t\\\'") == "a\nb\t\\'"
    assert unescape_string(r"é") == "é"
    with pytest.raises(ValueError):
        unescape_string(r"\q")


def test_case_insensitive_keywords():
    parse("match (n:P) where n.x = 1 return n order by n.x desc skip 1 limit 2")
    parse("Optional Match (n) Return Distinct n")
    parse("merge (n:P {a: 1}) on create set n.b = 1 on match set n.c = 2")
    parse("unwind [1] as x with x where x > 0 return x")


def test_keywords_as_identifiers():
    parse("MATCH (order:Order) RETURN order.match AS end, order.where, order.and2")
    parse("MATCH (n) RETURN n.count AS count, n.type AS type, n.in AS `in`")
    with pytest.raises(CypherSyntaxError):
        parse("MATCH (n) RETURN n.a AS")


def test_projection_text_preserved_for_column_names():
    items = single("RETURN  n.name,  count( * ) ,n . age AS  a")[0].projection.items
    assert [i.text for i in items] == ["n.name", "count( * )", "n . age"]
    assert items[2].alias == "a"


def test_union_parsing():
    q = parse("RETURN 1 AS x UNION ALL RETURN 2 AS x UNION ALL RETURN 3 AS x")
    assert len(q.parts) == 3 and q.union_all
    q = parse("RETURN 1 AS x UNION RETURN 2 AS x")
    assert len(q.parts) == 2 and not q.union_all
    with pytest.raises(CypherSyntaxError, match="mix"):
        parse("RETURN 1 AS x UNION RETURN 2 AS x UNION ALL RETURN 3 AS x")


def test_schema_statements():
    v = parse(
        "CREATE VECTOR INDEX idx IF NOT EXISTS FOR (n:L) ON (n.emb) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 4, `vector.similarity_function`: 'cosine'}}"
    )
    assert isinstance(v, A.CreateVectorIndex) and v.name == "idx" and v.if_not_exists
    assert (v.label, v.property) == ("L", "emb")
    f = parse("CREATE FULLTEXT INDEX ft FOR (n:A|B|C) ON EACH [n.x, n.y]")
    assert isinstance(f, A.CreateFulltextIndex) and f.labels == ["A", "B", "C"] and f.properties == ["x", "y"]
    p = parse("CREATE INDEX FOR (n:P) ON (n.name)")
    assert isinstance(p, A.CreatePropertyIndex) and p.properties == ["name"] and p.name is None
    assert isinstance(parse("DROP INDEX x IF EXISTS"), A.DropIndex)
    assert isinstance(parse("SHOW INDEXES"), A.ShowIndexes)


def test_call_and_yield():
    (c,) = single("CALL db.index.vector.queryNodes('i', 5, $q) YIELD node AS n, score WHERE score > 0.5")
    assert c.name == "db.index.vector.queryNodes" and len(c.args) == 3
    assert [(y.name, y.alias) for y in c.yields] == [("node", "n"), ("score", None)]
    assert c.where is not None
    (c,) = single("CALL db.labels()")
    assert c.yields is None and c.args == []


def test_set_items():
    (s,) = single("MATCH (n) SET n.a = 1, n += {b: 2}, n = {c: 3}, n:X:Y")[1:]
    assert [i.kind for i in s.items] == ["prop", "merge", "replace", "labels"]
    assert s.items[3].labels == ["X", "Y"]


def test_error_positions_and_messages():
    with pytest.raises(CypherSyntaxError) as ei:
        parse("MATCH (n)\nWHERE n.x = = 1\nRETURN n")
    assert ei.value.line == 2 and ei.value.column == 13
    assert "Unexpected" in str(ei.value)
    with pytest.raises(CypherSyntaxError) as ei:
        parse("RETURN 'unterminated")
    assert ei.value.line == 1


@pytest.mark.parametrize(
    "cy",
    [
        "MATCH p = shortestPath((a)-[*]-(b)) RETURN p",
        "MATCH p = allShortestPaths((a)-[*]-(b)) RETURN p",
        "FOREACH (x IN [1] | CREATE (:N))",
        "CALL { RETURN 1 } RETURN 1",
        "RETURN reduce(acc = 0, x IN [1, 2] | acc + x) AS s",
        "MATCH (a) RETURN [(a)-->(b) | b.name] AS x",
        "LOAD CSV FROM 'f' AS r RETURN r",
        "CREATE CONSTRAINT c FOR (n:P) REQUIRE n.x IS UNIQUE",
    ],
)
def test_unsupported_syntax_is_reported_as_unsupported(cy):
    with pytest.raises(CypherUnsupportedError):
        parse(cy)


def test_parse_is_cached():
    a = parse("MATCH (n) RETURN n")
    assert parse("MATCH (n) RETURN n") is a
