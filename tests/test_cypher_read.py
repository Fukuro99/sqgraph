from __future__ import annotations

import itertools
import random

import pytest

from sqgraph import (
    CypherRuntimeError,
    CypherSyntaxError,
    CypherUnsupportedError,
    GraphDB,
    Node,
    Relationship,
)


def q(db, cypher, **params):
    return db.query(cypher, params)


def col(rows, key):
    return [r[key] for r in rows]


# ------------------------------------------------------------------ basic MATCH / RETURN


def test_match_all_nodes_returns_node_objects(db, social):
    rows = q(db, "MATCH (n) RETURN n")
    assert len(rows) == 6
    assert all(isinstance(r["n"], Node) for r in rows)
    alice = next(r["n"] for r in rows if r["n"].get("name") == "Alice")
    assert alice.labels == ("Employee", "Person")
    assert alice.properties == {"name": "Alice", "age": 30, "active": True}


def test_label_filters(db, social):
    assert q(db, "MATCH (n:Person) RETURN count(*) AS c") == [{"c": 4}]
    assert q(db, "MATCH (n:Company) RETURN count(*) AS c") == [{"c": 2}]
    assert q(db, "MATCH (n:Person:Employee) RETURN n.name AS name") == [{"name": "Alice"}]
    assert q(db, "MATCH (n:Nope) RETURN n") == []


def test_property_map_match_and_params(db, social):
    assert q(db, "MATCH (n:Person {name: 'Bob'}) RETURN n.age AS age") == [{"age": 25}]
    assert q(db, "MATCH (n:Person {name: $name}) RETURN n.age AS age", name="Carol") == [{"age": 35}]
    assert q(db, "MATCH (n {age: 25, name: 'Bob'}) RETURN n.name AS n") == [{"n": "Bob"}]
    assert q(db, "MATCH (n:Person {name: 'Bob', age: 99}) RETURN n") == []
    assert q(db, "MATCH (n:Person $props) RETURN n.name AS n", props={"name": "Dave"}) == [{"n": "Dave"}]


def test_column_naming_follows_neo4j(db, social):
    rows = q(db, "MATCH (n:Person) RETURN n.name, n.age AS years, count(*), n.age + 1")
    assert set(rows[0]) == {"n.name", "years", "count(*)", "n.age + 1"}


def test_duplicate_column_names_rejected(db, social):
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n) RETURN n.name AS x, n.age AS x")


def test_return_star(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[r]->(b:Company) RETURN *")
    assert len(rows) == 1
    assert set(rows[0]) == {"a", "r", "b"}
    assert isinstance(rows[0]["r"], Relationship)
    assert rows[0]["r"].type == "WORKS_AT"


def test_return_star_omits_anonymous(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[:WORKS_AT]->(:Company) RETURN *")
    assert set(rows[0]) == {"a"}


def test_scalar_types_roundtrip(db):
    row = q(db, "RETURN 1 AS i, 1.5 AS f, 'x' AS s, true AS t, false AS u, null AS n, [1,'a',true] AS l, {a: 1, b: [2]} AS m")[0]
    assert row == {"i": 1, "f": 1.5, "s": "x", "t": True, "u": False, "n": None, "l": [1, "a", True], "m": {"a": 1, "b": [2]}}
    assert row["t"] is True and row["u"] is False


def test_boolean_properties_are_bool(db, social):
    rows = q(db, "MATCH (n:Person) WHERE n.active IS NOT NULL RETURN n.name AS name, n.active AS a ORDER BY name")
    assert rows == [{"name": "Alice", "a": True}, {"name": "Bob", "a": False}]
    assert q(db, "MATCH (n:Person) WHERE n.active RETURN n.name AS name") == [{"name": "Alice"}]
    assert q(db, "MATCH (n:Person) WHERE n.active = false RETURN n.name AS name") == [{"name": "Bob"}]


def test_list_property_roundtrip(db, social):
    assert q(db, "MATCH (n {name:'Carol'}) RETURN n.tags AS t") == [{"t": ["a", "b"]}]
    assert q(db, "MATCH (n {name:'Carol'}) RETURN size(n.tags) AS s, n.tags[0] AS h, head(n.tags) AS h2, last(n.tags) AS l") == [
        {"s": 2, "h": "a", "h2": "a", "l": "b"}
    ]


# ------------------------------------------------------------------ WHERE / null semantics / operators


def test_where_comparisons(db, social):
    names = lambda cy, **p: sorted(col(q(db, cy, **p), "n"))  # noqa: E731
    assert names("MATCH (p:Person) WHERE p.age > 28 RETURN p.name AS n") == ["Alice", "Carol", "Dave"]
    assert names("MATCH (p:Person) WHERE p.age >= 35 RETURN p.name AS n") == ["Carol", "Dave"]
    assert names("MATCH (p:Person) WHERE p.age < 30 RETURN p.name AS n") == ["Bob"]
    assert names("MATCH (p:Person) WHERE p.age <= 30 RETURN p.name AS n") == ["Alice", "Bob"]
    assert names("MATCH (p:Person) WHERE p.age <> 30 RETURN p.name AS n") == ["Bob", "Carol", "Dave"]
    assert names("MATCH (p:Person) WHERE p.age > 26 AND p.age < 36 RETURN p.name AS n") == ["Alice", "Carol"]
    assert names("MATCH (p:Person) WHERE p.age < 26 OR p.age > 39 RETURN p.name AS n") == ["Bob", "Dave"]
    assert names("MATCH (p:Person) WHERE NOT p.age > 26 RETURN p.name AS n") == ["Bob"]
    assert names("MATCH (p:Person) WHERE 26 < p.age < 36 RETURN p.name AS n") == ["Alice", "Carol"]
    assert names("MATCH (p:Person) WHERE p.age > $min RETURN p.name AS n", min=34) == ["Carol", "Dave"]


def test_missing_property_is_null_and_filters_out(db, social):
    assert q(db, "MATCH (n:Person) WHERE n.missing = 1 RETURN n") == []
    assert q(db, "MATCH (n:Person) WHERE NOT n.missing = 1 RETURN n") == []
    assert q(db, "MATCH (n:Person) WHERE n.missing <> 1 RETURN n") == []
    assert len(q(db, "MATCH (n:Person) WHERE n.missing IS NULL RETURN n")) == 4
    assert q(db, "MATCH (n:Person) WHERE n.tags IS NOT NULL RETURN n.name AS n") == [{"n": "Carol"}]
    assert q(db, "MATCH (n:Person {name:'Bob'}) RETURN n.tags AS t") == [{"t": None}]


def test_three_valued_logic(db):
    row = q(
        db,
        "RETURN null AND false AS a, null AND true AS b, null OR true AS c, null OR false AS d, "
        "NOT null AS e, null = null AS f, null <> 1 AS g, 1 = null AS h, true XOR false AS i, "
        "true XOR true AS j, null XOR true AS k",
    )[0]
    assert row == {"a": False, "b": None, "c": True, "d": None, "e": None, "f": None, "g": None, "h": None,
                   "i": True, "j": False, "k": None}


def test_is_null_operators(db):
    row = q(db, "RETURN null IS NULL AS a, 1 IS NULL AS b, null IS NOT NULL AS c, 1 IS NOT NULL AS d")[0]
    assert row == {"a": True, "b": False, "c": False, "d": True}


def test_in_operator(db, social):
    assert sorted(col(q(db, "MATCH (n:Person) WHERE n.name IN ['Alice','Bob'] RETURN n.name AS n"), "n")) == ["Alice", "Bob"]
    assert sorted(col(q(db, "MATCH (n:Person) WHERE n.name IN $names RETURN n.name AS n", names=["Dave", "Zed"]), "n")) == ["Dave"]
    assert q(db, "MATCH (n:Person) WHERE n.age IN [] RETURN n") == []
    assert q(db, "RETURN 2 IN [1,2,3] AS a, 5 IN [1,2,3] AS b, 'x' IN ['x'] AS c, 1 IN [null, 2] AS d, 2 IN [null, 2] AS e") == [
        {"a": True, "b": False, "c": True, "d": None, "e": True}
    ]
    assert q(db, "MATCH (n:Person) WHERE 'a' IN n.tags RETURN n.name AS n") == [{"n": "Carol"}]
    ids = [social["alice"], social["bob"]]
    assert sorted(col(q(db, "MATCH (n) WHERE id(n) IN $ids RETURN n.name AS n", ids=ids), "n")) == ["Alice", "Bob"]


def test_string_predicates(db, social):
    f = lambda cy: sorted(col(q(db, cy), "n"))  # noqa: E731
    assert f("MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.name AS n") == ["Alice"]
    assert f("MATCH (n:Person) WHERE n.name ENDS WITH 'e' RETURN n.name AS n") == ["Alice", "Dave"]
    assert f("MATCH (n:Person) WHERE n.name CONTAINS 'o' RETURN n.name AS n") == ["Bob", "Carol"]
    assert f("MATCH (n:Person) WHERE n.name =~ '[AB].*' RETURN n.name AS n") == ["Alice", "Bob"]
    assert f("MATCH (n:Person) WHERE n.name =~ 'lic' RETURN n.name AS n") == []  # full match required
    assert q(db, "RETURN 'abc' STARTS WITH '' AS a, 'abc' ENDS WITH '' AS b, 'abc' ENDS WITH 'abcd' AS c, 'abc' CONTAINS '' AS d") == [
        {"a": True, "b": True, "c": False, "d": True}
    ]
    assert q(db, "RETURN 'a%c' CONTAINS '%' AS a, 'abc' CONTAINS '_' AS b, 'ABC' CONTAINS 'b' AS c") == [
        {"a": True, "b": False, "c": False}
    ]


def test_regex_special_cases(db):
    assert q(db, r"RETURN 'foo123' =~ '[a-z]+\\d+' AS a, 'foo' =~ null AS b") == [{"a": True, "b": None}]


def test_label_predicate(db, social):
    assert sorted(col(q(db, "MATCH (n) WHERE n:Company RETURN n.name AS n"), "n")) == ["Acme", "Globex"]
    assert col(q(db, "MATCH (n) WHERE n:Person:Employee RETURN n.name AS n"), "n") == ["Alice"]
    assert q(db, "MATCH (n:Person) RETURN count(*) AS c") == [{"c": 4}]
    rows = q(db, "MATCH (n) WHERE NOT n:Person RETURN count(*) AS c")
    assert rows == [{"c": 2}]


def test_exists_subquery_and_function(db, social):
    rows = q(db, "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:WORKS_AT]->(:Company {name:'Acme'}) } RETURN p.name AS n")
    assert sorted(col(rows, "n")) == ["Alice", "Bob"]
    rows = q(db, "MATCH (p:Person) WHERE NOT EXISTS { (p)-[:WORKS_AT]->() } RETURN p.name AS n")
    assert col(rows, "n") == ["Dave"]
    rows = q(db, "MATCH (p:Person) WHERE exists(p.tags) RETURN p.name AS n")
    assert col(rows, "n") == ["Carol"]


def test_arithmetic(db):
    row = q(db, "RETURN 1 + 2 AS a, 7 - 10 AS b, 3 * 4 AS c, 7 / 2 AS d, 7.0 / 2 AS e, 7 % 3 AS f, 2 ^ 3 AS g, -5 % 3 AS h, 5.5 % 2 AS i, -(1 + 2) AS j, 2 * 3 + 4 * 5 AS k, 2 + 3 * 4 ^ 2 AS l")[0]
    assert row["a"] == 3 and row["b"] == -3 and row["c"] == 12
    assert row["d"] == 3 and isinstance(row["d"], int)
    assert row["e"] == 3.5
    assert row["f"] == 1
    assert row["g"] == 8.0 and isinstance(row["g"], float)
    assert row["h"] == -2
    assert row["i"] == 1.5
    assert row["j"] == -3
    assert row["k"] == 26
    assert row["l"] == 50.0


def test_arithmetic_with_null_and_properties(db, social):
    assert q(db, "RETURN 1 + null AS a, null * 2 AS b, -null AS c") == [{"a": None, "b": None, "c": None}]
    assert q(db, "MATCH (n:Person {name:'Alice'}) RETURN n.age + 1 AS a, n.age * 2 AS b, n.age - n.age AS z, n.age / 4 AS d, n.missing + 1 AS m") == [
        {"a": 31, "b": 60, "z": 0, "d": 7, "m": None}
    ]


def test_string_concatenation(db, social):
    row = q(db, "MATCH (n:Person {name:'Alice'}) RETURN n.name + '!' AS a, 'x' + 1 AS b, 1 + 'x' AS c, n.name + n.age AS d, 'a' + null AS e")[0]
    assert row == {"a": "Alice!", "b": "x1", "c": "1x", "d": "Alice30", "e": None}


def test_case_expressions(db, social):
    rows = q(
        db,
        "MATCH (p:Person) RETURN p.name AS n, CASE WHEN p.age < 30 THEN 'young' WHEN p.age < 40 THEN 'mid' ELSE 'old' END AS c ORDER BY n",
    )
    assert col(rows, "c") == ["mid", "young", "mid", "old"]
    rows = q(db, "MATCH (p:Person) RETURN p.name AS n, CASE p.name WHEN 'Alice' THEN 1 WHEN 'Bob' THEN 2 END AS c ORDER BY n")
    assert col(rows, "c") == [1, 2, None, None]
    assert q(db, "RETURN CASE WHEN null THEN 1 ELSE 2 END AS x") == [{"x": 2}]
    assert q(db, "RETURN CASE 1 WHEN 1 THEN 'a' ELSE 5 END AS x") == [{"x": "a"}]


# ------------------------------------------------------------------ lists, maps, functions


def test_list_operations(db):
    row = q(
        db,
        "RETURN [1,2,3][0] AS a, [1,2,3][-1] AS b, [1,2,3][5] AS c, [1,2,3,4][1..3] AS d, [1,2,3][..2] AS e, "
        "[1,2,3][1..] AS f, size([1,2,3]) AS g, head([1,2]) AS h, last([1,2]) AS i, tail([1,2,3]) AS j, "
        "reverse([1,2,3]) AS k, [1,2] + [3] AS l, [1] + 2 AS m, range(1,5) AS n, range(0,10,5) AS o, "
        "range(5,1,-2) AS p, range(3,1) AS q, head([]) AS r",
    )[0]
    assert row == {"a": 1, "b": 3, "c": None, "d": [2, 3], "e": [1, 2], "f": [2, 3], "g": 3, "h": 1, "i": 2,
                   "j": [2, 3], "k": [3, 2, 1], "l": [1, 2, 3], "m": [1, 2], "n": [1, 2, 3, 4, 5],
                   "o": [0, 5, 10], "p": [5, 3, 1], "q": [], "r": None}


def test_list_comprehension_and_quantifiers(db):
    row = q(
        db,
        "RETURN [x IN range(1,5) WHERE x % 2 = 1] AS a, [x IN range(1,3) | x * x] AS b, "
        "[x IN range(1,6) WHERE x > 2 | x + 100] AS c, [x IN [] | x] AS d, "
        "all(x IN [2,4] WHERE x % 2 = 0) AS e, any(x IN [1,2] WHERE x > 1) AS f, "
        "none(x IN [1,2] WHERE x > 5) AS g, single(x IN [1,2,3] WHERE x > 2) AS h, "
        "all(x IN [1,2] WHERE x > 1) AS i, any(x IN [] WHERE true) AS j",
    )[0]
    assert row == {"a": [1, 3, 5], "b": [1, 4, 9], "c": [103, 104, 105, 106], "d": [], "e": True, "f": True,
                   "g": True, "h": True, "i": False, "j": False}


def test_list_comprehension_over_nodes(db, social):
    rows = q(db, "MATCH (n:Person) WITH collect(n) AS ps RETURN [p IN ps WHERE p.age > 30 | p.name] AS names")
    assert sorted(rows[0]["names"]) == ["Carol", "Dave"]


def test_map_operations(db, social):
    row = q(db, "WITH {a: 1, b: 'x', c: [1,2]} AS m RETURN m.a AS a, m.b AS b, m.c AS c, m['a'] AS d, m.zzz AS e, keys(m) AS k")[0]
    assert row == {"a": 1, "b": "x", "c": [1, 2], "d": 1, "e": None, "k": ["a", "b", "c"]}
    row = q(db, "MATCH (n:Person {name:'Alice'}) RETURN properties(n) AS p, keys(n) AS k, n['name'] AS nm")[0]
    assert row["p"] == {"name": "Alice", "age": 30, "active": True}
    assert sorted(row["k"]) == ["active", "age", "name"]
    assert row["nm"] == "Alice"
    row = q(db, "RETURN {a: 1}.a AS a, {} AS e, {n: null} AS z")[0]
    assert row == {"a": 1, "e": {}, "z": {"n": None}}


def test_entity_functions(db, social):
    row = q(db, "MATCH (a:Person {name:'Alice'})-[r:KNOWS]->(b) RETURN id(a) AS ia, elementId(a) AS ea, labels(a) AS la, type(r) AS t, id(r) AS ir, startNode(r) AS s, endNode(r) AS e, properties(r) AS pr")[0]
    assert row["ia"] == social["alice"]
    assert row["ea"] == str(social["alice"])
    assert row["la"] == ["Employee", "Person"]
    assert row["t"] == "KNOWS"
    assert row["s"].id == social["alice"] and row["e"].id == social["bob"]
    assert row["pr"] == {"since": 2015}


def test_scalar_functions(db, social):
    row = q(
        db,
        "RETURN coalesce(null, 1) AS a, coalesce(null, null) AS b, toLower('ÄBC') AS c, toUpper('abc') AS d, "
        "trim('  x ') AS e, ltrim('  x ') AS f, rtrim(' x  ') AS g, toString(12) AS h, toString(1.5) AS i, "
        "toString(true) AS j, toInteger('42') AS k, toInteger(3.9) AS l, toFloat('2.5') AS m, toInteger('x') AS n, "
        "abs(-3) AS o, abs(-2.5) AS p, round(2.5) AS r, round(-2.5) AS s, round(1.2345, 2) AS t, size('héllo') AS u, "
        "substring('hello', 1, 3) AS v, substring('hello', 2) AS w, split('a,b,c', ',') AS x, replace('aXbX','X','-') AS y, "
        "left('hello', 2) AS z, right('hello', 3) AS aa, ceil(1.1) AS ab, floor(1.9) AS ac, sqrt(16) AS ad, sign(-5) AS ae, "
        "toBoolean('true') AS af, reverse('abc') AS ag, coalesce(toInteger('z'), 7) AS ah",
    )[0]
    assert row == {"a": 1, "b": None, "c": "äbc", "d": "ABC", "e": "x", "f": "x ", "g": " x", "h": "12", "i": "1.5",
                   "j": "true", "k": 42, "l": 3, "m": 2.5, "n": None, "o": 3, "p": 2.5, "r": 3.0, "s": -3.0,
                   "t": 1.23, "u": 5, "v": "ell", "w": "llo", "x": ["a", "b", "c"], "y": "a-b-", "z": "he",
                   "aa": "llo", "ab": 2.0, "ac": 1.0, "ad": 4.0, "ae": -1, "af": True, "ag": "cba", "ah": 7}


def test_timestamp_function(db):
    ts = q(db, "RETURN timestamp() AS t")[0]["t"]
    assert isinstance(ts, int) and ts > 1_600_000_000_000


def test_coalesce_on_properties(db, social):
    rows = q(db, "MATCH (n:Person) RETURN n.name AS n, coalesce(n.active, 'unknown') AS a ORDER BY n")
    assert col(rows, "a") == [True, False, "unknown", "unknown"]


def test_size_of_property_string_and_list(db, social):
    assert q(db, "MATCH (n:Person {name:'Carol'}) RETURN size(n.name) AS a, size(n.tags) AS b, length(n.name) AS c") == [
        {"a": 5, "b": 2, "c": 5}
    ]


# ------------------------------------------------------------------ aggregation


def test_aggregate_functions(db, social):
    row = q(
        db,
        "MATCH (p:Person) RETURN count(*) AS c, count(p.tags) AS ct, count(DISTINCT p.age) AS cd, sum(p.age) AS s, "
        "avg(p.age) AS a, min(p.age) AS mn, max(p.age) AS mx, min(p.name) AS mnn, max(p.name) AS mxn",
    )[0]
    assert row == {"c": 4, "ct": 1, "cd": 4, "s": 130, "a": 32.5, "mn": 25, "mx": 40, "mnn": "Alice", "mxn": "Dave"}


def test_collect(db, social):
    row = q(db, "MATCH (p:Person) RETURN collect(p.name) AS names, collect(p.tags) AS tags, collect(p) AS nodes")[0]
    assert sorted(row["names"]) == ["Alice", "Bob", "Carol", "Dave"]
    assert row["tags"] == [["a", "b"]]
    assert len(row["nodes"]) == 4 and all(isinstance(n, Node) for n in row["nodes"])


def test_collect_distinct_and_count_distinct(db):
    rows = q(db, "UNWIND [1,1,2,null,3,3] AS x RETURN collect(x) AS a, collect(DISTINCT x) AS b, count(x) AS c, count(DISTINCT x) AS d, count(*) AS e")
    assert rows == [{"a": [1, 1, 2, 3, 3], "b": [1, 2, 3], "c": 5, "d": 3, "e": 6}]


def test_aggregates_on_empty_input(db, social):
    row = q(db, "MATCH (n:Nope) RETURN count(*) AS c, count(n) AS c2, sum(n.x) AS s, avg(n.x) AS a, min(n.x) AS mn, max(n.x) AS mx, collect(n.x) AS l")[0]
    assert row == {"c": 0, "c2": 0, "s": 0, "a": None, "mn": None, "mx": None, "l": []}
    assert q(db, "MATCH (n:Nope) RETURN n.name AS k, count(*) AS c") == []


def test_grouping_uses_non_aggregate_items_as_keys(db, social):
    rows = q(db, "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN c.name AS company, count(*) AS n, collect(p.name) AS people ORDER BY company")
    assert rows[0]["company"] == "Acme" and rows[0]["n"] == 2 and sorted(rows[0]["people"]) == ["Alice", "Bob"]
    assert rows[1] == {"company": "Globex", "n": 1, "people": ["Carol"]}
    rows = q(db, "MATCH (p:Person) RETURN p.age > 28 AS old, count(*) AS n ORDER BY old")
    assert rows == [{"old": False, "n": 1}, {"old": True, "n": 3}]
    rows = q(db, "MATCH (a:Person)-[:KNOWS]->(b) RETURN a, count(b) AS n ORDER BY a.name")
    assert [r["n"] for r in rows] == [1, 1, 1] and all(isinstance(r["a"], Node) for r in rows)


def test_aggregate_in_expression(db, social):
    row = q(db, "MATCH (p:Person) RETURN count(*) + 1 AS a, max(p.age) - min(p.age) AS b, size(collect(p)) AS c, toString(count(*)) AS d")[0]
    assert row == {"a": 5, "b": 15, "c": 4, "d": "4"}


def test_sum_and_avg_types(db):
    row = q(db, "UNWIND [1,2,3] AS x RETURN sum(x) AS s, avg(x) AS a, sum(x * 1.5) AS f")[0]
    assert row["s"] == 6 and row["a"] == 2.0 and row["f"] == 9.0


# ------------------------------------------------------------------ ORDER BY / SKIP / LIMIT / DISTINCT


def test_order_by_variants(db, social):
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY p.age"), "n") == ["Bob", "Alice", "Carol", "Dave"]
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY p.age DESC"), "n") == ["Dave", "Carol", "Alice", "Bob"]
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY n DESC"), "n") == ["Dave", "Carol", "Bob", "Alice"]
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY p.age DESCENDING"), "n")[0] == "Dave"
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY p.age ASCENDING"), "n")[0] == "Bob"


def test_order_by_multiple_keys(db):
    rows = q(db, "UNWIND [{a:1,b:'x'},{a:2,b:'y'},{a:1,b:'z'},{a:2,b:'a'}] AS r RETURN r.a AS a, r.b AS b ORDER BY a DESC, b ASC")
    assert [(r["a"], r["b"]) for r in rows] == [(2, "a"), (2, "y"), (1, "x"), (1, "z")]


def test_order_by_null_placement(db, social):
    asc = col(q(db, "MATCH (p:Person) RETURN p.tags AS t ORDER BY p.tags"), "t")
    assert asc[0] == ["a", "b"] and asc[1:] == [None, None, None]
    desc = col(q(db, "MATCH (p:Person) RETURN p.tags AS t ORDER BY p.tags DESC"), "t")
    assert desc[:3] == [None, None, None] and desc[3] == ["a", "b"]


def test_order_by_expression_not_in_return_and_aggregate(db, social):
    rows = q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY -p.age")
    assert col(rows, "n") == ["Dave", "Carol", "Alice", "Bob"]
    rows = q(db, "MATCH (p:Person)-[:WORKS_AT]->(c) RETURN c.name AS c, count(*) AS n ORDER BY n DESC, c")
    assert rows == [{"c": "Acme", "n": 2}, {"c": "Globex", "n": 1}]


def test_skip_limit(db, social):
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY n SKIP 1 LIMIT 2"), "n") == ["Bob", "Carol"]
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY n LIMIT 1"), "n") == ["Alice"]
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY n SKIP 3"), "n") == ["Dave"]
    assert q(db, "MATCH (p:Person) RETURN p.name AS n LIMIT 0") == []
    assert col(q(db, "MATCH (p:Person) RETURN p.name AS n ORDER BY n SKIP $s LIMIT $l", s=2, l=1), "n") == ["Carol"]
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (p) RETURN p LIMIT -1")
    with pytest.raises(CypherRuntimeError):
        q(db, "MATCH (p) RETURN p LIMIT $l", l=-3)


def test_distinct(db, social):
    assert len(q(db, "MATCH (p:Person)-[:WORKS_AT]->(c) RETURN DISTINCT c.name AS c")) == 2
    assert len(q(db, "MATCH (p:Person)-[:WORKS_AT]->(c) RETURN c.name AS c")) == 3
    assert q(db, "UNWIND [1,1,2] AS x RETURN DISTINCT x ORDER BY x") == [{"x": 1}, {"x": 2}]
    assert len(q(db, "MATCH (a)-[:KNOWS]->(b) RETURN DISTINCT a")) == 3
    assert q(db, "UNWIND [[1],[1],[2]] AS x RETURN DISTINCT x ORDER BY x") == [{"x": [1]}, {"x": [2]}]


# ------------------------------------------------------------------ UNWIND / WITH


def test_unwind_forms(db):
    assert col(q(db, "UNWIND [1,2,3] AS x RETURN x"), "x") == [1, 2, 3]
    assert col(q(db, "UNWIND $xs AS x RETURN x * 2 AS y", xs=[1, 2]), "y") == [2, 4]
    assert q(db, "UNWIND null AS x RETURN x") == []
    assert q(db, "UNWIND [] AS x RETURN x") == []
    assert q(db, "UNWIND range(1,3) AS x RETURN sum(x) AS s") == [{"s": 6}]
    assert q(db, "UNWIND [[1,2],[3]] AS xs UNWIND xs AS x RETURN collect(x) AS all") == [{"all": [1, 2, 3]}]
    rows = q(db, "UNWIND $rows AS row RETURN row.name AS n, row.age AS a, row.ok AS ok", rows=[{"name": "a", "age": 1, "ok": True}, {"name": "b"}])
    assert rows == [{"n": "a", "a": 1, "ok": True}, {"n": "b", "a": None, "ok": None}]
    assert col(q(db, "UNWIND ['a', 'b'] AS s RETURN s"), "s") == ["a", "b"]
    assert col(q(db, "UNWIND [true, false] AS s RETURN s"), "s") == [True, False]


def test_unwind_cross_product_with_match(db, social):
    rows = q(db, "UNWIND ['Alice','Bob','Zed'] AS nm MATCH (p:Person {name: nm}) RETURN p.age AS a ORDER BY a")
    assert col(rows, "a") == [25, 30]


def test_unwind_property_list(db, social):
    rows = q(db, "MATCH (p:Person {name:'Carol'}) UNWIND p.tags AS t RETURN t")
    assert col(rows, "t") == ["a", "b"]
    with pytest.raises(CypherSyntaxError):
        q(db, "UNWIND 5 AS x RETURN x")


def test_with_pipeline(db, social):
    rows = q(db, "MATCH (p:Person) WITH p.age AS age WHERE age > 30 RETURN age ORDER BY age")
    assert col(rows, "age") == [35, 40]
    rows = q(db, "MATCH (p:Person) WITH p ORDER BY p.age DESC LIMIT 2 RETURN p.name AS n ORDER BY n")
    assert col(rows, "n") == ["Carol", "Dave"]
    rows = q(db, "MATCH (p:Person)-[:WORKS_AT]->(c) WITH c, count(*) AS n WHERE n > 1 RETURN c.name AS c, n")
    assert rows == [{"c": "Acme", "n": 2}]
    rows = q(db, "MATCH (p:Person) WITH DISTINCT p.age > 28 AS x RETURN x ORDER BY x")
    assert rows == [{"x": False}, {"x": True}]
    rows = q(db, "MATCH (p:Person) WITH count(*) AS n RETURN n * 10 AS m")
    assert rows == [{"m": 40}]
    rows = q(db, "MATCH (p:Person) WITH p.name AS n, p.age AS a MATCH (q:Person {name: n}) RETURN count(q) AS c")
    assert rows == [{"c": 4}]


def test_with_scoping_hides_old_variables(db, social):
    with pytest.raises(CypherSyntaxError, match="not defined"):
        q(db, "MATCH (p:Person) WITH p.name AS n RETURN p")
    with pytest.raises(CypherSyntaxError, match="aliased"):
        q(db, "MATCH (p:Person) WITH p.name RETURN 1")


def test_with_aggregation_then_collect_ordered(db, social):
    rows = q(db, "MATCH (p:Person) WITH p ORDER BY p.age DESC WITH collect(p.name) AS names RETURN names")
    assert rows == [{"names": ["Dave", "Carol", "Alice", "Bob"]}]


def test_with_star(db, social):
    rows = q(db, "MATCH (p:Person {name:'Bob'}) WITH *, 1 AS one RETURN p.name AS n, one")
    assert rows == [{"n": "Bob", "one": 1}]


# ------------------------------------------------------------------ OPTIONAL MATCH


def test_optional_match_null_rows(db, social):
    rows = q(db, "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name AS n, c.name AS c ORDER BY n")
    assert rows == [
        {"n": "Alice", "c": "Acme"},
        {"n": "Bob", "c": "Acme"},
        {"n": "Carol", "c": "Globex"},
        {"n": "Dave", "c": None},
    ]


def test_optional_match_returns_null_node_and_counts(db, social):
    rows = q(db, "MATCH (p:Person {name:'Dave'}) OPTIONAL MATCH (p)-[r:KNOWS]->(x) RETURN p.name AS n, r, x")
    assert rows == [{"n": "Dave", "r": None, "x": None}]
    rows = q(db, "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(x) RETURN p.name AS n, count(x) AS c ORDER BY n")
    assert col(rows, "c") == [1, 1, 1, 0]
    rows = q(db, "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(x) RETURN p.name AS n, count(*) AS c ORDER BY n")
    assert col(rows, "c") == [1, 1, 1, 1]


def test_optional_match_where_belongs_to_optional_part(db, social):
    rows = q(db, "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c) WHERE c.name = 'Globex' RETURN p.name AS n, c.name AS c ORDER BY n")
    assert rows == [
        {"n": "Alice", "c": None},
        {"n": "Bob", "c": None},
        {"n": "Carol", "c": "Globex"},
        {"n": "Dave", "c": None},
    ]


def test_optional_match_first_clause_and_no_match(db, social):
    assert q(db, "OPTIONAL MATCH (n:Nope) RETURN n") == [{"n": None}]
    rows = q(db, "OPTIONAL MATCH (n:Company) RETURN count(n) AS c")
    assert rows == [{"c": 2}]


def test_chained_optional_matches(db, social):
    rows = q(
        db,
        "MATCH (p:Person {name:'Alice'}) OPTIONAL MATCH (p)-[:KNOWS]->(f) OPTIONAL MATCH (f)-[:WORKS_AT]->(c) RETURN f.name AS f, c.name AS c",
    )
    assert rows == [{"f": "Bob", "c": "Acme"}]


# ------------------------------------------------------------------ relationship patterns


def test_directions(db, social):
    out = q(db, "MATCH (a {name:'Alice'})-[:KNOWS]->(b) RETURN b.name AS n")
    assert col(out, "n") == ["Bob"]
    inn = q(db, "MATCH (a {name:'Alice'})<-[:KNOWS]-(b) RETURN b.name AS n")
    assert col(inn, "n") == ["Carol"]
    both = q(db, "MATCH (a {name:'Alice'})-[:KNOWS]-(b) RETURN b.name AS n ORDER BY n")
    assert col(both, "n") == ["Bob", "Carol"]
    anyrel = q(db, "MATCH (a {name:'Alice'})--(b) RETURN b.name AS n ORDER BY n")
    assert col(anyrel, "n") == ["Acme", "Bob", "Carol"]
    anyrel2 = q(db, "MATCH (a {name:'Alice'})-->(b) RETURN b.name AS n ORDER BY n")
    assert col(anyrel2, "n") == ["Acme", "Bob"]
    anyrel3 = q(db, "MATCH (a {name:'Alice'})<--(b) RETURN b.name AS n")
    assert col(anyrel3, "n") == ["Carol"]


def test_multiple_rel_types_and_rel_properties(db, social):
    rows = q(db, "MATCH (a {name:'Alice'})-[r:KNOWS|WORKS_AT]->(b) RETURN b.name AS n ORDER BY n")
    assert col(rows, "n") == ["Acme", "Bob"]
    rows = q(db, "MATCH (a)-[r:KNOWS {since: 2018}]->(b) RETURN a.name AS a, b.name AS b")
    assert rows == [{"a": "Bob", "b": "Carol"}]
    rows = q(db, "MATCH (a)-[r:KNOWS]->(b) WHERE r.since > 2016 RETURN a.name AS a ORDER BY a")
    assert col(rows, "a") == ["Bob", "Carol"]
    rows = q(db, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")
    assert rows == [{"t": "KNOWS", "c": 3}, {"t": "WORKS_AT", "c": 3}]
    rows = q(db, "MATCH ()-[r:KNOWS|:WORKS_AT]->() RETURN count(*) AS c")
    assert rows == [{"c": 6}]


def test_chained_and_multiple_patterns(db, social):
    rows = q(db, "MATCH (a {name:'Alice'})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.name AS n")
    assert col(rows, "n") == ["Carol"]
    rows = q(db, "MATCH (a {name:'Alice'})-[:KNOWS]->(b)-[:WORKS_AT]->(c:Company) RETURN b.name AS b, c.name AS c")
    assert rows == [{"b": "Bob", "c": "Acme"}]
    rows = q(db, "MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'}) RETURN a.name AS a, b.name AS b")
    assert rows == [{"a": "Alice", "b": "Bob"}]
    rows = q(db, "MATCH (a:Person)-[:WORKS_AT]->(c), (b:Person)-[:WORKS_AT]->(c) WHERE a <> b AND id(a) < id(b) RETURN a.name AS a, b.name AS b")
    assert rows == [{"a": "Alice", "b": "Bob"}]
    assert q(db, "MATCH (a:Company), (b:Company) RETURN count(*) AS c") == [{"c": 4}]


def test_reusing_bound_variables_across_clauses(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'}) MATCH (a)-[:KNOWS]->(b) RETURN b.name AS n")
    assert col(rows, "n") == ["Bob"]
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[:KNOWS]->(b) MATCH (b)-[:KNOWS]->(c) RETURN c.name AS n")
    assert col(rows, "n") == ["Carol"]
    rows = q(db, "MATCH (a:Person)-[:KNOWS]->(b) WITH a, b MATCH (b)-[:WORKS_AT]->(c) RETURN a.name AS a, c.name AS c ORDER BY a")
    assert rows == [{"a": "Alice", "c": "Acme"}, {"a": "Bob", "c": "Globex"}, {"a": "Carol", "c": "Acme"}]
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[r:KNOWS]->(b) MATCH (a)-[r]->(c) RETURN c.name AS n")
    assert col(rows, "n") == ["Bob"]


def test_pattern_closing_a_cycle(db, social):
    rows = q(db, "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) RETURN a.name AS n ORDER BY n")
    assert col(rows, "n") == ["Alice", "Bob", "Carol"]


def test_relationship_uniqueness_within_a_pattern(db):
    a = db.create_node("X", {"n": "a"})
    b = db.create_node("X", {"n": "b"})
    db.create_edge(a.id, b.id, "R")
    # a single relationship may not be traversed twice in one pattern
    assert q(db, "MATCH (x)-[r1:R]-(y)-[r2:R]-(z) RETURN count(*) AS c") == [{"c": 0}]
    assert q(db, "MATCH (x)-[:R]-(y) RETURN count(*) AS c") == [{"c": 2}]
    # ...but two separate MATCH clauses each get their own scope
    assert q(db, "MATCH (x)-[r1:R]-(y) MATCH (y)-[r2:R]-(z) RETURN count(*) AS c") == [{"c": 2}]
    c = db.create_node("X", {"n": "c"})
    db.create_edge(b.id, c.id, "R")
    rows = q(db, "MATCH (x)-[r1:R]-(y)-[r2:R]-(z) RETURN x.n AS x, y.n AS y, z.n AS z ORDER BY x, z")
    assert [(r["x"], r["y"], r["z"]) for r in rows] == [("a", "b", "c"), ("c", "b", "a")]


def test_relationship_uniqueness_across_pattern_parts(db):
    a = db.create_node("X")
    b = db.create_node("X")
    db.create_edge(a.id, b.id, "R")
    assert q(db, "MATCH (x)-[r1:R]->(y), (x)-[r2:R]->(y) RETURN count(*) AS c") == [{"c": 0}]
    db.create_edge(a.id, b.id, "R")
    assert q(db, "MATCH (x)-[r1:R]->(y), (x)-[r2:R]->(y) RETURN count(*) AS c") == [{"c": 2}]


def test_self_loop_and_parallel_edges(db):
    a = db.create_node("X", {"n": "a"})
    db.create_edge(a.id, a.id, "SELF")
    b = db.create_node("X", {"n": "b"})
    db.create_edge(a.id, b.id, "P")
    db.create_edge(a.id, b.id, "P")
    assert q(db, "MATCH (x)-[:SELF]->(y) RETURN count(*) AS c") == [{"c": 1}]
    assert q(db, "MATCH (x)-[:SELF]-(y) RETURN count(*) AS c") == [{"c": 1}]
    assert q(db, "MATCH (x)-[:P]->(y) RETURN count(*) AS c") == [{"c": 2}]
    assert q(db, "MATCH (x {n:'a'})-[r]-(y {n:'b'}) RETURN count(*) AS c") == [{"c": 2}]
    assert q(db, "MATCH (x)-[r]->(x) RETURN count(*) AS c") == [{"c": 1}]


def test_unlabeled_anonymous_patterns(db, social):
    assert q(db, "MATCH ()-[r]->() RETURN count(r) AS c") == [{"c": 6}]
    assert q(db, "MATCH (a)-[]->(b) RETURN count(*) AS c") == [{"c": 6}]
    assert q(db, "MATCH (a)-[]-(b) RETURN count(*) AS c") == [{"c": 12}]
    assert q(db, "MATCH ()-[]->(:Company) RETURN count(*) AS c") == [{"c": 3}]


# ------------------------------------------------------------------ variable-length paths


def test_varlen_cycle_terminates_and_counts(db, social):
    # KNOWS forms the cycle Alice -> Bob -> Carol -> Alice
    def count(pattern):
        return q(db, f"MATCH (a:Person {{name:'Alice'}}){pattern}(b) RETURN count(*) AS c")[0]["c"]

    assert count("-[:KNOWS*1]->") == 1
    assert count("-[:KNOWS*2]->") == 1
    assert count("-[:KNOWS*3]->") == 1
    assert count("-[:KNOWS*4]->") == 0  # would reuse a relationship
    assert count("-[:KNOWS*1..3]->") == 3
    assert count("-[:KNOWS*1..10]->") == 3
    assert count("-[:KNOWS*]->") == 3
    assert count("-[:KNOWS*..2]->") == 2
    assert count("-[:KNOWS*2..]->") == 2
    assert count("-[:KNOWS*0..1]->") == 2
    assert count("-[:KNOWS*0..0]->") == 1


def test_varlen_endpoints_and_named_rel_list(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[r:KNOWS*1..3]->(b) RETURN b.name AS b, size(r) AS n, [x IN r | x.since] AS s ORDER BY n")
    assert rows == [
        {"b": "Bob", "n": 1, "s": [2015]},
        {"b": "Carol", "n": 2, "s": [2015, 2018]},
        {"b": "Alice", "n": 3, "s": [2015, 2018, 2020]},
    ]
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[r*2]->(b) RETURN b.name AS n ORDER BY n")
    assert col(rows, "n") == ["Carol", "Globex"] or col(rows, "n") == ["Acme", "Carol"]


def test_varlen_any_type_and_undirected(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[*1..2]->(b) RETURN b.name AS n ORDER BY n")
    assert col(rows, "n") == ["Acme", "Acme", "Bob", "Carol"]
    rows = q(db, "MATCH (a:Person {name:'Dave'})-[*1..3]-(b) RETURN count(*) AS c")
    assert rows == [{"c": 0}]
    rows = q(db, "MATCH (a:Company {name:'Globex'})-[*1..1]-(b) RETURN b.name AS n")
    assert col(rows, "n") == ["Carol"]


def test_varlen_with_property_filter_and_reverse_anchor(db, social):
    rows = q(db, "MATCH (a)-[:KNOWS*1..2 {since: 2018}]->(b) RETURN a.name AS a, b.name AS b")
    assert rows == [{"a": "Bob", "b": "Carol"}]
    rows = q(db, "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person {name:'Alice'}) RETURN a.name AS a ORDER BY a")
    assert col(rows, "a") == ["Bob", "Carol"]
    rows = q(db, "MATCH (b:Person {name:'Alice'}) MATCH (a)-[:KNOWS*2]->(b) RETURN a.name AS a")
    assert col(rows, "a") == ["Bob"]


def test_varlen_bound_both_ends(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'}), (c:Person {name:'Carol'}) MATCH (a)-[:KNOWS*1..3]->(c) RETURN count(*) AS n")
    assert rows == [{"n": 1}]


def test_named_paths(db, social):
    rows = q(db, "MATCH p = (a:Person {name:'Alice'})-[:KNOWS*2]->(c) RETURN length(p) AS l, [n IN nodes(p) | n.name] AS names, size(relationships(p)) AS r")
    assert rows == [{"l": 2, "names": ["Alice", "Bob", "Carol"], "r": 2}]
    rows = q(db, "MATCH p = (a:Person {name:'Alice'})-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN p")
    p = rows[0]["p"]
    assert [n.get("name") for n in p.nodes] == ["Alice", "Bob", "Acme"]
    assert [r.type for r in p.relationships] == ["KNOWS", "WORKS_AT"]
    assert len(p) == 2


def _brute_force_trails(edges, start, lo, hi, directed):
    """Count trails (no repeated relationship) of length lo..hi from start."""
    count = 0

    def rec(node, used, depth):
        nonlocal count
        if lo <= depth <= hi:
            count += 1
        if depth == hi:
            return
        for eid, s, d in edges:
            if eid in used:
                continue
            if s == node:
                rec(d, used | {eid}, depth + 1)
            elif not directed and d == node:
                rec(s, used | {eid}, depth + 1)

    rec(start, frozenset(), 0)
    return count


@pytest.mark.parametrize("seed", range(6))
def test_varlen_matches_brute_force_on_random_graphs(seed):
    rnd = random.Random(seed)
    with GraphDB() as db:
        ids = [db.create_node("V", {"i": i}).id for i in range(6)]
        edges = []
        for _ in range(11):
            s, d = rnd.choice(ids), rnd.choice(ids)  # self loops and parallel edges allowed
            e = db.create_edge(s, d, "E")
            edges.append((e.id, s, d))
        for directed, arrow in ((True, "->"), (False, "-")):
            for lo, hi in ((1, 3), (2, 4), (0, 2), (1, 5)):
                got = db.query(f"MATCH (a:V)-[:E*{lo}..{hi}]{arrow}(b) RETURN a.i AS a, count(*) AS c")
                by_start = {r["a"]: r["c"] for r in got}
                for i, nid in enumerate(ids):
                    expect = _brute_force_trails(edges, nid, lo, hi, directed)
                    assert by_start.get(i, 0) == expect, (seed, directed, lo, hi, i)


def test_default_max_var_length_is_configurable():
    with GraphDB(max_var_length=2) as db:
        ids = [db.create_node("V").id for _ in range(6)]
        for a, b in zip(ids, ids[1:]):
            db.create_edge(a, b, "E")
        assert db.query("MATCH (a:V)-[*]->(b) WHERE id(a) = $i RETURN count(*) AS c", {"i": ids[0]}) == [{"c": 2}]
    with GraphDB() as db:
        ids = [db.create_node("V").id for _ in range(15)]
        for a, b in zip(ids, ids[1:]):
            db.create_edge(a, b, "E")
        assert db.query("MATCH (a:V)-[*]->(b) WHERE id(a) = $i RETURN count(*) AS c", {"i": ids[0]}) == [{"c": 10}]
        assert db.query("MATCH (a:V)-[*1..12]->(b) WHERE id(a) = $i RETURN count(*) AS c", {"i": ids[0]}) == [{"c": 12}]


# ------------------------------------------------------------------ parameters, quoting, injection


def test_parameters_of_many_types(db):
    row = q(db, "RETURN $i AS i, $f AS f, $s AS s, $b AS b, $n AS n, $l AS l, $m AS m", i=1, f=2.5, s="x", b=True, n=None, l=[1, 2], m={"k": [1]})[0]
    assert row == {"i": 1, "f": 2.5, "s": "x", "b": True, "n": None, "l": [1, 2], "m": {"k": [1]}}


def test_node_parameter(db, social):
    alice = db.get_node(social["alice"])
    rows = q(db, "MATCH (n) WHERE n = $node RETURN n.name AS n", node=alice)
    assert rows == [{"n": "Alice"}]


def test_missing_parameter_raises(db):
    with pytest.raises(CypherRuntimeError, match="Missing parameter"):
        q(db, "RETURN $nope")


def test_backtick_identifiers(db):
    db.create_node(["My Label"], {"odd key": 1, "a.b": 2, 'q"uote': 3, "back`tick": 4})
    row = q(db, "MATCH (n:`My Label`) RETURN n.`odd key` AS a, n.`a.b` AS b, n.`q\"uote` AS c, n.`back``tick` AS d")[0]
    assert row == {"a": 1, "b": 2, "c": 3, "d": 4}
    assert q(db, "MATCH (n:`My Label` {`odd key`: 1}) RETURN count(*) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n) RETURN n.`odd key` AS `my alias`") == [{"my alias": 1}]
    assert q(db, "WITH 5 AS `x y` RETURN `x y` + 1 AS r") == [{"r": 6}]


def test_keywords_usable_as_names(db):
    db.create_node(["Order"], {"end": 1, "type": "t", "match": 2, "order": 3})
    row = q(db, "MATCH (n:Order) RETURN n.end AS e, n.type AS t, n.match AS m, n.order AS o")[0]
    assert row == {"e": 1, "t": "t", "m": 2, "o": 3}
    assert q(db, "MATCH (n:Order) WITH n.end AS end RETURN end") == [{"end": 1}]


def test_injection_via_parameters_is_inert(db, social):
    evil = "x' OR 1=1 --"
    assert q(db, "MATCH (n:Person {name: $n}) RETURN n", n=evil) == []
    assert q(db, "MATCH (n:Person) WHERE n.name = $n RETURN n", n="'; DROP TABLE nodes; --") == []
    assert q(db, "MATCH (n:Person) WHERE n.name IN $l RETURN n", l=[evil, "\"); DELETE FROM nodes; --"]) == []
    assert q(db, "RETURN $s AS s", s="'; DROP TABLE nodes; --") == [{"s": "'; DROP TABLE nodes; --"}]
    assert q(db, "MATCH (n) RETURN count(*) AS c")[0]["c"] == 6
    assert q(db, "MATCH (n:Person) RETURN n.name AS n LIMIT $l", l=1) != []


def test_injection_via_stored_property_values_is_inert(db):
    evil = "'); DROP TABLE nodes; --"
    db.create_node("P", {"name": evil, "v": '" OR 1=1'})
    rows = q(db, "MATCH (n:P) WHERE n.name = $x RETURN n.name AS n, n.v AS v", x=evil)
    assert rows == [{"n": evil, "v": '" OR 1=1'}]
    rows = q(db, "MATCH (n:P) WITH n.name AS a UNWIND [a] AS b RETURN b")
    assert rows == [{"b": evil}]
    assert db.stats()["nodes"] == 1


def test_injection_via_labels_and_keys_stays_inert(db):
    weird = "x'; DROP TABLE nodes; --"
    db.create_node([weird], {weird: 1, "k": 2})
    rows = q(db, "MATCH (n:`x'; DROP TABLE nodes; --`) RETURN n.`x'; DROP TABLE nodes; --` AS v, n.k AS k")
    assert rows == [{"v": 1, "k": 2}]
    assert q(db, "MATCH (n) WHERE n.`x'; DROP TABLE nodes; --` = 1 RETURN count(*) AS c") == [{"c": 1}]
    # passing a hostile string as a dynamic property key / relationship type also stays a value
    assert q(db, "MATCH (n) RETURN n[$k] AS v", k=weird) == [{"v": 1}]
    assert db.stats()["nodes"] == 1


def test_string_escapes_in_literals(db):
    cy = (
        r"RETURN 'it\'s' AS a, 'tab\there' AS c, 'nl\nx' AS d, 'unié' AS e, "
        r"'back\\slash' AS f, " + '"dq\\"x" AS g'
    )
    row = q(db, cy)[0]
    assert row["a"] == "it's" and row["c"] == "tab\there" and row["d"] == "nl\nx" and row["e"] == "unié"
    assert row["f"] == "back\\slash" and row["g"] == 'dq"x'
    with pytest.raises(CypherSyntaxError):
        q(db, r"RETURN '\q'")


def test_comments_and_semicolon(db, social):
    rows = q(db, "// leading\nMATCH (n:Company) /* inline */ RETURN count(*) AS c; ")
    assert rows == [{"c": 2}]


# ------------------------------------------------------------------ UNION


def test_union_and_union_all(db, social):
    rows = q(db, "MATCH (n:Company) RETURN n.name AS name UNION MATCH (m:Person {name:'Alice'}) RETURN m.name AS name")
    assert sorted(col(rows, "name")) == ["Acme", "Alice", "Globex"]
    rows = q(db, "RETURN 1 AS x UNION ALL RETURN 1 AS x")
    assert col(rows, "x") == [1, 1]
    rows = q(db, "RETURN 1 AS x UNION RETURN 1 AS x")
    assert col(rows, "x") == [1]
    with pytest.raises(CypherSyntaxError):
        q(db, "RETURN 1 AS x UNION RETURN 1 AS y")


# ------------------------------------------------------------------ errors


def test_undefined_variable(db):
    with pytest.raises(CypherSyntaxError, match="Variable `x` not defined"):
        q(db, "MATCH (n) RETURN x")
    with pytest.raises(CypherSyntaxError, match="not defined"):
        q(db, "MATCH (n) WHERE m.x = 1 RETURN n")


def test_syntax_errors_have_positions(db):
    with pytest.raises(CypherSyntaxError) as ei:
        q(db, "MATCH (n RETURN n")
    assert ei.value.line == 1 and ei.value.column == 10
    with pytest.raises(CypherSyntaxError) as ei:
        q(db, "MATCH (n)\nRETRUN n")
    assert ei.value.line == 2
    with pytest.raises(CypherSyntaxError):
        q(db, "")
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n)")
    with pytest.raises(CypherSyntaxError):
        q(db, "RETURN 1 RETURN 2")


def test_unsupported_features_raise_clearly(db):
    for cy in (
        "MATCH p = shortestPath((a)-[*]-(b)) RETURN p",
        "FOREACH (x IN [1] | CREATE (:N))",
        "CALL { MATCH (n) RETURN n } RETURN 1",
        "LOAD CSV FROM 'x' AS row RETURN row",
        "RETURN apoc.text.join(['a'], ',')",
        "RETURN nosuchfunction(1)",
        "CALL apoc.help('x') YIELD name RETURN name",
    ):
        with pytest.raises((CypherUnsupportedError, CypherSyntaxError)):
            q(db, cy)
    with pytest.raises(CypherUnsupportedError):
        q(db, "RETURN nosuchfunction(1)")
    with pytest.raises(CypherUnsupportedError):
        q(db, "MATCH p = shortestPath((a)-[*]-(b)) RETURN p")


def test_pattern_predicate_gives_hint(db, social):
    with pytest.raises(CypherSyntaxError, match="EXISTS"):
        q(db, "MATCH (a), (b) WHERE (a)-[:KNOWS]->(b) RETURN a")


def test_type_errors_are_reported(db, social):
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n) RETURN n.name.first.x + true")
    with pytest.raises(CypherSyntaxError):
        q(db, "RETURN 1.foo")
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n) RETURN labels(1)")
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n) WHERE count(*) > 1 RETURN n")


def test_write_query_needs_return_or_update(db):
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n) WITH n")


def test_explain_shows_sql(db, social):
    sql = db.explain("MATCH (a:Person)-[:KNOWS*1..3]->(b) WHERE a.age > $x RETURN b.name, count(*)", {"x": 1})
    assert "WITH RECURSIVE" in sql and "json_extract" in sql
    assert "SELECT" in db.explain("UNWIND [1,2] AS x RETURN x")
    sql = db.explain("MATCH (n) SET n.x = 1 RETURN n")
    assert "write step" in sql


def test_reads_are_pure_sql_no_python_row_processing(db, social):
    sql = db.explain("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS n, collect(p.age) AS ages ORDER BY n LIMIT 2")
    for token in ("json_group_array", "ORDER BY", "LIMIT", "WHERE"):
        assert token in sql
    sql = db.explain("UNWIND $xs AS x RETURN x", {"xs": [1]})
    assert "json_each" in sql


def test_large_graph_smoke(db):
    ids = db.create_nodes([("N", {"i": i}) for i in range(500)])
    db.create_edges([(ids[i], ids[(i + 1) % 500], "NEXT") for i in range(500)])
    assert db.query("MATCH (n:N)-[:NEXT]->(m:N) RETURN count(*) AS c") == [{"c": 500}]
    rows = db.query("MATCH (a:N {i: 0})-[:NEXT*3]->(b) RETURN b.i AS i")
    assert rows == [{"i": 3}]
    rows = db.query("MATCH (n:N) WHERE n.i % 100 = 0 RETURN n.i AS i ORDER BY i")
    assert [r["i"] for r in rows] == [0, 100, 200, 300, 400]


def test_integer_division_by_zero_is_an_error(db):
    with pytest.raises(CypherRuntimeError, match="by zero"):
        q(db, "RETURN 1 / 0 AS x")
    with pytest.raises(CypherRuntimeError, match="by zero"):
        q(db, "RETURN 5 % 0 AS x")
    assert q(db, "RETURN 1.0 / 0 AS x") == [{"x": None}]  # float division: no exception (NULL, not Infinity)


def test_invalid_regular_expression_is_reported(db):
    with pytest.raises(CypherRuntimeError, match="regular expression"):
        q(db, "RETURN 'a' =~ '(' AS x")


def test_implicit_grouping_expressions_are_rejected(db, social):
    with pytest.raises(CypherSyntaxError, match="implicit grouping"):
        q(db, "MATCH (n:Person) RETURN n.name, n.age + count(*) AS x")
    with pytest.raises(CypherSyntaxError, match="implicit grouping"):
        q(db, "MATCH (n:Person) RETURN count(*) + n.age AS x")
    # allowed: variables used inside aggregates, or repeated as grouping keys
    assert q(db, "MATCH (n:Person) RETURN count(*) + 1 AS x") == [{"x": 5}]
    rows = q(db, "MATCH (n:Person) RETURN n.age AS age, n.age + count(*) AS x ORDER BY age LIMIT 1")
    assert rows == [{"age": 25, "x": 26}]
    assert q(db, "MATCH (n:Person) RETURN size(collect(n.name)) AS s") == [{"s": 4}]


def test_aggregation_cannot_be_used_in_where(db, social):
    with pytest.raises(CypherSyntaxError, match="aggregating"):
        q(db, "MATCH (n:Person) WHERE count(*) > 1 RETURN n")
