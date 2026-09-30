from __future__ import annotations

import pytest

from sqgraph import (
    CypherRuntimeError,
    CypherSyntaxError,
    CypherUnsupportedError,
    GraphDB,
    Node,
    Relationship,
    SqgraphError,
)


def q(db, cypher, **params):
    return db.query(cypher, params)


def count(db, cypher="MATCH (n) RETURN count(*) AS c"):
    return db.query(cypher)[0]["c"]


# ------------------------------------------------------------------ CREATE


def test_create_node_with_labels_and_props(db):
    rows = q(db, "CREATE (n:Person:Employee {name: 'Alice', age: 30, tags: ['a','b'], ok: true}) RETURN n")
    n = rows[0]["n"]
    assert isinstance(n, Node)
    assert n.labels == ("Employee", "Person")
    assert n.properties == {"name": "Alice", "age": 30, "tags": ["a", "b"], "ok": True}
    assert db.get_node(n.id) == n
    assert db.get_node(n.id).properties == n.properties


def test_create_with_parameters_and_expressions(db):
    q(db, "CREATE (n:P {name: $name, double: $x * 2, s: 'a' + $name, l: [1, 2, $x]})", name="Zed", x=21)
    n = db.query("MATCH (n:P) RETURN n")[0]["n"]
    assert n.properties == {"name": "Zed", "double": 42, "s": "aZed", "l": [1, 2, 21]}
    q(db, "CREATE (n:Q $props)", props={"a": 1, "b": [1, 2], "c": None})
    assert db.query("MATCH (n:Q) RETURN n")[0]["n"].properties == {"a": 1, "b": [1, 2]}


def test_create_skips_null_properties(db):
    q(db, "CREATE (n:P {a: 1, b: null})")
    assert db.query("MATCH (n:P) RETURN n")[0]["n"].properties == {"a": 1}


def test_create_relationships(db):
    rows = q(db, "CREATE (a:Person {name:'A'})-[r:KNOWS {since: 2020}]->(b:Person {name:'B'}) RETURN a, r, b")
    r = rows[0]["r"]
    assert isinstance(r, Relationship)
    assert r.type == "KNOWS" and r.properties == {"since": 2020}
    assert r.start_node_id == rows[0]["a"].id and r.end_node_id == rows[0]["b"].id
    q(db, "CREATE (x:Person {name:'X'})<-[:LIKES]-(y:Person {name:'Y'})")
    rows = q(db, "MATCH (y)-[:LIKES]->(x) RETURN y.name AS y, x.name AS x")
    assert rows == [{"y": "Y", "x": "X"}]


def test_create_chain_and_multiple_patterns(db):
    q(db, "CREATE (a:N {i:1})-[:R]->(b:N {i:2})-[:R]->(c:N {i:3}), (d:M)")
    assert count(db, "MATCH (n:N) RETURN count(*) AS c") == 3
    assert count(db, "MATCH ()-[r:R]->() RETURN count(*) AS c") == 2
    assert count(db, "MATCH (n:M) RETURN count(*) AS c") == 1
    assert q(db, "MATCH (a)-[:R]->(b)-[:R]->(c) RETURN a.i AS a, c.i AS c") == [{"a": 1, "c": 3}]


def test_create_between_matched_nodes(db):
    q(db, "CREATE (:P {n:'a'}), (:P {n:'b'})")
    q(db, "MATCH (a:P {n:'a'}), (b:P {n:'b'}) CREATE (a)-[:R {w: 1}]->(b)")
    rows = q(db, "MATCH (a)-[r:R]->(b) RETURN a.n AS a, b.n AS b, r.w AS w")
    assert rows == [{"a": "a", "b": "b", "w": 1}]
    assert count(db) == 2


def test_create_new_node_variable_reuse_across_clauses(db):
    q(db, "CREATE (a:A) CREATE (b:B) CREATE (a)-[:R]->(b)")
    assert q(db, "MATCH (a:A)-[:R]->(b:B) RETURN count(*) AS c") == [{"c": 1}]
    rows = q(db, "CREATE (a:A {x: 1}) WITH a CREATE (a)-[:R]->(b:B {y: a.x + 1}) RETURN b.y AS y")
    assert rows == [{"y": 2}]


def test_create_per_input_row(db):
    q(db, "UNWIND range(1, 5) AS i CREATE (:Num {i: i, sq: i * i})")
    rows = q(db, "MATCH (n:Num) RETURN n.i AS i, n.sq AS sq ORDER BY i")
    assert [(r["i"], r["sq"]) for r in rows] == [(1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]
    q(db, "UNWIND $rows AS row CREATE (:P {name: row.name, age: row.age})", rows=[{"name": "a", "age": 1}, {"name": "b"}])
    assert q(db, "MATCH (p:P) RETURN p.name AS n, p.age AS a ORDER BY n") == [{"n": "a", "a": 1}, {"n": "b", "a": None}]
    q(db, "MATCH (n:Num) CREATE (n)-[:HAS]->(:Tag {of: n.i})")
    assert count(db, "MATCH (:Num)-[:HAS]->(t:Tag) RETURN count(*) AS c") == 5


def test_create_returns_value_and_can_continue(db):
    rows = q(db, "CREATE (n:X {a: 1}) WITH n MATCH (m:X) RETURN count(m) AS c, n.a AS a")
    assert rows == [{"c": 1, "a": 1}]
    assert q(db, "CREATE (n:X) RETURN id(n) AS i")[0]["i"] > 0
    assert q(db, "CREATE (n:X)") == []


def test_create_errors(db):
    with pytest.raises(CypherRuntimeError, match="directed"):
        q(db, "CREATE (a)-[:R]-(b)")
    with pytest.raises(CypherRuntimeError):
        q(db, "CREATE (a)-[:R|S]->(b)")
    with pytest.raises(CypherRuntimeError):
        q(db, "CREATE (a)-[]->(b)")
    with pytest.raises(CypherRuntimeError):
        q(db, "CREATE (a)-[:R*2]->(b)")
    with pytest.raises(CypherUnsupportedError):
        q(db, "CREATE p = (a)-[:R]->(b)")
    q(db, "CREATE (a:A)")
    with pytest.raises(CypherRuntimeError, match="already declared"):
        q(db, "MATCH (a:A) CREATE (a:B)")
    with pytest.raises(CypherRuntimeError, match="null"):
        q(db, "OPTIONAL MATCH (a:Nope) CREATE (a)-[:R]->(:X)")
    assert count(db) == 1


def test_create_property_type_errors(db):
    with pytest.raises(CypherRuntimeError, match="primitive"):
        q(db, "CREATE (n:X {m: {a: 1}})")
    with pytest.raises(CypherRuntimeError, match="primitive"):
        q(db, "CREATE (n:X {m: [[1], [2]]})")
    assert count(db) == 0


# ------------------------------------------------------------------ MERGE


def test_merge_node_is_idempotent(db):
    for _ in range(3):
        q(db, "MERGE (n:Person {name: 'Alice'})")
    assert count(db, "MATCH (n:Person) RETURN count(*) AS c") == 1
    rows = q(db, "MERGE (n:Person {name: 'Alice'}) RETURN n.name AS n")
    assert rows == [{"n": "Alice"}]
    q(db, "MERGE (n:Person {name: 'Bob'})")
    assert count(db, "MATCH (n:Person) RETURN count(*) AS c") == 2


def test_merge_on_create_and_on_match(db):
    q(db, "MERGE (n:P {id: 1}) ON CREATE SET n.created = true, n.hits = 0 ON MATCH SET n.hits = n.hits + 1")
    n = db.query("MATCH (n:P) RETURN n")[0]["n"]
    assert n.properties == {"id": 1, "created": True, "hits": 0}
    q(db, "MERGE (n:P {id: 1}) ON CREATE SET n.created = true, n.hits = 0 ON MATCH SET n.hits = n.hits + 1, n.seen = 'yes'")
    q(db, "MERGE (n:P {id: 1}) ON MATCH SET n.hits = n.hits + 1")
    n = db.query("MATCH (n:P) RETURN n")[0]["n"]
    assert n.properties == {"id": 1, "created": True, "hits": 2, "seen": "yes"}
    assert count(db) == 1


def test_merge_with_unwind_sees_earlier_rows(db):
    q(db, "UNWIND [1, 2, 1, 3, 2, 1] AS x MERGE (n:T {v: x}) ON CREATE SET n.c = 1 ON MATCH SET n.c = n.c + 1")
    rows = q(db, "MATCH (n:T) RETURN n.v AS v, n.c AS c ORDER BY v")
    assert rows == [{"v": 1, "c": 3}, {"v": 2, "c": 2}, {"v": 3, "c": 1}]


def test_merge_relationship_between_bound_nodes(db):
    q(db, "CREATE (:P {n:'a'}), (:P {n:'b'})")
    for _ in range(3):
        q(db, "MATCH (a:P {n:'a'}), (b:P {n:'b'}) MERGE (a)-[r:KNOWS {w: 1}]->(b)")
    assert count(db, "MATCH ()-[r:KNOWS]->() RETURN count(*) AS c") == 1
    q(db, "MATCH (a:P {n:'a'}), (b:P {n:'b'}) MERGE (a)-[r:KNOWS {w: 2}]->(b)")
    assert count(db, "MATCH ()-[r:KNOWS]->() RETURN count(*) AS c") == 2
    q(db, "MATCH (a:P {n:'a'}), (b:P {n:'b'}) MERGE (b)-[r:KNOWS]->(a)")
    assert count(db, "MATCH ()-[r:KNOWS]->() RETURN count(*) AS c") == 3
    # undirected MERGE matches either direction and does not create a duplicate
    q(db, "MATCH (a:P {n:'a'}), (b:P {n:'b'}) MERGE (a)-[r:KNOWS {w: 2}]-(b)")
    assert count(db, "MATCH ()-[r:KNOWS]->() RETURN count(*) AS c") == 3


def test_merge_creates_whole_missing_pattern(db):
    q(db, "MERGE (a:P {n:'a'})-[:R]->(b:P {n:'b'})")
    assert count(db) == 2
    q(db, "MERGE (a:P {n:'a'})-[:R]->(b:P {n:'b'})")
    assert count(db) == 2
    # when only part exists, MERGE creates the whole pattern anew (Neo4j semantics)
    q(db, "MERGE (a:P {n:'a'})-[:S]->(b:P {n:'c'})")
    assert count(db, "MATCH (n:P) RETURN count(*) AS c") == 4


def test_merge_returns_bound_values_and_multiple_matches(db):
    q(db, "CREATE (:D {k: 1}), (:D {k: 1})")
    rows = q(db, "MERGE (n:D {k: 1}) RETURN n")
    assert len(rows) == 2  # every match is returned
    rows = q(db, "MERGE (n:D {k: 9}) ON CREATE SET n.new = true RETURN n.new AS new")
    assert rows == [{"new": True}]


def test_merge_null_property_is_an_error(db):
    with pytest.raises(CypherRuntimeError, match="null"):
        q(db, "MERGE (n:P {name: null})")
    with pytest.raises(CypherRuntimeError, match="null"):
        q(db, "MERGE (n:P {name: $x})", x=None)
    assert count(db) == 0


def test_merge_on_match_set_labels_and_maps(db):
    q(db, "CREATE (:P {id: 1})")
    q(db, "MERGE (n:P {id: 1}) ON MATCH SET n:Seen, n += {extra: 5}")
    n = db.query("MATCH (n:P) RETURN n")[0]["n"]
    assert n.labels == ("P", "Seen") and n["extra"] == 5


# ------------------------------------------------------------------ SET / REMOVE


@pytest.fixture
def people(db):
    q(db, "CREATE (:P {name:'a', v:1}), (:P {name:'b', v:2}), (:P {name:'c'})")
    return db


def test_set_property(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) SET n.v = 10, n.w = 'x'")
    assert q(db, "MATCH (n:P {name:'a'}) RETURN n.v AS v, n.w AS w") == [{"v": 10, "w": "x"}]
    q(db, "MATCH (n:P) SET n.all = true")
    assert count(db, "MATCH (n:P) WHERE n.all RETURN count(*) AS c") == 3


def test_set_property_from_expressions_and_other_nodes(people):
    db = people
    q(db, "MATCH (n:P) WHERE n.v IS NOT NULL SET n.v = n.v * 10 + 1")
    assert q(db, "MATCH (n:P) WHERE n.v IS NOT NULL RETURN n.v AS v ORDER BY v") == [{"v": 11}, {"v": 21}]
    q(db, "MATCH (a:P {name:'a'}), (c:P {name:'c'}) SET c.v = a.v + 1")
    assert q(db, "MATCH (c:P {name:'c'}) RETURN c.v AS v") == [{"v": 12}]
    q(db, "MATCH (n:P {name:'a'}) SET n.list = [1, 2, 3], n.copy = n.name")
    n = db.query("MATCH (n:P {name:'a'}) RETURN n")[0]["n"]
    assert n["list"] == [1, 2, 3] and n["copy"] == "a"


def test_set_null_removes_property(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) SET n.v = null")
    assert q(db, "MATCH (n:P {name:'a'}) RETURN n.v AS v") == [{"v": None}]
    assert "v" not in db.query("MATCH (n:P {name:'a'}) RETURN n")[0]["n"].properties


def test_set_is_applied_sequentially_per_row(db):
    q(db, "CREATE (:C {n: 0})")
    q(db, "UNWIND range(1, 5) AS i MATCH (c:C) SET c.n = c.n + i")
    assert q(db, "MATCH (c:C) RETURN c.n AS n") == [{"n": 15}]


def test_set_replace_and_merge_maps(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) SET n += {v: 99, extra: 'e'}")
    assert db.query("MATCH (n:P {name:'a'}) RETURN n")[0]["n"].properties == {"name": "a", "v": 99, "extra": "e"}
    q(db, "MATCH (n:P {name:'a'}) SET n += {v: null}")
    assert db.query("MATCH (n:P {name:'a'}) RETURN n")[0]["n"].properties == {"name": "a", "extra": "e"}
    q(db, "MATCH (n:P {name:'a'}) SET n = {only: 1}")
    assert db.query("MATCH (n:P {only: 1}) RETURN n")[0]["n"].properties == {"only": 1}
    q(db, "MATCH (n:P {name:'b'}), (m:P {name:'c'}) SET m = properties(n)")
    assert db.query("MATCH (n:P {name:'b'}) RETURN n")[0]["n"].properties == {"name": "b", "v": 2}
    assert db.query("MATCH (n:P) WHERE n.name = 'b' RETURN count(*) AS c") == [{"c": 2}]
    q(db, "MATCH (n:P {only: 1}) SET n += $m", m={"z": [1, 2]})
    assert db.query("MATCH (n:P {only: 1}) RETURN n.z AS z") == [{"z": [1, 2]}]


def test_set_labels(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) SET n:Extra")
    q(db, "MATCH (n:P {name:'b'}) SET n:One:Two")
    assert q(db, "MATCH (n:Extra) RETURN n.name AS n") == [{"n": "a"}]
    assert q(db, "MATCH (n:Two:One) RETURN n.name AS n") == [{"n": "b"}]
    q(db, "MATCH (n:P {name:'a'}) SET n:Extra")  # idempotent
    assert db.get_node(q(db, "MATCH (n:Extra) RETURN id(n) AS i")[0]["i"]).labels == ("Extra", "P")


def test_set_relationship_properties(db):
    q(db, "CREATE (a:A)-[:R {w: 1}]->(b:B)")
    q(db, "MATCH ()-[r:R]->() SET r.w = r.w + 1, r.new = 'x'")
    assert q(db, "MATCH ()-[r:R]->() RETURN r.w AS w, r.new AS n") == [{"w": 2, "n": "x"}]
    q(db, "MATCH ()-[r:R]->() SET r += {a: 1}")
    assert q(db, "MATCH ()-[r:R]->() RETURN properties(r) AS p") == [{"p": {"w": 2, "new": "x", "a": 1}}]
    q(db, "MATCH ()-[r:R]->() SET r = {only: 1}")
    assert q(db, "MATCH ()-[r:R]->() RETURN properties(r) AS p") == [{"p": {"only": 1}}]


def test_set_on_null_from_optional_match_is_noop(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) OPTIONAL MATCH (n)-[:X]->(m) SET m.v = 1")
    assert count(db) == 3


def test_set_type_errors(people):
    db = people
    with pytest.raises(CypherRuntimeError, match="primitive"):
        q(db, "MATCH (n:P {name:'a'}) SET n.m = {a: 1}")
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n:P) SET x.v = 1")
    with pytest.raises(CypherSyntaxError):
        q(db, "MATCH (n:P) WITH n.name AS s SET s.v = 1")


def test_remove_property_and_label(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) SET n:Tmp")
    q(db, "MATCH (n:P {name:'a'}) REMOVE n.v, n:Tmp")
    n = db.query("MATCH (n:P {name:'a'}) RETURN n")[0]["n"]
    assert n.properties == {"name": "a"} and n.labels == ("P",)
    q(db, "MATCH (n:P {name:'b'}) REMOVE n:P")
    assert count(db, "MATCH (n:P) RETURN count(*) AS c") == 2
    assert count(db) == 3
    q(db, "MATCH (n:P {name:'a'}) REMOVE n.nonexistent")  # no-op


def test_remove_relationship_property(db):
    q(db, "CREATE (a)-[:R {w: 1, x: 2}]->(b)")
    q(db, "MATCH ()-[r:R]->() REMOVE r.w")
    assert q(db, "MATCH ()-[r:R]->() RETURN properties(r) AS p") == [{"p": {"x": 2}}]


# ------------------------------------------------------------------ DELETE


def test_delete_node_without_relationships(people):
    db = people
    q(db, "MATCH (n:P {name:'a'}) DELETE n")
    assert count(db) == 2
    assert q(db, "MATCH (n:P) DELETE n RETURN count(*) AS c") == [{"c": 2}]
    assert count(db) == 0


def test_delete_node_with_relationships_fails_and_rolls_back(db):
    q(db, "CREATE (a:A {n:1})-[:R]->(b:B {n:2}), (c:C)")
    with pytest.raises(CypherRuntimeError, match="DETACH DELETE"):
        q(db, "MATCH (c:C) DELETE c WITH 1 AS x MATCH (a:A) DELETE a")
    assert count(db) == 3  # the successful part (deleting C) was rolled back too
    with pytest.raises(CypherRuntimeError):
        q(db, "MATCH (n) DELETE n")
    assert count(db) == 3
    assert count(db, "MATCH ()-[r]->() RETURN count(*) AS c") == 1


def test_detach_delete(db):
    q(db, "CREATE (a:A)-[:R]->(b:B)-[:R]->(c:C), (a)-[:S]->(c)")
    q(db, "MATCH (b:B) DETACH DELETE b")
    assert count(db) == 2
    assert count(db, "MATCH ()-[r]->() RETURN count(*) AS c") == 1
    q(db, "MATCH (n) DETACH DELETE n")
    assert count(db) == 0
    assert count(db, "MATCH ()-[r]->() RETURN count(*) AS c") == 0


def test_delete_relationships(db):
    q(db, "CREATE (a:A)-[:R]->(b:B), (a)-[:S]->(b)")
    q(db, "MATCH ()-[r:R]->() DELETE r")
    assert count(db, "MATCH ()-[r]->() RETURN count(*) AS c") == 1
    assert count(db) == 2


def test_delete_node_and_its_relationships_together(db):
    q(db, "CREATE (a:A)-[:R]->(b:B)")
    q(db, "MATCH (a:A)-[r]-(b) DELETE a, r, b")
    assert count(db) == 0


def test_delete_paths_and_lists(db):
    q(db, "CREATE (a:A)-[:R]->(b:B)-[:R]->(c:C)")
    q(db, "MATCH p = (a:A)-[:R*2]->(c) DELETE p")
    assert count(db) == 0
    q(db, "CREATE (:L {i:1}), (:L {i:2})")
    q(db, "MATCH (n:L) WITH collect(n) AS ns DELETE ns[0]")
    assert count(db) == 1
    q(db, "MATCH (n:L) WITH collect(n) AS ns UNWIND ns AS x DELETE x")
    assert count(db) == 0


def test_delete_null_is_noop_and_bad_types_error(db):
    q(db, "CREATE (:A {x: 1})")
    q(db, "OPTIONAL MATCH (n:Nope) DELETE n")
    q(db, "MATCH (n:A) DELETE n.missing")
    assert count(db) == 1
    with pytest.raises(CypherRuntimeError):
        q(db, "MATCH (n:A) DELETE n.x")
    with pytest.raises(CypherRuntimeError):
        q(db, "WITH 1 AS x DELETE x")


def test_returning_deleted_node_is_an_error(db):
    q(db, "CREATE (:A)")
    with pytest.raises(CypherRuntimeError, match="deleted"):
        q(db, "MATCH (n:A) DELETE n RETURN n")
    assert count(db) == 1


# ------------------------------------------------------------------ atomicity


def test_rollback_on_failure_mid_query(db):
    with pytest.raises(CypherRuntimeError):
        q(db, "CREATE (a:T {v: 1}) WITH a MATCH (b:T) SET b.bad = {m: 1}")
    assert count(db) == 0
    q(db, "CREATE (:T {v: 1})")
    with pytest.raises(CypherRuntimeError):
        q(db, "UNWIND [1, 2, 3] AS i CREATE (:U {i: i}) WITH i MATCH (t:T) SET t.v = {x: i}")
    assert count(db, "MATCH (u:U) RETURN count(*) AS c") == 0
    assert q(db, "MATCH (t:T) RETURN t.v AS v") == [{"v": 1}]


def test_failed_query_inside_transaction_only_rolls_back_itself(db):
    with db.transaction():
        q(db, "CREATE (:Keep)")
        with pytest.raises(CypherRuntimeError):
            q(db, "CREATE (:Lost) WITH 1 AS x MATCH (k:Keep) SET k.m = {a: 1}")
        q(db, "CREATE (:Keep2)")
    assert count(db, "MATCH (n:Keep) RETURN count(*) AS c") == 1
    assert count(db, "MATCH (n:Lost) RETURN count(*) AS c") == 0
    assert count(db, "MATCH (n:Keep2) RETURN count(*) AS c") == 1


def test_query_writes_join_outer_transaction_rollback(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            q(db, "CREATE (:X)")
            raise RuntimeError
    assert count(db) == 0


def test_read_only_rejects_write_queries(tmp_path):
    path = tmp_path / "ro.db"
    with GraphDB(path) as db:
        db.create_node("P", {"n": 1})
    with GraphDB(path, read_only=True) as ro:
        for cy in (
            "CREATE (:X)",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) DELETE n",
            "MERGE (:X)",
            "MATCH (n) REMOVE n.n",
            "CREATE VECTOR INDEX v FOR (n:P) ON (n.e) OPTIONS {indexConfig: {`vector.dimensions`: 2}}",
        ):
            with pytest.raises(SqgraphError, match="read-only"):
                ro.query(cy)
        assert ro.query("MATCH (n) RETURN count(*) AS c") == [{"c": 1}]
    with GraphDB(path) as db:
        assert db.stats()["nodes"] == 1


def test_updates_visible_to_python_api(db):
    q(db, "CREATE (n:P {a: 1})")
    nid = q(db, "MATCH (n:P) RETURN id(n) AS i")[0]["i"]
    db.update_node(nid, {"b": 2})
    assert q(db, "MATCH (n:P) RETURN n.a AS a, n.b AS b") == [{"a": 1, "b": 2}]
    q(db, "MATCH (n:P) SET n.c = 3")
    assert db.get_node(nid).properties == {"a": 1, "b": 2, "c": 3}


def test_merge_then_match_pipeline(db):
    rows = q(
        db,
        "UNWIND ['x', 'y'] AS name MERGE (n:Tag {name: name}) WITH collect(n) AS tags "
        "CREATE (d:Doc {title: 'd'}) WITH d, tags UNWIND tags AS t CREATE (d)-[:TAGGED]->(t) "
        "RETURN count(*) AS c",
    )
    assert rows == [{"c": 2}]
    assert q(db, "MATCH (:Doc)-[:TAGGED]->(t:Tag) RETURN t.name AS n ORDER BY n") == [{"n": "x"}, {"n": "y"}]


def test_graphrag_style_upsert_with_unwind_maps(db):
    entities = [{"name": "A", "type": "org"}, {"name": "B", "type": "person"}, {"name": "A", "type": "org"}]
    q(db, "UNWIND $es AS e MERGE (n:Entity {name: e.name}) SET n.type = e.type", es=entities)
    assert q(db, "MATCH (n:Entity) RETURN n.name AS n, n.type AS t ORDER BY n") == [
        {"n": "A", "t": "org"},
        {"n": "B", "t": "person"},
    ]
    rels = [{"s": "A", "t": "B", "w": 1}]
    q(
        db,
        "UNWIND $rs AS r MATCH (a:Entity {name: r.s}), (b:Entity {name: r.t}) "
        "MERGE (a)-[x:RELATED_TO]->(b) SET x.w = r.w",
        rs=rels,
    )
    assert q(db, "MATCH (a)-[x:RELATED_TO]->(b) RETURN a.name AS a, b.name AS b, x.w AS w") == [{"a": "A", "b": "B", "w": 1}]
