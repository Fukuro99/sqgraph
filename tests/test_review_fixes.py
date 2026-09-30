"""Regression tests for the findings of the independent review (numbered as in the review)."""

from __future__ import annotations

import random
import time

import pytest

from sqgraph import CypherRuntimeError, DimensionMismatchError, GraphDB, SqgraphError
from sqgraph.cypher.compiler import Compiler
from sqgraph.cypher.parser import parse
from sqgraph.schema import is_identifier


def q(db, cypher, **params):
    return db.query(cypher, params)


def plan(db, cypher, params=None):
    comp = Compiler(db, params or {}, cypher)
    seg = comp.compile_segment(parse(cypher).parts[0].clauses)
    rows = db._conn.execute("EXPLAIN QUERY PLAN " + seg.sql, seg.binds).fetchall()
    return seg.sql, " | ".join(r[3] for r in rows)


def vec_ids(db, name="vi"):
    return sorted(r[0] for r in db._conn.execute(f"SELECT node_id FROM vec_{name}"))


def fts_ids(db, name="ft"):
    return sorted(r[0] for r in db._conn.execute(f"SELECT rowid FROM fts_{name}"))


def social_graph(n, seed=1):
    rnd = random.Random(seed)
    db = GraphDB()
    ids = db.create_nodes([("Person", {"pid": i}) for i in range(n)])
    db.create_edges([(ids[i % n], ids[rnd.randrange(n)], "KNOWS") for i in range(n * 3)])
    return db, ids


# ------------------------------------------------------------------ 1. planner traps on an un-analyzed database


def test_1_typed_varlen_on_fresh_database_is_fast_and_uses_the_adjacency_index():
    db, _ = social_graph(1500)  # never analyzed
    t = time.time()
    rows = db.query("MATCH (a:Person)-[:KNOWS*1..2]->(b) RETURN count(*) AS c")
    assert time.time() - t < 5.0  # was ~12 s (walked idx_edges_type)
    assert rows[0]["c"] > 4500
    sql, _ = plan(db, "MATCH (a:Person)-[:KNOWS*1..2]->(b) RETURN count(*) AS c")
    assert "INDEXED BY idx_edges_src" in sql
    sql, _ = plan(db, "MATCH (a:Person)<-[:KNOWS*1..2]-(b) RETURN count(*) AS c")
    assert "INDEXED BY idx_edges_dst" in sql
    t = time.time()
    db.query("MATCH (a:Person)<-[:KNOWS*1..2]-(b) RETURN count(*) AS c")
    assert time.time() - t < 5.0


def test_1_other_join_shapes_are_fast_without_statistics():
    db, ids = social_graph(1500)
    shapes = [
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN count(*) AS c",
        "MATCH (a:Person {pid: 5})-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*) AS c",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c:Person {pid: 5}) RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS*1..3]->(b:Person {pid: 5}) RETURN count(DISTINCT a) AS c",
        "MATCH (a:Person {pid: 3})-[:KNOWS*1..3]-(b) RETURN count(*) AS c",
    ]
    for cy in shapes:
        t = time.time()
        db.query(cy)
        assert time.time() - t < 5.0, cy


def test_1_redundant_type_index_is_gone_and_old_databases_are_migrated(tmp_path):
    with GraphDB() as db:
        names = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert "idx_edges_type" not in names and {"idx_edges_src", "idx_edges_dst"} <= names
    path = tmp_path / "old.db"
    with GraphDB(path) as db:
        a, b = db.create_node("A"), db.create_node("B")
        db.create_edge(a.id, b.id, "R")
        db._conn.execute("CREATE INDEX idx_edges_type ON edges(type)")  # what schema version 1 created
        db._conn.execute("UPDATE _meta SET value = '1' WHERE key = 'schema_version'")
    with GraphDB(path) as db:
        names = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert "idx_edges_type" not in names
        assert db._conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()[0] == "3"
        assert db.query("MATCH ()-[r:R]->() RETURN count(*) AS c") == [{"c": 1}]


# ------------------------------------------------------------------ 2. bm25 must never be flattened into the caller


def test_2_fulltext_call_followed_by_match_with_aggregates_and_order():
    db = GraphDB()
    db.create_fulltext_index("ft", "Chunk", "text")
    db.query(
        "CREATE (d:Doc {n:1})-[:HAS]->(:Chunk {text:'hello world'}), (d)-[:HAS]->(:Chunk {text:'hello there'}), "
        "(:Doc {n:2})-[:HAS]->(:Chunk {text:'other'})"
    )
    call = "CALL db.index.fulltext.queryNodes('ft', 'hello') YIELD node, score "
    m = "MATCH (d:Doc)-[:HAS]->(node) "
    assert q(db, call + m + "RETURN d.n AS d, count(*) AS c") == [{"d": 1, "c": 2}]
    (row,) = q(db, call + m + "RETURN d.n AS d, max(score) AS s")
    assert row["d"] == 1 and row["s"] > 0
    assert len(q(db, call + m + "RETURN d.n AS d, collect(score) AS s")[0]["s"]) == 2
    assert q(db, call + m + "RETURN d.n AS d, sum(score) AS s")[0]["s"] > 0
    assert len(q(db, call + "WITH node, score " + m + "RETURN d.n AS d, min(score) AS s")) == 1
    rows = q(db, call + m + "RETURN d.n AS d, score ORDER BY score DESC LIMIT 1")
    assert len(rows) == 1
    assert len(q(db, call + m + "RETURN DISTINCT d.n AS d, score")) >= 1
    assert q(
        db, "CALL db.index.fulltext.queryNodes('ft', 'hello', {limit: 5}) YIELD node, score " + m + "RETURN count(score) AS c"
    ) == [{"c": 2}]


# ------------------------------------------------------------------ 3. registries follow rollbacks


def fresh_doc_db():
    db = GraphDB()
    db.create_fulltext_index("ft", "Doc", ["title"])
    db.create_vector_index("vi", "Doc", "embedding", dimensions=2)
    return db


def test_3_failed_vector_index_creation_leaves_no_stale_registry():
    db = GraphDB()
    db.query("CREATE (:Doc {embedding: [1, 2, 3], t: 'x'})")
    with pytest.raises(DimensionMismatchError):
        db.create_vector_index("vi", "Doc", "embedding", dimensions=2)
    assert db.vector_indexes() == []
    db.query("CREATE (:Doc {t: 'y', embedding: [1, 2]})")  # must not touch a missing vec_vi table
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'vec_vi'").fetchone()[0] == 0
    db2 = GraphDB()
    db2.query("CREATE (:Doc {embedding: ['a', 'b']})")
    with pytest.raises(SqgraphError):
        db2.create_vector_index("vi", "Doc", "embedding", dimensions=2)
    assert db2.vector_indexes() == []


def test_3_index_creation_inside_rolled_back_transaction():
    db = GraphDB()
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.create_vector_index("zz", "Doc", dimensions=2)
            db.create_fulltext_index("fz", "Doc", "t")
            raise RuntimeError
    assert db.vector_indexes() == [] and db.fulltext_indexes() == []
    db.query("CREATE (:Doc {t: 'x', embedding: [1, 0]})")  # no ghost indexes to maintain
    assert db.stats()["nodes"] == 1
    db.create_vector_index("zz", "Doc", dimensions=2)  # and the names are free again
    assert [i.name for i in db.vector_indexes()] == ["zz"]


def test_3_drop_index_inside_rolled_back_transaction():
    db = fresh_doc_db()
    db.create_node("Doc", {"title": "hello", "embedding": [1, 0]})
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.drop_index("vi")
            db.drop_index("ft")
            raise RuntimeError
    assert [i.name for i in db.vector_indexes()] == ["vi"] and [i.name for i in db.fulltext_indexes()] == ["ft"]
    db.create_node("Doc", {"title": "second", "embedding": [0, 1]})
    assert vec_ids(db) == [1, 2] and fts_ids(db) == [1, 2]
    assert len(db.vector_search("vi", [1, 0], k=5)) == 2
    assert len(db.fulltext_search("ft", "second")) == 1


def test_3_registry_restored_by_inner_savepoint_rollback():
    db = GraphDB()
    with db.transaction():
        db.create_vector_index("outer", "Doc", dimensions=2)
        with pytest.raises(RuntimeError):
            with db.transaction():
                db.create_vector_index("inner", "Doc", dimensions=2)
                raise RuntimeError
        assert [i.name for i in db.vector_indexes()] == ["outer"]
    assert [i.name for i in db.vector_indexes()] == ["outer"]


# ------------------------------------------------------------------ 4/5. label changes move embeddings correctly


def test_4_removing_the_last_covering_label_keeps_the_embedding_as_a_property():
    db = fresh_doc_db()
    n = db.create_node("Doc", {"title": "hello", "embedding": [1.0, 0.0]})
    q(db, "MATCH (d:Doc) REMOVE d:Doc")
    assert vec_ids(db) == []
    assert db.get_node(n.id).properties["embedding"] == [1.0, 0.0]
    q(db, "MATCH (d) WHERE id(d) = $i SET d:Doc", i=n.id)  # and adding it back re-indexes (item 5)
    assert vec_ids(db) == [n.id] and "embedding" not in db.get_node(n.id).properties
    db.remove_labels(n.id, "Doc")
    assert db.get_node(n.id).properties["embedding"] == [1.0, 0.0]


def test_4_embedding_stays_indexed_while_another_index_still_covers_it():
    db = GraphDB()
    db.create_vector_index("a", "A", "embedding", dimensions=2)
    db.create_vector_index("b", "B", "embedding", dimensions=2)
    n = db.create_node(["A", "B"], {"embedding": [1.0, 0.0]})
    assert vec_ids(db, "a") == [n.id] and vec_ids(db, "b") == [n.id]
    db.remove_labels(n.id, "A")
    assert vec_ids(db, "a") == [] and vec_ids(db, "b") == [n.id]
    assert "embedding" not in db.get_node(n.id).properties
    db.remove_labels(n.id, "B")
    assert db.get_node(n.id).properties["embedding"] == [1.0, 0.0]


def test_5_adding_a_label_migrates_an_existing_embedding_property():
    db = fresh_doc_db()
    m = db.create_node("Other", {"title": "hello", "embedding": [1, 0]})
    db.add_labels(m.id, "Doc")
    assert vec_ids(db) == [m.id]
    assert db.get_node(m.id).properties == {"title": "hello"}
    assert db.vector_search("vi", [1, 0], k=1)[0][0].id == m.id
    n = db.create_node("Other", {"title": "second", "embedding": [0, 1]})
    q(db, "MATCH (x:Other {title: 'second'}) SET x:Doc")
    assert vec_ids(db) == sorted([m.id, n.id]) and "embedding" not in db.get_node(n.id).properties


def test_5_label_add_with_wrong_dimensions_fails_and_rolls_back():
    db = fresh_doc_db()
    m = db.create_node("Other", {"embedding": [1, 2, 3]})
    with pytest.raises(DimensionMismatchError):
        db.add_labels(m.id, "Doc")
    assert db.get_node(m.id).labels == ("Other",) and db.get_node(m.id).properties == {"embedding": [1, 2, 3]}
    assert vec_ids(db) == []
    with pytest.raises(SqgraphError):
        q(db, "MATCH (x:Other) SET x:Doc")
    assert db.get_node(m.id).labels == ("Other",)


# ------------------------------------------------------------------ 6. Neo4j type semantics


@pytest.fixture
def typed(db):
    q(db, "CREATE (:F {v: true}), (:F {v: false}), (:F {v: 1}), (:F {v: 0}), (:S {v: '1'}), (:S {v: 2})")
    return db


def test_6_boolean_is_not_an_integer(typed):
    db = typed
    assert q(db, "RETURN 1 = true AS a, true = true AS b, 0 = false AS c, 1 <> true AS d") == [
        {"a": False, "b": True, "c": False, "d": True}
    ]
    assert q(db, "MATCH (n:F {v: true}) RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:F {v: 1}) RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:F) WHERE n.v = 0 RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:F) WHERE n.v = false RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:F) WHERE n.v IN [1] RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:F) WHERE n.v IN [true, 5] RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:F) WHERE n.v > 0 RETURN n.v AS v") == [{"v": 1}]
    assert q(db, "MATCH (n:F) RETURN count(DISTINCT n.v) AS c") == [{"c": 4}]
    assert q(db, "MATCH (a:F), (b:F) WHERE a.v = b.v RETURN count(*) AS c") == [{"c": 4}]
    assert q(db, "MATCH (a:F), (b:F) WHERE a.v < b.v RETURN count(*) AS c") == [{"c": 2}]  # 0<1 and false<true


def test_6_merge_distinguishes_true_from_one(typed):
    db = typed
    q(db, "MERGE (n:F {v: true})")
    q(db, "MERGE (n:F {v: 1})")
    assert q(db, "MATCH (n:F) RETURN count(n) AS c") == [{"c": 4}]
    q(db, "MERGE (n:F {v: 7})")
    q(db, "UNWIND [true, 1, 7] AS x MERGE (n:F {v: x})")
    assert q(db, "MATCH (n:F) RETURN count(n) AS c") == [{"c": 5}]
    q(db, "MERGE (n:F {v: '1'})")
    assert q(db, "MATCH (n:F) RETURN count(n) AS c") == [{"c": 6}]


def test_6_cross_type_comparisons(typed):
    db = typed
    assert q(db, "RETURN 1 = '1' AS a, 1 < 'a' AS b, 'a' < 1 AS c, 1 <> 'a' AS d, null = 1 AS e, [1] = 1 AS f") == [
        {"a": False, "b": None, "c": None, "d": True, "e": None, "f": False}
    ]
    assert q(db, "MATCH (n:S) WHERE n.v >= 0 RETURN n.v AS v") == [{"v": 2}]
    assert q(db, "MATCH (n:S) WHERE n.v < 5 RETURN n.v AS v") == [{"v": 2}]
    assert q(db, "MATCH (n:S) WHERE n.v > '0' RETURN n.v AS v") == [{"v": "1"}]
    assert q(db, "MATCH (n:S) WHERE n.v = 1 RETURN count(n) AS c") == [{"c": 0}]
    assert q(db, "MATCH (n:S) WHERE n.v = '1' RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:S) WHERE n.v <> 2 RETURN n.v AS v") == [{"v": "1"}]
    assert q(db, "MATCH (n:S) WHERE n.missing = 1 RETURN count(n) AS c") == [{"c": 0}]
    assert q(db, "MATCH (n:S) WHERE n.missing <> 1 RETURN count(n) AS c") == [{"c": 0}]
    assert q(db, "RETURN CASE 1 WHEN true THEN 'bool' WHEN 1 THEN 'int' END AS r") == [{"r": "int"}]
    assert q(db, "RETURN 1 = 1.0 AS a, 2 < 2.5 AS b, 'a' < 'b' AS c, true > false AS d") == [
        {"a": True, "b": True, "c": True, "d": True}
    ]


def test_6_anchored_property_lookups_still_use_the_property_index():
    db = GraphDB()
    db.create_nodes([("P", {"name": f"n{i}", "k": i, "flag": bool(i % 2)}) for i in range(300)])
    for prop in ("name", "k", "flag"):
        db.create_property_index("P", prop)
    for cy, params, prop in (
        ("MATCH (n:P {name: $v}) RETURN n", {"v": "n5"}, "name"),
        ("MATCH (n {k: $v}) RETURN n", {"v": 5}, "k"),
        ("MATCH (n {name: 'n5'}) RETURN n", {}, "name"),
        ("MATCH (n {flag: $v}) RETURN n", {"v": True}, "flag"),
        ("MATCH (n) WHERE n.k = $v RETURN n", {"v": 7}, "k"),
    ):
        sql, p = plan(db, cy, params)
        assert "idx_p_prop_P_" + prop in p, (cy, p)
    assert q(db, "MATCH (n:P {k: 5}) RETURN n.name AS n") == [{"n": "n5"}]
    assert q(db, "MATCH (n:P {k: '5'}) RETURN n.name AS n") == []
    assert q(db, "MATCH (n:P {k: true}) RETURN count(n) AS c") == [{"c": 0}]
    # values coming from a row map are typed dynamically; equality must still be an index seek
    sql, p = plan(db, "UNWIND $rows AS row MATCH (n:P {name: row.name}) RETURN n", {"rows": [{"name": "n1"}]})
    assert "idx_p_prop_P_name" in p, p
    assert q(db, "UNWIND $rows AS row MATCH (n:P {name: row.name}) RETURN n.k AS k", rows=[{"name": "n1"}, {"name": 7}]) == [{"k": 1}]


def test_6_relationship_property_maps_use_typed_equality():
    db = GraphDB()
    q(db, "CREATE (a)-[:R {w: true}]->(b), (a)-[:R {w: 1}]->(b)")
    assert q(db, "MATCH ()-[r:R {w: true}]->() RETURN count(r) AS c") == [{"c": 1}]
    assert q(db, "MATCH ()-[r:R {w: 1}]->() RETURN count(r) AS c") == [{"c": 1}]
    assert q(db, "MATCH ()-[r:R*1..1 {w: 1}]->() RETURN count(r) AS c") == [{"c": 1}]


# ------------------------------------------------------------------ 7. three-valued IN


def test_7_in_with_null_lists_and_elements():
    db = GraphDB()
    assert q(db, "RETURN 'x' IN null AS a, 1 IN $l AS b, 2 IN [1, null] AS c, 1 IN [1, null] AS d, null IN [] AS e, null IN [1] AS f, NOT (2 IN [1, null]) AS g", l=None) == [
        {"a": None, "b": None, "c": None, "d": True, "e": False, "f": None, "g": None}
    ]
    q(db, "CREATE (:T {id: 1, tags: ['x']}), (:T {id: 2}), (:T {id: 3, tags: ['y']})")
    assert q(db, "MATCH (n:T) RETURN n.id AS id, 'x' IN n.tags AS r ORDER BY id") == [
        {"id": 1, "r": True}, {"id": 2, "r": None}, {"id": 3, "r": False},
    ]
    assert q(db, "MATCH (n:T) WHERE NOT 'x' IN n.tags RETURN n.id AS id") == [{"id": 3}]
    assert q(db, "RETURN 2 IN $l AS a, 1 IN $l AS b", l=[1, None]) == [{"a": None, "b": True}]
    assert q(db, "UNWIND [[1, null], [1, 2]] AS l RETURN 2 IN l AS r") == [{"r": None}, {"r": True}]


# ------------------------------------------------------------------ 8. procedure row order survives MATCH / LIMIT


def test_8_vector_call_then_match_limit_returns_global_best():
    rnd = random.Random(3)
    db = GraphDB()
    db.create_vector_index("vi", "Chunk", "embedding", dimensions=4)
    n = 300
    ids = db.create_nodes([("Chunk", {"embedding": [rnd.random() for _ in range(4)], "i": i}) for i in range(n)])
    docs = db.create_nodes([("Doc", {"d": i}) for i in range(10)])
    db.create_edges([(docs[i % 10], ids[i], "HAS") for i in range(n)])
    qv = [rnd.random() for _ in range(4)]
    best = db.query("CALL db.index.vector.queryNodes('vi', 50, $q) YIELD node, score RETURN node.i AS i, score", {"q": qv})
    for tail in (
        "MATCH (d:Doc)-[:HAS]->(node) RETURN node.i AS i, score LIMIT 5",
        "WITH node, score MATCH (d:Doc)-[:HAS]->(node) RETURN node.i AS i, score LIMIT 5",
        "MATCH (d:Doc)-[:HAS]->(node) WITH node, score RETURN node.i AS i, score LIMIT 5",
        "MATCH (d:Doc)-[:HAS]->(node) WITH node, score LIMIT 5 RETURN node.i AS i, score",
        "OPTIONAL MATCH (d:Doc)-[:HAS]->(node) RETURN node.i AS i, score LIMIT 5",
    ):
        rows = db.query("CALL db.index.vector.queryNodes('vi', 50, $q) YIELD node, score " + tail, {"q": qv})
        assert [r["i"] for r in rows] == [r["i"] for r in best[:5]], tail
    rows = db.query(
        "CALL db.index.vector.queryNodes('vi', 50, $q) YIELD node, score MATCH (d:Doc)-[:HAS]->(node) RETURN node.i AS i, score",
        {"q": qv},
    )
    assert [r["i"] for r in rows] == [r["i"] for r in best]
    # an explicit ORDER BY still wins
    rows = db.query(
        "CALL db.index.vector.queryNodes('vi', 50, $q) YIELD node, score MATCH (d:Doc)-[:HAS]->(node) "
        "RETURN node.i AS i ORDER BY i LIMIT 3",
        {"q": qv},
    )
    assert [r["i"] for r in rows] == sorted(r["i"] for r in best)[:3]


def test_8_fulltext_call_then_match_is_score_ordered():
    db = GraphDB()
    db.create_fulltext_index("ft", "Chunk", "text")
    docs = db.create_nodes([("Doc", {"d": i}) for i in range(3)])
    texts = ["common", "common common common", "common common", "unrelated"]
    chunks = db.create_nodes([("Chunk", {"text": t}) for t in texts])
    db.create_edges([(docs[i % 3], chunks[i], "HAS") for i in range(4)])
    rows = db.query(
        "CALL db.index.fulltext.queryNodes('ft', 'common') YIELD node, score MATCH (d:Doc)-[:HAS]->(node) RETURN node.text AS t, score"
    )
    assert [r["t"] for r in rows][0] == "common common common"
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ 9. seed var-length from the selective end


def test_9_far_end_anchor_seeds_the_recursion_and_gives_the_same_answer():
    db, ids = social_graph(3000, seed=2)
    db.create_property_index("Person", "pid")
    sql = db.explain("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person {pid: $v}) RETURN count(DISTINCT a) AS c", {"v": 5})
    # the anchored node is materialized first, and the recursion runs backwards (over dst)
    assert sql.index("'$.pid'") < sql.index("vl")
    assert "INDEXED BY idx_edges_dst" in sql
    t = time.time()
    got = q(db, "MATCH (a:Person)-[:KNOWS*1..3]->(b:Person {pid: $v}) RETURN count(DISTINCT a) AS c", v=5)
    assert time.time() - t < 3.0
    forward_form = q(
        db, "MATCH (b:Person {pid: $v}) MATCH (a:Person)-[:KNOWS*1..3]->(b) RETURN count(DISTINCT a) AS c", v=5
    )
    assert got == forward_form
    # id() equality counts as an anchor too
    got = q(db, "MATCH (a)-[:KNOWS*1..3]->(b) WHERE id(b) = $i RETURN count(DISTINCT a) AS c", i=ids[5])
    assert got == q(db, "MATCH (b) WHERE id(b) = $i MATCH (a)-[:KNOWS*1..3]->(b) RETURN count(DISTINCT a) AS c", i=ids[5])
    sql = db.explain("MATCH (a)-[:KNOWS*1..3]->(b) WHERE id(b) = $i RETURN count(DISTINCT a) AS c", {"i": ids[5]})
    assert "INDEXED BY idx_edges_dst" in sql
    # a selective left end is still used forwards
    sql = db.explain("MATCH (a:Person {pid: $v})-[:KNOWS*1..3]->(b:Person) RETURN count(DISTINCT b) AS c", {"v": 5})
    assert "INDEXED BY idx_edges_src" in sql and "INDEXED BY idx_edges_dst" not in sql


def test_9_reversed_traversal_keeps_semantics_on_random_graphs():
    for seed in range(4):
        rnd = random.Random(seed)
        db = GraphDB()
        ids = db.create_nodes([("P", {"pid": i}) for i in range(8)])
        for _ in range(16):
            db.create_edge(rnd.choice(ids), rnd.choice(ids), "E")
        for arrow in ("-[:E*1..3]->", "-[:E*2..3]-", "<-[:E*1..2]-"):
            for pid in range(8):
                a = q(db, f"MATCH (a:P){arrow}(b:P {{pid: $v}}) RETURN a.pid AS a ORDER BY a", v=pid)
                b = q(db, f"MATCH (b:P {{pid: $v}}) MATCH (a:P){arrow}(b) RETURN a.pid AS a ORDER BY a", v=pid)
                assert a == b, (seed, arrow, pid)


# ------------------------------------------------------------------ 10. index names


def test_10_create_index_honors_the_given_name():
    db = GraphDB()
    q(db, "CREATE INDEX my_idx FOR (n:A) ON (n.p)")
    assert [i["name"] for i in db.list_indexes()] == ["my_idx"]
    assert [r["name"] for r in q(db, "SHOW INDEXES")] == ["my_idx"]
    q(db, "CREATE INDEX my_idx IF NOT EXISTS FOR (n:A) ON (n.p)")
    with pytest.raises(SqgraphError):
        q(db, "CREATE INDEX my_idx FOR (n:A) ON (n.p)")
    with pytest.raises(SqgraphError):
        q(db, "CREATE INDEX my_idx IF NOT EXISTS FOR (n:B) ON (n.other)")
    q(db, "DROP INDEX my_idx")
    assert db.list_indexes() == []
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'idx_p_%'").fetchone()[0] == 0
    q(db, "DROP INDEX my_idx IF EXISTS")
    # a user name can never clobber internal objects
    q(db, "CREATE INDEX idx_edges_src FOR (n:A) ON (n.p)")
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'idx_edges_src'").fetchone()[0] == 1
    q(db, "DROP INDEX idx_edges_src")
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'idx_edges_src'").fetchone()[0] == 1
    q(db, "CREATE VECTOR INDEX vv FOR (n:A) ON (n.e) OPTIONS {indexConfig: {`vector.dimensions`: 2}}")
    with pytest.raises(SqgraphError):
        q(db, "CREATE INDEX vv FOR (n:A) ON (n.p)")


def test_10_generated_names_are_collision_free():
    db = GraphDB()
    n1 = db.create_property_index("A_b", "c")
    n2 = db.create_property_index("A", "b_c")
    n3 = db.create_property_index("A b", "c")
    assert len({n1, n2, n3}) == 3
    assert {i["name"] for i in db.list_indexes()} == {n1, n2, n3}
    assert db.create_property_index("A_b", "c") == n1  # idempotent
    for n in (n1, n2, n3):
        db.drop_index(n)
    assert db.list_indexes() == []
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'idx_p_%'").fetchone()[0] == 0


# ------------------------------------------------------------------ 11. identifiers must fullmatch


def test_11_trailing_newline_is_not_an_identifier():
    assert is_identifier("abc") and not is_identifier("abc\n") and not is_identifier("a b") and not is_identifier("")
    db = GraphDB()
    for bad in ("v\n", "v\r\n", "v x"):
        with pytest.raises(SqgraphError):
            db.create_vector_index(bad, "A", dimensions=2)
        with pytest.raises(SqgraphError):
            db.create_fulltext_index(bad, "A", "p")
        with pytest.raises(SqgraphError):
            db.create_property_index("A", "p", name=bad)
    with pytest.raises(SqgraphError):
        db.create_property_index("A", "a\n")
    assert db.vector_indexes() == [] and db.fulltext_indexes() == [] and db.list_indexes() == []
    # a property key ending in a newline is still an ordinary (parameter-quoted) key
    q(db, "CREATE (n:A {`a\n`: 1})")
    assert q(db, "MATCH (n:A {`a\n`: 1}) RETURN n.`a\n` AS v") == [{"v": 1}]
    assert q(db, "MATCH (n:A) RETURN n.a AS v") == [{"v": None}]
    with pytest.raises(SqgraphError):
        q(db, "CREATE INDEX FOR (n:A) ON (n.`a\n`)")


# ------------------------------------------------------------------ 12. three-valued quantifiers


def test_12_quantifiers_follow_three_valued_logic():
    db = GraphDB()
    row = q(
        db,
        "RETURN "
        "all(x IN [1, null] WHERE x > 0) AS a1, all(x IN [1, -1, null] WHERE x > 0) AS a2, all(x IN [1, 2] WHERE x > 0) AS a3, "
        "all(x IN [] WHERE x > 0) AS a4, "
        "any(x IN [null, 1] WHERE x > 5) AS n1, any(x IN [null, 9] WHERE x > 5) AS n2, any(x IN [1] WHERE x > 5) AS n3, "
        "none(x IN [null] WHERE x > 0) AS o1, none(x IN [null, 1] WHERE x > 0) AS o2, none(x IN [-1] WHERE x > 0) AS o3, "
        "single(x IN [null, 1] WHERE x > 0) AS s1, single(x IN [1, 2] WHERE x > 0) AS s2, "
        "single(x IN [1, -1] WHERE x > 0) AS s3, single(x IN [-1] WHERE x > 0) AS s4, single(x IN [null] WHERE x > 0) AS s5",
    )[0]
    assert row == {
        "a1": None, "a2": False, "a3": True, "a4": True,
        "n1": None, "n2": True, "n3": False,
        "o1": None, "o2": False, "o3": True,
        "s1": None, "s2": False, "s3": True, "s4": False, "s5": None,
    }
    assert q(db, "RETURN all(x IN null WHERE x > 0) AS a, any(x IN $l WHERE x > 0) AS b", l=None) == [{"a": None, "b": None}]
    q(db, "CREATE (:T {v: [1, 2]}), (:T {v: [1]}), (:T)")
    assert q(db, "MATCH (n:T) WHERE any(x IN n.v WHERE x = 2) RETURN count(n) AS c") == [{"c": 1}]
    assert q(db, "MATCH (n:T) WHERE none(x IN n.v WHERE x = 2) RETURN count(n) AS c") == [{"c": 1}]  # null list -> null -> filtered
