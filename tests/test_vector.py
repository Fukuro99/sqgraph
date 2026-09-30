from __future__ import annotations

import math

import pytest

from sqgraph import (
    CypherRuntimeError,
    CypherSyntaxError,
    DimensionMismatchError,
    GraphDB,
    Node,
    SqgraphError,
)


def q(db, cypher, **params):
    return db.query(cypher, params)


def vec_rows(db, index="emb"):
    return db._conn.execute(f"SELECT count(*) FROM vec_{index}").fetchone()[0]


@pytest.fixture
def vdb(db):
    """Three entities with 2-d embeddings: x-axis, y-axis and the diagonal."""
    db.create_vector_index("emb", "Entity", "embedding", 2)
    db.create_node("Entity", {"name": "x"})
    db.create_node("Entity", {"name": "y"})
    db.create_node("Entity", {"name": "d"})
    ids = {r["name"]: r["i"] for r in db.query("MATCH (n:Entity) RETURN n.name AS name, id(n) AS i")}
    db.set_node_vector(ids["x"], "embedding", [1.0, 0.0])
    db.set_node_vector(ids["y"], "embedding", [0.0, 1.0])
    db.set_node_vector(ids["d"], "embedding", [1.0, 1.0])
    db.ids = ids
    return db


# ------------------------------------------------------------------ index management


def test_create_vector_index_api(db):
    ix = db.create_vector_index("idx1", "Doc", "embedding", 4, "cosine")
    assert (ix.name, ix.label, ix.property, ix.dimensions, ix.similarity) == ("idx1", "Doc", "embedding", 4, "cosine")
    assert db.vector_indexes() == [ix]
    row = db._conn.execute("SELECT name, label, property, dimensions, similarity FROM _vector_indexes").fetchone()
    assert row == ("idx1", "Doc", "embedding", 4, "cosine")
    assert vec_rows(db, "idx1") == 0
    with pytest.raises(SqgraphError, match="already exists"):
        db.create_vector_index("idx1", "Doc", "embedding", 4)
    assert db.create_vector_index("idx1", "Doc", "embedding", 4, if_not_exists=True) == ix


def test_create_vector_index_validation(db):
    for bad in ("a b", "a;DROP TABLE nodes", "1abc", "", "a-b", "x' --"):
        with pytest.raises(SqgraphError):
            db.create_vector_index(bad, "L", "embedding", 3)
    with pytest.raises(SqgraphError):
        db.create_vector_index("ok", "L", "embedding", 0)
    with pytest.raises(SqgraphError):
        db.create_vector_index("ok", "L", "embedding", None)
    with pytest.raises(SqgraphError):
        db.create_vector_index("ok", "L", "embedding", 3, "manhattan")
    assert db.stats()["nodes"] == 0
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'vec_%'").fetchone()[0] == 0


def test_create_vector_index_via_cypher_ddl(db):
    q(
        db,
        "CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 4, `vector.similarity_function`: 'cosine'}}",
    )
    (ix,) = db.vector_indexes()
    assert (ix.name, ix.label, ix.property, ix.dimensions, ix.similarity) == ("entity_emb", "Entity", "embedding", 4, "cosine")
    # idempotent with IF NOT EXISTS
    q(
        db,
        "CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 4, `vector.similarity_function`: 'cosine'}}",
    )
    with pytest.raises(SqgraphError):
        q(
            db,
            "CREATE VECTOR INDEX entity_emb FOR (n:Entity) ON (n.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 4}}",
        )
    q(
        db,
        "CREATE VECTOR INDEX eu FOR (n:Chunk) ON (n.vec) "
        "OPTIONS {indexConfig: {`vector.dimensions`: $d, `vector.similarity_function`: 'euclidean'}}",
        d=8,
    )
    eu = next(i for i in db.vector_indexes() if i.name == "eu")
    assert (eu.dimensions, eu.similarity, eu.property) == (8, "euclidean", "vec")


def test_vector_ddl_requires_dimensions_and_valid_name(db):
    with pytest.raises(CypherRuntimeError, match="dimensions"):
        q(db, "CREATE VECTOR INDEX v FOR (n:E) ON (n.embedding)")
    with pytest.raises(SqgraphError):
        q(db, "CREATE VECTOR INDEX `bad name` FOR (n:E) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3}}")
    with pytest.raises(SqgraphError):
        q(db, "CREATE VECTOR INDEX v FOR (n:E) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'dot'}}")
    with pytest.raises(CypherSyntaxError):
        q(db, "CREATE VECTOR INDEX v FOR (n:E) ON (n.embedding")
    with pytest.raises(CypherRuntimeError, match="dimensions"):
        q(db, "CREATE VECTOR INDEX v FOR (n:E) ON n.embedding")  # parentheses are optional, dimensions are not
    assert db.vector_indexes() == []


def test_drop_index_cypher(db):
    db.create_vector_index("v", "E", "embedding", 2)
    q(db, "DROP INDEX v")
    assert db.vector_indexes() == []
    assert db._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'vec_v'").fetchone()[0] == 0
    q(db, "DROP INDEX v IF EXISTS")
    with pytest.raises(SqgraphError):
        q(db, "DROP INDEX v")


def test_show_indexes(db):
    db.create_vector_index("v", "E", "embedding", 2)
    db.create_fulltext_index("f", ["E"], ["name"])
    rows = q(db, "SHOW INDEXES")
    assert {(r["name"], r["type"]) for r in rows} == {("v", "VECTOR"), ("f", "FULLTEXT")}


# ------------------------------------------------------------------ storing vectors


def test_embeddings_are_not_stored_in_props(vdb):
    for row in vdb._conn.execute("SELECT props FROM nodes"):
        assert "embedding" not in row[0]
    assert vec_rows(vdb) == 3
    assert vdb.get_node(vdb.ids["x"]).properties == {"name": "x"}


def test_set_node_vector_api_replaces(vdb):
    vdb.set_node_vector(vdb.ids["x"], "embedding", [0.0, 1.0])
    assert vec_rows(vdb) == 3
    top = vdb.vector_search("emb", [0.0, 1.0], k=3)
    assert {n["name"] for n, s in top if s == pytest.approx(1.0)} == {"x", "y"}


def test_set_node_vector_errors(vdb):
    with pytest.raises(DimensionMismatchError):
        vdb.set_node_vector(vdb.ids["x"], "embedding", [1.0, 2.0, 3.0])
    with pytest.raises(SqgraphError, match="no vector index"):
        vdb.set_node_vector(vdb.ids["x"], "other_prop", [1.0, 2.0])
    other = vdb.create_node("Unindexed")
    with pytest.raises(SqgraphError, match="no vector index"):
        vdb.set_node_vector(other.id, "embedding", [1.0, 2.0])
    with pytest.raises(SqgraphError):
        vdb.set_node_vector(999999, "embedding", [1.0, 2.0])
    with pytest.raises(SqgraphError):
        vdb.set_node_vector(vdb.ids["x"], "embedding", "not a vector")
    assert vec_rows(vdb) == 3


def test_set_vector_via_cypher_procedure(db):
    db.create_vector_index("emb", "Entity", "embedding", 3)
    q(db, "CREATE (:Entity {name: 'a'}), (:Entity {name: 'b'})")
    q(db, "MATCH (n:Entity {name: 'a'}) CALL db.create.setNodeVectorProperty(n, 'embedding', [1.0, 0.0, 0.0])")
    q(db, "MATCH (n:Entity {name: 'b'}) CALL db.create.setNodeVectorProperty(n, 'embedding', $v)", v=[0.0, 1.0, 0.0])
    assert vec_rows(db) == 2
    rows = q(db, "CALL db.index.vector.queryNodes('emb', 1, [1.0, 0.0, 0.0]) YIELD node, score RETURN node.name AS n, score")
    assert rows == [{"n": "a", "score": pytest.approx(1.0)}]
    # UNWIND-driven bulk assignment
    q(
        db,
        "UNWIND $rows AS row MATCH (n:Entity {name: row.name}) "
        "CALL db.create.setNodeVectorProperty(n, 'embedding', row.vec)",
        rows=[{"name": "a", "vec": [0.0, 0.0, 1.0]}, {"name": "b", "vec": [0.0, 1.0, 1.0]}],
    )
    rows = q(db, "CALL db.index.vector.queryNodes('emb', 1, [0.0, 0.0, 1.0]) YIELD node, score RETURN node.name AS n")
    assert rows == [{"n": "a"}]
    assert vec_rows(db) == 2


def test_set_vector_procedure_errors_roll_back(db):
    db.create_vector_index("emb", "Entity", "embedding", 3)
    q(db, "CREATE (:Entity {name: 'a'})")
    with pytest.raises(DimensionMismatchError):
        q(db, "MATCH (n:Entity) CALL db.create.setNodeVectorProperty(n, 'embedding', [1.0, 2.0])")
    with pytest.raises(SqgraphError):
        q(db, "MATCH (n:Entity) CALL db.create.setNodeVectorProperty(n, 'nope', [1.0, 2.0, 3.0])")
    with pytest.raises(CypherRuntimeError):
        q(db, "MATCH (n:Entity) CALL db.create.setNodeVectorProperty(n, 'embedding', 'abc')")
    assert vec_rows(db) == 0


def test_embedding_in_create_and_set_is_routed_to_vector_table(db):
    db.create_vector_index("emb", "Entity", "embedding", 2)
    q(db, "CREATE (n:Entity {name: 'a', embedding: [1.0, 0.0]})")
    q(db, "CREATE (n:Entity {name: 'b'})")
    q(db, "MATCH (n:Entity {name: 'b'}) SET n.embedding = [0.0, 1.0]")
    assert vec_rows(db) == 2
    assert all("embedding" not in r[0] for r in db._conn.execute("SELECT props FROM nodes"))
    rows = q(db, "CALL db.index.vector.queryNodes('emb', 1, [0.0, 1.0]) YIELD node RETURN node.name AS n")
    assert rows == [{"n": "b"}]
    node = db.create_node("Entity", {"name": "c", "embedding": [1.0, 1.0]})
    assert "embedding" not in node.properties and vec_rows(db) == 3


def test_creating_an_index_migrates_existing_embedding_properties(db):
    a = db.create_node("Doc", {"t": "a", "embedding": [1.0, 0.0]})
    db.create_node("Doc", {"t": "no embedding"})
    db.create_vector_index("late", "Doc", "embedding", 2)
    assert vec_rows(db, "late") == 1
    assert "embedding" not in db.get_node(a.id).properties
    (hit, score), = db.vector_search("late", [1.0, 0.0], k=5)
    assert hit == a and score == pytest.approx(1.0)


def test_numpy_arrays_accepted(vdb):
    np = pytest.importorskip("numpy")
    vdb.set_node_vector(vdb.ids["x"], "embedding", np.array([0.5, 0.5], dtype=np.float32))
    (hit, score), *_ = vdb.vector_search("emb", np.array([1.0, 1.0]), k=3)
    assert score == pytest.approx(1.0, abs=1e-6)
    res = q(vdb, "CALL db.index.vector.queryNodes('emb', 3, $v) YIELD node RETURN count(*) AS c", v=np.array([1.0, 1.0]))
    assert res == [{"c": 3}]


# ------------------------------------------------------------------ search: ranking and score convention


def test_vector_search_cosine_scores(vdb):
    res = vdb.vector_search("emb", [1.0, 0.0], k=3)
    assert [n["name"] for n, _ in res] == ["x", "d", "y"]
    scores = [s for _, s in res]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx((1 + math.sqrt(0.5)) / 2)
    assert scores[2] == pytest.approx(0.5)
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(n, Node) for n, _ in res)
    assert len(vdb.vector_search("emb", [1.0, 0.0], k=2)) == 2
    assert len(vdb.vector_search("emb", [1.0, 0.0], k=50)) == 3


def test_vector_search_euclidean_scores(db):
    db.create_vector_index("eu", "P", "embedding", 2, "euclidean")
    for name, v in (("o", [0.0, 0.0]), ("a", [1.0, 0.0]), ("b", [3.0, 4.0])):
        db.create_node("P", {"name": name, "embedding": v})
    res = db.vector_search("eu", [0.0, 0.0], k=3)
    assert [n["name"] for n, _ in res] == ["o", "a", "b"]
    assert [s for _, s in res] == pytest.approx([1.0, 0.5, 1 / 26])
    rows = q(db, "CALL db.index.vector.queryNodes('eu', 3, [0.0, 0.0]) YIELD node, score RETURN node.name AS n, score")
    assert [r["n"] for r in rows] == ["o", "a", "b"]
    assert [r["score"] for r in rows] == pytest.approx([1.0, 0.5, 1 / 26])


def test_cypher_query_nodes_matches_python_api(vdb):
    api = vdb.vector_search("emb", [1.0, 0.2], k=3)
    rows = q(vdb, "CALL db.index.vector.queryNodes('emb', 3, $q) YIELD node, score RETURN node, score", q=[1.0, 0.2])
    assert [r["node"] for r in rows] == [n for n, _ in api]
    assert [r["score"] for r in rows] == pytest.approx([s for _, s in api])
    assert [r["node"]["name"] for r in rows] == ["x", "d", "y"]


def test_query_nodes_yield_alias_where_order_limit(vdb):
    rows = q(
        vdb,
        "CALL db.index.vector.queryNodes('emb', 3, [1.0, 0.0]) YIELD node AS hit, score AS s "
        "WHERE s > 0.6 RETURN hit.name AS name, s ORDER BY s DESC",
    )
    assert [r["name"] for r in rows] == ["x", "d"]
    rows = q(vdb, "CALL db.index.vector.queryNodes('emb', 3, [1.0, 0.0]) YIELD node, score RETURN node.name AS n ORDER BY score ASC LIMIT 1")
    assert rows == [{"n": "y"}]


def test_query_nodes_with_k_and_vector_parameters(vdb):
    rows = q(vdb, "CALL db.index.vector.queryNodes($idx, $k, $v) YIELD node RETURN node.name AS n", idx="emb", k=1, v=[0.0, 1.0])
    assert rows == [{"n": "y"}]


def test_query_nodes_errors(vdb):
    with pytest.raises(CypherRuntimeError, match="no such vector"):
        q(vdb, "CALL db.index.vector.queryNodes('nope', 3, [1.0, 0.0]) YIELD node RETURN node")
    with pytest.raises(DimensionMismatchError):
        q(vdb, "CALL db.index.vector.queryNodes('emb', 3, [1.0, 0.0, 0.5]) YIELD node RETURN node")
    with pytest.raises(DimensionMismatchError):
        vdb.vector_search("emb", [1.0])
    with pytest.raises(SqgraphError, match="no such vector"):
        vdb.vector_search("nope", [1.0, 0.0])
    with pytest.raises(CypherRuntimeError):
        q(vdb, "CALL db.index.vector.queryNodes('emb', 0, [1.0, 0.0]) YIELD node RETURN node")
    with pytest.raises(CypherSyntaxError):
        q(vdb, "CALL db.index.vector.queryNodes('emb', 3, [1.0, 0.0]) YIELD nope RETURN nope")
    with pytest.raises(CypherSyntaxError):
        q(vdb, "MATCH (n) CALL db.index.vector.queryNodes('emb', 3, [1.0, 0.0])")
    with pytest.raises(SqgraphError):
        vdb.vector_search("emb", [1.0, 0.0], k=0)


def test_call_combined_with_match_graphrag_local_search(vdb):
    ids = vdb.ids
    vdb.create_node("Chunk", {"text": "about x"})
    vdb.create_node("Chunk", {"text": "about y"})
    c = {r["text"]: r["i"] for r in vdb.query("MATCH (c:Chunk) RETURN c.text AS text, id(c) AS i")}
    vdb.create_edge(c["about x"], ids["x"], "MENTIONS")
    vdb.create_edge(c["about y"], ids["y"], "MENTIONS")
    vdb.create_edge(ids["x"], ids["d"], "RELATED_TO", {"w": 3})
    rows = q(
        vdb,
        "CALL db.index.vector.queryNodes('emb', 1, [1.0, 0.0]) YIELD node, score "
        "MATCH (node)-[r]-(nb) "
        "RETURN node.name AS node, type(r) AS rel, coalesce(nb.name, nb.text) AS nb, score ORDER BY nb",
    )
    assert rows == [
        {"node": "x", "rel": "MENTIONS", "nb": "about x", "score": pytest.approx(1.0)},
        {"node": "x", "rel": "RELATED_TO", "nb": "d", "score": pytest.approx(1.0)},
    ]
    rows = q(
        vdb,
        "CALL db.index.vector.queryNodes('emb', 2, [1.0, 0.0]) YIELD node, score "
        "OPTIONAL MATCH (node)<-[:MENTIONS]-(c:Chunk) "
        "RETURN node.name AS n, collect(c.text) AS chunks, score ORDER BY score DESC",
    )
    assert [r["n"] for r in rows] == ["x", "d"]
    assert rows[0]["chunks"] == ["about x"] and rows[1]["chunks"] == []


def test_call_after_match_and_where_on_matched_nodes(vdb):
    rows = q(
        vdb,
        "MATCH (seed:Entity {name: 'y'}) "
        "CALL db.index.vector.queryNodes('emb', 3, [0.0, 1.0]) YIELD node, score "
        "WHERE node <> seed RETURN seed.name AS s, node.name AS n ORDER BY score DESC",
    )
    assert rows == [{"s": "y", "n": "d"}, {"s": "y", "n": "x"}]


def test_query_nodes_only_returns_indexed_label(vdb):
    other = vdb.create_node("Other", {"name": "z", "embedding": [1.0, 0.0]})
    res = vdb.vector_search("emb", [1.0, 0.0], k=10)
    assert other not in [n for n, _ in res]
    assert len(res) == 3


# ------------------------------------------------------------------ lifecycle


def test_vectors_removed_when_node_deleted_api(vdb):
    vdb.delete_node(vdb.ids["x"])
    assert vec_rows(vdb) == 2
    assert vdb.ids["x"] not in [n.id for n, _ in vdb.vector_search("emb", [1.0, 0.0], k=10)]


def test_vectors_removed_when_node_deleted_cypher(vdb):
    q(vdb, "MATCH (n:Entity {name: 'y'}) DETACH DELETE n")
    assert vec_rows(vdb) == 2
    q(vdb, "MATCH (n:Entity) DELETE n")
    assert vec_rows(vdb) == 0
    assert vdb.vector_search("emb", [1.0, 0.0]) == []


def test_vector_removed_when_label_removed(vdb):
    q(vdb, "MATCH (n:Entity {name: 'x'}) REMOVE n:Entity")
    assert vec_rows(vdb) == 2


def test_deleting_node_id_reuse_does_not_leak_vectors(db):
    db.create_vector_index("emb", "E", "embedding", 2)
    n = db.create_node("E", {"embedding": [1.0, 0.0]})
    db.delete_node(n.id)
    m = db.create_node("E", {"name": "new"})  # may reuse the id
    assert vec_rows(db) == 0
    assert db.vector_search("emb", [1.0, 0.0]) == []
    assert m.properties == {"name": "new"}


def test_vector_rollback_with_transaction(vdb):
    with pytest.raises(RuntimeError):
        with vdb.transaction():
            vdb.set_node_vector(vdb.ids["x"], "embedding", [0.0, 1.0])
            vdb.create_node("Entity", {"name": "new", "embedding": [0.3, 0.3]})
            raise RuntimeError
    assert vec_rows(vdb) == 3
    top = vdb.vector_search("emb", [1.0, 0.0], k=1)
    assert top[0][0]["name"] == "x"


def test_vector_index_persists_and_works_read_only(tmp_path):
    path = tmp_path / "v.db"
    with GraphDB(path) as db:
        db.create_vector_index("emb", "E", "embedding", 2)
        db.create_node("E", {"name": "a", "embedding": [1.0, 0.0]})
        db.create_node("E", {"name": "b", "embedding": [0.0, 1.0]})
    with GraphDB(path) as db:
        assert [i.name for i in db.vector_indexes()] == ["emb"]
        assert db.vector_search("emb", [0.0, 1.0], k=1)[0][0]["name"] == "b"
        db.create_node("E", {"name": "c", "embedding": [1.0, 1.0]})
    with GraphDB(path, read_only=True) as ro:
        assert ro.vector_search("emb", [1.0, 1.0], k=1)[0][0]["name"] == "c"
        rows = ro.query("CALL db.index.vector.queryNodes('emb', 2, [1.0, 0.0]) YIELD node RETURN node.name AS n")
        assert [r["n"] for r in rows] == ["a", "c"]


def test_hostile_property_names_do_not_reach_ddl(db):
    db.create_vector_index("emb", "E", "embedding; DROP TABLE nodes", 2)
    n = db.create_node("E", {"x": 1})
    db.set_node_vector(n.id, "embedding; DROP TABLE nodes", [1.0, 0.0])
    assert db.stats()["nodes"] == 1
    assert vec_rows(db) == 1


def test_many_vectors_bulk_search(db):
    db.create_vector_index("emb", "E", "embedding", 8)
    import random

    rnd = random.Random(1)
    vecs = [[rnd.random() for _ in range(8)] for _ in range(300)]
    ids = db.create_nodes([("E", {"i": i, "embedding": v}) for i, v in enumerate(vecs)])
    assert len(ids) == 300 and vec_rows(db) == 300

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    query = [rnd.random() for _ in range(8)]
    expect = sorted(range(300), key=lambda i: -cos(vecs[i], query))[:5]
    got = [n["i"] for n, _ in db.vector_search("emb", query, k=5)]
    assert got == expect
