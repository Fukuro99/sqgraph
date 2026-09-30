"""Vector, fulltext and property indexes over relationships (Neo4j 5 style)."""

from __future__ import annotations

import json
import math
import threading
import urllib.request

import pytest

from sqgraph import (
    CypherRuntimeError,
    CypherUnsupportedError,
    DimensionMismatchError,
    FulltextIndex,
    GraphDB,
    Node,
    Relationship,
    SqgraphError,
    VectorIndex,
)
from sqgraph.cypher.compiler import Compiler
from sqgraph.cypher.parser import parse
from sqgraph.viewer.server import make_server


def q(db, cypher, **params):
    return db.query(cypher, params)


def count(db, table):
    return db._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def edge_ids(db, index):
    """Sorted entity ids held by a vec_/fts_ table."""
    col = db._conn.execute(f"PRAGMA table_info({index})").fetchone()[1] if index.startswith("vec_") else "rowid"
    return sorted(r[0] for r in db._conn.execute(f"SELECT {col} FROM {index}"))


@pytest.fixture
def fdb(db):
    """Two people and two FACT edges with 3-d embeddings, a vector and a trigram fulltext index over FACT."""
    db.create_relationship_vector_index("fact_emb", "FACT", "embedding", 3)
    db.create_relationship_fulltext_index("fact_ft", "FACT", ["fact"], tokenizer="trigram")
    a = db.create_node("Person", {"name": "Alice"})
    b = db.create_node("Person", {"name": "Bob"})
    c = db.create_node("Person", {"name": "Carol"})
    e1 = db.create_edge(a.id, b.id, "FACT", {"id": "f1", "fact": "Alice manages the Rails migration", "embedding": [1, 0, 0]})
    e2 = db.create_edge(b.id, c.id, "FACT", {"id": "f2", "fact": "Bob reviews Carol's pull requests", "embedding": [0, 1, 0]})
    db.ids = {"a": a.id, "b": b.id, "c": c.id, "e1": e1.id, "e2": e2.id}
    return db


# ------------------------------------------------------------------ metadata


def test_index_metadata_and_registries(fdb):
    vx = fdb.vector_indexes()
    fx = fdb.fulltext_indexes()
    assert vx == [VectorIndex("fact_emb", "FACT", "embedding", 3, "cosine", "relationship")]
    assert fx == [FulltextIndex("fact_ft", ("FACT",), ("fact",), "trigram", "relationship")]
    assert vx[0].key_column == "edge_id" and fx[0].key_column == "edge_id"
    assert fdb._conn.execute("SELECT entity_type FROM _vector_indexes").fetchone() == ("relationship",)
    assert fdb._conn.execute("SELECT entity_type FROM _fulltext_indexes").fetchone() == ("relationship",)
    # the backing tables are keyed by edge id
    assert [r[1] for r in fdb._conn.execute("PRAGMA table_info(vec_fact_emb)")][:1] == ["edge_id"]
    node_ix = fdb.create_vector_index("n_emb", "Person", "embedding", 3)
    assert node_ix.entity_type == "node" and node_ix.key_column == "node_id"


def test_list_indexes_reports_entity_type(fdb):
    fdb.create_vector_index("n_emb", "Person", "embedding", 3)
    fdb.create_fulltext_index("n_ft", "Person", "name")
    fdb.create_property_index("Person", "name")
    fdb.create_relationship_property_index("FACT", "id")
    by_name = {i["name"]: i for i in fdb.list_indexes()}
    assert {n: (i["type"], i["entityType"]) for n, i in by_name.items() if not n.startswith(("prop_", "rprop_"))} == {
        "fact_emb": ("VECTOR", "RELATIONSHIP"),
        "fact_ft": ("FULLTEXT", "RELATIONSHIP"),
        "n_emb": ("VECTOR", "NODE"),
        "n_ft": ("FULLTEXT", "NODE"),
    }
    assert by_name["fact_emb"]["labelsOrTypes"] == ["FACT"] and by_name["fact_emb"]["properties"] == ["embedding"]
    assert sorted(i["entityType"] for i in fdb.list_indexes() if i["type"] == "RANGE") == ["NODE", "RELATIONSHIP"]
    rows = q(fdb, "SHOW INDEXES")
    assert {(r["name"], r["entityType"]) for r in rows if r["type"] != "RANGE"} == {
        ("fact_emb", "RELATIONSHIP"), ("fact_ft", "RELATIONSHIP"), ("n_emb", "NODE"), ("n_ft", "NODE"),
    }


def test_index_names_are_unique_across_kinds(fdb):
    with pytest.raises(SqgraphError, match="already exists"):
        fdb.create_vector_index("fact_emb", "Person", "embedding", 3)
    with pytest.raises(SqgraphError, match="already exists"):
        fdb.create_relationship_vector_index("fact_emb", "FACT", "embedding", 3)
    with pytest.raises(SqgraphError, match="already exists"):
        fdb.create_fulltext_index("fact_ft", "Person", "name")
    assert fdb.create_relationship_vector_index("fact_emb", "FACT", "embedding", 3, if_not_exists=True).name == "fact_emb"
    assert fdb.create_relationship_fulltext_index("fact_ft", "FACT", "fact", if_not_exists=True).name == "fact_ft"
    # if_not_exists must not hand back an index of the other kind
    with pytest.raises(SqgraphError, match="already exists"):
        fdb.create_vector_index("fact_emb", "Person", "embedding", 3, if_not_exists=True)
    with pytest.raises(SqgraphError, match="already exists"):
        fdb.create_fulltext_index("fact_ft", "Person", "name", if_not_exists=True)


def test_relationship_index_validation(db):
    for bad in ("a b", "x;y", "", "1a"):
        with pytest.raises(SqgraphError):
            db.create_relationship_vector_index(bad, "T", "embedding", 3)
        with pytest.raises(SqgraphError):
            db.create_relationship_fulltext_index(bad, "T", "p")
    with pytest.raises(SqgraphError):
        db.create_relationship_vector_index("v", "T", "embedding", 0)
    with pytest.raises(SqgraphError):
        db.create_relationship_vector_index("v", "", "embedding", 3)
    with pytest.raises(SqgraphError):
        db.create_relationship_vector_index("v", "T", "embedding", 3, similarity="dot")
    with pytest.raises(SqgraphError, match="tokenizer"):
        db.create_relationship_fulltext_index("f", "T", "p", tokenizer="nope")
    with pytest.raises(SqgraphError):
        db.create_relationship_fulltext_index("f", [], "p")
    assert db.vector_indexes() == [] and db.fulltext_indexes() == []
    assert db.create_relationship_fulltext_index("f", ["A", "B"], ["x", "y"], tokenizer="porter").labels == ("A", "B")


def test_read_only_database_rejects_relationship_index_creation(tmp_path):
    path = tmp_path / "r.db"
    GraphDB(path).close()
    with GraphDB(path, read_only=True) as ro:
        with pytest.raises(SqgraphError):
            ro.create_relationship_vector_index("v", "T", "embedding", 3)
        with pytest.raises(SqgraphError):
            ro.create_relationship_fulltext_index("f", "T", "p")
        with pytest.raises(SqgraphError):
            ro.create_relationship_property_index("T", "p")
        with pytest.raises(SqgraphError):
            ro.set_edge_vector(1, "embedding", [1, 2, 3])


# ------------------------------------------------------------------ vector: Python API


def test_vector_search_returns_relationships(fdb):
    hits = fdb.vector_search("fact_emb", [1, 0.1, 0], k=5)
    assert [type(h[0]) for h in hits] == [Relationship, Relationship]
    assert [h[0].id for h in hits] == [fdb.ids["e1"], fdb.ids["e2"]]
    assert hits[0][0].start_node_id == fdb.ids["a"] and hits[0][0].end_node_id == fdb.ids["b"]
    assert hits[0][0]["id"] == "f1"
    cos = 1 / math.sqrt(1.01)
    assert hits[0][1] == pytest.approx((1 + cos) / 2, abs=1e-5)
    assert hits[0][1] > hits[1][1]
    assert fdb.vector_search("fact_emb", [1, 0, 0], k=1)[0][0].id == fdb.ids["e1"]


def test_euclidean_relationship_index_score(db):
    db.create_relationship_vector_index("e", "T", "v", 2, similarity="euclidean")
    a, b = db.create_node("N").id, db.create_node("N").id
    eid = db.create_edge(a, b, "T", {"v": [0.0, 0.0]}).id
    hits = db.vector_search("e", [3.0, 4.0], k=1)
    assert hits[0][0].id == eid and hits[0][1] == pytest.approx(1 / (1 + 25.0), abs=1e-5)


def test_embedding_is_not_stored_in_edge_props(fdb):
    for (props,) in fdb._conn.execute("SELECT props FROM edges"):
        assert "embedding" not in props
    assert fdb.get_edge(fdb.ids["e1"]).properties == {"id": "f1", "fact": "Alice manages the Rails migration"}
    assert count(fdb, "vec_fact_emb") == 2
    assert q(fdb, "MATCH ()-[r:FACT]->() RETURN r.embedding AS e") == [{"e": None}, {"e": None}]
    assert "embedding" not in q(fdb, "MATCH ()-[r:FACT {id: 'f1'}]->() RETURN properties(r) AS p")[0]["p"]
    assert "embedding" not in q(fdb, "MATCH ()-[r:FACT {id: 'f1'}]->() RETURN r")[0]["r"].properties


def test_set_edge_vector(fdb):
    e3 = fdb.create_edge(fdb.ids["a"], fdb.ids["c"], "FACT", {"id": "f3", "fact": "x"})
    assert count(fdb, "vec_fact_emb") == 2
    fdb.set_edge_vector(e3.id, "embedding", [0.0, 0.0, 1.0])
    assert count(fdb, "vec_fact_emb") == 3
    assert fdb.vector_search("fact_emb", [0, 0, 1], k=1)[0][0].id == e3.id
    fdb.set_edge_vector(e3.id, "embedding", [1.0, 0.0, 0.0])  # upsert
    assert count(fdb, "vec_fact_emb") == 3
    assert fdb.get_edge(e3.id).properties == {"id": "f3", "fact": "x"}
    with pytest.raises(DimensionMismatchError):
        fdb.set_edge_vector(e3.id, "embedding", [1.0, 2.0])
    with pytest.raises(SqgraphError, match="does not exist"):
        fdb.set_edge_vector(9999, "embedding", [1, 2, 3])
    with pytest.raises(SqgraphError, match="no vector index covers"):
        fdb.set_edge_vector(e3.id, "other", [1, 2, 3])
    other = fdb.create_edge(fdb.ids["a"], fdb.ids["c"], "OTHER")
    with pytest.raises(SqgraphError, match="no vector index covers"):
        fdb.set_edge_vector(other.id, "embedding", [1, 2, 3])
    with pytest.raises(SqgraphError, match="does not exist"):  # a node id is not an edge
        fdb.set_edge_vector(0, "embedding", [1, 2, 3])


def test_create_edge_only_touches_matching_type_and_property(fdb):
    other = fdb.create_edge(fdb.ids["a"], fdb.ids["b"], "OTHER", {"embedding": [1, 2, 3, 4, 5]})
    assert fdb.get_edge(other.id)["embedding"] == [1, 2, 3, 4, 5]  # not indexed: stays a plain property
    plain = fdb.create_edge(fdb.ids["a"], fdb.ids["b"], "FACT", {"fact": "no embedding"})
    assert count(fdb, "vec_fact_emb") == 2
    assert plain.id not in [h[0].id for h in fdb.vector_search("fact_emb", [1, 1, 1], k=10)]


def test_create_edge_dimension_mismatch_rolls_back(fdb):
    before = count(fdb, "edges")
    with pytest.raises(DimensionMismatchError):
        fdb.create_edge(fdb.ids["a"], fdb.ids["b"], "FACT", {"fact": "bad", "embedding": [1, 2]})
    assert count(fdb, "edges") == before and count(fdb, "vec_fact_emb") == 2 and count(fdb, "fts_fact_ft") == 2
    with pytest.raises(DimensionMismatchError):
        q(fdb, "MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'}) CREATE (a)-[:FACT {fact:'bad', embedding:[1.0]}]->(b)")
    assert count(fdb, "edges") == before and count(fdb, "vec_fact_emb") == 2


def test_create_edges_bulk_syncs_indexes(fdb):
    a, b = fdb.ids["a"], fdb.ids["b"]
    ids = fdb.create_edges(
        [
            (a, b, "FACT", {"fact": "bulk one", "embedding": [0.0, 0.0, 1.0]}),
            (b, a, "FACT", {"fact": "bulk two", "embedding": [0.0, 1.0, 1.0]}),
            (a, b, "OTHER", {"fact": "bulk three"}),
            (a, b, "FACT"),
        ]
    )
    assert len(ids) == 4
    assert count(fdb, "vec_fact_emb") == 4
    assert edge_ids(fdb, "fts_fact_ft") == sorted([fdb.ids["e1"], fdb.ids["e2"], ids[0], ids[1]])
    assert fdb.vector_search("fact_emb", [0, 0, 1], k=1)[0][0].id == ids[0]
    assert fdb.fulltext_search("fact_ft", "bulk two")[0][0].id == ids[1]
    assert "embedding" not in fdb.get_edge(ids[0]).properties
    with pytest.raises(SqgraphError):
        fdb.create_edges([(a, 99999, "FACT", {"embedding": [1, 2, 3]})])
    assert count(fdb, "vec_fact_emb") == 4


def test_update_edge_keeps_vector_in_sync(fdb):
    e1 = fdb.ids["e1"]
    fdb.update_edge(e1, {"embedding": [0.0, 0.0, 1.0], "note": "moved"})
    assert count(fdb, "vec_fact_emb") == 2
    assert fdb.vector_search("fact_emb", [0, 0, 1], k=1)[0][0].id == e1
    assert fdb.get_edge(e1).properties == {"id": "f1", "fact": "Alice manages the Rails migration", "note": "moved"}
    fdb.update_edge(e1, {"embedding": None})  # removal
    assert count(fdb, "vec_fact_emb") == 1
    assert e1 not in [h[0].id for h in fdb.vector_search("fact_emb", [0, 0, 1], k=5)]
    fdb.update_edge(e1, {"embedding": [1.0, 0.0, 0.0]})
    fdb.update_edge(e1, {"id": "f1b"}, replace=True)  # replace drops the embedding and the rest
    assert count(fdb, "vec_fact_emb") == 1
    assert fdb.get_edge(e1).properties == {"id": "f1b"}
    fdb.update_edge(e1, {"embedding": [1, 1, 1], "id": "f1c"}, replace=True)  # ...unless it is re-supplied
    assert count(fdb, "vec_fact_emb") == 2
    with pytest.raises(DimensionMismatchError):
        fdb.update_edge(e1, {"embedding": [1, 2]})
    assert count(fdb, "vec_fact_emb") == 2


# ------------------------------------------------------------------ fulltext: Python API


def test_fulltext_search_returns_relationships(fdb):
    hits = fdb.fulltext_search("fact_ft", "Rails")
    assert [type(h[0]) for h in hits] == [Relationship]
    assert hits[0][0].id == fdb.ids["e1"] and hits[0][1] > 0
    assert fdb.fulltext_search("fact_ft", "pull requests")[0][0].id == fdb.ids["e2"]
    assert fdb.fulltext_search("fact_ft", "nonexistent-term") == []
    assert len(fdb.fulltext_search("fact_ft", "Carol OR Alice", k=1)) == 1


def test_fulltext_trigram_japanese(db):
    db.create_relationship_fulltext_index("ja", "FACT", "fact", tokenizer="trigram")
    a, b = db.create_node("E").id, db.create_node("E").id
    e1 = db.create_edge(a, b, "FACT", {"fact": "来週から認証基盤の担当は Bob に移ります"}).id
    db.create_edge(a, b, "FACT", {"fact": "9/14-9/16 は休みです"})
    assert [h[0].id for h in db.fulltext_search("ja", "認証基盤")] == [e1]
    assert [h[0].id for h in db.fulltext_search("ja", "基盤の担当")] == [e1]
    assert db.fulltext_search("ja", "休みです")[0][0]["fact"].startswith("9/14")


def test_fulltext_backfills_existing_relationships(db):
    a, b = db.create_node("N").id, db.create_node("N").id
    e1 = db.create_edge(a, b, "KNOWS", {"note": "met at the conference"}).id
    e2 = db.create_edge(a, b, "LIKES", {"note": "conference buddy"}).id
    db.create_edge(a, b, "OTHER", {"note": "conference again"})
    db.create_edge(a, b, "KNOWS", {"since": 2020})  # no indexed property: no row
    db.create_relationship_fulltext_index("rel_ft", ["KNOWS", "LIKES"], ["note"])
    assert edge_ids(db, "fts_rel_ft") == sorted([e1, e2])
    assert {h[0].id for h in db.fulltext_search("rel_ft", "conference")} == {e1, e2}


def test_fulltext_multi_property_and_multi_type(db):
    db.create_relationship_fulltext_index("ft", ["A", "B"], ["title", "body"])
    n1, n2 = db.create_node("N").id, db.create_node("N").id
    e1 = db.create_edge(n1, n2, "A", {"title": "alpha"}).id
    e2 = db.create_edge(n1, n2, "B", {"body": "alpha beta"}).id
    db.create_edge(n1, n2, "C", {"title": "alpha"})
    assert {h[0].id for h in db.fulltext_search("ft", "alpha")} == {e1, e2}
    assert [h[0].id for h in db.fulltext_search("ft", "body:beta")] == [e2]
    assert [h[0].id for h in db.fulltext_search("ft", "title:beta")] == []


def test_update_edge_keeps_fulltext_in_sync(fdb):
    e1 = fdb.ids["e1"]
    fdb.update_edge(e1, {"fact": "Alice leads the Django rewrite"})
    assert fdb.fulltext_search("fact_ft", "Rails") == []
    assert fdb.fulltext_search("fact_ft", "Django")[0][0].id == e1
    fdb.update_edge(e1, {"fact": None})
    assert edge_ids(fdb, "fts_fact_ft") == [fdb.ids["e2"]]
    fdb.update_edge(e1, {"fact": "back again"})
    assert edge_ids(fdb, "fts_fact_ft") == sorted([e1, fdb.ids["e2"]])
    fdb.update_edge(e1, {"id": "only-id"}, replace=True)
    assert edge_ids(fdb, "fts_fact_ft") == [fdb.ids["e2"]]


def test_delete_edge_removes_index_rows(fdb):
    assert fdb.delete_edge(fdb.ids["e1"]) is True
    assert edge_ids(fdb, "fts_fact_ft") == [fdb.ids["e2"]] and edge_ids(fdb, "vec_fact_emb") == [fdb.ids["e2"]]
    assert [h[0].id for h in fdb.vector_search("fact_emb", [1, 0, 0], k=5)] == [fdb.ids["e2"]]
    assert fdb.fulltext_search("fact_ft", "Rails") == []
    assert fdb.delete_edge(fdb.ids["e1"]) is False


def test_detach_delete_node_removes_edge_index_rows(fdb):
    extra = fdb.create_edge(fdb.ids["c"], fdb.ids["a"], "FACT", {"fact": "Carol pings Alice", "embedding": [0, 0, 1]}).id
    assert count(fdb, "vec_fact_emb") == 3
    with pytest.raises(SqgraphError):
        fdb.delete_node(fdb.ids["a"])  # refused: nothing may change
    assert count(fdb, "vec_fact_emb") == 3 and count(fdb, "fts_fact_ft") == 3
    assert fdb.delete_node(fdb.ids["a"], detach=True) is True
    assert edge_ids(fdb, "vec_fact_emb") == [fdb.ids["e2"]] and edge_ids(fdb, "fts_fact_ft") == [fdb.ids["e2"]]
    assert extra not in [h[0].id for h in fdb.vector_search("fact_emb", [0, 0, 1], k=10)]
    assert fdb.fulltext_search("fact_ft", "pings") == []
    assert fdb.get_edge(fdb.ids["e1"]) is None and fdb.get_edge(extra) is None
    # a later edge reusing the freed rowid never resurrects the old rows
    e_new = fdb.create_edge(fdb.ids["b"], fdb.ids["c"], "FACT", {"fact": "fresh", "embedding": [1, 1, 1]}).id
    assert [h[0].id for h in fdb.fulltext_search("fact_ft", "fresh")] == [e_new]
    assert count(fdb, "vec_fact_emb") == 2


def test_cypher_delete_relationship_and_detach_delete(fdb):
    q(fdb, "MATCH ()-[r:FACT {id: 'f1'}]->() DELETE r")
    assert edge_ids(fdb, "vec_fact_emb") == [fdb.ids["e2"]] and edge_ids(fdb, "fts_fact_ft") == [fdb.ids["e2"]]
    q(fdb, "MATCH (n:Person {name: 'Carol'}) DETACH DELETE n")
    assert count(fdb, "vec_fact_emb") == 0 and count(fdb, "fts_fact_ft") == 0 and count(fdb, "edges") == 0


def test_node_operations_do_not_touch_relationship_indexes(fdb):
    fdb.create_vector_index("p_emb", "Person", "embedding", 3)
    fdb.create_fulltext_index("p_ft", "Person", "name")
    # same numeric ids on both sides: node 1 / edge 1 must not be confused
    fdb.set_node_vector(fdb.ids["a"], "embedding", [0, 1, 0])
    fdb.update_node(fdb.ids["a"], {"name": "Alicia"})
    fdb.add_labels(fdb.ids["a"], "Extra")
    fdb.remove_labels(fdb.ids["a"], "Person")
    assert count(fdb, "vec_fact_emb") == 2 and count(fdb, "fts_fact_ft") == 2
    assert edge_ids(fdb, "vec_p_emb") == []  # node "Person" label removed -> embedding dropped
    fdb.delete_node(fdb.ids["c"], detach=True)
    assert count(fdb, "vec_fact_emb") == 1 and count(fdb, "fts_fact_ft") == 1
    assert [h[0].id for h in fdb.fulltext_search("p_ft", "Bob")] == [fdb.ids["b"]]
    assert all(isinstance(h[0], Node) for h in fdb.fulltext_search("p_ft", "Bob"))


def test_relationship_type_named_like_a_node_label(db):
    """A relationship index over type X and node indexes over label X are independent."""
    db.create_relationship_vector_index("rv", "X", "embedding", 2)
    db.create_vector_index("nv", "X", "embedding", 2)
    n = db.create_node("X", {"embedding": [1.0, 0.0]})
    m = db.create_node("X", {"embedding": [0.0, 1.0]})
    e = db.create_edge(n.id, m.id, "X", {"embedding": [0.0, 1.0]})
    assert count(db, "vec_nv") == 2 and count(db, "vec_rv") == 1
    db.remove_labels(n.id, "X")  # must not try to read node ids out of the relationship table
    assert count(db, "vec_nv") == 1 and count(db, "vec_rv") == 1
    assert db.get_node(n.id)["embedding"] == [1.0, 0.0]  # no index covers it any more
    assert db.vector_search("rv", [0, 1], k=1)[0][0].id == e.id
    db.add_labels(n.id, "X")
    assert count(db, "vec_nv") == 2 and count(db, "vec_rv") == 1
    db.delete_node(n.id, detach=True)
    assert count(db, "vec_nv") == 1 and count(db, "vec_rv") == 0


def test_drop_relationship_indexes(fdb):
    assert fdb.drop_index("fact_emb") is True
    assert fdb.drop_index("fact_ft") is True
    assert fdb.vector_indexes() == [] and fdb.fulltext_indexes() == []
    names = {r[0] for r in fdb._conn.execute("SELECT name FROM sqlite_master")}
    assert not any(n.startswith(("vec_fact_emb", "fts_fact_ft")) for n in names)
    assert count(fdb, "_vector_indexes") == 0 and count(fdb, "_fulltext_indexes") == 0
    with pytest.raises(SqgraphError):
        fdb.vector_search("fact_emb", [1, 0, 0])
    # edges written afterwards keep the embedding as a plain property
    e = fdb.create_edge(fdb.ids["a"], fdb.ids["b"], "FACT", {"embedding": [1, 2, 3]})
    assert e.properties == {"embedding": [1, 2, 3]}
    q(fdb, "CREATE VECTOR INDEX v FOR ()-[r:FACT]-() ON (r.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3}}")
    q(fdb, "DROP INDEX v")
    with pytest.raises(SqgraphError):
        q(fdb, "DROP INDEX v")
    q(fdb, "DROP INDEX v IF EXISTS")


def test_existing_props_are_migrated_when_a_relationship_vector_index_is_created(db):
    a, b = db.create_node("N").id, db.create_node("N").id
    e1 = db.create_edge(a, b, "T", {"embedding": [1.0, 0.0], "k": 1}).id
    e2 = db.create_edge(a, b, "OTHER", {"embedding": [1.0, 0.0]}).id
    db.create_edge(a, b, "T", {"k": 2})
    db.create_relationship_vector_index("tv", "T", "embedding", 2)
    assert edge_ids(db, "vec_tv") == [e1]
    assert db.get_edge(e1).properties == {"k": 1}
    assert db.get_edge(e2)["embedding"] == [1.0, 0.0]
    assert db.vector_search("tv", [1, 0], k=3)[0][0].id == e1


def test_relationship_indexes_survive_reopen(tmp_path):
    path = tmp_path / "persist.db"
    with GraphDB(path) as db:
        db.create_relationship_vector_index("rv", "T", "embedding", 2)
        db.create_relationship_fulltext_index("rf", "T", "note")
        a, b = db.create_node("N").id, db.create_node("N").id
        eid = db.create_edge(a, b, "T", {"note": "hello world", "embedding": [0.0, 1.0]}).id
    with GraphDB(path) as db:
        assert [(i.name, i.entity_type) for i in db.vector_indexes()] == [("rv", "relationship")]
        assert [(i.name, i.entity_type) for i in db.fulltext_indexes()] == [("rf", "relationship")]
        assert db.vector_search("rv", [0, 1], k=1)[0][0].id == eid
        assert db.fulltext_search("rf", "hello")[0][0].id == eid
    with GraphDB(path, read_only=True) as db:
        assert db.vector_search("rv", [0, 1], k=1)[0][0].id == eid
        assert db.fulltext_search("rf", "hello")[0][0].id == eid
        assert {i["entityType"] for i in db.list_indexes()} == {"RELATIONSHIP"}


# ------------------------------------------------------------------ Cypher: DDL


def test_cypher_ddl_relationship_vector_index(db):
    q(
        db,
        "CREATE VECTOR INDEX fact_emb IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 4, `vector.similarity_function`: 'euclidean'}}",
    )
    q(  # idempotent
        db,
        "CREATE VECTOR INDEX fact_emb IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 4}}",
    )
    assert db.vector_indexes() == [VectorIndex("fact_emb", "FACT", "embedding", 4, "euclidean", "relationship")]
    q(db, "CREATE VECTOR INDEX out_emb FOR ()-[r:OUT]->() ON r.embedding OPTIONS {indexConfig: {`vector.dimensions`: 2}}")
    q(db, "create vector index in_emb for ()-[x:IN]-() on (x.v) options {indexConfig: {`vector.dimensions`: 2}}")
    assert {i.name: (i.label, i.property, i.entity_type) for i in db.vector_indexes()} == {
        "fact_emb": ("FACT", "embedding", "relationship"),
        "out_emb": ("OUT", "embedding", "relationship"),
        "in_emb": ("IN", "v", "relationship"),
    }
    with pytest.raises(CypherRuntimeError, match="already exists"):
        q(db, "CREATE VECTOR INDEX out_emb FOR ()-[r:OUT]-() ON (r.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 2}}")
    with pytest.raises(CypherRuntimeError, match="vector.dimensions"):
        q(db, "CREATE VECTOR INDEX nodim FOR ()-[r:OUT]-() ON (r.embedding)")


def test_cypher_ddl_relationship_fulltext_index(db):
    q(db, "CREATE FULLTEXT INDEX ft IF NOT EXISTS FOR ()-[r:A|B]-() ON EACH [r.x, r.y]")
    q(db, "CREATE FULLTEXT INDEX ft2 FOR ()-[r:FACT]->() ON EACH [r.fact] OPTIONS {indexConfig: {`fulltext.tokenizer`: 'trigram'}}")
    ix = {i.name: i for i in db.fulltext_indexes()}
    assert ix["ft"] == FulltextIndex("ft", ("A", "B"), ("x", "y"), "unicode61", "relationship")
    assert ix["ft2"].tokenizer == "trigram" and ix["ft2"].entity_type == "relationship"
    q(db, "CREATE (:N)-[:FACT {fact: 'graph databases are fun'}]->(:N)")
    q(db, "CREATE (:N)-[:A {x: 'graph theory'}]->(:N)")
    assert len(q(db, "CALL db.index.fulltext.queryRelationships('ft2', 'databases') YIELD relationship RETURN relationship")) == 1
    assert len(q(db, "CALL db.index.fulltext.queryRelationships('ft', 'graph') YIELD relationship RETURN relationship")) == 1


def test_cypher_ddl_still_creates_node_indexes(db):
    q(db, "CREATE VECTOR INDEX nv FOR (n:E) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 2}}")
    q(db, "CREATE FULLTEXT INDEX nf FOR (n:E|F) ON EACH [n.name]")
    q(db, "CREATE INDEX FOR (n:E) ON (n.name)")
    assert [(i.name, i.entity_type) for i in db.vector_indexes() + db.fulltext_indexes()] == [
        ("nv", "node"), ("nf", "node")
    ]
    assert {r["entityType"] for r in q(db, "SHOW INDEXES")} == {"NODE"}


def test_cypher_relationship_pattern_ddl_syntax_errors(db):
    from sqgraph import CypherSyntaxError

    for bad in (
        "CREATE VECTOR INDEX v FOR ()-[r]-() ON (r.p) OPTIONS {indexConfig: {`vector.dimensions`: 2}}",
        "CREATE VECTOR INDEX v FOR ()<-[r:T]-() ON (r.p) OPTIONS {indexConfig: {`vector.dimensions`: 2}}",
        "CREATE FULLTEXT INDEX f FOR ()-[r:T]-(m) ON EACH [r.p]",
        "CREATE INDEX FOR (a)-[r:T]-() ON (r.p)",
    ):
        with pytest.raises(CypherSyntaxError):
            q(db, bad)


# ------------------------------------------------------------------ Cypher: procedures


def test_query_relationships_vector_yields_a_relationship_variable(fdb):
    rows = q(
        fdb,
        "CALL db.index.vector.queryRelationships('fact_emb', 2, $v) YIELD relationship, score "
        "RETURN relationship, score",
        v=[1, 0.1, 0],
    )
    assert [r["relationship"].id for r in rows] == [fdb.ids["e1"], fdb.ids["e2"]]
    assert isinstance(rows[0]["relationship"], Relationship)
    assert rows[0]["relationship"].properties == {"id": "f1", "fact": "Alice manages the Rails migration"}
    assert rows[0]["score"] == pytest.approx(fdb.vector_search("fact_emb", [1, 0.1, 0], 2)[0][1], abs=1e-6)
    assert rows[0]["score"] > rows[1]["score"]


def test_query_relationships_combines_with_later_clauses(fdb):
    rows = q(
        fdb,
        "CALL db.index.vector.queryRelationships('fact_emb', 5, [1.0, 0.0, 0.0]) YIELD relationship AS r, score "
        "MATCH (s:Person)-[r]->(o:Person) "
        "WHERE score > 0.4 AND r.id STARTS WITH 'f' "
        "RETURN s.name AS s, o.name AS o, r.fact AS fact, type(r) AS t, startNode(r).name AS sn, "
        "endNode(r).name AS en, id(r) AS rid, r.embedding AS emb, score ORDER BY score DESC",
    )
    assert [(r["s"], r["o"], r["t"], r["sn"], r["en"], r["emb"]) for r in rows] == [
        ("Alice", "Bob", "FACT", "Alice", "Bob", None),
        ("Bob", "Carol", "FACT", "Bob", "Carol", None),
    ]
    assert rows[0]["fact"] == "Alice manages the Rails migration" and rows[0]["rid"] == fdb.ids["e1"]
    # the default YIELD name is usable as a bound relationship in a pattern
    rows = q(
        fdb,
        "CALL db.index.vector.queryRelationships('fact_emb', 1, $v) YIELD relationship, score "
        "MATCH (a)-[relationship]->(b) RETURN a.name AS a, b.name AS b, relationship.fact AS fact",
        v=[0, 1, 0],
    )
    assert rows == [{"a": "Bob", "b": "Carol", "fact": "Bob reviews Carol's pull requests"}]
    # YIELD ... WHERE, WITH, aggregation, LIMIT keeping the best rows
    rows = q(
        fdb,
        "CALL db.index.vector.queryRelationships('fact_emb', 5, [1.0, 0.0, 0.0]) YIELD relationship, score "
        "WHERE score > 0.9 RETURN count(relationship) AS c",
    )
    assert rows == [{"c": 1}]
    rows = q(
        fdb,
        "CALL db.index.vector.queryRelationships('fact_emb', 5, [0.0, 1.0, 0.0]) YIELD relationship, score "
        "WITH relationship, score RETURN relationship.id AS id LIMIT 1",
    )
    assert rows == [{"id": "f2"}]


def test_query_relationships_fulltext(fdb):
    rows = q(
        fdb,
        "CALL db.index.fulltext.queryRelationships('fact_ft', $q) YIELD relationship, score "
        "MATCH (s)-[relationship]->(o) RETURN s.name AS s, o.name AS o, relationship.id AS id, score",
        q="Rails",
    )
    assert [(r["s"], r["o"], r["id"]) for r in rows] == [("Alice", "Bob", "f1")]
    assert rows[0]["score"] > 0
    rows = q(fdb, "CALL db.index.fulltext.queryRelationships('fact_ft', 'Carol OR Alice', {limit: 1}) YIELD relationship RETURN relationship.id AS id")
    assert len(rows) == 1
    rows = q(
        fdb,
        "CALL db.index.fulltext.queryRelationships('fact_ft', 'reviews') YIELD relationship AS r, score "
        "RETURN startNode(r).name AS who, type(r) AS t, r.fact CONTAINS 'pull' AS pull",
    )
    assert rows == [{"who": "Bob", "t": "FACT", "pull": True}]
    assert q(fdb, "CALL db.index.fulltext.queryRelationships('fact_ft', '') YIELD relationship RETURN relationship") == []


def test_procedures_match_the_python_api(fdb):
    api = [(h[0].id, h[1]) for h in fdb.fulltext_search("fact_ft", "Carol OR Alice", 5)]
    cy = q(fdb, "CALL db.index.fulltext.queryRelationships('fact_ft', 'Carol OR Alice') YIELD relationship, score RETURN id(relationship) AS i, score")
    assert [(r["i"], r["score"]) for r in cy] == [(i, pytest.approx(s)) for i, s in api]


def test_wrong_kind_errors(fdb):
    fdb.create_vector_index("n_emb", "Person", "embedding", 3)
    fdb.create_fulltext_index("n_ft", "Person", "name")
    with pytest.raises(CypherRuntimeError, match="relationship vector index"):
        q(fdb, "CALL db.index.vector.queryNodes('fact_emb', 1, [1.0, 0.0, 0.0]) YIELD node RETURN node")
    with pytest.raises(CypherRuntimeError, match="node vector index"):
        q(fdb, "CALL db.index.vector.queryRelationships('n_emb', 1, [1.0, 0.0, 0.0]) YIELD relationship RETURN relationship")
    with pytest.raises(CypherRuntimeError, match="relationship fulltext index"):
        q(fdb, "CALL db.index.fulltext.queryNodes('fact_ft', 'Rails') YIELD node RETURN node")
    with pytest.raises(CypherRuntimeError, match="node fulltext index"):
        q(fdb, "CALL db.index.fulltext.queryRelationships('n_ft', 'Bob') YIELD relationship RETURN relationship")
    with pytest.raises(CypherRuntimeError, match="no such vector"):
        q(fdb, "CALL db.index.vector.queryRelationships('nope', 1, [1.0, 0.0, 0.0]) YIELD relationship RETURN relationship")
    with pytest.raises(CypherRuntimeError, match="no such fulltext"):
        q(fdb, "CALL db.index.fulltext.queryRelationships('nope', 'x') YIELD relationship RETURN relationship")
    with pytest.raises(SqgraphError, match="no output"):  # queryRelationships does not yield `node`
        q(fdb, "CALL db.index.vector.queryRelationships('fact_emb', 1, [1.0, 0.0, 0.0]) YIELD node RETURN node")
    with pytest.raises(DimensionMismatchError):
        q(fdb, "CALL db.index.vector.queryRelationships('fact_emb', 1, [1.0, 0.0]) YIELD relationship RETURN relationship")
    with pytest.raises(SqgraphError):
        fdb.vector_search("n_emb", [1, 2])  # fine for nodes but wrong dims
    # Python search works for both kinds
    assert isinstance(fdb.fulltext_search("n_ft", "Bob")[0][0], Node)


def test_set_relationship_vector_property(fdb):
    e3 = fdb.create_edge(fdb.ids["a"], fdb.ids["c"], "FACT", {"id": "f3", "fact": "third"}).id
    q(fdb, "MATCH ()-[r:FACT {id: 'f3'}]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', $v)", v=[0.0, 0.0, 1.0])
    assert count(fdb, "vec_fact_emb") == 3
    assert fdb.vector_search("fact_emb", [0, 0, 1], k=1)[0][0].id == e3
    assert fdb.get_edge(e3).properties == {"id": "f3", "fact": "third"}
    # several rows at once, driven by a MATCH that runs before the write
    q(fdb, "MATCH ()-[r:FACT]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', [1.0, 1.0, 1.0])")
    assert count(fdb, "vec_fact_emb") == 3
    assert {round(h[1], 4) for h in fdb.vector_search("fact_emb", [1, 1, 1], k=3)} == {1.0}
    with pytest.raises(DimensionMismatchError):
        q(fdb, "MATCH ()-[r:FACT {id: 'f3'}]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', [1.0])")
    with pytest.raises(SqgraphError, match="no vector index covers"):
        q(fdb, "MATCH ()-[r:FACT {id: 'f3'}]->() CALL db.create.setRelationshipVectorProperty(r, 'nope', [1.0, 1.0, 1.0])")
    with pytest.raises(SqgraphError, match="must be a relationship"):
        q(fdb, "MATCH (n:Person {name: 'Alice'}) CALL db.create.setRelationshipVectorProperty(n, 'embedding', [1.0, 1.0, 1.0])")
    with pytest.raises(SqgraphError, match="must be a node"):
        q(fdb, "MATCH ()-[r:FACT {id: 'f3'}]->() CALL db.create.setNodeVectorProperty(r, 'embedding', [1.0, 1.0, 1.0])")
    with pytest.raises(SqgraphError, match="YIELD"):
        q(fdb, "MATCH ()-[r:FACT]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', [1.0, 1.0, 1.0]) YIELD relationship")
    with pytest.raises(SqgraphError, match="takes 3 arguments"):
        q(fdb, "MATCH ()-[r:FACT]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding')")
    # a null relationship (OPTIONAL MATCH) is a no-op
    q(fdb, "OPTIONAL MATCH ()-[r:MISSING]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', [1.0, 1.0, 1.0])")


def test_set_relationship_vector_property_is_rejected_on_read_only(tmp_path):
    path = tmp_path / "ro.db"
    with GraphDB(path) as db:
        db.create_relationship_vector_index("rv", "T", "embedding", 2)
        a, b = db.create_node("N").id, db.create_node("N").id
        db.create_edge(a, b, "T")
    with GraphDB(path, read_only=True) as ro:
        with pytest.raises(CypherRuntimeError, match="read-only"):
            q(ro, "MATCH ()-[r:T]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', [1.0, 0.0])")


def test_cypher_create_populates_indexes(db):
    db.create_relationship_vector_index("fv", "FACT", "embedding", 2)
    db.create_relationship_fulltext_index("ff", "FACT", "fact")
    q(db, "CREATE (a:P {name:'a'})-[r:FACT {fact: 'a knows b', embedding: [1.0, 0.0]}]->(b:P {name:'b'})")
    q(db, "MATCH (a:P {name:'a'}), (b:P {name:'b'}) MERGE (a)-[r:FACT {fact: 'merged fact'}]->(b) ON CREATE SET r.embedding = [0.0, 1.0]")
    q(db, "UNWIND [[0.5, 0.5], [0.25, 0.75]] AS v MATCH (a:P {name:'b'}), (b:P {name:'a'}) CREATE (a)-[:FACT {fact: 'unwound', embedding: v}]->(b)")
    assert count(db, "edges") == 4 and count(db, "vec_fv") == 4 and count(db, "fts_ff") == 4
    assert db.fulltext_search("ff", "merged")[0][0]["fact"] == "merged fact"
    assert all("embedding" not in e.properties for e in db._load_edges(range(1, 5)).values())


def test_cypher_set_and_remove_keep_indexes_in_sync(fdb):
    e1 = fdb.ids["e1"]
    f1 = "MATCH ()-[r:FACT {id: 'f1'}]->() "
    q(fdb, f1 + "SET r.embedding = [0.0, 0.0, 1.0]")
    assert fdb.vector_search("fact_emb", [0, 0, 1], k=1)[0][0].id == e1 and count(fdb, "vec_fact_emb") == 2
    q(fdb, f1 + "SET r.fact = 'Alice owns the Kotlin port'")
    assert [h[0].id for h in fdb.fulltext_search("fact_ft", "Kotlin")] == [e1]
    assert fdb.fulltext_search("fact_ft", "Rails") == []
    q(fdb, f1 + "SET r += {fact: 'Alice owns the Golang port', embedding: [0.0, 1.0, 1.0], extra: 1}")
    assert [h[0].id for h in fdb.fulltext_search("fact_ft", "Golang")] == [e1]
    assert fdb.get_edge(e1).properties == {"id": "f1", "fact": "Alice owns the Golang port", "extra": 1}
    assert fdb.vector_search("fact_emb", [0, 1, 1], k=1)[0][0].id == e1
    q(fdb, f1 + "SET r.embedding = null")
    assert edge_ids(fdb, "vec_fact_emb") == [fdb.ids["e2"]]
    q(fdb, f1 + "SET r.embedding = [1.0, 1.0, 1.0]")
    q(fdb, f1 + "REMOVE r.embedding")
    assert edge_ids(fdb, "vec_fact_emb") == [fdb.ids["e2"]]
    q(fdb, f1 + "SET r.embedding = [1.0, 1.0, 1.0]")
    q(fdb, f1 + "SET r = {id: 'f1', fact: 'replaced text'}")  # replace: the embedding is gone, fts follows
    assert edge_ids(fdb, "vec_fact_emb") == [fdb.ids["e2"]]
    assert [h[0].id for h in fdb.fulltext_search("fact_ft", "replaced")] == [e1]
    q(fdb, f1 + "SET r = {id: 'f1', embedding: [1.0, 0.0, 0.0]}")  # replace keeping an embedding, dropping fact
    assert edge_ids(fdb, "vec_fact_emb") == sorted([e1, fdb.ids["e2"]])
    assert edge_ids(fdb, "fts_fact_ft") == [fdb.ids["e2"]]
    q(fdb, f1 + "REMOVE r.id")
    assert fdb.get_edge(e1).properties == {}
    with pytest.raises(DimensionMismatchError):
        q(fdb, "MATCH ()-[r:FACT]->() WHERE id(r) = $i SET r.embedding = [1.0]", i=e1)
    assert edge_ids(fdb, "vec_fact_emb") == sorted([e1, fdb.ids["e2"]])
    # copying the properties of one relationship onto another never copies an embedding
    q(fdb, "MATCH ()-[a:FACT {id: 'f2'}]->(), ()-[b:FACT]->() WHERE id(b) = $i SET b += a", i=e1)
    assert fdb.get_edge(e1).properties == {"id": "f2", "fact": "Bob reviews Carol's pull requests"}
    assert edge_ids(fdb, "fts_fact_ft") == sorted([e1, fdb.ids["e2"]])


def test_failed_write_query_rolls_back_relationship_index_rows(fdb):
    with pytest.raises(DimensionMismatchError):
        q(
            fdb,
            "MATCH ()-[r:FACT]->() SET r.fact = 'changed by a failing query' "
            "WITH r CALL db.create.setRelationshipVectorProperty(r, 'embedding', [1.0])",
        )
    assert fdb.fulltext_search("fact_ft", "failing") == []
    assert len(fdb.fulltext_search("fact_ft", "Rails")) == 1
    with pytest.raises(SqgraphError):
        with fdb.transaction():
            fdb.create_relationship_vector_index("tmp", "T", "embedding", 2)
            fdb.create_edge(fdb.ids["a"], fdb.ids["b"], "T", {"embedding": [1, 2]})
            raise SqgraphError("boom")
    assert [i.name for i in fdb.vector_indexes()] == ["fact_emb"]
    assert fdb._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'vec_tmp'").fetchone()[0] == 0


# ------------------------------------------------------------------ property indexes on relationships


def plan(db, cypher, params=None):
    # note: sqlite3 caches statements by SQL text and EXPLAIN output is not refreshed after a schema change,
    # so a plan asked for again after DROP INDEX needs different SQL text
    comp = Compiler(db, params or {}, cypher)
    seg = comp.compile_segment(parse(cypher).parts[0].clauses)
    return seg.sql, " | ".join(r[3] for r in db._conn.execute("EXPLAIN QUERY PLAN " + seg.sql, seg.binds))


@pytest.fixture
def pdb(db):
    a, b = db.create_node("P").id, db.create_node("P").id
    db.create_edges([(a, b, "FACT", {"id": f"f{i}", "n": i}) for i in range(400)])
    db.create_edges([(b, a, "OTHER", {"id": f"o{i}"}) for i in range(400)])
    return db


def test_relationship_property_index_api(pdb):
    name = pdb.create_relationship_property_index("FACT", "id")
    assert name.startswith("rprop_FACT_id_")
    assert pdb.create_relationship_property_index("FACT", "id") == name  # idempotent
    with pytest.raises(SqgraphError, match="already exists"):
        pdb.create_relationship_property_index("FACT", "id", if_not_exists=False)
    node_name = pdb.create_property_index("FACT", "id")  # a node index for a label with the same name: distinct
    assert node_name != name and node_name.startswith("prop_FACT_id_")
    row = pdb._conn.execute("SELECT label, property, entity_type FROM _property_indexes WHERE name = ?", (name,)).fetchone()
    assert row == ("FACT", "id", "relationship")
    sql = pdb._conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (f"idx_p_{name}",)).fetchone()[0]
    assert "ON edges(" in sql and "json_extract(props, '$.id')" in sql
    named = pdb.create_relationship_property_index("FACT", "n", name="fact_n")
    assert named == "fact_n"
    with pytest.raises(SqgraphError, match="different"):
        pdb.create_relationship_property_index("OTHER", "n", name="fact_n")
    with pytest.raises(SqgraphError, match="different"):
        pdb.create_property_index("FACT", "n", name="fact_n")
    with pytest.raises(SqgraphError):
        pdb.create_relationship_property_index("FACT", "bad name")
    with pytest.raises(SqgraphError):
        pdb.create_relationship_property_index("", "id")
    with pytest.raises(SqgraphError):
        pdb.create_relationship_property_index("FACT", "id", name="bad name")
    pdb.create_relationship_vector_index("shared_name", "FACT", "embedding", 2)
    with pytest.raises(SqgraphError, match="already exists"):  # index names are unique across index kinds
        pdb.create_relationship_property_index("FACT", "x", name="shared_name")
    assert pdb.drop_index(name) is True
    assert pdb._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = ?", (f"idx_p_{name}",)).fetchone()[0] == 0


def test_relationship_property_index_is_used_by_the_planner(pdb):
    name = pdb.create_relationship_property_index("FACT", "id")
    pdb.create_relationship_property_index("FACT", "n")
    for cy, params in (
        ("MATCH ()-[r:FACT {id: $v}]->() RETURN r.n AS n", {"v": "f7"}),
        ("MATCH ()-[r:FACT {id: 'f7'}]->() RETURN r.n AS n", {}),
        ("MATCH ()-[r:FACT]->() WHERE r.id = $v RETURN r.n AS n", {"v": "f7"}),
        ("MATCH (a)-[r:FACT {id: $v}]->(b) RETURN a, b", {"v": "f7"}),
        ("MATCH (a:P)-[r:FACT {id: $v}]->(b:P) RETURN r.n AS n", {"v": "f7"}),
        ("MATCH ()-[r {id: $v}]->() RETURN r.n AS n", {"v": "f7"}),
    ):
        sql, p = plan(pdb, cy, params)
        assert f"idx_p_{name}" in p, (cy, p)
        assert "'$.id'" in sql  # literal path, or the expression index cannot match
    sql, p = plan(pdb, "MATCH ()-[r:FACT {n: $v}]->() RETURN r.id AS id", {"v": 7})
    assert "idx_p_rprop_FACT_n_" in p, p
    assert q(pdb, "MATCH ()-[r:FACT {id: $v}]->() RETURN r.n AS n", v="f7") == [{"n": 7}]
    assert q(pdb, "MATCH ()-[r:FACT]->() WHERE r.n = 7 RETURN r.id AS id") == [{"id": "f7"}]
    assert q(pdb, "MATCH ()-[r:OTHER {id: 'f7'}]->() RETURN count(r) AS c") == [{"c": 0}]
    # without the index the same query scans the edge table
    pdb.drop_index(name)
    _, p = plan(pdb, "MATCH ()-[r:FACT {id: $v}]->() RETURN r.n + 1 AS n", {"v": "f7"})
    assert "idx_p_" not in p and "SCAN e" in p, p


def test_relationship_property_index_ddl(pdb):
    q(pdb, "CREATE INDEX fact_id FOR ()-[r:FACT]-() ON (r.id)")
    q(pdb, "CREATE INDEX fact_id IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.id)")
    q(pdb, "CREATE RANGE INDEX fact_n FOR ()-[r:FACT]->() ON (r.n)")
    q(pdb, "CREATE INDEX IF NOT EXISTS FOR ()-[r:OTHER]->() ON r.oid")
    idx = {r["name"]: r for r in q(pdb, "SHOW INDEXES")}
    assert idx["fact_id"]["type"] == "RANGE" and idx["fact_id"]["entityType"] == "RELATIONSHIP"
    assert idx["fact_id"]["labelsOrTypes"] == ["FACT"] and idx["fact_id"]["properties"] == ["id"]
    assert idx["fact_n"]["entityType"] == "RELATIONSHIP"
    assert len(idx) == 3
    with pytest.raises(CypherRuntimeError, match="already exists"):
        q(pdb, "CREATE INDEX fact_id FOR ()-[r:FACT]-() ON (r.id)")
    with pytest.raises(CypherUnsupportedError):
        q(pdb, "CREATE INDEX FOR ()-[r:FACT]-() ON (r.a, r.b)")
    with pytest.raises(SqgraphError):
        q(pdb, "CREATE INDEX FOR ()-[r:FACT]-() ON (r.`bad key`)")
    _, p = plan(pdb, "MATCH ()-[r:FACT {id: $v}]->() RETURN r.n AS n", {"v": "f7"})
    assert "idx_p_fact_id" in p
    q(pdb, "DROP INDEX fact_id")
    q(pdb, "DROP INDEX fact_n")
    assert [r["entityType"] for r in q(pdb, "SHOW INDEXES")] == ["RELATIONSHIP"]
    q(pdb, "DROP INDEX nonexistent IF EXISTS")


def test_relationship_property_index_analyze_and_write_paths(pdb):
    pdb.create_relationship_property_index("FACT", "id")
    e = pdb.create_edge(1, 2, "FACT", {"id": "fresh"})
    assert q(pdb, "MATCH ()-[r:FACT {id: 'fresh'}]->() RETURN id(r) AS i") == [{"i": e.id}]
    pdb.update_edge(e.id, {"id": "renamed"})
    assert q(pdb, "MATCH ()-[r:FACT {id: 'fresh'}]->() RETURN id(r) AS i") == []
    assert q(pdb, "MATCH ()-[r:FACT {id: 'renamed'}]->() RETURN id(r) AS i") == [{"i": e.id}]


# ------------------------------------------------------------------ schema migration


def make_v2_file(path):
    """A database as written by sqgraph before relationship indexes existed (schema version 2)."""
    with GraphDB(path) as db:
        db.create_vector_index("n_emb", "Entity", "embedding", 2)
        db.create_fulltext_index("n_ft", "Entity", "name", tokenizer="trigram")
        db.create_property_index("Entity", "name", name="ent_name")
        a = db.create_node("Entity", {"name": "Alice", "embedding": [1.0, 0.0]}).id
        b = db.create_node("Entity", {"name": "Bob", "embedding": [0.0, 1.0]}).id
        db.create_edge(a, b, "KNOWS", {"since": 2020})
        c = db._conn
        for table in ("_vector_indexes", "_fulltext_indexes", "_property_indexes"):
            c.execute(f"ALTER TABLE {table} DROP COLUMN entity_type")
        c.execute("UPDATE _meta SET value = '2' WHERE key = 'schema_version'")


def columns(path, table):
    import sqlite3

    c = sqlite3.connect(path)
    try:
        return [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
    finally:
        c.close()


def test_migration_from_schema_v2_writable(tmp_path):
    path = tmp_path / "v2.db"
    make_v2_file(path)
    assert "entity_type" not in columns(path, "_vector_indexes")
    with GraphDB(path) as db:
        assert db._conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()[0] == "3"
        for table in ("_vector_indexes", "_fulltext_indexes", "_property_indexes"):
            assert "entity_type" in columns(path, table)
        assert {(i["name"], i["entityType"]) for i in db.list_indexes()} == {
            ("n_emb", "NODE"), ("n_ft", "NODE"), ("ent_name", "NODE"),
        }
        assert db.vector_search("n_emb", [1, 0], 1)[0][0]["name"] == "Alice"
        assert isinstance(db.fulltext_search("n_ft", "Bob")[0][0], Node)
        # ... and the new kinds work on the migrated file
        db.create_relationship_vector_index("r_emb", "KNOWS", "embedding", 2)
        db.create_relationship_fulltext_index("r_ft", "KNOWS", "note")
        db.create_relationship_property_index("KNOWS", "since")
        eid = db.query("MATCH ()-[r:KNOWS]->() SET r.embedding = [1.0, 1.0], r.note = 'old friends' RETURN id(r) AS i")[0]["i"]
        assert db.vector_search("r_emb", [1, 1], 1)[0][0].id == eid
        assert db.fulltext_search("r_ft", "friends")[0][0].id == eid
    with GraphDB(path) as db:  # reopening the migrated file
        assert {i.entity_type for i in db.vector_indexes()} == {"node", "relationship"}
        assert db.vector_search("r_emb", [1, 1], 1)[0][0].id == eid


def test_migration_from_schema_v2_read_only(tmp_path):
    path = tmp_path / "v2ro.db"
    make_v2_file(path)
    with GraphDB(path, read_only=True) as db:  # cannot migrate: a missing column means 'node'
        assert "entity_type" not in columns(path, "_vector_indexes")
        assert {(i.name, i.entity_type) for i in db.vector_indexes()} == {("n_emb", "node")}
        assert {(i.name, i.entity_type) for i in db.fulltext_indexes()} == {("n_ft", "node")}
        assert {(i["name"], i["entityType"]) for i in db.list_indexes()} == {
            ("n_emb", "NODE"), ("n_ft", "NODE"), ("ent_name", "NODE"),
        }
        assert db.vector_search("n_emb", [0, 1], 1)[0][0]["name"] == "Bob"
        assert db.fulltext_search("n_ft", "Alice")[0][0]["name"] == "Alice"
        assert q(db, "CALL db.index.vector.queryNodes('n_emb', 1, [1.0, 0.0]) YIELD node RETURN node.name AS n") == [{"n": "Alice"}]
        assert {r["entityType"] for r in q(db, "SHOW INDEXES")} == {"NODE"}
        assert db.stats()["nodes"] == 2
        with pytest.raises(CypherRuntimeError, match="read-only"):
            q(db, "CREATE (:X)")
        with pytest.raises(CypherRuntimeError, match="read-only"):
            q(db, "CREATE INDEX FOR ()-[r:KNOWS]-() ON (r.since)")
        with pytest.raises(CypherRuntimeError, match="node vector index"):
            q(db, "CALL db.index.vector.queryRelationships('n_emb', 1, [1.0, 0.0]) YIELD relationship RETURN relationship")
    assert "entity_type" not in columns(path, "_vector_indexes")  # the file was left untouched


def test_fresh_database_has_schema_v3():
    with GraphDB() as db:
        assert db._conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()[0] == "3"
        for table in ("_vector_indexes", "_fulltext_indexes", "_property_indexes"):
            assert "entity_type" in [r[1] for r in db._conn.execute(f"PRAGMA table_info({table})")]


# ------------------------------------------------------------------ viewer


def test_viewer_lists_relationship_indexes_and_never_leaks_embeddings(tmp_path):
    path = tmp_path / "v.db"
    dims = 70
    with GraphDB(path) as db:
        db.create_relationship_vector_index("fact_emb", "FACT", "embedding", dims)
        db.create_relationship_fulltext_index("fact_ft", "FACT", "fact")
        db.create_relationship_property_index("FACT", "id")
        a, b = db.create_node("Entity", {"name": "A"}).id, db.create_node("Entity", {"name": "B"}).id
        db.create_edge(a, b, "FACT", {"id": "f1", "fact": "A relates to B", "embedding": [0.5] * dims, "plain": [0.25] * 100})
    server = make_server(path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def call(route, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(server.url.rstrip("/") + route, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        o = call("/api/overview")
        kinds = {(i["name"], i["type"], i["entityType"]) for i in o["indexes"]}
        assert ("fact_emb", "VECTOR", "RELATIONSHIP") in kinds and ("fact_ft", "FULLTEXT", "RELATIONSHIP") in kinds
        assert o["relationships"] == 1 and "embedding" not in o["propertyKeys"]
        g = call("/api/graph?limit=10")
        assert "embedding" not in g["edges"][0]["properties"]
        assert g["edges"][0]["properties"]["plain"] == "<vector: 100 dims>"  # un-indexed vectors are summarized
        assert g["edges"][0]["properties"]["fact"] == "A relates to B"
        node_id = g["nodes"][0]["id"]
        e = call(f"/api/expand/{node_id}")
        assert "embedding" not in e["edges"][0]["properties"]
        r = call(
            "/api/query",
            {"cypher": "CALL db.index.fulltext.queryRelationships('fact_ft', 'relates') YIELD relationship, score RETURN relationship, score"},
        )
        assert r["columns"] == ["relationship", "score"] and len(r["graph"]["edges"]) == 1
        assert "embedding" not in json.dumps(r)
        r = call("/api/query", {"cypher": "MATCH ()-[r:FACT]->() RETURN r.embedding AS e, r"})
        assert r["rows"][0][0] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
