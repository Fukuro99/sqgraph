from __future__ import annotations

import json

import pytest

from sqgraph import GraphDB, Node, Relationship, SqgraphError


def test_create_and_get_node(db):
    n = db.create_node(["Person"], {"name": "Alice", "age": 30})
    assert isinstance(n, Node)
    assert n.labels == ("Person",)
    assert n["name"] == "Alice"
    assert n.get("missing") is None
    assert n.get("missing", 5) == 5
    got = db.get_node(n.id)
    assert got == n
    assert got.properties == {"name": "Alice", "age": 30}
    assert db.get_node(9999) is None


def test_single_label_string_and_no_labels(db):
    a = db.create_node("Person")
    b = db.create_node(None, {"x": 1})
    c = db.create_node([], {})
    assert a.labels == ("Person",)
    assert b.labels == ()
    assert c.properties == {}


def test_multi_labels_sorted_and_deduplicated(db):
    n = db.create_node(["Zed", "Alpha", "Zed"])
    assert n.labels == ("Alpha", "Zed")
    db.add_labels(n.id, "Mid")
    assert db.get_node(n.id).labels == ("Alpha", "Mid", "Zed")
    db.remove_labels(n.id, ["Alpha"])
    assert db.get_node(n.id).labels == ("Mid", "Zed")


def test_property_types_roundtrip(db):
    props = {"s": "héllo 日本語", "i": 1, "f": 1.5, "b": True, "l": [1, 2, 3], "sl": ["a", "b"], "m": {"k": 1}}
    n = db.create_node("T", props)
    assert db.get_node(n.id).properties == props
    assert db.get_node(n.id)["b"] is True


def test_none_props_dropped_and_bad_values_rejected(db):
    n = db.create_node("T", {"a": None, "b": 1})
    assert n.properties == {"b": 1}
    with pytest.raises(SqgraphError):
        db.create_node("T", {"a": object()})
    with pytest.raises(SqgraphError):
        db.create_node("T", {"a": float("nan")})
    with pytest.raises(SqgraphError):
        db.create_node("T", {"": 1})
    with pytest.raises(SqgraphError):
        db.create_node([""])


def test_update_node_merge_and_replace(db):
    n = db.create_node("P", {"a": 1, "b": 2})
    n2 = db.update_node(n.id, {"b": 3, "c": 4})
    assert n2.properties == {"a": 1, "b": 3, "c": 4}
    n3 = db.update_node(n.id, {"a": None})
    assert n3.properties == {"b": 3, "c": 4}
    n4 = db.update_node(n.id, {"z": 1}, replace=True)
    assert n4.properties == {"z": 1}
    with pytest.raises(SqgraphError):
        db.update_node(12345, {"a": 1})


def test_create_edge_and_get(db):
    a, b = db.create_node("A"), db.create_node("B")
    e = db.create_edge(a.id, b.id, "REL", {"w": 0.5})
    assert isinstance(e, Relationship)
    assert (e.type, e.start_node_id, e.end_node_id, e.properties) == ("REL", a.id, b.id, {"w": 0.5})
    assert db.get_edge(e.id) == e
    assert db.get_edge(999) is None
    assert db.update_edge(e.id, {"w": 1}).properties == {"w": 1}


def test_create_edge_validates_endpoints(db):
    a = db.create_node("A")
    with pytest.raises(SqgraphError):
        db.create_edge(a.id, 999, "REL")
    with pytest.raises(SqgraphError):
        db.create_edge(a.id, a.id, "")
    assert db.stats()["relationships"] == 0


def test_self_loop_and_parallel_edges(db):
    a = db.create_node("A")
    db.create_edge(a.id, a.id, "SELF")
    db.create_edge(a.id, a.id, "SELF")
    assert db.stats()["relationships"] == 2


def test_node_and_edge_json_serializable_and_equality(db):
    a, b = db.create_node("A", {"x": 1}), db.create_node("B")
    e = db.create_edge(a.id, b.id, "R")
    json.dumps(a.to_dict())
    json.dumps(e.to_dict())
    assert a.to_dict() == {"id": a.id, "labels": ["A"], "properties": {"x": 1}}
    assert a == db.get_node(a.id) and a != b
    assert len({a, db.get_node(a.id)}) == 1
    assert a != e


def test_delete_node_requires_detach(db):
    a, b = db.create_node("A"), db.create_node("B")
    e = db.create_edge(a.id, b.id, "R")
    with pytest.raises(SqgraphError, match="relationship"):
        db.delete_node(a.id)
    assert db.get_node(a.id) is not None
    assert db.delete_node(a.id, detach=True) is True
    assert db.get_node(a.id) is None
    assert db.get_edge(e.id) is None
    assert db.get_node(b.id) is not None
    assert db.delete_node(a.id) is False


def test_delete_isolated_node_and_edge(db):
    a, b = db.create_node("A"), db.create_node("B")
    e = db.create_edge(a.id, b.id, "R")
    assert db.delete_edge(e.id) is True
    assert db.delete_edge(e.id) is False
    assert db.delete_node(a.id) is True
    assert db.stats() == {"nodes": 1, "relationships": 0, "labels": 1}


def _chain(db):
    a, b, c, d = (db.create_node("N", {"name": n}) for n in "abcd")
    db.create_edge(a.id, b.id, "R")
    db.create_edge(b.id, c.id, "R")
    db.create_edge(c.id, d.id, "S")
    return a, b, c, d


def test_neighbors_directions(db):
    a, b, c, d = _chain(db)
    assert [n.id for n in db.neighbors(b.id)] == [c.id]
    assert [n.id for n in db.neighbors(b.id, direction="in")] == [a.id]
    assert {n.id for n in db.neighbors(b.id, direction="both")} == {a.id, c.id}
    with pytest.raises(SqgraphError):
        db.neighbors(b.id, direction="sideways")


def test_neighbors_depth_types_labels(db):
    a, b, c, d = _chain(db)
    assert {n.id for n in db.neighbors(a.id, depth=2)} == {b.id, c.id}
    assert {n.id for n in db.neighbors(a.id, depth=3)} == {b.id, c.id, d.id}
    assert {n.id for n in db.neighbors(a.id, types=["R"], depth=5)} == {b.id, c.id}
    assert {n.id for n in db.neighbors(a.id, types="S", depth=5, direction="both")} == set()
    x = db.create_node("X")
    db.create_edge(a.id, x.id, "R")
    assert [n.id for n in db.neighbors(a.id, labels=["X"])] == [x.id]
    with pytest.raises(SqgraphError):
        db.neighbors(a.id, depth=0)


def test_neighbors_cycle_safe_and_distinct(db):
    a, b, c = (db.create_node("N") for _ in range(3))
    db.create_edge(a.id, b.id, "R")
    db.create_edge(b.id, c.id, "R")
    db.create_edge(c.id, a.id, "R")
    db.create_edge(a.id, b.id, "R")  # parallel edge must not duplicate results
    res = db.neighbors(a.id, depth=10)
    assert sorted(n.id for n in res) == sorted([b.id, c.id])
    both = db.neighbors(a.id, direction="both", depth=10)
    assert sorted(n.id for n in both) == sorted([b.id, c.id])


def test_transaction_commit_and_rollback(db):
    with db.transaction():
        a = db.create_node("A")
    assert db.get_node(a.id) is not None
    with pytest.raises(RuntimeError):
        with db.transaction():
            b = db.create_node("B")
            raise RuntimeError("boom")
    assert db.get_node(b.id) is None
    assert db.stats()["nodes"] == 1


def test_nested_transactions_use_savepoints(db):
    with db.transaction():
        a = db.create_node("A")
        try:
            with db.transaction():
                db.create_node("B")
                raise ValueError
        except ValueError:
            pass
        c = db.create_node("C")
    assert db.stats()["nodes"] == 2
    assert db.get_node(a.id) and db.get_node(c.id)


def test_failed_public_call_is_atomic(db):
    a = db.create_node("A")
    with pytest.raises(SqgraphError):
        db.create_edge(a.id, 999, "R")
    with pytest.raises(SqgraphError):
        db.create_nodes([("A", {"x": 1}), ("A", {"bad": object()})])
    assert db.stats()["nodes"] == 1


def test_bulk_create(db):
    ids = db.create_nodes([("P", {"i": i}) for i in range(100)] + [(["P", "Q"], None)])
    assert ids == list(range(ids[0], ids[0] + 101))
    assert db.stats()["nodes"] == 101
    assert db.get_node(ids[-1]).labels == ("P", "Q")
    eids = db.create_edges([(ids[i], ids[i + 1], "NEXT", {"i": i}) for i in range(100)] + [(ids[0], ids[1], "X")])
    assert len(eids) == 101
    assert db.get_edge(eids[5]).properties == {"i": 5}
    assert db.create_nodes([]) == [] and db.create_edges([]) == []
    with pytest.raises(SqgraphError):
        db.create_edges([(ids[0], 10**6, "BAD")])


def test_persistence_and_read_only(tmp_path):
    path = tmp_path / "g.db"
    with GraphDB(path) as db:
        a = db.create_node("P", {"name": "A"})
        b = db.create_node("P", {"name": "B"})
        db.create_edge(a.id, b.id, "KNOWS")
    with GraphDB(path, read_only=True) as ro:
        assert ro.get_node(a.id)["name"] == "A"
        assert ro.neighbors(a.id)[0].id == b.id
        assert ro.query("MATCH (n:P) RETURN count(*) AS c") == [{"c": 2}]
        with pytest.raises(SqgraphError):
            ro.create_node("P")
        with pytest.raises(SqgraphError):
            ro.create_edge(a.id, b.id, "X")
        with pytest.raises(SqgraphError):
            ro.delete_node(a.id, detach=True)
        with pytest.raises(SqgraphError):
            ro.update_node(a.id, {"x": 1})
        with pytest.raises(SqgraphError):
            ro.create_vector_index("v", "P", "embedding", 3)
    with GraphDB(path) as db:
        assert db.stats()["nodes"] == 2


def test_read_only_requires_existing_file(tmp_path):
    with pytest.raises(SqgraphError):
        GraphDB(tmp_path / "nope.db", read_only=True)
    with pytest.raises(SqgraphError):
        GraphDB(":memory:", read_only=True)


def test_read_only_rejects_non_sqgraph_file(tmp_path):
    import sqlite3

    p = tmp_path / "other.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE x(a)")
    c.commit()
    c.close()
    with pytest.raises(SqgraphError, match="not a sqgraph database"):
        GraphDB(p, read_only=True)


def test_pragmas(tmp_path):
    with GraphDB(tmp_path / "p.db") as db:
        assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with GraphDB() as m:
        assert m._conn.execute("PRAGMA journal_mode").fetchone()[0] == "memory"


def test_schema_tables_present(db):
    names = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')")}
    for t in ("nodes", "node_labels", "edges", "_meta", "_vector_indexes", "_fulltext_indexes",
              "idx_edges_src", "idx_edges_dst", "idx_node_labels_node"):
        assert t in names
    assert "idx_edges_type" not in names  # redundant with idx_edges_src(src, type, dst)
    assert db._conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()[0] == "3"


def test_property_index_used_by_planner(db):
    db.create_nodes([("P", {"name": f"n{i}"}) for i in range(50)])
    name = db.create_property_index("P", "name")
    assert db.create_property_index("P", "name") == name
    plan = db._conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM nodes WHERE json_extract(props, '$.name') = 'n5'"
    ).fetchall()
    assert any(name in row[3] for row in plan)
    with pytest.raises(SqgraphError):
        db.create_property_index("P", "bad name")
    db.drop_index(name)


def test_context_manager_closes(tmp_path):
    db = GraphDB(tmp_path / "c.db")
    with db:
        db.create_node("A")
    with pytest.raises(Exception):
        db.create_node("A")
