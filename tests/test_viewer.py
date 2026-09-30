"""JSON API of the graph viewer (sqgraph.viewer)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from sqgraph import GraphDB
from sqgraph.viewer.server import make_server, make_server_auto, serialize
from sqgraph.types import Node


@pytest.fixture(scope="module")
def viewer(tmp_path_factory):
    path = tmp_path_factory.mktemp("viewer") / "graph.db"
    with GraphDB(path) as db:
        db.create_vector_index("emb", "Entity", "embedding", dimensions=3)
        doc = db.create_node("Document", {"id": "d1", "title": "潮見市の概要"}).id
        chunk = db.create_node("Chunk", {"id": "d1#0", "text": "長い本文。" * 100}).id
        a = db.create_node("Entity", {"name": "蒼波エネルギー", "embedding": [0.1, 0.2, 0.3]}).id
        b = db.create_node("Entity", {"name": "Tidal Lab", "vec": [0.5] * 100}).id
        c = db.create_node("Entity", {"name": "Lonely"}).id
        com = db.create_node("Community", {"id": 0, "title": "エネルギー"}).id
        db.create_edge(doc, chunk, "HAS_CHUNK")
        db.create_edge(chunk, a, "MENTIONS")
        db.create_edge(chunk, b, "MENTIONS")
        db.create_edge(a, b, "RELATED_TO", {"description": "partner"})
        db.create_edge(a, com, "IN_COMMUNITY")
        db.create_edge(b, com, "IN_COMMUNITY")
        ids = {"doc": doc, "chunk": chunk, "a": a, "b": b, "c": c, "com": com}
    server = make_server(path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, ids
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def call(server, path, body=None, host=None):
    headers = {"Content-Type": "application/json"}
    if host:
        headers["Host"] = host
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(server.url.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_index_page(viewer):
    server, _ = viewer
    with urllib.request.urlopen(server.url) as r:
        html = r.read().decode("utf-8")
        assert r.headers["Content-Type"].startswith("text/html")
    assert "cytoscape" in html


def test_overview(viewer):
    server, _ = viewer
    status, o = call(server, "/api/overview")
    assert status == 200
    assert o["file"] == "graph.db"
    assert o["nodes"] == 6 and o["relationships"] == 6
    assert {"name": "Entity", "count": 3} in o["labels"]
    assert {"name": "MENTIONS", "count": 2} in o["types"]
    assert "name" in o["propertyKeys"] and "embedding" not in o["propertyKeys"]
    assert [i["type"] for i in o["indexes"]] == ["VECTOR"]
    assert o["hasCommunities"] is True


def test_graph_prefers_high_degree_and_never_leaks_vectors(viewer):
    server, ids = viewer
    status, g = call(server, "/api/graph?limit=3")
    assert status == 200
    got = {n["id"] for n in g["nodes"]}
    assert got == {ids["chunk"], ids["a"], ids["b"]}  # degree 3, 3, 3 beat the rest
    assert {e["type"] for e in g["edges"]} == {"MENTIONS", "RELATED_TO"}
    assert all(e["source"] in got and e["target"] in got for e in g["edges"])
    node = next(n for n in g["nodes"] if n["id"] == ids["a"])
    assert node["degree"] == 3 and node["community"] == ids["com"]
    assert g["communities"] == {str(ids["com"]): "エネルギー"}
    _, full = call(server, "/api/graph?limit=100")
    assert len(full["nodes"]) == 6
    assert "embedding" not in json.dumps(full)
    assert "vector" in json.dumps(full)  # >64 floats replaced with a marker
    status, only = call(server, "/api/graph?limit=100&labels=Entity,Community")
    assert {tuple(n["labels"]) for n in only["nodes"]} == {("Entity",), ("Community",)}


def test_node_full_properties(viewer):
    server, ids = viewer
    status, n = call(server, f"/api/node/{ids['chunk']}")
    assert status == 200
    assert n["labels"] == ["Chunk"] and n["degree"] == 3
    assert n["properties"]["text"] == "長い本文。" * 100  # not truncated
    assert {(r["type"], r["direction"], r["count"]) for r in n["relationships"]} == {
        ("HAS_CHUNK", "in", 1),
        ("MENTIONS", "out", 2),
    }
    _, light = call(server, "/api/graph?limit=100")
    truncated = next(x for x in light["nodes"] if x["id"] == ids["chunk"])["properties"]["text"]
    assert len(truncated) < 300
    assert call(server, "/api/node/999999")[0] == 404
    assert call(server, "/api/node/abc")[0] == 400


def test_expand(viewer):
    server, ids = viewer
    status, e = call(server, f"/api/expand/{ids['chunk']}")
    assert status == 200 and e["center"] == ids["chunk"]
    assert {n["id"] for n in e["nodes"]} == {ids["doc"], ids["a"], ids["b"]}
    assert len(e["edges"]) == 3 and e["total"] == 3 and e["truncated"] is False
    _, e = call(server, f"/api/expand/{ids['chunk']}?types=MENTIONS")
    assert {n["id"] for n in e["nodes"]} == {ids["a"], ids["b"]}
    _, e = call(server, f"/api/expand/{ids['chunk']}?limit=1")
    assert len(e["edges"]) == 1 and e["total"] == 3 and e["truncated"] is True
    _, e = call(server, f"/api/expand/{ids['c']}")
    assert e["nodes"] == [] and e["edges"] == []
    assert call(server, "/api/expand/999999")[0] == 404


def test_edges_among(viewer):
    server, ids = viewer
    status, r = call(server, "/api/edges", {"ids": [ids["a"], ids["b"]]})
    assert status == 200 and [e["type"] for e in r["edges"]] == ["RELATED_TO"]
    assert call(server, "/api/edges", {"ids": ["x"]})[0] == 400


def test_search(viewer):
    server, ids = viewer
    _, r = call(server, "/api/search?q=" + urllib.request.quote("蒼波"))
    assert [n["id"] for n in r["nodes"]] == [ids["a"]]
    _, r = call(server, "/api/search?q=TIDAL")  # case-insensitive
    assert [n["id"] for n in r["nodes"]] == [ids["b"]]
    _, r = call(server, "/api/search?q=d1")  # id property; names/titles rank first
    assert {n["id"] for n in r["nodes"]} == {ids["doc"], ids["chunk"]}
    _, r = call(server, "/api/search?q=%E9%95%B7%E3%81%84")  # text substring ("長い")
    assert [n["id"] for n in r["nodes"]] == [ids["chunk"]]
    _, r = call(server, f"/api/search?q=%23{ids['c']}")  # internal id
    assert r["nodes"][0]["id"] == ids["c"]
    _, r = call(server, "/api/search?q=")
    assert r["nodes"] == []
    assert len(call(server, "/api/search?q=l")[1]["nodes"]) == 2
    assert len(call(server, "/api/search?q=l&limit=1")[1]["nodes"]) == 1


def test_query_table_and_graph(viewer):
    server, ids = viewer
    status, r = call(
        server,
        "/api/query",
        {"cypher": "MATCH (e:Entity {name: $n})-[r:RELATED_TO]->(f) RETURN e, r, f.name AS other", "params": {"n": "蒼波エネルギー"}},
    )
    assert status == 200
    assert r["columns"] == ["e", "r", "other"] and r["rowCount"] == 1
    e, rel, other = r["rows"][0]
    assert e["kind"] == "node" and e["id"] == ids["a"] and e["labels"] == ["Entity"]
    assert rel == {
        "kind": "relationship", "id": rel["id"], "type": "RELATED_TO",
        "source": ids["a"], "target": ids["b"], "properties": {"description": "partner"},
    }
    assert other == "Tidal Lab"
    # the relationship's other endpoint is pulled into the graph even though only ``f.name`` was returned
    assert {n["id"] for n in r["graph"]["nodes"]} == {ids["a"], ids["b"]}
    assert [x["type"] for x in r["graph"]["edges"]] == ["RELATED_TO"]


def test_query_relationship_only_includes_endpoints(viewer):
    server, ids = viewer
    _, r = call(server, "/api/query", {"cypher": "MATCH ()-[r:HAS_CHUNK]->() RETURN r"})
    assert {n["id"] for n in r["graph"]["nodes"]} == {ids["doc"], ids["chunk"]}


def test_query_path_serialization(viewer):
    server, ids = viewer
    _, r = call(
        server,
        "/api/query",
        {"cypher": "MATCH p = (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity {name: 'Tidal Lab'}) RETURN p, nodes(p) AS ns"},
    )
    (p, ns), = r["rows"]
    assert p["kind"] == "path"
    assert [n["id"] for n in p["nodes"]] == [ids["doc"], ids["chunk"], ids["b"]]
    assert [x["type"] for x in p["relationships"]] == ["HAS_CHUNK", "MENTIONS"]
    assert [n["id"] for n in ns] == [ids["doc"], ids["chunk"], ids["b"]]
    assert len(r["graph"]["nodes"]) == 3 and len(r["graph"]["edges"]) == 2


def test_query_scalar_and_lists(viewer):
    server, _ = viewer
    _, r = call(server, "/api/query", {"cypher": "MATCH (n) RETURN labels(n) AS l, count(*) AS c ORDER BY c DESC, l"})
    assert r["columns"] == ["l", "c"] and r["rows"][0][1] == 3
    assert r["graph"]["nodes"] == []
    _, r = call(server, "/api/query", {"cypher": "MATCH (n:Entity) RETURN collect(n) AS ns"})
    assert len(r["rows"][0][0]) == 3 and len(r["graph"]["nodes"]) == 3
    _, r = call(server, "/api/query", {"cypher": "SHOW INDEXES"})
    assert r["rows"][0][r["columns"].index("type")] == "VECTOR"


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (n:Hacked {x: 1})",
        "MATCH (n:Entity) SET n.name = 'x'",
        "MATCH (n) DETACH DELETE n",
        "MERGE (n:Entity {name: 'new'})",
        "CREATE INDEX i FOR (n:Entity) ON (n.name)",
    ],
)
def test_query_rejects_writes(viewer, cypher):
    server, _ = viewer
    status, r = call(server, "/api/query", {"cypher": cypher})
    assert status == 400 and "read-only" in r["error"]
    _, o = call(server, "/api/overview")
    assert o["nodes"] == 6  # nothing changed


def test_query_errors(viewer):
    server, _ = viewer
    status, r = call(server, "/api/query", {"cypher": "MATCH (n RETURN n"})
    assert status == 400 and r["error"]
    assert call(server, "/api/query", {"cypher": "  "})[0] == 400
    assert call(server, "/api/query", {"cypher": "RETURN 1", "params": [1]})[0] == 400
    assert call(server, "/api/nope")[0] == 404


def test_rejects_foreign_host_header(viewer):
    server, _ = viewer
    assert call(server, "/api/overview", host="evil.example.com")[0] == 403
    assert call(server, "/api/overview", host="localhost:1234")[0] == 200


def test_serialize_defensive():
    n = Node(1, ("A",), {"v": [0.5] * 65, "small": [1, 2, 3], "s": "x" * 500, "f": float("nan")})
    full = serialize(n)
    assert full["properties"]["v"] == "<vector: 65 dims>"
    assert full["properties"]["small"] == [1, 2, 3]
    assert len(full["properties"]["s"]) == 500 and full["properties"]["f"] is None
    assert len(serialize(n, 200)["properties"]["s"]) == 201
    json.dumps(full, allow_nan=False)


def test_port_in_use_is_not_shared(viewer):
    """A second server must not silently share a port (SO_REUSEADDR on Windows would allow it)."""
    server, _ = viewer
    port = server.server_address[1]
    db_path = server.service.path
    with pytest.raises(OSError):
        make_server(db_path, port=port).server_close()
    other = make_server_auto(db_path)
    try:
        assert other.server_address[1] != port
    finally:
        other.server_close()
