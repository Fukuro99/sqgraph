"""Built-in procedures, schema commands, map projection and Neo4j-client-style (LangChain) queries."""

from __future__ import annotations

import pytest

from sqgraph import CypherRuntimeError, CypherSyntaxError, CypherUnsupportedError, SqgraphError
from sqgraph.cypher.compiler import Compiler
from sqgraph.cypher.parser import parse


def q(db, cypher, **params):
    return db.query(cypher, params)


# ------------------------------------------------------------------ db.* procedures


def test_db_labels_and_relationship_types(db, social):
    rows = q(db, "CALL db.labels() YIELD label RETURN label ORDER BY label")
    assert [r["label"] for r in rows] == ["Company", "Employee", "Person"]
    rows = q(db, "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
    assert [r["relationshipType"] for r in rows] == ["KNOWS", "WORKS_AT"]
    assert q(db, "CALL db.labels()") == [{"label": "Company"}, {"label": "Employee"}, {"label": "Person"}] or len(q(db, "CALL db.labels()")) == 3
    keys = q(db, "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey")
    assert [r["propertyKey"] for r in keys] == ["active", "age", "name", "since", "tags"]


def test_call_yield_where_and_alias(db, social):
    rows = q(db, "CALL db.labels() YIELD label AS l WHERE l STARTS WITH 'P' RETURN l")
    assert rows == [{"l": "Person"}]
    rows = q(db, "CALL db.labels() YIELD label RETURN count(*) AS c")
    assert rows == [{"c": 3}]


def test_call_combined_with_preceding_clauses(db, social):
    rows = q(db, "MATCH (c:Company {name:'Acme'}) CALL db.labels() YIELD label RETURN c.name AS c, count(label) AS n")
    assert rows == [{"c": "Acme", "n": 3}]


def test_unknown_procedure_and_bad_yield(db):
    with pytest.raises(CypherUnsupportedError, match="not supported"):
        q(db, "CALL apoc.meta.stats() YIELD labels RETURN labels")
    with pytest.raises(CypherSyntaxError):
        q(db, "CALL db.labels() YIELD nothing RETURN nothing")
    with pytest.raises(CypherSyntaxError):
        q(db, "CALL db.labels() YIELD label, label AS label RETURN label")


def test_index_creation_procedures(db):
    q(db, "CALL db.index.vector.createNodeIndex('emb', 'Chunk', 'embedding', 3, 'cosine')")
    q(db, "CALL db.index.fulltext.createNodeIndex('ft', ['Chunk'], ['text'])")
    assert [i.name for i in db.vector_indexes()] == ["emb"]
    assert db.vector_indexes()[0].dimensions == 3
    assert [i.name for i in db.fulltext_indexes()] == ["ft"]
    with pytest.raises(CypherRuntimeError):
        q(db, "CALL db.index.vector.createNodeIndex('emb', 'Chunk', 'embedding', 3, 'cosine')")
    q(db, "CALL db.index.vector.createNodeIndex($n, $l, $p, toInteger($d), $s)", n="e2", l="X", p="v", d=4, s="euclidean")
    assert next(i for i in db.vector_indexes() if i.name == "e2").similarity == "euclidean"


def test_set_vector_property_yields_node(db):
    db.create_vector_index("emb", "E", "embedding", 2)
    q(db, "CREATE (:E {name: 'a'})")
    rows = q(db, "MATCH (n:E) CALL db.create.setVectorProperty(n, 'embedding', [1.0, 0.0]) YIELD node RETURN node.name AS n")
    assert rows == [{"n": "a"}]
    assert db.vector_search("emb", [1.0, 0.0])[0][0]["name"] == "a"


# ------------------------------------------------------------------ property indexes and DDL


def test_create_property_index_via_cypher_and_used_in_queries(db):
    db.create_nodes([("P", {"name": f"n{i}"}) for i in range(200)])
    q(db, "CREATE INDEX FOR (n:P) ON (n.name)")
    q(db, "CREATE INDEX IF NOT EXISTS FOR (n:P) ON (n.name)")
    assert db.list_indexes()[0]["type"] == "RANGE"
    index_name = db.list_indexes()[0]["name"]
    cy = "MATCH (n {name: 'n7'}) RETURN n.name AS name"
    comp = Compiler(db, {}, cy)
    seg = comp.compile_segment(parse(cy).parts[0].clauses)
    assert "'$.name'" in seg.sql  # literal path so that the expression index can match
    plan = " ".join(r[3] for r in db._conn.execute("EXPLAIN QUERY PLAN " + seg.sql, seg.binds))
    assert f"idx_p_{index_name}" in plan
    assert q(db, cy) == [{"name": "n7"}]
    q(db, f"DROP INDEX {index_name}")
    assert db.list_indexes() == []


def test_property_index_ddl_errors(db):
    with pytest.raises(CypherUnsupportedError):
        q(db, "CREATE INDEX FOR (n:P) ON (n.a, n.b)")
    with pytest.raises(SqgraphError):
        q(db, "CREATE INDEX FOR (n:P) ON (n.`bad key`)")


def test_analyze_is_available(db, social):
    db.analyze()
    assert q(db, "MATCH (n:Person) RETURN count(*) AS c") == [{"c": 4}]


# ------------------------------------------------------------------ map projection & Neo4j client queries


def test_map_projection(db, social):
    rows = q(db, "MATCH (n:Person {name:'Alice'}) RETURN n {.name, .age, extra: 1 + 1, missing: n.zzz} AS m")
    assert rows == [{"m": {"name": "Alice", "age": 30, "extra": 2, "missing": None}}]
    rows = q(db, "MATCH (n:Person {name:'Alice'}) RETURN n {.*} AS m")
    assert rows == [{"m": {"name": "Alice", "age": 30, "active": True}}]
    rows = q(db, "MATCH (n:Person {name:'Alice'}) RETURN n {.*, age: null, name: 'X'} AS m")
    assert rows == [{"m": {"name": "X", "age": None, "active": True}}]
    rows = q(db, "MATCH (n:Person {name:'Alice'}) WITH n, 5 AS five RETURN n {.name, five} AS m")
    assert rows == [{"m": {"name": "Alice", "five": 5}}]
    rows = q(db, "OPTIONAL MATCH (n:Nope) RETURN n {.name} AS m")
    assert rows == [{"m": None}]
    rows = q(db, "MATCH ()-[r:KNOWS {since: 2015}]->() RETURN r {.since} AS m")
    assert rows == [{"m": {"since": 2015}}]


def test_langchain_neo4j_vector_style_retrieval_query(db):
    q(
        db,
        # the form used by LangChain's Neo4jVector (property without parentheses)
        "CREATE VECTOR INDEX vector IF NOT EXISTS FOR (m:Chunk) ON m.embedding "
        "OPTIONS {indexConfig: {`vector.dimensions`: 2, `vector.similarity_function`: 'cosine'}}",
    )
    q(
        db,
        "UNWIND $data AS row "
        "MERGE (c:`Chunk` {id: row.id}) "
        "WITH c, row "
        "CALL db.create.setVectorProperty(c, 'embedding', row.embedding) YIELD node "
        "SET c.text = row.text, c += row.metadata "
        "RETURN count(*) AS n",
        data=[
            {"id": "1", "text": "alpha", "embedding": [1.0, 0.0], "metadata": {"source": "a.txt"}},
            {"id": "2", "text": "beta", "embedding": [0.0, 1.0], "metadata": {"source": "b.txt"}},
        ],
    )
    rows = q(
        db,
        "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
        "RETURN node.`text` AS text, score, node {.*, `text`: Null, `embedding`: Null, id: Null } AS metadata "
        "ORDER BY score DESC",
        index="vector",
        k=2,
        embedding=[1.0, 0.1],
    )
    assert [r["text"] for r in rows] == ["alpha", "beta"]
    assert rows[0]["metadata"]["source"] == "a.txt"
    assert rows[0]["metadata"]["text"] is None and rows[0]["metadata"]["embedding"] is None
    assert rows[0]["score"] > rows[1]["score"]


def test_neo4j_graph_style_generic_queries(db, social):
    # typical helper queries used by Neo4j client libraries
    rows = q(db, "MATCH (n) RETURN labels(n) AS labels, count(*) AS c ORDER BY c DESC, labels")
    assert rows[0] == {"labels": ["Person"], "c": 3}
    rows = q(db, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")
    assert rows == [{"t": "KNOWS", "c": 3}, {"t": "WORKS_AT", "c": 3}]
    rows = q(db, "MATCH (n:Person) WHERE n.name = $name RETURN n{.*, id: id(n)} AS node", name="Bob")
    assert rows[0]["node"]["name"] == "Bob" and rows[0]["node"]["id"] == social["bob"]
    rows = q(db, "MATCH (n:Person)-[r:KNOWS]->(m) RETURN n.name AS start, type(r) AS type, m.name AS end ORDER BY start LIMIT $limit", limit=2)
    assert rows == [
        {"start": "Alice", "type": "KNOWS", "end": "Bob"},
        {"start": "Bob", "type": "KNOWS", "end": "Carol"},
    ]
    rows = q(db, "MATCH (n:Person) RETURN collect({name: n.name, age: n.age}) AS people")
    assert sorted(p["name"] for p in rows[0]["people"]) == ["Alice", "Bob", "Carol", "Dave"]
    rows = q(db, "MATCH (n:Person) WITH n ORDER BY n.age RETURN collect(n.name)[0..2] AS youngest")
    assert rows == [{"youngest": ["Bob", "Alice"]}]


def test_returning_maps_and_lists_with_entities(db, social):
    rows = q(db, "MATCH (a:Person {name:'Alice'})-[r:WORKS_AT]->(c) RETURN {person: a, rel: r, company: c} AS row")
    row = rows[0]["row"]
    assert row["person"].get("name") == "Alice"
    assert row["rel"].type == "WORKS_AT"
    assert row["company"].get("name") == "Acme"
    rows = q(db, "MATCH (p:Person) WITH collect(p) AS ps RETURN ps[0] AS first, size(ps) AS n")
    assert rows[0]["first"].get("name") in {"Alice", "Bob", "Carol", "Dave"} and rows[0]["n"] == 4
    with pytest.raises(CypherUnsupportedError):
        q(db, "MATCH (p:Person) RETURN [p, p.name] AS mixed")


def test_property_index_without_parentheses(db):
    q(db, "CREATE INDEX FOR (n:P) ON n.name")
    assert [i["properties"] for i in db.list_indexes()] == [["name"]]
