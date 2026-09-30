"""End-to-end GraphRAG scenario:

Document -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Entity -[:RELATED_TO]-> Entity,
Entity -[:IN_COMMUNITY]-> Community, plus vector and fulltext indexes.
"""

from __future__ import annotations

import pytest

from sqgraph import GraphDB

ENTITIES = {
    "OpenAI": [1.0, 0.0, 0.0],
    "GPT-4": [0.9, 0.1, 0.0],
    "Microsoft": [0.5, 0.5, 0.0],
    "Azure": [0.2, 0.8, 0.0],
    "Paris": [0.0, 0.0, 1.0],
    "France": [0.0, 0.1, 0.9],
}
RELATIONS = [
    ("OpenAI", "GPT-4", "created"),
    ("OpenAI", "Microsoft", "partner"),
    ("Microsoft", "Azure", "owns"),
    ("Paris", "France", "capital of"),
]
CHUNKS = [
    ("d1", "c1", "OpenAI created GPT-4 as a large language model", ["OpenAI", "GPT-4"]),
    ("d1", "c2", "Microsoft partners with OpenAI and runs the Azure cloud", ["Microsoft", "OpenAI", "Azure"]),
    ("d2", "c3", "Paris is the capital of France", ["Paris", "France"]),
    ("d2", "c4", "Azure offers managed cloud services", ["Azure"]),
]
COMMUNITIES = {
    "com1": ("AI industry", "Companies and models in the AI industry", ["OpenAI", "GPT-4", "Microsoft", "Azure"]),
    "com2": ("Geography", "European capitals and countries", ["Paris", "France"]),
}


@pytest.fixture
def rag(db):
    db.query(
        "CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}"
    )
    db.query("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]")
    db.query("CREATE INDEX FOR (n:Entity) ON (n.name)")
    db.query("UNWIND $docs AS d MERGE (:Document {id: d})", {"docs": ["d1", "d2"]})
    db.query(
        "UNWIND $rows AS row "
        "MATCH (d:Document {id: row.doc}) "
        "CREATE (c:Chunk {id: row.id, text: row.text}) CREATE (d)-[:HAS_CHUNK]->(c) "
        "WITH c, row UNWIND row.entities AS name "
        "MERGE (e:Entity {name: name}) CREATE (c)-[:MENTIONS]->(e)",
        {"rows": [{"doc": d, "id": c, "text": t, "entities": e} for d, c, t, e in CHUNKS]},
    )
    db.query(
        "UNWIND $rows AS row MATCH (e:Entity {name: row.name}) "
        "CALL db.create.setNodeVectorProperty(e, 'embedding', row.vec)",
        {"rows": [{"name": n, "vec": v} for n, v in ENTITIES.items()]},
    )
    db.query(
        "UNWIND $rels AS r MATCH (a:Entity {name: r.a}), (b:Entity {name: r.b}) "
        "MERGE (a)-[x:RELATED_TO]->(b) SET x.description = r.d",
        {"rels": [{"a": a, "b": b, "d": d} for a, b, d in RELATIONS]},
    )
    for cid, (title, summary, members) in COMMUNITIES.items():
        db.query(
            "MERGE (c:Community {id: $id}) SET c.title = $title, c.summary = $summary "
            "WITH c UNWIND $members AS m MATCH (e:Entity {name: m}) MERGE (e)-[:IN_COMMUNITY]->(c)",
            {"id": cid, "title": title, "summary": summary, "members": members},
        )
    return db


LOCAL_SEARCH = """
CALL db.index.vector.queryNodes('entity_emb', $k, $q) YIELD node, score
OPTIONAL MATCH (node)-[:RELATED_TO*1..2]-(nb:Entity)
WITH node, score, collect(DISTINCT nb) AS neighbors
OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(x:Entity) WHERE x = node OR x IN neighbors
WITH node, score, neighbors, collect(DISTINCT c.text) AS chunks
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(com:Community)
RETURN node.name AS entity, score,
       [n IN neighbors | n.name] AS neighbors,
       chunks,
       collect(DISTINCT com.summary) AS communities
ORDER BY score DESC
"""


def test_graph_was_built_as_expected(rag):
    assert rag.query("MATCH (n) RETURN count(*) AS c")[0]["c"] == 2 + 4 + 6 + 2
    rows = rag.query("MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d.id AS d, count(c) AS n ORDER BY d")
    assert rows == [{"d": "d1", "n": 2}, {"d": "d2", "n": 2}]
    assert rag.query("MATCH (e:Entity) RETURN count(*) AS c") == [{"c": 6}]
    assert rag.query("MATCH ()-[m:MENTIONS]->() RETURN count(*) AS c") == [{"c": 8}]
    assert rag.query("MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community) RETURN c.title AS t, count(e) AS n ORDER BY t") == [
        {"t": "AI industry", "n": 4},
        {"t": "Geography", "n": 2},
    ]
    assert rag._conn.execute("SELECT count(*) FROM vec_entity_emb").fetchone()[0] == 6
    assert not any("embedding" in r[0] for r in rag._conn.execute("SELECT props FROM nodes"))


def test_local_search_vector_topk_expand_hops_collect_chunks_and_communities(rag):
    rows = rag.query(LOCAL_SEARCH, {"k": 2, "q": [1.0, 0.05, 0.0]})
    assert [r["entity"] for r in rows] == ["OpenAI", "GPT-4"]
    assert rows[0]["score"] >= rows[1]["score"] > 0.9
    openai, gpt4 = rows
    assert sorted(openai["neighbors"]) == ["Azure", "GPT-4", "Microsoft"]
    assert sorted(gpt4["neighbors"]) == ["Microsoft", "OpenAI"]
    assert sorted(openai["chunks"]) == sorted(
        [CHUNKS[0][2], CHUNKS[1][2], CHUNKS[3][2]]
    )
    assert sorted(gpt4["chunks"]) == sorted([CHUNKS[0][2], CHUNKS[1][2]])
    assert openai["communities"] == ["Companies and models in the AI industry"]
    assert gpt4["communities"] == ["Companies and models in the AI industry"]


def test_local_search_on_a_different_topic(rag):
    rows = rag.query(LOCAL_SEARCH, {"k": 1, "q": [0.0, 0.05, 1.0]})
    assert len(rows) == 1
    (r,) = rows
    assert r["entity"] in ("Paris", "France")
    other = "France" if r["entity"] == "Paris" else "Paris"
    assert r["neighbors"] == [other]
    assert r["chunks"] == [CHUNKS[2][2]]
    assert r["communities"] == ["European capitals and countries"]


def test_local_search_with_relationship_details(rag):
    rows = rag.query(
        "CALL db.index.vector.queryNodes('entity_emb', 1, $q) YIELD node, score "
        "MATCH (node)-[r:RELATED_TO]-(nb:Entity) "
        "RETURN node.name AS entity, type(r) AS rel, r.description AS description, nb.name AS other, "
        "CASE WHEN startNode(r) = node THEN 'out' ELSE 'in' END AS direction "
        "ORDER BY other",
        {"q": [1.0, 0.0, 0.0]},
    )
    assert rows == [
        {"entity": "OpenAI", "rel": "RELATED_TO", "description": "created", "other": "GPT-4", "direction": "out"},
        {"entity": "OpenAI", "rel": "RELATED_TO", "description": "partner", "other": "Microsoft", "direction": "out"},
    ]


def test_local_search_matches_python_api_composition(rag):
    q = [0.4, 0.6, 0.0]
    api_hits = rag.vector_search("entity_emb", q, k=3)
    cypher_hits = rag.query(
        "CALL db.index.vector.queryNodes('entity_emb', 3, $q) YIELD node, score RETURN node, score ORDER BY score DESC",
        {"q": q},
    )
    assert [n.id for n, _ in api_hits] == [r["node"].id for r in cypher_hits]
    seed = api_hits[0][0]
    api_neighbors = {n["name"] for n in rag.neighbors(seed.id, types=["RELATED_TO"], direction="both", depth=2, labels=["Entity"])}
    rows = rag.query(
        "MATCH (s:Entity) WHERE id(s) = $id MATCH (s)-[:RELATED_TO*1..2]-(nb:Entity) RETURN DISTINCT nb.name AS n",
        {"id": seed.id},
    )
    assert {r["n"] for r in rows} == api_neighbors


def test_hybrid_fulltext_then_graph_expansion(rag):
    rows = rag.query(
        "CALL db.index.fulltext.queryNodes('chunk_text', 'capital') YIELD node AS chunk, score "
        "MATCH (chunk)-[:MENTIONS]->(e:Entity)-[:IN_COMMUNITY]->(com:Community) "
        "RETURN chunk.id AS chunk, collect(e.name) AS entities, com.title AS community ORDER BY chunk"
    )
    assert len(rows) == 1
    assert rows[0]["chunk"] == "c3" and sorted(rows[0]["entities"]) == ["France", "Paris"]
    assert rows[0]["community"] == "Geography"
    # chunk text search feeds an entity vector lookup: which entities are close to those mentioned in matching chunks?
    rows = rag.query(
        "CALL db.index.fulltext.queryNodes('chunk_text', 'Azure') YIELD node AS chunk "
        "MATCH (chunk)-[:MENTIONS]->(e:Entity {name: 'Azure'}) "
        "WITH DISTINCT e "
        "CALL db.index.vector.queryNodes('entity_emb', 2, [0.2, 0.8, 0.0]) YIELD node, score "
        "RETURN e.name AS seed, node.name AS hit ORDER BY score DESC"
    )
    assert [r["hit"] for r in rows][0] == "Azure"


def test_global_search_style_community_summaries(rag):
    rows = rag.query(
        "MATCH (c:Community)<-[:IN_COMMUNITY]-(e:Entity) "
        "WITH c, collect(e.name) AS members "
        "RETURN c.id AS id, c.summary AS summary, size(members) AS n ORDER BY n DESC"
    )
    assert rows == [
        {"id": "com1", "summary": "Companies and models in the AI industry", "n": 4},
        {"id": "com2", "summary": "European capitals and countries", "n": 2},
    ]
    top = rag.query("MATCH (e:Entity)-[r:RELATED_TO]-() RETURN e.name AS e, count(r) AS degree ORDER BY degree DESC, e LIMIT 2")
    assert top == [{"e": "Microsoft", "degree": 2}, {"e": "OpenAI", "degree": 2}]


def test_provenance_chain_document_to_entity_paths(rag):
    rows = rag.query(
        "MATCH p = (d:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity {name: 'GPT-4'}) "
        "RETURN d.id AS doc, length(p) AS hops, [n IN nodes(p) | labels(n)[0]] AS kinds"
    )
    assert rows == [{"doc": "d1", "hops": 2, "kinds": ["Document", "Chunk", "Entity"]}]


def test_idempotent_ingestion(rag):
    before = rag.stats()
    rag.query("UNWIND $rows AS row MERGE (e:Entity {name: row}) ", {"rows": list(ENTITIES)})
    rag.query(
        "UNWIND $rels AS r MATCH (a:Entity {name: r.a}), (b:Entity {name: r.b}) MERGE (a)-[x:RELATED_TO]->(b)",
        {"rels": [{"a": a, "b": b} for a, b, _ in RELATIONS]},
    )
    assert rag.stats() == before


def test_deleting_an_entity_cleans_up_indexes(rag):
    rag.query("MATCH (e:Entity {name: 'Azure'}) DETACH DELETE e")
    assert rag._conn.execute("SELECT count(*) FROM vec_entity_emb").fetchone()[0] == 5
    hits = [n["name"] for n, _ in rag.vector_search("entity_emb", [0.2, 0.8, 0.0], k=10)]
    assert "Azure" not in hits
    rows = rag.query(LOCAL_SEARCH, {"k": 1, "q": [1.0, 0.0, 0.0]})
    assert "Azure" not in rows[0]["neighbors"]


def test_graphrag_flow_survives_reopen_read_only(tmp_path):
    path = tmp_path / "rag.db"
    with GraphDB(path) as db:
        db.create_vector_index("entity_emb", "Entity", "embedding", 3)
        db.create_fulltext_index("chunk_text", ["Chunk"], ["text"])
        for name, vec in ENTITIES.items():
            db.create_node("Entity", {"name": name, "embedding": vec})
        c = db.create_node("Chunk", {"text": "OpenAI made GPT-4"})
        for name in ("OpenAI", "GPT-4"):
            eid = db.query("MATCH (e:Entity {name: $n}) RETURN id(e) AS i", {"n": name})[0]["i"]
            db.create_edge(c.id, eid, "MENTIONS")
    with GraphDB(path, read_only=True) as ro:
        rows = ro.query(
            "CALL db.index.vector.queryNodes('entity_emb', 1, [1.0, 0.0, 0.0]) YIELD node, score "
            "MATCH (c:Chunk)-[:MENTIONS]->(node) RETURN node.name AS e, c.text AS text"
        )
        assert rows == [{"e": "OpenAI", "text": "OpenAI made GPT-4"}]
        rows = ro.query(
            "CALL db.index.fulltext.queryNodes('chunk_text', 'GPT-4') YIELD node "
            "MATCH (node)-[:MENTIONS]->(e) RETURN e.name AS e ORDER BY e"
        )
        assert [r["e"] for r in rows] == ["GPT-4", "OpenAI"]
