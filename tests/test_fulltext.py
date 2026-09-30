from __future__ import annotations

import pytest

from sqgraph import CypherRuntimeError, CypherSyntaxError, GraphDB, Node, SqgraphError
from sqgraph.ftsquery import lucene_to_fts5


def q(db, cypher, **params):
    return db.query(cypher, params)


def fts_rows(db, name="ft"):
    return db._conn.execute(f"SELECT count(*) FROM fts_{name}").fetchone()[0]


@pytest.fixture
def fdb(db):
    db.create_fulltext_index("ft", ["Doc"], ["title", "body"])
    db.create_node("Doc", {"title": "Graph databases", "body": "A graph database stores nodes and relationships"})
    db.create_node("Doc", {"title": "Vector search", "body": "Vectors enable similarity search over embeddings"})
    db.create_node("Doc", {"title": "GraphRAG", "body": "Graph and vector search combined for retrieval augmented generation"})
    db.create_node("Other", {"title": "graph but wrong label", "body": "graph graph graph"})
    return db


def titles(results):
    return [n["title"] for n, _ in results]


# ------------------------------------------------------------------ index management


def test_create_fulltext_index_api(db):
    ix = db.create_fulltext_index("ft", ["A", "B"], ["p1", "p2"])
    assert (ix.name, ix.labels, ix.properties) == ("ft", ("A", "B"), ("p1", "p2"))
    row = db._conn.execute("SELECT name, labels, properties FROM _fulltext_indexes").fetchone()
    assert row == ("ft", '["A", "B"]', '["p1", "p2"]')
    with pytest.raises(SqgraphError, match="already exists"):
        db.create_fulltext_index("ft", ["A"], ["p1"])
    assert db.create_fulltext_index("ft", ["A"], ["p1"], if_not_exists=True) == ix
    assert db.create_fulltext_index("single", "A", "p1").labels == ("A",)


def test_fulltext_index_validation(db):
    for bad in ("a b", "x;y", "1x", "", "a'b"):
        with pytest.raises(SqgraphError):
            db.create_fulltext_index(bad, ["A"], ["p"])
    with pytest.raises(SqgraphError):
        db.create_fulltext_index("ok", [], ["p"])
    with pytest.raises(SqgraphError):
        db.create_fulltext_index("ok", ["A"], [])
    with pytest.raises(SqgraphError):
        db.create_fulltext_index("ok", ["A"], ["p"], tokenizer="evil'); DROP TABLE nodes; --")
    assert db.fulltext_indexes() == []


def test_create_fulltext_index_via_cypher_ddl(db):
    q(db, "CREATE FULLTEXT INDEX ft IF NOT EXISTS FOR (n:Article|Note) ON EACH [n.title, n.text]")
    (ix,) = db.fulltext_indexes()
    assert (ix.name, ix.labels, ix.properties) == ("ft", ("Article", "Note"), ("title", "text"))
    q(db, "CREATE FULLTEXT INDEX ft IF NOT EXISTS FOR (n:Article|Note) ON EACH [n.title, n.text]")
    with pytest.raises(SqgraphError):
        q(db, "CREATE FULLTEXT INDEX ft FOR (n:Article) ON EACH [n.title]")
    with pytest.raises(CypherSyntaxError):
        q(db, "CREATE FULLTEXT INDEX ft2 FOR (n:Article) ON [n.title]")


def test_drop_fulltext_index(fdb):
    q(fdb, "DROP INDEX ft")
    assert fdb.fulltext_indexes() == []
    assert fdb._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'fts_ft'").fetchone()[0] == 0
    fdb.create_node("Doc", {"title": "after drop"})  # writes no longer touch the dropped index


def test_existing_nodes_are_indexed_on_creation(db):
    db.create_node("Doc", {"title": "old one"})
    db.create_node("Doc", {"title": "old two"})
    db.create_node("Other", {"title": "old three"})
    db.create_fulltext_index("ft", ["Doc"], ["title"])
    assert fts_rows(db) == 2
    assert len(db.fulltext_search("ft", "old")) == 2


# ------------------------------------------------------------------ search


def test_search_ranking_and_scores(fdb):
    res = fdb.fulltext_search("ft", "graph")
    assert all(isinstance(n, Node) for n, _ in res)
    assert set(titles(res)) == {"Graph databases", "GraphRAG"} or set(titles(res)) == {"Graph databases", "GraphRAG"}
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)
    res = fdb.fulltext_search("ft", "vector search")
    assert titles(res)[0] == "Vector search"  # matches both terms in title and body
    assert len(res) == 2  # default OR semantics: the GraphRAG doc has both words in its body too
    assert res[0][1] > res[1][1]


def test_search_respects_label_restriction_and_k(fdb):
    assert "graph but wrong label" not in titles(fdb.fulltext_search("ft", "graph"))
    assert len(fdb.fulltext_search("ft", "search", k=1)) == 1
    assert len(fdb.fulltext_search("ft", "search", k=10)) == 2
    assert fdb.fulltext_search("ft", "zzzznothing") == []
    assert fdb.fulltext_search("ft", "") == []
    assert fdb.fulltext_search("ft", "   ") == []


def test_query_syntax_variants(fdb):
    assert titles(fdb.fulltext_search("ft", "similarity AND embeddings")) == ["Vector search"]
    assert set(titles(fdb.fulltext_search("ft", "database OR retrieval"))) == {"Graph databases", "GraphRAG"}
    assert titles(fdb.fulltext_search("ft", '"graph and vector"')) == ["GraphRAG"]
    assert set(titles(fdb.fulltext_search("ft", "search NOT graph"))) == {"Vector search"}
    assert set(titles(fdb.fulltext_search("ft", "search -graph"))) == {"Vector search"}
    assert set(titles(fdb.fulltext_search("ft", "embed*"))) == {"Vector search"}
    assert titles(fdb.fulltext_search("ft", "title:graphrag")) == ["GraphRAG"]
    assert set(titles(fdb.fulltext_search("ft", "(graph OR vector) AND retrieval"))) == {"GraphRAG"}


def test_special_characters_never_cause_syntax_errors(fdb):
    for nasty in ('C++', "foo:bar:baz", "'; DROP TABLE nodes; --", '"unbalanced', "(((", ")))", "AND", "OR OR", "NOT",
                  "*", "a^b~c", "title:", ":x", "graph OR", "- -", '""', "\\", "{}", "graph)"):
        res = fdb.fulltext_search("ft", nasty)
        assert isinstance(res, list)
    assert fdb.stats()["nodes"] == 4
    assert lucene_to_fts5("") == "" and lucene_to_fts5("  ") == ""
    assert lucene_to_fts5("a b") == '"a" OR "b"'
    assert lucene_to_fts5("a AND b") == '"a" AND "b"'
    assert lucene_to_fts5("a -b") == '("a") NOT ("b")'


def test_list_and_numeric_properties_are_indexed(db):
    db.create_fulltext_index("ft", ["T"], ["tags", "year"])
    db.create_node("T", {"tags": ["alpha", "beta"], "year": 1999})
    assert len(db.fulltext_search("ft", "beta")) == 1
    assert len(db.fulltext_search("ft", "1999")) == 1


def test_unicode_and_cjk(db):
    db.create_fulltext_index("ft", ["Doc"], ["text"], tokenizer="trigram")
    db.create_node("Doc", {"text": "グラフデータベースは関係を表現します"})
    db.create_node("Doc", {"text": "ベクトル検索は類似度を計算します"})
    res = db.fulltext_search("ft", "データベース")
    assert len(res) == 1 and "グラフ" in res[0][0]["text"]
    db2 = GraphDB()
    db2.create_fulltext_index("ft", ["Doc"], ["text"])
    db2.create_node("Doc", {"text": "Café résumé naïve"})
    assert len(db2.fulltext_search("ft", "cafe")) == 1  # unicode61 folds diacritics
    db2.close()


# ------------------------------------------------------------------ sync


def test_sync_on_update(fdb):
    n = fdb.query("MATCH (n:Doc {title:'GraphRAG'}) RETURN n")[0]["n"]
    fdb.update_node(n.id, {"body": "completely different words: banana"})
    assert fdb.fulltext_search("ft", "banana")[0][0].id == n.id
    assert n.id not in [x.id for x, _ in fdb.fulltext_search("ft", "retrieval")]
    fdb.update_node(n.id, {"title": None, "body": None})
    assert fts_rows(fdb) == 2  # no indexed properties left -> no row


def test_sync_on_delete(fdb):
    n = fdb.query("MATCH (n:Doc {title:'Vector search'}) RETURN n")[0]["n"]
    fdb.delete_node(n.id)
    assert fdb.fulltext_search("ft", "similarity") == []
    assert fts_rows(fdb) == 2
    q(fdb, "MATCH (n:Doc) DETACH DELETE n")
    assert fts_rows(fdb) == 0
    assert fdb.fulltext_search("ft", "graph") == []


def test_sync_on_cypher_writes(fdb):
    q(fdb, "CREATE (:Doc {title: 'Brand new', body: 'zebra crossing'})")
    assert titles(fdb.fulltext_search("ft", "zebra")) == ["Brand new"]
    q(fdb, "MATCH (n:Doc {title: 'Brand new'}) SET n.body = 'giraffe'")
    assert fdb.fulltext_search("ft", "zebra") == []
    assert titles(fdb.fulltext_search("ft", "giraffe")) == ["Brand new"]
    q(fdb, "MATCH (n:Doc {title: 'Brand new'}) SET n += {body: 'okapi'}")
    assert titles(fdb.fulltext_search("ft", "okapi")) == ["Brand new"]
    q(fdb, "MATCH (n:Doc {title: 'Brand new'}) REMOVE n.body, n.title")
    assert fdb.fulltext_search("ft", "okapi") == []
    q(fdb, "MERGE (n:Doc {title: 'Merged'}) ON CREATE SET n.body = 'quokka'")
    assert titles(fdb.fulltext_search("ft", "quokka")) == ["Merged"]
    q(fdb, "MERGE (n:Doc {title: 'Merged'}) ON MATCH SET n.body = 'wombat'")
    assert titles(fdb.fulltext_search("ft", "wombat")) == ["Merged"]
    assert fdb.fulltext_search("ft", "quokka") == []
    q(fdb, "MATCH (n:Doc {title: 'Merged'}) DELETE n")
    assert fdb.fulltext_search("ft", "wombat") == []


def test_sync_on_label_changes(fdb):
    q(fdb, "MATCH (n:Other) SET n:Doc")
    assert "graph but wrong label" in titles(fdb.fulltext_search("ft", "graph"))
    q(fdb, "MATCH (n:Other) REMOVE n:Doc")
    assert "graph but wrong label" not in titles(fdb.fulltext_search("ft", "graph"))
    assert fts_rows(fdb) == 3


def test_multi_index_and_multi_label_sync(db):
    db.create_fulltext_index("a", ["A", "B"], ["t"])
    db.create_fulltext_index("b", ["B"], ["t", "u"])
    n = db.create_node(["B"], {"t": "shared", "u": "only in b"})
    assert len(db.fulltext_search("a", "shared")) == 1
    assert len(db.fulltext_search("b", "shared")) == 1
    assert len(db.fulltext_search("a", "only")) == 0
    assert len(db.fulltext_search("b", "only")) == 1
    db.delete_node(n.id)
    assert fts_rows(db, "a") == 0 and fts_rows(db, "b") == 0


def test_bulk_creates_are_indexed(db):
    db.create_fulltext_index("ft", ["Doc"], ["t"])
    db.create_nodes([("Doc", {"t": f"word{i} common"}) for i in range(50)])
    assert len(db.fulltext_search("ft", "common", k=100)) == 50
    assert len(db.fulltext_search("ft", "word7")) == 1
    q(db, "UNWIND range(1, 5) AS i CREATE (:Doc {t: 'unwound ' + toString(i)})")
    assert len(db.fulltext_search("ft", "unwound", k=100)) == 5


def test_fulltext_rollback(fdb):
    with pytest.raises(RuntimeError):
        with fdb.transaction():
            fdb.create_node("Doc", {"title": "ghost", "body": "phantom"})
            raise RuntimeError
    assert fdb.fulltext_search("ft", "phantom") == []
    with pytest.raises(CypherRuntimeError):
        q(fdb, "CREATE (:Doc {title: 'ghost2', body: 'phantom2'}) WITH 1 AS x MATCH (n:Doc) SET n.bad = {m: 1}")
    assert fdb.fulltext_search("ft", "phantom2") == []
    assert fts_rows(fdb) == 3


# ------------------------------------------------------------------ Cypher procedure


def test_call_fulltext_query_nodes(fdb):
    rows = q(fdb, "CALL db.index.fulltext.queryNodes('ft', 'similarity') YIELD node, score RETURN node.title AS t, score")
    assert len(rows) == 1 and rows[0]["t"] == "Vector search" and rows[0]["score"] > 0
    rows = q(fdb, "CALL db.index.fulltext.queryNodes('ft', $q) YIELD node, score RETURN node.title AS t, score ORDER BY score DESC", q="graph")
    api = fdb.fulltext_search("ft", "graph", k=10)
    assert [r["t"] for r in rows] == titles(api)
    assert [r["score"] for r in rows] == pytest.approx([s for _, s in api])
    rows = q(fdb, "CALL db.index.fulltext.queryNodes('ft', 'graph') YIELD node AS d, score AS s WHERE s > 0 RETURN d.title AS t")
    assert len(rows) == 2


def test_call_fulltext_with_limit_option_and_default_order(fdb):
    rows = q(fdb, "CALL db.index.fulltext.queryNodes('ft', 'graph OR search OR vector', {limit: 1}) YIELD node, score RETURN node.title AS t")
    assert len(rows) == 1
    rows = q(fdb, "CALL db.index.fulltext.queryNodes('ft', 'graph OR search OR vector') YIELD node, score RETURN score")
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_call_fulltext_combined_with_match(fdb):
    author = fdb.create_node("Person", {"name": "Ann"})
    doc = fdb.query("MATCH (d:Doc {title:'GraphRAG'}) RETURN id(d) AS i")[0]["i"]
    fdb.create_edge(author.id, doc, "WROTE")
    rows = q(
        fdb,
        "CALL db.index.fulltext.queryNodes('ft', 'graph') YIELD node, score "
        "OPTIONAL MATCH (a:Person)-[:WROTE]->(node) "
        "RETURN node.title AS t, a.name AS author ORDER BY t",
    )
    assert rows == [{"t": "Graph databases", "author": None}, {"t": "GraphRAG", "author": "Ann"}]


def test_call_fulltext_errors(fdb):
    with pytest.raises(CypherRuntimeError, match="no such fulltext"):
        q(fdb, "CALL db.index.fulltext.queryNodes('nope', 'x') YIELD node RETURN node")
    with pytest.raises(CypherSyntaxError):
        q(fdb, "CALL db.index.fulltext.queryNodes('ft') YIELD node RETURN node")
    with pytest.raises(SqgraphError, match="no such fulltext"):
        fdb.fulltext_search("nope", "x")
    assert q(fdb, "CALL db.index.fulltext.queryNodes('ft', '') YIELD node RETURN node") == []
    assert q(fdb, "CALL db.index.fulltext.queryNodes('ft', '\"unbalanced') YIELD node RETURN count(*) AS c")[0]["c"] >= 0


def test_fulltext_persists_and_read_only(tmp_path):
    path = tmp_path / "f.db"
    with GraphDB(path) as db:
        db.create_fulltext_index("ft", ["Doc"], ["t"])
        db.create_node("Doc", {"t": "persistent text"})
    with GraphDB(path) as db:
        assert len(db.fulltext_search("ft", "persistent")) == 1
        db.create_node("Doc", {"t": "second persistent"})
    with GraphDB(path, read_only=True) as ro:
        assert len(ro.fulltext_search("ft", "persistent")) == 2
        rows = ro.query("CALL db.index.fulltext.queryNodes('ft', 'second') YIELD node RETURN node.t AS t")
        assert rows == [{"t": "second persistent"}]
