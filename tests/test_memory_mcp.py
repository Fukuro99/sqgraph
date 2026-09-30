"""sqgraph.memory.mcp: tool handler functions (no subprocess) and server tool registration."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from sqgraph.memory import FakeEmbedder, FakeLLM, Memory  # noqa: E402
from sqgraph.memory.mcp import MemoryTools, build_server  # noqa: E402

G = "guild"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        for text, ch, who, at in [
            ("[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", "pub", "alice", "2026-07-01 10:00"),
            ("[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", "pub", "bob", "2026-08-15 10:00"),
            ("[[Rails|Service]] [[Vault|Project]] {{Rails|FEEDS|Vault}}", "secret", "carol", "2026-08-16 10:00"),
        ]:
            mem.add_episode(text, group_id=G, channel=ch, author=who, occurred_at=at, source="test")
    return path


def tools_for(path, **kw):
    mem = Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G, read_only=kw.pop("db_read_only", False))
    return MemoryTools(mem, **kw), mem


def test_search_get_entity_and_episodes(db_path):
    t, mem = tools_for(db_path, group_id=G)
    res = t.search_memory("Rails OWNER", include_invalid=True)
    assert res["count"] >= 2 and res["group_id"] == G
    statuses = {f["object"]["name"]: f["status"] for f in res["facts"] if f["relation"] == "OWNER"}
    assert statuses == {"alice": "ended", "bob": "valid"}
    past = t.search_memory("Rails OWNER", as_of="2026-07-20")
    assert [f["object"]["name"] for f in past["facts"]] == ["alice"]
    ent = t.get_entity("Rails")
    assert ent["found"] and ent["entity"]["name"] == "Rails" and len(ent["timeline"]) == 3
    assert not t.get_entity("does-not-exist")["found"]
    found = t.search_entities("rails")
    assert found["entities"][0]["name"] == "Rails"
    eps = t.get_episodes(entity="Rails", limit=5)
    assert eps["count"] == 3
    mem.close()


def test_add_memory_and_read_only(db_path):
    t, mem = tools_for(db_path, group_id=G)
    out = t.add_memory("[[Rails|Service]] [[dave|Person]] {{dave|REVIEWS|Rails}}", channel="pub", author="dave", source_ref="x1")
    assert out["facts_created"] == 1 and not out["already_stored"]
    assert t.add_memory("[[Rails|Service]] {{dave|REVIEWS|Rails}}", channel="pub", author="dave", source_ref="x1")["already_stored"]
    assert any(f["relation"] == "REVIEWS" for f in t.search_memory("Rails REVIEWS")["facts"])
    with pytest.raises(ValueError):
        t.add_memory("  ")
    mem.close()

    ro, mem2 = tools_for(db_path, group_id=G, read_only=True, db_read_only=True)
    with pytest.raises(PermissionError):
        ro.add_memory("[[a|Concept]] hello world")
    assert ro.search_memory("Rails OWNER")["count"] >= 1  # reading still works
    mem2.close()


def test_group_lock(db_path):
    t, mem = tools_for(db_path, group_id=G)
    assert t.search_memory("Rails", group_id=G)["count"] > 0
    with pytest.raises(PermissionError):
        t.search_memory("Rails", group_id="another")
    mem.close()
    # unlocked server: caller chooses the group
    t2, mem2 = tools_for(db_path)
    assert t2.search_memory("Rails", group_id="another")["count"] == 0
    assert t2.search_memory("Rails")["count"] > 0  # falls back to Memory.default_group_id
    mem2.close()


def test_allowed_channels_enforced_server_side(db_path):
    t, mem = tools_for(db_path, group_id=G, allowed_channels=["pub"])
    res = t.search_memory("Rails Vault FEEDS", include_invalid=True)
    assert "Vault" not in repr(res) and "carol" not in repr(res)
    # the caller cannot widen the restriction ...
    wide = t.search_memory("Rails Vault FEEDS", channels=["pub", "secret"], include_invalid=True)
    assert "Vault" not in repr(wide)
    # ... and asking only for a forbidden channel yields nothing
    assert t.search_memory("Rails", channels=["secret"])["count"] == 0
    assert not t.get_entity("Vault")["found"]
    assert t.search_entities("Vault")["count"] == 0
    assert all(e["channel"] == "pub" for e in t.get_episodes()["episodes"])
    assert "Vault" not in repr(t.get_entity("Rails"))
    # writing outside the allowed channels is refused, inside is fine
    with pytest.raises(PermissionError):
        t.add_memory("[[a|Concept]] {{a|X|a}} hello", channel="secret")
    assert t.add_memory("[[Rails|Service]] [[eve|Person]] {{eve|USES|Rails}}", author="eve")["facts_created"] == 1  # default channel = first allowed
    mem.close()


def test_read_cypher(db_path):
    t, mem = tools_for(db_path, group_id=G)
    res = t.read_cypher(
        "MATCH (s:Entity)-[f:FACT]->(o:Entity) WHERE f.relation = $r RETURN f.fact AS fact, f.invalid_at AS inv ORDER BY f.valid_at",
        {"r": "OWNER"},
    )
    assert res["row_count"] == 2 and res["rows"][0]["inv"] is not None and res["rows"][1]["inv"] is None
    # row limit
    lim = t.read_cypher("MATCH (n) RETURN n.id AS id", limit=3)
    assert lim["row_count"] == 3 and lim["truncated"] and lim["total_rows"] > 3
    # nodes and fact edges are serialised and embeddings never appear
    node = t.read_cypher("MATCH (e:Entity {name:'Rails'}) RETURN e")["rows"][0]["e"]
    assert node["kind"] == "node" and "embedding" not in node["properties"]
    fact = t.read_cypher("MATCH ()-[f:FACT {relation:'OWNER'}]->() RETURN f LIMIT 1")["rows"][0]["f"]
    assert fact["kind"] == "relationship" and "embedding" not in fact["properties"]
    # the query patterns advertised in the schema text work
    cur = t.read_cypher("MATCH (s:Entity)-[f:FACT]->(o:Entity) WHERE f.invalid_at IS NULL AND f.relation = 'OWNER' RETURN s.name AS s, o.name AS o")
    assert cur["rows"] == [{"s": "Rails", "o": "bob"}]
    # writes are rejected (read-only connection)
    with pytest.raises(Exception):
        t.read_cypher("CREATE (:Evil {x: 1})")
    assert t.read_cypher("MATCH (n:Evil) RETURN count(n) AS c")["rows"] == [{"c": 0}]
    # invalid query -> error, not a crash
    with pytest.raises(Exception):
        t.read_cypher("MATCH (n RETURN")
    # char limit
    small = MemoryTools(mem, group_id=G, max_chars=200)
    r = small.read_cypher("MATCH (e:Episode) RETURN e.content AS c")
    assert r["truncated"] and r["row_count"] < r["total_rows"]
    mem.close()


def test_read_cypher_timeout(db_path):
    t, mem = tools_for(db_path, group_id=G, allow_cypher=True)
    t.query_timeout = 0.05
    with pytest.raises(TimeoutError):
        # cross product of nodes: far too slow for 50 ms
        t.read_cypher("MATCH (a), (b), (c), (d), (e), (f), (g), (h), (i), (j) RETURN count(*) AS n")
    mem.close()


def test_read_cypher_disabled_with_channel_acl(db_path):
    t, mem = tools_for(db_path, group_id=G, allowed_channels=["pub"])
    assert not t.cypher_enabled
    with pytest.raises(PermissionError, match="bypasses"):
        t.read_cypher("MATCH (n) RETURN n")
    assert t.get_schema()["read_cypher_enabled"] is False
    forced = MemoryTools(mem, group_id=G, allowed_channels=["pub"], allow_cypher=True)
    assert forced.read_cypher("MATCH ()-[f:FACT]->() RETURN count(f) AS c")["rows"][0]["c"] == 3
    mem.close()


def test_get_schema(db_path):
    t, mem = tools_for(db_path, group_id=G)
    s = t.get_schema()
    assert s["counts"]["nodes"]["Episode"] == 3 and "Fact" not in s["counts"]["nodes"]
    assert s["counts"]["facts_valid"] == 2 and s["counts"]["facts_invalidated"] == 1
    assert s["counts"]["relationships"]["FACT"] == 3 and "SUPERSEDES" not in s["counts"]["relationships"]
    assert {"fact_emb", "entity_emb", "fact_ft", "fact_id"} <= {i["name"] for i in s["indexes"]}
    assert "[:FACT" in s["description"] and "self-loop" in s["description"] and "SUBJECT_OF" not in s["description"]
    assert "Cypher" in s["cypher_notes"] or "openCypher" in s["cypher_notes"]
    assert s["read_cypher_enabled"] is True and s["channel_restriction"] is None
    mem.close()


def _tool_names(server) -> set[str]:
    async def go():
        return {t.name for t in await server.list_tools()}

    return asyncio.run(go())


def test_server_registers_tools_according_to_policy(db_path):
    t, mem = tools_for(db_path, group_id=G)
    full = _tool_names(build_server(t))
    assert full == {"add_memory", "search_memory", "search_entities", "get_entity", "get_episodes", "get_schema", "read_cypher"}
    mem.close()

    t2, mem2 = tools_for(db_path, group_id=G, allowed_channels=["pub"])
    assert "read_cypher" not in _tool_names(build_server(t2))
    mem2.close()

    t3, mem3 = tools_for(db_path, group_id=G, read_only=True, db_read_only=True)
    names = _tool_names(build_server(t3))
    assert "add_memory" not in names and "search_memory" in names
    mem3.close()


def test_server_tool_call_end_to_end(db_path):
    t, mem = tools_for(db_path, group_id=G)
    server = build_server(t)

    async def go():
        return await server.call_tool("search_memory", {"query": "Rails OWNER", "as_of": "2026-07-20"})

    result = asyncio.run(go())
    content = result.content if hasattr(result, "content") else result[0]
    payload = json.loads(content[0].text)
    assert [f["object"]["name"] for f in payload["facts"]] == ["alice"]
    mem.close()
