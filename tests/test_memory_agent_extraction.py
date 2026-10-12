"""Agent extraction (two phases, no server-side LLM): Memory.prepare_episode / commit_episode, the MCP tools and the CLI."""

from __future__ import annotations

import asyncio
import io
import json
import sys

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, Memory, cli
from sqgraph.memory.core import _decode_token, _encode_token
from sqgraph.memory.filelock import FileLock
from sqgraph.memory.prompts import AGENT_EXTRACTION_INSTRUCTIONS, AGENT_EXTRACTION_SCHEMA, ENTITY_TYPES

G = "guild"
AT = "2026-07-01 10:00"


def make(**kw):
    return Memory(":memory:", None, FakeEmbedder(), default_group_id=G, **kw)


def facts(mem):
    return [r["f"] for r in mem.db.query("MATCH ()-[f:FACT]->() RETURN f ORDER BY f.created_at, f.valid_at")]


def entity(mem, name):
    rows = mem.db.query("MATCH (e:Entity {name:$n}) RETURN e", {"n": name})
    assert len(rows) == 1, (name, len(rows))
    return rows[0]["e"]


def two_step(mem, text, extraction, *, channel="personal", author="user", at=AT, source_ref=None, source="test"):
    prep = mem.prepare_episode(text, group_id=G, source=source, channel=channel, author=author, occurred_at=at, source_ref=source_ref)
    if prep["already_stored"]:
        return prep, None
    return prep, mem.commit_episode(prep["token"], extraction)


PNPM = {
    "entities": [{"ref": "e1", "name": "pnpm", "type": "Service", "existing_id": None, "summary": "JS package manager the user prefers", "aliases": []}],
    "facts": [{"subject": "author", "object": "e1", "relation": "PREFERS", "fact": "ユーザーは npm ではなく pnpm を使ってほしい",
               "valid_at": None, "invalid_at": None, "modality": "definite", "confidence": 0.95, "duplicate_of": None, "contradicts": []}],
}


def owner(s_name, o_name, text, **kw):
    """Extraction: ``s_name`` OWNS ``o_name`` (both new entities unless an existing_id is given)."""
    return {
        "entities": [
            {"ref": "s", "name": s_name, "type": "Person", **kw.get("sub", {})},
            {"ref": "o", "name": o_name, "type": "Service", **kw.get("obj", {})},
        ],
        "facts": [{"subject": "s", "object": "o", "relation": "OWNS", "fact": text, **kw.get("fact", {})}],
    }


# --------------------------------------------------------------------------- phase 1


def test_prepare_returns_token_instructions_schema_and_context_and_writes_nothing():
    mem = make()
    prep = mem.prepare_episode("来週から pnpm を使ってほしい", group_id=G, source="claude-code", channel="personal", author="user", occurred_at=AT)
    assert prep["already_stored"] is False
    assert prep["instructions"] == AGENT_EXTRACTION_INSTRUCTIONS and prep["schema"] == AGENT_EXTRACTION_SCHEMA
    assert "commit_memory" in prep["next"] and "token" in prep["next"]
    for t in ENTITY_TYPES:
        assert t in AGENT_EXTRACTION_INSTRUCTIONS
    ctx = prep["context"]
    assert ctx["message"] == {"content": "来週から pnpm を使ってほしい", "author": "user", "occurred_at": "2026-07-01 10:00 (Wed, +09:00)", "channel": "personal"}
    assert ctx["author_ref"] == {"ref": "author", "name": "user", "existing_id": None}
    assert ctx["recent"] == [] and ctx["candidate_entities"] == [] and ctx["related_facts"] == []
    assert mem.stats(G)["nodes"] == {"Episode": 0, "Entity": 0}  # nothing is written by phase 1
    tok = Memory.read_token(prep["token"])
    assert tok == {"group_id": G, "source": "claude-code", "channel": "personal", "author": "user",
                   "occurred_at": "2026-07-01T01:00:00Z", "source_ref": None, "content": "来週から pnpm を使ってほしい"}
    assert prep["token"].startswith("sqm1.") and "{" not in prep["token"]
    json.dumps(prep)  # JSON serialisable


def test_prepare_does_not_need_the_write_lock(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, None, FakeEmbedder(), default_group_id=G, lock_timeout=0.2) as mem:
        with FileLock(str(path) + ".lock", timeout=1):  # another process / agent is writing
            assert mem.prepare_episode("pnpm を使う予定です", author="user")["token"]
            with pytest.raises(TimeoutError):
                mem.commit_episode(mem.prepare_episode("pnpm を使う予定です", author="user")["token"], {})


def test_prepare_validates_input():
    mem = make()
    with pytest.raises(ValueError, match="empty"):
        mem.prepare_episode("   ", group_id=G)
    with pytest.raises(ValueError, match="occurred_at"):
        mem.prepare_episode("pnpm", group_id=G, occurred_at="garbage")
    with pytest.raises(ValueError, match="group_id"):
        Memory(":memory:", None, None).prepare_episode("pnpm")


def test_prepare_context_candidates_related_facts_recent_and_author():
    mem = make()
    two_step(mem, "Rails の担当は太郎です", {
        "entities": [{"ref": "r", "name": "Rails", "type": "Service", "summary": "Web framework", "aliases": ["レイルズ"]},
                     {"ref": "t", "name": "太郎", "type": "Person"}],
        "facts": [{"subject": "t", "object": "r", "relation": "OWNS", "fact": "太郎 は Rails を担当している"},
                  {"subject": "t", "relation": "HAS_STATUS", "fact": "太郎 は元気", "invalid_at": "2026-07-02"}],
    }, author="alice", at="2026-06-30 09:00")
    # another channel: must stay invisible
    two_step(mem, "Secret の担当は次郎です", {
        "entities": [{"ref": "s", "name": "Secret", "type": "Project"}, {"ref": "j", "name": "次郎", "type": "Person"}],
        "facts": [{"subject": "j", "object": "s", "relation": "OWNS", "fact": "次郎 は Secret を担当している"}],
    }, channel="other", author="alice")

    prep = mem.prepare_episode("レイルズ の担当が 太郎 から 花子 に変わります。Secret は別件", group_id=G, channel="personal", author="alice", occurred_at=AT)  # names 太郎: his facts are related
    ctx = prep["context"]
    names = [c["name"] for c in ctx["candidate_entities"]]
    assert "Rails" in names and "alice" in names  # alias match + the author entity
    assert "Secret" not in names and "次郎" not in names  # other channel
    rails = next(c for c in ctx["candidate_entities"] if c["name"] == "Rails")
    assert rails["type"] == "Service" and rails["aliases"] == ["レイルズ"] and rails["summary"] == "Web framework" and rails["id"].startswith("ent_")
    assert ctx["author_ref"]["name"] == "alice" and ctx["author_ref"]["existing_id"] == entity(mem, "alice")["id"]
    texts = {f["fact"]: f for f in ctx["related_facts"]}
    assert "太郎 は Rails を担当している" in texts and "次郎 は Secret を担当している" not in texts
    f = texts["太郎 は Rails を担当している"]
    assert f["id"].startswith("fact_") and f["relation"] == "OWNS" and f["status"] == "valid"
    assert f["subject"]["name"] == "太郎" and f["object"]["name"] == "Rails"
    assert f["valid_at"] == "2026-06-30T09:00:00+09:00" and f["invalid_at"] is None
    # an attribute fact has no object, and a fact whose end lies after the message is still related
    assert texts["太郎 は元気"]["object"] is None and texts["太郎 は元気"]["invalid_at"] == "2026-07-02T00:00:00+09:00"
    assert ctx["recent"] == [{"at": "2026-06-30 09:00", "author": "alice", "text": "Rails の担当は太郎です"}]  # the other channel is excluded


def test_prepare_related_facts_exclude_ended_and_respect_limits():
    mem = make()
    two_step(mem, "A", owner("alice", "Rails", "alice は Rails を担当"), author="alice", at="2026-06-01 10:00")
    prep0 = mem.prepare_episode("Rails の担当", group_id=G, channel="personal", author="alice", occurred_at=AT)
    old = next(f["id"] for f in prep0["context"]["related_facts"])
    two_step(mem, "B", owner("bob", "Rails", "bob は Rails を担当", sub={}, fact={"contradicts": [old]}), author="bob", at="2026-06-15 10:00")
    prep = mem.prepare_episode("Rails の担当", group_id=G, channel="personal", author="carol", occurred_at=AT)
    assert [f["fact"] for f in prep["context"]["related_facts"]] == ["bob は Rails を担当"]  # the superseded one is not offered
    prep = mem.prepare_episode("Rails の担当", group_id=G, channel="personal", occurred_at="2026-06-10 10:00")
    assert {f["fact"]: f["status"] for f in prep["context"]["related_facts"]} == {"alice は Rails を担当": "valid", "bob は Rails を担当": "scheduled"}  # as of the message time
    capped = mem.prepare_episode("Rails の担当", group_id=G, channel="personal", occurred_at=AT, max_entities=1, max_facts=0)
    assert capped["context"]["related_facts"] == [] and len(capped["context"]["candidate_entities"]) <= 1


def test_prepare_finds_candidates_by_vector_similarity():
    mem = make()
    two_step(mem, "x", {"entities": [{"ref": "z", "name": "Zeta", "type": "Concept", "summary": "quantum flux capacitor calibration procedure"}]}, author=None)
    prep = mem.prepare_episode("quantum flux capacitor calibration procedure", group_id=G, channel="personal", occurred_at=AT)
    assert [c["name"] for c in prep["context"]["candidate_entities"]] == ["Zeta"]  # no name match: only the embedding
    assert prep["context"]["author_ref"] is None


# --------------------------------------------------------------------------- phase 2: writes


def test_commit_creates_entities_facts_episode_and_author():
    mem = make()
    prep, res = two_step(mem, "pnpm を使ってほしい", PNPM, source_ref="n1")
    assert not res.skipped and res.llm_calls == 0 and res.warnings == [] and res.usd is None
    assert len(res.new_entities) == 2 and len(res.entities) == 2 and len(res.facts_created) == 1
    user, pnpm = entity(mem, "user"), entity(mem, "pnpm")
    assert user["type"] == "Person" and pnpm["type"] == "Service"
    assert pnpm["summary"] == "JS package manager the user prefers" and pnpm["summaries"] == ["personal\tJS package manager the user prefers"]
    (f,) = facts(mem)
    assert f["subject_id"] == user["id"] and f["object_id"] == pnpm["id"] and f["relation"] == "PREFERS"
    assert f["valid_at"] == "2026-07-01T01:00:00Z" and f.get("invalid_at") is None  # null valid_at = the message time
    assert f["confidence"] == 0.95 and f["modality"] == "definite" and f["channels"] == ["personal"]
    ep = mem.db.query("MATCH (e:Episode) RETURN e")[0]["e"]
    assert ep["content"] == "pnpm を使ってほしい" and ep["source"] == "test" and ep["author"] == "user" and ep["source_ref"] == "n1"
    assert ep["occurred_at"] == "2026-07-01T01:00:00Z" and f["episode_ids"] == [ep["id"]]
    roles = mem.db.query("MATCH (:Episode)-[m:MENTIONS]->(e:Entity) RETURN e.name AS n, m.role AS r")
    assert {(r["n"], r["r"]) for r in roles} == {("user", "author"), ("pnpm", None)}
    # embeddings were computed for entities and the fact (search finds it by vector)
    assert mem.search("pnpm", group_id=G, channels=["personal"])[0].fact == "ユーザーは npm ではなく pnpm を使ってほしい"
    assert mem.db.vector_search("fact_emb", mem.embedder.embed(["pnpm"])[0], k=1)


def test_commit_accepts_a_json_string_and_reports_the_result_dict():
    mem = make()
    prep = mem.prepare_episode("pnpm を使ってほしい", group_id=G, author="user", occurred_at=AT)
    res = mem.commit_episode(prep["token"], json.dumps(PNPM, ensure_ascii=False))
    d = res.to_dict()
    assert d["warnings"] == [] and len(d["facts_created"]) == 1 and d["llm_calls"] == 0


def test_existing_id_reuses_the_entity_and_adds_the_spelling_as_alias():
    mem = make()
    two_step(mem, "a", owner("alice", "Rails", "alice は Rails を担当"), author="alice", at="2026-06-01 10:00")
    rails = entity(mem, "Rails")
    prep = mem.prepare_episode("RoR を bob が担当", group_id=G, channel="personal", author="bob", occurred_at=AT)
    ext = {"entities": [{"ref": "r", "name": "RoR", "type": "Service", "existing_id": rails["id"], "summary": "Web framework (RoR)"},
                        {"ref": "b", "name": "bob", "type": "Person"}],
           "facts": [{"subject": "b", "object": "r", "relation": "OWNS", "fact": "bob は RoR を担当している"}]}
    res = mem.commit_episode(prep["token"], ext)
    assert res.warnings == [] and rails["id"] in res.entities and len(res.new_entities) == 1  # only bob (the author) is new; "b" merged into it
    assert len(mem.db.query("MATCH (e:Entity {name:'Rails'}) RETURN e")) == 1
    p = mem.db.query("MATCH (e:Entity {id:$i}) RETURN e", {"i": rails["id"]})[0]["e"]
    assert "RoR" in p["aliases"] and p["name"] == "Rails"
    assert p["summaries"] == ["personal\tWeb framework (RoR)"]
    assert mem.search("RoR", group_id=G)  # the new alias is searchable


def test_exact_match_safety_net_merges_a_new_entity_into_the_existing_one():
    mem = make()
    two_step(mem, "a", owner("alice", "Rails", "alice は Rails を担当", obj={"aliases": ["Ruby on Rails"]}), author="alice", at="2026-06-01 10:00")
    rails_id = entity(mem, "Rails")["id"]
    # the agent missed the candidate: "ruby on rails" (case / spaces differ) and an alias key match
    prep = mem.prepare_episode("bob は ruby on rails をレビュー", group_id=G, channel="personal", author="bob", occurred_at=AT)
    ext = {"entities": [{"ref": "r", "name": "ruby on rails", "type": "Service", "existing_id": None}],
           "facts": [{"subject": "author", "object": "r", "relation": "REVIEWS", "fact": "bob は Rails をレビューしている"}]}
    res = mem.commit_episode(prep["token"], ext)
    assert len(mem.db.query("MATCH (e:Entity) WHERE e.name CONTAINS 'ails' RETURN e")) == 1
    assert rails_id in res.entities and any("matched the existing entity 'Rails'" in w for w in res.warnings)
    assert [f for f in facts(mem) if f["relation"] == "REVIEWS"][0]["object_id"] == rails_id

    # an incompatible type (Person vs Service) is NOT merged
    prep = mem.prepare_episode("Rails という人", group_id=G, channel="personal", author="bob", occurred_at=AT)
    res = mem.commit_episode(prep["token"], {"entities": [{"ref": "p", "name": "Rails", "type": "Person"}]})
    assert len(res.new_entities) == 1 and len(mem.db.query("MATCH (e:Entity {name:'Rails'}) RETURN e")) == 2


def test_same_entity_declared_twice_or_the_author_listed_merges():
    mem = make()
    ext = {"entities": [{"ref": "a", "name": "Redis", "type": "Service"}, {"ref": "b", "name": "redis", "type": "Service", "summary": "KV store"},
                        {"ref": "me", "name": "alice", "type": "Person"}],
           "facts": [{"subject": "a", "relation": "HAS_STATUS", "fact": "Redis は稼働中"},
                     {"subject": "b", "object": "me", "relation": "OWNED_BY", "fact": "Redis は alice が管理"}]}
    _, res = two_step(mem, "Redis と alice", ext, author="alice")
    assert len(res.entities) == 2 and len(res.new_entities) == 2  # Redis once, alice (author) once
    assert entity(mem, "Redis")["summary"] == "KV store" and len(facts(mem)) == 2
    redis, alice = entity(mem, "Redis")["id"], entity(mem, "alice")["id"]
    # OWNED_BY is stored in its active direction: alice OWNS Redis
    assert {(f["subject_id"], f["relation"], f.get("object_id")) for f in facts(mem)} == {
        (redis, "HAS_STATUS", None), (alice, "OWNS", redis)
    }


def test_duplicate_of_appends_the_episode_to_the_existing_fact():
    mem = make()
    two_step(mem, "a", owner("alice", "Rails", "alice は Rails を担当", fact={"confidence": 0.6}), author="alice", at="2026-06-01 10:00")
    (old,) = facts(mem)
    rails, alice = entity(mem, "Rails"), entity(mem, "alice")
    prep = mem.prepare_episode("alice が Rails の担当です", group_id=G, channel="personal", author="alice", occurred_at=AT)
    assert [f["id"] for f in prep["context"]["related_facts"]] == [old["id"]]
    ext = {"entities": [{"ref": "r", "name": "Rails", "type": "Service", "existing_id": rails["id"]}],
           "facts": [{"subject": "author", "object": "r", "relation": "OWNS", "fact": "alice は Rails を担当している", "duplicate_of": old["id"], "confidence": 0.9,
                      "contradicts": [old["id"]]}]}
    res = mem.commit_episode(prep["token"], ext)
    assert res.facts_duplicate == [old["id"]] and res.facts_created == [] and res.facts_invalidated == []
    (f,) = facts(mem)
    assert f["id"] == old["id"] and len(f["episode_ids"]) == 2 and f["confidence"] == 0.9 and f.get("invalid_at") is None
    assert any("duplicate contradicts nothing" in w for w in res.warnings)
    assert alice["id"] in res.entities


def test_contradicts_invalidates_and_links_superseded_by_like_the_pipeline():
    mem = make()
    two_step(mem, "a", owner("alice", "Rails", "alice は Rails を担当"), author="alice", at="2026-07-01 10:00")
    (old,) = facts(mem)
    ent = lambda n: entity(mem, n)["id"]  # noqa: E731
    prep = mem.prepare_episode("Rails の担当は bob に変わります", group_id=G, channel="personal", author="alice", occurred_at="2026-08-15 10:00")
    ext = owner("bob", "Rails", "bob は Rails を担当している", obj={"existing_id": ent("Rails")}, fact={"contradicts": [old["id"]]})
    res = mem.commit_episode(prep["token"], ext)
    assert res.facts_invalidated == [old["id"]] and len(res.facts_created) == 1
    rows = {f["fact"]: f for f in facts(mem)}
    old_p, new_p = rows["alice は Rails を担当"], rows["bob は Rails を担当している"]
    assert old_p["invalid_at"] == new_p["valid_at"] == "2026-08-15T01:00:00Z"
    assert old_p["superseded_by"] == new_p["id"] and new_p["supersedes"] == [old_p["id"]] and old_p["expired_at"]
    hits = {h.fact: h.status for h in mem.search("Rails 担当", group_id=G, include_invalid=True)}
    assert hits == {"alice は Rails を担当": "ended", "bob は Rails を担当している": "valid"}

    # the same through the pipeline gives the same structure (FakeLLM: same subject+relation, other object = contradiction)
    pipe = Memory(":memory:", FakeLLM(), FakeEmbedder(), default_group_id=G)
    pipe.add_episode("[[Rails|Service]] [[alice|Person]] {{alice|OWNS|Rails|2026-07-01}}", author="alice", occurred_at="2026-07-01 10:00")
    pipe.add_episode("[[Rails|Service]] [[alice|Person]] [[bob|Person]] {{alice|OWNS|bob|2026-08-15}}", author="alice", occurred_at="2026-08-15 10:00")
    pf = facts(pipe)
    def shape(fs):
        return sorted((f.get("invalid_at") is not None, bool(f.get("superseded_by")), bool(f.get("supersedes"))) for f in fs)

    assert shape(pf) == shape(facts(mem))


def test_a_historical_fact_ends_where_the_newer_existing_one_begins():
    mem = make()
    two_step(mem, "a", owner("bob", "Rails", "bob は Rails を担当", fact={"valid_at": "2026-08-15"}), author="bob", at="2026-08-15 10:00")
    (newer,) = facts(mem)
    prep = mem.prepare_episode("以前は alice が担当", group_id=G, channel="personal", occurred_at="2026-09-01 10:00")
    ext = owner("alice", "Rails", "alice は Rails を担当していた", obj={"existing_id": entity(mem, "Rails")["id"]},
                fact={"valid_at": "2026-07-01", "contradicts": [newer["id"]]})
    res = mem.commit_episode(prep["token"], ext)
    assert res.facts_invalidated == []
    alice_f = next(f for f in facts(mem) if f["fact"].startswith("alice"))
    assert alice_f["invalid_at"] == newer["valid_at"] and alice_f["superseded_by"] == newer["id"]


def test_times_local_iso_relative_dates_modality_and_confidence():
    mem = make()
    ext = {"entities": [{"ref": "o", "name": "Osaka", "type": "Place"}],
           "facts": [
               {"subject": "author", "relation": "absent", "fact": "ユーザーは 9/14-9/16 お休み", "valid_at": "2026-09-14", "invalid_at": "2026-09-17"},
               {"subject": "author", "object": "o", "relation": "plans-to move", "fact": "ユーザーは来週から Osaka に引っ越す", "valid_at": "2026-07-08T09:30:00",
                "modality": "planned"},
               {"subject": "author", "relation": "HAS_STATUS", "fact": "tentative one", "valid_at": "2026-07-01T12:00:00Z", "invalid_at": "2026-06-01", "modality": "tentative", "confidence": 0.2},
           ]}
    _, res = two_step(mem, "休みと引っ越し", ext)
    by = {f["fact"]: f for f in facts(mem)}
    a = by["ユーザーは 9/14-9/16 お休み"]
    assert a["relation"] == "ABSENT" and a["valid_at"] == "2026-09-13T15:00:00Z" and a["invalid_at"] == "2026-09-16T15:00:00Z"
    b = by["ユーザーは来週から Osaka に引っ越す"]
    assert b["relation"] == "PLANS_TO_MOVE" and b["valid_at"] == "2026-07-08T00:30:00Z" and b["modality"] == "planned" and b["confidence"] == 0.7
    c = by["tentative one"]
    assert c["valid_at"] == "2026-07-01T12:00:00Z" and "invalid_at" not in c.properties  # invalid_at before valid_at is dropped
    assert c["modality"] == "tentative" and c["confidence"] == 0.2
    assert any("invalid_at is not after valid_at" in w for w in res.warnings)
    scheduled = {h.fact: h.status for h in mem.search("お休み 引っ越す", group_id=G, as_of="2026-07-02", include_future=True)}
    assert scheduled.get("ユーザーは来週から Osaka に引っ越す") == "scheduled"


def test_empty_extraction_stores_the_episode_and_the_author_mention_only():
    mem = make()
    for ext in ({}, {"entities": [], "facts": []}, {"entities": None, "facts": None}):
        prep = mem.prepare_episode(f"ありがとう {len(str(ext))}", group_id=G, author="user", occurred_at=AT)
        res = mem.commit_episode(prep["token"], ext)
        assert res.facts_created == [] and len(res.entities) == 1 and not res.skipped
    assert mem.stats(G)["nodes"]["Episode"] == 3 and mem.stats(G)["nodes"]["Entity"] == 1
    prep = mem.prepare_episode("匿名のメモです", group_id=G, occurred_at=AT)  # no author at all
    assert mem.commit_episode(prep["token"], {}).entities == []
    assert mem.stats(G)["nodes"]["Episode"] == 4


def test_commit_without_an_embedder_stores_without_vectors():
    mem = Memory(":memory:", None, None, default_group_id=G)
    prep = mem.prepare_episode("pnpm を使ってほしい", group_id=G, author="user", occurred_at=AT)
    assert prep["context"]["candidate_entities"] == []
    res = mem.commit_episode(prep["token"], PNPM)
    assert len(res.facts_created) == 1
    (f,) = facts(mem)
    assert "embedding" not in f.properties and entity(mem, "pnpm")["summary"]
    assert mem.search("pnpm", group_id=G)  # fulltext still finds it
    # and the next prepare works too (fulltext + name matching only)
    nxt = mem.prepare_episode("pnpm のバージョン", group_id=G, author="user", occurred_at=AT)
    assert "pnpm" in [c["name"] for c in nxt["context"]["candidate_entities"]]
    assert [x["fact"] for x in nxt["context"]["related_facts"]] == [f["fact"]]


# --------------------------------------------------------------------------- validation / staleness


def test_validation_lists_every_problem_and_writes_nothing():
    mem = make()
    # an entity of another group living in the same database
    mem.db.create_node("Entity", {"id": "ent_aaaaaaaaaaaa", "group_id": "elsewhere", "name": "Foreign", "type": "Concept", "channels": ["x"]})
    bad = {
        "entities": [
            {"ref": "e0", "name": "Rails", "type": "Service", "existing_id": "ent_ffffffffffff"},
            {"ref": "e1", "name": "Dup", "type": "Service"},
            {"ref": "e1", "name": "Dup2", "type": "Service"},
            {"ref": "e3", "name": "Foreign", "type": "Concept", "existing_id": "ent_aaaaaaaaaaaa"},
            {"name": "no ref"},
            "not an object",
        ],
        "facts": [
            {"subject": "nope", "relation": "X", "fact": "unknown subject"},
            {"subject": "author", "object": "ghost", "relation": "X", "fact": "unknown object"},
            {"subject": "author", "relation": "X", "fact": "bad time", "valid_at": "next tuesday-ish"},
            {"subject": "author", "relation": "X", "fact": "bad modality", "modality": "maybe"},
            {"subject": "author", "relation": "X", "fact": "bad confidence", "confidence": 1.5},
            {"subject": "author", "relation": "X", "fact": "bad dup", "duplicate_of": "not-a-fact"},
            {"subject": "author", "relation": "X", "fact": "bad contradicts", "contradicts": ["fact_xyz"]},
            {"subject": "author", "relation": "X"},
            {"subject": "author", "relation": "X", "fact": "bad contradicts type", "contradicts": [5]},
        ],
    }
    prep = mem.prepare_episode("たくさんの間違い", group_id=G, author="user", occurred_at=AT)
    with pytest.raises(ValueError) as ei:
        mem.commit_episode(prep["token"], bad)
    msg = str(ei.value)
    for needle in [
        "entities[0].existing_id 'ent_ffffffffffff' does not exist",
        "entities[2].ref 'e1' is used twice",
        "entities[3].existing_id 'ent_aaaaaaaaaaaa' belongs to another group",
        "entities[4].ref is required",
        "entities[5] must be an object",
        "facts[0].subject 'nope' is not a declared entity ref",
        "facts[1].object 'ghost' is not a declared entity ref",
        "facts[2].valid_at 'next tuesday-ish' cannot be parsed",
        "facts[3].modality 'maybe' must be one of",
        "facts[4].confidence 1.5 must be a number between 0 and 1",
        "facts[5].duplicate_of: 'not-a-fact' is not a fact id",
        "facts[6].contradicts: 'fact_xyz' is not a fact id",
        "facts[8].contradicts must be an array of fact ids",
        "facts[7].fact is required",
    ]:
        assert needle in msg, needle
    assert msg.startswith("invalid extraction (") and mem.stats(G)["nodes"] == {"Episode": 0, "Entity": 0}

    for ext, pattern in [("{not json", "not valid JSON"), ([1], "must be a JSON object"), ({"facts": "x"}, "facts must be an array"),
                         ({"facts": [{"subject": "author", "fact": "x"}]}, None)]:
        if pattern:
            with pytest.raises(ValueError, match=pattern):
                mem.commit_episode(prep["token"], ext)
    # no author: "author" is not a ref
    anon = mem.prepare_episode("匿名", group_id=G, occurred_at=AT)
    with pytest.raises(ValueError, match="subject 'author' is not a declared entity ref"):
        mem.commit_episode(anon["token"], {"facts": [{"subject": "author", "relation": "X", "fact": "f"}]})
    # relation defaults, an "author" entry among the entities is ignored with a note
    res = mem.commit_episode(prep["token"], {"entities": [{"ref": "author", "name": "user", "type": "Person"}], "facts": [{"subject": "author", "fact": "ユーザーはテスト中"}]})
    assert facts(mem)[0]["relation"] == "RELATES_TO" and any("added automatically" in w for w in res.warnings)


def test_stale_ids_become_warnings_not_errors():
    mem = make()
    two_step(mem, "a", owner("alice", "Rails", "alice は Rails を担当"), author="alice", at="2026-06-01 10:00")
    two_step(mem, "b", {"entities": [{"ref": "x", "name": "Rails", "type": "Service"}], "facts": [{"subject": "x", "relation": "HAS_STATUS", "fact": "Rails は安定"}]},
             channel="work", author="alice", at="2026-06-02 10:00")
    owns = next(f for f in facts(mem) if f["relation"] == "OWNS")
    work_fact = next(f for f in facts(mem) if f["relation"] == "HAS_STATUS")
    # an ended fact (as would happen if another agent wrote in between)
    mem.db.update_edge(owns.id, {"invalid_at": "2026-06-20T00:00:00Z"})
    gone = "fact_0123456789ab"
    prep = mem.prepare_episode("Rails 更新", group_id=G, channel="personal", author="alice", occurred_at=AT)
    ext = {"entities": [{"ref": "r", "name": "Rails", "type": "Service"}],
           "facts": [
               {"subject": "author", "object": "r", "relation": "USES", "fact": "f-gone-dup", "duplicate_of": gone},
               {"subject": "author", "object": "r", "relation": "USES2", "fact": "f-gone-contra", "contradicts": [gone]},
               {"subject": "author", "object": "r", "relation": "USES3", "fact": "f-ended", "contradicts": [owns["id"]], "duplicate_of": owns["id"]},
               {"subject": "author", "object": "r", "relation": "USES4", "fact": "f-other-channel", "duplicate_of": work_fact["id"]},
           ]}
    res = mem.commit_episode(prep["token"], ext)
    w = "\n".join(res.warnings)
    assert f"fact {gone} no longer exists" in w
    assert "has already ended; stored as a new fact" in w and "belongs to another channel; stored as a separate fact" in w
    assert len(res.facts_created) == 4 and res.facts_duplicate == [] and res.facts_invalidated == []
    ended = mem.db.query("MATCH ()-[f:FACT {id:$i}]->() RETURN f", {"i": owns["id"]})[0]["f"]
    assert ended["invalid_at"] == "2026-06-20T00:00:00Z" and "superseded_by" not in ended.properties  # left untouched
    wf = mem.db.query("MATCH ()-[f:FACT {id:$i}]->() RETURN f", {"i": work_fact["id"]})[0]["f"]
    assert wf["channels"] == ["work"] and len(wf["episode_ids"]) == 1  # the other channel's fact was not extended

    # a fact that exists but turned ended after prepare is reported on contradicts as well
    prep = mem.prepare_episode("もう一度", group_id=G, channel="personal", author="alice", occurred_at=AT)
    res = mem.commit_episode(prep["token"], {"entities": [{"ref": "r", "name": "Rails", "type": "Service"}], "facts": [
        {"subject": "author", "object": "r", "relation": "USES", "fact": "again", "contradicts": [owns["id"]]}]})
    assert any("no longer valid" in x for x in res.warnings) and res.facts_invalidated == []


def test_source_ref_dedupe_at_both_phases():
    mem = make()
    prep, res = two_step(mem, "pnpm を使ってほしい", PNPM, source_ref="n1")
    assert res.episode_id
    again = mem.prepare_episode("pnpm を使ってほしい", group_id=G, source="test", channel="personal", author="user", source_ref="n1")
    assert again == {"already_stored": True, "episode_id": res.episode_id}
    # a token prepared before another agent committed the same (source, source_ref): commit is a no-op
    other = make()
    p1 = other.prepare_episode("メモ", group_id=G, source="s", channel="c", author="u", source_ref="r1")
    p2 = other.prepare_episode("メモ", group_id=G, source="s", channel="c", author="u", source_ref="r1")
    first = other.commit_episode(p1["token"], {})
    second = other.commit_episode(p2["token"], PNPM)
    assert second.skipped and second.episode_id == first.episode_id and second.facts_created == []
    assert other.stats(G)["nodes"]["Episode"] == 1
    # another source with the same ref is a different episode
    p3 = other.prepare_episode("メモ", group_id=G, source="s2", channel="c", author="u", source_ref="r1")
    assert p3["already_stored"] is False


def test_token_is_self_contained_and_validated():
    mem = make()
    prep = mem.prepare_episode("pnpm を使ってほしい", group_id=G, source="cli", channel="personal", author="user", occurred_at=AT, source_ref="r")
    token = prep["token"]
    # works across Memory instances / processes: nothing is kept server-side
    fresh = make()
    assert fresh.commit_episode(token, PNPM).facts_created
    assert Memory.read_token(token)["source_ref"] == "r"
    body = token.split(".")[1]
    for bad in [
        "", "   ", None, 5, "garbage", "sqm1.only-two", f"sqm2.{body}.{token.split('.')[2]}",
        token[:-3],                                     # truncated checksum
        token.replace(body, body[:-4]),                 # truncated body
        token.replace(body[10], "A" if body[10] != "A" else "B", 1),  # altered character
        f"sqm1.{body}.00000000",
        "sqm1.!!!!.deadbeef",
    ]:
        with pytest.raises(ValueError, match="invalid token"):
            fresh.commit_episode(bad, {})
    # a well-formed token of another version / with missing fields is rejected too
    import base64, hashlib, zlib

    def forge(payload):
        raw = zlib.compress(json.dumps(payload).encode("utf-8"))
        return f"sqm1.{base64.urlsafe_b64encode(raw).decode().rstrip('=')}.{hashlib.blake2b(raw, digest_size=4).hexdigest()}"

    with pytest.raises(ValueError, match="unsupported version"):
        _decode_token(forge({"v": 2}))
    with pytest.raises(ValueError, match="missing field"):
        _decode_token(forge({"v": 1, "g": "a"}))
    with pytest.raises(ValueError, match="bad time"):
        _decode_token(forge({"v": 1, "g": "a", "s": "s", "c": "c", "t": "x", "x": "text"}))
    assert _decode_token(_encode_token({"v": 1, "g": "a", "s": "s", "c": "c", "a": None, "t": "2026-07-01T00:00:00Z", "r": None, "x": "日本語 ✓"}))["content"] == "日本語 ✓"


def test_commit_holds_the_write_lock_for_the_whole_operation(tmp_path):
    path = tmp_path / "mem.db"
    seen: list[bool] = []

    class SpyEmbedder(FakeEmbedder):
        def embed(self, texts):
            holding = mem._file_lock.locked and mem._lock_depth >= 1
            seen.append(holding)
            if holding:
                with pytest.raises(TimeoutError):  # another process cannot write while we embed
                    FileLock(str(path) + ".lock", timeout=0.1).acquire()
            return super().embed(texts)

    with Memory(path, None, SpyEmbedder(), default_group_id=G, lock_timeout=0.3) as mem:
        prep = mem.prepare_episode("pnpm を使ってほしい", group_id=G, author="user", occurred_at=AT)
        n_prepare_embeds = len(seen)
        assert n_prepare_embeds >= 1 and not any(seen)  # phase 1 embeds the message without the lock
        res = mem.commit_episode(prep["token"], PNPM)
        assert res.facts_created and seen[n_prepare_embeds:] == [True]  # one batched embedding call inside the guard
        assert mem._lock_depth == 0 and not mem._file_lock.locked
        # and a held lock makes commit wait and time out, writing nothing
        with FileLock(str(path) + ".lock", timeout=1):
            with pytest.raises(TimeoutError, match="mem.db.lock"):
                mem.commit_episode(mem.prepare_episode("別のメモです", group_id=G, author="user")["token"], PNPM)
        assert mem.stats(G)["nodes"]["Episode"] == 1


def test_read_only_memory_cannot_commit_but_can_prepare(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, None, FakeEmbedder(), default_group_id=G) as mem:
        prep = mem.prepare_episode("pnpm", group_id=G, author="user", occurred_at=AT)
    with Memory(path, None, None, default_group_id=G, read_only=True) as ro:
        assert ro.prepare_episode("pnpm", group_id=G)["token"]
        with pytest.raises(Exception, match="read-only"):
            ro.commit_episode(prep["token"], {})


def test_add_episode_still_runs_the_one_shot_pipeline_and_result_has_warnings():
    mem = Memory(":memory:", FakeLLM(), FakeEmbedder(), default_group_id=G)
    r = mem.add_episode("[[Rails|Service]] {{alice|USES|Rails}} [[alice|Person]]", author="alice")
    assert r.llm_calls >= 2 and r.warnings == [] and r.to_dict()["warnings"] == []


# --------------------------------------------------------------------------- MCP


mcp_mod = pytest.importorskip("mcp")
from sqgraph.memory.mcp import MemoryTools, build_server  # noqa: E402


def agent_tools(**kw):
    mem = Memory(":memory:", None, FakeEmbedder(), default_group_id=G)
    return MemoryTools(mem, group_id=G, extractor="agent", **kw), mem


def test_mcp_add_memory_in_agent_mode_returns_the_prepare_payload_and_commit_memory_writes():
    t, mem = agent_tools()
    assert t.extractor == "agent" and t.write_blocker is None
    prep = t.add_memory("pnpm を使ってほしい", author="user", source="claude-code", occurred_at=AT, source_ref="m1")
    assert prep["already_stored"] is False and prep["token"] and prep["instructions"] and prep["schema"] and prep["context"]["message"]["author"] == "user"
    assert Memory.read_token(prep["token"])["channel"] == "personal" and Memory.read_token(prep["token"])["source"] == "claude-code"
    assert mem.stats(G)["nodes"]["Episode"] == 0
    out = t.commit_memory(prep["token"], PNPM)
    assert out["facts_created"] == 1 and out["new_entities"] == 2 and out["llm_calls"] == 0 and "warnings" not in out and not out["already_stored"]
    assert t.add_memory("pnpm を使ってほしい", author="user", source="claude-code", source_ref="m1") == {"already_stored": True, "episode_id": out["episode_id"]}
    assert t.commit_memory(prep["token"], PNPM)["already_stored"] is True  # same source_ref
    assert t.search_memory("pnpm")["facts"][0]["fact"].startswith("ユーザーは npm")
    assert t.commit_memory(t.add_memory("メモです、よろしく", author="user")["token"], json.dumps({"entities": [], "facts": []}))["facts_created"] == 0
    with pytest.raises(ValueError, match="invalid extraction"):
        t.commit_memory(t.add_memory("もう一つメモ", author="user")["token"], {"facts": [{"subject": "zz", "relation": "X", "fact": "f"}]})
    with pytest.raises(ValueError, match="empty"):
        t.add_memory("  ")
    mem.close()


def test_mcp_commit_memory_policy_group_channel_read_only():
    t, mem = agent_tools(allowed_channels=["pub"])
    prep = t.add_memory("pnpm を使ってほしい", author="user")
    assert Memory.read_token(prep["token"])["channel"] == "pub"  # the first allowed channel
    with pytest.raises(PermissionError):
        t.add_memory("secret", channel="secret")
    forged = mem.prepare_episode("secret note", group_id=G, channel="secret", author="user")["token"]
    with pytest.raises(PermissionError, match="allowed channels"):
        t.commit_memory(forged, {})
    other_group = mem.prepare_episode("x note", group_id="elsewhere", channel="pub", author="user")["token"]
    with pytest.raises(PermissionError, match="group_id"):
        t.commit_memory(other_group, {})
    with pytest.raises(ValueError, match="invalid token"):
        t.commit_memory("nonsense", {})
    assert t.commit_memory(prep["token"], PNPM)["facts_created"] == 1
    ro = MemoryTools(mem, group_id=G, read_only=True, extractor="agent")
    with pytest.raises(PermissionError):
        ro.add_memory("x")
    with pytest.raises(PermissionError):
        ro.commit_memory(prep["token"], {})
    mem.close()


def test_mcp_extractor_selection_and_openai_mode_keeps_both_tools():
    mem = Memory(":memory:", FakeLLM(), FakeEmbedder(), default_group_id=G)
    assert MemoryTools(mem).extractor == "openai"  # a Memory with an LLM defaults to the one-shot pipeline
    t = MemoryTools(mem, extractor="openai")
    out = t.add_memory("[[Rails|Service]] [[dave|Person]] {{dave|REVIEWS|Rails}}", author="dave")
    assert out["facts_created"] == 1 and "token" not in out
    prep = mem.prepare_episode("pnpm を使ってほしい", group_id=G, channel="personal", author="user")
    assert t.commit_memory(prep["token"], PNPM)["facts_created"] == 1  # commit_memory always works
    blocked = MemoryTools(mem, extractor="openai", write_blocker="OPENAI_API_KEY is not set")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        blocked.add_memory("何か長いテキスト")
    assert blocked.commit_memory(mem.prepare_episode("別のメモです", group_id=G, channel="personal", author="user")["token"], {})["entities"] == 1
    assert MemoryTools(Memory(":memory:", None, None, default_group_id=G)).extractor == "agent"
    with pytest.raises(ValueError):
        MemoryTools(mem, extractor="bogus")
    mem.close()


def test_mcp_server_registers_and_serves_both_steps():
    t, mem = agent_tools()
    server = build_server(t)

    async def go():
        tools = {x.name: x for x in await server.list_tools()}
        r1 = await server.call_tool("add_memory", {"content": "pnpm を使ってほしい", "author": "user", "source": "codex"})
        prep = json.loads((r1.content if hasattr(r1, "content") else r1[0])[0].text)
        r2 = await server.call_tool("commit_memory", {"token": prep["token"], "extraction": PNPM})
        out = json.loads((r2.content if hasattr(r2, "content") else r2[0])[0].text)
        r3 = await server.call_tool("commit_memory", {"token": json.loads((await server.call_tool("add_memory", {"content": "もう一件のメモ", "author": "user"})).content[0].text)["token"],
                                                      "extraction": json.dumps(PNPM)})
        return tools, prep, out, r3

    tools, prep, out, _ = asyncio.run(go())
    assert {"add_memory", "commit_memory"} <= set(tools)
    assert "Step 1 of 2" in tools["add_memory"].description and "Step 2 of 2" in tools["commit_memory"].description
    assert "commit_memory" in tools["add_memory"].description
    assert prep["token"] and out["facts_created"] == 1
    assert mem.db.query("MATCH (e:Episode {source:'codex'}) RETURN e.id AS i")
    mem.close()
    one_shot = Memory(":memory:", FakeLLM(), FakeEmbedder(), default_group_id=G)
    names = {x.name: x for x in asyncio.run(build_server(MemoryTools(one_shot, extractor="openai")).list_tools())}
    assert "analysed by an LLM" in names["add_memory"].description and "commit_memory" in names
    one_shot.close()


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mem-home"
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(h))
    for k in ("OPENAI_API_KEY", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP", "SQGRAPH_MEMORY_EXTRACTOR", "SQGRAPH_EMBED_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return h


@pytest.fixture
def fakes(monkeypatch, home):
    seen: list[dict] = []

    def make_providers(cfg, *, need_llm, need_embedder):
        seen.append({"need_llm": need_llm, "need_embedder": need_embedder})
        assert not need_llm, "agent mode must never ask for an LLM"
        return None, (FakeEmbedder() if need_embedder else None), []

    monkeypatch.setattr(cli, "make_providers", make_providers)
    return seen


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    capsys.readouterr()
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, (json.loads(cap.out) if cap.out.strip() else None), (json.loads(cap.err) if cap.err.strip() else None)


def test_cli_add_commit_round_trip(capsys, home, fakes, monkeypatch):
    code, prep, err = run(capsys, "add", "pnpm を使ってほしい", "--source", "claude-code", "--occurred-at", AT, "--source-ref", "n1")
    assert code == 0 and err is None
    assert prep["group_id"] == "me" and prep["token"] and prep["instructions"] and prep["schema"] and "sqgraph-memory commit" in prep["next"]
    assert prep["context"]["message"]["author"] == "user" and prep["context"]["message"]["channel"] == "personal"
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 0

    code, out, err = run(capsys, "commit", "--token", prep["token"], stdin=json.dumps(PNPM, ensure_ascii=False), monkeypatch=monkeypatch)
    assert code == 0 and err is None
    assert out["facts_created"] == 1 and out["new_entities"] == 2 and out["llm_calls"] == 0 and not out["already_stored"]
    _, found, _ = run(capsys, "search", "pnpm")
    assert found["count"] >= 1 and found["facts"][0]["fact"].startswith("ユーザーは npm")

    # the whole thing as one stdin object; ids of candidates now exist
    code, prep2, _ = run(capsys, "add", "-", "--occurred-at", "2026-07-02 10:00", stdin="やっぱり pnpm ではなく bun を使う", monkeypatch=monkeypatch)
    ids = {f["id"] for f in prep2["context"]["related_facts"]}
    assert [c["name"] for c in prep2["context"]["candidate_entities"]][:2] and found["facts"][0]["id"] in ids
    ext = {"entities": [{"ref": "b", "name": "bun", "type": "Service"}],
           "facts": [{"subject": "author", "object": "b", "relation": "PREFERS", "fact": "ユーザーは pnpm ではなく bun を使ってほしい", "contradicts": [found["facts"][0]["id"]]}]}
    code, out, _ = run(capsys, "commit", stdin=json.dumps({"token": prep2["token"], "extraction": ext}), monkeypatch=monkeypatch)
    assert code == 0 and out["facts_invalidated"] == 1 and out["facts_created"] == 1
    _, hist, _ = run(capsys, "search", "pnpm bun", "--history")
    assert {f["status"] for f in hist["facts"]} == {"valid", "ended"}

    # --extraction FILE, idempotency by source_ref
    f = home.parent / "ext.json"
    f.write_text(json.dumps(PNPM, ensure_ascii=False), encoding="utf-8")
    code, prep3, _ = run(capsys, "add", "別のメモ", "--source-ref", "n9")
    code, out, _ = run(capsys, "commit", "--token", prep3["token"], "--extraction", str(f))
    assert code == 0 and out["facts_created"] == 1
    code, again, _ = run(capsys, "add", "別のメモ", "--source-ref", "n9")
    assert again["already_stored"] is True and again["episode_id"] == out["episode_id"] and "token" not in again
    code, out2, _ = run(capsys, "commit", "--token", prep3["token"], "--extraction", str(f))
    assert out2["already_stored"] is True and out2["episode_id"] == out["episode_id"]
    assert all(not s["need_llm"] for s in fakes)


def test_cli_commit_errors(capsys, home, fakes, monkeypatch):
    _, prep, _ = run(capsys, "add", "pnpm を使ってほしい")
    code, _, err = run(capsys, "commit", stdin="", monkeypatch=monkeypatch)
    assert code == 2 and "no extraction" in err["error"]
    code, _, err = run(capsys, "commit", stdin="{not json", monkeypatch=monkeypatch)
    assert code == 2 and "not valid JSON" in err["error"]
    code, _, err = run(capsys, "commit", stdin="{}", monkeypatch=monkeypatch)
    assert code == 2 and "--token" in err["error"]
    code, _, err = run(capsys, "commit", "--token", "bogus", stdin="{}", monkeypatch=monkeypatch)
    assert code == 2 and "invalid token" in err["error"]
    code, _, err = run(capsys, "commit", "--token", prep["token"], stdin='{"facts":[{"subject":"x","relation":"R","fact":"f"}]}', monkeypatch=monkeypatch)
    assert code == 2 and "invalid extraction" in err["error"] and "facts[0].subject" in err["error"]
    code, _, err = run(capsys, "commit", "--token", prep["token"], "--extraction", str(home / "missing.json"))
    assert code == 2 and "cannot read" in err["error"]
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "other")
    code, _, err = run(capsys, "commit", "--token", prep["token"], stdin="{}", monkeypatch=monkeypatch)
    assert code == 2 and "belongs to group 'me'" in err["error"]
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 0


def test_cli_init_in_agent_mode_needs_no_openai_key(capsys, home, monkeypatch):
    code, out, err = run(capsys, "init", "--embed-base-url", "http://192.168.1.20:1234/v1", "--embed-model", "m", "--no-probe", stdin="", monkeypatch=monkeypatch)
    assert code == 0 and err is None
    assert out["extractor"] == "agent" and out["api_key"] == "not-needed" and "OPENAI_API_KEY" not in out.get("warning", "")
    assert not (home / ".env").exists()
    assert json.loads((home / "config.json").read_text(encoding="utf-8"))["extractor"] == "agent"
    # an environment key is not copied into .env (nothing needs it)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-needed")
    code, out, _ = run(capsys, "init")
    assert out["api_key"] == "not-needed" and not (home / ".env").exists()
    monkeypatch.delenv("OPENAI_API_KEY")
    # explicit --extractor openai: the key is required (warning) / stored from stdin
    code, out, _ = run(capsys, "init", "--extractor", "openai", stdin="", monkeypatch=monkeypatch)
    assert out["extractor"] == "openai" and out["api_key"] == "missing" and "OPENAI_API_KEY" in out["warning"]
    code, out, _ = run(capsys, "init", "--extractor", "openai", "--api-key-stdin", stdin="sk-test-1\n", monkeypatch=monkeypatch)
    assert out["api_key"] == "stored" and "sk-test-1" not in json.dumps(out)
    # the OpenAI embedder (no custom endpoint) in agent mode: no prompt, a warning about vector search only
    code, out, _ = run(capsys, "init", "--extractor", "agent", "--embed-base-url", "openai", stdin="", monkeypatch=monkeypatch)
    assert out["extractor"] == "agent" and out["api_key"] == "existing"


def test_cli_init_agent_mode_with_the_openai_embedder_warns_about_vector_search_only(capsys, home, monkeypatch):
    code, out, _ = run(capsys, "init", stdin="", monkeypatch=monkeypatch)
    assert out["extractor"] == "agent" and out["api_key"] == "missing"
    assert "vector" in out["warning"] and "agent extractor needs no key" in out["warning"]


def test_cli_config_shows_and_sets_extractor_and_env_override(capsys, home, monkeypatch):
    code, out, _ = run(capsys, "config")
    assert out["extractor"] == "agent"
    code, out, _ = run(capsys, "config", "--extractor", "openai")
    assert out["extractor"] == "openai" and out["changed"] == ["extractor"]
    assert json.loads((home / "config.json").read_text(encoding="utf-8"))["extractor"] == "openai"
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "agent")
    assert run(capsys, "config")[1]["extractor"] == "agent"  # env override
    assert json.loads((home / "config.json").read_text(encoding="utf-8"))["extractor"] == "openai"  # not persisted
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "bogus")
    code, _, err = run(capsys, "config")
    assert code == 1 and "SQGRAPH_MEMORY_EXTRACTOR" in err["error"]
    code, _, err = run(capsys, "config", "--extractor", "gpt")
    assert code == 2
    # a config.json without the field (or with garbage) means "agent"
    monkeypatch.delenv("SQGRAPH_MEMORY_EXTRACTOR")
    (home / "config.json").write_text(json.dumps({"group_id": "old"}), encoding="utf-8")
    assert run(capsys, "config")[1]["extractor"] == "agent"
    (home / "config.json").write_text(json.dumps({"extractor": "nonsense"}), encoding="utf-8")
    assert run(capsys, "config")[1]["extractor"] == "agent"


def test_cli_add_in_openai_mode_still_calls_the_llm(capsys, home, monkeypatch):
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "openai")
    monkeypatch.setattr(cli, "make_providers", lambda cfg, *, need_llm, need_embedder: (FakeLLM() if need_llm else None, FakeEmbedder(), []))
    code, out, _ = run(capsys, "add", "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}", "--occurred-at", AT)
    assert code == 0 and out["facts_created"] == 1 and out["llm_calls"] >= 2 and "token" not in out


def test_cli_mcp_forwards_the_extractor(capsys, home, monkeypatch):
    from sqgraph.memory import mcp as mcp_main

    seen: list[list[str]] = []
    monkeypatch.setattr(mcp_main, "main", lambda argv=None: seen.append(list(argv)))
    assert cli.main(["mcp"]) == 0
    assert seen[0][seen[0].index("--extractor") + 1] == "agent"
    assert cli.main(["mcp", "--extractor", "openai"]) == 0
    assert seen[1][seen[1].index("--extractor") + 1] == "openai"
    run(capsys, "config", "--extractor", "openai")
    seen.clear()
    assert cli.main(["mcp"]) == 0
    assert seen[0][seen[0].index("--extractor") + 1] == "openai"


def test_cli_doctor_reports_the_extractor(capsys, home, tmp_path, monkeypatch):
    fake_home = tmp_path / "fakeuser"
    fake_home.mkdir()
    monkeypatch.setattr(cli, "user_home", lambda: fake_home)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    names = {c["name"]: c for c in run(capsys, "doctor")[1]["checks"]}
    assert names["api_key"]["status"] == "skip" and "not needed" in names["api_key"]["detail"] and names["extractor"]["status"] == "ok"
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "openai")
    names = {c["name"]: c for c in run(capsys, "doctor")[1]["checks"]}
    assert names["api_key"]["status"] == "warn" and "openai" in names["extractor"]["detail"]


def test_mcp_main_selects_the_extractor(tmp_path, monkeypatch):
    from sqgraph.memory import mcp as mcp_mod

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list[MemoryTools] = []
    monkeypatch.setattr(mcp_mod, "build_server", lambda tools: seen.append(tools) or object())
    monkeypatch.setattr(mcp_mod, "run_server", lambda *a, **k: None)
    env = ["--env-file", str(tmp_path / "nope.env")]
    mcp_mod.main(["--db", str(tmp_path / "a.db"), "--group-id", "me", *env])
    assert seen[-1].extractor == "agent" and seen[-1].memory.llm is None and seen[-1].write_blocker is None
    mcp_mod.main(["--db", str(tmp_path / "b.db"), "--group-id", "me", *env, "--extractor", "openai"])
    assert seen[-1].extractor == "openai" and "OPENAI_API_KEY" in seen[-1].write_blocker
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    mcp_mod.main(["--db", str(tmp_path / "c.db"), "--group-id", "me", *env, "--extractor", "openai"])
    assert seen[-1].memory.llm is not None and seen[-1].write_blocker is None
    mcp_mod.main(["--db", str(tmp_path / "d.db"), "--group-id", "me", *env])
    assert seen[-1].extractor == "agent" and seen[-1].memory.llm is None  # a key does not switch the mode
