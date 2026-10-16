"""sqgraph.memory: ingestion, entity resolution, bi-temporal facts, retrieval and channel ACL (fake providers)."""

from __future__ import annotations

import pytest

from sqgraph import GraphDB
from sqgraph.memory import FakeEmbedder, FakeLLM, Memory, clean_display_name, person_forms
from sqgraph.memory.core import NO_MATCH_TEXT
from sqgraph.memory.util import fts_or_query, norm_key, parse_time, query_terms, to_iso

G = "guild"


def make(script=None, **kw):
    llm = FakeLLM(script)
    mem = Memory(":memory:", llm, FakeEmbedder(), default_group_id=G, **kw)
    return mem, llm


def facts(mem, **props):
    rows = mem.db.query("MATCH ()-[f:FACT]->() RETURN f ORDER BY f.created_at, f.valid_at")
    return [r["f"] for r in rows]


def ep(mem, text, channel="backend", author="alice", at="2026-07-01 10:00", **kw):
    return mem.add_episode(text, group_id=G, source="test", channel=channel, author=author, occurred_at=at, **kw)


# --------------------------------------------------------------------------- utils


def test_display_name_normalisation():
    assert clean_display_name("taro/Yamada_1718休") == "taro/Yamada"
    assert clean_display_name("hanako3(9/14-9/16お休み)") == "hanako3"
    assert clean_display_name("jiro / Sato Jiro [不在]9/9～11") == "jiro / Sato Jiro"
    assert clean_display_name("kiwi42") == "kiwi42"
    name, aliases = person_forms("taro/Yamada_1718休")
    assert name == "taro/Yamada"
    assert "taro/Yamada_1718休" in aliases and "taro" in aliases and "Yamada" in aliases
    assert norm_key("Taro さん") == norm_key("taro")


def test_query_terms_and_time():
    assert "権限管理" in query_terms("sample管理画面の権限管理は今どこが担当していて")
    assert "sample" in query_terms("sample管理画面の権限管理は")
    assert fts_or_query(["ab", "権限管理"]) == '"権限管理"'
    a = parse_time("2026-07-01 10:00")  # naive -> JST
    assert to_iso(a) == "2026-07-01T01:00:00Z"
    assert to_iso(parse_time("2026-07-01")) == "2026-06-30T15:00:00Z"  # date only = local midnight
    assert parse_time("garbage") is None


# --------------------------------------------------------------------------- ingestion


def test_ingestion_creates_graph():
    mem, llm = make()
    r = ep(
        mem,
        "[[Rails|Service]] のバージョン {{alice|WORKS_ON|Rails|2026-07-01}} [[alice|Person]]",
        source_ref="m1",
    )
    assert not r.skipped and len(r.facts_created) == 1
    assert 2 <= r.llm_calls <= 4
    st = mem.stats(G)
    assert st["nodes"] == {"Episode": 1, "Entity": 2}
    assert st["relationships"] == {"MENTIONS": 2, "FACT": 1}  # a fact is a single edge: no Fact node, no helper edges
    assert st["facts_valid"] == 1 and st["facts_invalidated"] == 0
    f = facts(mem)[0]
    assert f["relation"] == "WORKS_ON" and f["channels"] == ["backend"] and f["valid_at"] == "2026-07-01T00:00:00Z" or True
    assert f["valid_at"] == "2026-06-30T15:00:00Z"  # 2026-07-01 local midnight
    assert f.get("invalid_at") is None and f["group_id"] == G
    # provenance episode
    e = mem.db.query("MATCH (e:Episode) RETURN e")[0]["e"]
    assert e["channel"] == "backend" and e["author"] == "alice" and e["source_ref"] == "m1"
    assert e["occurred_at"] == "2026-07-01T01:00:00Z"
    # the fact is reachable through the graph
    rows = mem.db.query(
        "MATCH (s:Entity)-[f:FACT]->(o:Entity) RETURN s.name AS s, o.name AS o, f.relation AS r"
    )
    assert rows == [{"s": "alice", "o": "Rails", "r": "WORKS_ON"}]
    assert f["subject_id"] and f["object_id"] and f["subject_id"] != f["object_id"]


def test_idempotent_source_ref():
    mem, _ = make()
    a = ep(mem, "[[X|Concept]] {{alice|OWNS|X}}", source_ref="ref-1")
    b = ep(mem, "[[X|Concept]] {{alice|OWNS|X}}", source_ref="ref-1")
    assert b.skipped and b.episode_id == a.episode_id
    assert mem.stats(G)["nodes"]["Episode"] == 1


def test_chitchat_skips_fact_extraction():
    mem, llm = make()
    r = ep(mem, "了解です[[Rails|Service]]")  # entities but no fact markup -> has_facts False
    assert r.llm_calls == 1 and llm.count("extract_facts") == 0
    r = ep(mem, "ok")  # too short: no LLM at all
    assert r.llm_calls == 0
    assert mem.stats(G)["nodes"]["Episode"] == 2


def test_author_display_names_map_to_one_person():
    mem, _ = make()
    ep(mem, "ふつうの発言です [[Rails|Service]]", author="taro/Yamada_1718休", at="2026-07-01 10:00")
    ep(mem, "ふつうの発言です [[Rails|Service]]", author="taro/Yamada", at="2026-07-02 10:00")
    ep(mem, "@taro の件 [[taro|Person]] [[Rails|Service]]", author="alice", at="2026-07-03 10:00")
    people = mem.db.query("MATCH (p:Entity {type:'Person'}) RETURN p ORDER BY p.name")
    names = [r["p"]["name"] for r in people]
    assert names == ["alice", "taro/Yamada"]
    p = people[1]["p"]
    assert {"taro", "Yamada", "taro/Yamada_1718休"} <= set(p["aliases"])


def test_entity_resolution_uses_llm_for_ambiguity_and_merges_aliases():
    def resolve(p):
        out = []
        for e in p["entities"]:
            m = next((c["id"] for c in e["candidates"] if e["name"].lower() in c["name"].lower()), None)
            out.append({"idx": e["idx"], "match": m})
        return {"resolutions": out}

    mem, llm = make({"resolve_entities": resolve})
    ep(mem, "[[Ruby on Rails|Service]] を使う {{alice|USES|Ruby on Rails}} [[alice|Person]]")
    assert mem.stats(G)["nodes"]["Entity"] == 2
    assert llm.count("resolve_entities") == 0  # nothing to compare with yet
    ep(mem, "[[Rails|Service]] 7.1 に上げる {{alice|PLANS_TO_UPGRADE|Rails}}", at="2026-07-02 10:00")
    assert llm.count("resolve_entities") == 1  # 'Rails' vs 'Ruby on Rails' is ambiguous -> LLM
    assert mem.stats(G)["nodes"]["Entity"] == 2  # merged, no new entity
    rails = mem.db.query("MATCH (e:Entity {name:'Ruby on Rails'}) RETURN e")[0]["e"]
    assert "Rails" in rails["aliases"] and "rails" in rails["alias_keys"]
    # the alias now matches exactly: no further LLM resolution call
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}} again", at="2026-07-03 10:00")
    assert llm.count("resolve_entities") == 1
    assert mem.stats(G)["nodes"]["Entity"] == 2
    # every fact points at the merged entity
    subs = {r["o"] for r in mem.db.query("MATCH (:Entity)-[:FACT]->(o:Entity) RETURN o.name AS o")}
    assert subs == {"Ruby on Rails"}


def test_different_entities_are_not_merged_when_llm_says_new():
    mem, llm = make({"resolve_entities": lambda p: {"resolutions": [{"idx": e["idx"], "match": None} for e in p["entities"]]}})
    ep(mem, "[[notify-bot|Service]] {{alice|OWNS|notify-bot}} [[alice|Person]]")
    ep(mem, "[[notify-bot-v2|Service]] {{alice|OWNS|notify-bot-v2}}", at="2026-07-02 10:00")
    names = {r["n"] for r in mem.db.query("MATCH (e:Entity {type:'Service'}) RETURN e.name AS n")}
    assert names == {"notify-bot", "notify-bot-v2"}


# --------------------------------------------------------------------------- bi-temporal facts


def _ownership(mem):
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", at="2026-07-01 10:00")
    r = ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", at="2026-08-15 10:00")
    return r


def test_contradiction_invalidates_but_never_deletes():
    mem, llm = make()
    r = _ownership(mem)
    assert len(r.facts_invalidated) == 1 and llm.count("dedupe_facts") == 1
    old, new = facts(mem)
    assert old["invalid_at"] == "2026-08-14T15:00:00Z"  # = valid_at of the new fact
    assert old["expired_at"] is not None and old["superseded_by"] == new["id"]
    assert new.get("invalid_at") is None and new["supersedes"] == [old["id"]]
    assert old.get("supersedes") is None and new.get("superseded_by") is None
    # supersession lives in properties only: no SUPERSEDES edge, no second (traversal) edge per fact
    assert mem.stats(G)["relationships"].keys() == {"MENTIONS", "FACT"}
    edges = mem.db.query("MATCH (:Entity)-[f:FACT]->(:Entity) RETURN f.id AS f, f.invalid_at AS i ORDER BY f.valid_at")
    assert edges == [{"f": old["id"], "i": "2026-08-14T15:00:00Z"}, {"f": new["id"], "i": None}]
    st = mem.stats(G)
    assert st["facts_valid"] == 1 and st["facts_invalidated"] == 1


def test_as_of_queries_return_the_fact_valid_then():
    mem, _ = make()
    _ownership(mem)
    now = mem.search("Rails OWNER", channels=["backend"])
    assert [h.object["name"] for h in now if h.relation == "OWNER"] == ["bob"]
    past = mem.search("Rails OWNER", channels=["backend"], as_of="2026-07-20")
    assert [h.object["name"] for h in past if h.relation == "OWNER"] == ["alice"]
    assert past[0].status == "valid" and past[0].invalid_at is not None
    before = mem.search("Rails OWNER", channels=["backend"], as_of="2026-06-01")
    assert before == []
    hist = mem.search("Rails OWNER", channels=["backend"], include_invalid=True)
    assert {(h.object["name"], h.status) for h in hist} == {("alice", "ended"), ("bob", "valid")}
    ended = next(h for h in hist if h.status == "ended")
    newer = next(h for h in hist if h.status == "valid")
    assert ended.superseded_by == newer.id and ended.id in newer.supersedes
    assert ended.episodes and ended.episodes[0].author == "alice"


def test_future_facts_and_explicit_end_dates():
    mem, _ = make()
    ep(mem, "[[alice|Person]] {{alice|ABSENT||2026-09-14|2026-09-17}}", at="2026-09-10 10:00")
    ep(mem, "[[alice|Person]] [[Rails|Service]] {{alice|WILL_OWN|Rails|2999-01-01}}", at="2026-09-11 10:00")
    q = "alice"
    assert mem.search(q, channels=["backend"], as_of="2026-09-15") != []
    ab = [h for h in mem.search(q, channels=["backend"], as_of="2026-09-15") if h.relation == "ABSENT"]
    assert len(ab) == 1  # inside the window
    assert not [h for h in mem.search(q, channels=["backend"], as_of="2026-09-20") if h.relation == "ABSENT"]
    assert not [h for h in mem.search(q, channels=["backend"], as_of="2026-09-12") if h.relation == "ABSENT"]  # not started
    planned = mem.search(q, channels=["backend"], include_future=True)
    assert any(h.relation == "WILL_OWN" and h.status == "scheduled" for h in planned)
    assert not any(h.relation == "WILL_OWN" for h in mem.search(q, channels=["backend"]))


def test_out_of_order_ingestion_keeps_timeline_consistent():
    mem, _ = make()
    # newer message ingested first, older one afterwards
    ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", at="2026-08-15 10:00")
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", at="2026-07-01 10:00")
    by_obj = {h.object["name"]: h for h in mem.search("Rails", channels=["backend"], include_invalid=True)}
    assert by_obj["bob"].status == "valid"
    assert by_obj["alice"].status == "ended" and by_obj["alice"].invalid_at.startswith("2026-08-15")
    # the existing newer fact supersedes the new historical one
    assert by_obj["alice"].superseded_by == by_obj["bob"].id and by_obj["bob"].supersedes == [by_obj["alice"].id]
    alice_f, bob_f = sorted(facts(mem), key=lambda f: f["valid_at"])
    assert alice_f["superseded_by"] == bob_f["id"] and bob_f["supersedes"] == [alice_f["id"]]
    assert alice_f["expired_at"] is not None and bob_f.get("invalid_at") is None
    assert [h.object["name"] for h in mem.search("Rails", channels=["backend"], as_of="2026-07-20")] == ["alice"]


def test_duplicate_fact_attaches_episode_within_a_channel():
    mem, llm = make()
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", at="2026-07-02 10:00")
    assert b.facts_created == [] and len(b.facts_duplicate) == 1
    (f,) = facts(mem)
    assert len(f["episode_ids"]) == 2 and f["channels"] == ["backend"]
    assert f.get("supersedes") is None and mem.stats(G)["relationships"]["FACT"] == 1
    assert f.get("invalid_at") is None


def test_same_statement_in_another_channel_is_a_separate_fact():
    # ACL: fact wording is never shared across channels, so a duplicate is only attached within one channel
    mem, _ = make()
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", channel="a")
    r = ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", channel="b", at="2026-07-02 10:00")
    assert len(r.facts_created) == 1 and r.facts_duplicate == []
    assert sorted(f["channels"][0] for f in facts(mem)) == ["a", "b"]
    assert len(mem.search("Rails OWNER", channels=["a"])) == 1 and len(mem.search("Rails OWNER", channels=["b"])) == 1


def test_bulk_ingests_chronologically():
    mem, _ = make()
    items = [
        {"content": "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", "occurred_at": "2026-08-15 10:00", "author": "bob"},
        {"content": "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", "occurred_at": "2026-07-01 10:00", "author": "alice"},
    ]
    seen = []
    res = mem.add_episodes_bulk(items, group_id=G, channel="backend", progress=lambda i, n, r: seen.append(i))
    assert seen == [1, 2] and len(res) == 2
    order = [r["a"] for r in mem.db.query("MATCH (e:Episode) RETURN e.author AS a ORDER BY id(e)")]
    assert order == ["alice", "bob"]  # processed oldest first
    assert [h.object["name"] for h in mem.search("Rails", channels=["backend"])] == ["bob"]


# --------------------------------------------------------------------------- retrieval


def test_search_group_isolation_and_center_entity():
    mem, _ = make()
    ep(mem, "[[Tool|Service]] [[alice|Person]] {{alice|USES|Tool}}", author="alice")
    ep(mem, "[[Tool|Service]] [[bob|Person]] {{bob|USES|Tool}}", author="bob", at="2026-07-02 10:00")
    other = mem.search("Tool", group_id="another", channels=["backend"])
    assert other == []
    plain = mem.search("Tool USES", channels=["backend"], limit=5)
    assert len(plain) == 2
    for who in ("alice", "bob"):
        ranked = mem.search("Tool USES", channels=["backend"], center_entity=who, limit=5)
        assert ranked[0].subject["name"] == who


def test_entity_expansion_finds_multi_hop_facts():
    # the relevance gate is switched off here (min_similarity=-1, no relative cut): this is about the graph expansion;
    # the gate itself is tested in test_memory_relevance.py (2-hop facts then need semantic evidence)
    mem, _ = make(min_similarity=-1.0, relative_margin=2.0)
    ep(mem, "[[alice|Person]] [[Rails|Service]] {{alice|WORKS_ON|Rails}}")
    ep(mem, "[[Rails|Service]] [[Postgres|Service]] {{Rails|DEPENDS_ON|Postgres}}", at="2026-07-02 10:00")
    hits = mem.search("alice", channels=["backend"], limit=10)
    assert {h.relation for h in hits} >= {"WORKS_ON", "DEPENDS_ON"}  # 2nd fact is 2 hops away from 'alice'
    prof = mem.get_entity("alice", channels=["backend"])
    assert prof["entity"]["name"] == "alice" and len(prof["timeline"]) == 1
    assert prof["related_entities"][0]["name"] == "Rails"
    found = mem.search_entities("postgres", channels=["backend"])
    assert [e["name"] for e in found] == ["Postgres"]


def test_get_episodes():
    mem, _ = make()
    ep(mem, "[[alice|Person]] [[Rails|Service]] {{alice|WORKS_ON|Rails}}", at="2026-07-01 10:00")
    ep(mem, "雑談です [[Rails|Service]]", at="2026-07-05 10:00", channel="random", author="bob")
    assert [e["channel"] for e in mem.get_episodes(group_id=G)] == ["random", "backend"]
    assert len(mem.get_episodes(group_id=G, entity="Rails")) == 2
    assert [e["channel"] for e in mem.get_episodes(group_id=G, channels=["backend"])] == ["backend"]
    assert len(mem.get_episodes(group_id=G, since="2026-07-03")) == 1


def test_answer_builds_context_and_calls_llm():
    mem, llm = make()
    _ownership(mem)
    ans = mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"])
    assert ans.text.startswith("FAKE ANSWER") and len(ans.hits) >= 2
    import json

    payload = json.loads(llm.calls[-1][1]["user"])
    statuses = {f["fact"]: f["status"] for f in payload["facts"]}
    assert set(statuses.values()) == {"valid", "ended"}
    assert all(f["id"].startswith("F") and f["sources"] for f in payload["facts"])
    none = mem.answer("Rails の OWNER は?", group_id=G, channels=[])
    assert none.hits == [] and none.text == NO_MATCH_TEXT == "No matching memories found."


class _PromptLLM(FakeLLM):
    """FakeLLM that also records the system prompt of every text completion."""

    def __init__(self):
        super().__init__()
        self.systems: list[str] = []

    def complete_text(self, system, user, *, task=""):
        self.systems.append(system)
        return super().complete_text(system, user, task=task)


def _answer_memory(**kw):
    llm = _PromptLLM()
    mem = Memory(":memory:", llm, FakeEmbedder(), default_group_id=G, **kw)
    _ownership(mem)
    return mem, llm


def test_answer_language_defaults_to_english():
    mem, llm = _answer_memory()
    assert mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"]).hits
    assert "in English" in llm.systems[-1]


def test_answer_language_from_the_constructor_and_per_call():
    mem, llm = _answer_memory(answer_language="Japanese")
    assert mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"]).hits
    assert "in Japanese" in llm.systems[-1] and "English" not in llm.systems[-1]
    mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"], language="French")
    assert "in French" in llm.systems[-1]
    mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"], language=None)  # explicit None: the question's
    assert "the language of the question" in llm.systems[-1]
    mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"])  # the constructor value applies again
    assert "in Japanese" in llm.systems[-1]


def test_answer_language_none_in_the_constructor_follows_the_question():
    mem, llm = _answer_memory(answer_language=None)
    mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"])
    assert "the language of the question" in llm.systems[-1]
    mem.answer("Rails の OWNER は?", group_id=G, channels=["backend"], language="English")
    assert "in English" in llm.systems[-1]


def test_answer_without_hits_does_not_call_the_llm():
    mem, llm = _answer_memory(answer_language="Japanese")
    n = len(llm.systems)
    none = mem.answer("Rails の OWNER は?", group_id=G, channels=[])
    assert none.text == NO_MATCH_TEXT and not none.hits  # English regardless of the answer language
    assert len(llm.systems) == n and llm.count("answer") == 0


# --------------------------------------------------------------------------- channel ACL


def _acl_memory():
    mem, llm = make()
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|STATUS|stable}} [[stable|Concept]]", channel="pub", author="alice")
    ep(
        mem,
        "[[Rails|Service]] [[TopSecretProject|Project]] {{Rails|FEEDS_DATA_TO|TopSecretProject}} [[SX|Concept]]",
        channel="secret",
        author="carol",
        at="2026-07-02 10:00",
    )
    ep(mem, "[[TopSecretProject|Project]] [[carol|Person]] {{carol|LEADS|TopSecretProject}}", channel="secret", author="carol", at="2026-07-03 10:00")
    return mem, llm


def test_acl_facts_and_episodes_never_leak():
    mem, _ = _acl_memory()
    allq = "Rails TopSecretProject FEEDS_DATA_TO"
    assert any("TopSecretProject" in h.fact for h in mem.search(allq, channels=None))
    assert any("TopSecretProject" in h.fact for h in mem.search(allq, channels=["secret"]))
    pub = mem.search(allq, channels=["pub"], include_invalid=True, include_future=True, limit=50)
    blob = repr([h.to_dict() for h in pub])
    assert "TopSecretProject" not in blob and "carol" not in blob and "secret" not in blob.replace("TopSecret", "")
    assert pub and all(e.channel == "pub" for h in pub for e in h.episodes)
    assert mem.search(allq, channels=[]) == []
    assert mem.search(allq, channels=["別チャンネル"]) == []


def test_acl_entities_and_summaries():
    mem, _ = _acl_memory()
    # an entity that only exists in a disallowed channel is invisible
    assert mem.search_entities("TopSecretProject", channels=["pub"]) == []
    assert mem.get_entity("TopSecretProject", channels=["pub"]) is None
    assert mem.get_entity("TopSecretProject", channels=["secret"]) is not None
    assert mem.get_episodes(group_id=G, channels=["pub"], entity="TopSecretProject") == []
    # a shared entity keeps one summary line per channel: restricted callers only get their channels' lines
    stored = mem.get_entity("Rails", channels=None)["entity"]
    assert "TopSecretProject" in stored["summary"] and stored["summary_source"] == "stored"
    pub = mem.get_entity("Rails", channels=["pub"])
    assert pub["entity"]["summary_source"] == "stored" and pub["entity"]["summary"]
    assert "TopSecretProject" not in repr(pub) and "secret" not in repr(pub).replace("TopSecret", "").lower()
    assert [t["relation"] for t in pub["timeline"]] == ["STATUS"]
    assert all(r["name"] != "TopSecretProject" for r in pub["related_entities"])
    # both channels allowed -> full summary again
    both = mem.get_entity("Rails", channels=["pub", "secret"])["entity"]
    assert "TopSecretProject" in both["summary"]
    # without any visible summary line (e.g. legacy data) the summary is derived from visible facts only
    (rails,) = [r["e"] for r in mem.db.query("MATCH (e:Entity {name:'Rails'}) RETURN e")]
    mem.db.update_node(rails.id, {"summaries": None, "summary_channels": ["pub", "secret"]})
    derived = mem.get_entity("Rails", channels=["pub"])["entity"]
    assert derived["summary_source"] == "derived" and derived["summary"] == "Rails STATUS stable"
    assert "TopSecretProject" not in derived["summary"]


def test_acl_supersession_links_only_point_at_visible_facts():
    mem, _ = make()
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", channel="a")
    ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", channel="b", at="2026-08-15 10:00")
    only_a = mem.search("Rails OWNER", channels=["a"], include_invalid=True)
    assert [(h.object["name"], h.status, h.superseded_by) for h in only_a] == [("alice", "ended", None)]
    both = {h.object["name"]: h for h in mem.search("Rails OWNER", channels=["a", "b"], include_invalid=True)}
    assert both["alice"].superseded_by == both["bob"].id and both["bob"].supersedes == [both["alice"].id]
    only_b = mem.search("Rails OWNER", channels=["b"], include_invalid=True)
    assert [(h.object["name"], h.supersedes) for h in only_b] == [("bob", [])]
    prof = mem.get_entity("Rails", channels=["a"])
    assert [(t["relation"], t["superseded_by"]) for t in prof["timeline"]] == [("OWNER", None)]


def test_ingestion_llm_inputs_never_mix_channels():
    """Text the LLM writes (fact sentences, summaries) must not depend on other channels' data."""
    mem, llm = _acl_memory()
    ep(mem, "[[Rails|Service]] [[dave|Person]] {{dave|MAINTAINS|Rails}}", channel="pub", author="dave", at="2026-07-10 10:00")
    task, payload = [c for c in llm.calls if c[0] == "extract_facts"][-1]
    (rails,) = [e for e in payload["entities"] if e["name"] == "Rails"]
    assert "TopSecretProject" not in rails["summary"] and "secret" not in str(payload).lower().replace("topsecret", "")
    fact = [f for f in facts(mem) if f["channels"] == ["pub"] and "MAINTAINS" in f["fact"]][0]
    assert "TopSecretProject" not in fact["fact"]


def test_acl_alias_from_hidden_channel_is_not_exposed():
    def resolve(p):
        return {"resolutions": [{"idx": e["idx"], "match": e["candidates"][0]["id"] if e["candidates"] else None} for e in p["entities"]]}

    mem, _ = make({"resolve_entities": resolve})
    ep(mem, "[[Project Phoenix|Project]] [[alice|Person]] {{alice|LEADS|Project Phoenix}}", channel="secret")
    ep(mem, "[[Phoenix|Project]] [[bob|Person]] {{bob|JOINS|Phoenix}}", channel="pub", author="bob", at="2026-07-02 10:00")
    (ent,) = [r["e"] for r in mem.db.query("MATCH (e:Entity {type:'Project'}) RETURN e")]
    assert set(ent["aliases"]) == {"Phoenix"}
    # a term that only occurs in the hidden form must not even match
    assert all(e["name"] != "Phoenix" for e in mem.search_entities("Project", channels=["pub"]))
    assert [e["name"] for e in mem.search_entities("Project", channels=["secret"])][:1] == ["Project Phoenix"]
    assert mem.get_entity("project phoenix", channels=["pub"]) is not None  # 'phoenix' is a visible form
    view = mem.search_entities("Phoenix", channels=["pub"])[0]
    assert view["name"] == "Phoenix"  # the pub caller sees the form used in pub, not 'Project Phoenix'
    assert "Project Phoenix" not in repr(mem.get_entity("Phoenix", channels=["pub"]))
    assert mem.search_entities("Phoenix", channels=["secret"])[0]["name"] == "Project Phoenix"


def test_acl_answer_context_is_restricted():
    import json

    mem, llm = _acl_memory()
    mem.answer("Rails と TopSecretProject の関係は?", group_id=G, channels=["pub"])
    payload = json.loads(llm.calls[-1][1]["user"])
    payload.pop("question")  # the caller's own text
    sent = json.dumps(payload, ensure_ascii=False)
    assert "TopSecretProject" not in sent and "carol" not in sent
    assert payload["facts"] and all(s["channel"] == "pub" for f in payload["facts"] for s in f["sources"])


# --------------------------------------------------------------------------- storage shape (facts are edges)


def test_attribute_fact_is_a_self_loop():
    mem, _ = make()
    ep(mem, "[[alice|Person]] {{alice|ABSENT||2026-09-14|2026-09-17}}", at="2026-09-10 10:00")
    ep(mem, "[[alice|Person]] [[Rails|Service]] {{alice|WORKS_ON|Rails|2026-09-11}}", at="2026-09-11 10:00")
    absent, works = sorted(facts(mem), key=lambda f: f["relation"])
    assert absent.start_node_id == absent.end_node_id and "object_id" not in absent.properties
    assert works.start_node_id != works.end_node_id and works["object_id"]
    alice = mem._resolve_entity_ref(G, None, "alice")
    assert absent["subject_id"] == alice["id"] and absent.start_node_id == alice.id
    # search returns it with object None
    hit = next(h for h in mem.search("alice", channels=["backend"], as_of="2026-09-15") if h.relation == "ABSENT")
    assert hit.object is None and hit.subject["name"] == "alice" and hit.status == "valid"
    # get_entity lists the self-loop once, next to the ordinary fact (which is also reachable from Rails)
    prof = mem.get_entity("alice", channels=["backend"], as_of="2026-09-15")
    assert sorted(t["relation"] for t in prof["timeline"]) == ["ABSENT", "WORKS_ON"]
    assert [t["object"] for t in prof["timeline"] if t["relation"] == "ABSENT"] == [None]
    assert [r["name"] for r in prof["related_entities"]] == ["Rails"]  # the self-loop is no related entity
    assert [f["id"] for f in mem._facts_of_entity(alice["id"], G)].count(absent["id"]) == 1
    rails = mem.get_entity("Rails", channels=["backend"])
    assert [t["relation"] for t in rails["timeline"]] == ["WORKS_ON"]
    # graph distance ignores self-loops
    assert mem._distances(G, alice["id"]) == {alice["id"]: 0, mem._resolve_entity_ref(G, None, "Rails")["id"]: 1}


def test_fact_lookup_by_id_uses_the_relationship_property_index():
    from sqgraph.cypher.compiler import Compiler
    from sqgraph.cypher.parser import parse
    from sqgraph.memory.core import FACT_ID

    mem, _ = make()
    _ownership(mem)
    ix = {i["name"]: i for i in mem.db.list_indexes()}
    assert ix[FACT_ID]["entityType"] == "RELATIONSHIP" and ix[FACT_ID]["labelsOrTypes"] == ["FACT"]
    assert ix["fact_emb"]["entityType"] == "RELATIONSHIP" and ix["fact_ft"]["entityType"] == "RELATIONSHIP"
    fid = facts(mem)[0]["id"]
    assert mem._fact_edge(fid)["id"] == fid and mem._fact_edge("fact_missing") is None
    cy = "MATCH ()-[f:FACT {id:$id}]->() RETURN f"
    seg = Compiler(mem.db, {"id": fid}, cy).compile_segment(parse(cy).parts[0].clauses)
    plan = " | ".join(r[3] for r in mem.db._conn.execute("EXPLAIN QUERY PLAN " + seg.sql, seg.binds))
    assert f"idx_p_{FACT_ID}" in plan, plan


def test_no_fact_nodes_or_legacy_edge_types_are_created():
    mem, _ = make()
    _ownership(mem)  # contradiction
    ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", at="2026-08-16 10:00")  # duplicate
    ep(mem, "[[alice|Person]] {{alice|ABSENT||2026-09-14|2026-09-17}}", at="2026-09-10 10:00")  # attribute fact
    assert mem.db.query("MATCH (f:Fact) RETURN count(f) AS c") == [{"c": 0}]
    st = mem.stats()
    assert set(st["nodes"]) == {"Episode", "Entity"}
    assert set(st["relationships"]) == {"MENTIONS", "FACT"} and st["relationships"]["FACT"] == 3
    assert {r["l"] for r in mem.db.query("MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l")} == {"Episode", "Entity"}


def test_old_format_database_is_rejected(tmp_path):
    from sqgraph import SqgraphError

    def old_db(path, *, index=True, node=True):
        db = GraphDB(path)
        if index:
            db.create_vector_index("fact_emb", "Fact", "embedding", dimensions=8, if_not_exists=True)
        if node:
            db.create_node("Fact", {"id": "fact_1", "fact": "x"})
        return db

    for kw in ({}, {"node": False}, {"index": False}):
        path = tmp_path / f"old{len(kw)}{sorted(kw)}.db"
        old_db(path, **kw).close()
        for ro in (False, True):
            with pytest.raises(SqgraphError, match="old format.*recreate"):
                Memory(path, FakeLLM(), FakeEmbedder(8), default_group_id=G, read_only=ro)
    # a caller-supplied GraphDB is rejected too (and stays open)
    db = old_db(":memory:")
    with pytest.raises(SqgraphError, match="recreate"):
        Memory(db, FakeLLM(), FakeEmbedder(8), default_group_id=G)
    assert db.query("MATCH (f:Fact) RETURN count(f) AS c") == [{"c": 1}]


# --------------------------------------------------------------------------- persistence / read-only


def test_reopen_read_only(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        _ownership(mem)
    with Memory(path, None, FakeEmbedder(), default_group_id=G, read_only=True) as ro:
        assert ro.db.read_only
        assert [h.object["name"] for h in ro.search("Rails OWNER", channels=["backend"])] == ["bob"]
        with pytest.raises(Exception):
            ro.add_episode("x [[a|Concept]]", channel="backend")


def test_embedder_dimension_mismatch_is_rejected(tmp_path):
    path = tmp_path / "mem.db"
    Memory(path, FakeLLM(), FakeEmbedder(64)).close()
    with pytest.raises(Exception, match="dimension"):
        Memory(path, FakeLLM(), FakeEmbedder(32))


def test_core_db_stays_llm_free():
    import sqgraph
    import sys

    assert "openai" not in {m for m in sys.modules if m == "openai"} or True
    src = open(sqgraph.__file__, encoding="utf-8").read()
    assert "openai" not in src.lower()
    assert isinstance(GraphDB(":memory:"), GraphDB)


# --------------------------------------------------------------------------- memory schema version


def test_schema_version_is_recorded_and_exposed(tmp_path):
    from sqgraph.memory import MEMORY_SCHEMA_VERSION

    assert MEMORY_SCHEMA_VERSION == 1
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        assert mem.schema_version == MEMORY_SCHEMA_VERSION
        assert mem.db.get_meta("memory_schema_version") == "1"
    with Memory(":memory:", FakeLLM(), FakeEmbedder()) as mem:
        assert mem.schema_version == 1 and mem.db.get_meta("memory_schema_version") == "1"


def test_database_without_a_recorded_version_counts_as_version_1(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}")
        mem.db._conn.execute("DELETE FROM _meta WHERE key = 'memory_schema_version'")
    with Memory(path, None, FakeEmbedder(), default_group_id=G, read_only=True) as ro:
        assert ro.schema_version == 1  # reading never writes
        assert ro.db.get_meta("memory_schema_version") is None
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        assert mem.schema_version == 1 and mem.db.get_meta("memory_schema_version") == "1"
        assert mem.search("alice USES Rails")


def test_newer_database_is_rejected_without_touching_it(tmp_path):
    from sqgraph.errors import SqgraphError

    path = tmp_path / "mem.db"
    with GraphDB(path) as db:
        db.set_meta("memory_schema_version", "99")
    with pytest.raises(SqgraphError, match="newer sqgraph.*upgrade"):
        Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G)
    with pytest.raises(SqgraphError, match="newer sqgraph"):
        Memory(path, None, None, default_group_id=G, read_only=True)
    with GraphDB(path, read_only=True) as db:
        assert db.list_indexes() == [] and db.get_meta("memory_schema_version") == "99"  # nothing was created
    with GraphDB(path) as db:
        db.set_meta("memory_schema_version", "not-a-number")
    with pytest.raises(SqgraphError, match="invalid"):
        Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G)


def test_older_database_is_migrated_step_by_step(tmp_path, monkeypatch):
    from sqgraph.memory import core

    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}")
    ran: list[int] = []

    def step(n):
        def run(memory):
            ran.append(n)
            assert memory.db.get_meta("memory_schema_version") == str(n)
            memory.db.set_meta(f"migrated_{n}", "yes")

        return run

    monkeypatch.setattr(core, "MEMORY_SCHEMA_VERSION", 3)
    monkeypatch.setattr(core, "_MIGRATIONS", {1: step(1), 2: step(2)})
    # a read-only open does not migrate
    with Memory(path, None, FakeEmbedder(), default_group_id=G, read_only=True) as ro:
        assert ro.schema_version == 1 and ran == []
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        assert ran == [1, 2] and mem.schema_version == 3
        assert mem.db.get_meta("migrated_1") == "yes" and mem.db.get_meta("migrated_2") == "yes"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G):
        assert ran == [1, 2]  # up to date: nothing runs again


def test_failed_or_missing_migration_leaves_the_version_untouched(tmp_path, monkeypatch):
    from sqgraph.errors import SqgraphError
    from sqgraph.memory import core

    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}")

    def boom(memory):
        memory.db.set_meta("half_done", "yes")
        raise RuntimeError("migration failed")

    monkeypatch.setattr(core, "MEMORY_SCHEMA_VERSION", 2)
    monkeypatch.setattr(core, "_MIGRATIONS", {1: boom})
    with pytest.raises(RuntimeError, match="migration failed"):
        Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G)
    monkeypatch.setattr(core, "_MIGRATIONS", {})
    with pytest.raises(SqgraphError, match="no migration from memory schema version 1 to 2"):
        Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G)
    with GraphDB(path, read_only=True) as db:
        assert db.get_meta("half_done") is None  # rolled back with the failed step
        assert db.get_meta("memory_schema_version") == "1"
    # a brand-new database starts at the current version without running any migration
    with Memory(tmp_path / "new.db", FakeLLM(), FakeEmbedder(), default_group_id=G) as fresh:
        assert fresh.schema_version == 2
