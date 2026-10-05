"""sqgraph.memory forget API: episode / fact / entity / range / purge, cascades, dry run, physical deletion."""

from __future__ import annotations

import pytest

from sqgraph import GraphDB
from sqgraph.errors import SqgraphError
from sqgraph.memory import FakeEmbedder, FakeLLM, ForgetResult, Memory
from sqgraph.memory.core import ENTITY_INDEX

G = "personal"
H = "other"


@pytest.fixture
def mem(tmp_path):
    m = Memory(tmp_path / "mem.db", FakeLLM(), FakeEmbedder(), default_group_id=G)
    yield m
    m.close()


def ep(mem, text, channel="chat", author="alice", at="2026-07-01 10:00", group=G, **kw):
    return mem.add_episode(text, group_id=group, source="test", channel=channel, author=author, occurred_at=at, **kw)


def facts(mem, group=G):
    return {r["f"]["id"]: r["f"] for r in mem.db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f", {"g": group})}


def entity(mem, name, group=G):
    rows = mem.db.query("MATCH (e:Entity {group_id:$g, name:$n}) RETURN e", {"g": group, "n": name})
    return rows[0]["e"] if rows else None


def counts(mem, group=G):
    st = mem.stats(group)
    return st["nodes"]["Episode"], st["nodes"]["Entity"], st["facts_valid"] + st["facts_invalidated"]


def snapshot(mem):
    """Everything stored (props of all nodes / edges and the index tables) for 'nothing was written' checks."""
    c = mem.db._conn
    return (
        c.execute("SELECT id, props FROM nodes ORDER BY id").fetchall(),
        c.execute("SELECT id, src, dst, type, props FROM edges ORDER BY id").fetchall(),
        c.execute("SELECT * FROM vec_entity_emb ORDER BY node_id").fetchall(),
        c.execute("SELECT * FROM vec_fact_emb ORDER BY edge_id").fetchall(),
    )


# --------------------------------------------------------------------------- result type


def test_forget_result_to_dict():
    r = ForgetResult(episodes_deleted=["a"], facts_restored=["f"], dry_run=True)
    d = r.to_dict()
    assert d == {
        "episodes_deleted": ["a"],
        "facts_deleted": [],
        "facts_restored": ["f"],
        "entities_deleted": [],
        "entities_updated": [],
        "dry_run": True,
    }
    d["episodes_deleted"].append("b")
    assert r.episodes_deleted == ["a"]


# --------------------------------------------------------------------------- forget_episode


def test_forget_episode_deletes_sole_source_facts_and_orphans(mem):
    keep = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    gone = ep(mem, "[[Vault|Project]] [[carol|Person]] {{carol|OWNS|Vault}}", author="bob", at="2026-07-02 10:00")
    assert counts(mem) == (2, 5, 2)
    res = mem.forget_episode(gone.episode_id)
    assert res.episodes_deleted == [gone.episode_id] and not res.dry_run
    assert res.facts_deleted == gone.facts_created
    assert set(res.entities_deleted) == set(gone.new_entities)  # Vault, carol, bob: mentioned only there
    assert entity(mem, "Vault") is None and entity(mem, "carol") is None and entity(mem, "bob") is None
    assert entity(mem, "Rails") is not None and entity(mem, "alice") is not None
    assert counts(mem) == (1, 2, 1)
    assert mem.get_episodes(limit=10)[0]["id"] == keep.episode_id
    # searching finds nothing of the forgotten episode, neither by fact nor by fulltext
    assert all("Vault" not in h.fact for h in mem.search("Vault OWNS", include_invalid=True))
    assert mem.db.fulltext_search("episode_ft", '"vault"') == []
    assert mem.db.fulltext_search("fact_ft", '"vault"') == []
    assert mem.db.fulltext_search("entity_ft", '"vault"') == []
    # the vectors are gone as well
    assert mem.db._conn.execute("SELECT count(*) FROM vec_fact_emb").fetchone()[0] == 1
    assert mem.db._conn.execute("SELECT count(*) FROM vec_entity_emb").fetchone()[0] == 2


def test_forget_episode_keeps_multi_source_facts(mem):
    a = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00", source_ref="m1")
    b = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-02 10:00", source_ref="m2")
    fid = a.facts_created[0]
    assert b.facts_duplicate == [fid]
    assert facts(mem)[fid]["episode_ids"] == [a.episode_id, b.episode_id]
    res = mem.forget_episode(source="test", source_ref="m1")  # by (source, source_ref)
    assert res.episodes_deleted == [a.episode_id] and res.facts_deleted == []
    f = facts(mem)[fid]
    assert f["episode_ids"] == [b.episode_id] and f["channels"] == ["chat"]
    assert entity(mem, "Rails") is not None  # still mentioned by b
    assert mem.search("alice USES Rails")[0].id == fid
    # now the last source goes: the fact and the entities follow
    res2 = mem.forget_episode(b.episode_id)
    assert res2.facts_deleted == [fid] and facts(mem) == {}
    assert counts(mem) == (0, 0, 0)


def test_forget_episode_recomputes_fact_channels(mem):
    a = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", channel="work", source_ref="m1")
    # a duplicate from another channel is its own fact (ACL), so attach a same-channel duplicate manually
    b = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", channel="work", at="2026-07-02 10:00", source_ref="m2")
    fid = a.facts_created[0]
    mem.db.update_edge(
        mem._fact_edge(fid).id, {"channels": ["work", "side"], "episode_ids": [a.episode_id, b.episode_id]}
    )
    mem.forget_episode(a.episode_id)
    assert facts(mem)[fid]["channels"] == ["work"]


def test_forget_episode_unknown_and_validation(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}")
    before = counts(mem)
    assert mem.forget_episode("ep_nope").episodes_deleted == []
    assert mem.forget_episode(source="test", source_ref="nope").to_dict()["facts_deleted"] == []
    assert counts(mem) == before
    with pytest.raises(ValueError):
        mem.forget_episode()
    with pytest.raises(ValueError):
        mem.forget_episode(source="test")  # a source alone is not an episode


def test_forget_episode_is_scoped_to_the_group(mem):
    a = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", group=H)
    assert mem.forget_episode(a.episode_id).episodes_deleted == []  # default group is G
    assert counts(mem, H) == (1, 2, 1)
    assert mem.forget_episode(a.episode_id, group_id=H).episodes_deleted == [a.episode_id]


# --------------------------------------------------------------------------- forget_fact / supersession


def _owner_history(mem):
    a = ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", author="alice", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", author="bob", at="2026-08-15 10:00")
    return a.facts_created[0], b.facts_created[0], a, b


def test_forget_fact_restores_the_superseded_fact(mem):
    old, new, _, b = _owner_history(mem)
    assert facts(mem)[old]["superseded_by"] == new and facts(mem)[old]["invalid_at"]
    assert facts(mem)[new]["supersedes"] == [old]
    res = mem.forget_fact(new)
    assert res.facts_deleted == [new] and res.facts_restored == [old] and res.episodes_deleted == []
    f = facts(mem)[old]
    assert "invalid_at" not in f.properties and "expired_at" not in f.properties and "superseded_by" not in f.properties
    hits = mem.search("Rails OWNER")
    assert [h.id for h in hits if h.relation == "OWNER"] == [old]
    assert hits[0].status == "valid" and hits[0].superseded_by is None
    assert mem.stats(G)["facts_invalidated"] == 0
    # the episode of the forgotten fact is still stored, and bob is still mentioned there
    assert mem.get_episodes(limit=5)[0]["id"] == b.episode_id and entity(mem, "bob") is not None


def test_forget_historical_fact_updates_the_newer_facts_supersedes(mem):
    old, new, _, _ = _owner_history(mem)
    res = mem.forget_fact(old)
    assert res.facts_deleted == [old] and res.facts_restored == []
    f = facts(mem)[new]
    assert "supersedes" not in f.properties  # emptied list is removed
    assert mem.search("Rails OWNER", include_invalid=True)[0].supersedes == []


def test_forget_fact_in_a_chain_relinks_the_older_fact_to_the_newer(mem):
    a = ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-01}}", at="2026-08-01 10:00")
    c = ep(mem, "[[Rails|Service]] [[carol|Person]] {{Rails|OWNER|carol|2026-09-01}}", at="2026-09-01 10:00")
    fa, fb, fc = a.facts_created[0], b.facts_created[0], c.facts_created[0]
    res = mem.forget_fact(fb)
    assert res.facts_deleted == [fb] and res.facts_restored == []  # alice's fact stays ended: carol still owns it
    fs = facts(mem)
    assert fs[fa]["superseded_by"] == fc and fs[fa]["invalid_at"] == fs[fc]["valid_at"]
    assert fs[fc]["supersedes"] == [fa]
    # and with both later facts gone, the first one is valid again
    res2 = mem.forget_fact(fc)
    assert res2.facts_restored == [fa] and "superseded_by" not in facts(mem)[fa].properties


def test_forget_fact_keeps_a_stated_end_time_of_a_historical_fact(mem):
    newer = ep(mem, "[[Rails|Service]] [[bob|Person]] {{Rails|OWNER|bob|2026-08-15}}", author="bob", at="2026-08-15 10:00")
    # ingested later but older, with its own earlier end date: it ends where *it* says
    older = ep(mem, "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01|2026-07-10}}", at="2026-07-01 10:00")
    fn, fo = newer.facts_created[0], older.facts_created[0]
    assert facts(mem)[fo]["superseded_by"] == fn and facts(mem)[fo]["invalid_at"] == "2026-07-09T15:00:00Z"
    res = mem.forget_fact(fn)
    assert res.facts_restored == [fo]
    f = facts(mem)[fo]
    assert f["invalid_at"] == "2026-07-09T15:00:00Z"  # its own end time survives
    assert "superseded_by" not in f.properties and "expired_at" not in f.properties


def test_forget_fact_unknown_is_empty_and_group_scoped(mem):
    old, new, _, _ = _owner_history(mem)
    assert mem.forget_fact("fact_nope").facts_deleted == []
    assert mem.forget_fact(new, group_id=H).facts_deleted == []
    assert new in facts(mem)


def test_forgetting_an_episode_restores_what_its_facts_superseded(mem):
    old, new, _, b = _owner_history(mem)
    res = mem.forget_episode(b.episode_id)
    assert res.facts_deleted == [new] and res.facts_restored == [old]
    assert mem.search("Rails OWNER")[0].id == old
    assert entity(mem, "bob") is None  # only existed there


def test_forgetting_an_episode_keeps_other_facts_of_surviving_entities(mem):
    a = ep(mem, "[[Rails|Service]] [[alice|Person]] {{alice|WORKS_ON|Rails}}", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] [[dave|Person]] {{dave|WORKS_ON|Rails}}", author="dave", at="2026-07-02 10:00")
    res = mem.forget_episode(b.episode_id)
    assert res.facts_deleted == b.facts_created and set(res.entities_deleted) == set(b.new_entities)  # dave
    assert entity(mem, "Rails") is not None and entity(mem, "alice") is not None
    assert list(facts(mem)) == a.facts_created
    assert entity(mem, "Rails")["id"] in res.entities_updated  # survived but was touched


# --------------------------------------------------------------------------- forget_entity


def test_forget_entity_removes_entity_facts_and_mentioning_episodes(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    v1 = ep(mem, "[[Vault|Project]] [[carol|Person]] {{carol|OWNS|Vault}}", author="bob", at="2026-07-02 10:00")
    v2 = ep(mem, "[[Vault|Project]] [[Rails|Service]] {{Vault|DEPENDS_ON|Rails}}", author="bob", at="2026-07-03 10:00")
    res = mem.forget_entity("Vault")
    assert set(res.episodes_deleted) == {v1.episode_id, v2.episode_id}
    assert set(res.facts_deleted) == set(v1.facts_created + v2.facts_created)
    assert entity(mem, "Vault") is None and entity(mem, "carol") is None and entity(mem, "bob") is None
    rails = entity(mem, "Rails")
    assert rails is not None and "Vault" not in repr(rails.properties)
    assert counts(mem) == (1, 2, 1)
    assert mem.get_entity("Vault") is None


def test_forget_entity_without_episodes_keeps_the_raw_notes(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    v = ep(mem, "[[Vault|Project]] [[carol|Person]] {{carol|OWNS|Vault}}", author="bob", at="2026-07-02 10:00")
    res = mem.forget_entity("vault", with_episodes=False)  # name matching is normalised
    assert res.episodes_deleted == [] and v.facts_created[0] in res.facts_deleted
    assert entity(mem, "Vault") is None
    assert mem.get_episodes(limit=10)[0]["id"] == v.episode_id
    # the surviving mentioned entities lost the summary that spoke about the forgotten fact
    carol = entity(mem, "carol")
    assert carol is not None and carol.get("summary") == ""
    assert "Vault" not in repr(carol.properties)


def test_forget_entity_by_id_unknown_and_ambiguous(mem):
    r = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}")
    rails_id = r.entities[1] if entity(mem, "alice").id != r.entities[1] else r.entities[0]
    assert mem.forget_entity("nobody-here").entities_deleted == []
    assert mem.forget_entity("Rai").entities_deleted == []  # no fuzzy matching when deleting
    assert counts(mem) == (1, 2, 1)
    # same name, different entity types: ambiguous
    ep(mem, "[[Mercury|Service]] {{alice|USES|Mercury}}", at="2026-07-02 10:00")
    ep(mem, "[[Mercury|Person]] {{Mercury|KNOWS|alice}}", at="2026-07-03 10:00")
    names = mem.db.query("MATCH (e:Entity {name:'Mercury'}) RETURN e.id AS id")
    assert len(names) == 2
    with pytest.raises(ValueError, match="ambiguous"):
        mem.forget_entity("Mercury")
    assert mem.forget_entity(names[0]["id"], dry_run=True).entities_deleted == [names[0]["id"]]
    assert mem.forget_entity(rails_id).entities_deleted[0] == rails_id


# --------------------------------------------------------------------------- entity maintenance


def test_orphan_entities_are_deleted_but_entities_with_facts_survive(mem):
    ep(mem, "[[Rails|Service]] [[alice|Person]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    # an episode with an entity but no fact still mentions it
    lone = ep(mem, "[[Lonely|Concept]] note", author="alice", at="2026-07-02 10:00")
    both = ep(mem, "[[Lonely|Concept]] [[Rails|Service]] {{alice|LIKES|Lonely}}", at="2026-07-03 10:00")
    res = mem.forget_episode(both.episode_id)
    assert entity(mem, "Lonely") is not None  # still mentioned by the 'lone' episode
    assert entity(mem, "Rails") is not None
    res = mem.forget_episode(lone.episode_id)
    assert entity(mem, "Lonely") is None and "Lonely" not in " ".join(
        r["e"]["search_text"] for r in mem.db.query("MATCH (e:Entity) RETURN e")
    )
    assert len(res.entities_deleted) == 1


def test_alias_pruning_keeps_only_forms_still_used(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}} known as Ruby on Rails", at="2026-07-01 10:00")
    second = ep(mem, "[[Rails|Service]] {{alice|LIKES|Rails}} ror", at="2026-07-02 10:00")
    rails = entity(mem, "Rails")
    mem.db.update_node(
        rails.id,
        {
            "aliases": ["ror", "Ruby on Rails"],
            "alias_keys": ["rails", "ror", "rubyonrails"],
            "alias_src": ["chat\tror", "chat\tRuby on Rails", "chat\tRails"],
            "search_text": "rails\nror\nruby on rails",
        },
    )
    assert mem.db.query("MATCH (e:Entity) WHERE 'ror' IN e.alias_keys RETURN e.name AS n") == [{"n": "Rails"}]
    res = mem.forget_episode(second.episode_id)
    assert rails["id"] in res.entities_updated
    after = entity(mem, "Rails")
    assert after["aliases"] == ["Ruby on Rails"]  # 'ror' only occurred in the forgotten message
    assert after["alias_keys"] == ["rails", "rubyonrails"] and after["search_text"] == "rails\nruby on rails"
    assert after["alias_src"] == ["chat\tRuby on Rails", "chat\tRails"]
    assert mem.db.query("MATCH (e:Entity) WHERE 'ror' IN e.alias_keys RETURN e") == []
    assert mem.get_entity("Ruby on Rails")["entity"]["name"] == "Rails"





def test_summary_rows_of_affected_channels_are_dropped(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", channel="work", at="2026-07-01 10:00")
    secret = ep(mem, "[[Rails|Service]] [[Vault|Project]] {{Vault|DEPENDS_ON|Rails}}", channel="diary", author="bob", at="2026-07-02 10:00")
    rails = entity(mem, "Rails")
    assert {e.partition("\t")[0] for e in rails["summaries"]} == {"work", "diary"}
    res = mem.forget_episode(secret.episode_id)
    assert rails["id"] in res.entities_updated
    after = entity(mem, "Rails")
    assert [e.partition("\t")[0] for e in after["summaries"]] == ["work"]
    assert after["summary_channels"] == ["work"] and "Vault" not in after["summary"]
    assert after["channels"] == ["work"]


def test_summary_falls_back_to_valid_facts_when_all_rows_dropped(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] {{alice|LIKES|Rails}}", at="2026-07-02 10:00")
    res = mem.forget_episode(b.episode_id)
    rails = entity(mem, "Rails")
    assert rails["summary"] == "" and rails["summaries"] == []
    view = mem.get_entity("Rails")["entity"]
    assert view["summary_source"] == "derived" and "alice USES Rails" in view["summary"]
    assert res.entities_updated


def test_embedding_is_recomputed_with_an_embedder(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] [[Vault|Project]] {{Vault|DEPENDS_ON|Rails}}", author="bob", at="2026-07-02 10:00")
    rails = entity(mem, "Rails")
    ix = next(i for i in mem.db.vector_indexes() if i.name == ENTITY_INDEX)
    before = mem.db._vec_read(ix, rails.id)
    mem.forget_episode(b.episode_id)
    after = mem.db._vec_read(ix, rails.id)
    assert after is not None and after != before
    # no summary line is left (one channel): the vector is built from the name and the valid facts
    assert after == pytest.approx(FakeEmbedder().embed(["Rails: alice USES Rails"])[0])


def test_embedding_is_removed_without_an_embedder(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as m:
        ep(m, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
        b = ep(m, "[[Rails|Service]] [[Vault|Project]] {{Vault|DEPENDS_ON|Rails}}", author="bob", at="2026-07-02 10:00")
    with Memory(path, None, None, default_group_id=G) as m:  # reads the dimension from the database
        rails = entity(m, "Rails")
        ix = next(i for i in m.db.vector_indexes() if i.name == ENTITY_INDEX)
        assert m.db._vec_read(ix, rails.id) is not None
        m.forget_episode(b.episode_id)
        assert m.db._vec_read(ix, rails.id) is None  # a stale vector could encode the forgotten text
        assert entity(m, "Rails") is not None


def test_forget_succeeds_when_the_embedder_fails(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    b = ep(mem, "[[Rails|Service]] [[Vault|Project]] {{Vault|DEPENDS_ON|Rails}}", author="bob", at="2026-07-02 10:00")
    rails = entity(mem, "Rails")

    def offline(texts):
        raise ConnectionError("embedding API unreachable")

    mem.embedder.embed = offline
    res = mem.forget_episode(b.episode_id)
    assert res.episodes_deleted == [b.episode_id]
    ix = next(i for i in mem.db.vector_indexes() if i.name == ENTITY_INDEX)
    assert mem.db._vec_read(ix, rails.id) is None  # falls back to removing the stale vector


# --------------------------------------------------------------------------- dry run


def test_dry_run_reports_the_same_result_and_writes_nothing(mem):
    old, new, _, b = _owner_history(mem)
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-09-01 10:00")
    before = snapshot(mem)
    for call in (
        lambda **k: mem.forget_episode(b.episode_id, **k),
        lambda **k: mem.forget_fact(new, **k),
        lambda **k: mem.forget_entity("bob", **k),
        lambda **k: mem.forget_range(since="2026-08-01", **k),
        lambda **k: mem.purge(**k),
    ):
        preview = call(dry_run=True)
        assert preview.dry_run and snapshot(mem) == before
        assert preview.episodes_deleted or preview.facts_deleted
    # identical to a real run (on a separate, identically built database)
    other_path = mem.db.path + ".copy"
    import shutil

    mem.db.checkpoint()
    shutil.copy(mem.db.path, other_path)
    with Memory(other_path, FakeLLM(), FakeEmbedder(), default_group_id=G) as twin:
        real = twin.forget_episode(b.episode_id)
    prev = mem.forget_episode(b.episode_id, dry_run=True)
    assert {k: v for k, v in real.to_dict().items() if k != "dry_run"} == {k: v for k, v in prev.to_dict().items() if k != "dry_run"}
    assert real.facts_restored == [old] and not real.dry_run
    assert snapshot(mem) == before


def test_dry_run_on_a_read_only_database(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as m:
        r = ep(m, "[[Rails|Service]] {{alice|USES|Rails}}")
    with Memory(path, None, None, default_group_id=G, read_only=True) as ro:
        assert ro.forget_episode(r.episode_id, dry_run=True).episodes_deleted == [r.episode_id]
        with pytest.raises(SqgraphError, match="read-only"):
            ro.forget_episode(r.episode_id)
        with pytest.raises(SqgraphError, match="read-only"):
            ro.purge()


# --------------------------------------------------------------------------- purge / range


def test_purge_empties_the_group_but_not_another(mem):
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    ep(mem, "[[Vault|Project]] [[carol|Person]] {{carol|OWNS|Vault}}", at="2026-07-02 10:00")
    other = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", group=H)
    res = mem.purge()
    assert len(res.episodes_deleted) == 2 and len(res.facts_deleted) == 2 and len(res.entities_deleted) == 4
    assert counts(mem) == (0, 0, 0)
    assert counts(mem, H) == (1, 2, 1)
    assert mem.search("Rails", group_id=H)[0].id == other.facts_created[0]
    assert mem.db._conn.execute("SELECT count(*) FROM vec_fact_emb").fetchone()[0] == 1
    assert mem.purge().empty  # idempotent
    # an empty group can be written again
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-03 10:00")
    assert counts(mem) == (1, 2, 1)


def test_forget_range_by_since_until_channel_and_source(mem):
    e1 = ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", channel="work", at="2026-07-01 10:00")
    e2 = ep(mem, "[[Vault|Project]] {{alice|OWNS|Vault}}", channel="work", at="2026-07-10 10:00")
    e3 = ep(mem, "[[Redis|Service]] {{alice|USES|Redis}}", channel="diary", at="2026-07-20 10:00")
    e4 = mem.add_episode("[[Kafka|Service]] {{alice|USES|Kafka}}", group_id=G, source="cli", channel="work", author="alice", occurred_at="2026-07-25 10:00")
    # since is inclusive, until exclusive; naive times are JST
    r = mem.forget_range(since="2026-07-10 10:00", until="2026-07-20 10:00", dry_run=True)
    assert r.episodes_deleted == [e2.episode_id]
    r = mem.forget_range(since="2026-07-10", until="2026-07-21")
    assert set(r.episodes_deleted) == {e2.episode_id, e3.episode_id}
    assert entity(mem, "Vault") is None and entity(mem, "Redis") is None and entity(mem, "Rails") is not None
    assert mem.forget_range(channel="diary").episodes_deleted == []  # already gone
    assert mem.forget_range(channel="work", until="2026-07-02").episodes_deleted == [e1.episode_id]
    assert mem.forget_range(source="cli").episodes_deleted == [e4.episode_id]
    assert counts(mem) == (0, 0, 0)


def test_forget_range_validation_and_group(mem):
    with pytest.raises(ValueError):
        mem.forget_range()
    with pytest.raises(ValueError, match="cannot parse since"):
        mem.forget_range(since="not a date")
    ep(mem, "[[Rails|Service]] {{alice|USES|Rails}}", group=H, at="2026-07-01 10:00")
    assert mem.forget_range(since="2026-01-01").episodes_deleted == []
    assert mem.forget_range(since="2026-01-01", group_id=H).episodes_deleted != []


# --------------------------------------------------------------------------- physical deletion


MARKER = "ZQXV-secret-7f3a91"


def _raw_bytes(path):
    out = {}
    for suffix in ("", "-wal"):
        try:
            with open(str(path) + suffix, "rb") as fh:
                out[suffix or "db"] = fh.read()
        except FileNotFoundError:
            out[suffix or "db"] = b""
    return out


def test_forgotten_text_is_physically_gone_from_the_database_file(tmp_path):
    path = tmp_path / "mem.db"
    m = Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G)
    ep(m, "[[Rails|Service]] {{alice|USES|Rails}} ordinary note", at="2026-07-01 10:00")
    secret = ep(
        m,
        f"[[Rails|Service]] [[{MARKER}|Project]] {{{{{MARKER}|DEPENDS_ON|Rails}}}} {MARKER} diary",
        channel="diary",
        author="bob",
        at="2026-07-02 10:00",
    )
    m.db.checkpoint()
    raw = _raw_bytes(path)
    assert MARKER.encode() in raw["db"] or MARKER.encode() in raw["-wal"]  # it was really written
    res = m.forget_episode(secret.episode_id)
    assert res.episodes_deleted == [secret.episode_id]
    m.close()
    for name, data in _raw_bytes(path).items():
        assert MARKER.encode() not in data, f"forgotten text found in the {name} file"
        assert MARKER.lower().encode() not in data, f"forgotten text (case-folded) found in the {name} file"
    # the rest of the database is intact
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as m2:
        assert counts(m2) == (1, 2, 1)
        assert m2.search("alice USES Rails")


def test_physical_deletion_also_covers_entity_summaries_and_purge(tmp_path):
    path = tmp_path / "mem.db"
    m = Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G)
    ep(m, "[[Rails|Service]] {{alice|USES|Rails}}", at="2026-07-01 10:00")
    secret = ep(m, f"[[Rails|Service]] [[Vault|Project]] {{{{Vault|DEPENDS_ON|Rails}}}} {MARKER}", author="bob", at="2026-07-02 10:00")
    assert "Vault" in entity(m, "Rails")["summary"]  # the summary of the surviving entity knew about it
    m.forget_episode(secret.episode_id)
    m.close()
    raw = b"".join(_raw_bytes(path).values())
    assert MARKER.encode() not in raw and b"DEPENDS_ON" not in raw and b"Vault" not in raw
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G) as m2:
        m2.purge()
    raw = b"".join(_raw_bytes(path).values())
    assert b"USES" not in raw and b"alice" not in raw


def test_secure_delete_pragma_is_on_for_memory_databases(tmp_path):
    with Memory(tmp_path / "a.db", FakeLLM(), FakeEmbedder(), default_group_id=G) as m:
        assert m.db._conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    shared = GraphDB(tmp_path / "b.db")
    assert shared._conn.execute("PRAGMA secure_delete").fetchone()[0] == 0
    with Memory(shared, FakeLLM(), FakeEmbedder(), default_group_id=G):
        assert shared._conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    shared.close()
