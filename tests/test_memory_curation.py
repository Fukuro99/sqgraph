"""Curation of the memory: relation vocabulary, merge_entities, invalidate_fact, rename_relation, review, upcoming."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sqgraph.memory import FakeEmbedder, Memory
from sqgraph.memory.core import canonical_relation, normalize_relation

G = "me"
AT = "2026-07-01 10:00"
NOW = datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)  # 2026-07-10 12:00 JST


def make(**kw):
    return Memory(":memory:", None, FakeEmbedder(), default_group_id=G, clock=lambda: NOW, **kw)


def add(mem, text, extraction, *, at=AT, channel="personal", author="user"):
    prep = mem.prepare_episode(text, group_id=G, source="test", channel=channel, author=author, occurred_at=at)
    return prep, mem.commit_episode(prep["token"], extraction)


def ent(ref, name, typ="Service", **kw):
    return {"ref": ref, "name": name, "type": typ, **kw}


def fact(s, o, rel, text, **kw):
    return {"subject": s, "object": o, "relation": rel, "fact": text, **kw}


def facts(mem):
    return {r["f"]["id"]: r["f"] for r in mem.db.query("MATCH ()-[f:FACT]->() RETURN f")}


def entities(mem):
    return {r["e"]["name"]: r["e"] for r in mem.db.query("MATCH (e:Entity) RETURN e")}


# --------------------------------------------------------------------------- relation vocabulary


def test_normalize_and_canonical_relation():
    assert normalize_relation(" owned-by ") == "OWNED_BY"
    assert normalize_relation("works on") == "WORKS_ON"
    assert normalize_relation(None) == "RELATES_TO"
    assert normalize_relation("担当する") == "担当する"
    assert canonical_relation("owned by", True) == ("OWNS", True)
    assert canonical_relation("OWNED_BY", False) == ("OWNED_BY", False)  # an attribute has nothing to swap with
    assert canonical_relation("USES", True) == ("USES", False)


def test_passive_relation_is_stored_in_active_direction():
    mem = make()
    _, res = add(
        mem,
        "sqgraph のリポジトリは Bob が管理している",
        {"entities": [ent("r", "sqgraph"), ent("b", "Bob", "Person")], "facts": [fact("r", "b", "owned_by", "sqgraph は Bob が所有している")]},
    )
    (f,) = facts(mem).values()
    names = {e["id"]: n for n, e in entities(mem).items()}
    assert f["relation"] == "OWNS"
    assert names[f["subject_id"]] == "Bob" and names[f["object_id"]] == "sqgraph"


def test_known_relations_are_offered_to_the_extractor():
    mem = make()
    add(mem, "pnpm を使う", {"entities": [ent("p", "pnpm")], "facts": [fact("author", "p", "PREFERS", "ユーザーは pnpm を好む")]})
    add(mem, "uv を使う", {"entities": [ent("u", "uv")], "facts": [fact("author", "u", "PREFERS", "ユーザーは uv を好む")]})
    add(mem, "Rails を使う", {"entities": [ent("r", "Rails")], "facts": [fact("author", "r", "USES", "ユーザーは Rails を使う")]})
    add(mem, "秘密", {"entities": [ent("x", "X")], "facts": [fact("author", "x", "HIDES", "ユーザーは X を隠す")]}, channel="other")
    prep = mem.prepare_episode("次は何を使う?", group_id=G, channel="personal", author="user", occurred_at=AT)
    assert prep["context"]["known_relations"] == ["PREFERS", "USES"]  # most used first, only this channel's facts
    assert "known_relations" in prep["instructions"]


# --------------------------------------------------------------------------- merge_entities


def _two_rails(mem):
    add(mem, "Rails を使う", {"entities": [ent("r", "Rails")], "facts": [fact("author", "r", "USES", "ユーザーは Rails を使う")]})
    add(
        mem,
        "RoR を使う。RoR は Ruby 製",
        {
            "entities": [ent("r", "Ruby on Rails", aliases=["RoR"]), ent("rb", "Ruby", "Concept")],
            "facts": [
                fact("author", "r", "USES", "ユーザーは Rails を使う"),
                fact("r", "rb", "WRITTEN_IN", "Ruby on Rails は Ruby で書かれている"),
            ],
        },
        at="2026-07-02 10:00",
    )


def test_merge_entities_moves_facts_folds_duplicates_and_keeps_aliases():
    mem = make()
    _two_rails(mem)
    before = entities(mem)
    assert {"Rails", "Ruby on Rails"} <= set(before)
    plan = mem.merge_entities("Rails", "Ruby on Rails", dry_run=True)
    assert plan["dry_run"] and len(facts(mem)) == 3 and "Ruby on Rails" in entities(mem)  # nothing written
    res = mem.merge_entities("Rails", "RoR")  # an alias resolves too
    assert res["kept"]["name"] == "Rails" and res["merged"]["name"] == "Ruby on Rails"
    assert len(res["facts_moved"]) == 1 and len(res["facts_folded"]) == 1 and res["mentions_moved"] == 1
    assert set(res["aliases_added"]) == {"Ruby on Rails", "RoR"}
    after = entities(mem)
    assert "Ruby on Rails" not in after
    rails = after["Rails"]
    assert {"Ruby on Rails", "RoR"} <= set(rails["aliases"])
    fs = facts(mem)
    assert len(fs) == 2
    uses = next(f for f in fs.values() if f["relation"] == "USES")
    assert len(uses["episode_ids"]) == 2  # the duplicate's source was folded in
    written = next(f for f in fs.values() if f["relation"] == "WRITTEN_IN")
    assert written["subject_id"] == rails["id"]
    eps = mem.db.query("MATCH (e:Episode)-[:MENTIONS]->(x:Entity {name:'Rails'}) RETURN count(e) AS n")
    assert eps[0]["n"] == 2
    assert mem.get_entity("RoR")["entity"]["id"] == rails["id"]
    hits = mem.search("Ruby で書かれている")
    assert hits and hits[0].subject["name"] == "Rails"  # the moved fact kept its vector


def test_merge_entities_fact_between_the_two_becomes_attribute_and_errors():
    mem = make()
    add(
        mem,
        "k8s は Kubernetes の略",
        {"entities": [ent("a", "k8s"), ent("b", "Kubernetes")], "facts": [fact("a", "b", "ABBREVIATES", "k8s は Kubernetes の略称")]},
    )
    res = mem.merge_entities("Kubernetes", "k8s")
    assert res["warnings"] and "attribute" in res["warnings"][0]
    (f,) = facts(mem).values()
    assert f.get("object_id") is None and f["subject_id"] == entities(mem)["Kubernetes"]["id"]
    with pytest.raises(ValueError, match="no entity"):
        mem.merge_entities("Kubernetes", "nothing")
    with pytest.raises(ValueError, match="same entity"):
        mem.merge_entities("Kubernetes", "k8s")  # k8s is now an alias of Kubernetes


def test_merge_redirects_supersede_links_of_folded_facts():
    mem = make()
    add(mem, "A は v1", {"entities": [ent("a", "A")], "facts": [fact("a", None, "HAS_STATUS", "A は v1")]})
    old = next(iter(facts(mem)))
    _, r2 = add(
        mem, "A2 は v2",
        {"entities": [ent("a", "A2")], "facts": [fact("a", None, "HAS_STATUS", "A は v2")]},
        at="2026-07-02 10:00",
    )
    _, r3 = add(
        mem, "A は v2 (再掲)",
        {"entities": [ent("a", "A", existing_id=entities(mem)["A"]["id"])], "facts": [fact("a", None, "HAS_STATUS", "A は v2", contradicts=[old])]},
        at="2026-07-03 10:00",
    )
    survivor = r3.facts_created[0]
    mem.merge_entities("A", "A2")
    fs = facts(mem)
    assert set(fs) == {old, survivor}  # A2's "A は v2" folded into A's
    assert fs[old]["superseded_by"] == survivor


# --------------------------------------------------------------------------- invalidate_fact


def test_invalidate_fact_closes_and_keeps_history():
    mem = make()
    add(mem, "pnpm", {"entities": [ent("p", "pnpm")], "facts": [fact("author", "p", "PREFERS", "ユーザーは pnpm を好む")]})
    fid = next(iter(facts(mem)))
    hit = mem.invalidate_fact(fid, reason="もう使っていない")
    assert hit.status == "ended" and hit.invalid_at.startswith("2026-07-10T12:00")
    f = facts(mem)[fid]
    assert f["ended_by"] == "manual" and f["invalid_reason"] == "もう使っていない"
    assert mem.search("pnpm") == [] and mem.search("pnpm", include_invalid=True)
    with pytest.raises(ValueError, match="already ended"):
        mem.invalidate_fact(fid)
    with pytest.raises(ValueError, match="no fact"):
        mem.invalidate_fact("fact_000000000000")


def test_invalidate_a_plan_before_it_starts_cancels_it():
    mem = make()
    add(
        mem, "来月から大阪",
        {"entities": [ent("o", "大阪", "Place")], "facts": [fact("author", "o", "PLANS_TO", "ユーザーは 8/1 から大阪に住む", valid_at="2026-08-01", modality="planned")]},
    )
    fid = next(iter(facts(mem)))
    with pytest.raises(ValueError, match="before the fact became valid"):
        mem.invalidate_fact(fid, at="2026-07-20")
    hit = mem.invalidate_fact(fid)
    assert hit.status == "ended" and hit.invalid_at == hit.valid_at
    assert mem.upcoming(60) == []  # a cancelled plan is not upcoming


# --------------------------------------------------------------------------- rename_relation / review


def test_rename_relation_with_swap_and_subset():
    mem = make()
    add(
        mem, "PR と Issue",
        {
            "entities": [ent("b", "Bob", "Person"), ent("p", "PR #1", "Ticket"), ent("i", "Issue #2", "Ticket")],
            "facts": [fact("p", "b", "REVIEWER_IS", "PR #1 のレビュアーは Bob"), fact("i", "b", "REVIEWER_IS", "Issue #2 のレビュアーは Bob")],
        },
    )
    ids = sorted(facts(mem))
    dry = mem.rename_relation("reviewer is", "REVIEWED", swap=True, dry_run=True)
    assert dry["renamed"] == dry["swapped"] and len(dry["renamed"]) == 2 and {f["relation"] for f in facts(mem).values()} == {"REVIEWER_IS"}
    res = mem.rename_relation("REVIEWER_IS", "REVIEWED", swap=True, fact_ids=[ids[0]])
    assert res["renamed"] == [ids[0]] and res["swapped"] == [ids[0]]
    fs = facts(mem)
    bob = entities(mem)["Bob"]["id"]
    assert fs[ids[0]]["relation"] == "REVIEWED" and fs[ids[0]]["subject_id"] == bob
    assert fs[ids[1]]["relation"] == "REVIEWER_IS"
    res = mem.rename_relation("REVIEWER_IS", "reviewed_by")  # passive: stored as the active form, turned around
    assert res["to"] == "REVIEWED" and res["swapped"] == [ids[1]]
    assert facts(mem)[ids[1]]["subject_id"] == bob
    with pytest.raises(ValueError, match="not facts"):
        mem.rename_relation("REVIEWED", "X", fact_ids=["fact_000000000000"])


def test_review_lists_duplicates_relation_suggestions_and_stale_plans():
    mem = make()
    _two_rails(mem)
    mem.db.query("MATCH ()-[f:FACT {relation:'WRITTEN_IN'}]->() SET f.relation = 'OWNED_BY'")  # legacy passive name
    add(mem, "uses", {"entities": [ent("x", "X")], "facts": [fact("author", "x", "Uses", "ユーザーは X を使う")]})
    mem.db.query("MATCH ()-[f:FACT {fact:'ユーザーは X を使う'}]->() SET f.relation = 'USE'")
    add(
        mem, "6/1 に引っ越す予定",
        {"entities": [ent("k", "京都", "Place")], "facts": [fact("author", "k", "PLANS_TO", "ユーザーは 6/1 に京都へ引っ越す", valid_at="2026-06-01", modality="planned")]},
    )
    rep = mem.review()
    pair = rep["duplicate_entities"][0]
    assert {pair["a"]["name"], pair["b"]["name"]} == {"Rails", "Ruby on Rails"}
    assert "one name contains the other" in pair["reasons"]
    sugg = {(s["relation"], s["suggest"], s["swap"]) for s in rep["relation_suggestions"]}
    assert ("OWNED_BY", "OWNS", True) in sugg and ("USE", "USES", False) in sugg
    assert "OWNED_BY" in rep["relations_used_once"]
    assert [p["fact"] for p in rep["stale_plans"]] == ["ユーザーは 6/1 に京都へ引っ越す"]
    assert "merge_entities" in rep["next"]


# --------------------------------------------------------------------------- upcoming


def test_upcoming_lists_starts_and_stated_ends_soonest_first():
    mem = make()
    add(
        mem, "7/14-7/16 休み",
        {"entities": [], "facts": [fact("author", None, "ABSENT", "ユーザーは 7/14-7/16 休み", valid_at="2026-07-14", invalid_at="2026-07-17")]},
    )
    add(
        mem, "8/1 から大阪",
        {"entities": [ent("o", "大阪", "Place")], "facts": [fact("author", "o", "PLANS_TO", "ユーザーは 8/1 から大阪", valid_at="2026-08-01", modality="planned")]},
    )
    add(mem, "昔の予定", {"entities": [], "facts": [fact("author", None, "ABSENT", "ユーザーは 7/1 休み", valid_at="2026-07-01", invalid_at="2026-07-02")]})
    items = mem.upcoming(7)
    assert [(i["event"], i["at"][:10]) for i in items] == [("starts", "2026-07-14"), ("ends", "2026-07-17")]
    assert items[0]["fact"]["status"] == "scheduled"
    assert [i["event"] for i in mem.upcoming(30)][-1] == "starts" and len(mem.upcoming(30)) == 3
    assert [i["at"][:10] for i in mem.upcoming(1, past_days=10)] == ["2026-07-01", "2026-07-02"]
    assert mem.upcoming(7, channels=["other"]) == []
