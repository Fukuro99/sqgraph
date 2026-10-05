"""Relevance gate of Memory.search / answer / prepare_episode, its configuration plumbing (config / CLI / MCP) and the
forget_memory argument aliases. A deterministic embedder with hand-set vectors gives exact control over the similarities."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, Memory, cli
from sqgraph.memory.config import MemoryConfig
from sqgraph.memory import mcp as mcp_mod
from sqgraph.memory.mcp import MemoryTools, build_server
from sqgraph.memory.core import (
    MIN_SIMILARITY_NO_MODEL,
    MIN_SIMILARITY_UNKNOWN_MODEL,
    NO_HITS_HINT,
    PREPARE_MIN_RELATED,
    default_min_similarity,
)

G = "me"
AT = "2026-07-01 10:00"
needs_mcp = pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the mcp package")


# --------------------------------------------------------------------------- helpers


class VecEmbedder:
    """Hand-set vectors: the first table key contained in a text decides its vector (the first axes, ``cos`` between
    two texts is then known exactly). Any other text gets an axis of its own, i.e. similarity 0 to everything else."""

    def __init__(self, table: dict[str, tuple[float, ...]] | None = None, dims: int = 64, model: str | None = None):
        self.table = list((table or {}).items())
        self.dimensions = dims
        self._fresh: dict[str, int] = {}
        self.fail = False
        if model:
            self.model = model

    def embed(self, texts):
        if self.fail:
            raise ConnectionError("embedding endpoint is down")
        out = []
        for t in texts:
            vec = [0.0] * self.dimensions
            for key, v in self.table:
                if key in t:
                    vec[: len(v)] = list(v)
                    break
            else:
                axis = self._fresh.setdefault(t, 8 + len(self._fresh))
                vec[axis] = 1.0
            n = sum(x * x for x in vec) ** 0.5
            out.append([x / n for x in vec])
        return out


def at_cos(c: float) -> tuple[float, float]:
    """A unit vector with cosine ``c`` to the first axis ``(1, 0)``."""
    return (c, (1 - c * c) ** 0.5)


def remember(mem: Memory, text: str, *, ents=(), subject="author", obj=None, relation="LIKES", **kw):
    """Store one fact the way an agent would (prepare + commit). ``ents``: ``[(ref, name, type)]``."""
    prep = mem.prepare_episode(text, group_id=G, source="t", channel="personal", author="user", occurred_at=AT)
    cands = {c["name"]: c["id"] for c in prep["context"]["candidate_entities"]}
    entities = [{"ref": r, "name": n, "type": t, "existing_id": cands.get(n)} for r, n, t in ents]
    fact = {"subject": subject, "object": obj, "relation": relation, "fact": text, **kw}
    return mem.commit_episode(prep["token"], {"entities": entities, "facts": [fact]})


def make(table=None, **kw) -> Memory:
    kw.setdefault("min_similarity", 0.6)
    return Memory(":memory:", FakeLLM(), VecEmbedder(table), default_group_id=G, **kw)


def texts(hits) -> list[str]:
    return [h.fact for h in hits]


def personal_memory(**kw) -> Memory:
    """Four facts of the user, all of the form "ユーザーは ..." (every fact has the user as subject)."""
    mem = make({"猫": (1.0, 0.0), "犬": (0.0, 1.0)}, **kw)
    for t in ("ユーザーは猫を飼っている", "ユーザーは犬が苦手", "ユーザーは朝型の生活をしている", "ユーザーはコーヒーが好き"):
        remember(mem, t)
    return mem


# --------------------------------------------------------------------------- the gate


def test_unrelated_query_on_a_small_memory_returns_nothing():
    mem = personal_memory()
    assert mem.search("ユーザーの年収はいくら？") == []  # the old RRF-only search returned all four facts
    assert mem.search("明日の天気は？") == []
    assert mem.search("Kubernetes のクラスタ構成") == []
    mem.close()


def test_related_query_returns_only_the_related_fact_with_its_similarity():
    mem = personal_memory()
    hits = mem.search("猫について")
    assert texts(hits) == ["ユーザーは猫を飼っている"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)
    d = hits[0].to_dict()
    assert d["similarity"] == 1.0 and "score" in d and d["score"] > 0
    mem.close()


def test_persona_words_are_no_fulltext_evidence():
    mem = personal_memory()
    # "ユーザー" occurs in every fact: a fulltext hit on it must not count
    assert mem.search("ユーザー") == []
    assert mem.search("ユーザーが好きな映画") == []
    mem.close()


def test_fulltext_hit_is_always_kept_even_without_semantic_similarity():
    mem = personal_memory()
    hits = mem.search("コーヒー")  # a rare word of exactly one fact; its vector is unrelated to the query's
    assert texts(hits) == ["ユーザーはコーヒーが好き"]
    assert hits[0].similarity is not None and hits[0].similarity < 0.6
    mem.close()


def test_a_term_in_half_of_all_facts_is_not_fulltext_evidence():
    mem = make()
    for t in ("A のサーバーはリージョン us-east-1 にある", "B のサーバーはリージョン ap-northeast-1 にある", "C のサーバーはリージョン eu-west-1 にある", "ユーザーは猫が好き"):
        remember(mem, t)
    # "リージョン" is in 3 of 4 facts (>= half of them): no evidence; the region name is
    assert mem.search("リージョン") == []
    assert texts(mem.search("リージョン ap-northeast-1")) == ["B のサーバーはリージョン ap-northeast-1 にある"]
    mem.close()


def test_strong_entity_match_keeps_its_hop1_facts_regardless_of_similarity():
    mem = make()
    remember(mem, "田中は先月入社した", ents=[("t", "田中", "Person")], subject="t")
    remember(mem, "田中の席は3階にある", ents=[("t", "田中", "Person")], subject="t")
    remember(mem, "ユーザーは猫が好き")
    hits = mem.search("田中さんについて教えて")
    assert sorted(texts(hits)) == ["田中の席は3階にある", "田中は先月入社した"]
    assert all(h.similarity is not None and h.similarity < 0.6 for h in hits)
    mem.close()


def test_literal_mention_of_the_user_hub_is_only_a_weak_match():
    mem = make()
    for i in range(6):
        remember(mem, f"ユーザーの好みその{i}は紅茶")
    assert mem.search("user の好きな食べ物") == []  # the author entity "user" is named, but it is the hub of the memory
    assert mem.search("ユーザーの好きな食べ物") == []
    # center_entity is an explicit request: its facts are kept
    assert len(mem.search("好きな食べ物", center_entity="user", limit=10)) == 6
    mem.close()


def test_weak_entity_expansion_needs_similarity_above_threshold_minus_slack():
    table = {"sqgraph": (1.0, 0.0), "説明A": at_cos(0.55), "説明B": at_cos(0.40)}
    mem = make(table, relative_margin=2.0, expand_slack=0.1)  # thr 0.6: slack bar 0.5
    ents = [("e", "sqgraph-memory", "Project")]
    remember(mem, "ツールの説明A", ents=ents, subject="e")
    remember(mem, "ツールの説明B", ents=ents, subject="e")
    remember(mem, "ツールの説明C", ents=ents, subject="e")
    # "sqgraph" is only a fragment of the name "sqgraph-memory": a weak match; the facts do not contain the word
    assert texts(mem.search("sqgraph")) == ["ツールの説明A"]  # A: 0.55 >= 0.5; B: 0.40 and C: 0.0 are below
    mem.close()


def test_two_hops_from_a_strong_match_need_similarity_but_weak_two_hops_never_expand():
    def build(c):
        mem = make({"alice": (1.0, 0.0), "Postgres": at_cos(c)}, relative_margin=2.0, expand_slack=0.1)
        remember(mem, "alice は Rails を担当している", ents=[("a", "alice", "Person"), ("r", "Rails", "Service")], subject="a", obj="r")
        remember(mem, "Rails は Postgres に依存している", ents=[("r", "Rails", "Service"), ("p", "Postgres", "Service")], subject="r", obj="p")
        return mem

    near, far = build(0.55), build(0.30)
    assert texts(near.search("alice")) == ["alice は Rails を担当している", "Rails は Postgres に依存している"]
    assert texts(far.search("alice")) == ["alice は Rails を担当している"]
    near.close()
    far.close()


def test_relative_margin_drops_vector_only_candidates_far_below_the_best():
    table = {"検索語": (1.0, 0.0), "九割": at_cos(0.9), "七割": at_cos(0.7)}

    def run(**kw):
        mem = make(table, min_similarity=0.5, **kw)
        remember(mem, "事実の九割")
        remember(mem, "事実の七割")
        out = texts(mem.search("検索語"))
        mem.close()
        return out

    assert run(relative_margin=0.1) == ["事実の九割"]
    assert sorted(run(relative_margin=0.3)) == ["事実の七割", "事実の九割"]


def test_relative_margin_never_drops_fulltext_hits():
    table = {"検索語": (1.0, 0.0), "近い": at_cos(0.95)}
    mem = make(table, min_similarity=0.5, relative_margin=0.05)
    remember(mem, "近い事実")
    remember(mem, "遠い事実マルゲリータ")  # fresh vector, but a keyword hit
    out = texts(mem.search("検索語 マルゲリータ"))
    assert sorted(out) == ["近い事実", "遠い事実マルゲリータ"]  # the keyword hit is 0.95 below the best, and stays
    mem.close()


def test_per_call_override_and_constructor_threshold():
    table = {"検索語": (1.0, 0.0), "九割": at_cos(0.9), "七割": at_cos(0.7)}
    mem = make(table, min_similarity=0.6, relative_margin=2.0)
    remember(mem, "事実の九割")
    remember(mem, "事実の七割")
    assert len(mem.search("検索語")) == 2
    assert texts(mem.search("検索語", min_similarity=0.8)) == ["事実の九割"]
    assert mem.search("検索語", min_similarity=0.95) == []
    assert len(mem.search("検索語", min_similarity=-1.0)) == 2
    assert mem.effective_min_similarity() == 0.6 and mem.effective_min_similarity(0.8) == 0.8
    with pytest.raises(ValueError, match="min_similarity"):
        mem.search("検索語", min_similarity=1.5)
    with pytest.raises(ValueError, match="min_similarity"):
        Memory(":memory:", None, None, min_similarity="high")  # type: ignore[arg-type]
    mem.close()


def test_without_an_embedder_only_keyword_and_named_entity_evidence_remains(tmp_path):
    path = tmp_path / "m.db"
    emb = VecEmbedder({"猫": (1.0, 0.0)})
    with Memory(path, None, emb, default_group_id=G) as mem:
        for t in ("ユーザーは猫を飼っている", "ユーザーは朝型の生活をしている", "ユーザーはコーヒーが好き"):
            remember(mem, t)
        remember(mem, "田中は先月入社した", ents=[("t", "田中", "Person")], subject="t")
        remember(mem, "田中の席は3階にある", ents=[("t", "田中", "Person")], subject="t")
        remember(mem, "ツールの説明", ents=[("e", "sqgraph-memory", "Project")], subject="e")
    with Memory(path, None, None, default_group_id=G) as mem:  # no embedder: no vectors at all
        assert mem.search("猫について") == []  # no keyword, no entity: nothing (not everything)
        assert mem.search("明日の天気は") == []
        hits = mem.search("コーヒー")
        assert texts(hits) == ["ユーザーはコーヒーが好き"] and hits[0].similarity is None
        assert hits[0].to_dict()["similarity"] is None
        assert sorted(texts(mem.search("田中さんは？"))) == ["田中の席は3階にある", "田中は先月入社した"]  # strong entity
        assert texts(mem.search("sqgraph")) == ["ツールの説明"]  # weak (name fragment): no vector evidence, hop 1 kept


def test_an_embedder_that_fails_at_search_time_degrades_instead_of_raising(tmp_path):
    emb = VecEmbedder({"猫": (1.0, 0.0)})
    mem = Memory(tmp_path / "m.db", None, emb, default_group_id=G, min_similarity=0.6)
    remember(mem, "ユーザーは猫を飼っている")
    remember(mem, "ユーザーはコーヒーが好き")
    emb.fail = True
    assert texts(mem.search("コーヒー")) == ["ユーザーはコーヒーが好き"]  # keyword evidence still works
    assert mem.search("猫について") == []  # semantic evidence is unavailable: nothing, not everything
    mem.close()


def test_answer_uses_the_gate():
    mem = personal_memory()
    llm = mem.llm
    ans = mem.answer("ユーザーの年収はいくら？", group_id=G)
    assert ans.hits == [] and "見つかりません" in ans.text and llm.count("answer") == 0
    ans = mem.answer("猫について", group_id=G)
    assert [h.fact for h in ans.hits] == ["ユーザーは猫を飼っている"] and llm.count("answer") == 1
    assert mem.answer("猫について", group_id=G, min_similarity=0.99).hits  # explicit per-call threshold
    mem.close()


# --------------------------------------------------------------------------- default thresholds


def test_default_threshold_by_model_name():
    assert default_min_similarity("text-embedding-3-small") == 0.34
    assert default_min_similarity("text-embedding-3-large") == 0.32
    assert default_min_similarity("text-embedding-nomic-embed-text-v1.5") == 0.72  # LM Studio's id
    assert default_min_similarity("nomic-ai/nomic-embed-text-v1.5") == 0.72
    assert default_min_similarity("nomic-embed-text") == 0.72  # Ollama
    assert default_min_similarity("text-embedding-qwen3-embedding-0.6b") == 0.46  # LM Studio's id
    assert default_min_similarity("Qwen/Qwen3-Embedding-0.6B") == 0.46
    assert default_min_similarity("bge-m3") == 0.45
    assert default_min_similarity("some-future-model") == MIN_SIMILARITY_UNKNOWN_MODEL
    assert default_min_similarity(None) == MIN_SIMILARITY_NO_MODEL


def test_effective_threshold_follows_the_stored_embedding_identity(tmp_path):
    path = tmp_path / "m.db"
    emb = VecEmbedder(model="text-embedding-nomic-embed-text-v1.5")
    emb.endpoint = "http://lan:1234/v1"
    with Memory(path, None, emb, default_group_id=G) as mem:
        assert mem.effective_min_similarity() == 0.72
        assert mem.effective_min_similarity(0.4) == 0.4
    small = VecEmbedder(model="text-embedding-nomic-embed-text-v1.5")  # same model, no endpoint: still the DB's identity
    with Memory(path, None, small, default_group_id=G) as mem:
        assert mem.effective_min_similarity() == 0.72
    with Memory(path, None, None, default_group_id=G) as mem:
        assert mem.effective_min_similarity() == MIN_SIMILARITY_UNKNOWN_MODEL  # no embedder: no vectors anyway
    with Memory(":memory:", None, FakeEmbedder()) as mem:  # an embedder without a model name
        assert mem.effective_min_similarity() == MIN_SIMILARITY_NO_MODEL
    with Memory(":memory:", None, FakeEmbedder(), min_similarity=0.5) as mem:
        assert mem.effective_min_similarity() == 0.5


def test_fake_embedder_default_keeps_the_old_intent():
    mem = Memory(":memory:", FakeLLM(), FakeEmbedder(), default_group_id=G)
    remember(mem, "ユーザーは pnpm を使っている")
    remember(mem, "ユーザーは朝 6 時に起きる")
    remember(mem, "ユーザーは京都へ旅行する予定")
    assert texts(mem.search("pnpm を使っている")) == ["ユーザーは pnpm を使っている"]
    assert mem.search("Kubernetes のクラスタ構成は？") == []
    mem.close()


# --------------------------------------------------------------------------- prepare_episode


def test_prepare_related_facts_keep_named_entity_facts_and_drop_unrelated_ones():
    table = {"pnpm": at_cos(0.9)}
    mem = make(table, min_similarity=0.6)
    remember(mem, "田中は先月入社した", ents=[("t", "田中", "Person")], subject="t")
    remember(mem, "ユーザーは pnpm を使っている")
    for i in range(12):
        remember(mem, f"ユーザーのメモその{i}は無関係な話題")
    prep = mem.prepare_episode("ユーザーは pnpm をやめて bun にする", group_id=G, channel="personal", author="user", occurred_at=AT)
    rel = [f["fact"] for f in prep["context"]["related_facts"]]
    assert "ユーザーは pnpm を使っている" in rel  # the contradiction candidate
    assert len(rel) <= 2 * PREPARE_MIN_RELATED - 1 < 13  # not the 12 unrelated facts of the user (the best few are always offered)
    prep2 = mem.prepare_episode("田中さんは来月異動する", group_id=G, channel="personal", author="user", occurred_at=AT)
    assert "田中は先月入社した" in [f["fact"] for f in prep2["context"]["related_facts"]]
    assert "田中" in [c["name"] for c in prep2["context"]["candidate_entities"]]
    mem.close()


# --------------------------------------------------------------------------- MCP / CLI plumbing

@needs_mcp
def test_mcp_search_memory_passes_min_similarity_and_reports_the_gate():
    mem = personal_memory()
    t = MemoryTools(mem, group_id=G)
    res = t.search_memory("猫について")
    assert res["min_similarity"] == 0.6 and res["count"] == 1 and res["facts"][0]["similarity"] == 1.0 and "hint" not in res
    empty = t.search_memory("ユーザーの年収はいくら？")
    assert empty["count"] == 0 and empty["hint"] == NO_HITS_HINT and "min_similarity" in empty["hint"]
    loose = t.search_memory("ユーザーの年収はいくら？", min_similarity=-1.0)
    assert loose["min_similarity"] == -1.0 and loose["count"] == 4
    server = build_server(t)

    async def go():
        tools = {x.name: x for x in await server.list_tools()}
        schema = getattr(tools["search_memory"], "inputSchema", None) or tools["search_memory"].input_schema
        assert "min_similarity" in schema["properties"]
        assert "min_similarity" in tools["search_memory"].description
        r = await server.call_tool("search_memory", {"query": "猫について", "min_similarity": 1.0})
        return r

    r = asyncio.run(go())
    content = r.content if hasattr(r, "content") else r[0]
    assert json.loads(content[0].text)["min_similarity"] == 1.0
    mem.close()


@needs_mcp
def test_mcp_main_forwards_min_similarity(tmp_path, monkeypatch):
    seen: dict = {}

    class Spy(Memory):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(mcp_mod, "Memory", Spy)
    monkeypatch.setattr(mcp_mod, "run_server", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mcp_mod.main(["--db", str(tmp_path / "m.db"), "--group-id", G, "--min-similarity", "0.41", "--env-file", str(tmp_path / "none.env")])
    assert seen["min_similarity"] == 0.41
    seen.clear()
    mcp_mod.main(["--db", str(tmp_path / "m.db"), "--group-id", G, "--env-file", str(tmp_path / "none.env")])
    assert seen["min_similarity"] is None


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mem-home"
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(h))
    for k in ("OPENAI_API_KEY", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP", "SQGRAPH_MEMORY_EXTRACTOR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "openai")

    def providers(cfg, *, need_llm, need_embedder):
        return (FakeLLM() if need_llm else None), (FakeEmbedder() if (need_llm or need_embedder) else None), []

    monkeypatch.setattr(cli, "make_providers", providers)
    return h


def run(capsys, *argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    capsys.readouterr()
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, (json.loads(cap.out) if cap.out.strip() else None), (json.loads(cap.err) if cap.err.strip() else None)


def test_config_search_min_similarity_roundtrip_and_validation(capsys, home):
    code, out, _ = run(capsys, "config")
    assert code == 0 and out["search_min_similarity"] is None
    assert out["search_min_similarity_effective"] == default_min_similarity("text-embedding-3-small")
    code, out, _ = run(capsys, "config", "--search-min-similarity", "0.55")
    assert code == 0 and out["search_min_similarity"] == 0.55 and out["search_min_similarity_effective"] == 0.55
    assert out["changed"] == ["search_min_similarity"]
    assert json.loads((home / "config.json").read_text(encoding="utf-8"))["search_min_similarity"] == 0.55
    assert MemoryConfig.load(home).search_min_similarity == 0.55
    code, out, _ = run(capsys, "config", "--embed-model", "nomic-embed-text", "--embed-base-url", "http://h:1/v1", "--embed-dimensions", "8", "--no-probe", "--search-min-similarity", "default")
    assert code == 0 and out["search_min_similarity"] is None and out["search_min_similarity_effective"] == 0.72
    code, _, err = run(capsys, "config", "--search-min-similarity", "1.5")
    assert code == 2 and "between -1 and 1" in err["error"]
    code, _, err = run(capsys, "config", "--search-min-similarity", "high")
    assert code == 2 and "number" in err["error"]


def test_config_json_with_garbage_similarity_falls_back_to_default(home):
    home.mkdir(parents=True)
    (home / "config.json").write_text(json.dumps({"search_min_similarity": "high"}), encoding="utf-8")
    assert MemoryConfig.load(home).search_min_similarity is None
    (home / "config.json").write_text(json.dumps({"search_min_similarity": 7}), encoding="utf-8")
    assert MemoryConfig.load(home).search_min_similarity is None


def test_cli_search_min_similarity_and_hint(capsys, home):
    run(capsys, "add", "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}", "--author", "太郎")
    code, out, _ = run(capsys, "search", "Rails 担当")
    assert code == 0 and out["count"] >= 1 and out["min_similarity"] == MIN_SIMILARITY_NO_MODEL and "hint" not in out
    assert out["facts"][0]["similarity"] is not None
    code, out, _ = run(capsys, "search", "Kubernetes のクラスタ構成は？", "--min-similarity", "0.9")
    assert code == 0 and out["count"] == 0 and out["min_similarity"] == 0.9 and out["hint"] == NO_HITS_HINT
    run(capsys, "config", "--search-min-similarity", "0.4")
    code, out, _ = run(capsys, "search", "Rails 担当")
    assert out["min_similarity"] == 0.4  # from config.json
    code, out, _ = run(capsys, "search", "Rails 担当", "--min-similarity", "0.2")
    assert out["min_similarity"] == 0.2
    code, _, err = run(capsys, "search", "Rails", "--min-similarity", "2")
    assert code == 2 and "between -1 and 1" in err["error"]


def test_mcp_argv_carries_the_configured_threshold(home):
    cfg = MemoryConfig.load(home)

    class A:
        transport = "stdio"
        read_only = False
        extractor = None
        host = "127.0.0.1"
        port = 8000

    assert "--min-similarity" not in cli.mcp_argv(cfg, A())
    cfg.search_min_similarity = 0.55
    argv = cli.mcp_argv(cfg, A())
    assert argv[argv.index("--min-similarity") + 1] == "0.55"


# --------------------------------------------------------------------------- forget_memory aliases and messages


@pytest.fixture
def forget_tools(tmp_path):
    mem = Memory(tmp_path / "m.db", FakeLLM(), FakeEmbedder(), default_group_id=G)
    mem.add_episode(
        "[[Rails|Service]] [[alice|Person]] {{Rails|OWNER|alice|2026-07-01}}",
        group_id=G, channel="personal", author="alice", occurred_at="2026-07-01 10:00", source="claude-code", source_ref="ref-1",
    )
    mem.add_episode(
        "[[Vault|Project]] [[bob|Person]] {{Vault|OWNER|bob|2026-07-02}}",
        group_id=G, channel="personal", author="bob", occurred_at="2026-07-02 10:00", source="codex", source_ref="ref-2",
    )
    t = MemoryTools(mem, group_id=G)
    yield t, mem
    mem.close()


def first_episode(mem, who="alice"):
    return next(e["id"] for e in mem.get_episodes(group_id=G) if e["author"] == who)


def test_forget_memory_accepts_aliases(forget_tools):
    t, mem = forget_tools
    ep = first_episode(mem)
    fact = next(h.id for h in mem.search("Rails OWNER", min_similarity=-1.0))
    for kw in ({"episode_id": ep}, {"episode": ep}, {"episode": ep, "episode_id": ep}, {"source_ref": "ref-1"}, {"source_ref": "ref-1", "source": "claude-code"}):
        res = t.forget_memory(dry_run=True, **kw)
        assert res["dry_run"] and res["episodes_deleted"] == [ep], kw
    assert t.forget_memory(fact_id=fact, dry_run=True)["facts_deleted"] == [fact]
    assert t.forget_memory(fact=fact, dry_run=True)["facts_deleted"] == [fact]
    ent = next(e["id"] for e in mem.search_entities("Vault", group_id=G))
    for kw in ({"entity": "Vault"}, {"entity_id": ent}, {"name": "Vault"}, {"entity": "Vault", "name": "Vault"}):
        assert ent in t.forget_memory(dry_run=True, **kw)["entities_deleted"], kw
    # a real delete through an alias
    res = t.forget_memory(episode=ep)
    assert res["episodes_deleted"] == [ep] and len(mem.get_episodes(group_id=G)) == 1


def test_forget_memory_alias_conflicts_and_stray_qualifiers(forget_tools):
    t, mem = forget_tools
    with pytest.raises(ValueError, match="conflicting.*episode_id.*episode"):
        t.forget_memory(episode_id="ep_a", episode="ep_b")
    with pytest.raises(ValueError, match="conflicting.*entity"):
        t.forget_memory(entity="Rails", name="Vault")
    with pytest.raises(ValueError, match="source.*qualifies"):
        t.forget_memory(episode_id="ep_x", source="codex")
    # source alone is a range target (like the CLI); with source_ref it is a qualifier
    assert len(t.forget_memory(source="codex", dry_run=True)["episodes_deleted"]) == 1


def test_forget_memory_selector_errors_list_the_accepted_forms(forget_tools):
    t, _ = forget_tools
    for kw, got in (({}, "none"), ({"dry_run": True}, "none"), ({"episode": "ep_x", "fact": "fact_x"}, "episode_id, fact_id"), ({"name": "Rails", "since": "2026-01-01"}, "entity, since/until/channel")):
        with pytest.raises(ValueError) as exc:
            t.forget_memory(**kw)
        msg = str(exc.value)
        assert "exactly one selector" in msg and f"got {got}" in msg, msg
        for needle in ("episode_id (alias episode)", "fact_id (alias fact)", "entity (aliases entity_id, name)", "source_ref", "since / until / channel", 'forget_memory(episode_id="ep_...", dry_run=True)'):
            assert needle in msg, (needle, msg)


@needs_mcp
def test_forget_memory_tool_over_the_server_shows_alias_params_and_readable_errors(forget_tools):
    t, mem = forget_tools
    server = build_server(t)
    ep = first_episode(mem)

    async def go():
        tools = {x.name: x for x in await server.list_tools()}
        tool = tools["forget_memory"]
        schema = getattr(tool, "inputSchema", None) or tool.input_schema
        for name in ("episode", "fact", "entity_id", "name", "source_ref", "source"):
            assert name in schema["properties"], name
        assert "alias" in json.dumps(schema["properties"]["episode"])
        assert not schema.get("required")
        assert tool.description.startswith('Example: forget_memory(episode_id="ep_...", dry_run=True)')
        ok = await server.call_tool("forget_memory", {"episode": ep, "dry_run": True})  # the call that failed in real use
        for bad in ({}, {"episodes": ep}):  # unknown argument names are dropped by the SDK: no selector is left
            with pytest.raises(Exception) as exc:
                await server.call_tool("forget_memory", bad)
            text = str(exc.value)
            assert "episode_id (alias episode)" in text and "exactly one selector" in text, text  # the message reaches the client
        return ok

    ok = asyncio.run(go())
    content = ok.content if hasattr(ok, "content") else ok[0]
    payload = json.loads(content[0].text)
    assert payload["dry_run"] and payload["episodes_deleted"] == [ep]
    assert len(mem.get_episodes(group_id=G)) == 2  # dry run: still there


@needs_mcp
def test_mcp_expected_errors_keep_their_message_over_the_server(forget_tools):
    t, mem = forget_tools
    ro = MemoryTools(mem, group_id=G, read_only=True)
    server = build_server(MemoryTools(mem, group_id=G))

    async def go():
        with pytest.raises(Exception, match="min_similarity must be a number between -1 and 1"):
            await server.call_tool("search_memory", {"query": "x", "min_similarity": 5})

    asyncio.run(go())
    with pytest.raises(PermissionError, match="read-only"):
        ro.forget_memory(episode="ep_x")


def test_cli_forget_aliases(capsys, home):
    run(capsys, "add", "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}", "--author", "太郎")
    eps = run(capsys, "episodes")[1]["episodes"]
    ep = eps[0]["id"]
    code, out, _ = run(capsys, "forget", "--episode-id", ep, "--dry-run")
    assert code == 0 and out["dry_run"] and out["episodes_deleted"] == [ep]
    fid = run(capsys, "search", "Rails 担当", "--min-similarity", "-1")[1]["facts"][0]["id"]
    code, out, _ = run(capsys, "forget", "--fact-id", fid, "--dry-run")
    assert code == 0 and out["facts_deleted"] == [fid]
    code, out, _ = run(capsys, "forget", "--episode", ep, "--dry-run")  # the original spelling still works
    assert code == 0 and out["episodes_deleted"] == [ep]
    code, _, err = run(capsys, "forget", "--episode-id", ep, "--fact-id", fid)
    assert code == 2 and "--episode-id" in err["error"] and "--fact-id" in err["error"] and "got: --episode, --fact" in err["error"]
    code, _, err = run(capsys, "forget")
    assert code == 2 and "example: sqgraph-memory forget --episode-id" in err["error"]
