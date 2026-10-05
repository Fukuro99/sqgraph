"""Embeddings from an OpenAI-compatible server (a stdlib stub on localhost): provider, identity, reembed, CLI, MCP."""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from sqgraph import GraphDB
from sqgraph.errors import SqgraphError
from sqgraph.memory import FakeEmbedder, FakeLLM, Memory, OpenAIEmbedder, UsageTracker, cli, probe_dimensions
from sqgraph.memory.core import ENTITY_INDEX, FACT_INDEX
from sqgraph.memory.providers import EmbedderError, build_embedder, normalize_base_url

TARO = "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}"
HANAKO = "[[Rails|Service]] [[花子|Person]] {{Rails|担当|花子|2026-08-15}}"


# --------------------------------------------------------------------------- the stub server


class EmbeddingStub:
    """``POST /v1/embeddings`` in the OpenAI response shape; vectors come from :class:`FakeEmbedder` (``dims``)."""

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims
        self.reject_dimensions = True  # like servers that 400 on the OpenAI-only ``dimensions`` parameter
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, obj: dict[str, Any]) -> None:
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append(
                    {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                )
                if self.path != "/v1/embeddings":
                    return self._send(404, {"error": {"message": "not found"}})
                if outer.reject_dimensions and "dimensions" in body:
                    return self._send(400, {"error": {"message": "dimensions is not supported"}})
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vecs = FakeEmbedder(outer.dims).embed(inputs)
                tokens = sum(len(t) for t in inputs)
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                        "model": body.get("model"),
                        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                    },
                )

            def log_message(self, *args: object) -> None:  # silence
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(autouse=True)
def fast_failures(monkeypatch):
    """An unreachable host must not make the tests wait (Windows takes seconds to refuse a connection)."""
    from sqgraph.memory import providers

    monkeypatch.setattr(providers, "CUSTOM_EMBED_TIMEOUT", 1.0)
    monkeypatch.setattr(providers, "CUSTOM_EMBED_CONNECT_TIMEOUT", 0.3)
    monkeypatch.setattr(providers, "CUSTOM_EMBED_RETRIES", 0)


@pytest.fixture
def stub():
    s = EmbeddingStub(384)
    try:
        yield s
    finally:
        s.close()


def dead_url() -> str:
    """A localhost URL nothing listens on (connection refused immediately)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


class Named(FakeEmbedder):
    """A fake embedder that names its model / endpoint; different models give different vectors."""

    def __init__(self, model: str, dims: int = 64, endpoint: str = "test://one") -> None:
        super().__init__(dims)
        self.model = model
        self.endpoint = endpoint

    def embed(self, texts):
        return super().embed([f"{self.model}|{t}" for t in texts])


# --------------------------------------------------------------------------- provider


def test_embedder_against_stub_sends_no_dimensions(stub, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    tracker = UsageTracker()
    emb = OpenAIEmbedder("bge-m3", dimensions=384, base_url=stub.url, tracker=tracker)
    vecs = emb.embed(["こんにちは", "hello", "   "])
    assert len(vecs) == 3 and all(len(v) == 384 for v in vecs)
    assert emb.endpoint == stub.url
    req = stub.requests[0]
    assert req["path"] == "/v1/embeddings"
    assert "dimensions" not in req["body"]  # servers may reject it
    assert req["body"]["model"] == "bge-m3" and req["body"]["encoding_format"] == "float"
    assert req["body"]["input"][2] == "."  # blank text is replaced
    assert "sk-must-not-leak" not in json.dumps(req["headers"])  # the OpenAI key never reaches a custom endpoint
    assert req["headers"]["authorization"] == "Bearer sqgraph-local"  # placeholder
    # unknown model: tokens tracked, cost 0, summary does not crash
    assert tracker.total_calls() == 1 and tracker.total_usd() == 0.0
    assert "bge-m3" in tracker.summary()


def test_embedder_api_key_batching_and_send_dimensions(stub):
    emb = OpenAIEmbedder("m", dimensions=384, base_url=stub.url, api_key="secret-lan-key")
    assert emb.batch_size == 32  # TEI-friendly default for custom endpoints
    emb.embed([f"t{i}" for i in range(70)])
    assert [len(r["body"]["input"]) for r in stub.requests] == [32, 32, 6]
    assert stub.requests[0]["headers"]["authorization"] == "Bearer secret-lan-key"

    stub.reject_dimensions = False
    stub.requests.clear()
    OpenAIEmbedder("m", dimensions=384, base_url=stub.url, send_dimensions=True).embed(["x"])
    assert stub.requests[0]["body"]["dimensions"] == 384


def test_embedder_validates_dimensions_and_count(stub):
    emb = OpenAIEmbedder("m", dimensions=100, base_url=stub.url)
    with pytest.raises(EmbedderError, match="384-dimensional.*100 dimensions are configured") as ei:
        emb.embed(["x"])
    assert ei.value.actual == 384

    class Short:  # fewer vectors than texts
        class embeddings:
            @staticmethod
            def create(**kw):
                class R:
                    data = []
                    usage = None

                return R()

    with pytest.raises(EmbedderError, match="0 vectors for 1 texts"):
        OpenAIEmbedder("m", dimensions=4, client=Short()).embed(["x"])


def test_openai_default_behaviour_unchanged():
    sent: list[dict[str, Any]] = []

    class Client:
        class embeddings:
            @staticmethod
            def create(**kw):
                sent.append(kw)

                class D:
                    def __init__(self, i):
                        self.index, self.embedding = i, [0.0] * kw.get("dimensions", 1536)

                class R:
                    data = [D(i) for i in range(len(kw["input"]))]
                    usage = None

                return R()

    OpenAIEmbedder(client=Client()).embed(["a"])
    OpenAIEmbedder("text-embedding-3-large", dimensions=256, client=Client()).embed(["a"])
    assert "dimensions" not in sent[0] and "encoding_format" not in sent[0]
    assert sent[1]["dimensions"] == 256


def test_probe_dimensions(stub):
    assert probe_dimensions({"model": "m", "base_url": stub.url}) == 384
    assert probe_dimensions(model="m", base_url=stub.url) == 384
    assert probe_dimensions(FakeEmbedder(17)) == 17
    # an embedder with the wrong configured dimension still reports what the server really produces
    assert probe_dimensions(OpenAIEmbedder("m", dimensions=5, base_url=stub.url)) == 384
    stub.dims = 768
    assert probe_dimensions({"model": "m", "base_url": stub.url}) == 768
    emb, probed = build_embedder("m", base_url=stub.url)
    assert probed and emb.dimensions == 768
    emb, probed = build_embedder("m", base_url=stub.url, dimensions=10)
    assert not probed and emb.dimensions == 10


def test_probe_unreachable_endpoint_fails_fast():
    with pytest.raises(Exception) as ei:
        probe_dimensions({"model": "m", "base_url": dead_url()})
    assert type(ei.value).__name__ in ("APIConnectionError", "APITimeoutError")


def test_normalize_base_url():
    assert normalize_base_url("http://pc:11434") == "http://pc:11434/v1"
    assert normalize_base_url("http://pc:1234/v1/") == "http://pc:1234/v1"
    assert normalize_base_url(" https://x.example/api/v1 ") == "https://x.example/api/v1"
    for bad in ("pc:11434", "ftp://pc", ""):
        with pytest.raises(ValueError):
            normalize_base_url(bad)


# --------------------------------------------------------------------------- identity in the database


def meta(path, key):
    with GraphDB(str(path), read_only=True) as db:
        return db.get_meta(key)


def test_identity_stored_and_checked(tmp_path):
    path = tmp_path / "m.db"
    with Memory(path, FakeLLM(), Named("m1", 64, "test://one"), default_group_id="g") as mem:
        assert mem.embedding_identity() == {"model": "m1", "dimensions": 64, "endpoint": "test://one"}
    assert (meta(path, "embed_model"), meta(path, "embed_dimensions"), meta(path, "embed_endpoint")) == ("m1", "64", "test://one")

    # a different model with the SAME dimensions is refused, with the way out
    with pytest.raises(SqgraphError, match=r"embedding model mismatch.*'m1'.*'m2'.*reembed"):
        Memory(path, FakeLLM(), Named("m2", 64))
    # different dimensions: same family of error, mentions reembed too
    with pytest.raises(SqgraphError, match=r"mismatch.*reembed"):
        Memory(path, FakeLLM(), Named("m1", 32))
    # read-only opens check as well, but never write
    with pytest.raises(SqgraphError, match="embedding model mismatch"):
        Memory(path, None, Named("m2", 64), read_only=True)
    with Memory(path, None, Named("m1", 64, "test://moved"), read_only=True):
        pass
    assert meta(path, "embed_endpoint") == "test://one"

    # the endpoint may change (same model + dimensions): just recorded
    with Memory(path, FakeLLM(), Named("m1", 64, "test://two")):
        pass
    assert meta(path, "embed_endpoint") == "test://two"

    # no embedder: no check; the escape hatch skips it
    with Memory(path, None, None) as mem:
        assert mem.embedding_identity()["model"] == "m1"
    with Memory(path, None, Named("m9", 64), allow_embedder_change=True):
        pass
    assert meta(path, "embed_model") == "m1"  # allow_embedder_change does not rewrite the identity by itself


def test_database_without_identity_adopts_the_embedder(tmp_path):
    path = tmp_path / "legacy.db"
    with Memory(path, FakeLLM(), FakeEmbedder(64), default_group_id="g") as mem:  # embedder without a model name
        mem.add_episode(TARO, group_id="g", channel="c", author="a")
        assert mem.embedding_identity() is None
    assert meta(path, "embed_model") is None
    with Memory(path, FakeLLM(), Named("m1", 64)) as mem:  # adopts
        assert mem.embedding_identity()["model"] == "m1"
    with pytest.raises(SqgraphError, match="dimension mismatch"):  # dimensions are still checked for legacy DBs
        Memory(tmp_path / "legacy.db", FakeLLM(), FakeEmbedder(32))


# --------------------------------------------------------------------------- reembed


def seeded(path, embedder, texts=(TARO, HANAKO)):
    mem = Memory(path, FakeLLM(), embedder, default_group_id="g")
    for i, t in enumerate(texts):
        mem.add_episode(t, group_id="g", channel="c", author="u", occurred_at=f"2026-0{7 + i}-01 10:00", source="t")
    return mem


def counts(mem):
    st = mem.stats("g")
    return st["nodes"]["Entity"], st["facts_valid"] + st["facts_invalidated"]


def test_reembed_new_model_same_dimensions(tmp_path):
    path = tmp_path / "m.db"
    seeded(path, Named("m1", 64)).close()
    with pytest.raises(SqgraphError, match="reembed"):
        Memory(path, None, Named("m2", 64))
    seen: list[tuple[str, int, int]] = []
    with Memory(path, None, Named("m2", 64, "test://two"), default_group_id="g", allow_embedder_change=True) as mem:
        ents, facts = counts(mem)
        res = mem.reembed(batch_size=2, progress=lambda *a: seen.append(a))
        assert (res["entities"], res["facts"]) == (ents, facts) and ents >= 3 and facts == 2
        assert res["model"] == "m2" and res["dimensions"] == 64 and res["indexes_recreated"] is False
        assert res["previous"]["model"] == "m1"
        assert seen[-1][0] == "facts" and seen[-1][1:] == (facts, facts) and {k for k, _, _ in seen} == {"entities", "facts"}
    with Memory(path, None, Named("m2", 64, "test://two"), default_group_id="g") as mem:  # opens normally now
        assert mem.embedding_identity() == {"model": "m2", "dimensions": 64, "endpoint": "test://two"}
        hits = mem.search("Rails 担当", group_id="g", include_invalid=True)
        assert {h.object["name"] for h in hits if h.relation == "担当"} == {"太郎", "花子"}
        # the vectors really are the new model's: a fact is nearest to its own text embedded with m2
        qv = Named("m2", 64).embed(["Rails 担当 花子"])[0]
        with_vec = mem.db.vector_search(FACT_INDEX, qv, k=1)[0][0]
        assert with_vec["fact"].endswith("花子")


def test_reembed_dimension_change_recreates_indexes(tmp_path):
    path = tmp_path / "m.db"
    seeded(path, Named("small", 384)).close()
    with Memory(path, None, Named("big", 768), default_group_id="g", allow_embedder_change=True) as mem:
        res = mem.reembed()
        assert res["indexes_recreated"] is True and res["dimensions"] == 768
        dims = {i.name: i.dimensions for i in mem.db.vector_indexes()}
        assert dims[ENTITY_INDEX] == dims[FACT_INDEX] == 768
    with Memory(path, FakeLLM(), Named("big", 768), default_group_id="g") as mem:
        ents, facts = counts(mem)
        qv = Named("big", 768).embed(["Rails"])[0]
        assert len(mem.db.vector_search(ENTITY_INDEX, qv, k=50)) == ents  # every entity has a vector again
        assert len(mem.db.vector_search(FACT_INDEX, qv, k=50)) == facts
        hits = mem.search("Rails 担当", group_id="g")
        assert [h.object["name"] for h in hits if h.relation == "担当"] == ["花子"]
        # ingestion keeps working on the new indexes
        r = mem.add_episode("[[Rails|Service]] [[次郎|Person]] {{Rails|担当|次郎|2026-09-01}}", group_id="g", channel="c", author="u")
        assert len(r.facts_created) == 1
    with pytest.raises(SqgraphError, match="mismatch"):
        Memory(path, None, Named("small", 384))


def test_reembed_failure_leaves_the_database_untouched(tmp_path):
    path = tmp_path / "m.db"
    seeded(path, Named("m1", 64)).close()

    class Flaky(Named):
        def embed(self, texts):
            if getattr(self, "armed", False):
                raise ConnectionError("server went away")
            self.armed = True
            return super().embed(texts)

    with Memory(path, None, Flaky("m2", 32), default_group_id="g", allow_embedder_change=True) as mem:
        with pytest.raises(ConnectionError):
            mem.reembed(batch_size=1)
        assert {i.dimensions for i in mem.db.vector_indexes() if i.name in (ENTITY_INDEX, FACT_INDEX)} == {64}
    assert meta(path, "embed_model") == "m1"
    with Memory(path, None, Named("m1", 64), default_group_id="g") as mem:  # still fully usable with the old model
        assert mem.search("Rails 担当", group_id="g")


def test_reembed_single_group_rules(tmp_path):
    path = tmp_path / "m.db"
    mem = seeded(path, Named("m1", 64))
    mem.add_episode("[[Vault|Service]] [[次郎|Person]] {{Vault|担当|次郎|2026-07-01}}", group_id="other", channel="c", author="u")
    mem.close()
    with Memory(path, None, Named("m2", 64), default_group_id="g", allow_embedder_change=True) as mem:
        with pytest.raises(SqgraphError, match="other groups"):
            mem.reembed("g")
        res = mem.reembed()  # all groups
        assert res["group_id"] is None and res["entities"] >= 5
    with Memory(path, None, Named("m2", 64), default_group_id="g") as mem:
        assert mem.reembed("g")["entities"] >= 3  # same model: refreshing one group is fine


def test_reembed_needs_embedder_and_writable(tmp_path):
    path = tmp_path / "m.db"
    seeded(path, Named("m1", 64)).close()
    with Memory(path, None, None) as mem:
        with pytest.raises(RuntimeError, match="embedder"):
            mem.reembed()
    with Memory(path, None, Named("m1", 64), read_only=True) as mem:
        with pytest.raises(SqgraphError, match="read-only"):
            mem.reembed()


# --------------------------------------------------------------------------- CLI (real providers against the stub)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mem-home"
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(h))
    for k in ("OPENAI_API_KEY", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP", "SQGRAPH_EMBED_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return h


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    capsys.readouterr()
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, (json.loads(cap.out) if cap.out.strip() else None), (json.loads(cap.err) if cap.err.strip() else None)


def read_config(home):
    return json.loads((home / "config.json").read_text(encoding="utf-8"))


def test_init_with_embed_base_url_probes_and_saves(capsys, home, stub, monkeypatch):
    code, out, err = run(capsys, "init", "--embed-base-url", stub.url.removesuffix("/v1"), "--embed-model", "bge-m3", stdin="", monkeypatch=monkeypatch)
    assert code == 0 and err is None
    assert out["probed"] is True
    assert out["embedder"] == {"endpoint": stub.url, "model": "bge-m3", "dimensions": 384, "api_key_env": "SQGRAPH_EMBED_API_KEY", "api_key": "not set (optional)"}
    cfg = read_config(home)
    assert (cfg["embed_base_url"], cfg["embed_model"], cfg["embed_dimensions"], cfg["embed_api_key_env"]) == (stub.url, "bge-m3", 384, "SQGRAPH_EMBED_API_KEY")
    assert "dimensions" not in stub.requests[0]["body"]
    assert "OPENAI_API_KEY" in out["warning"]  # the LLM key is still missing

    # a second init without options keeps everything and does not probe again
    n = len(stub.requests)
    code, out, _ = run(capsys, "init", stdin="", monkeypatch=monkeypatch)
    assert out["embedder"]["dimensions"] == 384 and "probed" not in out and len(stub.requests) == n


def test_init_probe_failures_and_no_probe(capsys, home, stub, monkeypatch):
    code, _, err = run(capsys, "init", "--embed-base-url", dead_url(), "--embed-model", "m", stdin="", monkeypatch=monkeypatch)
    assert code == 1 and "cannot get an embedding" in err["error"] and "--no-probe" in err["error"]
    assert not (home / "config.json").exists()  # nothing saved on a failed probe

    code, out, _ = run(capsys, "init", "--embed-base-url", dead_url(), "--embed-model", "m", "--no-probe", stdin="", monkeypatch=monkeypatch)
    assert code == 0 and out["embedder"]["dimensions"] is None
    assert read_config(home)["embed_dimensions"] is None and "not known yet" in out["warning"]

    # an explicit dimension that disagrees with the server is rejected
    code, _, err = run(capsys, "init", "--embed-base-url", stub.url, "--embed-model", "m", "--embed-dimensions", "100", stdin="", monkeypatch=monkeypatch)
    assert code == 2 and "384" in err["error"]
    code, out, _ = run(capsys, "init", "--embed-base-url", stub.url, "--embed-model", "m", "--embed-dimensions", "384", stdin="", monkeypatch=monkeypatch)
    assert code == 0 and out["embedder"]["dimensions"] == 384
    code, _, err = run(capsys, "init", "--embed-base-url", "not a url", stdin="", monkeypatch=monkeypatch)
    assert code == 2 and "invalid base URL" in err["error"]


def test_init_stores_embed_key_from_env_without_echo(capsys, home, stub, monkeypatch):
    monkeypatch.setenv("SQGRAPH_EMBED_API_KEY", "lan-secret-42")
    code, out, _ = run(capsys, "init", "--embed-base-url", stub.url, "--embed-model", "m", stdin="", monkeypatch=monkeypatch)
    assert code == 0 and out["embedder"]["api_key"] == "set"
    assert "lan-secret-42" not in json.dumps(out)
    assert (home / ".env").read_text(encoding="utf-8") == "SQGRAPH_EMBED_API_KEY=lan-secret-42\n"
    assert stub.requests[0]["headers"]["authorization"] == "Bearer lan-secret-42"


def populate(home, stub, model="bge-m3", texts=(TARO, HANAKO)):
    """Write a database the way the CLI would (FakeLLM for the extraction, the stub for the embeddings)."""
    with Memory(home / "memory.db", FakeLLM(), OpenAIEmbedder(model, dimensions=stub.dims, base_url=stub.url), default_group_id="me", tz="Asia/Tokyo") as mem:
        for i, t in enumerate(texts):
            mem.add_episode(t, group_id="me", channel="personal", author="user", occurred_at=f"2026-0{7 + i}-01 10:00", source="cli")


def test_search_uses_the_lan_embedder_without_openai_key(capsys, home, stub, monkeypatch):
    run(capsys, "init", "--embed-base-url", stub.url, "--embed-model", "bge-m3", stdin="", monkeypatch=monkeypatch)
    populate(home, stub)
    assert meta(home / "memory.db", "embed_endpoint") == stub.url and meta(home / "memory.db", "embed_model") == "bge-m3"
    stub.requests.clear()
    code, out, err = run(capsys, "search", "Rails 担当", "--history")
    assert code == 0 and err is None and "warning" not in out  # vector search is on, no OPENAI_API_KEY needed
    assert out["count"] >= 2 and stub.requests and all("dimensions" not in r["body"] for r in stub.requests)
    code, out, _ = run(capsys, "entities", "Rails")
    assert out["entities"][0]["name"] == "Rails"

    # forget re-embeds the entities it touches through the same endpoint
    stub.requests.clear()
    _, found, _ = run(capsys, "search", "Rails 担当", "--history")
    hanako = next(f["id"] for f in found["facts"] if f["object"]["name"] == "花子")
    stub.requests.clear()
    code, out, _ = run(capsys, "forget", "--fact", hanako)
    assert code == 0 and out["facts_deleted"] == [hanako] and stub.requests

    # `add` is the only command that needs the OpenAI LLM key
    code, _, err = run(capsys, "add", "覚えておいてください、これは長いテキストです")
    assert code == 1 and "OPENAI_API_KEY" in err["error"]


def test_config_show_and_change_warns_about_existing_database(capsys, home, stub, monkeypatch):
    run(capsys, "init", "--embed-base-url", stub.url, "--embed-model", "bge-m3", stdin="", monkeypatch=monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-config-secret")
    code, out, _ = run(capsys, "config")
    assert code == 0 and out["embedder"]["model"] == "bge-m3" and out["embedder"]["endpoint"] == stub.url
    assert out["openai_api_key"] == "configured" and out["db_embedding"] is None and "changed" not in out
    assert "sk-config-secret" not in json.dumps(out)

    populate(home, stub)
    code, out, _ = run(capsys, "config")
    assert out["db_embedding"] == {"model": "bge-m3", "dimensions": 384, "endpoint": stub.url} and "warning" not in out

    stub.dims = 768  # the other model produces other vectors
    code, out, _ = run(capsys, "config", "--embed-model", "nomic-embed-text")
    assert code == 0 and out["changed"] == ["embed_model", "embed_dimensions"] and out["probed"] is True
    assert out["embedder"]["dimensions"] == 768 and out["reembed_required"] is True and "reembed" in out["warning"]
    assert read_config(home)["embed_model"] == "nomic-embed-text" and read_config(home)["embed_dimensions"] == 768

    # searching with the mismatching configuration is refused with the way out
    code, _, err = run(capsys, "search", "Rails")
    assert code == 1 and "reembed" in err["error"]
    code, out, _ = run(capsys, "doctor")
    names = {c["name"]: c for c in out["checks"]}
    assert names["database embedding"]["status"] == "fail" and "reembed" in names["database embedding"]["detail"]
    assert code == 1

    # reembed: the plan first, then for real
    code, out, _ = run(capsys, "reembed")
    assert code == 0 and out["executed"] is False and out["facts"] == 2 and out["from"]["model"] == "bge-m3"
    assert out["to"] == {"model": "nomic-embed-text", "dimensions": 768, "endpoint": stub.url} and "--yes" in out["next"]
    code, out, _ = run(capsys, "reembed", "--yes")
    assert code == 0 and out["executed"] is True and out["indexes_recreated"] is True and out["dimensions"] == 768
    assert out["previous"]["model"] == "bge-m3" and out["entities"] >= 3 and out["facts"] == 2

    code, out, _ = run(capsys, "search", "Rails 担当", "--history")
    assert code == 0 and out["count"] >= 2
    code, out, _ = run(capsys, "config")
    assert out["db_embedding"]["model"] == "nomic-embed-text" and out["db_embedding"]["dimensions"] == 768 and "warning" not in out

    # back to the OpenAI API: the stored endpoint settings are dropped, the warning comes back
    code, out, _ = run(capsys, "config", "--embed-base-url", "openai", "--embed-model", "text-embedding-3-small")
    assert out["embedder"] == {"endpoint": "openai", "model": "text-embedding-3-small", "dimensions": 1536} and out["reembed_required"] is True
    assert read_config(home)["embed_base_url"] is None and read_config(home)["embed_dimensions"] is None


def test_reembed_without_database_and_bad_args(capsys, home):
    code, out, _ = run(capsys, "reembed", "--yes")
    assert code == 0 and out["executed"] is False and "nothing to re-embed" in out["message"]
    code, _, err = run(capsys, "config", "--embed-dimensions", "0")
    assert code == 2 and "positive" in err["error"]
    code, _, err = run(capsys, "config", "--embed-api-key-env", "bad name")
    assert code == 2


def test_mcp_argv_forwards_embedder_settings(capsys, home, monkeypatch):
    home.mkdir()
    (home / "config.json").write_text(
        json.dumps({"embed_model": "bge-m3", "embed_base_url": "http://nas:11434/v1", "embed_dimensions": 1024, "embed_api_key_env": "MY_KEY"}),
        encoding="utf-8",
    )
    from sqgraph.memory import mcp as mcp_mod

    seen: list[list[str]] = []
    monkeypatch.setattr(mcp_mod, "main", lambda argv=None: seen.append(list(argv)))
    assert cli.main(["mcp"]) == 0
    pairs = dict(zip(seen[0][::2], seen[0][1::2]))
    assert pairs["--embed-base-url"] == "http://nas:11434/v1" and pairs["--embed-dimensions"] == "1024"
    assert pairs["--embed-api-key-env"] == "MY_KEY" and pairs["--embed-model"] == "bge-m3"
    # an old config.json without the new fields -> no embedder options forwarded
    (home / "config.json").write_text(json.dumps({"group_id": "x", "embed_model": "text-embedding-3-small"}), encoding="utf-8")
    seen.clear()
    assert cli.main(["mcp"]) == 0
    assert not {"--embed-base-url", "--embed-dimensions", "--embed-api-key-env"} & set(seen[0])


def test_doctor_reports_the_embedder(capsys, home, stub, tmp_path, monkeypatch):
    fake_home = tmp_path / "fakeuser"
    fake_home.mkdir()
    monkeypatch.setattr(cli, "user_home", lambda: fake_home)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    run(capsys, "init", "--embed-base-url", stub.url, "--embed-model", "bge-m3", stdin="", monkeypatch=monkeypatch)
    populate(home, stub)
    stub.requests.clear()
    code, out, _ = run(capsys, "doctor")
    names = {c["name"]: c for c in out["checks"]}
    assert stub.requests == []  # offline doctor does not touch the network
    assert names["embedder"]["status"] == "ok" and stub.url in names["embedder"]["detail"] and "bge-m3" in names["embedder"]["detail"]
    assert "384" in names["embedder"]["detail"]
    assert names["database embedding"]["status"] == "ok" and "matches" in names["database embedding"]["detail"]

    code, out, _ = run(capsys, "doctor", "--online")
    names = {c["name"]: c for c in out["checks"]}
    assert names["online embedding"]["status"] == "ok" and "384" in names["online embedding"]["detail"] and stub.requests

    stub.dims = 768  # the server now serves another model under the same name
    code, out, _ = run(capsys, "doctor", "--online")
    names = {c["name"]: c for c in out["checks"]}
    assert names["online embedding"]["status"] == "fail" and "768" in names["online embedding"]["detail"] and code == 1


def test_doctor_online_unreachable_endpoint(capsys, home, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "user_home", lambda: tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    run(capsys, "init", "--embed-base-url", dead_url(), "--embed-model", "m", "--embed-dimensions", "8", "--no-probe", stdin="", monkeypatch=monkeypatch)
    code, out, _ = run(capsys, "doctor", "--online")
    names = {c["name"]: c for c in out["checks"]}
    assert names["online embedding"]["status"] == "fail" and code == 1


# --------------------------------------------------------------------------- MCP server


def test_mcp_main_with_lan_embedder_and_no_openai_key(tmp_path, stub, monkeypatch):
    from sqgraph.memory import mcp as mcp_mod

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "memory.db"
    with Memory(path, FakeLLM(), OpenAIEmbedder("bge-m3", dimensions=384, base_url=stub.url), default_group_id="me") as mem:
        mem.add_episode(TARO, group_id="me", channel="personal", author="user", occurred_at="2026-07-01 10:00", source="t")
    stub.requests.clear()

    tools_seen: list[Any] = []
    checks: list[Any] = []
    monkeypatch.setattr(mcp_mod, "build_server", lambda tools: tools_seen.append(tools) or object())
    # main() closes the Memory after the server stops: run the checks while it is "serving"
    monkeypatch.setattr(mcp_mod, "run_server", lambda *a, **k: checks.pop()(tools_seen[-1]))
    env = ["--env-file", str(tmp_path / "nope.env")]  # no .env: only the process environment counts

    def lan(tools):
        res = tools.search_memory("Rails 担当")
        assert res["count"] >= 1 and stub.requests and "dimensions" not in stub.requests[0]["body"]
        assert tools.get_entity("Rails")["found"] is True
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):  # writing needs the LLM: a clear error, nothing stored
            tools.add_memory("覚えておいてください、これは長いテキストです")
        assert tools.memory.stats("me")["nodes"]["Episode"] == 1

    checks.append(lan)
    mcp_mod.main(["--db", str(path), "--group-id", "me", *env, "--embed-model", "bge-m3", "--embed-base-url", stub.url, "--embed-dimensions", "384"])  # fmt: skip

    def degraded(tools):  # no custom embedder and no key: the read tools fall back to fulltext instead of crashing
        assert tools.memory.embedder is None
        assert tools.search_memory("Rails")["count"] >= 0

    checks.append(degraded)
    mcp_mod.main(["--db", str(path), "--group-id", "me", *env, "--read-only"])
    assert not checks
