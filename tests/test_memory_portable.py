"""Export / import (JSONL) and database snapshots (sqgraph.memory.portable + CLI)."""

from __future__ import annotations

import io
import json

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, Memory
from sqgraph.memory.portable import FORMAT, backup_database, export_memory, import_memory, read_export, rotate_backups

TARO = "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}"
HANAKO = "[[Rails|Service]] [[花子|Person]] {{Rails|担当|花子|2026-08-15}}"


class NamedEmbedder(FakeEmbedder):
    model = "fake-model"


@pytest.fixture
def src(tmp_path):
    mem = Memory(tmp_path / "src.db", FakeLLM(), NamedEmbedder(), default_group_id="me")
    mem.add_episode(TARO, occurred_at="2026-07-01 10:00", source="test", source_ref="m1", channel="pub")
    mem.add_episode(HANAKO, occurred_at="2026-08-15 10:00", source="test", source_ref="m2", channel="pub")
    yield mem
    mem.close()


def dump(mem, **kw):
    buf = io.StringIO()
    counts = export_memory(mem, buf, **kw)
    return buf.getvalue(), counts


def test_export_format(src):
    text, counts = dump(src)
    lines = [json.loads(x) for x in text.splitlines()]
    assert lines[0]["kind"] == "header" and lines[0]["format"] == FORMAT and lines[0]["group_id"] == "me"
    assert counts == {"group_id": "me", "entities": 3, "episodes": 2, "mentions": 4, "facts": 2, "with_embeddings": False}
    kinds = [x["kind"] for x in lines[1:]]
    assert kinds == sorted(kinds, key=["entity", "episode", "mention", "fact"].index)
    ent = next(x for x in lines if x["kind"] == "entity" and x["name"] == "Rails")
    assert ent["type"] == "Service" and "alias_keys" not in ent and "embedding" not in ent and "group_id" not in ent
    fact = next(x for x in lines if x["kind"] == "fact")
    assert "text_norm" not in fact and fact["valid_at"].endswith("Z")
    header, recs = read_export(text.splitlines())
    assert header["counts"]["facts"] == 2 and len(recs["fact"]) == 2


def test_round_trip_keeps_history_and_is_idempotent(src, tmp_path):
    text, _ = dump(src, with_embeddings=True)
    dst = Memory(tmp_path / "dst.db", None, NamedEmbedder(), default_group_id="other")
    res = import_memory(dst, text.splitlines(), group_id="other")
    assert res["entities"] == 3 and res["facts"] == 2 and res["episodes"] == 2 and res["vectors"] == "reused"
    assert dst.stats("other")["facts_invalidated"] == 1  # 太郎 -> 花子: superseded fact kept
    hits = dst.search("Rails 担当", include_invalid=True)
    assert {h.status for h in hits} == {"valid", "ended"}
    assert any(h.superseded_by for h in hits)
    assert dst.get_entity("Rails")["entity"]["name"] == "Rails"
    again = import_memory(dst, text.splitlines(), group_id="other")
    assert again["facts"] == 0 and again["skipped"] == {"entities": 3, "episodes": 2, "facts": 2}
    dst.close()


def test_import_reembeds_with_another_model_and_dry_run(src, tmp_path):
    text, _ = dump(src, with_embeddings=True)
    dst = Memory(tmp_path / "dst.db", None, FakeEmbedder(64), default_group_id="me")
    plan = import_memory(dst, text.splitlines(), dry_run=True)
    assert plan["dry_run"] and plan["facts"] == 2 and dst.stats("me")["nodes"]["Entity"] == 0
    res = import_memory(dst, text.splitlines())
    assert res["vectors"] == "computed" and res["group_id"] == "me"
    assert dst.search("Rails")
    dst.close()


def test_import_without_embedder_and_bad_input(src, tmp_path):
    text, _ = dump(src)
    dst = Memory(tmp_path / "dst.db", None, None, default_group_id="me")
    res = import_memory(dst, text.splitlines())
    assert res["vectors"] == "none" and "reembed" in res["warnings"][0]
    with pytest.raises(ValueError, match="header"):
        import_memory(dst, ['{"kind": "entity", "id": "x", "name": "y"}'])
    with pytest.raises(ValueError, match="line 2"):
        import_memory(dst, [text.splitlines()[0], "not json"])
    bad = [text.splitlines()[0], json.dumps({"kind": "fact", "id": "fact_000000000001", "fact": "x", "subject_id": "ent_missing", "valid_at": "2026-01-01T00:00:00Z"})]
    with pytest.raises(ValueError, match="unknown entity"):
        import_memory(dst, bad)
    dst.close()


def test_backup_and_rotation(src, tmp_path):
    dest = tmp_path / "bk" / "memory-20260101-000000.db"
    res = backup_database(src, dest)
    assert res["bytes"] > 0
    copy = Memory(dest, None, NamedEmbedder(), default_group_id="me", read_only=True)
    assert copy.stats("me")["facts_valid"] == 1 and copy.search("Rails")
    copy.close()
    with pytest.raises(FileExistsError):
        backup_database(src, dest)
    for i in range(2, 5):
        (dest.parent / f"memory-2026010{i}-000000.db").write_bytes(b"x")
    (dest.parent / "keep-me.db").write_bytes(b"x")
    removed = rotate_backups(dest.parent, 2)
    assert len(removed) == 2 and sorted(p.name for p in dest.parent.iterdir()) == [
        "keep-me.db", "memory-20260103-000000.db", "memory-20260104-000000.db"
    ]


def test_cli_export_import_backup(tmp_path, monkeypatch, capsys):
    from sqgraph.memory import cli

    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "openai")
    monkeypatch.setattr(cli, "make_providers", lambda cfg, *, need_llm, need_embedder: (FakeLLM() if need_llm else None, FakeEmbedder(), []))

    def run(*argv):
        capsys.readouterr()
        code = cli.main(list(argv))
        cap = capsys.readouterr()
        return code, cap.out, (json.loads(cap.err) if cap.err.strip() else None)

    assert run("add", TARO)[0] == 0
    out_file = tmp_path / "mem.jsonl"
    code, out, _ = run("export", str(out_file))
    assert code == 0 and json.loads(out)["facts"] == 1 and out_file.is_file()
    code, _, err = run("export", str(out_file))
    assert code == 2 and "--force" in err["error"]
    code, out, _ = run("export", "-")
    assert code == 0 and json.loads(out.splitlines()[0])["kind"] == "header"
    code, out, _ = run("import", str(out_file), "--into-group", "copy")
    assert code == 0 and json.loads(out)["skipped"]["facts"] == 1  # same database: the ids already exist
    code, out, _ = run("backup")
    res = json.loads(out)
    assert code == 0 and "backups" in res["path"] and res["removed"] == []
