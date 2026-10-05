"""sqgraph-memory CLI: in-process runs against a temporary home with fake LLM / embedder providers."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, cli

TARO = "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}"
HANAKO = "[[Rails|Service]] [[花子|Person]] {{Rails|担当|花子|2026-08-15}}"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mem-home"
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(h))
    for k in ("OPENAI_API_KEY", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP"):
        monkeypatch.delenv(k, raising=False)
    return h


@pytest.fixture
def fakes(monkeypatch, home):
    """Replace the OpenAI provider factory by deterministic fakes."""

    def make(cfg, *, need_llm, need_embedder):
        return (FakeLLM() if need_llm else None), (FakeEmbedder() if (need_llm or need_embedder) else None), []

    monkeypatch.setattr(cli, "make_providers", make)


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    capsys.readouterr()
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    out = json.loads(cap.out) if cap.out.strip() else None
    err = json.loads(cap.err) if cap.err.strip() else None
    return code, out, err


# --------------------------------------------------------------------------- init / config


def test_init_stores_key_from_stdin_without_echoing_it(capsys, home, monkeypatch):
    code, out, err = run(capsys, "init", "--api-key-stdin", "--group-id", "taro", "--tz", "UTC", stdin="sk-test-SECRET-123\n", monkeypatch=monkeypatch)
    assert code == 0 and err is None
    assert out["api_key"] == "stored" and out["group_id"] == "taro" and out["tz"] == "UTC"
    assert "SECRET" not in json.dumps(out)
    assert (home / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-test-SECRET-123\n"
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    assert cfg["group_id"] == "taro" and cfg["tz"] == "UTC" and cfg["db"] is None
    assert cfg["llm_model"] and cfg["embed_model"]
    assert out["db"] == str(home / "memory.db")

    # a second init keeps the stored key and settings
    code, out, _ = run(capsys, "init")
    assert code == 0 and out["api_key"] == "existing" and out["group_id"] == "taro"
    assert (home / ".env").read_text(encoding="utf-8").count("OPENAI_API_KEY") == 1


def test_init_key_from_env_and_missing_key_warning(capsys, home, monkeypatch):
    code, out, _ = run(capsys, "init", monkeypatch=monkeypatch, stdin="")
    assert code == 0 and out["api_key"] == "missing" and "warning" in out
    assert not (home / ".env").exists()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    code, out, _ = run(capsys, "init")
    assert out["api_key"] == "stored-from-env"
    assert "sk-from-env" in (home / ".env").read_text(encoding="utf-8")
    assert "sk-from-env" not in json.dumps(out)


def test_init_rejects_empty_stdin_key_and_bad_tz(capsys, home, monkeypatch):
    code, _, err = run(capsys, "init", "--api-key-stdin", stdin="\n", monkeypatch=monkeypatch)
    assert code == 2 and "stdin" in err["error"]
    code, _, err = run(capsys, "init", "--tz", "Not/AZone")
    assert code == 2 and "timezone" in err["error"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_init_private_permissions(capsys, home, monkeypatch):
    run(capsys, "init", "--api-key-stdin", stdin="sk-x\n", monkeypatch=monkeypatch)
    assert (home.stat().st_mode & 0o777) == 0o700
    assert ((home / ".env").stat().st_mode & 0o777) == 0o600
    assert ((home / "config.json").stat().st_mode & 0o777) == 0o600


def test_env_overrides_for_db_and_group(capsys, home, tmp_path, fakes, monkeypatch):
    other = tmp_path / "elsewhere" / "x.db"
    monkeypatch.setenv("SQGRAPH_MEMORY_DB", str(other))
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envgroup")
    code, out, _ = run(capsys, "add", TARO)
    assert code == 0 and out["group_id"] == "envgroup"
    assert other.is_file() and not (home / "memory.db").exists()


# --------------------------------------------------------------------------- add / search / entity (Japanese)


def test_add_search_entity_episodes_stats_japanese(capsys, home, fakes, monkeypatch):
    code, out, err = run(capsys, "add", TARO, "--source", "claude-code", "--occurred-at", "2026-07-01 10:00", "--source-ref", "n1")
    assert code == 0 and err is None
    assert out["facts_created"] == 1 and out["new_entities"] >= 2 and not out["already_stored"] and out["episode_id"]

    code, again, _ = run(capsys, "add", TARO, "--source", "claude-code", "--source-ref", "n1")
    assert again["already_stored"] is True  # idempotent per (source, source-ref)

    code, out, _ = run(capsys, "add", "-", "--source", "codex", "--occurred-at", "2026-08-15 10:00", stdin=HANAKO, monkeypatch=monkeypatch)
    assert code == 0 and out["facts_created"] == 1

    code, out, _ = run(capsys, "search", "Rails 担当", "--history")
    assert code == 0 and out["group_id"] == "me" and out["count"] >= 2
    owners = {(f["object"]["name"], f["status"]) for f in out["facts"] if f["relation"] == "担当"}
    assert owners == {("花子", "valid"), ("太郎", "ended")}  # the later owner superseded the earlier one

    code, out, _ = run(capsys, "search", "Rails 担当", "--as-of", "2026-07-20")
    assert [f["object"]["name"] for f in out["facts"] if f["relation"] == "担当"] == ["太郎"]

    code, out, _ = run(capsys, "entity", "Rails")
    assert out["found"] is True and out["entity"]["name"] == "Rails" and len(out["timeline"]) == 2
    code, out, _ = run(capsys, "entity", "存在しない人")
    assert code == 0 and out["found"] is False and "存在しない人" in out["message"]

    code, out, _ = run(capsys, "entities", "Rails", "--limit", "3")
    assert out["entities"][0]["name"] == "Rails"

    code, out, _ = run(capsys, "episodes", "--entity", "花子", "--limit", "5")
    assert out["count"] == 1 and "花子" in out["episodes"][0]["content"] and out["episodes"][0]["source"] == "codex"

    code, out, _ = run(capsys, "stats")
    assert out["counts"]["nodes"]["Episode"] == 2 and out["counts"]["facts_valid"] >= 1
    assert out["db"] == str(home / "memory.db") and out["db_bytes"] > 0


def test_search_includes_scheduled_facts_by_default(capsys, home, fakes):
    run(capsys, "add", "[[太郎|Person]] [[Osaka|Place]] {{太郎|引っ越す|Osaka|2099-01-01}}", "--occurred-at", "2026-07-01 10:00")
    _, out, _ = run(capsys, "search", "太郎 引っ越す")
    assert any(f["status"] == "scheduled" for f in out["facts"])


def test_output_is_valid_utf8_json_unescaped_and_compact(capsys, home, fakes):
    run(capsys, "add", TARO, "--occurred-at", "2026-07-01 10:00")
    capsys.readouterr()
    assert cli.main(["search", "太郎", "--compact"]) == 0
    raw = capsys.readouterr().out
    assert raw.count("\n") == 1 and "太郎" in raw and "\\u592a" not in raw
    assert cli.main(["search", "太郎"]) == 0
    assert capsys.readouterr().out.count("\n") > 5  # indented by default


def test_errors_are_json_on_stderr(capsys, home, monkeypatch):
    code, out, err = run(capsys, "add", "覚えておいてください、これは長いテキストです")  # real providers, no key
    assert code == 1 and out is None and "OPENAI_API_KEY" in err["error"]
    code, out, err = run(capsys, "add", "-", stdin="   ", monkeypatch=monkeypatch)
    assert code == 2 and "empty" in err["error"]
    code, out, err = run(capsys, "bogus-command")
    assert code == 2 and out is None and "error" in err
    code, out, err = run(capsys, "search")  # missing argument
    assert code == 2 and "error" in err


def test_search_without_api_key_degrades_with_warning(capsys, home):
    code, out, err = run(capsys, "search", "太郎")
    assert code == 0 and out["count"] == 0 and "OPENAI_API_KEY" in out["warning"]


def test_stdout_is_utf8_even_when_console_codec_is_not(tmp_path):
    """On Windows the default codec (cp932) garbles Japanese: the CLI must force UTF-8 on its streams."""
    env = {**os.environ, "SQGRAPH_MEMORY_HOME": str(tmp_path / "h"), "PYTHONIOENCODING": "cp932", "PYTHONPATH": str(Path(cli.__file__).parents[2])}
    env.pop("OPENAI_API_KEY", None)
    p = subprocess.run([sys.executable, "-m", "sqgraph.memory", "entity", "太郎さん"], env=env, capture_output=True, timeout=120)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.decode("utf-8"))
    assert "太郎さん" in out["message"]


# --------------------------------------------------------------------------- forget / purge (contract of the parallel API)


def _first_fact_id(capsys, query="Rails"):
    _, out, _ = run(capsys, "search", query, "--history")
    return out["facts"][0]["id"]


def test_forget_requires_exactly_one_target(capsys, home, fakes):
    code, _, err = run(capsys, "forget")
    assert code == 2 and "exactly one" in err["error"]
    code, _, err = run(capsys, "forget", "--fact", "x", "--entity", "Rails")
    assert code == 2
    code, _, err = run(capsys, "forget", "--episode", "e1", "--keep-episodes")
    assert code == 2


def test_forget_by_source_ref_dry_run_then_real(capsys, home, fakes):
    run(capsys, "add", TARO, "--source", "claude-code", "--source-ref", "n1", "--occurred-at", "2026-07-01 10:00")
    code, out, _ = run(capsys, "forget", "--source-ref", "n1", "--source", "claude-code", "--dry-run")
    assert code == 0 and out["dry_run"] is True and len(out["episodes_deleted"]) == 1
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 1  # nothing deleted yet
    code, out, _ = run(capsys, "forget", "--source-ref", "n1", "--source", "claude-code")
    assert code == 0 and out["dry_run"] is False and len(out["episodes_deleted"]) == 1
    assert len(out["facts_deleted"]) == 1
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 0
    assert run(capsys, "search", "Rails 担当", "--history")[1]["count"] == 0


def test_forget_fact_entity_and_range(capsys, home, fakes):
    run(capsys, "add", TARO, "--source", "cli", "--channel", "work", "--occurred-at", "2026-07-01 10:00")
    run(capsys, "add", HANAKO, "--source", "codex", "--channel", "personal", "--occurred-at", "2026-08-15 10:00")

    fid = _first_fact_id(capsys)
    code, out, _ = run(capsys, "forget", "--fact", fid, "--dry-run")
    assert code == 0 and out["dry_run"] is True and len(out["facts_deleted"]) >= 1

    code, out, _ = run(capsys, "forget", "--channel", "work", "--dry-run")
    assert code == 0 and out["dry_run"] is True and len(out["episodes_deleted"]) == 1
    code, out, _ = run(capsys, "forget", "--source", "codex")  # source alone = range target
    assert code == 0 and len(out["episodes_deleted"]) == 1
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 1

    code, out, _ = run(capsys, "forget", "--entity", "太郎", "--keep-episodes")
    assert code == 0 and len(out["episodes_deleted"]) == 0
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 1


def test_purge_requires_yes(capsys, home, fakes):
    run(capsys, "add", TARO, "--occurred-at", "2026-07-01 10:00")
    code, out, err = run(capsys, "purge")
    assert code == 2 and "--yes" in err["error"]
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 1
    code, out, _ = run(capsys, "purge", "--yes", "--dry-run")
    assert code == 0 and out["dry_run"] is True and len(out["episodes_deleted"]) == 1
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 1
    code, out, _ = run(capsys, "purge", "--yes")
    assert code == 0 and out["dry_run"] is False
    counts = run(capsys, "stats")[1]["counts"]
    assert counts["nodes"] == {"Episode": 0, "Entity": 0} and counts["facts_valid"] == 0


# --------------------------------------------------------------------------- mcp / setup / doctor


def test_mcp_argv_built_from_config(capsys, home, monkeypatch):
    (home).mkdir()
    (home / "config.json").write_text(
        json.dumps({"group_id": "taro", "tz": "UTC", "llm_model": "m-llm", "embed_model": "m-emb"}), encoding="utf-8"
    )
    from sqgraph.memory import mcp as mcp_mod

    seen: list[list[str]] = []
    monkeypatch.setattr(mcp_mod, "main", lambda argv=None: seen.append(list(argv)))
    assert cli.main(["mcp"]) == 0
    assert capsys.readouterr().out == ""  # stdout belongs to the MCP protocol
    argv = seen[0]
    pairs = dict(zip(argv[::2], argv[1::2]))
    assert pairs["--db"] == str(home / "memory.db") and pairs["--group-id"] == "taro"
    assert pairs["--env-file"] == str(home / ".env") and pairs["--tz"] == "UTC"
    assert pairs["--llm-model"] == "m-llm" and pairs["--embed-model"] == "m-emb" and pairs["--transport"] == "stdio"
    assert "--read-only" not in argv and "--host" not in argv

    seen.clear()
    assert cli.main(["mcp", "--transport", "streamable-http", "--port", "9000", "--read-only"]) == 0
    argv = seen[0]
    assert "--read-only" in argv and argv[argv.index("--port") + 1] == "9000" and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert home.is_dir()


def test_setup_and_uninstall_via_cli(capsys, home, tmp_path, monkeypatch):
    fake_home = tmp_path / "fakeuser"
    fake_home.mkdir()
    monkeypatch.setattr(cli, "user_home", lambda: fake_home)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli.setup_agents, "default_runner", lambda argv: pytest.fail(f"must not run {argv}")
    )

    code, out, _ = run(capsys, "setup", "--agents", "cursor,codex", "--command", "my-launcher mcp")
    assert code == 0 and out["command"] == ["my-launcher", "mcp"]
    assert {a["agent"]: a["action"] for a in out["agents"]} == {"cursor": "registered", "codex": "registered"}
    assert "note_init" in out  # no API key configured yet
    mcp = json.loads((fake_home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    # SQGRAPH_MEMORY_HOME is set (by the fixture): it is forwarded so agents use the same home
    forwarded = {"SQGRAPH_MEMORY_HOME": os.path.abspath(str(home))}
    assert mcp["mcpServers"]["sqgraph-memory"] == {"command": "my-launcher", "args": ["mcp"], "env": forwarded}
    assert out["env"] == forwarded and all(a["env"] == forwarded for a in out["agents"])
    assert "SQGRAPH_MEMORY_HOME = " in (fake_home / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert (fake_home / ".codex" / "config.toml").is_file()
    assert (fake_home / ".agents" / "skills" / "sqgraph-memory" / "SKILL.md").is_file()

    code, out, _ = run(capsys, "setup", "--agents", "cursor,codex", "--uninstall")
    assert {a["agent"]: a["action"] for a in out["agents"]} == {"cursor": "removed", "codex": "removed"}
    assert not (fake_home / ".agents" / "skills" / "sqgraph-memory").exists()

    code, _, err = run(capsys, "setup", "--agents", "vim")
    assert code == 2 and "unknown agent" in err["error"]

    # nothing detected in the fake home and no explicit selection
    code, out, _ = run(capsys, "setup", "--dry-run")
    assert code == 0 and out["dry_run"] is True


def test_doctor_report_structure(capsys, home, tmp_path, monkeypatch):
    fake_home = tmp_path / "fakeuser"
    fake_home.mkdir()
    monkeypatch.setattr(cli, "user_home", lambda: fake_home)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    code, out, _ = run(capsys, "doctor")
    names = {c["name"]: c for c in out["checks"]}
    for n in ("python", "sqlite3", "enable_load_extension", "sqlite_vec", "import openai", "import mcp", "home", "api_key", "database"):
        assert n in names, n
    assert names["python"]["status"] == "ok" and names["sqlite_vec"]["status"] == "ok"
    assert names["home"]["status"] == "warn" and names["api_key"]["status"] == "warn"
    assert all(c["status"] in ("ok", "warn", "fail", "skip") for c in out["checks"])
    assert code == (0 if out["ok"] else 1)
    assert "sk-" not in json.dumps(out)

    # after init + a write, the database check reports counts
    run(capsys, "init", "--api-key-stdin", stdin="sk-doctor\n", monkeypatch=monkeypatch)
    monkeypatch.setattr(cli, "make_providers", lambda cfg, *, need_llm, need_embedder: (FakeLLM() if need_llm else None, FakeEmbedder(), []))
    run(capsys, "add", TARO, "--occurred-at", "2026-07-01 10:00")
    code, out, _ = run(capsys, "doctor", "--online")
    names = {c["name"]: c for c in out["checks"]}
    assert names["api_key"]["status"] == "ok" and names["home"]["status"] == "ok"
    assert names["database"]["status"] == "ok" and "1 episodes" in names["database"]["detail"]
    assert names["online embedding"]["status"] == "ok"
