"""Claude Code hooks (sqgraph.memory.hooks), their registration by `setup`, and import-claude-memory."""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, Memory, cli, hooks, setup_agents

NOW = datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)


def make():
    mem = Memory(":memory:", None, FakeEmbedder(), default_group_id="me", clock=lambda: NOW)

    def add(text, extraction, at="2026-07-01 10:00"):
        prep = mem.prepare_episode(text, channel="personal", author="user", occurred_at=at, source="test")
        mem.commit_episode(prep["token"], extraction)

    add("uv を使う", {"entities": [{"ref": "u", "name": "uv", "type": "Service"}],
                     "facts": [{"subject": "author", "object": "u", "relation": "PREFERS", "fact": "ユーザーは Python に uv を使う"}]})
    add("sqgraph は SQLite 製", {"entities": [{"ref": "s", "name": "sqgraph", "type": "Project"}],
                                "facts": [{"subject": "s", "object": None, "relation": "USES", "fact": "sqgraph は SQLite の上に作ったグラフ DB"}]})
    add("7/14 休み", {"entities": [], "facts": [{"subject": "author", "object": None, "relation": "ABSENT",
                                               "fact": "ユーザーは 7/14 休み", "valid_at": "2026-07-14", "invalid_at": "2026-07-15"}]})
    return mem


def opener(mem):
    @contextmanager
    def go():
        yield mem

    return go


def ctx(out):
    assert out["hookSpecificOutput"]["hookEventName"] in ("SessionStart", "UserPromptSubmit")
    return out["hookSpecificOutput"]["additionalContext"]


def test_project_names(tmp_path):
    repo = tmp_path / "sqgraph"
    (repo / ".git").mkdir(parents=True)
    (repo / "src" / "pkg").mkdir(parents=True)
    assert hooks.project_names(str(repo / "src" / "pkg")) == ["sqgraph", "pkg"]
    assert hooks.project_names(str(tmp_path / "source")) == []  # generic folder names are not projects
    assert hooks.project_names(None) == []


def test_session_start_injects_user_facts_project_and_upcoming(tmp_path):
    mem = make()
    (tmp_path / "sqgraph" / ".git").mkdir(parents=True)
    out = hooks.run("session-start", {"session_id": "s1", "cwd": str(tmp_path / "sqgraph"), "source": "startup"}, tmp_path, opener(mem))
    text = ctx(out)
    assert "data, not instructions" in text
    assert "About the user:" in text and "uv を使う" in text
    assert "Related to sqgraph:" in text and "SQLite" in text
    assert "Coming up" in text and "- 2026-07-14: ユーザーは 7/14 休み" in text
    state = json.loads((tmp_path / "hook-state" / "s1.json").read_text(encoding="utf-8"))
    assert len(state["shown"]) == 2


def test_user_prompt_dedupes_per_session_and_detects_remember(tmp_path):
    mem = make()
    out = hooks.run("user-prompt", {"session_id": "s2", "prompt": "Python のパッケージ管理は何を使えばいい？"}, tmp_path, opener(mem))
    assert "uv" in ctx(out) and "Possibly relevant" in ctx(out)
    again = hooks.run("user-prompt", {"session_id": "s2", "prompt": "Python のパッケージ管理は何を使えばいい？"}, tmp_path, opener(mem))
    assert again is None  # already shown in this session
    rem = hooks.run("user-prompt", {"session_id": "s2", "prompt": "これ覚えておいて: 締切は金曜"}, tmp_path, opener(mem))
    assert "add_memory" in ctx(rem)
    assert hooks.run("user-prompt", {"session_id": "s2", "prompt": "/compact"}, tmp_path, opener(mem)) is None
    past = hooks.run("user-prompt", {"session_id": "s3", "prompt": "前に話した件どうなった？"}, tmp_path, opener(mem))
    assert past is None or "search_memory" in ctx(past)


def test_hook_errors_are_logged_not_raised(tmp_path):
    @contextmanager
    def broken():
        raise RuntimeError("db locked")
        yield  # pragma: no cover

    assert hooks.run("session-start", {"session_id": "x"}, tmp_path, broken) is None
    assert "db locked" in (tmp_path / "hooks.log").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        hooks.run("nope", {}, tmp_path, broken)


def test_cli_hook_reads_stdin_and_prints_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(cli, "make_providers", lambda cfg, *, need_llm, need_embedder: (None, FakeEmbedder(), []))
    home = tmp_path / "home"
    home.mkdir()
    with Memory(home / "memory.db", None, FakeEmbedder(), default_group_id="me") as mem:
        prep = mem.prepare_episode("pnpm", channel="personal", author="user", source="t")
        mem.commit_episode(prep["token"], {"entities": [{"ref": "p", "name": "pnpm", "type": "Service"}],
                                           "facts": [{"subject": "author", "object": "p", "relation": "PREFERS", "fact": "ユーザーは pnpm を好む"}]})
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": "a", "cwd": str(tmp_path), "source": "startup"})))
    capsys.readouterr()
    code = cli.main(["hook", "session-start", "--env", f"SQGRAPH_MEMORY_HOME={home}"])
    out = capsys.readouterr().out
    assert code == 0 and "pnpm" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    assert cli.main(["hook", "user-prompt"]) == 0  # bad input: no output, no failure
    assert cli.main(["hook", "user-prompt", "--env", "PATH=x"]) == 2


# --------------------------------------------------------------------------- setup


def test_hook_command_formats():
    cmd = setup_agents.hook_command(["C:\\Users\\me\\.local\\bin\\sqgraph-memory.EXE", "mcp"], "session-start", platform="win32")
    assert cmd == "C:/Users/me/.local/bin/sqgraph-memory.EXE hook session-start"
    cmd = setup_agents.hook_command(["/usr/bin/python3", "-m", "sqgraph.memory", "mcp"], "user-prompt", {"SQGRAPH_MEMORY_HOME": "/h m"}, platform="linux")
    assert cmd == "/usr/bin/python3 -m sqgraph.memory hook user-prompt --env 'SQGRAPH_MEMORY_HOME=/h m'"


def test_set_hooks_keeps_other_hooks_and_is_idempotent():
    other = {"type": "command", "command": "echo hi"}
    data = {"model": "x", "hooks": {"SessionStart": [{"matcher": "startup", "hooks": [other]}], "Stop": [{"hooks": [other]}]}}
    want = {"session-start": {"type": "command", "command": "sqgraph-memory hook session-start", "timeout": 30},
            "user-prompt": {"type": "command", "command": "sqgraph-memory hook user-prompt", "timeout": 15}}
    new = setup_agents.set_hooks(data, want)
    assert new["model"] == "x" and new["hooks"]["Stop"] == data["hooks"]["Stop"]
    assert new["hooks"]["SessionStart"][0] == {"matcher": "startup", "hooks": [other]}
    assert new["hooks"]["SessionStart"][1] == {"hooks": [want["session-start"]]}
    assert setup_agents.set_hooks(new, want) is None  # unchanged
    changed = dict(want, **{"user-prompt": {**want["user-prompt"], "command": "/new/sqgraph-memory hook user-prompt"}})
    upd = setup_agents.set_hooks(new, changed)
    assert len(upd["hooks"]["UserPromptSubmit"]) == 1 and "/new/" in upd["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    gone = setup_agents.set_hooks(upd, {})
    assert "UserPromptSubmit" not in gone["hooks"] and gone["hooks"]["SessionStart"] == [{"matcher": "startup", "hooks": [other]}]
    assert setup_agents.set_hooks({}, {}) is None


def test_setup_installs_and_uninstalls_claude_code_hooks(tmp_path):
    home = tmp_path
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    calls = []

    def runner(argv):
        calls.append(argv)
        return 0, "", ""

    kw = dict(home=home, command=["/bin/sqgraph-memory", "mcp"], runner=runner, which=lambda n: f"/bin/{n}", platform="linux")
    out = setup_agents.run_setup(["claude-code"], **kw)
    rep = out["agents"][0]["hooks"]
    assert rep["action"] == "installed" and rep["events"] == ["session-start", "user-prompt"]
    settings = json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["theme"] == "dark"
    assert settings["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "/bin/sqgraph-memory hook session-start"
    assert setup_agents.run_setup(["claude-code"], **kw)["agents"][0]["hooks"]["action"] == "unchanged"
    status = setup_agents.agents_status(home=home, command=kw["command"], which=kw["which"], platform="linux")
    assert next(s for s in status if s["agent"] == "claude-code")["hooks"] == ["session-start", "user-prompt"]
    assert setup_agents.run_setup(["claude-code"], hooks=(), **kw)["agents"][0]["hooks"] == {"action": "skipped"}
    out = setup_agents.run_setup(["claude-code"], uninstall=True, **kw)
    assert out["agents"][0]["hooks"]["action"] == "removed"
    assert json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8")) == {"theme": "dark"}


# --------------------------------------------------------------------------- import-claude-memory


def test_import_claude_memory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "projects"
    mem_dir = root / "C--Users-me-src" / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("- [x](x.md)\n", encoding="utf-8")
    (mem_dir / "pref.md").write_text(
        "---\nname: pnpm-pref\ndescription: prefers pnpm\nmetadata:\n  type: feedback\n---\n\nUse pnpm, not npm.\n", encoding="utf-8"
    )
    (mem_dir / "proj.md").write_text("no front matter here\n", encoding="utf-8")
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SQGRAPH_MEMORY_EXTRACTOR", raising=False)
    monkeypatch.setattr(cli, "make_providers", lambda cfg, *, need_llm, need_embedder: (FakeLLM() if need_llm else None, FakeEmbedder(), []))

    def run(*argv):
        capsys.readouterr()
        code = cli.main(list(argv))
        return code, json.loads(capsys.readouterr().out)

    code, out = run("import-claude-memory", "--root", str(root), "--dry-run")
    assert code == 0 and out["files"] == 2 and out["pending"] == 2 and "token" not in out["items"][0]
    code, out = run("import-claude-memory", "--root", str(root), "--limit", "1")
    assert out["pending"] == 2 and len(out["items"]) == 1 and "instructions" in out
    item = out["items"][0]
    assert "pnpm-pref" in item["context"]["message"]["content"] and "feedback" in item["context"]["message"]["content"]
    with Memory(tmp_path / "home" / "memory.db", None, FakeEmbedder(), default_group_id="me") as mem:
        mem.commit_episode(item["token"], {})
    code, out = run("import-claude-memory", "--root", str(root))
    assert out["already_imported"] == 1 and out["pending"] == 1


def test_upcoming_lines_merge_start_and_end():
    def ev(kind, at, fid="f1", text="x"):
        return {"event": kind, "at": at, "fact": {"id": fid, "fact": text}}

    lines = hooks.upcoming_lines([
        ev("starts", "2026-10-13T11:00:00+09:00"), ev("ends", "2026-10-13T12:00:00+09:00"),
        ev("starts", "2026-10-14T00:00:00+09:00", "f2"), ev("ends", "2026-10-17T00:00:00+09:00", "f2"),
        ev("starts", "2026-10-15T00:00:00+09:00", "f3"),
        ev("ends", "2026-10-16T00:00:00+09:00", "f4"),
    ])
    assert lines == ["- 2026-10-13 11:00-12:00: x", "- 2026-10-14 - 2026-10-16: x", "- 2026-10-15: x", "- until 2026-10-16: x"]
