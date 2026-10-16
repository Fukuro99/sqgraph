"""Separate memories per agent: ``agent_groups`` in the config, ``--agent`` on the CLI, ``groups``, ``setup --group``."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, cli, hooks
from sqgraph.memory import setup_agents as sa
from sqgraph.memory.config import (
    MemoryConfig,
    validate_agent_name,
    validate_group_id,
)

TARO = "[[Rails|Service]] [[太郎|Person]] {{Rails|担当|太郎|2026-07-01}}"
HANAKO = "[[Rails|Service]] [[花子|Person]] {{Rails|担当|花子|2026-08-15}}"
BOB = "[[Bob|Person]] [[Go|Service]] {{Bob|使う|Go|2026-07-01}}"
ENV_KEYS = ("OPENAI_API_KEY", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP", "SQGRAPH_MEMORY_EXTRACTOR")


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mem-home"
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(h))
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    return h


@pytest.fixture
def fakes(monkeypatch, home):
    """Deterministic fake providers and the one-shot ``openai`` extractor (``add`` stores in one step)."""
    monkeypatch.setenv("SQGRAPH_MEMORY_EXTRACTOR", "openai")

    def make(cfg, *, need_llm, need_embedder):
        return (FakeLLM() if need_llm else None), (FakeEmbedder() if (need_llm or need_embedder) else None), []

    monkeypatch.setattr(cli, "make_providers", make)


@pytest.fixture
def agent_mode(monkeypatch, home):
    """Fake embedder only: the default ``agent`` extractor (``add`` prints a token, ``commit`` stores)."""
    monkeypatch.setattr(cli, "make_providers", lambda cfg, *, need_llm, need_embedder: (None, FakeEmbedder(), []))


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    fh = tmp_path / "fakeuser"
    fh.mkdir()
    monkeypatch.setattr(cli, "user_home", lambda: fh)
    monkeypatch.setenv("APPDATA", str(fh / "AppData" / "Roaming"))  # Claude Desktop's config dir must not be the real one
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.setup_agents, "default_runner", lambda argv: pytest.fail(f"must not run {argv}"))
    return fh


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    capsys.readouterr()
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    out = json.loads(cap.out) if cap.out.strip() else None
    err = json.loads(cap.err) if cap.err.strip() else None
    return code, out, err


def write_config(home: Path, **data) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(json.dumps(data), encoding="utf-8")


def stored_config(home: Path) -> dict:
    return json.loads((home / "config.json").read_text(encoding="utf-8"))


def object_names(capsys, *argv) -> set[str]:
    return {f["object"]["name"] for f in run(capsys, "search", "Rails 担当", "--history", *argv)[1]["facts"] if f["relation"] == "担当"}


# --------------------------------------------------------------------------- config


def test_agent_groups_default_empty_and_roundtrip(home):
    cfg = MemoryConfig.load()
    assert cfg.agent_groups == {} and cfg.to_json()["agent_groups"] == {}
    cfg.agent_groups = {"codex": "work", "claude-code": "me.private"}
    cfg.save()
    assert stored_config(home)["agent_groups"] == {"codex": "work", "claude-code": "me.private"}
    assert MemoryConfig.load().agent_groups == {"codex": "work", "claude-code": "me.private"}


def test_agent_groups_loaded_leniently(home):
    write_config(
        home,
        agent_groups={"codex": "work", "Bad Name": "x", "cursor": "", "opencode": 5, "x": ["a"], "": "g", "claude-code": "-nope"},
    )
    assert MemoryConfig.load().agent_groups == {"codex": "work"}
    for junk in ([], "codex=work", 3, None):
        write_config(home, agent_groups=junk)
        assert MemoryConfig.load().agent_groups == {}


def test_name_validation():
    for ok in ("codex", "claude-code", "my_client2", "0x"):
        assert validate_agent_name(ok) == ok
    for bad in ("", "Codex", "-x", "a b", "a/b", None, 3):
        with pytest.raises(ValueError):
            validate_agent_name(bad)  # type: ignore[arg-type]
    for ok in ("work", "me", "Team-A", "a.b:c_d", "9", "x" * 64):
        assert validate_group_id(ok) == ok
    for bad in ("", "-w", ".w", "w w", "w/x", "é", "x" * 65, None, 3):
        with pytest.raises(ValueError):
            validate_group_id(bad)  # type: ignore[arg-type]


def test_group_for_precedence_env_over_mapping_over_default(home, monkeypatch):
    write_config(home, group_id="taro", agent_groups={"codex": "work"})
    cfg = MemoryConfig.load()
    assert cfg.group_for("codex") == "work"  # the mapping beats the default group
    assert cfg.group_for("cursor") == "taro" and cfg.group_for(None) == "taro" and cfg.group_for("") == "taro"
    assert cfg.for_agent("codex").group_id == "work" and cfg.group_id == "taro"  # a copy

    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    cfg = MemoryConfig.load()
    assert cfg.group_id == "envg"
    assert cfg.group_for("codex") == "envg" and cfg.group_for("cursor") == "envg"  # env beats the mapping
    assert cfg.for_agent("codex").group_for("codex") == "envg"
    assert cfg.to_json()["group_id"] == "envg" and "group_from_env" not in cfg.to_json()
    # without the env (what `config` persists) the mapping is intact
    assert MemoryConfig.load(apply_env=False).group_for("codex") == "work"


# --------------------------------------------------------------------------- CLI: --agent


def test_add_search_stats_resolve_the_group_of_the_agent(capsys, home, fakes):
    write_config(home, agent_groups={"codex": "work"})
    code, out, _ = run(capsys, "add", TARO, "--source", "codex", "--agent", "codex", "--occurred-at", "2026-07-01 10:00")
    assert code == 0 and out["group_id"] == "work"
    code, out, _ = run(capsys, "add", HANAKO, "--source", "cli", "--occurred-at", "2026-08-15 10:00")
    assert code == 0 and out["group_id"] == "me"

    _, out, _ = run(capsys, "search", "Rails 担当", "--agent", "codex", "--history")
    assert out["group_id"] == "work"
    assert object_names(capsys, "--agent", "codex") == {"太郎"}
    _, out, _ = run(capsys, "search", "Rails 担当", "--history")
    assert out["group_id"] == "me"
    assert object_names(capsys) == {"花子"}

    assert run(capsys, "stats", "--agent", "codex")[1]["group_id"] == "work"
    assert run(capsys, "stats", "--agent", "codex")[1]["counts"]["nodes"]["Episode"] == 1
    assert run(capsys, "stats")[1]["group_id"] == "me"
    assert run(capsys, "stats", "--agent", "cursor")[1]["group_id"] == "me"  # no assignment: the default group
    assert run(capsys, "entities", "Rails", "--agent", "codex")[1]["group_id"] == "work"
    assert run(capsys, "episodes", "--agent", "codex")[1]["group_id"] == "work"


def test_agents_in_different_groups_cannot_see_each_other_but_unmapped_agents_share(capsys, home, fakes):
    write_config(home, agent_groups={"codex": "work", "opencode": "play"})
    run(capsys, "add", TARO, "--agent", "codex", "--occurred-at", "2026-07-01 10:00")
    run(capsys, "add", HANAKO, "--agent", "opencode", "--occurred-at", "2026-08-15 10:00")
    run(capsys, "add", BOB, "--agent", "claude-code", "--occurred-at", "2026-07-01 10:00")

    assert object_names(capsys, "--agent", "codex") == {"太郎"}
    assert object_names(capsys, "--agent", "opencode") == {"花子"}
    assert object_names(capsys, "--agent", "cursor") == set()
    assert run(capsys, "entity", "花子", "--agent", "codex")[1]["found"] is False
    assert run(capsys, "entity", "花子", "--agent", "opencode")[1]["found"] is True
    # claude-code, cursor and "no agent" are all in the default group: they see each other's notes, not the others'
    assert run(capsys, "entity", "Bob", "--agent", "cursor")[1]["found"] is True
    assert run(capsys, "entity", "Bob")[1]["found"] is True
    assert run(capsys, "entity", "太郎", "--agent", "cursor")[1]["found"] is False

    # forget / purge stay inside the group of the agent
    code, out, _ = run(capsys, "purge", "--yes", "--agent", "codex")
    assert code == 0 and out["dry_run"] is False
    assert run(capsys, "stats", "--agent", "codex")[1]["counts"]["nodes"]["Episode"] == 0
    assert run(capsys, "stats", "--agent", "opencode")[1]["counts"]["nodes"]["Episode"] == 1
    assert run(capsys, "stats")[1]["counts"]["nodes"]["Episode"] == 1


def test_env_group_overrides_the_agent_mapping(capsys, home, fakes, monkeypatch):
    write_config(home, agent_groups={"codex": "work"})
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    code, out, _ = run(capsys, "add", TARO, "--agent", "codex", "--occurred-at", "2026-07-01 10:00")
    assert out["group_id"] == "envg"
    assert run(capsys, "stats", "--agent", "codex")[1]["group_id"] == "envg"


def test_commit_token_must_be_committed_in_the_same_group(capsys, home, agent_mode, monkeypatch):
    write_config(home, agent_groups={"codex": "work"})
    code, out, _ = run(capsys, "add", "太郎はRailsの担当になった", "--agent", "codex", "--occurred-at", "2026-07-01 10:00")
    assert code == 0 and out["group_id"] == "work" and out["token"]
    token = out["token"]
    code, _, err = run(capsys, "commit", "--token", token, stdin="{}", monkeypatch=monkeypatch)  # forgot --agent
    assert code == 2 and "work" in err["error"] and "--agent" in err["error"]
    code, out, err = run(capsys, "commit", "--token", token, "--agent", "codex", stdin="{}", monkeypatch=monkeypatch)
    assert code == 0 and err is None and out["group_id"] == "work"
    assert run(capsys, "episodes", "--agent", "codex")[1]["count"] == 1
    assert run(capsys, "episodes")[1]["count"] == 0


def test_hook_uses_the_group_of_its_agent(home, fakes, monkeypatch):
    write_config(home, agent_groups={"codex": "work", "claude-code": "cc"})
    seen: list[tuple[str | None, str]] = []

    def fake_run(event, payload, home_, memory, agent=None):
        with memory() as mem:
            seen.append((agent, mem.default_group_id))

    monkeypatch.setattr(hooks, "run", fake_run)
    for argv in (["hook", "session-start", "--agent", "codex"], ["hook", "session-start"], ["hook", "user-prompt", "--agent", "cursor"]):
        monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
        assert cli.main(argv) == 0
    assert seen == [("codex", "work"), ("claude-code", "cc"), ("cursor", "me")]  # the hook default agent is claude-code


def test_mcp_argv_uses_the_mapped_group(home, monkeypatch):
    write_config(home, group_id="taro", agent_groups={"codex": "work"})
    from sqgraph.memory import mcp as mcp_mod

    seen: list[list[str]] = []
    monkeypatch.setattr(mcp_mod, "main", lambda argv=None: seen.append(list(argv)))

    def group_of(argv):
        return argv[argv.index("--group-id") + 1]

    assert cli.main(["mcp", "--agent", "codex"]) == 0 and group_of(seen[-1]) == "work"
    assert cli.main(["mcp", "--agent", "cursor"]) == 0 and group_of(seen[-1]) == "taro"
    assert cli.main(["mcp"]) == 0 and group_of(seen[-1]) == "taro"
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    assert cli.main(["mcp", "--agent", "codex"]) == 0 and group_of(seen[-1]) == "envg"


def test_agent_option_is_not_offered_by_commands_without_a_group(capsys, home):
    for argv in (["init", "--agent", "x"], ["doctor", "--agent", "x"], ["groups", "--agent", "x"]):
        code, _, err = run(capsys, *argv)
        assert code == 2 and "error" in err


# --------------------------------------------------------------------------- CLI: config --agent-group


def test_config_agent_group_add_and_remove(capsys, home):
    code, out, _ = run(capsys, "config")
    assert code == 0 and out["agent_groups"] == {}
    code, out, _ = run(capsys, "config", "--agent-group", "codex=work", "--agent-group", "opencode=play")
    assert code == 0 and out["agent_groups"] == {"codex": "work", "opencode": "play"} and out["changed"] == ["agent_groups"]
    assert stored_config(home)["agent_groups"] == {"codex": "work", "opencode": "play"}
    assert "changed" not in run(capsys, "config", "--agent-group", "codex=work")[1]  # same value: no change
    # AGENT= and AGENT=default remove the assignment
    assert run(capsys, "config", "--agent-group", "codex=")[1]["agent_groups"] == {"opencode": "play"}
    assert run(capsys, "config", "--agent-group", "opencode=default")[1]["agent_groups"] == {}
    assert stored_config(home)["agent_groups"] == {}
    assert run(capsys, "config", "--agent-group", "never-set=default")[1]["agent_groups"] == {}


def test_config_agent_group_rejects_bad_input_and_saves_nothing(capsys, home):
    run(capsys, "config", "--agent-group", "codex=work")
    for bad in ("codex", "Codex=work", "codex=-x", "codex=a b", "=work", "codex=" + "x" * 65):
        code, out, err = run(capsys, "config", "--agent-group", bad)
        assert code == 2 and out is None and "--agent-group" in err["error"], bad
    code, _, _ = run(capsys, "config", "--agent-group", "cursor=ok", "--agent-group", "bad name=x")
    assert code == 2
    assert stored_config(home)["agent_groups"] == {"codex": "work"}  # the valid half of a failed call is not saved


def test_config_does_not_persist_the_env_group(capsys, home, monkeypatch):
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    code, out, _ = run(capsys, "config", "--agent-group", "codex=work")
    assert out["group_id"] == "envg" and out["agent_groups"] == {"codex": "work"}
    assert stored_config(home)["group_id"] == "me"  # the env override is not written to config.json


# --------------------------------------------------------------------------- CLI: groups


def test_groups_lists_counts_agents_and_configured_empty_groups(capsys, home, fakes):
    write_config(home, group_id="me", agent_groups={"codex": "work", "cursor": "work", "opencode": "later"})
    run(capsys, "add", TARO, "--agent", "codex", "--occurred-at", "2026-07-01 10:00")
    run(capsys, "add", HANAKO, "--agent", "claude-code", "--occurred-at", "2026-08-15 10:00")
    run(capsys, "add", BOB, "--agent", "claude-code", "--occurred-at", "2026-07-01 10:00")

    code, out, _ = run(capsys, "groups")
    assert code == 0 and out["default_group"] == "me" and out["db"] == str(home / "memory.db")
    by = {g["group_id"]: g for g in out["groups"]}
    assert [g["group_id"] for g in out["groups"]] == ["me", "later", "work"]  # default first, then by name
    assert by["me"]["episodes"] == 2 and by["me"]["facts_valid"] == 2 and by["me"]["entities"] >= 4
    assert by["me"]["agents"] == ["(default)"] and by["me"]["default"] is True
    assert by["work"]["episodes"] == 1 and by["work"]["facts_valid"] == 1 and by["work"]["agents"] == ["codex", "cursor"]
    assert by["work"]["default"] is False
    assert by["later"] == {"group_id": "later", "episodes": 0, "entities": 0, "facts_valid": 0, "agents": ["opencode"], "default": False}
    assert "note" not in out


def test_groups_read_only_without_a_database_and_with_env(capsys, home, monkeypatch):
    code, out, _ = run(capsys, "groups")
    assert code == 0 and not home.exists()  # nothing is created
    assert out["groups"] == [{"group_id": "me", "episodes": 0, "entities": 0, "facts_valid": 0, "agents": ["(default)"], "default": True}]
    write_config(home, agent_groups={"codex": "work"})
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    code, out, _ = run(capsys, "groups")
    assert out["default_group"] == "envg" and "SQGRAPH_MEMORY_GROUP" in out["note"]
    assert [g["group_id"] for g in out["groups"]] == ["envg"] and out["groups"][0]["agents"] == ["(default)", "codex"]


def test_groups_does_not_modify_the_database(capsys, home, fakes):
    run(capsys, "add", TARO, "--occurred-at", "2026-07-01 10:00")
    before = (home / "memory.db").read_bytes()
    assert run(capsys, "groups")[0] == 0
    assert (home / "memory.db").read_bytes() == before


# --------------------------------------------------------------------------- doctor


def test_doctor_reports_the_assignments(capsys, home, fake_home):
    def check():
        return next(c for c in run(capsys, "doctor")[1]["checks"] if c["name"] == "agent groups")

    c = check()
    assert c["status"] == "ok" and "shares group 'me'" in c["detail"]
    write_config(home, group_id="me", agent_groups={"codex": "work", "cursor": "play"})
    c = check()
    assert c["status"] == "ok" and "codex -> work" in c["detail"] and "cursor -> play" in c["detail"] and "'me'" in c["detail"]


def test_doctor_notes_the_env_override(capsys, home, fake_home, monkeypatch):
    write_config(home, agent_groups={"codex": "work"})
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    c = next(c for c in run(capsys, "doctor")[1]["checks"] if c["name"] == "agent groups")
    assert c["status"] == "ok" and "envg" in c["detail"] and "SQGRAPH_MEMORY_GROUP" in c["detail"]


# --------------------------------------------------------------------------- setup


def test_setup_writes_agent_into_every_mcp_and_hook_registration(tmp_path):
    claude_runs: list[list[str]] = []

    def runner(argv):
        claude_runs.append(argv)
        return 0, "", ""

    which = lambda n: "/usr/bin/claude" if n == "claude" else None  # noqa: E731
    cmd = ["/opt/bin/sqgraph-memory", "mcp"]
    out = sa.run_setup(list(sa.AGENTS), home=tmp_path, command=cmd, runner=runner, which=which, platform="linux")
    assert {r["agent"]: r["action"] for r in out["agents"]} == dict.fromkeys(sa.AGENTS, "registered")

    assert claude_runs == [["/usr/bin/claude", "mcp", "add", "--scope", "user", "sqgraph-memory", "--", *cmd, "--agent", "claude-code"]]
    cursor = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert cursor["mcpServers"]["sqgraph-memory"]["args"] == ["mcp", "--agent", "cursor"]
    desktop = json.loads((tmp_path / ".config" / "Claude" / "claude_desktop_config.json").read_text(encoding="utf-8"))
    assert desktop["mcpServers"]["sqgraph-memory"]["args"] == ["mcp", "--agent", "claude-desktop"]
    codex = sa.codex_current((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
    assert codex["command"] == cmd[0] and codex["args"] == ["mcp", "--agent", "codex"]
    oc = json.loads((tmp_path / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert oc["mcp"]["sqgraph-memory"]["command"] == [*cmd, "--agent", "opencode"]

    cc_hooks = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))["hooks"]
    assert cc_hooks["SessionStart"][0]["hooks"][0]["command"] == "/opt/bin/sqgraph-memory hook session-start --agent claude-code"
    assert cc_hooks["UserPromptSubmit"][0]["hooks"][0]["command"] == "/opt/bin/sqgraph-memory hook user-prompt --agent claude-code"
    cx_hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    assert cx_hooks["SessionStart"][0]["hooks"][0]["command"] == "/opt/bin/sqgraph-memory hook session-start --agent codex"
    plugin = (tmp_path / ".config" / "opencode" / "plugins" / "sqgraph-memory.js").read_text(encoding="utf-8")
    assert '"hook", event, "--agent", "opencode"' in plugin

    # idempotent (claude-code's MCP entry is read from ~/.claude.json, which the fake runner did not write)
    again = sa.run_setup(list(sa.AGENTS), home=tmp_path, command=cmd, runner=runner, which=which, platform="linux")
    actions = {r["agent"]: r["action"] for r in again["agents"]}
    assert all(a == "unchanged" for n, a in actions.items() if n != "claude-code")
    assert all(r["hooks"]["action"] == "unchanged" for r in again["agents"] if r["agent"] in ("claude-code", "codex", "opencode"))


def test_command_override_gets_the_agent_appended_after_it(tmp_path):
    cmd = ["uvx", "--from", "sqgraph[mcp]", "sqgraph-memory", "mcp"]
    sa.run_setup(["cursor", "codex"], home=tmp_path, command=cmd, which=lambda n: None, platform="linux")
    entry = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["sqgraph-memory"]
    assert entry == {"command": "uvx", "args": ["--from", "sqgraph[mcp]", "sqgraph-memory", "mcp", "--agent", "cursor"]}
    hooks_json = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    assert hooks_json["SessionStart"][0]["hooks"][0]["command"] == "uvx --from 'sqgraph[mcp]' sqgraph-memory hook session-start --agent codex"


def test_setup_updates_old_registrations_without_agent_in_place(tmp_path):
    cmd = ["/opt/bin/sqgraph-memory", "mcp"]
    cur = tmp_path / ".cursor"
    cur.mkdir()
    (cur / "mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "x"}, "sqgraph-memory": {"command": cmd[0], "args": ["mcp"], "env": {"KEEP": "1"}}}}),
        encoding="utf-8",
    )
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text(sa.codex_add("", cmd), encoding="utf-8")
    (tmp_path / ".claude").mkdir()
    old_hooks = {
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": "/opt/bin/sqgraph-memory hook session-start", "timeout": 30}]}],
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "/opt/bin/sqgraph-memory hook user-prompt", "timeout": 15}]}],
        }
    }
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(old_hooks), encoding="utf-8")

    out = sa.run_setup(["cursor", "codex", "claude-code"], home=tmp_path, command=cmd, which=lambda n: None, platform="linux")
    by = {r["agent"]: r for r in out["agents"]}
    assert by["cursor"]["action"] == "updated" and by["codex"]["action"] == "updated"
    assert by["claude-code"]["hooks"]["action"] == "installed"
    servers = json.loads((cur / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert servers["sqgraph-memory"] == {"command": cmd[0], "args": ["mcp", "--agent", "cursor"], "env": {"KEEP": "1"}} and "other" in servers
    assert sa.codex_current((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))["args"] == ["mcp", "--agent", "codex"]
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))["hooks"]
    for event, name in (("SessionStart", "session-start"), ("UserPromptSubmit", "user-prompt")):
        assert [h["command"] for g in settings[event] for h in g["hooks"]] == [f"/opt/bin/sqgraph-memory hook {name} --agent claude-code"]
    st = {s["agent"]: s for s in sa.agents_status(home=tmp_path, command=cmd, which=lambda n: None, platform="linux")}
    assert st["cursor"]["up_to_date"] and st["codex"]["up_to_date"]


def test_setup_group_assigns_the_given_agents_and_registers_with_agent(capsys, home, fake_home):
    code, out, _ = run(capsys, "setup", "--agents", "codex", "--group", "work", "--command", "my-launcher mcp")
    assert code == 0
    assert out["group_assignment"] == {"group": "work", "agents": ["codex"], "action": "assigned", "agent_groups": {"codex": "work"}}
    assert stored_config(home)["agent_groups"] == {"codex": "work"}
    assert stored_config(home)["group_id"] == "me"
    assert sa.codex_current((fake_home / ".codex" / "config.toml").read_text(encoding="utf-8"))["args"] == ["mcp", "--agent", "codex"]

    # more agents add to the mapping; the first stays
    code, out, _ = run(capsys, "setup", "--agents", "cursor,opencode", "--group", "play")
    assert out["group_assignment"]["agent_groups"] == {"codex": "work", "cursor": "play", "opencode": "play"}
    assert run(capsys, "config")[1]["agent_groups"] == {"codex": "work", "cursor": "play", "opencode": "play"}

    # the same assignment again is a no-op; `default` removes it
    assert run(capsys, "setup", "--agents", "codex", "--group", "work")[1]["group_assignment"]["action"] == "unchanged"
    code, out, _ = run(capsys, "setup", "--agents", "cursor,opencode", "--group", "default")
    assert out["group_assignment"]["action"] == "removed" and out["group_assignment"]["agent_groups"] == {"codex": "work"}
    assert stored_config(home)["agent_groups"] == {"codex": "work"}


def test_setup_group_without_agents_applies_to_the_detected_ones(capsys, home, fake_home):
    code, out, _ = run(capsys, "setup", "--group", "work")
    assert code == 0 and out["group_assignment"]["agents"] == [] and out["group_assignment"]["action"] == "unchanged"
    assert "note" in out["group_assignment"] and not (home / "config.json").exists()
    (fake_home / ".cursor").mkdir()
    (fake_home / ".codex").mkdir()
    code, out, _ = run(capsys, "setup", "--group", "work")
    assert sorted(out["group_assignment"]["agents"]) == ["codex", "cursor"]
    assert stored_config(home)["agent_groups"] == {"codex": "work", "cursor": "work"}


def test_setup_group_dry_run_writes_nothing(capsys, home, fake_home):
    code, out, _ = run(capsys, "setup", "--agents", "codex", "--group", "work", "--dry-run")
    assert code == 0 and out["dry_run"] is True
    assert out["group_assignment"] == {"group": "work", "agents": ["codex"], "action": "would-assign", "agent_groups": {"codex": "work"}}
    assert not (home / "config.json").exists() and not (fake_home / ".codex").exists()

    run(capsys, "config", "--agent-group", "codex=work")
    code, out, _ = run(capsys, "setup", "--agents", "codex", "--group", "default", "--dry-run")
    assert out["group_assignment"]["action"] == "would-remove" and out["group_assignment"]["agent_groups"] == {}
    assert stored_config(home)["agent_groups"] == {"codex": "work"}


def test_setup_group_validation_and_uninstall_leave_the_mapping_alone(capsys, home, fake_home):
    run(capsys, "setup", "--agents", "codex", "--group", "work")
    code, _, err = run(capsys, "setup", "--agents", "codex", "--group", "bad group")
    assert code == 2 and "--group" in err["error"]
    code, _, err = run(capsys, "setup", "--agents", "codex", "--group", "work", "--uninstall")
    assert code == 2 and "--uninstall" in err["error"]
    code, out, _ = run(capsys, "setup", "--agents", "codex", "--uninstall")
    assert code == 0 and out["agents"][0]["action"] == "removed" and "group_assignment" not in out
    assert stored_config(home)["agent_groups"] == {"codex": "work"}  # uninstall does not touch agent_groups


def test_setup_group_warns_when_the_env_group_is_set(capsys, home, fake_home, monkeypatch):
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "envg")
    code, out, _ = run(capsys, "setup", "--agents", "codex", "--group", "work", "--dry-run")
    assert "SQGRAPH_MEMORY_GROUP" in out["group_assignment"]["warning"]
