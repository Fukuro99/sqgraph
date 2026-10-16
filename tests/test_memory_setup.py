"""sqgraph.memory.setup_agents: MCP registration and Skill installation against a fake home directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqgraph.memory import setup_agents as sa

CMD = ["/opt/bin/sqgraph-memory", "mcp"]


def no_which(_name: str) -> str | None:
    return None


class FakeRunner:
    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.code = code

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        self.calls.append(list(argv))
        return self.code, "", "boom" if self.code else ""


def run_setup_for(home: Path, agents, *, command=CMD, **kw):
    kw.setdefault("which", no_which)
    kw.setdefault("platform", "linux")
    return sa.run_setup(agents, home=home, command=command, **kw)


def report(res, agent):
    return next(r for r in res["agents"] if r["agent"] == agent)


# --------------------------------------------------------------------------- JSON targets


def test_cursor_merge_preserves_others_backup_idempotent_uninstall(tmp_path):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    original = {"mcpServers": {"other": {"command": "node", "args": ["x.js"]}}, "extra": 1}
    cfg.write_text(json.dumps(original), encoding="utf-8")

    r1 = report(run_setup_for(tmp_path, ["cursor"]), "cursor")
    assert r1["action"] == "registered" and r1["backup"]
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["extra"] == 1 and data["mcpServers"]["other"] == original["mcpServers"]["other"]
    assert data["mcpServers"]["sqgraph-memory"] == {"command": "/opt/bin/sqgraph-memory", "args": ["mcp", "--agent", "cursor"]}
    assert json.loads(Path(r1["backup"]).read_text(encoding="utf-8")) == original  # backup = pre-modification state

    before = cfg.read_text(encoding="utf-8")
    r2 = report(run_setup_for(tmp_path, ["cursor"]), "cursor")
    assert r2["action"] == "unchanged" and cfg.read_text(encoding="utf-8") == before

    # a changed launch command updates the entry in place and keeps user additions such as env
    data["mcpServers"]["sqgraph-memory"]["env"] = {"A": "1"}
    cfg.write_text(json.dumps(data), encoding="utf-8")
    r3 = report(run_setup_for(tmp_path, ["cursor"], command=["/new/sqgraph-memory", "mcp"]), "cursor")
    assert r3["action"] == "updated"
    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["sqgraph-memory"]
    assert entry["command"] == "/new/sqgraph-memory" and entry["env"] == {"A": "1"}

    r4 = report(run_setup_for(tmp_path, ["cursor"], uninstall=True), "cursor")
    assert r4["action"] == "removed"
    after = json.loads(cfg.read_text(encoding="utf-8"))
    assert "sqgraph-memory" not in after["mcpServers"] and "other" in after["mcpServers"]
    assert report(run_setup_for(tmp_path, ["cursor"], uninstall=True), "cursor")["action"] == "not-registered"


def test_cursor_creates_missing_config_and_invalid_json_untouched(tmp_path):
    r = report(run_setup_for(tmp_path, ["cursor"]), "cursor")
    assert r["action"] == "registered" and "backup" in r and r["backup"] is None
    cfg = tmp_path / ".cursor" / "mcp.json"
    assert json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["sqgraph-memory"]["args"] == ["mcp", "--agent", "cursor"]

    cfg.write_text("{ not json", encoding="utf-8")
    r = report(run_setup_for(tmp_path, ["cursor"]), "cursor")
    assert r["action"] == "failed" and cfg.read_text(encoding="utf-8") == "{ not json"


def test_dry_run_writes_nothing(tmp_path):
    (tmp_path / ".cursor").mkdir()
    res = run_setup_for(tmp_path, ["cursor", "codex"], dry_run=True)
    assert report(res, "cursor")["action"] == "would-register"
    assert report(res, "codex")["action"] == "would-register"
    assert report(res, "cursor")["skill"]["action"] == "would-install"
    assert not (tmp_path / ".cursor" / "mcp.json").exists()
    assert not (tmp_path / ".codex").exists() and not (tmp_path / ".agents").exists()


@pytest.mark.parametrize(
    "platform, rel",
    [
        ("linux", ".config/Claude/claude_desktop_config.json"),
        ("darwin", "Library/Application Support/Claude/claude_desktop_config.json"),
        ("win32", "AppData/Roaming/Claude/claude_desktop_config.json"),
    ],
)
def test_claude_desktop_paths_and_detection(tmp_path, platform, rel):
    cfg = tmp_path / rel
    # not installed -> not detected, nothing created for the default (detected-only) selection
    res = run_setup_for(tmp_path, None, platform=platform, appdata=tmp_path / "AppData" / "Roaming")
    assert res["agents"] == [] and "note" in res
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"mcpServers": {"fs": {"command": "npx"}}, "theme": "dark"}), encoding="utf-8")
    res = run_setup_for(tmp_path, None, platform=platform, appdata=tmp_path / "AppData" / "Roaming")
    r = report(res, "claude-desktop")
    assert r["action"] == "registered" and Path(r["config"]) == cfg
    assert r["skill"]["action"] == "unsupported"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["theme"] == "dark" and set(data["mcpServers"]) == {"fs", "sqgraph-memory"}


# --------------------------------------------------------------------------- Codex (TOML)

CODEX_TOML = """\
model = "gpt-5"

[mcp_servers.blender]
command = 'C:\\Users\\x\\uvx.exe'
args = ["--python", "3.11", "blender-mcp"]

[mcp_servers.blender.env]
BLENDER_HOST = "127.0.0.1"

[projects.'C:\\work']
trust_level = "trusted"
"""


def test_codex_toml_add_remove_preserves_everything_else(tmp_path):
    cfg = tmp_path / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(CODEX_TOML, encoding="utf-8", newline="\n")

    r = report(run_setup_for(tmp_path, ["codex"]), "codex")
    assert r["action"] == "registered" and Path(r["backup"]).read_text(encoding="utf-8") == CODEX_TOML
    text = cfg.read_text(encoding="utf-8")
    assert text.startswith(CODEX_TOML)  # existing content untouched, ours appended
    assert "[mcp_servers.sqgraph-memory]" in text
    assert "command = '/opt/bin/sqgraph-memory'" in text and "args = ['mcp', '--agent', 'codex']" in text

    assert sa.codex_current(text) == {"command": "/opt/bin/sqgraph-memory", "args": ["mcp", "--agent", "codex"], "env": {}}
    try:
        import tomllib
    except ImportError:  # Python 3.10
        pass
    else:
        parsed = tomllib.loads(text)
        assert parsed["mcp_servers"]["sqgraph-memory"]["args"] == ["mcp", "--agent", "codex"]
        assert parsed["mcp_servers"]["blender"]["env"]["BLENDER_HOST"] == "127.0.0.1"

    assert report(run_setup_for(tmp_path, ["codex"]), "codex")["action"] == "unchanged"
    assert cfg.read_text(encoding="utf-8") == text

    r = report(run_setup_for(tmp_path, ["codex"], uninstall=True), "codex")
    assert r["action"] == "removed"
    assert cfg.read_text(encoding="utf-8") == CODEX_TOML
    assert report(run_setup_for(tmp_path, ["codex"], uninstall=True), "codex")["action"] == "not-registered"


def test_codex_update_keeps_user_keys_and_subtables_removed_with_it(tmp_path):
    cfg = tmp_path / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(
        'a = 1\n\n[mcp_servers.sqgraph-memory]\ncommand = "old"\nargs = [\n  "x",\n  "y",\n]\ntool_timeout_sec = 90\n\n'
        '[mcp_servers.sqgraph-memory.env]\nK = "v"\n\n[tail]\nz = 2\n',
        encoding="utf-8",
    )
    assert report(run_setup_for(tmp_path, ["codex"]), "codex")["action"] == "updated"
    text = cfg.read_text(encoding="utf-8")
    assert text.count("command =") == 1 and "old" not in text and '"x"' not in text
    assert "tool_timeout_sec = 90" in text and 'K = "v"' in text and "[tail]" in text
    assert sa.codex_current(text) == {"command": "/opt/bin/sqgraph-memory", "args": ["mcp", "--agent", "codex"], "env": {"K": "v"}}

    removed = sa.codex_remove(text)
    assert "sqgraph-memory" not in removed and "K = " not in removed
    assert "a = 1" in removed and "[tail]\nz = 2" in removed


def test_codex_toml_quotes_windows_paths_and_spaces():
    text = sa.codex_add("", [r"C:\Users\me\.local\bin\sqgraph-memory.exe", "mcp"])
    assert r"command = 'C:\Users\me\.local\bin\sqgraph-memory.exe'" in text
    cur = sa.codex_current(text)
    assert cur == {"command": r"C:\Users\me\.local\bin\sqgraph-memory.exe", "args": ["mcp"], "env": {}}
    odd = sa.codex_add("", ["it's here", "mcp"])
    assert sa.codex_current(odd)["command"] == "it's here"


def test_toml_header_inside_multiline_string_is_not_a_table():
    text = 'x = """\n[mcp_servers.sqgraph-memory]\n"""\n\n[other]\ny = 1\n'
    assert sa.codex_remove(text) == text  # nothing of ours: the header is inside a string


# --------------------------------------------------------------------------- Claude Code


def test_claude_code_via_runner(tmp_path):
    runner = FakeRunner()
    which = lambda n: "/usr/bin/claude" if n == "claude" else None  # noqa: E731
    res = run_setup_for(tmp_path, ["claude-code"], runner=runner, which=which)
    r = report(res, "claude-code")
    assert r["action"] == "registered"
    assert runner.calls == [["/usr/bin/claude", "mcp", "add", "--scope", "user", "sqgraph-memory", "--", *CMD, "--agent", "claude-code"]]

    # already registered with the same command (as recorded in ~/.claude.json): nothing to do
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"sqgraph-memory": {"type": "stdio", "command": CMD[0], "args": ["mcp", "--agent", "claude-code"]}}}), encoding="utf-8"
    )
    runner2 = FakeRunner()
    assert report(run_setup_for(tmp_path, ["claude-code"], runner=runner2, which=which), "claude-code")["action"] == "unchanged"
    assert runner2.calls == []

    # different command: remove then add
    runner3 = FakeRunner()
    r = report(run_setup_for(tmp_path, ["claude-code"], command=["/other/sqgraph-memory", "mcp"], runner=runner3, which=which), "claude-code")
    assert r["action"] == "updated"
    assert runner3.calls[0][:6] == ["/usr/bin/claude", "mcp", "remove", "--scope", "user", "sqgraph-memory"]
    assert runner3.calls[1][-4:] == ["/other/sqgraph-memory", "mcp", "--agent", "claude-code"]

    runner4 = FakeRunner()
    r = report(run_setup_for(tmp_path, ["claude-code"], uninstall=True, runner=runner4, which=which), "claude-code")
    assert r["action"] == "removed"
    assert runner4.calls == [["/usr/bin/claude", "mcp", "remove", "--scope", "user", "sqgraph-memory"]]


def test_claude_code_without_cli_reports_manual_command_and_failure(tmp_path):
    r = report(run_setup_for(tmp_path, ["claude-code"]), "claude-code")
    assert r["action"] == "manual"
    assert r["manual_command"] == "claude mcp add --scope user sqgraph-memory -- /opt/bin/sqgraph-memory mcp --agent claude-code"

    bad = FakeRunner(code=1)
    r = report(run_setup_for(tmp_path, ["claude-code"], runner=bad, which=lambda n: "/usr/bin/claude"), "claude-code")
    assert r["action"] == "failed" and "boom" in r["error"] and r["manual_command"]


def test_claude_code_dry_run_does_not_call_runner(tmp_path):
    runner = FakeRunner()
    r = report(run_setup_for(tmp_path, ["claude-code"], runner=runner, which=lambda n: "/usr/bin/claude", dry_run=True), "claude-code")
    assert r["action"] == "would-register" and runner.calls == []


# --------------------------------------------------------------------------- Skill


def test_skill_text_is_packaged_and_has_frontmatter():
    text = sa.skill_text()
    assert text.startswith("---\nname: sqgraph-memory\ndescription: ")
    assert "forget" in text and "search_memory" in text


def test_skill_install_update_uninstall(tmp_path):
    res = run_setup_for(tmp_path, ["claude-code", "codex", "cursor"], runner=FakeRunner(), which=lambda n: "/x/claude" if n == "claude" else None)
    dirs = {
        "claude-code": tmp_path / ".claude" / "skills" / "sqgraph-memory",
        "codex": tmp_path / ".agents" / "skills" / "sqgraph-memory",
        "cursor": tmp_path / ".cursor" / "skills" / "sqgraph-memory",
    }
    for agent, d in dirs.items():
        assert report(res, agent)["skill"]["action"] == "installed"
        assert (d / "SKILL.md").read_text(encoding="utf-8") == sa.skill_text()

    again = run_setup_for(tmp_path, ["codex"])
    assert report(again, "codex")["skill"]["action"] == "unchanged"

    stale = dirs["codex"] / "SKILL.md"
    stale.write_text("old", encoding="utf-8")
    upd = report(run_setup_for(tmp_path, ["codex"]), "codex")["skill"]
    assert upd["action"] == "updated" and Path(upd["backup"]).read_text(encoding="utf-8") == "old"

    gone = run_setup_for(tmp_path, ["codex"], uninstall=True)
    assert report(gone, "codex")["skill"]["action"] == "removed"
    assert not (dirs["codex"] / "SKILL.md").exists() and not dirs["codex"].exists()
    assert (dirs["cursor"] / "SKILL.md").exists()  # other agents untouched


def test_no_skill_flag(tmp_path):
    res = run_setup_for(tmp_path, ["cursor"], install_skill=False)
    assert report(res, "cursor")["skill"]["action"] == "skipped"
    assert not (tmp_path / ".cursor" / "skills").exists()


# --------------------------------------------------------------------------- command resolution / status


def test_resolve_launch_command(tmp_path):
    exe = tmp_path / "sqgraph-memory"
    exe.write_text("")
    cmd, warns = sa.resolve_launch_command(which=lambda n: str(exe), executable=str(tmp_path / "python"))
    assert cmd == [str(exe), "mcp"] and warns == []

    cmd, _ = sa.resolve_launch_command(which=no_which, argv0="nope", executable="/py/python")
    assert cmd == ["/py/python", "-m", "sqgraph.memory", "mcp"]

    cmd, _ = sa.resolve_launch_command('uvx --from "sqgraph[mcp]" sqgraph-memory mcp', which=no_which)
    assert cmd == ["uvx", "--from", "sqgraph[mcp]", "sqgraph-memory", "mcp"]

    _, warns = sa.resolve_launch_command(
        which=lambda n: "C:/Users/me/AppData/Local/uv/cache/archive-v0/abc/Scripts/sqgraph-memory.exe", executable="C:/py/python.exe"
    )
    assert warns and "uv tool install" in warns[0]
    assert sa.looks_ephemeral("/home/me/.cache/uv/archive-v0/x/bin/python")
    assert not sa.looks_ephemeral("/home/me/.local/share/uv/tools/sqgraph/bin/sqgraph-memory")


def test_parse_agents():
    assert sa.parse_agents(None) is None
    assert sa.parse_agents("all") == list(sa.AGENTS)
    assert sa.parse_agents("codex, cursor") == ["codex", "cursor"]
    with pytest.raises(ValueError):
        sa.parse_agents("vim")


def test_agents_status(tmp_path):
    run_setup_for(tmp_path, ["cursor", "codex"])
    st = {s["agent"]: s for s in sa.agents_status(home=tmp_path, command=CMD, which=no_which, platform="linux")}
    assert st["cursor"]["registered"] and st["cursor"]["up_to_date"] and st["cursor"]["skill_installed"]
    assert st["codex"]["registered"] and st["codex"]["up_to_date"]
    assert not st["claude-code"]["detected"] and st["claude-code"]["registered"] is False
    assert st["claude-desktop"]["skill_installed"] is None
    changed = {s["agent"]: s for s in sa.agents_status(home=tmp_path, command=["/z", "mcp"], which=no_which, platform="linux")}
    assert changed["cursor"]["registered"] and changed["cursor"]["up_to_date"] is False


# --------------------------------------------------------------------------- forwarded SQGRAPH_MEMORY_* environment

ENV = {"SQGRAPH_MEMORY_HOME": "/data/mem", "SQGRAPH_MEMORY_GROUP": "taro"}


def test_forwarded_env_collects_set_variables_with_absolute_paths(tmp_path, monkeypatch):
    for k in sa.FORWARD_ENV:
        monkeypatch.delenv(k, raising=False)
    assert sa.forwarded_env() == {}
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", "rel/home")
    monkeypatch.setenv("SQGRAPH_MEMORY_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("SQGRAPH_MEMORY_GROUP", "g")
    env = sa.forwarded_env()
    assert Path(env["SQGRAPH_MEMORY_HOME"]).is_absolute() and env["SQGRAPH_MEMORY_HOME"].endswith("home")
    assert env["SQGRAPH_MEMORY_DB"] == str(tmp_path / "x.db") and env["SQGRAPH_MEMORY_GROUP"] == "g"


def test_env_json_targets_merge_ours_win_and_update_when_env_changes(tmp_path):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_text(
        json.dumps({"mcpServers": {"sqgraph-memory": {"command": CMD[0], "args": ["mcp"], "env": {"KEEP": "1", "SQGRAPH_MEMORY_GROUP": "old"}}}}),
        encoding="utf-8",
    )
    r = report(run_setup_for(tmp_path, ["cursor"], env=ENV), "cursor")
    assert r["action"] == "updated" and r["env"] == ENV  # same command, but the env differed
    entry = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["sqgraph-memory"]
    assert entry["env"] == {"KEEP": "1", "SQGRAPH_MEMORY_GROUP": "taro", "SQGRAPH_MEMORY_HOME": "/data/mem"}
    assert report(run_setup_for(tmp_path, ["cursor"], env=ENV), "cursor")["action"] == "unchanged"
    # without forwarded variables nothing is added or removed
    assert report(run_setup_for(tmp_path, ["cursor"]), "cursor")["action"] == "unchanged"


def test_env_claude_desktop_and_dry_run_report(tmp_path):
    cfg = tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    cfg.parent.mkdir(parents=True)
    res = run_setup_for(tmp_path, ["claude-desktop"], env=ENV, dry_run=True)
    assert report(res, "claude-desktop")["env"] == ENV and res["env"] == ENV and not cfg.exists()
    run_setup_for(tmp_path, ["claude-desktop"], env=ENV)
    assert json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["sqgraph-memory"]["env"] == ENV


def test_env_codex_subtable(tmp_path):
    cfg = tmp_path / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(CODEX_TOML, encoding="utf-8", newline="\n")
    assert report(run_setup_for(tmp_path, ["codex"], env=ENV), "codex")["action"] == "registered"
    text = cfg.read_text(encoding="utf-8")
    assert "[mcp_servers.sqgraph-memory.env]" in text and "SQGRAPH_MEMORY_GROUP = 'taro'" in text
    assert text.startswith(CODEX_TOML)
    assert sa.codex_current(text)["env"] == ENV
    try:
        import tomllib
    except ImportError:
        pass
    else:
        assert tomllib.loads(text)["mcp_servers"]["sqgraph-memory"]["env"] == ENV
    assert report(run_setup_for(tmp_path, ["codex"], env=ENV), "codex")["action"] == "unchanged"

    # a changed value replaces ours, user keys in the env table survive
    text2 = text.replace("SQGRAPH_MEMORY_GROUP = 'taro'", "SQGRAPH_MEMORY_GROUP = 'x'\nUSER = 'u'")
    cfg.write_text(text2, encoding="utf-8", newline="\n")
    assert report(run_setup_for(tmp_path, ["codex"], env=ENV), "codex")["action"] == "updated"
    cur = sa.codex_current(cfg.read_text(encoding="utf-8"))
    assert cur["env"] == {**ENV, "USER": "u"}
    assert cfg.read_text(encoding="utf-8").count("[mcp_servers.sqgraph-memory.env]") == 1

    # uninstall drops the env sub-table too
    run_setup_for(tmp_path, ["codex"], uninstall=True)
    assert cfg.read_text(encoding="utf-8") == CODEX_TOML

    # inline env in the existing table cannot be merged safely: reported, file untouched
    cfg.write_text('[mcp_servers.sqgraph-memory]\ncommand = "a"\nenv = { A = "1" }\n', encoding="utf-8")
    r = report(run_setup_for(tmp_path, ["codex"], env=ENV), "codex")
    assert r["action"] == "failed" and "inline" in r["error"]


def test_env_claude_code_flags(tmp_path):
    runner = FakeRunner()
    which = lambda n: "/usr/bin/claude" if n == "claude" else None  # noqa: E731
    r = report(run_setup_for(tmp_path, ["claude-code"], runner=runner, which=which, env=ENV), "claude-code")
    assert r["action"] == "registered" and r["env"] == ENV
    assert runner.calls == [
        [
            "/usr/bin/claude", "mcp", "add", "--scope", "user", "sqgraph-memory",
            "-e", "SQGRAPH_MEMORY_HOME=/data/mem", "-e", "SQGRAPH_MEMORY_GROUP=taro",
            "--", *CMD, "--agent", "claude-code",
        ]
    ]  # fmt: skip
    dry = report(run_setup_for(tmp_path, ["claude-code"], runner=FakeRunner(), which=which, env=ENV, dry_run=True), "claude-code")
    assert "-e SQGRAPH_MEMORY_GROUP=taro" in dry["manual_command"]

    # registered with the right command but without the env: re-registered
    (tmp_path / ".claude.json").write_text(
        json.dumps({"mcpServers": {"sqgraph-memory": {"command": CMD[0], "args": ["mcp"]}}}), encoding="utf-8"
    )
    runner2 = FakeRunner()
    assert report(run_setup_for(tmp_path, ["claude-code"], runner=runner2, which=which, env=ENV), "claude-code")["action"] == "updated"
    assert runner2.calls[0][2] == "remove"
