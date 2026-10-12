"""`setup` for Codex hooks and OpenCode (MCP in opencode.json, shared skill, plugin); the plugin itself under Node."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap

import pytest

from sqgraph.memory import setup_agents as sa

CMD = ["/bin/sqgraph-memory", "mcp"]


def runner(argv):
    return 0, "", ""


def setup(home, agents, **kw):
    return sa.run_setup(agents, home=home, command=kw.pop("command", CMD), runner=runner, which=lambda n: None, platform="linux", **kw)


def agent(out, name):
    return next(a for a in out["agents"] if a["agent"] == name)


# --------------------------------------------------------------------------- JSONC


def test_strip_jsonc_keeps_strings():
    text = '{\n  // comment\n  "url": "http://x//y", /* block */ "a": [1, 2,],\n  "s": "say \\"//\\"",\n}'
    assert json.loads(sa.strip_jsonc(text)) == {"url": "http://x//y", "a": [1, 2], "s": 'say "//"'}
    assert sa.has_jsonc_comments(text)
    assert not sa.has_jsonc_comments('{"a": "//not a comment", "b": [1,],}')


# --------------------------------------------------------------------------- Codex hooks


def test_codex_hooks_in_hooks_json(tmp_path):
    (tmp_path / ".codex").mkdir()
    out = setup(tmp_path, ["codex"])
    rep = agent(out, "codex")
    assert rep["action"] == "registered" and rep["hooks"]["action"] == "installed" and "/hooks" in rep["hooks"]["note"]
    data = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert data["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "/bin/sqgraph-memory hook session-start --agent codex"
    assert data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["timeout"] == 15
    assert agent(setup(tmp_path, ["codex"]), "codex")["hooks"]["action"] == "unchanged"
    st = sa.agents_status(home=tmp_path, command=CMD, which=lambda n: None, platform="linux")
    assert next(s for s in st if s["agent"] == "codex")["hooks"] == ["session-start", "user-prompt"]
    out = setup(tmp_path, ["codex"], uninstall=True)
    assert agent(out, "codex")["hooks"]["action"] == "removed"
    assert json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8")) == {}


# --------------------------------------------------------------------------- OpenCode


def oc_dir(home):
    d = home / ".config" / "opencode"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_opencode_register_beside_a_commented_jsonc(tmp_path):
    d = oc_dir(tmp_path)
    jsonc = '{\n  // my provider\n  "provider": {"x": {}},\n}\n'
    (d / "opencode.jsonc").write_text(jsonc, encoding="utf-8")
    out = setup(tmp_path, ["opencode"], env={"SQGRAPH_MEMORY_HOME": "/h"})
    rep = agent(out, "opencode")
    assert rep["action"] == "registered" and rep["skill"]["action"] == "installed" and rep["hooks"]["action"] == "installed"
    assert (d / "opencode.jsonc").read_text(encoding="utf-8") == jsonc  # never rewritten
    cfg = json.loads((d / "opencode.json").read_text(encoding="utf-8"))
    assert cfg["mcp"]["sqgraph-memory"] == {
        "type": "local", "command": CMD, "enabled": True, "timeout": 30000, "environment": {"SQGRAPH_MEMORY_HOME": "/h"}
    }
    assert (tmp_path / ".agents" / "skills" / "sqgraph-memory" / "SKILL.md").is_file()
    plugin = (d / "plugins" / "sqgraph-memory.js").read_text(encoding="utf-8")
    assert 'const COMMAND = ["/bin/sqgraph-memory"]' in plugin and '"SQGRAPH_MEMORY_HOME": "/h"' in plugin
    assert "__SQGRAPH" not in plugin
    again = agent(setup(tmp_path, ["opencode"], env={"SQGRAPH_MEMORY_HOME": "/h"}), "opencode")
    assert again["action"] == "unchanged" and again["hooks"]["action"] == "unchanged"
    st = next(s for s in sa.agents_status(home=tmp_path, command=CMD, which=lambda n: None, platform="linux", env={"SQGRAPH_MEMORY_HOME": "/h"}) if s["agent"] == "opencode")
    assert st["registered"] and st["up_to_date"] and st["hooks"] == ["session-start", "user-prompt"]


def test_opencode_existing_opencode_json_is_merged_and_uninstall_cleans_up(tmp_path):
    d = oc_dir(tmp_path)
    (d / "opencode.json").write_text(json.dumps({"model": "a/b", "mcp": {"other": {"type": "remote", "url": "u"}}}), encoding="utf-8")
    setup(tmp_path, ["opencode"])
    cfg = json.loads((d / "opencode.json").read_text(encoding="utf-8"))
    assert cfg["model"] == "a/b" and set(cfg["mcp"]) == {"other", "sqgraph-memory"}
    out = setup(tmp_path, ["opencode"], uninstall=True)
    rep = agent(out, "opencode")
    assert rep["action"] == "removed" and rep["hooks"]["action"] == "removed" and rep["skill"]["action"] == "removed"
    assert json.loads((d / "opencode.json").read_text(encoding="utf-8")) == {"model": "a/b", "mcp": {"other": {"type": "remote", "url": "u"}}}
    assert not (d / "plugins" / "sqgraph-memory.js").exists()


def test_opencode_created_file_is_removed_and_shared_skill_kept_for_codex(tmp_path):
    d = oc_dir(tmp_path)
    (tmp_path / ".codex").mkdir()
    setup(tmp_path, ["codex", "opencode"])
    out = setup(tmp_path, ["opencode"], uninstall=True)
    rep = agent(out, "opencode")
    assert not (d / "opencode.json").exists()  # only ours was in it
    assert rep["skill"]["action"] == "kept" and (tmp_path / ".agents" / "skills" / "sqgraph-memory" / "SKILL.md").is_file()


def test_opencode_entry_kept_in_jsonc_or_commented_json_is_manual(tmp_path):
    d = oc_dir(tmp_path)
    (d / "opencode.jsonc").write_text('{\n // mine\n "mcp": {"sqgraph-memory": {"type": "local", "command": ["old", "mcp"]}}\n}', encoding="utf-8")
    rep = agent(setup(tmp_path, ["opencode"]), "opencode")
    assert rep["action"] == "manual" and "opencode.jsonc" in rep["note"] and rep["manual_entry"]["sqgraph-memory"]["command"] == CMD
    assert not (d / "opencode.json").exists()
    (d / "opencode.jsonc").unlink()
    (d / "opencode.json").write_text('{\n // comment\n "model": "x"\n}', encoding="utf-8")
    rep = agent(setup(tmp_path, ["opencode"]), "opencode")
    assert rep["action"] == "manual" and "comments" in rep["note"]


def test_opencode_dry_run_and_no_hooks(tmp_path):
    d = oc_dir(tmp_path)
    rep = agent(setup(tmp_path, ["opencode"], dry_run=True), "opencode")
    assert rep["action"] == "would-register" and rep["hooks"]["action"] == "would-install"
    assert not (d / "opencode.json").exists() and not (d / "plugins").exists()
    rep = agent(setup(tmp_path, ["opencode"], hooks=()), "opencode")
    assert rep["hooks"] == {"action": "skipped"} and not (d / "plugins").exists()


# --------------------------------------------------------------------------- the plugin, run by Node


FAKE_HOOK = textwrap.dedent(
    """
    import json, sys
    event, agent = sys.argv[sys.argv.index("hook") + 1], sys.argv[sys.argv.index("--agent") + 1]
    payload = json.load(sys.stdin)
    text = f"{event}|{agent}|{payload.get('session_id')}|{payload.get('prompt', payload.get('cwd'))}"
    if payload.get("prompt") == "nothing":
        sys.exit(0)
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "x", "additionalContext": text}}))
    """
)

DRIVER = textwrap.dedent(
    """
    const { SqgraphMemory } = await import(process.argv[2])
    const hooks = await SqgraphMemory({ directory: "/proj" })
    const user = (id, text) => ({ id, sessionID: "s1", messageID: "m1", type: "text", text })
    const out1 = { message: {}, parts: [user("prt_01", "hello")] }
    await hooks["chat.message"]({ sessionID: "s1" }, out1)
    const out2 = { message: {}, parts: [user("prt_02", "nothing")] }
    await hooks["chat.message"]({ sessionID: "s1" }, out2)
    const sys1 = { system: ["base"] }
    await hooks["experimental.chat.system.transform"]({ sessionID: "s1", model: {} }, sys1)
    const sys2 = { system: ["base"] }
    await hooks["experimental.chat.system.transform"]({ model: {} }, sys2)
    console.log(JSON.stringify({ out1: out1.parts, out2: out2.parts, sys1: sys1.system, sys2: sys2.system }))
    """
)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_plugin_under_node(tmp_path):
    hook = tmp_path / "fake_hook.py"
    hook.write_text(FAKE_HOOK, encoding="utf-8")
    plugin = tmp_path / "sqgraph-memory.mjs"
    plugin.write_text(sa.opencode_plugin_text([sys.executable, str(hook), "mcp"]), encoding="utf-8")
    driver = tmp_path / "driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    res = subprocess.run(["node", str(driver), plugin.as_uri()], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    added = got["out1"][1]
    assert added == {"id": "prt_01sqgmem", "sessionID": "s1", "messageID": "m1", "type": "text",
                     "text": "user-prompt|opencode|s1|hello", "synthetic": True}
    assert len(got["out2"]) == 1  # no memories: nothing added
    # appended to the last system text: a second system message is rejected by some chat templates (Qwen on vLLM)
    assert got["sys1"] == ["base\n\nsession-start|opencode|s1|/proj"]
    assert got["sys2"] == ["base"]  # no session (title generation): nothing added
