"""Register ``sqgraph-memory`` as an MCP server (and install the Agent Skill) for local agents.

Targets and the formats written (verified against each product's documentation):

* **Claude Code** -- ``claude mcp add --scope user sqgraph-memory -- <cmd> mcp`` (stored in ``~/.claude.json``);
  skill: ``~/.claude/skills/sqgraph-memory/SKILL.md``; hooks (``SessionStart`` / ``UserPromptSubmit`` running
  ``<cmd> hook session-start|user-prompt``, see :mod:`sqgraph.memory.hooks`) in ``~/.claude/settings.json``
  (https://code.claude.com/docs/en/hooks).
* **Codex CLI** -- ``~/.codex/config.toml``, a ``[mcp_servers.sqgraph-memory]`` table with ``command`` / ``args``
  (https://developers.openai.com/codex/mcp); skill: ``~/.agents/skills/sqgraph-memory/SKILL.md``
  (https://developers.openai.com/codex/skills); hooks (the same two as Claude Code) in ``~/.codex/hooks.json``
  (https://learn.chatgpt.com/docs/hooks; Codex asks the user to trust new hooks once, with ``/hooks``).
* **OpenCode** -- ``mcp.sqgraph-memory`` (``type: local``, ``command`` array) in ``~/.config/opencode/opencode.json``.
  OpenCode merges ``config.json``, ``opencode.json`` and ``opencode.jsonc`` of that directory, so a commented
  ``opencode.jsonc`` is never rewritten; skill: the shared ``~/.agents/skills/sqgraph-memory/SKILL.md`` (OpenCode reads
  it, https://opencode.ai/docs/skills/); plugin ``~/.config/opencode/plugins/sqgraph-memory.js`` running the hooks
  (https://opencode.ai/docs/plugins/).
* **Cursor** -- ``~/.cursor/mcp.json`` ``mcpServers`` (https://cursor.com/docs/context/mcp); skill:
  ``~/.cursor/skills/sqgraph-memory/SKILL.md`` (https://cursor.com/docs/context/skills).
* **Claude Desktop** -- ``claude_desktop_config.json`` ``mcpServers`` (``%APPDATA%\\Claude`` on Windows,
  ``~/Library/Application Support/Claude`` on macOS, ``~/.config/Claude`` on Linux). No skill (not supported).

Every config file is backed up to ``<file>.bak`` before it is modified, edits are idempotent and only touch our
own entry. All locations derive from the ``home`` argument so tests can use a fake home.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SERVER_NAME = "sqgraph-memory"
EXE_NAME = "sqgraph-memory"
AGENTS = ("claude-code", "codex", "opencode", "cursor", "claude-desktop")
STARTUP_TIMEOUT_SEC = 30  # Codex default is 10s; the first start imports sqlite-vec / openai / mcp
OPENCODE_PLUGIN = "sqgraph-memory.js"

Runner = Callable[[list[str]], "tuple[int, str, str]"]

# Claude Code hooks: our event name -> (Claude Code event, timeout in seconds)
HOOKS: dict[str, tuple[str, int]] = {"session-start": ("SessionStart", 30), "user-prompt": ("UserPromptSubmit", 15)}
_OUR_HOOK_RE = re.compile(r"\bhook (session-start|user-prompt)\b")


# --------------------------------------------------------------------------- helpers

FORWARD_ENV = ("SQGRAPH_MEMORY_HOME", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP", "SQGRAPH_MEMORY_EXTRACTOR")


def forwarded_env(environ: Any = None) -> dict[str, str]:
    """The ``SQGRAPH_MEMORY_*`` variables set right now (paths made absolute): agents start the server
    without our shell environment, so they must be written into each registration."""
    src = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for k in FORWARD_ENV:
        v = src.get(k)
        if not v:
            continue
        out[k] = v if k in ("SQGRAPH_MEMORY_GROUP", "SQGRAPH_MEMORY_EXTRACTOR") else os.path.abspath(os.path.expanduser(v))
    return out


def skill_text() -> str:
    """The packaged ``SKILL.md``."""
    from importlib import resources

    return resources.files("sqgraph.memory").joinpath("skill", "SKILL.md").read_text(encoding="utf-8")


def default_runner(argv: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return p.returncode, p.stdout, p.stderr


def format_command(argv: list[str], platform: str | None = None) -> str:
    """A copy-pasteable shell command line."""
    if (platform or sys.platform) == "win32":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def looks_ephemeral(path: str | os.PathLike[str]) -> bool:
    """True if ``path`` is inside a ``uvx`` / ``uv run`` throw-away environment (it may disappear)."""
    s = str(path).replace("\\", "/").lower()
    if "archive-v0" in s or "/uv/cache/" in s or "/.cache/uv/" in s or "/uv/environments" in s:
        return True
    cache = os.environ.get("UV_CACHE_DIR")
    if cache:
        c = cache.replace("\\", "/").lower().rstrip("/") + "/"
        if s.startswith(c):
            return True
    return False


def resolve_launch_command(
    override: str | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    argv0: str | None = None,
    executable: str | None = None,
) -> tuple[list[str], list[str]]:
    """``(argv, warnings)``: how an agent should launch the MCP server."""
    warnings: list[str] = []
    if override:
        cmd = shlex.split(override, posix=(sys.platform != "win32"))
        if sys.platform == "win32":  # non-POSIX mode keeps the quotes (backslashes in paths stay literal)
            cmd = [c[1:-1] if len(c) >= 2 and c[0] == c[-1] and c[0] in "\"'" else c for c in cmd]
        if not cmd:
            raise ValueError("--command is empty")
        return cmd, warnings
    exe = which(EXE_NAME)
    if not exe:
        a0 = argv0 if argv0 is not None else (sys.argv[0] if sys.argv else "")
        if a0 and Path(a0).stem.lower() == EXE_NAME and Path(a0).exists():
            exe = os.path.abspath(a0)
    if exe:
        cmd = [os.path.abspath(exe), "mcp"]
        probe: str = exe
    else:
        py = executable or sys.executable
        cmd = [py, "-m", "sqgraph.memory", "mcp"]
        probe = py
    if looks_ephemeral(probe) or looks_ephemeral(executable or sys.executable):
        warnings.append(
            "this looks like a temporary uvx/uv cache environment; the registered path may disappear. "
            'Install permanently with: uv tool install "sqgraph[mcp]"  and run `sqgraph-memory setup` again.'
        )
    return cmd, warnings


def _backup(path: Path) -> str | None:
    if not path.exists():
        return None
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    return str(bak)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


@dataclass
class Context:
    home: Path
    command: list[str]
    dry_run: bool = False
    install_skill: bool = True
    runner: Runner = default_runner
    which: Callable[[str], str | None] = shutil.which
    platform: str = field(default_factory=lambda: sys.platform)
    appdata: Path | None = None  # Windows %APPDATA% override
    env: dict[str, str] = field(default_factory=dict)  # environment forwarded into every registration
    hooks: tuple[str, ...] = tuple(HOOKS)  # Claude Code hooks to install (empty: leave hooks alone)


# --------------------------------------------------------------------------- skill files


def _skill_path(skill_dir: Path) -> Path:
    return skill_dir / "SKILL.md"


def skill_installed(skill_dir: Path | None) -> bool | None:
    if skill_dir is None:
        return None
    return _skill_path(skill_dir).is_file()


def _install_skill(ctx: Context, skill_dir: Path | None) -> dict[str, Any]:
    if skill_dir is None:
        return {"action": "unsupported"}
    if not ctx.install_skill:
        return {"action": "skipped", "path": str(_skill_path(skill_dir))}
    target = _skill_path(skill_dir)
    text = skill_text()
    cur = target.read_text(encoding="utf-8") if target.is_file() else None
    if cur == text:
        return {"action": "unchanged", "path": str(target)}
    res: dict[str, Any] = {"action": "would-install" if ctx.dry_run else ("updated" if cur is not None else "installed"), "path": str(target)}
    if not ctx.dry_run:
        if cur is not None:
            res["backup"] = _backup(target)
        skill_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, text)
    return res


def _uninstall_skill(ctx: Context, skill_dir: Path | None) -> dict[str, Any]:
    if skill_dir is None:
        return {"action": "unsupported"}
    target = _skill_path(skill_dir)
    if not target.is_file():
        return {"action": "not-installed", "path": str(target)}
    if ctx.dry_run:
        return {"action": "would-remove", "path": str(target)}
    target.unlink()
    bak = target.with_name(target.name + ".bak")
    if bak.is_file():
        bak.unlink()  # created by an earlier update
    try:
        skill_dir.rmdir()  # only if empty
    except OSError:
        pass
    return {"action": "removed", "path": str(target)}


# --------------------------------------------------------------------------- base target


class Target:
    name = ""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    # to override
    def detected(self) -> bool:
        raise NotImplementedError

    def config_path(self) -> Path | None:
        return None

    def skill_dir(self) -> Path | None:
        return None

    def current(self) -> dict[str, Any] | None:
        """``{"command": str, "args": [...]}`` of the registered server, or ``None``."""
        raise NotImplementedError

    def _register(self, report: dict[str, Any]) -> None:
        raise NotImplementedError

    def _unregister(self, report: dict[str, Any]) -> None:
        raise NotImplementedError

    # shared
    def _matches(self, cur: dict[str, Any] | None) -> bool:
        if not cur:
            return False
        have = cur.get("env") or {}
        return (
            cur.get("command") == self.ctx.command[0]
            and list(cur.get("args") or []) == self.ctx.command[1:]
            and all(have.get(k) == v for k, v in self.ctx.env.items())
        )

    def status(self) -> dict[str, Any]:
        try:
            cur = self.current()
            err = None
        except Exception as exc:  # unreadable config
            cur, err = None, str(exc)
        cfg = self.config_path()
        out: dict[str, Any] = {
            "agent": self.name,
            "detected": self.detected(),
            "registered": cur is not None,
            "up_to_date": self._matches(cur) if cur is not None else None,
            "config": str(cfg) if cfg else None,
            "skill_installed": skill_installed(self.skill_dir()),
        }
        if cur is not None:
            out["command"] = cur
        if err:
            out["error"] = err
        hooks = self.hooks_status()
        if hooks is not None:
            out["hooks"] = hooks
        return out

    def hooks_status(self) -> list[str] | None:
        """Our hook events installed for this agent (``None``: the agent has no hooks)."""
        return None

    def uninstall_skill(self) -> dict[str, Any]:
        return _uninstall_skill(self.ctx, self.skill_dir())

    def install(self) -> dict[str, Any]:
        report: dict[str, Any] = {"agent": self.name, "detected": self.detected(), "config": str(self.config_path()) if self.config_path() else None}
        if self.ctx.env:
            report["env"] = dict(self.ctx.env)
        try:
            self._register(report)
        except Exception as exc:
            report.update(action="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            report["skill"] = _install_skill(self.ctx, self.skill_dir())
        except Exception as exc:
            report["skill"] = {"action": "failed", "error": str(exc)}
        return report

    def uninstall(self) -> dict[str, Any]:
        report: dict[str, Any] = {"agent": self.name, "detected": self.detected(), "config": str(self.config_path()) if self.config_path() else None}
        try:
            self._unregister(report)
        except Exception as exc:
            report.update(action="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            report["skill"] = self.uninstall_skill()
        except Exception as exc:
            report["skill"] = {"action": "failed", "error": str(exc)}
        return report


# --------------------------------------------------------------------------- JSON-config targets


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


class JsonTarget(Target):
    """``{"mcpServers": {"sqgraph-memory": {"command", "args"}}}`` merged into a JSON file."""

    def current(self) -> dict[str, Any] | None:
        p = self.config_path()
        if p is None:
            return None
        servers = _read_json_object(p).get("mcpServers")
        if isinstance(servers, dict) and isinstance(servers.get(SERVER_NAME), dict):
            return servers[SERVER_NAME]
        return None

    def _register(self, report: dict[str, Any]) -> None:
        p = self.config_path()
        assert p is not None
        data = _read_json_object(p)  # raises on invalid JSON: the file is left untouched
        servers = data.get("mcpServers")
        if servers is None:
            servers = data["mcpServers"] = {}
        if not isinstance(servers, dict):
            raise ValueError("'mcpServers' is not an object")
        cur = servers.get(SERVER_NAME)
        if isinstance(cur, dict) and self._matches(cur):
            report["action"] = "unchanged"
            return
        entry = dict(cur) if isinstance(cur, dict) else {}  # keep user additions (env, ...)
        entry["command"] = self.ctx.command[0]
        entry["args"] = self.ctx.command[1:]
        if self.ctx.env:  # merge with user-added env; ours win for our keys
            entry["env"] = {**(entry.get("env") if isinstance(entry.get("env"), dict) else {}), **self.ctx.env}
        servers[SERVER_NAME] = entry
        report["action"] = "would-register" if self.ctx.dry_run else ("updated" if cur is not None else "registered")
        if not self.ctx.dry_run:
            report["backup"] = _backup(p)
            _atomic_write(p, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def _unregister(self, report: dict[str, Any]) -> None:
        p = self.config_path()
        assert p is not None
        data = _read_json_object(p)
        servers = data.get("mcpServers")
        if not (isinstance(servers, dict) and SERVER_NAME in servers):
            report["action"] = "not-registered"
            return
        report["action"] = "would-remove" if self.ctx.dry_run else "removed"
        if not self.ctx.dry_run:
            del servers[SERVER_NAME]
            report["backup"] = _backup(p)
            _atomic_write(p, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class CursorTarget(JsonTarget):
    name = "cursor"

    def detected(self) -> bool:
        return (self.ctx.home / ".cursor").is_dir() or bool(self.ctx.which("cursor"))

    def config_path(self) -> Path:
        return self.ctx.home / ".cursor" / "mcp.json"

    def skill_dir(self) -> Path:
        return self.ctx.home / ".cursor" / "skills" / SERVER_NAME


def desktop_config_path(home: Path, platform: str, appdata: Path | None = None) -> Path:
    if platform == "win32":
        base = appdata or (Path(os.environ["APPDATA"]) if os.environ.get("APPDATA") else home / "AppData" / "Roaming")
        std = base / "Claude" / "claude_desktop_config.json"
        if not std.parent.is_dir():  # Microsoft Store installs live under Packages\Claude_*
            pk = home / "AppData" / "Local" / "Packages"
            if pk.is_dir():
                for d in sorted(pk.glob("Claude_*")):
                    cand = d / "LocalCache" / "Roaming" / "Claude"
                    if cand.is_dir():
                        return cand / "claude_desktop_config.json"
        return std
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return home / ".config" / "Claude" / "claude_desktop_config.json"


class DesktopTarget(JsonTarget):
    name = "claude-desktop"

    def config_path(self) -> Path:
        return desktop_config_path(self.ctx.home, self.ctx.platform, self.ctx.appdata)

    def detected(self) -> bool:
        return self.config_path().parent.is_dir()


# --------------------------------------------------------------------------- Claude Code


class CommandHooks:
    """Command hooks in a JSON file shaped like Claude Code's settings (``{"hooks": {Event: [{"hooks": [...]}]}}``),
    which Codex's ``hooks.json`` shares. Mixed into a :class:`Target` that defines :meth:`hooks_file`."""

    ctx: Context
    name: str

    def hooks_file(self) -> Path:
        raise NotImplementedError

    def hook_handler(self, event: str) -> dict[str, Any]:
        command = hook_command(self.ctx.command, event, self.ctx.env, self.ctx.platform, agent=self.name)
        return {"type": "command", "command": command, "timeout": HOOKS[event][1]}

    def hooks_status(self) -> list[str] | None:
        try:
            return sorted(installed_hooks(_read_json_object(self.hooks_file())))
        except Exception:  # noqa: BLE001 - unreadable file: nothing of ours is known to be there
            return []

    def install(self) -> dict[str, Any]:
        report = super().install()  # type: ignore[misc]
        report["hooks"] = self._set_hooks(self.ctx.hooks) if self.ctx.hooks else {"action": "skipped"}
        return report

    def uninstall(self) -> dict[str, Any]:
        report = super().uninstall()  # type: ignore[misc]
        report["hooks"] = self._set_hooks(())
        return report

    def _set_hooks(self, events: tuple[str, ...]) -> dict[str, Any]:
        p = self.hooks_file()
        try:
            data = _read_json_object(p)  # raises on invalid JSON: the file is left untouched
            want = {ev: self.hook_handler(ev) for ev in events}
            new = set_hooks(data, want)
            if new is None:
                return {"action": "unchanged" if want else "not-installed", "path": str(p), "events": sorted(want)}
            if self.ctx.dry_run:
                action = "would-install" if want else "would-remove"
            else:
                action = "installed" if want else "removed"
            res: dict[str, Any] = {"action": action, "path": str(p), "events": sorted(want)}
            if not self.ctx.dry_run:
                res["backup"] = _backup(p)
                _atomic_write(p, json.dumps(new, ensure_ascii=False, indent=2) + "\n")
            note = self.hooks_note(bool(want))
            if note:
                res["note"] = note
            return res
        except Exception as exc:  # noqa: BLE001
            return {"action": "failed", "path": str(p), "error": f"{type(exc).__name__}: {exc}"}

    def hooks_note(self, installing: bool) -> str | None:
        return None


class ClaudeCodeTarget(CommandHooks, Target):
    name = "claude-code"

    def detected(self) -> bool:
        return bool(self.ctx.which("claude")) or (self.ctx.home / ".claude").is_dir()

    def config_path(self) -> Path:
        return self.ctx.home / ".claude.json"

    def skill_dir(self) -> Path:
        return self.ctx.home / ".claude" / "skills" / SERVER_NAME

    def current(self) -> dict[str, Any] | None:
        p = self.config_path()
        if not p.is_file():
            return None
        servers = _read_json_object(p).get("mcpServers")
        if isinstance(servers, dict) and isinstance(servers.get(SERVER_NAME), dict):
            return servers[SERVER_NAME]
        return None

    def _add_argv(self, claude: str) -> list[str]:
        env_args = [x for k, v in self.ctx.env.items() for x in ("-e", f"{k}={v}")]
        # the name goes before -e (it takes a variadic list); `--` ends the options
        return [claude, "mcp", "add", "--scope", "user", SERVER_NAME, *env_args, "--", *self.ctx.command]

    def _register(self, report: dict[str, Any]) -> None:
        cur = self.current()
        claude = self.ctx.which("claude")
        manual = format_command(self._add_argv("claude"), self.ctx.platform)
        if cur is not None and self._matches(cur):
            report["action"] = "unchanged"
            return
        if not claude:
            report.update(action="manual", manual_command=manual, note="`claude` is not on PATH; run the command yourself")
            return
        if self.ctx.dry_run:
            report.update(action="would-register", manual_command=manual)
            return
        if cur is not None:  # replace a stale registration
            self.ctx.runner([claude, "mcp", "remove", "--scope", "user", SERVER_NAME])
        code, out, err = self.ctx.runner(self._add_argv(claude))
        if code != 0:
            report.update(action="failed", manual_command=manual, error=(err or out).strip()[-400:])
        else:
            report["action"] = "updated" if cur is not None else "registered"

    def hooks_file(self) -> Path:
        return self.ctx.home / ".claude" / "settings.json"

    def _unregister(self, report: dict[str, Any]) -> None:
        cur = self.current()
        claude = self.ctx.which("claude")
        argv = [claude or "claude", "mcp", "remove", "--scope", "user", SERVER_NAME]
        if cur is None:
            report["action"] = "not-registered"
            return
        if not claude:
            report.update(action="manual", manual_command=format_command(argv, self.ctx.platform))
            return
        if self.ctx.dry_run:
            report.update(action="would-remove", manual_command=format_command(argv, self.ctx.platform))
            return
        code, out, err = self.ctx.runner(argv)
        if code != 0:
            report.update(action="failed", error=(err or out).strip()[-400:])
        else:
            report["action"] = "removed"


def hook_argv(command: list[str], event: str, env: dict[str, str] | None = None, agent: str | None = None) -> list[str]:
    """The server launch command with ``hook <event> [--agent A] [--env K=V ...]`` instead of ``mcp``."""
    base = command[:-1] if command and command[-1] == "mcp" else list(command)
    tail = ["--agent", agent] if agent and agent != "claude-code" else []
    return [*base, "hook", event, *tail, *[x for k, v in (env or {}).items() for x in ("--env", f"{k}={v}")]]


def hook_command(
    command: list[str], event: str, env: dict[str, str] | None = None, platform: str | None = None, agent: str | None = None
) -> str:
    """The shell command an agent's hook runs (:func:`hook_argv` as one command line).

    On Windows, drive paths use forward slashes (valid for both Git Bash and cmd.exe, either of which may run hooks)."""
    plat = platform or sys.platform
    argv = hook_argv(command, event, env, agent)
    if plat == "win32":
        argv = [a.replace("\\", "/") if re.match(r"^[A-Za-z]:[\\/]", a) else a for a in argv]
    return format_command(argv, plat)


def _is_our_hook(h: Any) -> bool:
    cmd = h.get("command") if isinstance(h, dict) else None
    return isinstance(cmd, str) and "sqgraph" in cmd.lower() and bool(_OUR_HOOK_RE.search(cmd))


def installed_hooks(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Our hook handlers in a Claude Code settings object, by our event name."""
    out: dict[str, list[dict[str, Any]]] = {}
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return out
    for ev, (cc_event, _) in HOOKS.items():
        found: list[dict[str, Any]] = []
        groups = hooks.get(cc_event)
        for group in groups if isinstance(groups, list) else []:
            if isinstance(group, dict):
                found += [h for h in group.get("hooks") or [] if _is_our_hook(h)]
        if found:
            out[ev] = found
    return out


def set_hooks(data: dict[str, Any], want: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """A copy of ``data`` whose only sqgraph hooks are ``want`` (our event -> handler), every other hook kept as is;
    ``None`` when nothing changes. Our handlers get matcher groups of their own (no matcher: every occurrence)."""
    if {ev: [h] for ev, h in want.items()} == installed_hooks(data):
        return None
    new = json.loads(json.dumps(data))
    hooks = new.get("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("'hooks' in settings.json is not an object")
    for ev, (cc_event, _) in HOOKS.items():
        groups = hooks.get(cc_event) or []
        if not isinstance(groups, list):
            raise ValueError(f"'hooks.{cc_event}' in settings.json is not a list")
        kept: list[Any] = []
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("hooks"), list) and any(_is_our_hook(h) for h in group["hooks"]):
                rest = [h for h in group["hooks"] if not _is_our_hook(h)]
                if rest:
                    kept.append({**group, "hooks": rest})
            else:
                kept.append(group)
        if ev in want:
            kept.append({"hooks": [want[ev]]})
        if kept:
            hooks[cc_event] = kept
        else:
            hooks.pop(cc_event, None)
    if hooks:
        new["hooks"] = hooks
    else:
        new.pop("hooks", None)
    return new


# --------------------------------------------------------------------------- Codex (TOML, textual)

_TRIPLE = re.compile(r"\"\"\"|'''")


_HEADER_LINE = re.compile(r"^\s*\[\[?\s*([A-Za-z0-9_\-\"' .]+?)\s*\]\]?\s*(?:#.*)?$")


def _norm_header(line: str) -> str | None:
    """Normalised dotted key of a ``[table]`` / ``[[array]]`` header line, else ``None``."""
    m = _HEADER_LINE.match(line.rstrip("\r\n"))
    if not m:
        return None
    parts = re.findall(r"\"[^\"]*\"|'[^']*'|[^.\s]+", m.group(1))
    return ".".join(p.strip("\"'") for p in parts)


def toml_blocks(text: str) -> list[tuple[str | None, list[str]]]:
    """Split TOML text into ``(header_key or None for the preamble, lines)`` blocks (line endings kept)."""
    blocks: list[tuple[str | None, list[str]]] = [(None, [])]
    in_multi: str | None = None
    for line in text.splitlines(keepends=True):
        if in_multi is None:
            h = _norm_header(line)
            if h is not None:
                blocks.append((h, [line]))
                continue
        for m in _TRIPLE.finditer(line):
            tok = m.group(0)
            if in_multi is None:
                in_multi = tok
            elif in_multi == tok:
                in_multi = None
        blocks[-1][1].append(line)
    return blocks


def _toml_str(s: str) -> str:
    if "'" not in s and "\n" not in s and "\r" not in s:
        return f"'{s}'"
    return json.dumps(s, ensure_ascii=False)


def _is_ours(key: str | None) -> bool:
    base = f"mcp_servers.{SERVER_NAME}"
    return key is not None and (key == base or key.startswith(base + "."))


def _strip_keys(lines: list[str], keys: tuple[str, ...]) -> list[str]:
    """Drop ``key = value`` lines (and continuation lines of multi-line arrays) for ``keys``."""
    out: list[str] = []
    skipping = False
    depth = 0
    for ln in lines:
        if skipping:
            depth += ln.count("[") - ln.count("]")
            if depth <= 0:
                skipping = False
            continue
        m = re.match(r"\s*([A-Za-z0-9_-]+)\s*=\s*(.*)$", ln)
        if m and m.group(1) in keys:
            v = m.group(2)
            depth = v.count("[") - v.count("]")
            skipping = depth > 0
            continue
        out.append(ln)
    return out


def _toml_kv_lines(env: dict[str, str], nl: str) -> list[str]:
    return [f"{k} = {_toml_str(v)}{nl}" for k, v in env.items()]


def codex_add(text: str, command: list[str], env: dict[str, str] | None = None) -> str:
    """Return ``text`` with our ``[mcp_servers.sqgraph-memory]`` table added / its command and args updated.

    ``env`` goes into the ``[mcp_servers.sqgraph-memory.env]`` sub-table (other keys already there are kept).
    """
    base = f"mcp_servers.{SERVER_NAME}"
    env_key = base + ".env"
    env = env or {}
    nl = "\r\n" if "\r\n" in text else "\n"
    new_kv = [
        f"command = {_toml_str(command[0])}{nl}",
        f"args = [{', '.join(_toml_str(a) for a in command[1:])}]{nl}",
    ]
    blocks = toml_blocks(text)
    main = next((i for i, (k, _) in enumerate(blocks) if k == base), None)
    if main is None:
        body = text
        if body and not body.endswith("\n"):
            body += nl
        if body.strip():
            body += nl
        body += f"[{base}]{nl}" + "".join(new_kv) + f"startup_timeout_sec = {STARTUP_TIMEOUT_SEC}{nl}"
        if env:
            body += f"{nl}[{env_key}]{nl}" + "".join(_toml_kv_lines(env, nl))
        return body
    key, lines = blocks[main]
    if any(re.match(r"\s*env\s*=", ln) for ln in lines[1:]):
        raise ValueError("the existing table uses an inline `env = {...}`; edit it by hand or remove the table first")
    kept = _strip_keys(lines[1:], ("command", "args"))
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += nl
    blocks[main] = (key, [lines[0], *new_kv, *kept])
    if env:
        ei = next((i for i, (k, _) in enumerate(blocks) if k == env_key), None)
        if ei is None:
            blocks.insert(main + 1, (env_key, [f"{nl}[{env_key}]{nl}", *_toml_kv_lines(env, nl)]))
        else:
            ek, el = blocks[ei]
            rest = _strip_keys(el[1:], tuple(env))
            if rest and not rest[-1].endswith("\n"):
                rest[-1] += nl
            blocks[ei] = (ek, [el[0], *_toml_kv_lines(env, nl), *rest])
    return "".join("".join(ls) for _, ls in blocks)


def codex_remove(text: str) -> str:
    """Return ``text`` without our table and its sub-tables (everything else is preserved verbatim)."""
    blocks = [(k, ls) for k, ls in toml_blocks(text) if not _is_ours(k)]
    out = "".join("".join(ls) for _, ls in blocks)
    return out.rstrip("\r\n") + "\n" if out.strip() else ""


def codex_current(text: str) -> dict[str, Any] | None:
    base = f"mcp_servers.{SERVER_NAME}"
    env: dict[str, str] = {}
    for k, ls in toml_blocks(text):
        if k == base + ".env":
            for ln in ls[1:]:
                m = re.match(r"\s*([A-Za-z0-9_-]+)\s*=\s*(.+?)\s*$", ln)
                if m:
                    env[m.group(1)] = _unquote_toml(m.group(2))
    for k, ls in toml_blocks(text):
        if k == base:
            body = "".join(ls)
            try:
                import tomllib  # type: ignore[import-not-found]  # Python 3.11+

                data = tomllib.loads(body)
                srv = data["mcp_servers"][SERVER_NAME]
                return {"command": srv.get("command"), "args": list(srv.get("args", [])), "env": env}
            except Exception:
                pass
            cm = re.search(r"^\s*command\s*=\s*(.+?)\s*$", body, re.M)
            am = re.search(r"^\s*args\s*=\s*(\[.*?\])\s*$", body, re.M | re.S)
            cmd = _unquote_toml(cm.group(1)) if cm else None
            args: list[str] = []
            if am:
                args = [_unquote_toml(x) for x in re.findall(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"", am.group(1))]
            return {"command": cmd, "args": args, "env": env}
    return None


def _unquote_toml(v: str) -> str:
    v = v.strip()
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    try:
        return json.loads(v)
    except ValueError:
        return v.strip("\"'")


class CodexTarget(CommandHooks, Target):
    name = "codex"

    def hooks_file(self) -> Path:
        return self.ctx.home / ".codex" / "hooks.json"

    def hooks_note(self, installing: bool) -> str | None:
        if installing and not self.ctx.dry_run:
            return "Codex runs new hooks only after you trust them: start Codex, run /hooks and trust the sqgraph-memory hooks"
        return None

    def detected(self) -> bool:
        return (self.ctx.home / ".codex").is_dir() or bool(self.ctx.which("codex"))

    def config_path(self) -> Path:
        return self.ctx.home / ".codex" / "config.toml"

    def skill_dir(self) -> Path:
        return self.ctx.home / ".agents" / "skills" / SERVER_NAME

    def _read(self) -> str:
        p = self.config_path()
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def current(self) -> dict[str, Any] | None:
        return codex_current(self._read())

    def _register(self, report: dict[str, Any]) -> None:
        text = self._read()
        cur = codex_current(text)
        if cur is not None and self._matches(cur):
            report["action"] = "unchanged"
            return
        new = codex_add(text, self.ctx.command, self.ctx.env)
        report["action"] = "would-register" if self.ctx.dry_run else ("updated" if cur is not None else "registered")
        if not self.ctx.dry_run:
            report["backup"] = _backup(self.config_path())
            _atomic_write(self.config_path(), new)

    def _unregister(self, report: dict[str, Any]) -> None:
        text = self._read()
        if not any(_is_ours(k) for k, _ in toml_blocks(text)):
            report["action"] = "not-registered"
            return
        report["action"] = "would-remove" if self.ctx.dry_run else "removed"
        if not self.ctx.dry_run:
            report["backup"] = _backup(self.config_path())
            _atomic_write(self.config_path(), codex_remove(text))


# --------------------------------------------------------------------------- OpenCode


def strip_jsonc(text: str) -> str:
    """JSON text from JSONC: ``//`` and ``/* */`` comments and trailing commas removed (strings are kept intact)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] not in "\r\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def has_jsonc_comments(text: str) -> bool:
    return strip_jsonc(text) != re.sub(r",(\s*[}\]])", r"\1", text)


def _read_jsonc_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        return {}
    data = json.loads(strip_jsonc(raw))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def opencode_plugin_text(command: list[str], env: dict[str, str] | None = None) -> str:
    """The packaged OpenCode plugin with the launch command (and forwarded environment) filled in."""
    from importlib import resources

    base = command[:-1] if command and command[-1] == "mcp" else list(command)
    text = resources.files("sqgraph.memory").joinpath("opencode", OPENCODE_PLUGIN).read_text(encoding="utf-8")
    text = text.replace("__SQGRAPH_COMMAND__", json.dumps(base, ensure_ascii=False))
    return text.replace("__SQGRAPH_ENV__", json.dumps(env or {}, ensure_ascii=False))


class OpenCodeTarget(Target):
    """MCP in ``opencode.json`` (our own file: a commented ``opencode.jsonc`` is merged by OpenCode, never rewritten),
    the Skill shared with Codex in ``~/.agents/skills``, and the hooks as a plugin."""

    name = "opencode"
    CONFIG_FILES = ("config.json", "opencode.json", "opencode.jsonc")  # the order OpenCode merges them in

    def config_dir(self) -> Path:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        return (Path(xdg) if xdg and self.ctx.home == Path.home() else self.ctx.home / ".config") / "opencode"

    def detected(self) -> bool:
        return self.config_dir().is_dir() or bool(self.ctx.which("opencode"))

    def config_path(self) -> Path:
        return self.config_dir() / "opencode.json"

    def skill_dir(self) -> Path:
        return self.ctx.home / ".agents" / "skills" / SERVER_NAME

    def plugin_path(self) -> Path:
        return self.config_dir() / "plugins" / OPENCODE_PLUGIN

    def _entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"type": "local", "command": list(self.ctx.command), "enabled": True, "timeout": STARTUP_TIMEOUT_SEC * 1000}
        if self.ctx.env:
            entry["environment"] = dict(self.ctx.env)
        return entry

    def _found(self) -> tuple[Path, dict[str, Any]] | None:
        """The file that defines our server last (the effective one) and its entry."""
        hit = None
        for name in self.CONFIG_FILES:
            p = self.config_dir() / name
            mcp = _read_jsonc_object(p).get("mcp")
            if isinstance(mcp, dict) and isinstance(mcp.get(SERVER_NAME), dict):
                hit = (p, mcp[SERVER_NAME])
        return hit

    def current(self) -> dict[str, Any] | None:
        found = self._found()
        if found is None:
            return None
        e = found[1]
        cmd = e.get("command") if isinstance(e.get("command"), list) else []
        return {"command": cmd[0] if cmd else None, "args": cmd[1:], "env": e.get("environment") or {}, "file": str(found[0])}

    def _register(self, report: dict[str, Any]) -> None:
        p = self.config_path()
        found = self._found()
        if found is not None and found[0] != p:  # the user keeps it in another file (e.g. a commented opencode.jsonc)
            if self._matches(self.current()):
                report["action"] = "unchanged"
                return
            report.update(
                action="manual",
                note=f"{found[0]} defines mcp.{SERVER_NAME}; update it there (it overrides opencode.json)",
                manual_entry={SERVER_NAME: self._entry()},
            )
            return
        raw = p.read_text(encoding="utf-8-sig") if p.is_file() else ""
        if raw.strip() and has_jsonc_comments(raw):
            report.update(
                action="manual",
                note=f"{p} contains comments, which a rewrite would lose: add this under \"mcp\" yourself",
                manual_entry={SERVER_NAME: self._entry()},
            )
            return
        data = _read_jsonc_object(p) if raw.strip() else {"$schema": "https://opencode.ai/config.json"}
        mcp = data.get("mcp")
        if mcp is None:
            mcp = data["mcp"] = {}
        if not isinstance(mcp, dict):
            raise ValueError("'mcp' is not an object")
        cur = mcp.get(SERVER_NAME)
        entry = {**(cur if isinstance(cur, dict) else {}), **self._entry()}
        if self.ctx.env and isinstance(cur, dict) and isinstance(cur.get("environment"), dict):
            entry["environment"] = {**cur["environment"], **self.ctx.env}
        if cur == entry:
            report["action"] = "unchanged"
            return
        mcp[SERVER_NAME] = entry
        report["action"] = "would-register" if self.ctx.dry_run else ("updated" if cur is not None else "registered")
        if not self.ctx.dry_run:
            report["backup"] = _backup(p)
            _atomic_write(p, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def _unregister(self, report: dict[str, Any]) -> None:
        p = self.config_path()
        found = self._found()
        if found is None:
            report["action"] = "not-registered"
            return
        if found[0] != p:
            report.update(action="manual", note=f"remove mcp.{SERVER_NAME} from {found[0]} yourself")
            return
        raw = p.read_text(encoding="utf-8-sig")
        if has_jsonc_comments(raw):
            report.update(action="manual", note=f"{p} contains comments: remove mcp.{SERVER_NAME} yourself")
            return
        data = _read_jsonc_object(p)
        report["action"] = "would-remove" if self.ctx.dry_run else "removed"
        if not self.ctx.dry_run:
            del data["mcp"][SERVER_NAME]
            if not data["mcp"]:
                del data["mcp"]
            report["backup"] = _backup(p)
            if set(data) <= {"$schema"}:  # nothing but ours was there: the file we created goes
                p.unlink()
            else:
                _atomic_write(p, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def hooks_status(self) -> list[str] | None:
        return sorted(HOOKS) if self.plugin_path().is_file() else []

    def install(self) -> dict[str, Any]:
        report = super().install()
        report["hooks"] = self._install_plugin() if self.ctx.hooks else {"action": "skipped"}
        return report

    def uninstall_skill(self) -> dict[str, Any]:
        if CodexTarget(self.ctx).current() is not None:  # the skill directory is shared: keep it while Codex uses it
            return {"action": "kept", "path": str(_skill_path(self.skill_dir())), "note": "shared with Codex"}
        return super().uninstall_skill()

    def uninstall(self) -> dict[str, Any]:
        report = super().uninstall()
        p = self.plugin_path()
        if not p.is_file():
            report["hooks"] = {"action": "not-installed", "path": str(p)}
        elif self.ctx.dry_run:
            report["hooks"] = {"action": "would-remove", "path": str(p)}
        else:
            p.unlink()
            report["hooks"] = {"action": "removed", "path": str(p)}
        return report

    def _install_plugin(self) -> dict[str, Any]:
        p = self.plugin_path()
        try:
            text = opencode_plugin_text(self.ctx.command, self.ctx.env)
            cur = p.read_text(encoding="utf-8") if p.is_file() else None
            if cur == text:
                return {"action": "unchanged", "path": str(p), "events": sorted(HOOKS)}
            res: dict[str, Any] = {"action": "would-install" if self.ctx.dry_run else "installed", "path": str(p), "events": sorted(HOOKS)}
            if not self.ctx.dry_run:
                _atomic_write(p, text)
            return res
        except Exception as exc:  # noqa: BLE001
            return {"action": "failed", "path": str(p), "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- public API

TARGETS: dict[str, type[Target]] = {
    "claude-code": ClaudeCodeTarget,
    "codex": CodexTarget,
    "opencode": OpenCodeTarget,
    "cursor": CursorTarget,
    "claude-desktop": DesktopTarget,
}


def parse_agents(spec: str | None) -> list[str] | None:
    """``None`` = every detected agent; otherwise validated names (``all`` = every known agent)."""
    if spec is None:
        return None
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if names == ["all"] or "all" in names:
        return list(AGENTS)
    bad = [n for n in names if n not in TARGETS]
    if bad:
        raise ValueError(f"unknown agent(s): {', '.join(bad)} (choose from {', '.join(AGENTS)} or all)")
    return names


def make_targets(ctx: Context, agents: list[str] | None = None, *, detected_only: bool = False) -> list[Target]:
    names = agents if agents is not None else list(AGENTS)
    targets = [TARGETS[n](ctx) for n in names]
    return [t for t in targets if t.detected()] if detected_only else targets


def run_setup(
    agents: list[str] | None,
    *,
    home: Path,
    command: list[str],
    install_skill: bool = True,
    dry_run: bool = False,
    uninstall: bool = False,
    runner: Runner = default_runner,
    which: Callable[[str], str | None] = shutil.which,
    platform: str | None = None,
    appdata: Path | None = None,
    warnings: list[str] | None = None,
    env: dict[str, str] | None = None,
    hooks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Install or uninstall on ``agents`` (``None`` = those detected on this machine). Returns a JSON-able report.

    ``hooks``: the Claude Code hooks to install (default: all of ``HOOKS``; empty: leave hooks as they are)."""
    ctx = Context(
        home=home,
        command=command,
        dry_run=dry_run,
        install_skill=install_skill,
        runner=runner,
        which=which,
        platform=platform or sys.platform,
        appdata=appdata,
        env=dict(env or {}),
        hooks=tuple(HOOKS) if hooks is None else tuple(hooks),
    )
    targets = make_targets(ctx, agents, detected_only=agents is None)
    reports = [t.uninstall() if uninstall else t.install() for t in targets]
    out: dict[str, Any] = {
        "mode": "uninstall" if uninstall else "install",
        "dry_run": dry_run,
        "server": SERVER_NAME,
        "command": command,
        "agents": reports,
    }
    if env:
        out["env"] = dict(env)
    if warnings:
        out["warnings"] = list(warnings)
    if agents is None and not targets:
        out["note"] = "no supported agent detected; pass --agents claude-code,codex,opencode,cursor,claude-desktop"
    return out


def agents_status(
    *,
    home: Path,
    command: list[str],
    which: Callable[[str], str | None] = shutil.which,
    platform: str | None = None,
    appdata: Path | None = None,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    ctx = Context(home=home, command=command, which=which, platform=platform or sys.platform, appdata=appdata, env=dict(env or {}))
    return [t.status() for t in make_targets(ctx)]
