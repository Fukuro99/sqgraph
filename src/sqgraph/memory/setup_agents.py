"""Register ``sqgraph-memory`` as an MCP server (and install the Agent Skill) for local agents.

Targets and the formats written (verified against each product's documentation):

* **Claude Code** -- ``claude mcp add --scope user sqgraph-memory -- <cmd> mcp`` (stored in ``~/.claude.json``);
  skill: ``~/.claude/skills/sqgraph-memory/SKILL.md``.
* **Codex CLI** -- ``~/.codex/config.toml``, a ``[mcp_servers.sqgraph-memory]`` table with ``command`` / ``args``
  (https://developers.openai.com/codex/mcp); skill: ``~/.agents/skills/sqgraph-memory/SKILL.md``
  (https://developers.openai.com/codex/skills).
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
from typing import Any, Callable

SERVER_NAME = "sqgraph-memory"
EXE_NAME = "sqgraph-memory"
AGENTS = ("claude-code", "codex", "cursor", "claude-desktop")
STARTUP_TIMEOUT_SEC = 30  # Codex default is 10s; the first start imports sqlite-vec / openai / mcp

Runner = Callable[[list[str]], "tuple[int, str, str]"]


# --------------------------------------------------------------------------- helpers

FORWARD_ENV = ("SQGRAPH_MEMORY_HOME", "SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP")


def forwarded_env(environ: Any = None) -> dict[str, str]:
    """The ``SQGRAPH_MEMORY_*`` variables set right now (paths made absolute): agents start the server
    without our shell environment, so they must be written into each registration."""
    src = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for k in FORWARD_ENV:
        v = src.get(k)
        if not v:
            continue
        out[k] = v if k == "SQGRAPH_MEMORY_GROUP" else os.path.abspath(os.path.expanduser(v))
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
        return out

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
            report["skill"] = _uninstall_skill(self.ctx, self.skill_dir())
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


class ClaudeCodeTarget(Target):
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


class CodexTarget(Target):
    name = "codex"

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


# --------------------------------------------------------------------------- public API

TARGETS: dict[str, type[Target]] = {
    "claude-code": ClaudeCodeTarget,
    "codex": CodexTarget,
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
) -> dict[str, Any]:
    """Install or uninstall on ``agents`` (``None`` = those detected on this machine). Returns a JSON-able report."""
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
        out["note"] = "no supported agent detected; pass --agents claude-code,codex,cursor,claude-desktop"
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
