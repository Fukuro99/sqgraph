"""``sqgraph-memory``: personal long-term memory CLI (also ``python -m sqgraph.memory``).

All output is JSON on stdout (UTF-8); errors are ``{"error": ...}`` on stderr with a non-zero exit code.

    sqgraph-memory init [--api-key-stdin] [--group-id G] [--tz TZ]
    sqgraph-memory add TEXT|- [--source claude-code] [--channel personal] ...
    sqgraph-memory search QUERY [--limit 10] [--as-of DATE] [--history] [--center ENTITY]
    sqgraph-memory entity NAME | entities QUERY | episodes | stats
    sqgraph-memory forget (--episode ID | --source-ref REF | --fact ID | --entity NAME | --since/--until/--channel) [--dry-run]
    sqgraph-memory purge --yes [--dry-run]
    sqgraph-memory mcp [--transport stdio|streamable-http] [--read-only]
    sqgraph-memory setup [--agents ...] [--uninstall] | doctor [--online]

The home directory (``$SQGRAPH_MEMORY_HOME`` or ``~/.sqgraph-memory``) holds ``memory.db`` -- one SQLite file shared
by every agent on the machine -- plus ``.env`` (OPENAI_API_KEY) and ``config.json``. See :mod:`sqgraph.memory.config`.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NoReturn, Sequence

from . import setup_agents
from .config import KEY_NAME, MemoryConfig, ensure_private_dir
from .core import FACT_INDEX, Memory
from .providers import DEFAULT_EMBED_DIMS, Embedder, LLMProvider, OpenAIEmbedder, OpenAILLM, UsageTracker, load_env

EXTRAS_HINT = 'pip install "sqgraph[mcp]"   (or: uv tool install "sqgraph[mcp]")'


class CliError(Exception):
    """A user-facing error: printed as JSON on stderr."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliError(f"{self.prog}: {message}", 2)


# --------------------------------------------------------------------------- output


def _utf8() -> None:
    """Windows consoles default to cp932 etc., which garbles Japanese: force UTF-8 on all three streams."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def emit(obj: Any, compact: bool = False) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=None if compact else 2, default=str) + "\n")
    sys.stdout.flush()


def emit_error(message: str) -> None:
    sys.stderr.write(json.dumps({"error": message}, ensure_ascii=False) + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- memory construction


def user_home() -> Path:
    """The user's home directory (agent configs live below it). Monkeypatched in tests."""
    return Path.home()


def _existing_dims(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        from ..db import GraphDB

        with GraphDB(str(path), read_only=True) as db:
            for ix in db.vector_indexes():
                if ix.name == FACT_INDEX:
                    return ix.dimensions
    except Exception:
        return None
    return None


def make_providers(
    cfg: MemoryConfig, *, need_llm: bool, need_embedder: bool
) -> tuple[LLMProvider | None, Embedder | None, list[str]]:
    """Build the OpenAI providers from ``cfg`` (+ ``<home>/.env``). Tests monkeypatch this with fakes.

    Without an API key the embedder is ``None`` (fulltext + entity search still work) and a warning is returned;
    ``need_llm`` without a key is an error.
    """
    warnings: list[str] = []
    load_env(cfg.env_path)  # existing environment variables win
    if not os.environ.get(KEY_NAME):
        if need_llm:
            raise CliError(f"{KEY_NAME} is not set: run `sqgraph-memory init` (or set the environment variable)")
        if need_embedder:
            warnings.append(f"{KEY_NAME} is not set: semantic (vector) search is disabled; run `sqgraph-memory init`")
        return None, None, warnings
    if importlib.util.find_spec("openai") is None:
        raise CliError(f'the "openai" package is missing: {EXTRAS_HINT}')
    tracker = UsageTracker()
    llm = OpenAILLM(cfg.llm_model, tracker=tracker) if need_llm else None
    embedder = None
    if need_llm or need_embedder:
        dims = _existing_dims(cfg.db_path) or DEFAULT_EMBED_DIMS
        embedder = OpenAIEmbedder(cfg.embed_model, dimensions=dims, tracker=tracker)
    return llm, embedder, warnings


@contextmanager
def open_memory(
    cfg: MemoryConfig, *, need_llm: bool = False, need_embedder: bool = False
) -> Iterator[tuple[Memory, list[str]]]:
    ensure_private_dir(cfg.home)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    llm, embedder, warnings = make_providers(cfg, need_llm=need_llm, need_embedder=need_embedder)
    mem = Memory(cfg.db_path, llm, embedder, default_group_id=cfg.group_id, tz=cfg.tz)
    try:
        yield mem, warnings
    finally:
        mem.close()


def _with_warning(out: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if warnings:
        out["warning"] = "; ".join(warnings)
    return out


# --------------------------------------------------------------------------- commands


def cmd_init(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    cfg = MemoryConfig.load(cfg.home, apply_env=False)  # persist stored values, not env overrides
    if args.group_id:
        cfg.group_id = args.group_id
    if args.tz:
        try:
            from .util import resolve_tz

            resolve_tz(args.tz)
        except Exception as exc:
            raise CliError(f"invalid timezone {args.tz!r}: {exc}", 2) from exc
        cfg.tz = args.tz
    cfg.save()

    key_state = "missing"
    key: str | None = None
    if args.api_key_stdin:
        key = (sys.stdin.readline() or "").strip()
        if not key:
            raise CliError("--api-key-stdin: no key received on stdin", 2)
        key_state = "stored"
    elif os.environ.get(KEY_NAME):
        key, key_state = os.environ[KEY_NAME].strip(), "stored-from-env"
    elif MemoryConfig.load(cfg.home).has_api_key():
        key_state = "existing"
    elif sys.stdin is not None and sys.stdin.isatty():
        key = getpass.getpass("OpenAI API key (input hidden, saved to %s): " % cfg.env_path).strip()
        key_state = "stored" if key else "missing"
    if key:
        cfg.write_api_key(key)

    db = MemoryConfig.load(cfg.home)  # with env overrides: where the DB really lives
    out: dict[str, Any] = {
        "home": str(cfg.home),
        "config": str(cfg.config_path),
        "db": str(db.db_path),
        "db_exists": db.db_path.is_file(),  # created on the first `add`
        "group_id": db.group_id,
        "tz": cfg.tz,
        "api_key": key_state,
    }
    if key_state == "missing":
        out["warning"] = (
            f"no {KEY_NAME} configured: set the env var and re-run `sqgraph-memory init`, or pipe it: "
            "`... | sqgraph-memory init --api-key-stdin`"
        )
    out["next"] = "sqgraph-memory setup   # register with Claude Code / Codex / Cursor / Claude Desktop"
    return out


def cmd_add(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    text = sys.stdin.read() if args.text == "-" else args.text
    if not text or not text.strip():
        raise CliError("text must not be empty", 2)
    with open_memory(cfg, need_llm=True) as (mem, _):
        res = mem.add_episode(
            text,
            group_id=cfg.group_id,
            source=args.source,
            channel=args.channel,
            author=args.author,
            occurred_at=args.occurred_at,
            source_ref=args.source_ref,
        )
    return {
        "group_id": cfg.group_id,
        "episode_id": res.episode_id,
        "already_stored": res.skipped,
        "entities": len(res.entities),
        "new_entities": len(res.new_entities),
        "facts_created": len(res.facts_created),
        "facts_duplicate": len(res.facts_duplicate),
        "facts_invalidated": len(res.facts_invalidated),
        "llm_calls": res.llm_calls,
    }


def cmd_search(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        hits = mem.search(
            args.query,
            group_id=cfg.group_id,
            as_of=args.as_of,
            center_entity=args.center,
            limit=max(1, min(args.limit, 50)),
            include_invalid=args.history,
            include_future=args.as_of is None,  # scheduled facts too, like the MCP tool
        )
    out = {"group_id": cfg.group_id, "as_of": args.as_of, "count": len(hits), "facts": [h.to_dict() for h in hits]}
    return _with_warning(out, warnings)


def cmd_entity(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        res = mem.get_entity(args.name, group_id=cfg.group_id, as_of=args.as_of)
    out = {"found": False, "message": f"no entity matching {args.name!r}"} if res is None else {"found": True, **res}
    return _with_warning(out, warnings)


def cmd_entities(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        ents = mem.search_entities(args.query, group_id=cfg.group_id, limit=max(1, min(args.limit, 50)))
    return _with_warning({"group_id": cfg.group_id, "count": len(ents), "entities": ents}, warnings)


def cmd_episodes(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg, need_embedder=bool(args.entity)) as (mem, warnings):
        eps = mem.get_episodes(
            group_id=cfg.group_id, entity=args.entity, query=args.query, limit=max(1, min(args.limit, 100))
        )
    return _with_warning({"group_id": cfg.group_id, "count": len(eps), "episodes": eps}, warnings)


def cmd_stats(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg) as (mem, _):
        counts = mem.stats(cfg.group_id)
        schema = getattr(mem, "schema_version", None)
    out: dict[str, Any] = {"group_id": cfg.group_id, "db": str(cfg.db_path), "counts": counts}
    if cfg.db_path.is_file():
        out["db_bytes"] = cfg.db_path.stat().st_size
    if schema is not None:
        out["schema_version"] = schema
    return out


def cmd_forget(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    has_range = any(v is not None for v in (args.since, args.until, args.channel))
    kinds = [
        k
        for k, present in (
            ("--episode", args.episode is not None),
            ("--source-ref", args.source_ref is not None),
            ("--fact", args.fact is not None),
            ("--entity", args.entity is not None),
            ("--since/--until/--channel/--source", has_range or (args.source is not None and args.source_ref is None)),
        )
        if present
    ]
    if len(kinds) != 1:
        raise CliError(
            "specify exactly one forget target: --episode ID | --source-ref REF [--source S] | --fact ID | "
            f"--entity NAME [--keep-episodes] | --since/--until/--channel [--source S] (got: {', '.join(kinds) or 'none'})",
            2,
        )
    if args.keep_episodes and args.entity is None:
        raise CliError("--keep-episodes only applies to --entity", 2)
    if args.source is not None and args.episode is not None:
        raise CliError("--source cannot be combined with --episode", 2)
    with open_memory(cfg) as (mem, _):
        g = cfg.group_id
        if args.episode is not None:
            res = mem.forget_episode(args.episode, group_id=g, dry_run=args.dry_run)
        elif args.source_ref is not None:
            res = mem.forget_episode(source=args.source, source_ref=args.source_ref, group_id=g, dry_run=args.dry_run)
        elif args.fact is not None:
            res = mem.forget_fact(args.fact, group_id=g, dry_run=args.dry_run)
        elif args.entity is not None:
            res = mem.forget_entity(args.entity, with_episodes=not args.keep_episodes, group_id=g, dry_run=args.dry_run)
        else:
            res = mem.forget_range(
                since=args.since,
                until=args.until,
                channel=args.channel,
                source=args.source,
                group_id=g,
                dry_run=args.dry_run,
            )
    return res.to_dict()


def cmd_purge(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    if not args.yes and not args.dry_run:
        raise CliError(
            f"refusing to purge ALL memories of group {cfg.group_id!r} without --yes (use --dry-run to preview)", 2
        )
    with open_memory(cfg) as (mem, _):
        res = mem.purge(group_id=cfg.group_id, dry_run=args.dry_run)
    return res.to_dict()


def mcp_argv(cfg: MemoryConfig, args: argparse.Namespace) -> list[str]:
    """Arguments for :func:`sqgraph.memory.mcp.main`, filled from the config."""
    argv = [
        "--db", str(cfg.db_path),
        "--group-id", cfg.group_id,
        "--env-file", str(cfg.env_path),
        "--tz", cfg.tz,
        "--llm-model", cfg.llm_model,
        "--embed-model", cfg.embed_model,
        "--transport", args.transport,
    ]  # fmt: skip
    if args.transport != "stdio":
        argv += ["--host", args.host, "--port", str(args.port)]
    if args.read_only:
        argv.append("--read-only")
    return argv


def cmd_mcp(args: argparse.Namespace, cfg: MemoryConfig) -> None:
    from . import mcp as mcp_mod  # attribute lookup at call time (monkeypatchable)

    ensure_private_dir(cfg.home)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mcp_mod.main(mcp_argv(cfg, args))
    except ImportError as exc:
        raise CliError(f"the MCP server needs extra packages ({exc}): {EXTRAS_HINT}") from exc
    return None


def cmd_setup(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    try:
        agents = setup_agents.parse_agents(args.agents)
        command, warnings = setup_agents.resolve_launch_command(args.command, which=lambda n: shutil.which(n))
    except ValueError as exc:
        raise CliError(str(exc), 2) from exc
    out = setup_agents.run_setup(
        agents,
        home=user_home(),
        command=command,
        install_skill=not args.no_skill,
        dry_run=args.dry_run,
        uninstall=args.uninstall,
        runner=setup_agents.default_runner,
        which=lambda n: shutil.which(n),
        warnings=warnings,
        env=None if args.uninstall else setup_agents.forwarded_env(),
    )
    if not args.uninstall and not cfg.has_api_key():
        out["note_init"] = "no API key configured yet: run `sqgraph-memory init` (agents launch the server without your shell environment)"
    return out


# --------------------------------------------------------------------------- doctor


def _check(name: str, status: str, detail: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def run_doctor(cfg: MemoryConfig, *, online: bool = False) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    add = checks.append

    v = sys.version_info
    add(_check("python", "ok" if v >= (3, 10) else "fail", f"{v.major}.{v.minor}.{v.micro} (need >= 3.10)"))
    sv = sqlite3.sqlite_version_info
    add(_check("sqlite3", "ok" if sv >= (3, 38) else "fail", f"{sqlite3.sqlite_version} (need >= 3.38)"))
    can_ext = hasattr(sqlite3.Connection, "enable_load_extension")
    add(_check("enable_load_extension", "ok" if can_ext else "fail",
               "available" if can_ext else "this Python's sqlite3 cannot load extensions (use python.org / uv-managed Python)"))  # fmt: skip
    try:
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            ver = conn.execute("select vec_version()").fetchone()[0]
        finally:
            conn.close()
        add(_check("sqlite_vec", "ok", f"loaded (vec {ver})"))
    except Exception as exc:
        add(_check("sqlite_vec", "fail", f"{type(exc).__name__}: {exc}"))
    for mod in ("openai", "mcp"):
        spec = importlib.util.find_spec(mod)
        add(_check(f"import {mod}", "ok" if spec else "fail", "importable" if spec else EXTRAS_HINT))

    if cfg.home.is_dir():
        detail = str(cfg.home)
        status = "ok"
        if os.name == "posix":
            mode = cfg.home.stat().st_mode & 0o777
            if mode & 0o077:
                status, detail = "warn", f"{cfg.home} has mode {mode:o}; run: chmod 700 {cfg.home}"
        add(_check("home", status, detail))
    else:
        add(_check("home", "warn", f"{cfg.home} does not exist: run `sqgraph-memory init`"))
    add(_check("api_key", "ok" if cfg.has_api_key() else "warn",
               "configured" if cfg.has_api_key() else f"{KEY_NAME} not found (env or {cfg.env_path}): run `sqgraph-memory init`"))  # fmt: skip

    if cfg.db_path.is_file():
        try:
            with Memory(cfg.db_path, read_only=True, default_group_id=cfg.group_id) as mem:
                schema = getattr(mem, "schema_version", None)
                st = mem.stats(cfg.group_id)
            extra = f", schema_version={schema}" if schema is not None else ""
            add(_check("database", "ok", f"{cfg.db_path} ({st['nodes']['Episode']} episodes, {st['facts_valid']} valid facts{extra})"))
        except Exception as exc:
            add(_check("database", "fail", f"{cfg.db_path}: {type(exc).__name__}: {exc}"))
    else:
        add(_check("database", "warn", f"{cfg.db_path} not created yet (created on the first `sqgraph-memory add`)"))

    try:
        command, _ = setup_agents.resolve_launch_command(None, which=lambda n: shutil.which(n))
        for st in setup_agents.agents_status(home=user_home(), command=command, which=lambda n: shutil.which(n), env=setup_agents.forwarded_env()):
            name = st["agent"]
            if not st["detected"]:
                add(_check(f"agent {name}", "skip", "not detected"))
                continue
            if st.get("error"):
                add(_check(f"agent {name}", "warn", f"cannot read config: {st['error']}"))
            elif not st["registered"]:
                add(_check(f"agent {name}", "warn", "MCP server not registered: run `sqgraph-memory setup`"))
            elif not st["up_to_date"]:
                add(_check(f"agent {name}", "warn", "registered with a different command: run `sqgraph-memory setup`"))
            else:
                add(_check(f"agent {name}", "ok", "MCP server registered"))
            sk = st["skill_installed"]
            if sk is not None:
                add(_check(f"skill {name}", "ok" if sk else "warn", "installed" if sk else "not installed: run `sqgraph-memory setup`"))
    except Exception as exc:
        add(_check("agents", "warn", f"{type(exc).__name__}: {exc}"))

    if online:
        try:
            _, embedder, _ = make_providers(cfg, need_llm=False, need_embedder=True)
            if embedder is None:
                add(_check("online embedding", "fail", f"{KEY_NAME} not set"))
            else:
                vec = embedder.embed(["sqgraph-memory doctor"])[0]
                add(_check("online embedding", "ok", f"{len(vec)} dimensions"))
        except Exception as exc:
            add(_check("online embedding", "fail", f"{type(exc).__name__}: {str(exc)[:200]}"))

    ok = not any(c["status"] == "fail" for c in checks)
    return {"ok": ok, "home": str(cfg.home), "checks": checks}


def cmd_doctor(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    return run_doctor(cfg, online=args.online)


# --------------------------------------------------------------------------- parser / main


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="sqgraph-memory", description="Personal long-term memory shared by local agents (sqgraph.memory).")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--compact", action="store_true", help="one-line JSON instead of indented")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", required=True, parser_class=_Parser)

    p = sub.add_parser("init", parents=[common], help="create the home directory, config and API key file")
    p.add_argument("--api-key-stdin", action="store_true", help="read the OpenAI API key from stdin (never pass it as an argument)")
    p.add_argument("--group-id", help="namespace of your memories (default: me)")
    p.add_argument("--tz", help="timezone for naive timestamps (default: Asia/Tokyo)")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("add", parents=[common], help="remember something (LLM extraction, takes a few seconds)")
    p.add_argument("text", metavar="TEXT", help="the text to remember; '-' reads stdin")
    p.add_argument("--source", default="cli", help="who is writing: claude-code / codex / cursor / cli ...")
    p.add_argument("--author", default="user")
    p.add_argument("--channel", default="personal")
    p.add_argument("--occurred-at", help="ISO 8601 time of the event (default: now)")
    p.add_argument("--source-ref", help="idempotency key: the same (source, source-ref) is stored once")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("search", parents=[common], help="search facts (hybrid semantic + keyword + graph)")
    p.add_argument("query", metavar="QUERY")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--as-of", help="answer as of this date/time")
    p.add_argument("--history", action="store_true", help="include superseded / ended facts")
    p.add_argument("--center", metavar="ENTITY", help="rank facts near this entity first")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("entity", parents=[common], help="one entity's profile and full fact timeline")
    p.add_argument("name", metavar="NAME", help="entity name, alias or id")
    p.add_argument("--as-of")
    p.set_defaults(fn=cmd_entity)

    p = sub.add_parser("entities", parents=[common], help="find entities by name / alias / description")
    p.add_argument("query", metavar="QUERY")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_entities)

    p = sub.add_parser("episodes", parents=[common], help="list raw stored notes (newest first)")
    p.add_argument("--entity")
    p.add_argument("--query")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_episodes)

    p = sub.add_parser("stats", parents=[common], help="counts and database info")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("forget", parents=[common], help="delete memories (exactly one target kind)")
    p.add_argument("--episode", metavar="ID")
    p.add_argument("--source-ref", metavar="REF")
    p.add_argument("--source", help="with --source-ref, or alone / with a range to forget everything from that source")
    p.add_argument("--fact", metavar="ID")
    p.add_argument("--entity", metavar="NAME")
    p.add_argument("--keep-episodes", action="store_true", help="with --entity: keep the source notes")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--channel")
    p.add_argument("--dry-run", action="store_true", help="show what would be removed without deleting")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("purge", parents=[common], help="delete ALL memories of the configured group")
    p.add_argument("--yes", action="store_true", help="required to actually purge")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_purge)

    p = sub.add_parser("mcp", help="run the MCP server (what agents launch)")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--read-only", action="store_true")
    p.set_defaults(fn=cmd_mcp, raw=True)

    p = sub.add_parser("setup", parents=[common], help="register the MCP server and install the Skill for local agents")
    p.add_argument("--agents", help="comma-separated: claude-code,codex,cursor,claude-desktop or all (default: detected)")
    p.add_argument("--command", help='full launch command (e.g. "uvx --from sqgraph[mcp] sqgraph-memory mcp")')
    p.add_argument("--no-skill", action="store_true", help="do not install SKILL.md")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--uninstall", action="store_true", help="remove the registration and the Skill")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("doctor", parents=[common], help="check the installation (exit code 1 on failure)")
    p.add_argument("--online", action="store_true", help="also make one embedding API call")
    p.set_defaults(fn=cmd_doctor)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    _utf8()
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
        cfg = MemoryConfig.load()
        result = args.fn(args, cfg)
        if getattr(args, "raw", False):
            return 0
        emit(result, compact=getattr(args, "compact", False))
        if args.fn is cmd_doctor and not result["ok"]:
            return 1
        return 0
    except CliError as exc:
        emit_error(str(exc))
        return exc.code
    except KeyboardInterrupt:
        emit_error("interrupted")
        return 130
    except ImportError as exc:
        emit_error(f"missing dependency ({exc}): {EXTRAS_HINT}")
        return 1
    except Exception as exc:
        emit_error(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
