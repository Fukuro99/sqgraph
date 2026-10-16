"""``sqgraph-memory``: personal long-term memory CLI (also ``python -m sqgraph.memory``).

All output is JSON on stdout (UTF-8); errors are ``{"error": ...}`` on stderr with a non-zero exit code.

    sqgraph-memory init [--extractor agent|openai] [--api-key-stdin] [--group-id G] [--tz TZ] [--embed-base-url URL] ...
    sqgraph-memory config [--extractor agent|openai] [--embed-base-url URL|openai] [--embed-model M] [--llm-model M]
                          [--search-min-similarity X|default] ...
    sqgraph-memory reembed [--yes]
    sqgraph-memory add TEXT|- [--source claude-code] [--channel personal] ...      (agent mode: prints what to extract)
    sqgraph-memory commit [--token T] [--extraction FILE|-]                          (agent mode: stores the extraction)
    sqgraph-memory search QUERY [--limit 10] [--as-of DATE] [--history] [--center ENTITY] [--min-similarity X]
    sqgraph-memory entity NAME | entities QUERY | episodes | stats
    sqgraph-memory forget (--episode ID | --source-ref REF | --fact ID | --entity NAME | --since/--until/--channel) [--dry-run]
                          (--episode-id / --fact-id are aliases of --episode / --fact)
    sqgraph-memory purge --yes [--dry-run]
    sqgraph-memory merge KEEP MERGE [--dry-run] | invalidate FACT_ID [--at T] [--reason R]
    sqgraph-memory relations | rename-relation OLD NEW [--swap] [--fact ID] [--dry-run] | review | upcoming [--days 7]
    sqgraph-memory export FILE|- [--with-embeddings] | import FILE|- [--into-group G] [--dry-run] | backup [DEST] [--keep 10]
    sqgraph-memory hook session-start|user-prompt [--agent A]   (agent hooks, registered by `setup`)
    sqgraph-memory import-claude-memory [--limit 5] [--dry-run]
    sqgraph-memory mcp [--transport stdio|streamable-http] [--read-only] [--extractor agent|openai] [--agent A]
    sqgraph-memory setup [--agents ...] [--group G] [--uninstall] | doctor [--online]
    sqgraph-memory groups | config --agent-group AGENT=GROUP     (separate memories per agent; every memory command takes --agent A)

The home directory (``$SQGRAPH_MEMORY_HOME`` or ``~/.sqgraph-memory``) holds ``memory.db`` -- one SQLite file shared
by every agent on the machine (optionally split into groups per agent, see ``agent_groups``) -- plus ``.env`` and ``config.json``. See :mod:`sqgraph.memory.config`.
Who extracts the entities / facts of a note is the ``extractor`` setting: ``agent`` (default) means the calling agent
itself: ``add`` prints the instructions and context, the agent answers with an extraction and ``commit`` stores it
(no OpenAI key needed); ``openai`` makes ``add`` call an OpenAI LLM in one step. The embeddings can come from any
OpenAI-compatible server (``--embed-base-url``, e.g. LM Studio / Ollama on another PC of the LAN).
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NoReturn, Sequence

from . import setup_agents
from .config import (
    DEFAULT_GROUP_ALIAS,
    EXTRACTORS,
    KEY_NAME,
    MemoryConfig,
    ensure_private_dir,
    validate_agent_name,
    validate_group_id,
)
from .core import (
    FACT_INDEX,
    META_EMBED_DIMS,
    META_EMBED_ENDPOINT,
    META_EMBED_MODEL,
    NO_HITS_HINT,
    Memory,
    default_min_similarity,
)
from .providers import (
    DEFAULT_EMBED_DIMS,
    Embedder,
    LLMProvider,
    OpenAILLM,
    UsageTracker,
    build_embedder,
    load_env,
    normalize_base_url,
    probe_dimensions,
)

EXTRAS_HINT = 'uv tool install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"   (or: pip install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph")'


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


def _stored_identity(path: Path) -> dict[str, Any] | None:
    """Embedding identity recorded in the database (``model`` is ``None`` for databases from before it was recorded)."""
    if not path.is_file():
        return None
    try:
        from ..db import GraphDB

        with GraphDB(str(path), read_only=True) as db:
            model, dims, endpoint = (db.get_meta(k) for k in (META_EMBED_MODEL, META_EMBED_DIMS, META_EMBED_ENDPOINT))
            index_dims = next((ix.dimensions for ix in db.vector_indexes() if ix.name == FACT_INDEX), None)
    except Exception:
        return None
    if model is None and index_dims is None:
        return None
    return {"model": model, "dimensions": int(dims) if dims and dims.isdigit() else index_dims, "endpoint": endpoint}


def _describe_embedding(model: str | None, dims: int | None, endpoint: str | None) -> str:
    return f"model {model or 'unknown'!r} ({dims if dims else 'unknown'} dimensions, endpoint {endpoint or 'unknown'})"


def _identity_mismatch(cfg: MemoryConfig, stored: dict[str, Any] | None) -> str | None:
    """Why the database's vectors do not fit the configured embedder (``None`` = they fit / nothing stored)."""
    if not stored:
        return None
    dims = cfg.effective_embed_dimensions()
    wrong_model = bool(stored["model"]) and stored["model"] != cfg.embed_model
    wrong_dims = bool(stored["dimensions"]) and bool(dims) and stored["dimensions"] != dims
    if not (wrong_model or wrong_dims):
        return None
    return (
        f"the database was built with {_describe_embedding(stored['model'], stored['dimensions'], stored['endpoint'])} but "
        f"the configuration says {_describe_embedding(cfg.embed_model, dims, cfg.embed_base_url or 'openai')}: vectors of "
        "different models are not comparable, so run `sqgraph-memory reembed` (or switch the configuration back)"
    )


def _embed_key(cfg: MemoryConfig) -> str | None:
    """The optional API key of the custom embedder endpoint (from the environment / ``.env``; never printed)."""
    load_env(cfg.env_path)
    return os.environ.get(cfg.embed_api_key_env) or None


def probe_endpoint(cfg: MemoryConfig) -> int:
    """Embed one text at the configured custom endpoint and return the model's dimensions. Tests monkeypatch this."""
    if importlib.util.find_spec("openai") is None:
        raise CliError(f'the "openai" package is missing: {EXTRAS_HINT}')
    try:
        return probe_dimensions(model=cfg.embed_model, base_url=cfg.embed_base_url, api_key=_embed_key(cfg))
    except Exception as exc:
        raise CliError(
            f"cannot get an embedding from {cfg.embed_base_url} with model {cfg.embed_model!r} "
            f"({type(exc).__name__}: {str(exc)[:200]}). Check the URL and model name, that the server is running and "
            "listens on the LAN interface (e.g. OLLAMA_HOST=0.0.0.0); `--no-probe` saves the settings without checking"
        ) from exc


def _save_probed_dims(cfg: MemoryConfig, dims: int) -> None:
    """Remember the dimensions found by probing a custom endpoint (best effort)."""
    cfg.embed_dimensions = dims
    try:
        stored = MemoryConfig.load(cfg.home, apply_env=False)
        if (stored.embed_base_url, stored.embed_model) == (cfg.embed_base_url, cfg.embed_model) and not stored.embed_dimensions:
            stored.embed_dimensions = dims
            stored.save()
    except (OSError, ValueError):
        pass


def make_providers(
    cfg: MemoryConfig, *, need_llm: bool, need_embedder: bool
) -> tuple[LLMProvider | None, Embedder | None, list[str]]:
    """Build the providers from ``cfg`` (+ ``<home>/.env``). Tests monkeypatch this with fakes.

    The LLM is always OpenAI (``need_llm`` without ``OPENAI_API_KEY`` is an error). The embedder is OpenAI too, or the
    OpenAI-compatible server at ``cfg.embed_base_url``, which needs no OpenAI key; without any usable embedder it is
    ``None`` (fulltext + entity search still work) and a warning is returned.
    """
    warnings: list[str] = []
    load_env(cfg.env_path)  # existing environment variables win
    have_key = bool(os.environ.get(KEY_NAME))
    if need_llm and not have_key:
        raise CliError(f"{KEY_NAME} is not set: run `sqgraph-memory init` (or set the environment variable)")
    want_embedder = (need_llm or need_embedder) and (cfg.custom_embedder or have_key)
    if need_embedder and not want_embedder:
        warnings.append(f"{KEY_NAME} is not set: semantic (vector) search is disabled; run `sqgraph-memory init`")
    if not (need_llm or want_embedder):
        return None, None, warnings
    if importlib.util.find_spec("openai") is None:
        raise CliError(f'the "openai" package is missing: {EXTRAS_HINT}')
    tracker = UsageTracker()
    llm = OpenAILLM(cfg.llm_model, tracker=tracker) if need_llm else None
    embedder = None
    if want_embedder:
        try:
            embedder, probed = build_embedder(
                cfg.embed_model,
                base_url=cfg.embed_base_url,
                dimensions=cfg.embed_dimensions,
                existing_dims=_existing_dims(cfg.db_path),
                api_key=_embed_key(cfg),
                tracker=tracker,
            )
        except Exception as exc:
            raise CliError(
                f"cannot reach the embedding endpoint {cfg.embed_base_url} to find its dimensions "
                f"({type(exc).__name__}: {str(exc)[:200]}): start the server or run `sqgraph-memory config --embed-dimensions N`"
            ) from exc
        if probed:
            _save_probed_dims(cfg, embedder.dimensions)
    return llm, embedder, warnings


@contextmanager
def open_memory(
    cfg: MemoryConfig, *, need_llm: bool = False, need_embedder: bool = False, allow_embedder_change: bool = False
) -> Iterator[tuple[Memory, list[str]]]:
    ensure_private_dir(cfg.home)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    llm, embedder, warnings = make_providers(cfg, need_llm=need_llm, need_embedder=need_embedder)
    mem = Memory(
        cfg.db_path,
        llm,
        embedder,
        default_group_id=cfg.group_id,
        tz=cfg.tz,
        allow_embedder_change=allow_embedder_change,
        min_similarity=cfg.search_min_similarity,
    )
    try:
        yield mem, warnings
    finally:
        mem.close()


def _with_warning(out: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if warnings:
        out["warning"] = "; ".join(warnings)
    return out


# --------------------------------------------------------------------------- commands


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPENAI_ALIASES = ("", "openai", "default", "none")


def _apply_settings(cfg: MemoryConfig, args: argparse.Namespace) -> list[str]:
    """Apply the settings options of ``init`` / ``config`` to ``cfg``; returns the names of the fields that changed."""
    changed: list[str] = []

    def put(field: str, value: Any) -> None:
        if getattr(cfg, field) != value:
            setattr(cfg, field, value)
            changed.append(field)

    if args.group_id:
        put("group_id", args.group_id)
    if getattr(args, "agent_group", None):
        groups = dict(cfg.agent_groups)
        for spec in args.agent_group:
            agent, sep, group = spec.partition("=")
            if not sep:
                raise CliError(f"--agent-group expects AGENT=GROUP (or AGENT=default to remove it), got {spec!r}", 2)
            try:
                agent = validate_agent_name(agent.strip())
                group = group.strip()
                if group in ("", DEFAULT_GROUP_ALIAS):
                    groups.pop(agent, None)
                else:
                    groups[agent] = validate_group_id(group)
            except ValueError as exc:
                raise CliError(f"--agent-group {spec!r}: {exc}", 2) from exc
        put("agent_groups", groups)
    if args.tz:
        try:
            from .util import resolve_tz

            resolve_tz(args.tz)
        except Exception as exc:
            raise CliError(f"invalid timezone {args.tz!r}: {exc}", 2) from exc
        put("tz", args.tz)
    if getattr(args, "extractor", None):
        put("extractor", args.extractor)
    if getattr(args, "llm_model", None):
        put("llm_model", args.llm_model.strip())
    before = (cfg.embed_base_url, cfg.embed_model)
    if args.embed_base_url is not None:
        url = args.embed_base_url.strip()
        if url.lower() in _OPENAI_ALIASES:
            put("embed_base_url", None)
        else:
            try:
                put("embed_base_url", normalize_base_url(url))
            except ValueError as exc:
                raise CliError(str(exc), 2) from exc
    if args.embed_model:
        put("embed_model", args.embed_model.strip())
    if args.embed_api_key_env:
        if not _ENV_NAME_RE.match(args.embed_api_key_env):
            raise CliError(f"invalid environment variable name {args.embed_api_key_env!r}", 2)
        put("embed_api_key_env", args.embed_api_key_env)
    if (cfg.embed_base_url, cfg.embed_model) != before:
        put("embed_dimensions", None)  # the stored dimensions belonged to the old model: probe / ask again
    if args.embed_dimensions is not None:
        if args.embed_dimensions < 1:
            raise CliError("--embed-dimensions must be a positive integer", 2)
        put("embed_dimensions", args.embed_dimensions)
    if getattr(args, "search_min_similarity", None) is not None:
        put("search_min_similarity", _parse_similarity(args.search_min_similarity, "--search-min-similarity", allow_default=True))
    return changed


def _parse_similarity(raw: Any, flag: str, *, allow_default: bool = False) -> float | None:
    """A cosine similarity threshold in [-1, 1]; ``default`` / ``auto`` / ``none`` (config only) mean "per embedding model"."""
    if allow_default and str(raw).strip().lower() in ("default", "auto", "none", ""):
        return None
    suffix = " or `default`" if allow_default else ""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise CliError(f"{flag} must be a number between -1 and 1 (cosine similarity){suffix}, got {raw!r}", 2) from None
    if not -1.0 <= v <= 1.0:
        raise CliError(f"{flag} must be between -1 and 1 (cosine similarity), got {raw!r}", 2)
    return v


def _probe_embedder(cfg: MemoryConfig, args: argparse.Namespace, changed: list[str]) -> dict[str, Any] | None:
    """Probe the custom endpoint when embedder options were given (fills ``embed_dimensions``); ``None`` = not probed."""
    given = any(getattr(args, k) is not None for k in ("embed_base_url", "embed_model", "embed_dimensions"))
    if not (cfg.custom_embedder and given) or args.no_probe:
        return None
    explicit = args.embed_dimensions
    before_dims = cfg.embed_dimensions
    dims = probe_endpoint(cfg)
    if explicit is not None and explicit != dims:
        raise CliError(
            f"{cfg.embed_base_url} produces {dims}-dimensional vectors for model {cfg.embed_model!r}, not {explicit}: "
            "drop --embed-dimensions to use the detected value (or --no-probe to force yours)",
            2,
        )
    if cfg.embed_dimensions != dims:
        cfg.embed_dimensions = dims
    if "embed_dimensions" not in changed and dims != before_dims:
        changed.append("embed_dimensions")
    return {"dimensions": dims}


def _embedder_report(cfg: MemoryConfig) -> dict[str, Any]:
    """The embedder settings for output (never the key value)."""
    rep: dict[str, Any] = {
        "endpoint": cfg.embed_base_url or "openai",
        "model": cfg.embed_model,
        "dimensions": cfg.effective_embed_dimensions(),
    }
    if cfg.custom_embedder:
        rep["api_key_env"] = cfg.embed_api_key_env
        rep["api_key"] = "set" if cfg.has_embed_api_key() else "not set (optional)"
    return rep


def _embedder_warnings(cfg: MemoryConfig, probed: bool = False) -> list[str]:
    out: list[str] = []
    mismatch = _identity_mismatch(cfg, _stored_identity(cfg.db_path))
    if mismatch:
        out.append(mismatch)
    if cfg.custom_embedder and cfg.embed_dimensions is None and not probed:
        out.append("embedding dimensions are not known yet: they are probed on first use (or set --embed-dimensions)")
    return out


def cmd_init(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    cfg = MemoryConfig.load(cfg.home, apply_env=False)  # persist stored values, not env overrides
    env_key = os.environ.get(KEY_NAME)  # before .env is loaded: tells "from the environment" from "already stored"
    env_embed_key = os.environ.get(cfg.embed_api_key_env)
    changed = _apply_settings(cfg, args)
    probe = _probe_embedder(cfg, args, changed)  # fails before anything is saved
    cfg.save()

    # An OpenAI key is needed by the "openai" extractor and by the OpenAI embedder; with the agent extractor and a
    # custom (LAN) embedder nothing needs it, and init neither asks for nor copies one (--api-key-stdin still works).
    needs_key = cfg.extractor == "openai" or not cfg.custom_embedder
    key_state = "missing" if needs_key else "not-needed"
    key: str | None = None
    if args.api_key_stdin:
        key = (sys.stdin.readline() or "").strip()
        if not key:
            raise CliError("--api-key-stdin: no key received on stdin", 2)
        key_state = "stored"
    elif not needs_key:
        pass
    elif env_key:
        key, key_state = env_key.strip(), "stored-from-env"
    elif MemoryConfig.load(cfg.home).has_api_key():
        key_state = "existing"
    elif cfg.extractor == "openai" and sys.stdin is not None and sys.stdin.isatty():
        key = getpass.getpass("OpenAI API key (input hidden, saved to %s): " % cfg.env_path).strip()
        key_state = "stored" if key else "missing"
    if key:
        cfg.write_api_key(key)
    if cfg.custom_embedder and env_embed_key and env_embed_key.strip():
        cfg.write_api_key(env_embed_key, cfg.embed_api_key_env)  # agents start without the shell environment

    db = MemoryConfig.load(cfg.home)  # with env overrides: where the DB really lives
    out: dict[str, Any] = {
        "home": str(cfg.home),
        "config": str(cfg.config_path),
        "db": str(db.db_path),
        "db_exists": db.db_path.is_file(),  # created on the first `add`
        "group_id": db.group_id,
        "tz": cfg.tz,
        "extractor": cfg.extractor,
        "api_key": key_state,
        "embedder": _embedder_report(db),
    }
    if probe:
        out["probed"] = True
    warnings: list[str] = []
    if key_state == "missing" and cfg.extractor == "openai":
        warnings.append(
            f"no {KEY_NAME} configured: set the env var and re-run `sqgraph-memory init`, or pipe it: "
            "`... | sqgraph-memory init --api-key-stdin`"
        )
    elif key_state == "missing":
        warnings.append(
            f"no embedding endpoint or {KEY_NAME} configured: semantic (vector) search is off (the agent extractor needs no key); "
            "use --embed-base-url for a LAN embedding server, or provide the key"
        )
    warnings += _embedder_warnings(db, probed=probe is not None)
    if warnings:
        out["warning"] = "; ".join(warnings)
    out["next"] = "sqgraph-memory setup   # register with Claude Code / Codex / Cursor / Claude Desktop"
    return out


def _effective_similarity(cfg: MemoryConfig) -> float:
    """The relevance threshold search really uses: the configured one, else the default of the model of the database
    (the embedder the vectors were built with), else of the configured embedding model."""
    if cfg.search_min_similarity is not None:
        return cfg.search_min_similarity
    stored = _stored_identity(cfg.db_path)
    return default_min_similarity((stored or {}).get("model") or cfg.embed_model)


def cmd_config(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    stored = MemoryConfig.load(cfg.home, apply_env=False)
    changed = _apply_settings(stored, args)
    probe = _probe_embedder(stored, args, changed)  # fails before anything is saved
    if changed:
        stored.save()
    eff = MemoryConfig.load(cfg.home)  # with env overrides
    out: dict[str, Any] = {
        "home": str(eff.home),
        "config": str(eff.config_path),
        "db": str(eff.db_path),
        "group_id": eff.group_id,
        "agent_groups": eff.agent_groups,
        "tz": eff.tz,
        "extractor": eff.extractor,
        "llm_model": eff.llm_model,
        "embedder": _embedder_report(eff),
        "search_min_similarity": eff.search_min_similarity,
        "search_min_similarity_effective": _effective_similarity(eff),
        "openai_api_key": "configured" if eff.has_api_key() else "missing",
        "db_embedding": _stored_identity(eff.db_path),
    }
    if changed:
        out["changed"] = changed
    if probe:
        out["probed"] = True
    warnings = _embedder_warnings(eff, probed=probe is not None)
    if warnings:
        out["warning"] = "; ".join(warnings)
        if any("reembed" in w for w in warnings):
            out["reembed_required"] = True
    return out


def cmd_reembed(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    if not cfg.db_path.is_file():
        return {"executed": False, "db": str(cfg.db_path), "message": "no database yet: nothing to re-embed"}
    if args.batch_size < 1:
        raise CliError("--batch-size must be a positive integer", 2)
    stored = _stored_identity(cfg.db_path)
    with Memory(cfg.db_path, read_only=True) as m:
        st = m.stats(None)
    entities, facts = st["nodes"]["Entity"], st["facts_valid"] + st["facts_invalidated"]
    target = _embedder_report(cfg)
    if not args.yes:
        return {
            "executed": False,
            "db": str(cfg.db_path),
            "entities": entities,
            "facts": facts,
            "from": stored,
            "to": {k: target[k] for k in ("model", "dimensions", "endpoint")},
            "next": "re-run with --yes: every entity and fact is embedded again with the configured embedder (the "
            "database is write-locked meanwhile; with an OpenAI embedder this costs a few cents per 10k facts)",
        }
    if not cfg.embed_dimensions:  # the dimensions of the *configured* model, not those of the old vectors
        cfg.embed_dimensions = probe_endpoint(cfg) if cfg.custom_embedder else DEFAULT_EMBED_DIMS
    _, embedder, _ = make_providers(cfg, need_llm=False, need_embedder=True)
    if embedder is None:
        raise CliError(f"{KEY_NAME} is not set: run `sqgraph-memory init` (or configure --embed-base-url)")

    def progress(kind: str, done: int, total: int) -> None:
        if args.progress:
            sys.stderr.write(json.dumps({"progress": kind, "done": done, "total": total}) + "\n")
            sys.stderr.flush()

    mem = Memory(
        cfg.db_path, None, embedder, default_group_id=cfg.group_id, tz=cfg.tz, allow_embedder_change=True
    )
    try:
        res = mem.reembed(batch_size=args.batch_size, progress=progress)
    finally:
        mem.close()
    if cfg.custom_embedder and not MemoryConfig.load(cfg.home, apply_env=False).embed_dimensions:
        _save_probed_dims(cfg, res["dimensions"])
    return {"executed": True, "db": str(cfg.db_path), **res}


def _episode_summary(cfg: MemoryConfig, res: Any) -> dict[str, Any]:
    out = {
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
    if res.warnings:
        out["warnings"] = list(res.warnings)
    return out


COMMIT_NEXT = (
    "Write the extraction JSON (see `schema`, `instructions`), then run: sqgraph-memory commit --token <token> "
    "< extraction.json   (or pipe a single object {\"token\": ..., \"extraction\": ...} on stdin)"
)


def cmd_add(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    text = sys.stdin.read() if args.text == "-" else args.text
    if not text or not text.strip():
        raise CliError("text must not be empty", 2)
    if cfg.extractor == "agent":  # step 1 of 2: no LLM here, the calling agent extracts (see `commit`)
        with open_memory(cfg, need_embedder=True) as (mem, warnings):
            prep = mem.prepare_episode(
                text,
                group_id=cfg.group_id,
                source=args.source,
                channel=args.channel,
                author=args.author,
                occurred_at=args.occurred_at,
                source_ref=args.source_ref,
            )
        if prep.get("already_stored"):
            prep["next"] = "already stored: nothing to commit"
        else:
            prep["next"] = COMMIT_NEXT
        return _with_warning({"group_id": cfg.group_id, **prep}, warnings)
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
    return _episode_summary(cfg, res)


def _read_commit_input(args: argparse.Namespace) -> tuple[str, Any]:
    """``(token, extraction)`` from ``--token`` / ``--extraction FILE|-`` (default stdin); the input may also be one
    object ``{"token": ..., "extraction": ...}``."""
    src = args.extraction if args.extraction not in (None, "-") else None
    try:
        raw = Path(src).read_text(encoding="utf-8-sig") if src else sys.stdin.read()
    except OSError as exc:
        raise CliError(f"cannot read the extraction file: {exc}", 2) from exc
    if not raw.strip():
        raise CliError("no extraction received (pass --extraction FILE, or pipe the JSON on stdin)", 2)
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise CliError(f"the extraction is not valid JSON: {exc}", 2) from exc
    token = args.token
    if isinstance(data, dict) and "extraction" in data and isinstance(data.get("token"), str):
        token = token or data["token"]
        data = data["extraction"]
    if not token:
        raise CliError('--token is required (or send {"token": ..., "extraction": ...} on stdin)', 2)
    return token.strip(), data


def cmd_commit(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    token, extraction = _read_commit_input(args)
    try:
        tok = Memory.read_token(token)
    except ValueError as exc:
        raise CliError(str(exc), 2) from exc
    if tok["group_id"] != cfg.group_id:
        raise CliError(
            f"the token belongs to group {tok['group_id']!r}, but this command works in group {cfg.group_id!r}: "
            "pass the same --agent as to the `add` that printed the token",
            2,
        )
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        try:
            res = mem.commit_episode(token, extraction)
        except ValueError as exc:
            raise CliError(str(exc), 2) from exc
    return _with_warning(_episode_summary(cfg, res), warnings)


def cmd_search(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    min_sim = _parse_similarity(args.min_similarity, "--min-similarity") if args.min_similarity is not None else None
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        hits = mem.search(
            args.query,
            group_id=cfg.group_id,
            as_of=args.as_of,
            center_entity=args.center,
            limit=max(1, min(args.limit, 50)),
            include_invalid=args.history,
            include_future=args.as_of is None,  # scheduled facts too, like the MCP tool
            min_similarity=min_sim,
        )
        threshold = mem.effective_min_similarity(min_sim)
    out: dict[str, Any] = {
        "group_id": cfg.group_id,
        "as_of": args.as_of,
        "min_similarity": round(threshold, 4),
        "count": len(hits),
        "facts": [h.to_dict() for h in hits],
    }
    if not hits:
        out["hint"] = NO_HITS_HINT
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


DEFAULT_MARK = "(default)"  # in `groups`: the agents without an assignment


def cmd_groups(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    """The groups of the database (with counts) and the agents working in each. Read-only."""
    counts: dict[str, dict[str, int]] = {}
    if cfg.db_path.is_file():
        with Memory(cfg.db_path, read_only=True, default_group_id=cfg.group_id) as mem:
            counts = mem.group_stats()
    # who works where: SQGRAPH_MEMORY_GROUP overrides every assignment
    agents: dict[str, list[str]] = {cfg.group_id: [DEFAULT_MARK]}
    for agent in sorted(cfg.agent_groups):
        agents.setdefault(cfg.group_for(agent), []).append(agent)
    names = sorted(set(counts) | set(agents), key=lambda g: (g != cfg.group_id, g))
    zero = {"episodes": 0, "entities": 0, "facts_valid": 0}
    out: dict[str, Any] = {
        "db": str(cfg.db_path),
        "default_group": cfg.group_id,
        "groups": [
            {"group_id": g, **counts.get(g, zero), "agents": agents.get(g, []), "default": g == cfg.group_id} for g in names
        ],
    }
    if cfg.group_from_env:
        out["note"] = "SQGRAPH_MEMORY_GROUP is set: it overrides agent_groups, every agent works in the default group"
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
            "specify exactly one forget target: --episode ID (alias --episode-id) | --source-ref REF [--source S] | "
            "--fact ID (alias --fact-id) | --entity NAME|ID (alias --entity-id) [--keep-episodes] | "
            f"--since/--until/--channel [--source S] (got: {', '.join(kinds) or 'none'}); "
            "example: sqgraph-memory forget --episode-id ep_... --dry-run",
            2,
        )
    if args.keep_episodes and args.entity is None:
        raise CliError("--keep-episodes only applies to --entity", 2)
    if args.source is not None and args.episode is not None:
        raise CliError("--source cannot be combined with --episode", 2)
    with open_memory(cfg, need_embedder=True) as (mem, _):  # touched entities get fresh embeddings
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


def cmd_merge(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg, need_embedder=True) as (mem, warnings):  # the kept entity gets a fresh embedding
        try:
            res = mem.merge_entities(args.keep, args.merge, group_id=cfg.group_id, dry_run=args.dry_run)
        except ValueError as exc:
            raise CliError(str(exc), 2) from exc
    return _with_warning({"group_id": cfg.group_id, **res}, warnings)


def cmd_invalidate(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg) as (mem, _):
        try:
            hit = mem.invalidate_fact(args.fact_id, at=args.at, reason=args.reason, group_id=cfg.group_id)
        except ValueError as exc:
            raise CliError(str(exc), 2) from exc
    return {"group_id": cfg.group_id, "fact": hit.to_dict()}


def cmd_relations(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg) as (mem, _):
        rels = mem.relation_stats(cfg.group_id)
    return {"group_id": cfg.group_id, "count": len(rels), "relations": rels}


def cmd_rename_relation(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg) as (mem, _):
        try:
            res = mem.rename_relation(
                args.old, args.new, swap=args.swap, fact_ids=args.fact or None, group_id=cfg.group_id, dry_run=args.dry_run
            )
        except ValueError as exc:
            raise CliError(str(exc), 2) from exc
    return {"group_id": cfg.group_id, **res}


def cmd_review(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        res = mem.review(cfg.group_id, limit=max(1, min(args.limit, 100)))
    return _with_warning(res, warnings)


def cmd_upcoming(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    with open_memory(cfg) as (mem, _):
        items = mem.upcoming(args.days, group_id=cfg.group_id, past_days=args.past_days)
    return {"group_id": cfg.group_id, "days": args.days, "count": len(items), "events": items}


def cmd_export(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any] | None:
    from . import portable

    with open_memory(cfg) as (mem, _):
        if args.file == "-":
            portable.export_memory(mem, sys.stdout, group_id=cfg.group_id, with_embeddings=args.with_embeddings)
            return None
        dest = Path(args.file)
        if dest.exists() and not args.force:
            raise CliError(f"{dest} already exists (use --force to overwrite)", 2)
        tmp = dest.with_name(dest.name + ".part")
        with tmp.open("w", encoding="utf-8", newline="\n") as out:
            counts = portable.export_memory(mem, out, group_id=cfg.group_id, with_embeddings=args.with_embeddings)
        tmp.replace(dest)
    return {"path": str(dest), "bytes": dest.stat().st_size, **counts}


def cmd_import(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    from . import portable

    def progress(kind: str, done: int, total: int) -> None:
        sys.stderr.write(json.dumps({"progress": kind, "done": done, "total": total}) + "\n")

    try:
        src = sys.stdin if args.file == "-" else Path(args.file).open(encoding="utf-8-sig")
    except OSError as exc:
        raise CliError(f"cannot read {args.file}: {exc}", 2) from exc
    try:
        with open_memory(cfg, need_embedder=True) as (mem, warnings):
            res = portable.import_memory(
                mem, src, group_id=args.into_group or cfg.group_id, dry_run=args.dry_run, progress=progress if args.progress else None
            )
    except ValueError as exc:
        raise CliError(str(exc), 2) from exc
    finally:
        if src is not sys.stdin:
            src.close()
    return _with_warning(res, warnings)


def cmd_backup(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    import time

    from . import portable

    directory = cfg.home / "backups"
    dest = Path(args.dest) if args.dest else directory / time.strftime("memory-%Y%m%d-%H%M%S.db")
    with open_memory(cfg) as (mem, _):
        try:
            res = portable.backup_database(mem, dest)
        except FileExistsError as exc:
            raise CliError(str(exc), 2) from exc
    if not args.dest:
        res["removed"] = portable.rotate_backups(directory, args.keep)
    return res


def cmd_hook(args: argparse.Namespace, cfg: MemoryConfig) -> None:
    """A Claude Code hook: event JSON on stdin, ``hookSpecificOutput`` JSON (or nothing) on stdout, always exit 0."""
    from . import hooks

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    @contextmanager
    def memory() -> Iterator[Memory]:
        with open_memory(cfg, need_embedder=True) as (mem, _):
            yield mem

    out = hooks.run(args.event, payload, cfg.home, memory, agent=args.agent)
    if out is not None:
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return None


def _claude_memory_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/memory/*.md") if p.name != "MEMORY.md" and p.is_file())


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """``(fields, body)`` of a Markdown file with ``---`` front matter (flat ``key: value``; nested keys are dotted)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    fields: dict[str, str] = {}
    parent = None
    for ln in m.group(1).splitlines():
        km = re.match(r"^(\s*)([A-Za-z0-9_-]+):\s*(.*)$", ln)
        if not km:
            continue
        indent, key, value = km.groups()
        if not indent:
            parent = key if not value else None
            if value:
                fields[key] = value.strip().strip("\"'")
        elif parent:
            fields[f"{parent}.{key}"] = value.strip().strip("\"'")
    return fields, m.group(2)


def cmd_import_claude_memory(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    """Claude Code's file-based auto-memory (``~/.claude/projects/*/memory/*.md``) as episodes (one per file)."""
    import hashlib
    from datetime import datetime, timezone

    root = Path(args.root).expanduser() if args.root else user_home() / ".claude" / "projects"
    if not root.is_dir():
        raise CliError(f"no Claude Code projects directory at {root}", 2)
    items = []
    for p in _claude_memory_files(root):
        raw = p.read_text(encoding="utf-8-sig")
        fields, body = _parse_front_matter(raw)
        project = p.parent.parent.name
        title = fields.get("name") or p.stem
        head = f"Claude Code memory ({fields.get('metadata.type') or fields.get('type') or 'note'}, project {project}): {title}"
        if fields.get("description"):
            head += f" -- {fields['description']}"
        content = f"{head}\n\n{body.strip()}"[:6000]
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        items.append({"file": str(p), "source_ref": f"{project}/{p.name}#{digest}", "occurred_at": mtime, "content": content})
    source = "claude-memory"
    if cfg.extractor == "openai" and not args.dry_run:
        stored = []
        with open_memory(cfg, need_llm=True) as (mem, _):
            for it in items[: args.limit]:
                res = mem.add_episode(it["content"], group_id=cfg.group_id, source=source, channel=args.channel, author="user",
                                      occurred_at=it["occurred_at"], source_ref=it["source_ref"])
                stored.append({"file": it["file"], **_episode_summary(cfg, res)})
        return {"group_id": cfg.group_id, "files": len(items), "stored": stored}
    pending: list[dict[str, Any]] = []
    done = 0
    with open_memory(cfg, need_embedder=True) as (mem, warnings):
        for it in items:
            if mem._episode_by_ref(cfg.group_id, source, it["source_ref"]) is not None:
                done += 1
                continue
            if args.dry_run or len(pending) >= args.limit:
                pending.append({"file": it["file"], "source_ref": it["source_ref"]})
                continue
            prep = mem.prepare_episode(it["content"], group_id=cfg.group_id, source=source, channel=args.channel, author="user",
                                       occurred_at=it["occurred_at"], source_ref=it["source_ref"])
            pending.append({"file": it["file"], **prep})
    ready = [p for p in pending if "token" in p]
    out: dict[str, Any] = {
        "group_id": cfg.group_id,
        "files": len(items),
        "already_imported": done,
        "pending": len(pending),
        "items": pending if args.dry_run else ready,
    }
    if ready:
        out["instructions"] = ready[0]["instructions"]
        out["schema"] = ready[0]["schema"]
        for p in ready:
            p.pop("instructions", None)
            p.pop("schema", None)
            p.pop("next", None)
        out["next"] = (
            "For each item: write the extraction from its context (see instructions) and run "
            "`sqgraph-memory commit --token <token>` (or commit_memory). Then run this command again for the next "
            f"{args.limit} files until pending is 0."
        )
    return _with_warning(out, warnings)


def mcp_argv(cfg: MemoryConfig, args: argparse.Namespace) -> list[str]:
    """Arguments for :func:`sqgraph.memory.mcp.main`, filled from the config."""
    argv = [
        "--db", str(cfg.db_path),
        "--group-id", cfg.group_for(getattr(args, "agent", None)),
        "--env-file", str(cfg.env_path),
        "--tz", cfg.tz,
        "--extractor", getattr(args, "extractor", None) or cfg.extractor,
        "--llm-model", cfg.llm_model,
        "--embed-model", cfg.embed_model,
        "--transport", args.transport,
    ]  # fmt: skip
    if cfg.embed_base_url:
        argv += ["--embed-base-url", cfg.embed_base_url, "--embed-api-key-env", cfg.embed_api_key_env]
    if cfg.embed_dimensions:
        argv += ["--embed-dimensions", str(cfg.embed_dimensions)]
    if cfg.search_min_similarity is not None:
        argv += ["--min-similarity", str(cfg.search_min_similarity)]
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


def _assign_group(args: argparse.Namespace, cfg: MemoryConfig, agents: list[str] | None) -> dict[str, Any]:
    """``setup --group G``: record ``agent_groups[agent] = G`` (``default`` removes it) for ``agents`` (``None`` = the detected ones)."""
    raw = args.group.strip()
    removing = raw == DEFAULT_GROUP_ALIAS
    try:
        group = None if removing else validate_group_id(raw)
    except ValueError as exc:
        raise CliError(f"--group: {exc}", 2) from exc
    names = agents if agents is not None else setup_agents.detected_agents(home=user_home(), which=lambda n: shutil.which(n))
    stored = MemoryConfig.load(cfg.home, apply_env=False)
    before = dict(stored.agent_groups)
    for name in names:
        if group is None:
            stored.agent_groups.pop(name, None)
        else:
            stored.agent_groups[name] = group
    changed = stored.agent_groups != before
    if changed and not args.dry_run:
        stored.save()
    if not changed:
        action = "unchanged"
    elif args.dry_run:
        action = "would-remove" if group is None else "would-assign"
    else:
        action = "removed" if group is None else "assigned"
    out: dict[str, Any] = {"group": DEFAULT_GROUP_ALIAS if group is None else group, "agents": names, "action": action, "agent_groups": stored.agent_groups}
    if not names:
        out["note"] = "no agent detected: pass --agents claude-code,codex,opencode,cursor,claude-desktop"
    if cfg.group_from_env:
        out["warning"] = "SQGRAPH_MEMORY_GROUP is set in this environment; it overrides agent_groups for every agent that sees it"
    return out


def cmd_setup(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    try:
        agents = setup_agents.parse_agents(args.agents)
        command, warnings = setup_agents.resolve_launch_command(args.command, which=lambda n: shutil.which(n))
    except ValueError as exc:
        raise CliError(str(exc), 2) from exc
    if args.group is not None and args.uninstall:
        raise CliError("--group cannot be combined with --uninstall (uninstall leaves agent_groups as it is)", 2)
    assignment = _assign_group(args, cfg, agents) if args.group is not None else None
    spec = "none" if args.no_hooks else (args.hooks or "all")
    hooks = [h.strip() for h in spec.split(",") if h.strip()]
    if hooks == ["all"]:
        hooks = list(setup_agents.HOOKS)
    elif hooks == ["none"]:
        hooks = []
    bad = [h for h in hooks if h not in setup_agents.HOOKS]
    if bad:
        raise CliError(f"unknown hook(s): {', '.join(bad)} (choose from {', '.join(setup_agents.HOOKS)}, all or none)", 2)
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
        hooks=hooks,
    )
    if assignment is not None:
        out["group_assignment"] = assignment
    if not args.uninstall and cfg.extractor == "openai" and not cfg.has_api_key():
        out["note_init"] = "no API key configured yet: run `sqgraph-memory init` (agents launch the server without your shell environment)"
    elif not args.uninstall and not cfg.custom_embedder and not cfg.has_api_key():
        out["note_init"] = (
            "no embedding endpoint configured: semantic search is off until you run `sqgraph-memory init --embed-base-url URL` "
            "(a LAN embedding server) or provide an OpenAI key; adding memories needs no key (the agent extracts)"
        )
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
    if cfg.extractor == "agent":
        add(_check("extractor", "ok", "agent: the calling agent extracts entities / facts (`add` then `commit`), no LLM key needed"))
        add(_check("api_key", "ok" if cfg.has_api_key() else "skip",
                   f"{KEY_NAME} configured (used for OpenAI embeddings)" if cfg.has_api_key() else f"{KEY_NAME} not needed (extractor=agent)"))  # fmt: skip
    else:
        add(_check("extractor", "ok", f"openai: `add` calls the OpenAI LLM {cfg.llm_model!r}"))
        add(_check("api_key", "ok" if cfg.has_api_key() else "warn",
                   "configured" if cfg.has_api_key() else f"{KEY_NAME} not found (env or {cfg.env_path}): needed for `add` (the LLM): run `sqgraph-memory init`"))  # fmt: skip
    emb = _embedder_report(cfg)
    emb_detail = f"endpoint={emb['endpoint']}, model={emb['model']}, dimensions={emb['dimensions'] or 'unknown'}"
    if cfg.custom_embedder:
        emb_detail += f", api key {emb['api_key']} (${cfg.embed_api_key_env})"
        known = bool(cfg.embed_dimensions)
        add(_check("embedder", "ok" if known else "warn",
                   emb_detail if known else emb_detail + ": dimensions not probed yet, run `sqgraph-memory config --embed-base-url ...`"))  # fmt: skip
    else:
        add(_check("embedder", "ok" if cfg.has_api_key() else "warn", emb_detail if cfg.has_api_key() else f"{emb_detail}: {KEY_NAME} missing, vector search disabled"))  # fmt: skip

    if cfg.group_from_env:
        add(_check("agent groups", "ok", f"all agents work in group {cfg.group_id!r} (SQGRAPH_MEMORY_GROUP overrides agent_groups)"))
    elif cfg.agent_groups:
        mapped = ", ".join(f"{a} -> {g}" for a, g in sorted(cfg.agent_groups.items()))
        add(_check("agent groups", "ok", f"{mapped}; every other agent uses the default group {cfg.group_id!r}"))
    else:
        add(_check("agent groups", "ok", f"no assignments: every agent shares group {cfg.group_id!r}"))

    if cfg.db_path.is_file():
        try:
            with Memory(cfg.db_path, read_only=True, default_group_id=cfg.group_id) as mem:
                schema = getattr(mem, "schema_version", None)
                st = mem.stats(cfg.group_id)
            extra = f", schema_version={schema}" if schema is not None else ""
            add(_check("database", "ok", f"{cfg.db_path} ({st['nodes']['Episode']} episodes, {st['facts_valid']} valid facts{extra})"))
        except Exception as exc:
            add(_check("database", "fail", f"{cfg.db_path}: {type(exc).__name__}: {exc}"))
        stored = _stored_identity(cfg.db_path)
        mismatch = _identity_mismatch(cfg, stored)
        if mismatch:
            add(_check("database embedding", "fail", mismatch))
        elif stored is None or stored["model"] is None:
            add(_check("database embedding", "ok", "no embedding model recorded yet (it is set by the first write with an embedder)"))  # fmt: skip
        else:
            add(_check("database embedding", "ok", f"{_describe_embedding(stored['model'], stored['dimensions'], stored['endpoint'])} matches the configuration"))  # fmt: skip
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
                n = probe_dimensions(embedder)
                where = cfg.embed_base_url or "openai"
                if n != embedder.dimensions:
                    add(_check("online embedding", "fail", f"{where} produces {n} dimensions for {cfg.embed_model!r}, the configuration says {embedder.dimensions}"))  # fmt: skip
                else:
                    add(_check("online embedding", "ok", f"{where}: {n} dimensions"))
        except Exception as exc:
            add(_check("online embedding", "fail", f"{type(exc).__name__}: {str(exc)[:200]}"))

    ok = not any(c["status"] == "fail" for c in checks)
    return {"ok": ok, "home": str(cfg.home), "checks": checks}


def cmd_doctor(args: argparse.Namespace, cfg: MemoryConfig) -> dict[str, Any]:
    return run_doctor(cfg, online=args.online)


# --------------------------------------------------------------------------- parser / main


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="sqgraph-memory", description="Personal long-term memory shared by local agents (sqgraph.memory).")
    from .. import __version__

    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--compact", action="store_true", help="one-line JSON instead of indented")
    scoped = argparse.ArgumentParser(add_help=False, parents=[common])  # commands that work on one memory group
    scoped.add_argument("--agent", metavar="NAME", help="the agent you are (claude-code, codex, opencode, cursor, ...): use the memory group assigned to it (default: the default group, see `groups`)")  # fmt: skip
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", required=True, parser_class=_Parser)

    def embed_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--embed-base-url", metavar="URL", help="OpenAI-compatible endpoint serving the embeddings, e.g. http://192.168.1.20:11434/v1 (Ollama); 'openai' = the OpenAI API")  # fmt: skip
        p.add_argument("--embed-model", metavar="NAME", help="embedding model name (as the server knows it, e.g. bge-m3)")
        p.add_argument("--embed-dimensions", type=int, metavar="N", help="vector size of the model (default: 1536 for OpenAI, else probed from the endpoint)")  # fmt: skip
        p.add_argument("--embed-api-key-env", metavar="NAME", help="env var / .env key with an optional API key for the endpoint (default: SQGRAPH_EMBED_API_KEY)")  # fmt: skip
        p.add_argument("--no-probe", action="store_true", help="do not call the endpoint to check it / detect the dimensions")

    def extractor_option(p: argparse.ArgumentParser) -> None:
        p.add_argument("--extractor", choices=list(EXTRACTORS), help="who extracts entities / facts: 'agent' (default: the calling agent, no OpenAI key) or 'openai' (one-shot OpenAI LLM, for bots)")  # fmt: skip

    p = sub.add_parser("init", parents=[common], help="create the home directory and config (an OpenAI key only with --extractor openai)")
    p.add_argument("--api-key-stdin", action="store_true", help="read the OpenAI API key from stdin (never pass it as an argument)")
    extractor_option(p)
    p.add_argument("--group-id", help="namespace of your memories (default: me)")
    p.add_argument("--tz", help="timezone for naive timestamps (default: Asia/Tokyo)")
    embed_options(p)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("config", parents=[common], help="show the settings (no secrets) or change them")
    p.add_argument("--group-id")
    p.add_argument("--tz")
    p.add_argument("--agent-group", action="append", metavar="AGENT=GROUP", help="give AGENT its own memory group (repeatable; AGENT= or AGENT=default removes the assignment)")  # fmt: skip
    extractor_option(p)
    p.add_argument("--llm-model", metavar="NAME", help="chat model used by `add` with --extractor openai")
    p.add_argument("--search-min-similarity", metavar="X|default", help="relevance gate of search: minimum cosine similarity (-1..1) for a fact to count as related; 'default' = per embedding model")  # fmt: skip
    embed_options(p)
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("reembed", parents=[common], help="recompute every embedding with the configured embedder (after changing the model)")  # fmt: skip
    p.add_argument("--yes", action="store_true", help="required to actually run (without it the plan is shown)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--progress", action="store_true", help="print JSON progress lines on stderr")
    p.set_defaults(fn=cmd_reembed)

    p = sub.add_parser("add", parents=[scoped], help="remember something (agent mode: prints the context to extract from, then run `commit`; openai mode: LLM extraction)")  # fmt: skip
    p.add_argument("text", metavar="TEXT", help="the text to remember; '-' reads stdin")
    p.add_argument("--source", default="cli", help="who is writing: claude-code / codex / cursor / cli ...")
    p.add_argument("--author", default="user")
    p.add_argument("--channel", default="personal")
    p.add_argument("--occurred-at", help="ISO 8601 time of the event (default: now)")
    p.add_argument("--source-ref", help="idempotency key: the same (source, source-ref) is stored once")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("commit", parents=[scoped], help="store the extraction of an `add` (agent mode): --token T, extraction JSON on stdin or --extraction FILE")  # fmt: skip
    p.add_argument("--token", help='the token printed by `add` (may instead be inside the stdin JSON: {"token": ..., "extraction": ...})')
    p.add_argument("--extraction", metavar="FILE|-", help="file with the extraction JSON; '-' or omitted: stdin")
    p.set_defaults(fn=cmd_commit)

    p = sub.add_parser("search", parents=[scoped], help="search facts (hybrid semantic + keyword + graph)")
    p.add_argument("query", metavar="QUERY")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--as-of", help="answer as of this date/time")
    p.add_argument("--history", action="store_true", help="include superseded / ended facts")
    p.add_argument("--center", metavar="ENTITY", help="rank facts near this entity first")
    p.add_argument("--min-similarity", metavar="X", help="minimum cosine similarity (-1..1) for a semantic match (default: per embedding model, see `config`); lower it to see weaker matches")  # fmt: skip
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("entity", parents=[scoped], help="one entity's profile and full fact timeline")
    p.add_argument("name", metavar="NAME", help="entity name, alias or id")
    p.add_argument("--as-of")
    p.set_defaults(fn=cmd_entity)

    p = sub.add_parser("entities", parents=[scoped], help="find entities by name / alias / description")
    p.add_argument("query", metavar="QUERY")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_entities)

    p = sub.add_parser("episodes", parents=[scoped], help="list raw stored notes (newest first)")
    p.add_argument("--entity")
    p.add_argument("--query")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_episodes)

    p = sub.add_parser("stats", parents=[scoped], help="counts and database info")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("groups", parents=[common], help="the memory groups in the database (episodes / entities / facts) and the agents assigned to each")  # fmt: skip
    p.set_defaults(fn=cmd_groups)

    p = sub.add_parser("forget", parents=[scoped], help="delete memories (exactly one target kind)")
    p.add_argument("--episode", "--episode-id", dest="episode", metavar="ID", help="one stored note (--episode-id is an alias)")
    p.add_argument("--source-ref", metavar="REF")
    p.add_argument("--source", help="with --source-ref, or alone / with a range to forget everything from that source")
    p.add_argument("--fact", "--fact-id", dest="fact", metavar="ID", help="one fact (--fact-id is an alias)")
    p.add_argument("--entity", "--entity-id", dest="entity", metavar="NAME", help="an entity id or exact name (--entity-id is an alias)")
    p.add_argument("--keep-episodes", action="store_true", help="with --entity: keep the source notes")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--channel")
    p.add_argument("--dry-run", action="store_true", help="show what would be removed without deleting")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("merge", parents=[scoped], help="merge entity MERGE into KEEP (the same real-world thing); history is kept")
    p.add_argument("keep", metavar="KEEP", help="entity id or exact name / alias that remains")
    p.add_argument("merge", metavar="MERGE", help="entity id or exact name / alias folded into KEEP")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("invalidate", parents=[scoped], help="mark a fact as no longer true (kept in the history)")
    p.add_argument("fact_id", metavar="FACT_ID")
    p.add_argument("--at", help="when it stopped being true (ISO 8601; default now)")
    p.add_argument("--reason", help="why (stored with the fact)")
    p.set_defaults(fn=cmd_invalidate)

    p = sub.add_parser("relations", parents=[scoped], help="the relation vocabulary with counts and an example each")
    p.set_defaults(fn=cmd_relations)

    p = sub.add_parser("rename-relation", parents=[scoped], help="rename a relation on its facts (--swap also turns them around)")
    p.add_argument("old", metavar="OLD")
    p.add_argument("new", metavar="NEW")
    p.add_argument("--swap", action="store_true", help="also swap subject and object")
    p.add_argument("--fact", action="append", metavar="ID", help="only this fact (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_rename_relation)

    p = sub.add_parser("review", parents=[scoped], help="housekeeping report: duplicate entities, relation suggestions, stale plans")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("upcoming", parents=[scoped], help="facts starting or ending soon (plans, absences, deadlines)")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--past-days", type=float, default=0, help="also include the last N days")
    p.set_defaults(fn=cmd_upcoming)

    p = sub.add_parser("export", parents=[scoped], help="write the group's memory as JSONL (portable, human-readable)")
    p.add_argument("file", metavar="FILE", help="output file; '-' writes the JSONL to stdout")
    p.add_argument("--with-embeddings", action="store_true", help="include the vectors (reused on import with the same model)")
    p.add_argument("--force", action="store_true", help="overwrite FILE")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("import", parents=[scoped], help="load a JSONL export (records already present are skipped)")
    p.add_argument("file", metavar="FILE", help="an export file; '-' reads stdin")
    p.add_argument("--into-group", metavar="G", help="target group (default: the configured group)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--progress", action="store_true", help="print JSON progress lines on stderr")
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("backup", parents=[common], help="consistent snapshot of the database file (default: <home>/backups, rotated)")
    p.add_argument("dest", metavar="DEST", nargs="?", help="target file (default: <home>/backups/memory-YYYYmmdd-HHMMSS.db)")
    p.add_argument("--keep", type=int, default=10, help="snapshots kept in the default directory (default 10)")
    p.set_defaults(fn=cmd_backup)

    p = sub.add_parser("hook", help="agent hook (event JSON on stdin; Claude Code / Codex / the OpenCode plugin): session-start | user-prompt")
    p.add_argument("event", choices=["session-start", "user-prompt"])
    p.add_argument("--agent", default="claude-code", help="the client running the hook (claude-code, codex, opencode); its assigned memory group is used")  # fmt: skip
    p.add_argument("--env", action="append", metavar="KEY=VALUE", help="set an SQGRAPH_MEMORY_* variable first (repeatable)")
    p.set_defaults(fn=cmd_hook, raw=True)

    p = sub.add_parser("import-claude-memory", parents=[scoped], help="import Claude Code's file memory (~/.claude/projects/*/memory/*.md), a few files per run")  # fmt: skip
    p.add_argument("--root", metavar="DIR", help="projects directory (default: ~/.claude/projects)")
    p.add_argument("--limit", type=int, default=5, help="files prepared per run (agent mode) / stored (openai mode)")
    p.add_argument("--channel", default="personal")
    p.add_argument("--dry-run", action="store_true", help="only list the files not imported yet")
    p.set_defaults(fn=cmd_import_claude_memory)

    p = sub.add_parser("purge", parents=[scoped], help="delete ALL memories of the configured group")
    p.add_argument("--yes", action="store_true", help="required to actually purge")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_purge)

    p = sub.add_parser("mcp", help="run the MCP server (what agents launch)")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--read-only", action="store_true")
    p.add_argument("--agent", metavar="NAME", help="the agent this server is for: it serves the memory group assigned to that agent (default: the default group)")  # fmt: skip
    extractor_option(p)
    p.set_defaults(fn=cmd_mcp, raw=True)

    p = sub.add_parser("setup", parents=[common], help="register the MCP server and install the Skill for local agents")
    p.add_argument("--agents", help="comma-separated: claude-code,codex,opencode,cursor,claude-desktop or all (default: detected)")
    p.add_argument("--command", help="full launch command (e.g. uvx --from 'sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph' sqgraph-memory mcp)")
    p.add_argument("--no-skill", action="store_true", help="do not install SKILL.md")
    p.add_argument("--hooks", metavar="LIST", help="hooks to install (Claude Code / Codex hooks, the OpenCode plugin): session-start,user-prompt | all (default) | none")
    p.add_argument("--no-hooks", action="store_true", help="same as --hooks none (existing hooks are left as they are)")
    p.add_argument("--group", metavar="G", help="give the agents (--agents, default: the detected ones) their own memory group G; 'default' removes their assignment")  # fmt: skip
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--uninstall", action="store_true", help="remove the registration and the Skill")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("doctor", parents=[common], help="check the installation (exit code 1 on failure)")
    p.add_argument("--online", action="store_true", help="also make one embedding call (probes the embedding endpoint)")
    p.set_defaults(fn=cmd_doctor)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    _utf8()
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
        for kv in getattr(args, "env", None) or []:  # hooks: the registration may carry SQGRAPH_MEMORY_* settings
            key, sep, value = kv.partition("=")
            if not sep or not key.startswith("SQGRAPH_MEMORY_"):
                raise CliError(f"--env expects SQGRAPH_MEMORY_*=VALUE, got {kv!r}", 2)
            os.environ[key] = value
        cfg = MemoryConfig.load()
        if hasattr(args, "agent"):  # a command that works on memory: use the group of the agent (--agent; None = default group)
            cfg = cfg.for_agent(args.agent)
        result = args.fn(args, cfg)
        if getattr(args, "raw", False) or result is None:  # the command wrote its own output (mcp, export -)
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
        hint = ""
        if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
            hint = " (is the embedding / API server reachable? see `sqgraph-memory doctor --online`)"
        emit_error(f"{type(exc).__name__}: {exc}{hint}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
