"""Configuration of the personal memory (``sqgraph-memory``): home directory, ``config.json`` and ``.env``.

Layout of the home directory (``$SQGRAPH_MEMORY_HOME`` or ``~/.sqgraph-memory``)::

    memory.db      the SQLite database shared by every local agent
    memory.db.lock cross-process write lock (created by sqgraph.memory)
    .env           OPENAI_API_KEY=...   (mode 0600 on POSIX; optionally SQGRAPH_EMBED_API_KEY=... for a custom embedder endpoint)
    config.json    {"group_id", "tz", "llm_model", "embed_model", "embed_base_url", "embed_dimensions",
                    "embed_api_key_env", "db"}

Environment overrides: ``SQGRAPH_MEMORY_HOME``, ``SQGRAPH_MEMORY_DB``, ``SQGRAPH_MEMORY_GROUP``.
API key values are never printed or logged by this module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_DIMS, DEFAULT_EMBED_MODEL

ENV_HOME = "SQGRAPH_MEMORY_HOME"
ENV_DB = "SQGRAPH_MEMORY_DB"
ENV_GROUP = "SQGRAPH_MEMORY_GROUP"
DEFAULT_GROUP = "me"
DEFAULT_TZ = "Asia/Tokyo"
KEY_NAME = "OPENAI_API_KEY"
DEFAULT_EMBED_KEY_ENV = "SQGRAPH_EMBED_API_KEY"

_POSIX = os.name == "posix"


def default_home() -> Path:
    """The memory home directory (env override, else ``~/.sqgraph-memory``)."""
    env = os.environ.get(ENV_HOME)
    return Path(env).expanduser() if env else Path.home() / ".sqgraph-memory"


def ensure_private_dir(path: Path) -> None:
    """Create ``path`` (0700 on POSIX, best effort)."""
    path.mkdir(parents=True, exist_ok=True)
    if _POSIX:
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def write_private(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8) to ``path`` with mode 0600 on POSIX (best effort elsewhere)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _POSIX:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        finally:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


@dataclass
class MemoryConfig:
    """Resolved settings of the personal memory."""

    home: Path
    group_id: str = DEFAULT_GROUP
    tz: str = DEFAULT_TZ
    llm_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_base_url: str | None = None  # None = the OpenAI API; else an OpenAI-compatible /v1 endpoint (e.g. on the LAN)
    embed_dimensions: int | None = None  # None = 1536 (OpenAI), or probed from a custom endpoint and then saved
    embed_api_key_env: str = DEFAULT_EMBED_KEY_ENV  # env var / .env key holding an optional key for that endpoint
    db: str | None = None  # None = <home>/memory.db

    # -- paths ---------------------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    @property
    def env_path(self) -> Path:
        return self.home / ".env"

    @property
    def db_path(self) -> Path:
        return Path(self.db).expanduser() if self.db else self.home / "memory.db"

    # -- load / save ---------------------------------------------------------------------------

    @classmethod
    def load(cls, home: str | os.PathLike[str] | None = None, *, apply_env: bool = True) -> "MemoryConfig":
        """Read ``config.json`` (missing file = defaults). ``apply_env`` layers the env overrides on top."""
        h = Path(home).expanduser() if home is not None else default_home()
        cfg = cls(home=h)
        p = cfg.config_path
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot read {p}: {exc}") from exc
            if isinstance(data, dict):
                cfg.group_id = str(data.get("group_id") or cfg.group_id)
                cfg.tz = str(data.get("tz") or cfg.tz)
                cfg.llm_model = str(data.get("llm_model") or cfg.llm_model)
                cfg.embed_model = str(data.get("embed_model") or cfg.embed_model)
                cfg.embed_base_url = str(data.get("embed_base_url") or "").strip() or None
                dims = data.get("embed_dimensions")
                cfg.embed_dimensions = dims if isinstance(dims, int) and not isinstance(dims, bool) and dims > 0 else None
                cfg.embed_api_key_env = str(data.get("embed_api_key_env") or cfg.embed_api_key_env)
                cfg.db = data.get("db") or None
        if apply_env:
            if os.environ.get(ENV_GROUP):
                cfg.group_id = os.environ[ENV_GROUP]
            if os.environ.get(ENV_DB):
                cfg.db = os.environ[ENV_DB]
        return cfg

    def to_json(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "tz": self.tz,
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
            "embed_base_url": self.embed_base_url,
            "embed_dimensions": self.embed_dimensions,
            "embed_api_key_env": self.embed_api_key_env,
            "db": self.db,
        }

    @property
    def custom_embedder(self) -> bool:
        """True when embeddings come from ``embed_base_url`` instead of the OpenAI API."""
        return bool(self.embed_base_url)

    def effective_embed_dimensions(self) -> int | None:
        """Configured dimensions; the OpenAI default for the OpenAI API; ``None`` (= probe) for a custom endpoint."""
        if self.embed_dimensions:
            return self.embed_dimensions
        return None if self.custom_embedder else DEFAULT_EMBED_DIMS

    def save(self) -> None:
        ensure_private_dir(self.home)
        write_private(self.config_path, json.dumps(self.to_json(), ensure_ascii=False, indent=2) + "\n")

    # -- API key (.env) ------------------------------------------------------------------------

    def write_api_key(self, key: str, name: str = KEY_NAME) -> None:
        """Store ``name`` (default ``OPENAI_API_KEY``) in ``<home>/.env`` (other lines are kept). The value is never echoed."""
        key = key.strip()
        if not key or "\n" in key or "\r" in key:
            raise ValueError("invalid API key")
        ensure_private_dir(self.home)
        lines: list[str] = []
        if self.env_path.is_file():
            lines = self.env_path.read_text(encoding="utf-8").splitlines()
        pat = re.compile(rf"\s*(?:export\s+)?{re.escape(name)}\s*=")
        out = [ln for ln in lines if not pat.match(ln)]
        out.append(f"{name}={key}")
        write_private(self.env_path, "\n".join(out) + "\n")

    def has_embed_api_key(self) -> bool:
        """True if the optional key of the custom embedder endpoint (``embed_api_key_env``) is set."""
        return self.has_api_key(self.embed_api_key_env)

    def has_api_key(self, name: str = KEY_NAME) -> bool:
        """True if ``name`` (default ``OPENAI_API_KEY``) is in the environment or in ``<home>/.env`` (the value is not returned)."""
        if os.environ.get(name):
            return True
        if self.env_path.is_file():
            try:
                for ln in self.env_path.read_text(encoding="utf-8").splitlines():
                    m = re.match(rf"\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.*)\s*$", ln)
                    if m and m.group(1).strip().strip("'\""):
                        return True
            except OSError:
                return False
        return False
