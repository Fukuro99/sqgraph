"""MCP server exposing the temporal memory to agents (Claude Code, other MCP clients).

    python -m sqgraph.memory.mcp --db memory.db [--group-id G] [--allowed-channels a,b] [--extractor agent|openai]
                                 [--read-only] [--transport stdio|streamable-http] [--port 8000]

The handler logic lives in :class:`MemoryTools` (plain synchronous methods, easy to unit test);
:func:`build_server` registers them with the official ``mcp`` Python SDK (v1 ``FastMCP`` and v2
``MCPServer`` are both supported).

Who extracts the entities / facts of a new memory is ``--extractor``: ``agent`` (default) means the calling agent
itself: ``add_memory`` returns the context + instructions + a token and ``commit_memory(token, extraction)`` stores the
agent's extraction (no OpenAI key on the server); ``openai`` makes ``add_memory`` call an OpenAI LLM in one step (for
bots; ``commit_memory`` stays available).

Access control is enforced here, server-side: with ``--allowed-channels`` every read is restricted to
those channels and ``add_memory`` / ``commit_memory`` may only write into them. ``read_cypher`` is raw graph access that
bypasses that ACL, so it is disabled when ``--allowed-channels`` is set (unless ``--allow-cypher``).
"""

import argparse
import functools
import json
import logging
import os
import sys
import threading
import time
from typing import Any

from ..db import GraphDB
from ..errors import SqgraphError
from .core import FACT_INDEX, Memory
from .providers import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    OpenAILLM,
    UsageTracker,
    build_embedder,
    load_env,
)

log = logging.getLogger("sqgraph.memory.mcp")

CYPHER_NOTES = (
    "sqgraph supports an openCypher SUBSET: MATCH/OPTIONAL MATCH/WHERE/WITH/RETURN/ORDER BY/SKIP/LIMIT/UNWIND/UNION, "
    "variable-length paths (*1..3), list/map expressions, CASE, count/collect/sum/avg/min/max, toLower/coalesce/size/etc. "
    "Property values are primitives or lists of primitives (no maps). NOT supported: CALL {} subqueries, FOREACH, "
    "shortestPath(), pattern predicates in WHERE (use EXISTS { MATCH ... }), datetime types/functions (times are ISO "
    "strings in UTC, compare them as strings), reduce(), APOC. Use parameters ($p) instead of string concatenation."
)

SCHEMA_TEXT = """Graph schema (all nodes and FACT relationships have group_id):
  (:Episode {id, source, channel, author, content, occurred_at, ingested_at, source_ref})  raw messages
  (:Entity  {id, name, type, summary, aliases, alias_keys, channels, created_at, updated_at})
  (subject:Entity)-[:FACT {id, fact, relation, subject_id, object_id, valid_at, invalid_at, created_at, expired_at,
                           channels, episode_ids, confidence, modality, superseded_by, supersedes}]->(object:Entity)
  (Episode)-[:MENTIONS]->(Entity)
A fact is one FACT relationship from its subject entity to its object entity; the verb is f.relation (OWNER, WORKS_ON, ...).
An attribute fact without an object (a status, an absence, ...) is a self-loop (s)-[:FACT]->(s) with object_id absent.
Look up a fact by id with MATCH ()-[f:FACT {id: $id}]->() (indexed). Typical query:
  MATCH (s:Entity)-[f:FACT]->(o:Entity) WHERE f.invalid_at IS NULL AND f.relation = 'OWNER' RETURN s.name, o.name, f.fact
Times are UTC ISO-8601 strings ('2026-07-01T09:52:00Z'). A fact is valid at T when valid_at <= T and
(invalid_at IS NULL or invalid_at > T). Contradicted facts are kept: invalid_at is set and superseded_by holds the id of the
newer fact, whose supersedes property lists the ids of the older ones."""


class MemoryTools:
    """The MCP tool handlers. Construct with a :class:`Memory` and the server-side policy."""

    def __init__(
        self,
        memory: Memory,
        *,
        group_id: str | None = None,
        allowed_channels: list[str] | None = None,
        read_only: bool = False,
        allow_cypher: bool = False,
        max_rows: int = 100,
        max_chars: int = 20000,
        query_timeout: float = 10.0,
        default_channel: str = "personal",
        write_blocker: str | None = None,
        extractor: str | None = None,
    ) -> None:
        self.memory = memory
        # "agent": add_memory returns the prepare payload and the agent calls commit_memory; "openai": add_memory
        # runs the server-side LLM pipeline. None = by what the Memory has: an LLM -> openai, else agent.
        self.extractor = extractor or ("openai" if memory.llm is not None else "agent")
        if self.extractor not in ("agent", "openai"):
            raise ValueError("extractor must be 'agent' or 'openai'")
        self.write_blocker = write_blocker  # why the openai add_memory cannot work (e.g. no OpenAI key); None = it can
        self.default_channel = default_channel
        self.group_id = group_id or memory.default_group_id
        self.locked_group = group_id is not None
        self.allowed = None if allowed_channels is None else list(dict.fromkeys(allowed_channels))
        self.read_only = read_only or memory.db.read_only
        self.cypher_enabled = self.allowed is None or allow_cypher
        self.max_rows = max_rows
        self.max_chars = max_chars
        self.query_timeout = query_timeout
        self._ro: GraphDB | None = None
        self._ro_lock = threading.Lock()

    # -- policy -------------------------------------------------------------------------------

    def _group(self, group_id: str | None) -> str:
        if self.locked_group and group_id and group_id != self.group_id:
            raise PermissionError(f"this server is restricted to group_id={self.group_id!r}")
        g = self.group_id if self.locked_group else (group_id or self.group_id)
        if not g:
            raise ValueError("group_id is required (the server has no default --group-id)")
        return g

    def _channels(self, requested: list[str] | None) -> list[str] | None:
        """Effective allowed channels: the server ACL, optionally narrowed by the caller."""
        if self.allowed is None:
            return requested
        if requested is None:
            return list(self.allowed)
        return [c for c in requested if c in self.allowed]

    # -- tools --------------------------------------------------------------------------------

    def _write_channel(self, channel: str | None) -> str:
        if channel is None:
            channel = self.allowed[0] if self.allowed else self.default_channel
        if self.allowed is not None and channel not in self.allowed:
            raise PermissionError(f"channel {channel!r} is not among the allowed channels")
        return channel

    def add_memory(
        self,
        content: str,
        channel: str | None = None,
        author: str = "claude",
        source: str = "mcp",
        occurred_at: str | None = None,
        source_ref: str | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        if self.read_only:
            raise PermissionError("this server is read-only (add_memory is disabled)")
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        channel = self._write_channel(channel)
        if self.extractor == "agent":
            # step 1 of 2: no LLM on the server; the caller extracts entities / facts and calls commit_memory
            return self.memory.prepare_episode(
                content,
                group_id=self._group(group_id),
                source=source,
                channel=channel,
                author=author,
                occurred_at=occurred_at,
                source_ref=source_ref,
            )
        if self.write_blocker:
            raise RuntimeError(self.write_blocker)
        res = self.memory.add_episode(
            content,
            group_id=self._group(group_id),
            source=source,
            channel=channel,
            author=author,
            occurred_at=occurred_at,
            source_ref=source_ref,
        )
        return self._episode_summary(res)

    @staticmethod
    def _episode_summary(res: Any) -> dict[str, Any]:
        out = {
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

    def commit_memory(
        self, token: str, extraction: dict[str, Any] | str, group_id: str | None = None
    ) -> dict[str, Any]:
        """Step 2 of the agent extraction: store ``extraction`` for the ``token`` returned by ``add_memory``."""
        if self.read_only:
            raise PermissionError("this server is read-only (commit_memory is disabled)")
        tok = Memory.read_token(token)
        group = self._group(group_id)
        if tok["group_id"] != group:
            raise PermissionError(f"this token belongs to group_id={tok['group_id']!r}, not {group!r}")
        if self.allowed is not None and tok["channel"] not in self.allowed:
            raise PermissionError(f"channel {tok['channel']!r} is not among the allowed channels")
        return self._episode_summary(self.memory.commit_episode(token, extraction))

    def forget_memory(
        self,
        episode_id: str | None = None,
        fact_id: str | None = None,
        entity: str | None = None,
        since: str | None = None,
        until: str | None = None,
        channel: str | None = None,
        dry_run: bool = False,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """Permanently delete memory; exactly one selector: ``episode_id`` | ``fact_id`` | ``entity`` | a
        ``since`` / ``until`` / ``channel`` range. Disabled on read-only servers and with ``--allowed-channels``."""
        if self.read_only:
            raise PermissionError("this server is read-only (forget_memory is disabled)")
        if self.allowed is not None:
            raise PermissionError("forget_memory is not available with a channel restriction (--allowed-channels)")
        selectors = [
            name
            for name, given in (
                ("episode_id", episode_id),
                ("fact_id", fact_id),
                ("entity", entity),
                ("since/until/channel", since or until or channel),
            )
            if given
        ]
        if len(selectors) != 1:
            raise ValueError("give exactly one of: episode_id, fact_id, entity, or a since/until/channel range")
        group = self._group(group_id)
        if episode_id:
            res = self.memory.forget_episode(episode_id, group_id=group, dry_run=dry_run)
        elif fact_id:
            res = self.memory.forget_fact(fact_id, group_id=group, dry_run=dry_run)
        elif entity:
            res = self.memory.forget_entity(entity, group_id=group, dry_run=dry_run)
        else:
            res = self.memory.forget_range(
                since=since or None, until=until or None, channel=channel or None, group_id=group, dry_run=dry_run
            )
        return {"group_id": group, **res.to_dict()}

    def search_memory(
        self,
        query: str,
        group_id: str | None = None,
        channels: list[str] | None = None,
        as_of: str | None = None,
        center_entity: str | None = None,
        limit: int = 10,
        include_invalid: bool = False,
        include_planned: bool = True,
    ) -> dict[str, Any]:
        group = self._group(group_id)
        hits = self.memory.search(
            query,
            group_id=group,
            channels=self._channels(channels),
            as_of=as_of,
            center_entity=center_entity,
            limit=max(1, min(limit, 50)),
            include_invalid=include_invalid,
            include_future=include_planned and as_of is None,
        )
        return {"group_id": group, "as_of": as_of, "count": len(hits), "facts": [h.to_dict() for h in hits]}

    def search_entities(
        self,
        query: str,
        group_id: str | None = None,
        channels: list[str] | None = None,
        types: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        group = self._group(group_id)
        ents = self.memory.search_entities(
            query, group_id=group, channels=self._channels(channels), limit=max(1, min(limit, 50)), types=types
        )
        return {"group_id": group, "count": len(ents), "entities": ents}

    def get_entity(
        self,
        name_or_id: str,
        group_id: str | None = None,
        channels: list[str] | None = None,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        group = self._group(group_id)
        res = self.memory.get_entity(name_or_id, group_id=group, channels=self._channels(channels), as_of=as_of)
        if res is None:
            return {"found": False, "message": f"no visible entity matching {name_or_id!r}"}
        return {"found": True, **res}

    def get_episodes(
        self,
        entity: str | None = None,
        query: str | None = None,
        channel: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
        group_id: str | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        group = self._group(group_id)
        eps = self.memory.get_episodes(
            group_id=group,
            channels=self._channels(channels),
            entity=entity,
            channel=channel,
            query=query,
            since=since,
            until=until,
            limit=max(1, min(limit, 100)),
        )
        return {"group_id": group, "count": len(eps), "episodes": eps}

    def get_schema(self, group_id: str | None = None) -> dict[str, Any]:
        group = self._group(group_id) if (self.locked_group or group_id or self.group_id) else None
        st = self.memory.stats(group)
        return {
            "group_id": group,
            "description": SCHEMA_TEXT,
            "counts": st,
            "indexes": [
                {k: v for k, v in i.items() if k in ("name", "type", "labelsOrTypes", "properties")}
                for i in self.memory.db.list_indexes()
            ],
            "channel_restriction": self.allowed,
            "read_only": self.read_only,
            "read_cypher_enabled": self.cypher_enabled,
            "cypher_notes": CYPHER_NOTES,
        }

    # -- raw cypher -----------------------------------------------------------------------------

    def _ro_db(self) -> GraphDB:
        with self._ro_lock:
            if self._ro is None:
                base = self.memory.db
                if base.read_only:
                    self._ro = base
                else:
                    if base.path == ":memory:":
                        raise ValueError("read_cypher needs a file-backed database")
                    self._ro = GraphDB(base.path, read_only=True)
            return self._ro

    def read_cypher(
        self, query: str, params: dict[str, Any] | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        if not self.cypher_enabled:
            raise PermissionError(
                "read_cypher is disabled: it is raw graph access that bypasses the channel ACL "
                "(--allowed-channels is set). Use search_memory / get_entity instead."
            )
        from ..viewer.server import serialize

        cap = max(1, min(limit or self.max_rows, self.max_rows))
        db = self._ro_db()  # a read-only connection: write clauses fail
        deadline = time.monotonic() + self.query_timeout

        def guard() -> int:
            return 1 if time.monotonic() > deadline else 0

        with db._lock:
            db._conn.set_progress_handler(guard, 20000)
            try:
                rows = db.query(query, params or {})
            except Exception as exc:
                if "interrupt" in str(exc).lower():
                    raise TimeoutError(f"query exceeded {self.query_timeout:g}s") from exc
                raise
            finally:
                db._conn.set_progress_handler(None, 0)
        total = len(rows)
        out: list[Any] = []
        chars = 0
        truncated = total > cap
        for r in rows[:cap]:
            ser = serialize(r, 500)
            n = len(json.dumps(ser, ensure_ascii=False))
            if chars + n > self.max_chars:
                truncated = True
                break
            chars += n
            out.append(ser)
        return {"row_count": len(out), "total_rows": total, "truncated": truncated, "rows": out}


# --------------------------------------------------------------------------- server wiring

INSTRUCTIONS = (
    "Long-term memory of a team or a person: a temporal knowledge graph built from chat messages and notes. "
    "Use search_memory first for questions about who/what/when/status; use get_entity for the full history of one "
    "person/service/project; add_memory to record durable facts (decisions, ownership, status changes, plans). "
    "When add_memory returns a `token` (agent extraction) nothing is stored yet: YOU extract the entities and facts "
    "from the returned `context` following `instructions` / `schema`, then call commit_memory(token, extraction). "
    "Facts have validity intervals (valid_at / invalid_at); superseded facts stay available as history. "
    "forget_memory (when present) deletes memory permanently: use it only when the user explicitly asks to forget "
    "something, and preview with dry_run=true."
)

TOOL_DOCS = {
    "add_memory": (
        "Store a new piece of knowledge (a message, decision, status update, plan) in long-term memory. Use it when the "
        "user tells you something worth remembering later (who owns what, a decision, a schedule, an absence, a change of "
        "state). The text is analysed by an LLM: entities and facts are extracted, merged with existing ones, and older "
        "contradicting facts are marked as superseded (never deleted). Write self-contained sentences; put dates in the "
        "text or in occurred_at (ISO 8601, default now). Pass source = your client name (e.g. 'claude-code'). Not available on read-only servers."
    ),
    "add_memory_agent": (
        "Step 1 of 2 to store a new piece of knowledge (a message, decision, preference, plan) in long-term memory; "
        "NOTHING is stored yet. Use it when the user tells you something worth remembering later (who owns what, a "
        "decision, a lasting preference, a schedule, an absence, a change of state). It returns `token`, `instructions`, "
        "`schema` and `context` (the message, the author ref, recent notes, existing candidate entities and currently valid "
        "related facts). YOU (the calling agent) are the extractor: read `instructions`, produce the `extraction` JSON "
        "(entities with refs, facts as subject/relation/object sentences with valid_at/invalid_at), reuse `existing_id` of "
        "candidate entities that are the same thing, set `contradicts` for related facts that are no longer true and "
        "`duplicate_of` for ones it merely restates, then call commit_memory(token, extraction). Never invent facts that "
        "are not in the content. Write self-contained sentences; put dates in the text or in occurred_at (ISO 8601, "
        "default now). Pass source = your client name (e.g. 'claude-code') and author='user' for what the user said. "
        "If the result says already_stored there is nothing to commit. Not available on read-only servers."
    ),
    "commit_memory": (
        "Step 2 of 2 of add_memory: store the extraction you produced. Pass the `token` returned by add_memory unchanged and "
        "`extraction` = {\"entities\": [{ref, name, type, existing_id?, summary?, aliases?}], \"facts\": [{subject, object, "
        "relation, fact, valid_at?, invalid_at?, modality?, confidence?, duplicate_of?, contradicts?}]} (see add_memory's "
        "`schema`; the author is the ref \"author\"). Entities are merged with existing ones, contradicted facts are marked "
        "superseded (never deleted). If it reports problems in the JSON, fix them and call it again with the same token. "
        "The result lists what was created and `warnings` for anything the server dropped or merged. An empty extraction "
        "just stores the note. Not available on read-only servers."
    ),
    "forget_memory": (
        "PERMANENTLY delete memory, only when the user explicitly asks to forget something. Give exactly one of: "
        "episode_id (one stored message/note), fact_id, entity (id or exact name: the entity, all its facts and the "
        "episodes mentioning it) or a time/channel range (since / until / channel: episodes with since <= occurred_at "
        "< until). Facts that only the deleted episodes supported go too; facts they had superseded become valid "
        "again; entities left without any mention are removed. Use dry_run=true first to preview what would be "
        "deleted (ids returned), and confirm with the user. Returns the ids deleted / restored / updated. Not "
        "available on read-only servers or with --allowed-channels (forgetting across channel ACLs is unsupported)."
    ),
    "search_memory": (
        "Search long-term memory for facts (hybrid semantic + keyword + graph search). Use it FIRST for questions about "
        "people, services, projects, decisions, ownership, status or schedules, e.g. 'who owns X now?', 'what is the "
        "status of Y?', 'what was decided about Z?'. Returns facts with validity intervals, status (valid / ended / "
        "scheduled) and source episodes (author, time, channel, source_ref) that you should cite. as_of='2026-07-20' "
        "answers as of a past date; include_invalid=true also returns superseded/ended facts (history, 'how did it "
        "change?'); center_entity ranks facts near a person/service first."
    ),
    "search_entities": (
        "Find entities (people, services, repos, teams, tickets, projects, events, concepts) by name, alias or "
        "description. Use it to resolve who/what a name refers to, or to list what exists, before calling get_entity."
    ),
    "get_entity": (
        "Get one entity's profile and its complete fact timeline, including superseded facts, sorted by time, plus "
        "related entities. Use it for 'tell me everything about X' or 'how did X change over time'. Accepts an id or a "
        "name/alias."
    ),
    "get_episodes": (
        "List the raw source messages/notes (newest first), optionally about an entity, matching a text, in a channel or "
        "time window. Use it to verify a fact against the original wording or to read the surrounding conversation."
    ),
    "get_schema": (
        "Describe the graph schema (labels, relationship types, properties, indexes, counts) and the Cypher subset. Call "
        "it before writing a read_cypher query."
    ),
    "read_cypher": (
        "Run a READ-ONLY openCypher query against the memory graph (row/char/time limited). Use only when the other tools "
        "cannot express what you need (aggregations, custom joins). Call get_schema first. " + CYPHER_NOTES + " "
        "WARNING: raw access ignores channel restrictions; it is disabled when the server runs with --allowed-channels."
    ),
}


def build_server(tools: MemoryTools, name: str = "sqgraph-memory") -> Any:
    """Register ``tools`` with the MCP SDK and return the server object (``.run(transport)``)."""
    import anyio

    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

    server = _Server(name, instructions=INSTRUCTIONS)

    def wrap(fn: Any) -> Any:
        @functools.wraps(fn)
        async def run(**kwargs: Any) -> dict[str, Any]:
            return await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))

        return run

    # Explicit signatures below give the SDK a clean JSON schema for every tool.
    if not tools.read_only:

        @server.tool(
            name="add_memory", description=TOOL_DOCS["add_memory_agent" if tools.extractor == "agent" else "add_memory"]
        )
        async def add_memory(
            content: str,
            channel: str | None = None,
            author: str = "claude",
            source: str = "mcp",
            occurred_at: str | None = None,
            source_ref: str | None = None,
            group_id: str | None = None,
        ) -> dict[str, Any]:
            return await wrap(tools.add_memory)(
                content=content,
                channel=channel,
                author=author,
                source=source,
                occurred_at=occurred_at,
                source_ref=source_ref,
                group_id=group_id,
            )

    if not tools.read_only:

        @server.tool(name="commit_memory", description=TOOL_DOCS["commit_memory"])
        async def commit_memory(token: str, extraction: dict[str, Any] | str, group_id: str | None = None) -> dict[str, Any]:
            return await wrap(tools.commit_memory)(token=token, extraction=extraction, group_id=group_id)

    if not tools.read_only and tools.allowed is None:

        @server.tool(name="forget_memory", description=TOOL_DOCS["forget_memory"])
        async def forget_memory(
            episode_id: str | None = None,
            fact_id: str | None = None,
            entity: str | None = None,
            since: str | None = None,
            until: str | None = None,
            channel: str | None = None,
            dry_run: bool = False,
            group_id: str | None = None,
        ) -> dict[str, Any]:
            return await wrap(tools.forget_memory)(
                episode_id=episode_id,
                fact_id=fact_id,
                entity=entity,
                since=since,
                until=until,
                channel=channel,
                dry_run=dry_run,
                group_id=group_id,
            )

    @server.tool(name="search_memory", description=TOOL_DOCS["search_memory"])
    async def search_memory(
        query: str,
        as_of: str | None = None,
        center_entity: str | None = None,
        limit: int = 10,
        include_invalid: bool = False,
        include_planned: bool = True,
        channels: list[str] | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        return await wrap(tools.search_memory)(
            query=query,
            group_id=group_id,
            channels=channels,
            as_of=as_of,
            center_entity=center_entity,
            limit=limit,
            include_invalid=include_invalid,
            include_planned=include_planned,
        )

    @server.tool(name="search_entities", description=TOOL_DOCS["search_entities"])
    async def search_entities(
        query: str,
        types: list[str] | None = None,
        limit: int = 10,
        channels: list[str] | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        return await wrap(tools.search_entities)(query=query, group_id=group_id, channels=channels, types=types, limit=limit)

    @server.tool(name="get_entity", description=TOOL_DOCS["get_entity"])
    async def get_entity(
        name_or_id: str,
        as_of: str | None = None,
        channels: list[str] | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        return await wrap(tools.get_entity)(name_or_id=name_or_id, group_id=group_id, channels=channels, as_of=as_of)

    @server.tool(name="get_episodes", description=TOOL_DOCS["get_episodes"])
    async def get_episodes(
        entity: str | None = None,
        query: str | None = None,
        channel: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
        channels: list[str] | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        return await wrap(tools.get_episodes)(
            entity=entity, query=query, channel=channel, since=since, until=until, limit=limit, group_id=group_id, channels=channels
        )

    @server.tool(name="get_schema", description=TOOL_DOCS["get_schema"])
    async def get_schema(group_id: str | None = None) -> dict[str, Any]:
        return await wrap(tools.get_schema)(group_id=group_id)

    if tools.cypher_enabled:

        @server.tool(name="read_cypher", description=TOOL_DOCS["read_cypher"])
        async def read_cypher(query: str, params: dict[str, Any] | None = None, limit: int | None = None) -> dict[str, Any]:
            return await wrap(tools.read_cypher)(query=query, params=params, limit=limit)

    return server


def run_server(server: Any, transport: str, host: str, port: int) -> None:
    if transport == "stdio":
        server.run("stdio")
        return
    import anyio

    if hasattr(server, "run_streamable_http_async"):
        import inspect

        params = inspect.signature(server.run_streamable_http_async).parameters
        if "host" in params:  # mcp >= 2: host / port are arguments
            anyio.run(functools.partial(server.run_streamable_http_async, host=host, port=port))
            return
    server.settings.host = host  # mcp 1.x FastMCP
    server.settings.port = port
    server.run("streamable-http")


def _existing_dims(path: str) -> int | None:
    try:
        with GraphDB(path, read_only=True) as db:
            for ix in db.vector_indexes():
                if ix.name == FACT_INDEX:
                    return ix.dimensions
    except SqgraphError:
        return None
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sqgraph.memory.mcp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, help="path of the memory database (created when writable)")
    ap.add_argument("--group-id", help="fix the namespace (tool calls cannot use another group)")
    ap.add_argument("--allowed-channels", help="comma-separated channels readable/writable through this server")
    ap.add_argument(
        "--default-channel",
        default="personal",
        help="channel add_memory writes to when none is given (and no --allowed-channels); facts only merge within a channel",
    )
    ap.add_argument("--read-only", action="store_true", help="open the DB read-only and disable add_memory / commit_memory")
    ap.add_argument(
        "--extractor",
        choices=["agent", "openai"],
        default="agent",
        help="who extracts entities / facts: 'agent' (default: the calling agent, via add_memory + commit_memory; no OpenAI key) "
        "or 'openai' (add_memory calls an OpenAI LLM in one step; needs OPENAI_API_KEY)",
    )
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-cypher", action="store_true", help="keep read_cypher enabled even with --allowed-channels (bypasses the ACL!)")
    ap.add_argument("--env-file", help=".env with OPENAI_API_KEY (default: search from the current directory)")
    ap.add_argument("--llm-model", default=DEFAULT_CHAT_MODEL, help="chat model of --extractor openai")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument(
        "--embed-base-url",
        help="OpenAI-compatible /v1 endpoint serving the embeddings (e.g. http://192.168.1.20:11434/v1); default: the OpenAI API",
    )
    ap.add_argument("--embed-dimensions", type=int, help="dimensions of the embedding model (default: 1536 for OpenAI, else probed)")
    ap.add_argument(
        "--embed-api-key-env",
        default="SQGRAPH_EMBED_API_KEY",
        help="environment variable (or .env key) holding an optional API key for --embed-base-url",
    )
    ap.add_argument("--tz", default="Asia/Tokyo", help="timezone for naive timestamps / displayed times")
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--query-timeout", type=float, default=10.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    load_env(args.env_file)
    allowed = [c.strip() for c in args.allowed_channels.split(",") if c.strip()] if args.allowed_channels else None
    tracker = UsageTracker()
    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    embedder = None
    if args.embed_base_url or have_key:  # a LAN embedder needs no OpenAI key
        try:
            embedder, _ = build_embedder(
                args.embed_model,
                base_url=args.embed_base_url,
                dimensions=args.embed_dimensions,
                existing_dims=_existing_dims(args.db),
                api_key=os.environ.get(args.embed_api_key_env) or None,
                tracker=tracker,
            )
        except Exception as exc:
            raise SystemExit(
                f"cannot reach the embedding endpoint {args.embed_base_url} to find its dimensions ({type(exc).__name__}: "
                f"{str(exc)[:200]}): start the server, or pass --embed-dimensions"
            ) from exc
    else:
        log.warning("OPENAI_API_KEY is not set: semantic (vector) search is disabled")
    llm = None
    write_blocker = None
    if not args.read_only and args.extractor == "openai":
        if have_key:
            llm = OpenAILLM(args.llm_model, tracker=tracker)
        else:
            write_blocker = "OPENAI_API_KEY is not set: add_memory needs the OpenAI API (run `sqgraph-memory init`), or use --extractor agent"
    memory = Memory(args.db, llm, embedder, default_group_id=args.group_id, tz=args.tz, read_only=args.read_only)
    tools = MemoryTools(
        memory,
        group_id=args.group_id,
        allowed_channels=allowed,
        read_only=args.read_only,
        allow_cypher=args.allow_cypher,
        max_rows=args.max_rows,
        query_timeout=args.query_timeout,
        default_channel=args.default_channel,
        write_blocker=write_blocker,
        extractor=args.extractor,
    )
    if write_blocker:
        log.warning("%s", write_blocker)
    if allowed is not None and args.allow_cypher:
        log.warning("read_cypher is enabled together with --allowed-channels: it BYPASSES the channel ACL")
    elif allowed is not None:
        log.info("read_cypher disabled (channel restriction active)")
    server = build_server(tools)
    try:
        run_server(server, args.transport, args.host, args.port)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
