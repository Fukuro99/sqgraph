"""Moving memory between databases: a JSONL export / import and consistent SQLite snapshots.

Export format (one JSON object per line, UTF-8)::

    {"kind": "header", "format": "sqgraph-memory-export", "version": 1, "group_id", "exported_at", "embedding", "counts"}
    {"kind": "entity",  "id", "name", "type", "summary", "summaries", "aliases", "alias_src", "channels", ...}
    {"kind": "episode", "id", "source", "channel", "author", "content", "occurred_at", "ingested_at", "source_ref"}
    {"kind": "mention", "episode_id", "entity_id", "role"}
    {"kind": "fact",    "id", "fact", "relation", "subject_id", "object_id", "valid_at", "invalid_at", ...}

Times are the stored UTC ISO strings. Derived fields (``alias_keys``, ``search_text``, ``text_norm``) are not exported;
they are rebuilt on import. Embeddings are only exported with ``with_embeddings`` (``"embedding": [...]``) and only reused
on import when the exporting embedder (``header.embedding``) is the one configured; otherwise everything is re-embedded.
The file is human-readable: it is also a backup that does not depend on the database format.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable, Iterable

from ..errors import SqgraphError
from ..types import Node
from .core import ENTITY_INDEX, FACT_INDEX, FACT_TYPE, Memory
from .util import nfkc, norm_key

FORMAT = "sqgraph-memory-export"
VERSION = 1
_DERIVED = {"kind", "group_id", "alias_keys", "search_text", "text_norm", "embedding"}
_REQUIRED = {
    "entity": ("id", "name"),
    "episode": ("id", "content", "occurred_at"),
    "mention": ("episode_id", "entity_id"),
    "fact": ("id", "fact", "subject_id", "valid_at"),
}


def _clean(props: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if k not in _DERIVED and v is not None}


def export_memory(mem: Memory, out: IO[str], *, group_id: str | None = None, with_embeddings: bool = False) -> dict[str, Any]:
    """Write the group's memory to ``out`` as JSONL; returns the counts written."""
    group = mem._group(group_id)
    db = mem.db
    g = {"g": group}
    entities = [r["e"] for r in db.query("MATCH (e:Entity {group_id:$g}) RETURN e ORDER BY e.created_at", g)]
    episodes = [r["e"] for r in db.query("MATCH (e:Episode {group_id:$g}) RETURN e ORDER BY e.occurred_at", g)]
    mentions = db.query(
        "MATCH (e:Episode {group_id:$g})-[m:MENTIONS]->(x:Entity) RETURN e.id AS episode_id, x.id AS entity_id, m.role AS role", g
    )
    facts = [r["f"] for r in db.query("MATCH ()-[f:FACT {group_id:$g}]->() RETURN f ORDER BY f.created_at", g)]
    ixs = {i.name: i for i in db.vector_indexes()} if with_embeddings else {}
    counts = {"entities": len(entities), "episodes": len(episodes), "mentions": len(mentions), "facts": len(facts)}
    header = {
        "kind": "header",
        "format": FORMAT,
        "version": VERSION,
        "group_id": group,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding": mem.embedding_identity() if with_embeddings else None,
        "counts": counts,
    }

    def line(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    line(header)
    for n in entities:
        rec = {"kind": "entity", **_clean(n.properties)}
        if with_embeddings and ENTITY_INDEX in ixs:
            v = db._vec_read(ixs[ENTITY_INDEX], n.id)
            if v:
                rec["embedding"] = [round(float(x), 7) for x in v]
        line(rec)
    for n in episodes:
        line({"kind": "episode", **_clean(n.properties)})
    for r in mentions:
        line({"kind": "mention", "episode_id": r["episode_id"], "entity_id": r["entity_id"], **({"role": r["role"]} if r["role"] else {})})
    for f in facts:
        rec = {"kind": "fact", **_clean(f.properties)}
        if with_embeddings and FACT_INDEX in ixs:
            v = db._vec_read(ixs[FACT_INDEX], f.id)
            if v:
                rec["embedding"] = [round(float(x), 7) for x in v]
        line(rec)
    return {"group_id": group, **counts, "with_embeddings": with_embeddings}


def read_export(lines: Iterable[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Parse an export: ``(header, {"entity": [...], "episode": [...], "mention": [...], "fact": [...]})``."""
    header: dict[str, Any] | None = None
    recs: dict[str, list[dict[str, Any]]] = {k: [] for k in _REQUIRED}
    for no, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"line {no}: not JSON ({exc})") from None
        if not isinstance(obj, dict):
            raise ValueError(f"line {no}: not a JSON object")
        kind = obj.pop("kind", None)
        if header is None:
            if kind != "header" or obj.get("format") != FORMAT:
                raise ValueError(f"not a {FORMAT} file (the first line must be its header)")
            if obj.get("version") != VERSION:
                raise ValueError(f"unsupported export version {obj.get('version')!r} (this sqgraph reads version {VERSION})")
            header = obj
            continue
        if kind not in _REQUIRED:
            raise ValueError(f"line {no}: unknown record type {kind!r}")
        missing = [k for k in _REQUIRED[kind] if not obj.get(k)]
        if missing:
            raise ValueError(f"line {no}: {kind} without {', '.join(missing)}")
        recs[kind].append(obj)
    if header is None:
        raise ValueError("empty export")
    return header, recs


def import_memory(
    mem: Memory,
    lines: Iterable[str],
    *,
    group_id: str | None = None,
    dry_run: bool = False,
    batch_size: int = 64,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Load an export into ``mem`` (group ``group_id``, default: the exported group). Records whose id already exists
    are skipped (importing twice is a no-op); an episode whose ``(source, source_ref)`` is already stored is skipped too.
    Vectors are reused when the export carries them and was made with the configured embedding model, else computed with
    the configured embedder (none: stored without vectors, vector search ignores them). Everything is written in one
    transaction after all embeddings succeeded."""
    if not dry_run:
        mem._need_writable()
    header, recs = read_export(lines)
    group = group_id or header.get("group_id") or mem._group(None)
    db = mem.db
    warnings: list[str] = []
    with mem._write_guard():

        def exists(label: str, ids: list[str]) -> set[str]:
            found: set[str] = set()
            for i in range(0, len(ids), 200):
                found |= {r["id"] for r in db.query(f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n.id AS id", {"ids": ids[i : i + 200]})}
            return found

        have_ent = exists("Entity", [e["id"] for e in recs["entity"]])
        have_ep = exists("Episode", [e["id"] for e in recs["episode"]])
        have_fact = {f["id"] for f in recs["fact"] if mem._fact_edge(f["id"]) is not None}
        for e in recs["episode"]:
            if e["id"] not in have_ep and e.get("source_ref") and mem._episode_by_ref(group, e.get("source", "api"), e["source_ref"]):
                have_ep.add(e["id"])
        new_ents = [e for e in recs["entity"] if e["id"] not in have_ent]
        new_eps = [e for e in recs["episode"] if e["id"] not in have_ep]
        new_facts = [f for f in recs["fact"] if f["id"] not in have_fact]
        known_ents = {e["id"] for e in recs["entity"]} | have_ent
        for f in new_facts:
            for side in ("subject_id", "object_id"):
                if f.get(side) and f[side] not in known_ents and not mem._entity_node(group, f[side]):
                    raise ValueError(f"fact {f['id']} refers to the unknown entity {f[side]}")
        new_ep_ids = {e["id"] for e in new_eps}
        new_mentions = [m for m in recs["mention"] if m["episode_id"] in new_ep_ids]
        result: dict[str, Any] = {
            "group_id": group,
            "from_group": header.get("group_id"),
            "entities": len(new_ents),
            "episodes": len(new_eps),
            "mentions": len(new_mentions),
            "facts": len(new_facts),
            "skipped": {
                "entities": len(recs["entity"]) - len(new_ents),
                "episodes": len(recs["episode"]) - len(new_eps),
                "facts": len(recs["fact"]) - len(new_facts),
            },
            "dry_run": dry_run,
        }
        if dry_run or not (new_ents or new_eps or new_facts):
            result["vectors"] = "none needed" if not (new_ents or new_facts) else "not computed (dry run)"
            return result

        # -- vectors ------------------------------------------------------------------------
        emb = mem.embedder
        reuse = False
        if emb is not None:
            ident, src = mem.embedder_identity(emb), header.get("embedding") or {}
            reuse = bool(ident and src and src.get("model") == ident["model"] and src.get("dimensions") == ident["dimensions"])
        if emb is None:
            warnings.append("no embedder configured: imported entities / facts have no vectors (run `reembed` later)")
            result["vectors"] = "none"
        else:
            now = mem._now()
            facts_by_ent: dict[str, list[Any]] = {}
            for f in new_facts:
                for x in {f.get("subject_id"), f.get("object_id")} - {None}:
                    facts_by_ent.setdefault(x, []).append(_FactView(f))
            todo: list[tuple[dict[str, Any], str]] = []
            for e in new_ents:
                if not (reuse and e.get("embedding")):
                    todo.append((e, mem._reembed_entity_text(Node(0, ("Entity",), e), facts_by_ent.get(e["id"], ()), now)))
            for f in new_facts:
                if not (reuse and f.get("embedding")):
                    todo.append((f, f["fact"]))
            for i in range(0, len(todo), max(1, batch_size)):
                chunk = todo[i : i + batch_size]
                for (rec, _), v in zip(chunk, emb.embed([t for _, t in chunk])):
                    if len(v) != emb.dimensions:
                        raise SqgraphError(f"the embedder returned a {len(v)}-dimensional vector, expected {emb.dimensions}")
                    rec["embedding"] = v
                if progress is not None:
                    progress("embeddings", min(i + batch_size, len(todo)), len(todo))
            result["vectors"] = "reused" if reuse and not todo else ("computed" if not reuse else "reused + computed")
        if emb is None:  # exported vectors of an unknown model must not reach the index
            for rec in (*new_ents, *new_facts):
                rec.pop("embedding", None)

        # -- write --------------------------------------------------------------------------
        with db.transaction():
            node_ids: dict[str, int] = {}
            for e in new_ents:
                props = _clean(e)
                aliases = list(props.get("aliases", []))
                props.update(
                    group_id=group,
                    alias_keys=sorted({k for k in (norm_key(x) for x in [props["name"], *aliases]) if k}),
                    search_text=nfkc("\n".join([props["name"], *aliases])).casefold(),
                )
                props.setdefault("type", "Other")
                props.setdefault("channels", [])
                if e.get("embedding"):
                    props["embedding"] = e["embedding"]
                node_ids[e["id"]] = db.create_node("Entity", props).id
            ep_ids: dict[str, int] = {}
            for e in new_eps:
                props = _clean(e)
                props.update(group_id=group, text_norm=nfkc(props["content"]).casefold())
                props.setdefault("channel", "default")
                props.setdefault("source", "import")
                ep_ids[e["id"]] = db.create_node("Episode", props).id

            def node_id(eid: str) -> int:
                if eid not in node_ids:
                    n = mem._entity_node(group, eid)
                    if n is None:
                        raise ValueError(f"unknown entity {eid}")
                    node_ids[eid] = n.id
                return node_ids[eid]

            for m in new_mentions:
                if m["entity_id"] in known_ents:
                    db.create_edge(ep_ids[m["episode_id"]], node_id(m["entity_id"]), "MENTIONS", {"role": m["role"]} if m.get("role") else {})
            for f in new_facts:
                props = _clean(f)
                props.update(group_id=group, text_norm=nfkc(props["fact"]).casefold())
                if f.get("embedding"):
                    props["embedding"] = f["embedding"]
                src = node_id(f["subject_id"])
                dst = node_id(f["object_id"]) if f.get("object_id") else src
                db.create_edge(src, dst, FACT_TYPE, props)
            ident = mem.embedder_identity(emb) if emb is not None else None
            if ident is not None and mem.embedding_identity() is None:
                mem._store_embedding_identity(ident)
        if warnings:
            result["warnings"] = warnings
        return result


class _FactView:
    """Just enough of a :class:`Relationship` for ``Memory._reembed_entity_text`` (``get`` / ``[]`` / ``properties``)."""

    def __init__(self, props: dict[str, Any]) -> None:
        self.properties = props

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.properties[key]


# --------------------------------------------------------------------------- snapshots

BACKUP_RE = re.compile(r"^memory-\d{8}-\d{6}\.db$")


def backup_database(mem: Memory, dest: str | Path) -> dict[str, Any]:
    """A consistent copy of the whole database file (SQLite online backup) at ``dest``; refuses to overwrite a file."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    if mem.db.path == ":memory:":
        raise ValueError("an in-memory database has no file to back up")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    target = sqlite3.connect(str(tmp))
    try:
        with mem.db._lock:
            mem.db._conn.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")  # one self-contained file (no -wal / -shm beside it)
        ok = target.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        target.close()
    if ok != "ok":
        tmp.unlink(missing_ok=True)
        raise SqgraphError(f"the backup failed its integrity check: {ok}")
    tmp.replace(dest)
    return {"path": str(dest), "bytes": dest.stat().st_size}


def rotate_backups(directory: Path, keep: int) -> list[str]:
    """Delete the oldest ``memory-YYYYmmdd-HHMMSS.db`` snapshots in ``directory`` beyond the newest ``keep``."""
    if keep < 1:
        return []
    snaps = sorted(p for p in directory.glob("memory-*.db") if BACKUP_RE.match(p.name))
    removed = []
    for p in snaps[:-keep]:
        p.unlink()
        for side in ("-wal", "-shm", "-journal"):
            p.with_name(p.name + side).unlink(missing_ok=True)
        removed.append(str(p))
    return removed
