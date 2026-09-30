"""HTTP server and JSON API of the graph viewer (standard library only)."""

from __future__ import annotations

import argparse
import json
import math
import errno
import os
import socket
import sqlite3
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path as FsPath
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, unquote, urlparse

from ..db import GraphDB
from ..errors import SqgraphError
from ..types import Node, Path, Relationship

STATIC_DIR = FsPath(__file__).parent / "static"
MAX_BODY = 1_000_000
MAX_LIMIT = 5000
MAX_TABLE_ROWS = 2000
MAX_EDGES = 20000
QUERY_TIMEOUT = 30.0  # seconds
LIGHT_STR = 200  # strings in graph elements are truncated; /api/node/<id> returns everything
NAME_KEYS = ("name", "title")
SEARCH_KEYS = ("name", "title", "id", "key", "text")


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------- serialization


def is_vector(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 64
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
    )


def serialize(value: Any, max_str: int | None = None) -> Any:
    """Convert query values to JSON-safe data; Node/Relationship/Path get a ``kind`` tag."""
    if isinstance(value, Node):
        return {
            "kind": "node",
            "id": value.id,
            "labels": list(value.labels),
            "properties": serialize(value.properties, max_str),
        }
    if isinstance(value, Relationship):
        return {
            "kind": "relationship",
            "id": value.id,
            "type": value.type,
            "source": value.start_node_id,
            "target": value.end_node_id,
            "properties": serialize(value.properties, max_str),
        }
    if isinstance(value, Path):
        return {
            "kind": "path",
            "nodes": [serialize(n, max_str) for n in value.nodes],
            "relationships": [serialize(r, max_str) for r in value.relationships],
        }
    if isinstance(value, str):
        return value if max_str is None or len(value) <= max_str else value[:max_str] + "…"
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): serialize(v, max_str) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if is_vector(value):
            return f"<vector: {len(value)} dims>"
        return [serialize(v, max_str) for v in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)


def caption(node: dict[str, Any]) -> str:
    """Short human-readable name of a serialized node."""
    props = node.get("properties", {})
    for key in (*NAME_KEYS, "id"):
        if props.get(key) not in (None, ""):
            return str(props[key])
    if props.get("text"):
        return str(props["text"])[:20]
    return (node.get("labels") or [str(node["id"])])[0]


def collect_graph(values: Iterable[Any]) -> tuple[dict[int, Node], dict[int, Relationship]]:
    """Gather every Node / Relationship (also inside Paths, lists and maps) found in ``values``."""
    nodes: dict[int, Node] = {}
    rels: dict[int, Relationship] = {}

    def walk(v: Any) -> None:
        if isinstance(v, Node):
            nodes[v.id] = v
        elif isinstance(v, Relationship):
            rels[v.id] = v
        elif isinstance(v, Path):
            for x in (*v.nodes, *v.relationships):
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for v in values:
        walk(v)
    return nodes, rels


# ---------------------------------------------------------------------- service


class ViewerService:
    """Read-only query layer over one database. All access is serialized by a lock."""

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.path = str(db_path)
        self.db = GraphDB(self.path, read_only=True)
        self.lock = threading.RLock()
        self._deadline = float("inf")
        conn = self.db._conn
        conn.set_progress_handler(lambda: 1 if time.monotonic() > self._deadline else 0, 20000)
        conn.create_function(
            "sqv_lower", 1, lambda s: s.lower() if isinstance(s, str) else s, deterministic=True
        )
        self._has_communities = self._detect_communities()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # -- helpers (call with the lock held)

    def _sql(self, sql: str, *args: Any) -> list[tuple[Any, ...]]:
        return self.db._conn.execute(sql, args).fetchall()

    def _detect_communities(self) -> bool:
        return bool(
            self._sql("SELECT 1 FROM node_labels WHERE label = 'Community' LIMIT 1")
            and self._sql("SELECT 1 FROM edges WHERE type = 'IN_COMMUNITY' LIMIT 1")
        )

    @staticmethod
    def _clamp(value: Any, default: int, maximum: int = MAX_LIMIT) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, min(n, maximum))

    def _annotate(self, nodes: list[Node]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Serialize nodes (long strings truncated) with ``degree`` and ``community``.

        Also returns ``{community id: caption}`` for the communities referenced.
        """
        ids = json.dumps([n.id for n in nodes])
        deg = {
            r[0]: r[1]
            for r in self._sql(
                "SELECT value, (SELECT count(*) FROM edges WHERE src = value)"
                " + (SELECT count(*) FROM edges WHERE dst = value) FROM json_each(?)",
                ids,
            )
        }
        comm: dict[int, int] = {}
        if self._has_communities and nodes:
            comm = dict(
                self._sql(
                    "SELECT src, min(dst) FROM edges WHERE type = 'IN_COMMUNITY' "
                    "AND src IN (SELECT value FROM json_each(?)) GROUP BY src",
                    ids,
                )
            )
        names = {
            str(cn.id): caption(serialize(cn, LIGHT_STR))
            for cn in self.db._load_nodes(set(comm.values())).values()
        }
        out = []
        for n in nodes:
            d = serialize(n, LIGHT_STR)
            d["degree"] = deg.get(n.id, 0)
            d["community"] = comm.get(n.id)
            out.append(d)
        return out, names

    def _edges_among(self, ids: list[int]) -> list[Relationship]:
        if not ids:
            return []
        j = json.dumps(ids)
        rows = self._sql(
            "SELECT id, type, src, dst, props FROM edges WHERE src IN (SELECT value FROM json_each(?)) "
            "AND dst IN (SELECT value FROM json_each(?)) ORDER BY id LIMIT ?",
            j,
            j,
            MAX_EDGES,
        )
        return [Relationship(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows]

    def _payload(self, nodes: list[Node], rels: list[Relationship], **extra: Any) -> dict[str, Any]:
        node_dicts, communities = self._annotate(nodes)
        return {
            "nodes": node_dicts,
            "edges": [serialize(r, LIGHT_STR) for r in rels],
            "communities": communities,
            **extra,
        }

    def _node(self, node_id: int) -> Node:
        node = self.db._load_nodes([node_id]).get(node_id)
        if node is None:
            raise ApiError(404, f"node {node_id} not found")
        return node

    # -- API

    def overview(self) -> dict[str, Any]:
        with self.lock:
            self._has_communities = self._detect_communities()
            keys = self._sql(
                "SELECT DISTINCT j.key FROM (SELECT props FROM nodes LIMIT 5000) n, json_each(n.props) j "
                "ORDER BY j.key"
            )
            p = FsPath(self.path)
            return {
                "file": p.name,
                "size": p.stat().st_size,
                "nodes": self._sql("SELECT count(*) FROM nodes")[0][0],
                "relationships": self._sql("SELECT count(*) FROM edges")[0][0],
                "labels": [
                    {"name": name, "count": c}
                    for name, c in self._sql(
                        "SELECT label, count(*) c FROM node_labels GROUP BY label ORDER BY c DESC, label"
                    )
                ],
                "types": [
                    {"name": name, "count": c}
                    for name, c in self._sql(
                        "SELECT type, count(*) c FROM edges GROUP BY type ORDER BY c DESC, type"
                    )
                ],
                "propertyKeys": [r[0] for r in keys],
                "indexes": self.db.list_indexes(),
                "hasCommunities": self._has_communities,
            }

    def graph(self, limit: Any = 200, labels: str = "") -> dict[str, Any]:
        """The ``limit`` highest-degree nodes (optionally of the given labels) and the edges among them."""
        n = self._clamp(limit, 200)
        wanted = [x for x in labels.split(",") if x]
        if wanted:
            cand = "SELECT DISTINCT node_id AS id FROM node_labels WHERE label IN (SELECT value FROM json_each(?))"
            args: tuple[Any, ...] = (json.dumps(wanted), n)
        else:
            cand = "SELECT id FROM nodes"
            args = (n,)
        with self.lock:
            ids = [
                r[0]
                for r in self._sql(
                    f"WITH c AS ({cand}) SELECT c.id FROM c ORDER BY (SELECT count(*) FROM edges WHERE src = c.id)"
                    " + (SELECT count(*) FROM edges WHERE dst = c.id) DESC, c.id LIMIT ?",
                    *args,
                )
            ]
            loaded = self.db._load_nodes(ids)
            nodes = [loaded[i] for i in ids if i in loaded]
            return self._payload(nodes, self._edges_among(ids))

    def node(self, node_id: int) -> dict[str, Any]:
        with self.lock:
            node = self._node(node_id)
            [d], comms = self._annotate([node])
            d["properties"] = serialize(node.properties)  # full text, no truncation
            groups = self._sql(
                "SELECT type, 'out', count(*) FROM edges WHERE src = ?1 GROUP BY type "
                "UNION ALL SELECT type, 'in', count(*) FROM edges WHERE dst = ?1 GROUP BY type",
                node_id,
            )
            d["relationships"] = [{"type": t, "direction": di, "count": c} for t, di, c in groups]
            d["communities"] = comms
            return d

    def expand(self, node_id: int, limit: Any = 100, types: str = "") -> dict[str, Any]:
        """One hop around ``node_id``: incident edges (up to ``limit``) and the neighbor nodes."""
        n = self._clamp(limit, 100)
        wanted = json.dumps([x for x in types.split(",") if x])
        flt = "(?2 = '[]' OR type IN (SELECT value FROM json_each(?2)))"
        with self.lock:
            self._node(node_id)
            base = (
                f"SELECT id, type, src, dst, props FROM edges WHERE src = ?1 AND {flt} "
                f"UNION ALL SELECT id, type, src, dst, props FROM edges WHERE dst = ?1 AND src <> ?1 AND {flt}"
            )
            total = self._sql(f"SELECT count(*) FROM ({base})", node_id, wanted)[0][0]
            rows = self._sql(f"{base} ORDER BY id LIMIT ?3", node_id, wanted, n)
            rels = [Relationship(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows]
            other = {i for r in rels for i in (r.start_node_id, r.end_node_id)} - {node_id}
            loaded = self.db._load_nodes(other)
            nodes = [loaded[i] for i in sorted(other) if i in loaded]
            return self._payload(nodes, rels, center=node_id, total=total, truncated=total > len(rels))

    def edges(self, ids: list[int]) -> dict[str, Any]:
        with self.lock:
            return {"edges": [serialize(r, LIGHT_STR) for r in self._edges_among(ids)]}

    def search(self, q: str, limit: Any = 30) -> dict[str, Any]:
        """Case-insensitive substring search over name/title/id/key/text (names rank first)."""
        q = q.strip()
        n = self._clamp(limit, 30, 100)
        if not q:
            return {"nodes": [], "communities": {}}

        def field(key: str) -> str:
            return f"instr(sqv_lower(coalesce(json_extract(props, '$.{key}'), '')), ?1) > 0"

        rank = (
            f"CASE WHEN {' OR '.join(field(k) for k in NAME_KEYS)} THEN 0 "
            f"WHEN {field('id')} OR {field('key')} THEN 1 ELSE 2 END"
        )
        cond = " OR ".join(field(k) for k in SEARCH_KEYS)
        by_id = int(q.lstrip("#")) if q.lstrip("#").isdigit() else -1
        with self.lock:
            rows = self._sql(
                f"SELECT id FROM (SELECT id, {rank} AS r FROM nodes WHERE {cond} OR id = ?2) "
                "ORDER BY (id = ?2) DESC, r, id LIMIT ?3",
                q.lower(),
                by_id,
                n,
            )
            ids = [r[0] for r in rows]
            loaded = self.db._load_nodes(ids)
            nodes, comms = self._annotate([loaded[i] for i in ids if i in loaded])
            return {"nodes": nodes, "communities": comms}

    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run read-only Cypher; returns a table plus the graph elements found in the result."""
        if not isinstance(cypher, str) or not cypher.strip():
            raise ApiError(400, "empty query")
        if params is not None and not isinstance(params, dict):
            raise ApiError(400, "params must be an object")
        with self.lock:
            started = time.monotonic()
            self._deadline = started + QUERY_TIMEOUT
            try:
                rows = self.db.query(cypher, params or {})
            except sqlite3.OperationalError as exc:
                if "interrupt" in str(exc).lower():
                    raise ApiError(408, f"query timed out after {QUERY_TIMEOUT:.0f}s") from exc
                raise
            finally:
                self._deadline = float("inf")
            elapsed = time.monotonic() - started
            columns = list(rows[0].keys()) if rows else []
            nodes, rels = collect_graph(v for row in rows for v in row.values())
            missing = {i for r in rels.values() for i in (r.start_node_id, r.end_node_id)} - nodes.keys()
            nodes.update(self.db._load_nodes(missing))
            payload = self._payload(
                [nodes[i] for i in sorted(nodes)], [rels[i] for i in sorted(rels)]
            )
            return {
                "columns": columns,
                "rows": [[serialize(row[c]) for c in columns] for row in rows[:MAX_TABLE_ROWS]],
                "rowCount": len(rows),
                "truncated": len(rows) > MAX_TABLE_ROWS,
                "elapsedMs": round(elapsed * 1000, 1),
                "graph": payload,
            }


# ---------------------------------------------------------------------- HTTP


def _host_name(header: str) -> str:
    """Host part of a ``Host`` header value (handles ``[::1]:8765``)."""
    if header.startswith("["):
        return header[1 : header.find("]")].lower()
    return header.rsplit(":", 1)[0].lower() if header.count(":") == 1 else header.lower()


class Handler(BaseHTTPRequestHandler):
    server: "ViewerServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _check_host(self) -> None:
        """Reject foreign Host headers (DNS rebinding) unless bound to a wildcard address."""
        if not self.server.any_host and _host_name(self.headers.get("Host") or "") not in self.server.allowed_hosts:
            raise ApiError(403, "forbidden host")

    def _dispatch(self, fn: Callable[[], Any]) -> None:
        try:
            self._check_host()
            self._json(200, fn())
        except ApiError as exc:
            self._json(exc.status, {"error": exc.message})
        except SqgraphError as exc:
            self._json(400, {"error": str(exc), "kind": type(exc).__name__})
        except sqlite3.Error as exc:
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - report to the UI instead of dropping the connection
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    @staticmethod
    def _int(text: str) -> int:
        try:
            return int(unquote(text))
        except ValueError:
            raise ApiError(400, f"invalid id: {text!r}") from None

    def _index(self) -> bytes:
        self._check_host()
        return (STATIC_DIR / "index.html").read_bytes()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        qs = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        route = [p for p in url.path.split("/") if p]
        svc = self.server.service
        if not route:
            try:
                self._send(200, self._index(), "text/html; charset=utf-8")
            except ApiError as exc:
                self._json(exc.status, {"error": exc.message})
            return
        if route[0] != "api":
            return self._json(404, {"error": "not found"})
        route = route[1:]
        if route == ["overview"]:
            self._dispatch(svc.overview)
        elif route == ["graph"]:
            self._dispatch(lambda: svc.graph(qs.get("limit", 200), qs.get("labels", "")))
        elif route == ["search"]:
            self._dispatch(lambda: svc.search(qs.get("q", ""), qs.get("limit", 30)))
        elif len(route) == 2 and route[0] == "node":
            self._dispatch(lambda: svc.node(self._int(route[1])))
        elif len(route) == 2 and route[0] == "expand":
            self._dispatch(lambda: svc.expand(self._int(route[1]), qs.get("limit", 100), qs.get("types", "")))
        else:
            self._json(404, {"error": "not found"})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "request body too large")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ApiError(400, f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError(400, "request body must be a JSON object")
        return data

    def do_POST(self) -> None:
        route = [p for p in urlparse(self.path).path.split("/") if p]
        svc = self.server.service

        def query() -> dict[str, Any]:
            data = self._body()
            return svc.run_query(data.get("cypher", ""), data.get("params"))

        def edges() -> dict[str, Any]:
            ids = self._body().get("ids", [])
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise ApiError(400, "ids must be a list of integers")
            return svc.edges(ids)

        if route == ["api", "query"]:
            self._dispatch(query)
        elif route == ["api", "edges"]:
            self._dispatch(edges)
        else:
            self._json(404, {"error": "not found"})


DEFAULT_PORT = 8765


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is already in use,
    # silently leaving requests to the first one; claim the port exclusively there instead.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, service: ViewerService, host: str, port: int, verbose: bool = False) -> None:
        self.service = service
        self.verbose = verbose
        self.any_host = host in ("", "0.0.0.0", "::")
        self.allowed_hosts = {"localhost", "127.0.0.1", "::1", host.lower()}
        super().__init__((host, port), Handler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{'[' + host + ']' if ':' in host else host}:{port}/"

    def server_close(self) -> None:
        super().server_close()
        self.service.close()


def make_server(
    db_path: str | os.PathLike[str], host: str = "127.0.0.1", port: int = 8765, verbose: bool = False
) -> ViewerServer:
    """Open ``db_path`` read-only and create (but do not start) the server."""
    service = ViewerService(db_path)
    try:
        return ViewerServer(service, host, port, verbose)
    except BaseException:
        service.close()
        raise


def _port_in_use(exc: OSError) -> bool:
    return exc.errno in (errno.EADDRINUSE, errno.EACCES) or getattr(exc, "winerror", None) in (10048, 10013)


def make_server_auto(
    db_path: str | os.PathLike[str], host: str = "127.0.0.1", port: int | None = None, verbose: bool = False
) -> ViewerServer:
    """Like make_server, but with ``port=None`` try DEFAULT_PORT and the next ports until one is free."""
    if port is not None:
        return make_server(db_path, host, port, verbose)
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + 20):
        try:
            return make_server(db_path, host, candidate, verbose)
        except OSError as exc:
            if not _port_in_use(exc):
                raise
    return make_server(db_path, host, 0, verbose)


def serve(
    db_path: str | os.PathLike[str],
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
    verbose: bool = False,
) -> None:
    server = make_server_auto(db_path, host, port, verbose)
    print(f"sqgraph viewer: {server.service.path}\n  {server.url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (server.url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sqgraph.viewer", description="Browse a sqgraph database in the browser."
    )
    ap.add_argument("db", help="path to a sqgraph database file")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help=f"port (default: first free from {DEFAULT_PORT})")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    try:
        serve(args.db, args.host, args.port, not args.no_browser, args.verbose)
    except (SqgraphError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    return 0
