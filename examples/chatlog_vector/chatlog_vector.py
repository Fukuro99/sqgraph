#!/usr/bin/env python
"""Embedding-only (no LLM at index time) searchable graph of chat messages, built on sqgraph.

Subcommands (see README.md):
    index   --messages-json PATH --db PATH [--guild-id ID]     build / update the graph (embeddings only)
    search  "query" --db PATH [-k 10] [--author X] [--channel X] [--since D] [--until D]
    ask     "question" --db PATH [--model gpt-5.4-mini]          search, then let an LLM answer with citations
    stats   --db PATH                                           node / edge counts

Graph model::

    (:User {user_id, name}) -[:POSTED]-> (:Message {id, text, author, author_id, channel, thread, container,
                                                    posted_at, url})
    (:Message) -[:IN]-> (:Thread {id, name}) -[:IN]-> (:Channel {id, name})      (or Message -[:IN]-> Channel)
    (:Message) -[:MENTIONS]-> (:User)
    (:Message) -[:NEXT]-> (:Message)        chronological chain inside one thread / channel

Retrieval = vector (cosine) + trigram fulltext fused with RRF, optional metadata filters, and graph context
expansion (previous / next 2 messages through NEXT edges).  No company data is hard-coded here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path

from sqgraph import GraphDB
from sqgraph.memory import load_env
from sqgraph.memory.util import fts_or_query, nfkc, query_terms

REPO = Path(__file__).resolve().parents[2]
EMBED_MODEL = "text-embedding-3-small"
DIMS = 1536
VEC_INDEX = "msg_emb"
FT_INDEX = "msg_text"
EMBED_BATCH = 100
MAX_EMBED_CHARS = 6000
CONTEXT_WINDOW = 2  # previous / next messages attached to each hit
JPY_PER_USD = 150  # approximate

# USD per 1M tokens: (input, cached input, output)
PRICES = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4": (2.50, 0.25, 15.00),
    EMBED_MODEL: (0.02, 0.02, 0.0),
}


# --------------------------------------------------------------------------- cost tracking
class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, int]] = defaultdict(lambda: dict(calls=0, inp=0, cached=0, out=0))

    def add(self, model: str, inp: int, cached: int = 0, out: int = 0) -> None:
        with self.lock:
            r = self.rows[model]
            r["calls"] += 1
            r["inp"] += inp
            r["cached"] += cached
            r["out"] += out

    @staticmethod
    def usd(model: str, r: dict[str, int]) -> float | None:
        if model not in PRICES:
            return None
        pin, pcached, pout = PRICES[model]
        return ((r["inp"] - r["cached"]) * pin + r["cached"] * pcached + r["out"] * pout) / 1e6

    def total_usd(self) -> float:
        return sum(self.usd(m, r) or 0.0 for m, r in self.rows.items())

    def summary(self, title: str) -> None:
        print(f"\n=== OpenAI cost: {title} ===")
        print(f"  {'model':<26}{'calls':>6}{'in tok':>10}{'(cached)':>10}{'out tok':>9}{'USD':>11}")
        for model, r in sorted(self.rows.items()):
            u = self.usd(model, r)
            cost = f"{u:>11.6f}" if u is not None else f"{'?':>11}"
            print(f"  {model:<26}{r['calls']:>6}{r['inp']:>10}{r['cached']:>10}{r['out']:>9}{cost}")
        tot = self.total_usd()
        print(f"  TOTAL: ${tot:.6f} (approx. {tot * JPY_PER_USD:.2f} JPY at {JPY_PER_USD} JPY/USD)")


usage = Usage()
_client = None


def client():
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            load_env(REPO / ".env")
            load_env()
        from openai import OpenAI

        _client = OpenAI(max_retries=4)
    return _client


def embed(texts: list[str], batch: int = EMBED_BATCH) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t[:MAX_EMBED_CHARS] or " " for t in texts[i : i + batch]]
        r = client().embeddings.create(model=EMBED_MODEL, input=chunk)
        if r.usage:
            usage.add(EMBED_MODEL, r.usage.total_tokens)
        out += [d.embedding for d in r.data]
    return out


# --------------------------------------------------------------------------- helpers
def s(value) -> str | None:
    return None if value is None or value == "" else str(value)


def norm_msg(m: dict, guild_id: str | None) -> dict:
    """Normalise one input message dict into the fields the graph needs."""
    thread_id = s(m.get("thread_id"))
    channel_id = s(m.get("channel_id"))
    container = thread_id or channel_id or "-"
    text = (m.get("text") if m.get("text") is not None else m.get("content")) or ""
    posted = m.get("posted_at") or m.get("t") or ""
    link_id = thread_id or channel_id
    url = f"https://discord.com/channels/{guild_id}/{link_id}/{m['id']}" if guild_id and link_id else ""
    return {
        "id": str(m["id"]),
        "text": text.strip(),
        "author": m.get("username") or m.get("author") or "unknown",
        "author_id": s(m.get("user_id")) or (m.get("username") or m.get("author") or "unknown"),
        "channel": m.get("channel_name") or m.get("channel") or "",
        "channel_id": channel_id or "-",
        "thread": m.get("thread_name") or m.get("thread") or None,
        "thread_id": thread_id,
        "container": container,
        "posted_at": str(posted),
        "url": url,
        "mentions": [
            {"id": str(x["id"]), "name": x.get("name") or str(x["id"])} for x in (m.get("mentions") or []) if x.get("id")
        ],
    }


def context_string(m: dict) -> str:
    where = f"{m['channel']} / {m['thread']}" if m["thread"] else m["channel"]
    return f"[{where}] {m['author']}: {m['text']}"


def open_ro(path: str) -> GraphDB:
    if not Path(path).exists():
        sys.exit(f"{path} not found; run `index` first")
    return GraphDB(path, read_only=True)


# --------------------------------------------------------------------------- index
def ensure_schema(db: GraphDB) -> None:
    for label, prop in (
        ("Message", "id"),
        ("Message", "container"),
        ("User", "user_id"),
        ("Thread", "id"),
        ("Channel", "id"),
    ):
        db.create_property_index(label, prop)
    db.create_vector_index(VEC_INDEX, "Message", "embedding", dimensions=DIMS, similarity="cosine", if_not_exists=True)
    db.create_fulltext_index(FT_INDEX, "Message", ["text"], if_not_exists=True, tokenizer="trigram")


def id_map(db: GraphDB, label: str, key: str) -> dict[str, int]:
    rows = db.query(f"MATCH (n:{label}) RETURN n.{key} AS k, id(n) AS nid")
    return {str(r["k"]): r["nid"] for r in rows}


def relink_next(db: GraphDB, containers: set[str]) -> int:
    """Rebuild the chronological NEXT chain of the given containers."""
    n_edges = 0
    for c in sorted(containers):
        rows = db.query(
            "MATCH (m:Message {container: $c}) RETURN id(m) AS nid, m.posted_at AS t, m.id AS mid", {"c": c}
        )
        rows.sort(key=lambda r: (r["t"], int(r["mid"]) if str(r["mid"]).isdigit() else 0, str(r["mid"])))
        ids = [r["nid"] for r in rows]
        db.query(
            "MATCH (a:Message {container: $c})-[r:NEXT]->(b:Message) DELETE r", {"c": c}
        )
        pairs = [(a, b, "NEXT") for a, b in zip(ids, ids[1:])]
        if pairs:
            db.create_edges(pairs)
        n_edges += len(pairs)
    return n_edges


def cmd_index(args) -> None:
    usage.rows.clear()
    raw = json.loads(Path(args.messages_json).read_text(encoding="utf-8"))
    msgs = [norm_msg(m, args.guild_id) for m in raw]
    msgs = [m for m in msgs if m["text"]]
    msgs.sort(key=lambda m: (m["posted_at"], m["id"]))
    print(f"[1/4] loaded {len(msgs)} messages from {args.messages_json}")

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    with GraphDB(args.db) as db:
        ensure_schema(db)
        existing = set(id_map(db, "Message", "id"))
        new = [m for m in msgs if m["id"] not in existing]
        print(f"[2/4] already indexed: {len(existing)}, new: {len(new)}")
        if not new:
            print("nothing to do")
            return

        vecs = embed([context_string(m) for m in new])
        print(f"[3/4] embedded {len(new)} messages ({DIMS} dims, {EMBED_MODEL})")

        users = id_map(db, "User", "user_id")
        threads = id_map(db, "Thread", "id")
        channels = id_map(db, "Channel", "id")
        edges: list[tuple[int, int, str]] = []
        touched: set[str] = set()
        with db.transaction():
            for m, vec in zip(new, vecs):
                # channel / thread / users (MERGE by key)
                if m["channel_id"] not in channels:
                    channels[m["channel_id"]] = db.create_node("Channel", {"id": m["channel_id"], "name": m["channel"]}).id
                container_node = channels[m["channel_id"]]
                if m["thread_id"]:
                    if m["thread_id"] not in threads:
                        t = db.create_node("Thread", {"id": m["thread_id"], "name": m["thread"] or ""})
                        threads[m["thread_id"]] = t.id
                        edges.append((t.id, channels[m["channel_id"]], "IN"))
                    container_node = threads[m["thread_id"]]
                for uid, name in [(m["author_id"], m["author"])] + [(x["id"], x["name"]) for x in m["mentions"]]:
                    if uid not in users:
                        users[uid] = db.create_node("User", {"user_id": uid, "name": name}).id
                node = db.create_node(
                    "Message",
                    {
                        "id": m["id"],
                        "text": m["text"],
                        "author": m["author"],
                        "author_id": m["author_id"],
                        "channel": m["channel"],
                        "thread": m["thread"] or "",
                        "container": m["container"],
                        "posted_at": m["posted_at"],
                        "url": m["url"],
                        "embedding": vec,
                    },
                )
                edges.append((users[m["author_id"]], node.id, "POSTED"))
                edges.append((node.id, container_node, "IN"))
                for x in m["mentions"]:
                    edges.append((node.id, users[x["id"]], "MENTIONS"))
                touched.add(m["container"])
            db.create_edges(edges)
            n_next = relink_next(db, touched)
        db.analyze()
        print(f"[4/4] wrote {len(new)} messages, {len(edges) + n_next} edges (NEXT relinked in {len(touched)} containers)")
        print_stats(db)
    usage.summary("index (embeddings only, no LLM)")
    r = usage.rows[EMBED_MODEL]
    print(f"  embedding tokens: {r['inp']}")


def print_stats(db: GraphDB) -> None:
    nodes = db.query("MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS c ORDER BY c DESC")
    edges = db.query("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC")
    print("  nodes: " + ", ".join(f"{r['label']}={r['c']}" for r in nodes))
    print("  edges: " + ", ".join(f"{r['t']}={r['c']}" for r in edges))


def cmd_stats(args) -> None:
    with open_ro(args.db) as db:
        print_stats(db)
        for k, v in db.stats().items():
            print(f"  {k}: {v}")


# --------------------------------------------------------------------------- search
def _key(m: dict) -> tuple[str, str]:
    return (m.get("posted_at") or "", str(m.get("id")))


def passes(p: dict, author: str | None, channel: str | None, since: str | None, until: str | None) -> bool:
    if author and nfkc(author).casefold() not in nfkc(p.get("author") or "").casefold():
        return False
    if channel:
        hay = nfkc(f"{p.get('channel') or ''} {p.get('thread') or ''}").casefold()
        if nfkc(channel).casefold() not in hay:
            return False
    t = p.get("posted_at") or ""
    if since and t[: len(since)] < since:
        return False
    if until and t[: len(until)] > until:
        return False
    return True


def rrf(lists: list[list[int]], k: int = 60) -> dict[int, float]:
    score: dict[int, float] = defaultdict(float)
    for lst in lists:
        for rank, nid in enumerate(lst):
            score[nid] += 1.0 / (k + rank + 1)
    return score


def neighbours(db: GraphDB, mid: str, window: int) -> tuple[list[dict], list[dict]]:
    q_prev = "MATCH (a:Message)-[:NEXT*1..%d]->(m:Message {id: $id}) RETURN DISTINCT a AS n" % window
    q_next = "MATCH (m:Message {id: $id})-[:NEXT*1..%d]->(b:Message) RETURN DISTINCT b AS n" % window
    prev = sorted((r["n"].properties for r in db.query(q_prev, {"id": mid})), key=_key)
    nxt = sorted((r["n"].properties for r in db.query(q_next, {"id": mid})), key=_key)
    return prev, nxt


def retrieve(
    db: GraphDB,
    query: str,
    k: int = 10,
    author: str | None = None,
    channel: str | None = None,
    since: str | None = None,
    until: str | None = None,
    window: int = CONTEXT_WINDOW,
) -> list[dict]:
    """Hybrid retrieval. Returns [{msg, score, sources, prev, next}] best first."""
    filtered = bool(author or channel or since or until)
    pool_n = 400 if filtered else max(50, k * 5)
    qvec = embed([query])[0]
    vec_hits = db.vector_search(VEC_INDEX, qvec, k=pool_n)
    terms = query_terms(query)
    fq = fts_or_query(terms)
    ft_hits = db.fulltext_search(FT_INDEX, fq, k=pool_n) if fq else []

    nodes = {n.id: n for n, _ in vec_hits}
    nodes.update({n.id: n for n, _ in ft_hits})
    ok = lambda n: passes(n.properties, author, channel, since, until)  # noqa: E731
    vec_ids = [n.id for n, _ in vec_hits if ok(n)]
    ft_ids = [n.id for n, _ in ft_hits if ok(n)]
    fused = rrf([vec_ids, ft_ids])
    vrank = {nid: i + 1 for i, nid in enumerate(vec_ids)}
    frank = {nid: i + 1 for i, nid in enumerate(ft_ids)}
    top = sorted(fused, key=lambda i: -fused[i])[:k]
    out = []
    for nid in top:
        p = nodes[nid].properties
        prev, nxt = neighbours(db, p["id"], window)
        src = []
        if nid in vrank:
            src.append(f"vec#{vrank[nid]}")
        if nid in frank:
            src.append(f"fts#{frank[nid]}")
        out.append({"msg": p, "score": fused[nid], "sources": src, "prev": prev, "next": nxt})
    return out


def where(p: dict) -> str:
    return f"{p['channel']} / {p['thread']}" if p.get("thread") else p.get("channel", "")


def clip(text: str, n: int) -> str:
    text = re.sub(r"\s*\n\s*", " / ", (text or "").strip())
    return text if len(text) <= n else text[: n - 1] + "…"


def print_hits(hits: list[dict], text_width: int = 400) -> None:
    for i, h in enumerate(hits, 1):
        p = h["msg"]
        print(f"\n#{i}  score={h['score']:.4f} ({', '.join(h['sources'])})")
        print(f"  {p['posted_at']}  [{where(p)}]  {p['author']}")
        for q in h["prev"]:
            print(f"    | {q['posted_at'][5:16]} {q['author']}: {clip(q['text'], 160)}")
        print(f"  > {clip(p['text'], text_width)}")
        for q in h["next"]:
            print(f"    | {q['posted_at'][5:16]} {q['author']}: {clip(q['text'], 160)}")
        if p.get("url"):
            print(f"  {p['url']}")


def cmd_search(args) -> None:
    usage.rows.clear()
    with open_ro(args.db) as db:
        hits = retrieve(db, args.query, args.k, args.author, args.channel, args.since, args.until)
    print_hits(hits)
    usage.summary("search (1 query embedding)")


# --------------------------------------------------------------------------- ask
ANSWER_SYSTEM = (
    "あなたは社内チャットログ (Discord) の検索アシスタントです。与えられた「検索結果」だけを根拠に、質問へ日本語で簡潔に答えてください。\n"
    "- 各検索結果は [S番号] のブロックで、> が本命の発言、その前後は文脈の発言です。\n"
    "- 主張ごとに根拠の発言の日時 (YYYY-MM-DD HH:MM) と URL を括弧で引用してください。\n"
    "- 検索結果に無いことは推測せず「ログからは分からない」と述べてください。\n"
    "- 期間や人物が絞れる場合は、時系列や人ごとに整理してください。"
)


def build_context(hits: list[dict], width: int = 500) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        p = h["msg"]
        lines = [f"[S{i}] {where(p)}"]
        for q in h["prev"]:
            lines.append(f"  {q['posted_at']} {q['author']}: {clip(q['text'], width)} <{q.get('url', '')}>")
        lines.append(f"> {p['posted_at']} {p['author']}: {clip(p['text'], width)} <{p.get('url', '')}>")
        for q in h["next"]:
            lines.append(f"  {q['posted_at']} {q['author']}: {clip(q['text'], width)} <{q.get('url', '')}>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def cmd_ask(args) -> None:
    usage.rows.clear()
    with open_ro(args.db) as db:
        hits = retrieve(db, args.query, args.k, args.author, args.channel, args.since, args.until)
    if not hits:
        print("該当する発言が見つかりませんでした。")
        usage.summary("ask")
        return
    ctx = build_context(hits)
    r = client().chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"## 検索結果\n{ctx}\n\n## 質問\n{args.query}"},
        ],
    )
    u = r.usage
    if u:
        details = getattr(u, "prompt_tokens_details", None)
        usage.add(args.model, u.prompt_tokens, getattr(details, "cached_tokens", 0) or 0, u.completion_tokens)
    print((r.choices[0].message.content or "").strip())
    print("\n--- 参照した発言 ---")
    for i, h in enumerate(hits, 1):
        p = h["msg"]
        print(f"[S{i}] {p['posted_at']} [{where(p)}] {p['author']}  {p.get('url', '')}")
    usage.summary(f"ask ({args.model} + 1 query embedding)")


# --------------------------------------------------------------------------- main
def add_filters(p: argparse.ArgumentParser, default_k: int) -> None:
    p.add_argument("query")
    p.add_argument("--db", required=True)
    p.add_argument("-k", type=int, default=default_k, help="number of hits (default %(default)s)")
    p.add_argument("--author", help="author name (substring, case-insensitive)")
    p.add_argument("--channel", help="channel or thread name (substring)")
    p.add_argument("--since", help="YYYY-MM-DD[THH:MM] (inclusive)")
    p.add_argument("--until", help="YYYY-MM-DD[THH:MM] (inclusive)")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build / update the graph from a messages JSON (embeddings only)")
    p.add_argument("--messages-json", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--guild-id", help="used to build https://discord.com/channels/<guild>/<channel>/<message> urls")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("search", help="hybrid retrieval (vector + trigram fulltext, RRF) with graph context")
    add_filters(p, 10)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("ask", help="search then answer in Japanese with an LLM")
    add_filters(p, 8)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("stats", help="node / edge counts")
    p.add_argument("--db", required=True)
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
