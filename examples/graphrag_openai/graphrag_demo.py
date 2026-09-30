"""GraphRAG demo on sqgraph + OpenAI.

    python graphrag_demo.py index [--db PATH] [--force] [--messages-json chat.json]
    python graphrag_demo.py local  "question" [--db PATH]
    python graphrag_demo.py global "question" [--db PATH]
    python graphrag_demo.py basic  "question" [--db PATH]
    python graphrag_demo.py demo   [--db PATH] [--messages-json chat.json] [-q "question" ...]

--messages-json takes a JSON array of {"t": "YYYY-MM-DD HH:MM", "u": user, "c": text} chat messages
instead of the bundled fictional corpus. Every command ends with an OpenAI cost summary.

Graph schema:
    (Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)-[:RELATED_TO {description, weight}]->(Entity)
    (Entity)-[:IN_COMMUNITY]->(Community {title, summary})
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
try:
    from sqgraph import GraphDB
except ImportError:  # not pip-installed: use the in-repo sources
    sys.path.insert(0, str(REPO / "src"))
    from sqgraph import GraphDB
DEFAULT_DB = HERE / "graphrag.db"
CHAT_MODEL = "gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-small"
DIMS = 1536
WORKERS = 6
CORPUS_CHUNK_SIZE = 250  # characters per chunk for the bundled prose corpus
CHAT_CHUNK_SIZE = 600  # characters per chunk for chat logs (whole message lines are never split)

DEMO_QUESTIONS = [
    # multi-hop (several documents chained together)
    "2021年の赤潮事件の原因調査を主導した機関の所長が、共同研究をしている企業の創業者は誰ですか?",
    "赤潮事件のとき流出の疑いをかけられた企業のCTOが発見した藻類を使った製品を、非常用照明に採用した鉄道会社の社長は、誰の弟ですか?",
    # aggregation over many chunks
    "月凪祭に関わっている人物・組織をすべて挙げ、それぞれの役割を説明してください。",
    # global sensemaking
    "この街では、産業・科学・伝統文化はどのように結びついていますか? 全体の主要なテーマを教えてください。",
    "赤潮事件とその後の出来事から、関係者間の信頼回復についてどんな教訓が読み取れますか?",
]


# --------------------------------------------------------------------------- cost tracking
# USD per 1M tokens: (input, cached input, output)
PRICES = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
}
JPY_PER_USD = 150  # approximate


class Usage:
    """Thread-safe token / cost accumulator keyed by (phase, model)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.phase = "-"
        self.rows: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: dict(calls=0, inp=0, cached=0, out=0))

    def add(self, model: str, inp: int, cached: int, out: int) -> None:
        with self.lock:
            r = self.rows[(self.phase, model)]
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
        return sum(self.usd(m, r) or 0.0 for (_, m), r in self.rows.items())

    def summary(self, title: str) -> None:
        print(f"\n=== OpenAI cost: {title} ===")
        print(f"  {'phase':<26}{'model':<24}{'calls':>6}{'in tok':>9}{'(cached)':>10}{'out tok':>9}{'USD':>10}")
        for (phase, model), r in sorted(self.rows.items()):
            u = self.usd(model, r)
            cost = f"{u:>10.5f}" if u is not None else f"{'?':>10}"
            print(f"  {phase:<26}{model:<24}{r['calls']:>6}{r['inp']:>9}{r['cached']:>10}{r['out']:>9}{cost}")
        tot = self.total_usd()
        calls = sum(r["calls"] for r in self.rows.values())
        print(f"  TOTAL: {calls} calls, ${tot:.5f} (approx. {tot * JPY_PER_USD:.2f} JPY at {JPY_PER_USD} JPY/USD)")

    def reset(self) -> None:
        with self.lock:
            self.rows.clear()


usage = Usage()


# --------------------------------------------------------------------------- OpenAI helpers
def load_env() -> None:
    """Load OPENAI_API_KEY from <repo>/.env if it is not already in the environment (no extra deps)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                os.environ.setdefault(m.group(1), m.group(2).strip().strip("'\""))


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        load_env()
        _client = OpenAI(max_retries=4)
    return _client


def chat(system: str, user: str, schema: dict | None = None):
    """One chat completion. With ``schema`` -> strict JSON structured output returned as a dict."""
    kwargs = {}
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }
    r = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    u = r.usage
    if u:
        details = getattr(u, "prompt_tokens_details", None)
        usage.add(CHAT_MODEL, u.prompt_tokens, getattr(details, "cached_tokens", 0) or 0, u.completion_tokens)
    text = r.choices[0].message.content or ""
    return json.loads(text) if schema else text.strip()


def obj_schema(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def embed(texts: list[str], batch: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        r = client().embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
        if r.usage:
            usage.add(EMBED_MODEL, r.usage.total_tokens, 0, 0)
        out += [d.embedding for d in r.data]
    return out


def pmap(fn, items):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items))


# --------------------------------------------------------------------------- chunking / extraction
def chunk_text(text: str, size: int, overlap_sentences: int = 1) -> list[str]:
    """Pack sentences (split at 。) into chunks of ~size chars; the last sentence(s) are repeated as overlap."""
    sents = [s for s in re.split(r"(?<=[。\n])", text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(map(len, cur)) + len(s) > size:
            chunks.append("".join(cur).strip())
            cur = cur[-overlap_sentences:] if overlap_sentences else []
        cur.append(s)
    if cur and (not chunks or "".join(cur).strip() != chunks[-1]):
        chunks.append("".join(cur).strip())
    return chunks


# --------------------------------------------------------------------------- chat-log corpus
# Display names often carry temporary status suffixes: "alice(9/14-9/16お休み)", "name [不在]9/9～11", "name_1718休".
_ABSENCE_RES = [
    re.compile(r"\s*[\(（][^)）]*(?:お休み|休み|不在|休暇|OOO)[^)）]*[\)）]"),
    re.compile(r"\s*[\[【][^\]】]*(?:不在|お休み|休み|休暇)[^\]】]*[\]】]\S*"),
    re.compile(r"[_\-\s]?\d{3,4}休"),
]


def clean_username(name: str) -> str:
    """Strip vacation / absence markers so one person keeps one stable name."""
    prev = None
    while prev != name:
        prev = name
        for rx in _ABSENCE_RES:
            name = rx.sub("", name).strip()
    return name or "unknown"


def clean_message(text: str) -> str:
    """One message -> one line (newlines folded), absence markers in @mentions removed."""
    text = re.sub(r"(@[^\s(（]+)[\(（][^)）]*(?:お休み|不在|休み)[^)）]*[\)）]", r"\1", text)
    return re.sub(r"\s*\n\s*", " / ", text.strip())


def build_chat_docs(msgs: list[dict], gap_hours: float = 6, max_msgs: int = 12, max_chars: int = 1500) -> list[dict]:
    """Group time-sorted chat messages into conversation documents (time gap / message count / size limits)."""
    msgs = sorted(msgs, key=lambda m: m["t"])
    groups: list[list[dict]] = []
    cur: list[dict] = []
    size, last = 0, None
    for m in msgs:
        line = f"[{m['t']}] {clean_username(m['u'])}: {clean_message(m['c'])}"
        ts = datetime.strptime(m["t"], "%Y-%m-%d %H:%M")
        if cur and (
            (ts - last).total_seconds() > gap_hours * 3600 or len(cur) >= max_msgs or size + len(line) > max_chars
        ):
            groups.append(cur)
            cur, size = [], 0
        cur.append({"t": m["t"], "user": clean_username(m["u"]), "line": line})
        size += len(line)
        last = ts
    if cur:
        groups.append(cur)
    return [
        {
            "id": f"conv-{i:03d}",
            "title": f"{g[0]['t']} - {g[-1]['t']} ({len(g)} messages)",
            "lines": [x["line"] for x in g],
            "speakers": sorted({x["user"] for x in g}),
        }
        for i, g in enumerate(groups)
    ]


def chunk_lines(lines: list[str], size: int, overlap_lines: int = 1) -> list[str]:
    """Pack whole lines into chunks of ~size chars (a line is never split); last line(s) are repeated as overlap."""
    chunks: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if cur and sum(len(x) + 1 for x in cur) + len(ln) > size:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_lines:] if overlap_lines else []
            if cur and sum(len(x) + 1 for x in cur) + len(ln) > size:
                cur = []  # overlap would make the chunk too big
        cur.append(ln)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def norm(name: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).lower()


EXTRACT_SCHEMA = obj_schema(
    entities={
        "type": "array",
        "items": obj_schema(name={"type": "string"}, type={"type": "string"}, description={"type": "string"}),
    },
    relationships={
        "type": "array",
        "items": obj_schema(
            source={"type": "string"},
            target={"type": "string"},
            description={"type": "string"},
            weight={"type": "integer"},
        ),
    },
)
EXTRACT_SYSTEM = (
    "あなたは文章からナレッジグラフを抽出するアシスタントです。与えられたテキストから、"
    "エンティティ(人物・組織・場所・出来事・製品・概念など)と、エンティティ間の関係を抽出してください。\n"
    "- name: 正式名称で統一する(敬称・役職は含めない。例: 「鶴岡澄江」)。同一対象は常に同じ表記にする。"
    "「同研究所」「市長」「祭り」のような一般的な呼び方は文脈から正式名称に解決し、"
    "解決できない汎用語や、人数・日付だけの語はエンティティにしない。\n"
    "- type: PERSON / ORGANIZATION / LOCATION / EVENT / PRODUCT / CONCEPT のいずれか。\n"
    "- description: そのエンティティについてテキストから分かる事実を1〜2文で。\n"
    "- relationships: source と target は抽出したエンティティの name と完全一致させる。"
    "description に関係の内容(役職・提携・設立・親族関係など)を具体的に書き、"
    "weight は関係の強さを 1〜10 の整数で示す。"
)


EXTRACT_SYSTEM_CHAT = (
    "あなたはチャットログ(社内ワークスペースの会話)からナレッジグラフを抽出するアシスタントです。"
    "各行は `[日時] 発言者: 発言内容` の形式です。与えられたログから、エンティティと、エンティティ間の関係を抽出してください。\n"
    "- name: 同一対象は常に同じ表記にする。人物は「発言者」欄に示された表示名をそのまま使う"
    "(@メンション、「〜さん」などの呼び名が特定の発言者を指すと判断できる場合はその表示名に統合。"
    "1人に複数の名前候補があっても、発言者一覧にある名前を優先する)。"
    "サービス・リポジトリ・システム・チケットはログ中の識別子(例: api-gateway, PROJ-123)を name にする。"
    "解決できない汎用語(「それ」「あの件」など)や、日付・人数だけの語、URL そのものはエンティティにしない。\n"
    "- type: PERSON / ORGANIZATION(組織・チーム) / SERVICE(サービス・リポジトリ・システム・ツール) / "
    "TICKET(チケット・課題) / EVENT(会議・リリース・障害・作業) / TECHNOLOGY(言語・フレームワーク・ミドルウェア) / "
    "DOCUMENT(ドキュメント・議事録) / CONCEPT(方針・課題・トピック) のいずれか。\n"
    "- description: そのエンティティについてログから分かる事実(いつ・誰が・何を・なぜ)を1〜2文で。日付は YYYY-MM-DD で書く。\n"
    "- relationships: source と target は抽出したエンティティの name と完全一致させる。"
    "「誰が何を担当・提案・依頼・質問・回答・レビューした」「A は B に依存する」「A は B の所有者である」などを "
    "description に具体的に(日付や期限も含めて)書き、weight は関係の強さを 1〜10 の整数で示す。"
)


def extract(chunk: dict) -> dict:
    if chunk.get("speakers"):
        user = "## 発言者(正規化済みの表示名)\n" + ", ".join(chunk["speakers"]) + "\n\n## ログ\n" + chunk["text"]
        return chat(EXTRACT_SYSTEM_CHAT, user, EXTRACT_SCHEMA)
    return chat(EXTRACT_SYSTEM, chunk["text"], EXTRACT_SCHEMA)


def merge_extractions(chunks: list[dict], results: list[dict]):
    """Merge duplicate entities by normalized name; merge relationships per unordered entity pair."""
    ents: dict[str, dict] = {}
    rels: dict[tuple, dict] = {}
    mentions: set[tuple[str, str]] = set()

    def ent(name: str, etype: str | None = None, desc: str = "") -> str:
        k = norm(name)
        e = ents.setdefault(k, {"key": k, "name": name.strip(), "types": Counter(), "descs": []})
        if etype:  # only entity records vote for a type (relationship endpoints carry none)
            e["types"][etype.upper()] += 1
        if desc and desc not in e["descs"]:
            e["descs"].append(desc)
        return k

    for ch, res in zip(chunks, results):
        for e in res["entities"]:
            if e["name"].strip():
                mentions.add((ch["id"], ent(e["name"], e["type"], e["description"])))
        for r in res["relationships"]:
            if not r["source"].strip() or not r["target"].strip():
                continue
            a, b = ent(r["source"]), ent(r["target"])
            if a == b:
                continue
            mentions |= {(ch["id"], a), (ch["id"], b)}
            x = rels.setdefault(tuple(sorted((a, b))), {"src": a, "dst": b, "descs": [], "weight": 0})
            x["weight"] += max(1, min(10, r["weight"]))
            if r["description"] and r["description"] not in x["descs"]:
                x["descs"].append(r["description"])
    return ents, rels, mentions


def summarize_descriptions(name: str, descs: list[str]) -> str:
    return chat(
        "同一エンティティについての複数の記述を、矛盾なく1つの説明(3文以内、日本語)に統合してください。説明文のみを出力。",
        f"エンティティ: {name}\n" + "\n".join(f"- {d}" for d in descs),
    )


# --------------------------------------------------------------------------- community detection
def louvain_communities(nodes: list[str], edges: dict[tuple[str, str], float], seed: int = 42) -> list[list[str]]:
    """Louvain (local-moving + aggregation), pure Python, deterministic for a given seed."""
    rng = random.Random(seed)
    adj: dict[str, dict[str, float]] = {n: defaultdict(float) for n in nodes}
    for (a, b), w in edges.items():
        adj[a][b] += w
        adj[b][a] += w
    members = {n: [n] for n in nodes}  # super-node -> original nodes
    while True:
        total = sum(sum(nb.values()) for nb in adj.values()) / 2 or 1.0
        deg = {n: sum(nb.values()) for n, nb in adj.items()}
        comm = {n: n for n in adj}
        tot = dict(deg)
        moved_any = True
        while moved_any:
            moved_any = False
            order = sorted(adj)
            rng.shuffle(order)
            for n in order:
                links: dict[str, float] = defaultdict(float)
                for m, w in adj[n].items():
                    if m != n:
                        links[comm[m]] += w
                old = comm[n]
                tot[old] -= deg[n]
                best, best_gain = old, links.get(old, 0.0) - tot[old] * deg[n] / (2 * total)
                for c in sorted(links):
                    gain = links[c] - tot[c] * deg[n] / (2 * total)
                    if gain > best_gain + 1e-12:
                        best, best_gain = c, gain
                tot[best] += deg[n]
                comm[n] = best
                if best != old:
                    moved_any = True
        groups = sorted(set(comm.values()))
        if len(groups) == len(adj):
            break
        new_members = {g: [x for n in adj if comm[n] == g for x in members[n]] for g in groups}
        new_adj: dict[str, dict[str, float]] = {g: defaultdict(float) for g in groups}
        for n, nb in adj.items():
            for m, w in nb.items():
                new_adj[comm[n]][comm[m]] += w
        adj, members = new_adj, new_members
    return sorted((sorted(v) for v in members.values()), key=lambda c: (-len(c), c[0]))


COMMUNITY_SCHEMA = obj_schema(title={"type": "string"}, summary={"type": "string"})


def community_report(comm: list[str], ents: dict, rels: dict) -> dict:
    s = set(comm)
    lines = [f"- {ents[k]['name']} ({ents[k]['type']}): {ents[k]['description']}" for k in comm]
    rlines = [
        f"- {ents[r['src']]['name']} -> {ents[r['dst']]['name']}: {r['description']}"
        for r in rels.values()
        if r["src"] in s and r["dst"] in s
    ]
    return chat(
        "あなたはナレッジグラフのコミュニティ(関連の強いエンティティ群)のレポートを書くアナリストです。"
        "与えられたエンティティと関係から、このコミュニティを表す短いタイトルと、"
        "主要な事実・関係・出来事を具体的な固有名詞込みで整理した要約(300字程度、日本語)を JSON で出力してください。",
        "## エンティティ\n" + "\n".join(lines) + "\n\n## 関係\n" + "\n".join(rlines),
        COMMUNITY_SCHEMA,
    )


# --------------------------------------------------------------------------- index
def print_stats(db: GraphDB) -> None:
    print("\n=== Graph stats ===")
    for r in db.query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC"):
        print(f"  node :{r['label']:<10} {r['c']}")
    for r in db.query("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"):
        print(f"  rel  :{r['t']:<10} {r['c']}")


def cmd_index(args) -> None:
    path = Path(args.db)
    if path.exists():
        if not args.force:
            sys.exit(f"{path} already exists (use --force to rebuild)")
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)

    usage.reset()

    # 1. chunk
    docs, chunks = [], []
    if args.messages_json:  # chat-log corpus: conversation documents, chunked on whole message lines
        msgs = json.loads(Path(args.messages_json).read_text(encoding="utf-8"))
        size = args.chunk_size or CHAT_CHUNK_SIZE
        for d in build_chat_docs(msgs):
            docs.append({"id": d["id"], "title": d["title"]})
            for i, t in enumerate(chunk_lines(d["lines"], size)):
                chunks.append({"id": f"{d['id']}#{i}", "doc": d["id"], "idx": i, "text": t, "speakers": d["speakers"]})
    else:
        size = args.chunk_size or CORPUS_CHUNK_SIZE
        for f in sorted((HERE / "corpus").glob("*.txt")):
            title, _, body = f.read_text(encoding="utf-8").partition("\n")
            docs.append({"id": f.stem, "title": title.strip()})
            for i, t in enumerate(chunk_text(body.strip(), size)):
                chunks.append({"id": f"{f.stem}#{i}", "doc": f.stem, "idx": i, "text": t})
    print(f"[1/6] {len(docs)} documents -> {len(chunks)} chunks")

    # 2. extract entities / relationships (one LLM call per chunk)
    usage.phase = "index/extraction"
    results = pmap(extract, chunks)
    ents, rels, mentions = merge_extractions(chunks, results)
    print(f"[2/6] extracted: {len(ents)} entities (after merge), {len(rels)} relationships, {len(mentions)} mentions")

    # 3. finalize entity type / description (LLM-summarize merged descriptions)
    usage.phase = "index/entity-merge"
    multi = [e for e in ents.values() if len(e["descs"]) > 1]
    for e, d in zip(multi, pmap(lambda e: summarize_descriptions(e["name"], e["descs"]), multi)):
        e["descs"] = [d]
    for e in ents.values():
        e["type"] = e["types"].most_common(1)[0][0] if e["types"] else "CONCEPT"
        e["description"] = " ".join(e["descs"])
    for r in rels.values():
        r["description"] = " / ".join(r["descs"])
    print(f"[3/6] summarized descriptions of {len(multi)} multiply-mentioned entities")

    # 4. embeddings
    usage.phase = "index/embeddings"
    keys = list(ents)
    ent_vecs = embed([f"{ents[k]['name']}: {ents[k]['description']}" for k in keys])
    chunk_vecs = embed([c["text"] for c in chunks])
    print(f"[4/6] embedded {len(keys)} entities + {len(chunks)} chunks ({DIMS} dims)")

    # 5. store in sqgraph
    db = GraphDB(path)
    for label, prop in (("Document", "id"), ("Chunk", "id"), ("Entity", "key"), ("Community", "id")):
        db.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
    for name, label, prop in (("entity_emb", "Entity", "embedding"), ("chunk_emb", "Chunk", "embedding")):
        db.query(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop}) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {DIMS}, `vector.similarity_function`: 'cosine'}}}}"
        )
    db.query(
        "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text] "
        "OPTIONS {indexConfig: {`fulltext.tokenizer`: 'trigram'}}"
    )
    db.query("UNWIND $rows AS row CREATE (:Document {id: row.id, title: row.title})", {"rows": docs})
    db.query(
        "UNWIND $rows AS row MATCH (d:Document {id: row.doc}) "
        "CREATE (c:Chunk {id: row.id, idx: row.idx, text: row.text}) CREATE (d)-[:HAS_CHUNK]->(c)",
        {"rows": chunks},
    )
    db.query(
        "UNWIND $rows AS row MATCH (c:Chunk {id: row.id}) CALL db.create.setNodeVectorProperty(c, 'embedding', row.vec)",
        {"rows": [{"id": c["id"], "vec": v} for c, v in zip(chunks, chunk_vecs)]},
    )
    db.query(
        "UNWIND $rows AS row CREATE (:Entity {key: row.key, name: row.name, type: row.type, description: row.description})",
        {"rows": [{k: ents[key][k] for k in ("key", "name", "type", "description")} for key in keys]},
    )
    db.query(
        "UNWIND $rows AS row MATCH (e:Entity {key: row.key}) CALL db.create.setNodeVectorProperty(e, 'embedding', row.vec)",
        {"rows": [{"key": k, "vec": v} for k, v in zip(keys, ent_vecs)]},
    )
    db.query(
        "UNWIND $rows AS row MATCH (c:Chunk {id: row.c}), (e:Entity {key: row.e}) MERGE (c)-[:MENTIONS]->(e)",
        {"rows": [{"c": c, "e": e} for c, e in sorted(mentions)]},
    )
    db.query(
        "UNWIND $rows AS row MATCH (a:Entity {key: row.a}), (b:Entity {key: row.b}) "
        "CREATE (a)-[:RELATED_TO {description: row.description, weight: row.weight}]->(b)",
        {"rows": [{"a": r["src"], "b": r["dst"], "description": r["description"], "weight": r["weight"]} for r in rels.values()]},
    )
    print("[5/6] stored documents / chunks / entities / relationships with vector + fulltext indexes")

    # 6. communities (Louvain) + LLM reports
    usage.phase = "index/community-reports"
    comms = [c for c in louvain_communities(keys, {(r["src"], r["dst"]): r["weight"] for r in rels.values()}) if len(c) >= 2]
    reports = pmap(lambda c: community_report(c, ents, rels), comms)
    for i, (c, rep) in enumerate(zip(comms, reports)):
        db.query(
            "CREATE (com:Community {id: $id, title: $title, summary: $summary, size: $size}) "
            "WITH com UNWIND $members AS k MATCH (e:Entity {key: k}) CREATE (e)-[:IN_COMMUNITY]->(com)",
            {"id": f"community-{i}", "title": rep["title"], "summary": rep["summary"], "size": len(c), "members": c},
        )
        print(f"      community-{i} ({len(c)} entities): {rep['title']}")
    print(f"[6/6] {len(comms)} communities")
    db.analyze()
    print_stats(db)
    db.close()
    usage.summary("index")


# --------------------------------------------------------------------------- query modes
ANSWER_SYSTEM = (
    "あなたは提供されたコンテキストだけを根拠に、日本語で簡潔かつ正確に質問へ答えるアシスタントです。"
    "コンテキストにない事柄は推測せず「不明」と述べてください。"
    "根拠にしたエンティティ名は [エンティティ名] の形で、チャンクは (出典ID) の形で文中に引用してください。"
)


def open_db(path: str) -> GraphDB:
    if not Path(path).exists():
        sys.exit(f"{path} not found; run `index` first")
    return GraphDB(path, read_only=True)


def basic_search(db: GraphDB, question: str, k: int = 4) -> str:
    q = embed([question])[0]
    rows = db.query(
        "CALL db.index.vector.queryNodes('chunk_emb', $k, $q) YIELD node, score "
        "RETURN node.id AS id, node.text AS text, score ORDER BY score DESC",
        {"k": k, "q": q},
    )
    print("  retrieved chunks:", ", ".join(f"{r['id']}({r['score']:.3f})" for r in rows))
    ctx = "\n\n".join(f"({r['id']}) {r['text']}" for r in rows)
    return chat(ANSWER_SYSTEM, f"## コンテキスト\n{ctx}\n\n## 質問\n{question}")


def local_search(db: GraphDB, question: str, k: int = 3, max_chunks: int = 6) -> str:
    q = embed([question])[0]
    # 1. seed entities: vector top-k
    seeds = db.query(
        "CALL db.index.vector.queryNodes('entity_emb', $k, $q) YIELD node, score "
        "RETURN node.key AS key, node.name AS name, node.type AS type, node.description AS description, score "
        "ORDER BY score DESC",
        {"k": k, "q": q},
    )
    seed_keys = [s["key"] for s in seeds]
    # 2. 1..2 hop neighbourhood over RELATED_TO (distance = shortest of the paths found)
    hops = db.query(
        "MATCH p = (s:Entity)-[:RELATED_TO*1..2]-(nb:Entity) WHERE s.key IN $keys "
        "RETURN nb.key AS key, min(length(p)) AS hops",
        {"keys": seed_keys},
    )
    dist = {h["key"]: h["hops"] for h in hops} | {k_: 0 for k_ in seed_keys}
    # 3. entities + relationship descriptions inside the neighbourhood
    ents = db.query(
        "MATCH (e:Entity) WHERE e.key IN $keys RETURN e.key AS key, e.name AS name, e.type AS type, e.description AS description",
        {"keys": list(dist)},
    )
    rels = db.query(
        "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) WHERE a.key IN $keys AND b.key IN $keys "
        "RETURN a.name AS src, b.name AS dst, r.description AS description, r.weight AS weight ORDER BY r.weight DESC",
        {"keys": list(dist)},
    )
    # 4. chunks mentioning those entities (most entities first), with their document
    chunks = db.query(
        "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.key IN $keys "
        "RETURN c.id AS id, c.text AS text, d.title AS doc, count(DISTINCT e) AS hits ORDER BY hits DESC, id LIMIT $n",
        {"keys": list(dist), "n": max_chunks},
    )
    # 5. communities the entities belong to
    comms = db.query(
        "MATCH (e:Entity)-[:IN_COMMUNITY]->(com:Community) WHERE e.key IN $keys "
        "RETURN com.title AS title, com.summary AS summary, count(e) AS n ORDER BY n DESC LIMIT 3",
        {"keys": list(dist)},
    )
    print("  seed entities:", ", ".join(f"{s['name']}({s['score']:.3f})" for s in seeds))
    print(f"  context: {len(ents)} entities ({sum(1 for v in dist.values() if v == 1)} 1-hop, "
          f"{sum(1 for v in dist.values() if v == 2)} 2-hop), {len(rels)} relationships, "
          f"{len(chunks)} chunks, {len(comms)} communities")
    ent_lines = [f"- [{e['name']}] ({e['type']}, {dist[e['key']]}ホップ): {e['description']}" for e in sorted(ents, key=lambda e: dist[e["key"]])]
    ctx = (
        "## エンティティ\n" + "\n".join(ent_lines)
        + "\n\n## 関係\n" + "\n".join(f"- [{r['src']}] -> [{r['dst']}]: {r['description']}" for r in rels)
        + "\n\n## 関連コミュニティ\n" + "\n".join(f"- {c['title']}: {c['summary']}" for c in comms)
        + "\n\n## 原文チャンク\n" + "\n".join(f"({c['id']}) {c['text']}" for c in chunks)
    )
    return chat(ANSWER_SYSTEM, f"{ctx}\n\n## 質問\n{question}")


MAP_SCHEMA = obj_schema(answer={"type": "string"}, score={"type": "integer"})


def global_search(db: GraphDB, question: str, top: int = 5) -> str:
    comms = db.query("MATCH (c:Community) RETURN c.id AS id, c.title AS title, c.summary AS summary, c.size AS size ORDER BY size DESC")

    def map_one(c):
        return chat(
            "以下のコミュニティレポートだけを根拠に、質問への部分的な回答を日本語で書き、"
            "その回答が質問にどれだけ役立つかを score (0〜100) で示してください。無関係なら answer は空で score は 0。",
            f"## コミュニティレポート: {c['title']}\n{c['summary']}\n\n## 質問\n{question}",
            MAP_SCHEMA,
        )

    parts = sorted(zip(comms, pmap(map_one, comms)), key=lambda x: -x[1]["score"])
    print("  map scores:", ", ".join(f"{c['title']}={p['score']}" for c, p in parts))
    used = [(c, p) for c, p in parts if p["score"] > 0][:top]
    ctx = "\n\n".join(f"### {c['title']} (役立ち度 {p['score']})\n{p['answer']}" for c, p in used)
    return chat(
        "複数のコミュニティ分析から得られた部分回答を統合し、質問に対する包括的な回答を日本語で書いてください。"
        "部分回答に含まれない事柄は推測しないでください。根拠にしたコミュニティ名は【コミュニティ名】の形で引用してください。",
        f"## 部分回答\n{ctx}\n\n## 質問\n{question}",
    )


MODES = {"basic": basic_search, "local": local_search, "global": global_search}


def run_mode(mode: str, db: GraphDB, question: str) -> float:
    """Answer ``question`` in one mode; returns the USD spent on this call."""
    usage.phase = f"query/{mode}"
    before = usage.total_usd()
    print(f"\n--- {mode.upper()} ---")
    print(MODES[mode](db, question).strip())
    spent = usage.total_usd() - before
    print(f"  [cost: ${spent:.5f}]")
    return spent


def cmd_demo(args) -> None:
    if not Path(args.db).exists():
        cmd_index(argparse.Namespace(db=args.db, force=False, chunk_size=args.chunk_size, messages_json=args.messages_json))
    usage.reset()
    db = open_db(args.db)
    questions = args.question or DEMO_QUESTIONS
    per_q: list[dict[str, float]] = []
    for i, q in enumerate(questions, 1):
        print("\n" + "=" * 78 + f"\nQ{i}: {q}\n" + "=" * 78)
        per_q.append({mode: run_mode(mode, db, q) for mode in MODES})
    db.close()
    print("\n=== Cost per question (USD) ===")
    print(f"  {'':<4}" + "".join(f"{m:>11}" for m in MODES) + f"{'total':>11}")
    for i, c in enumerate(per_q, 1):
        print(f"  Q{i:<3}" + "".join(f"{c[m]:>11.5f}" for m in MODES) + f"{sum(c.values()):>11.5f}")
    usage.summary("demo (query phase)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "local", "global", "basic", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--db", default=str(DEFAULT_DB))
        if name in ("index", "demo"):
            p.add_argument(
                "--chunk-size", type=int, default=None,
                help=f"target chunk length in characters (default {CORPUS_CHUNK_SIZE}, or {CHAT_CHUNK_SIZE} with --messages-json)",
            )
            p.add_argument("--messages-json", help='index a chat log: JSON array of {"t": "YYYY-MM-DD HH:MM", "u": user, "c": text}')
        if name == "index":
            p.add_argument("--force", action="store_true", help="rebuild if the DB exists")
        if name == "demo":
            p.add_argument("-q", "--question", action="append", help="question to ask (repeatable; default: built-in demo questions)")
        if name in MODES:
            p.add_argument("question")
    args = ap.parse_args()
    if args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    else:
        usage.reset()
        db = open_db(args.db)
        run_mode(args.cmd, db, args.question)
        db.close()
        usage.summary(args.cmd)


if __name__ == "__main__":
    main()
