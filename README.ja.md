# sqgraph

[![CI](https://github.com/Fukuro99/sqgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Fukuro99/sqgraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)

[English](README.md)

SQLite と [sqlite-vec](https://github.com/asg017/sqlite-vec) の上に作った組み込み型のプロパティグラフ DB です。openCypher のサブセットで問い合わせます。GraphRAG のストレージ / クエリ層と、ローカルの AI エージェントの長期記憶として使うことを想定しています。

- **1 ファイル、サーバー不要。** ノード・リレーションシップ・ラベル・JSON プロパティを 1 つの SQLite ファイル (または `:memory:`) に保存します。
- **openCypher を SQL にコンパイル。** 読み取りは JOIN・再帰 CTE・`json_extract` / `json_each` などの純粋な SQL として SQLite が実行します。
- **Neo4j 互換の検索。** ノードにもリレーションシップにも、ベクトル索引 (sqlite-vec) と全文索引 (FTS5, BM25) を張れます。DDL とプロシージャは Neo4j 5 と同じ (`CREATE VECTOR INDEX`、`db.index.vector.queryNodes`、`db.index.fulltext.queryRelationships` など) なので、`Neo4jGraph` / `Neo4jVector` 向けのコードを最小限の変更で使えます。
- **エージェントの記憶。** `sqgraph.memory` は時系列ナレッジグラフです。ファクトは有効期間を持ち、古くなったファクトは削除せず履歴として残し、ベクトル・全文・グラフのホップを組み合わせて検索します。MCP サーバー、CLI、フック、Skill を同梱し、Claude Code・Codex CLI・OpenCode・Cursor・Claude Desktop に登録できます。
- **軽量。** ランタイム依存は `sqlite-vec` と `lark` だけです。

## インストール

PyPI にはまだ公開していないので、GitHub から入れます:

```bash
pip install "sqgraph @ git+https://github.com/Fukuro99/sqgraph"
```

| extra | 追加される依存 | 用途 |
|---|---|---|
| `memory` | `openai` | OpenAI または OpenAI 互換の埋め込みサーバーを使う `sqgraph.memory` |
| `mcp` | `mcp`, `openai` | MCP サーバーと `sqgraph-memory` CLI |
| `dev` | `pytest` | テストの実行 |

要件: Python 3.10 以上で、`sqlite3` モジュールが SQLite 3.38 以上かつ拡張ロードに対応していること。uv が管理する Python、python.org の Windows インストーラー、多くの Linux ディストリビューションは対応しています。macOS の一部の Python は非対応です (`sqgraph-memory doctor` で確認できます)。

## クイックスタート

```python
from sqgraph import GraphDB

with GraphDB("graph.db") as db:
    alice = db.create_node("Person", {"name": "Alice", "age": 30})
    bob = db.create_node("Person", {"name": "Bob"})
    db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})

    db.query(
        "MATCH (a:Person {name: $name})-[r:KNOWS]->(b) RETURN b.name AS friend, r.since AS since",
        {"name": "Alice"},
    )
    # [{'friend': 'Bob', 'since': 2020}]

    print(db.explain("MATCH (a)-[:KNOWS*1..3]->(b) RETURN count(*)"))  # 生成された SQL
```

ベクトル検索・全文検索は Neo4j と同じ構文です:

```python
db.query("""
  CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
""")
db.create_node("Entity", {"name": "OpenAI", "embedding": vector})  # 索引対象のプロパティは sqlite-vec に入る
db.query(
    "CALL db.index.vector.queryNodes('entity_emb', 5, $q) YIELD node, score RETURN node.name, score",
    {"q": query_vector},
)
```

同梱のビューワーでグラフを眺められます:

```bash
python -m sqgraph.viewer graph.db
```

## ローカルエージェントの長期記憶

同じマシンのエージェント全員で、1 つのローカルな記憶を共有します。抽出は呼び出したエージェント自身が行うので LLM の API キーは要りません。埋め込みは LM Studio や Ollama などのローカルサーバーで動かせます。

```bash
uv tool install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"
sqgraph-memory init      # ~/.sqgraph-memory を作る
sqgraph-memory setup     # 見つかったエージェントに MCP サーバー・Skill・フックを登録
sqgraph-memory doctor    # 環境の診断
```

bot からは Python API を直接使えます:

```python
from sqgraph.memory import Memory, OpenAILLM, OpenAIEmbedder

mem = Memory("memory.db", OpenAILLM("gpt-5.4-mini"), OpenAIEmbedder("text-embedding-3-small"),
             default_group_id="my-team", tz="Asia/Tokyo")
mem.add_episode("来週から認証基盤の担当は Bob に移ります", source="discord", author="alice")
for hit in mem.search("認証基盤は誰が担当？"):
    print(hit.status, hit.fact, hit.valid_at, hit.invalid_at)
```

## ドキュメント

- [グラフ DB](docs/ja/graph-db.md): Python API、ベクトル検索・全文検索、索引、Cypher のサポート範囲、仕組み、ビューワー
- [エージェント長期記憶](docs/ja/memory.md): データモデル、Python API、エージェント抽出、アクセス制御、MCP サーバー、記憶の整理、エクスポート
- [個人メモリとして使う](docs/ja/personal-memory.md): セットアップ、フック、CLI リファレンス、埋め込みサーバー
- [クラウドへのデプロイ](docs/ja/deployment.md): AWS / Cloudflare での構成、Litestream によるバックアップ
- [サンプル](examples/): OpenAI を使った GraphRAG、チャットログのベクトル検索

## 開発

```bash
pip install -e ".[dev,mcp]"
python -m pytest -q
```

## ライセンス

[MIT](LICENSE)
