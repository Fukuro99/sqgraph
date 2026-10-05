# sqgraph

SQLite と [sqlite-vec](https://github.com/asg017/sqlite-vec) の上に作った **組み込み型プロパティグラフ DB** です。GraphRAG のストレージ / クエリ層として使うことを想定しています。

- 単一ファイル (または `:memory:`) の SQLite に、ノード・リレーションシップ・ラベル・プロパティ (JSON) を保存
- **openCypher のサブセット**を SQL にコンパイルして実行 (読み取りは JOIN・再帰 CTE・`json_extract` / `json_each` / `json_group_array` などの純粋な SQL として SQLite が実行します)
- Neo4j 互換のプロシージャ `db.index.vector.queryNodes` / `db.index.fulltext.queryNodes` / `db.create.setNodeVectorProperty` と、リレーションシップ版 (Neo4j 5) の `db.index.vector.queryRelationships` / `db.index.fulltext.queryRelationships` / `db.create.setRelationshipVectorProperty`
- Neo4j 互換の DDL `CREATE VECTOR INDEX ...` / `CREATE FULLTEXT INDEX ...` / `CREATE INDEX ...` (ノードにも `()-[r:TYPE]-()` のリレーションシップにも張れます)
- ベクトル検索は sqlite-vec (`vec0`)、全文検索は FTS5 (BM25)
- ランタイム依存は `sqlite-vec` と `lark` のみ (numpy は任意: `numpy.ndarray` をそのまま渡せますが必須ではありません)

Neo4j (LangChain の `Neo4jGraph` / `Neo4jVector` など) 向けに書いたコードを、最小限の変更でローカルの SQLite に向けられることが目標です。

## インストール

```bash
pip install -e .            # 開発インストール (依存: sqlite-vec, lark)
pip install -e ".[dev]"     # テスト用 (pytest)
```

要件: Python 3.10 以上、SQLite 3.38 以上 (JSON の `->` 演算子を使用)、拡張ロード可能な `sqlite3` モジュール。

ローカルのエージェント用の個人メモリとして使うだけなら `uv tool install "sqgraph[mcp]"` から (「個人メモリとして使う」を参照)。

## クイックスタート (Python API)

```python
from sqgraph import GraphDB

with GraphDB("graph.db") as db:            # ":memory:" も可 / read_only=True で読み取り専用
    alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
    bob = db.create_node("Person", {"name": "Bob"})
    db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})

    db.neighbors(alice.id, types=["KNOWS"], direction="out", depth=2)   # -> list[Node]
    db.get_node(alice.id)["name"]                                        # 'Alice'

    # Cypher (パラメータは常に SQL パラメータとして束縛されます)
    rows = db.query(
        "MATCH (a:Person {name: $name})-[r:KNOWS]->(b) RETURN b.name AS friend, r.since AS since",
        {"name": "Alice"},
    )
    # [{'friend': 'Bob', 'since': 2020}]

    # トランザクション (ネスト可: SAVEPOINT)
    with db.transaction():
        db.create_node("Person", {"name": "Carol"})
        db.query("MATCH (p:Person {name:'Bob'}) SET p.age = 25")

    print(db.explain("MATCH (a)-[:KNOWS*1..3]->(b) RETURN count(*)"))   # 生成された SQL を表示
```

主な API:

| メソッド | 説明 |
|---|---|
| `GraphDB(path=":memory:", read_only=False, max_var_length=10)` | 開く。コンテキストマネージャ / `close()` |
| `transaction()` | 原子的なブロック (ネスト可)。公開メソッドと `query()` はそれぞれ単独でも原子的 |
| `create_node / get_node / update_node / delete_node(detach=False)` | ノード CRUD。関係が残っているノードは `detach=True` なしでは削除できません |
| `add_labels / remove_labels` | ラベル操作 |
| `create_edge / get_edge / update_edge / delete_edge` | リレーションシップ CRUD |
| `create_nodes([(labels, props), ...])` / `create_edges([(src, dst, type[, props]), ...])` | `executemany` による一括作成 (ID のリストを返す) |
| `neighbors(node_id, types=None, direction="out", depth=1, labels=None)` | 再帰 CTE による近傍探索 (循環安全・重複なし・起点は除く) |
| `create_vector_index / set_node_vector / vector_search` | ベクトル |
| `create_fulltext_index / fulltext_search` | 全文検索 |
| `create_relationship_vector_index / set_edge_vector` | リレーションシップのベクトル索引 (検索は同じ `vector_search`) |
| `create_relationship_fulltext_index` | リレーションシップの全文索引 (検索は同じ `fulltext_search`) |
| `create_property_index(label, property, name=None)` | `json_extract(props, '$.prop')` の式インデックス (`CREATE INDEX name FOR ...` の名前も有効) |
| `create_relationship_property_index(type, property, name=None)` | `edges` テーブルに対する同じ式インデックス (`MATCH ()-[r:T {p: $v}]->()` を全エッジ走査にしない) |
| `list_indexes()` | `SHOW INDEXES` 相当。`entityType` は `NODE` / `RELATIONSHIP` |
| `analyze()` | クエリプランナ統計の更新 (大量投入後に推奨) |
| `query(cypher, params=None)` / `explain(cypher, params=None)` | Cypher 実行 / 生成 SQL の確認 |

`Node` は `id`, `labels` (タプル), `properties` を持ち、`node["name"]`, `node.get("name")`, `.to_dict()` が使えます。`Relationship` は `id`, `type`, `start_node_id`, `end_node_id`, `properties`。どちらも `id` で等価判定されます。名前付きパス (`p = (a)-[*]->(b)`) は `Path` (`nodes`, `relationships`) として返ります。

## ベクトル検索

```python
db.create_vector_index("entity_emb", "Entity", "embedding", dimensions=384, similarity="cosine")
# Cypher の DDL でも同じ (Neo4j と同じ構文):
db.query("""
  CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
""")

# 埋め込みは props JSON には保存されず、sqlite-vec のテーブル (vec_<index名>) にのみ保存されます
db.set_node_vector(node_id, "embedding", vector)            # list[float] / numpy.ndarray
db.create_node("Entity", {"name": "OpenAI", "embedding": vector})   # 索引対象プロパティは自動でベクトル表へ
db.query("MATCH (n:Entity {name: 'x'}) CALL db.create.setNodeVectorProperty(n, 'embedding', $v)", {"v": vector})

# Python API: [(Node, score)] をスコア降順で返す
db.vector_search("entity_emb", query_vec, k=5)
```

スコアは Neo4j と同じ規約です: cosine は `(1 + cos_sim) / 2`、euclidean は `1 / (1 + d^2)`。

### リレーションシップのベクトル索引

「事実 (ファクト) をエッジとして保存し、埋め込みで検索する」用途向けに、Neo4j 5 と同じくリレーションシップにもベクトル索引を張れます。索引名はノード / リレーションシップをまたいで一意です (バックエンドは同じ `vec_<index名>`。リレーションシップ版の `vec0` は主キー列が `edge_id`、ノード版は `node_id`)。

```python
db.create_relationship_vector_index("fact_emb", "FACT", "embedding", dimensions=384, similarity="cosine")
# Cypher の DDL (Neo4j 5 と同じ構文。`()-[r:T]->()` の向き付きも可):
db.query("""
  CREATE VECTOR INDEX fact_emb IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
""")

# 索引対象プロパティはエッジの props JSON には入らず、ベクトル表にだけ保存されます (RETURN r.embedding は null)
db.create_edge(a.id, b.id, "FACT", {"fact": "Alice は Rails 移行を担当", "embedding": vector})
db.set_edge_vector(edge_id, "embedding", vector)
db.query("MATCH ()-[r:FACT {id: $id}]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', $v)", {"id": "f1", "v": vector})

# Python API は索引の種類に応じて [(Node, score)] / [(Relationship, score)] を返す
db.vector_search("fact_emb", query_vec, k=5)

# Cypher: yield された relationship は通常のリレーションシップ変数として後続句で使えます
db.query("""
  CALL db.index.vector.queryRelationships('fact_emb', $k, $q) YIELD relationship, score
  MATCH (s)-[relationship]->(o) WHERE score > 0.6
  RETURN s.name AS subject, relationship.fact AS fact, o.name AS object, type(relationship) AS type, score
""", {"k": 10, "q": query_vec})
```

作成・更新・削除は自動で同期されます (`create_edge` / `create_edges` / `update_edge` (`replace=True` を含む) / `delete_edge` / `delete_node(detach=True)`、Cypher の `CREATE` / `MERGE` / `SET r.p = ...` / `SET r += {...}` / `SET r = {...}` / `REMOVE r.p` / `SET r.p = null` / `DELETE r` / `DETACH DELETE n`)。索引作成時に、対象タイプのエッジの props にすでに配列で入っている埋め込みはベクトル表へ移されます (ノード版と同じ)。ノード用の索引に `queryRelationships`、リレーションシップ用の索引に `queryNodes` を使うと (逆も) 明確なエラーになります。

GraphRAG のローカルサーチ (ベクトル上位 → 1〜2 ホップ展開 → チャンク/コミュニティ収集):

```cypher
CALL db.index.vector.queryNodes('entity_emb', $k, $q) YIELD node, score
OPTIONAL MATCH (node)-[:RELATED_TO*1..2]-(nb:Entity)
WITH node, score, collect(DISTINCT nb) AS neighbors
OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(x:Entity) WHERE x = node OR x IN neighbors
WITH node, score, neighbors, collect(DISTINCT c.text) AS chunks
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(com:Community)
RETURN node.name, score, [n IN neighbors | n.name] AS neighbors, chunks, collect(DISTINCT com.summary) AS communities
ORDER BY score DESC
```

`CALL` の結果は後続の `MATCH` / `WITH` / `WHERE` と自由に組み合わせられます (`tests/test_graphrag_scenario.py` 参照)。

## 全文検索

```python
db.create_fulltext_index("chunk_text", ["Chunk"], ["text"])          # 日本語などは tokenizer="trigram" も可
db.query("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (n:Chunk|Note) ON EACH [n.text, n.title]")

db.fulltext_search("chunk_text", "graph AND (vector OR retrieval)", k=10)   # [(Node, score)]  score = -bm25 (大きいほど良い)
db.query("CALL db.index.fulltext.queryNodes('chunk_text', $q) YIELD node, score RETURN node.text, score", {"q": "graph"})
```

クエリ文字列は Lucene 風です (単語は既定で OR、`AND` / `OR` / `NOT`、`"フレーズ"`、`-除外`、`前方一致*`、`field:term`)。特殊文字を含む入力で FTS5 の構文エラーになることはありません。インデックスはノード / ラベルの作成・更新・削除に合わせて (Python API でも Cypher でも) 自動的に同期されます。

### リレーションシップの全文索引

```python
db.create_relationship_fulltext_index("fact_ft", ["FACT", "STATE"], ["fact"], tokenizer="trigram")   # types は str でもリストでも可
db.query("CREATE FULLTEXT INDEX fact_ft IF NOT EXISTS FOR ()-[r:FACT|STATE]-() ON EACH [r.fact, r.note] OPTIONS {indexConfig: {`fulltext.tokenizer`: 'trigram'}}")

db.fulltext_search("fact_ft", "認証基盤", k=10)      # [(Relationship, score)]
db.query("""
  CALL db.index.fulltext.queryRelationships('fact_ft', $q) YIELD relationship, score
  MATCH (s)-[relationship]->(o) RETURN s.name, relationship.fact, o.name, score
""", {"q": "認証基盤"})
```

`rowid` がエッジ ID の FTS5 表 (`fts_<index名>`) に保存され、エッジの作成・更新・削除 (ノードの `DETACH DELETE` で消えるエッジを含む) に合わせて同期されます。索引作成時に、対象タイプの既存エッジも索引に入ります。

### リレーションシップのプロパティ索引

```cypher
CREATE INDEX fact_id IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.id)     // RANGE も可。向き付き ()-[r:FACT]->() も可
MATCH ()-[r:FACT {id: $id}]->() RETURN r.fact                        // edges の式インデックスが使われる (WHERE r.id = $id も同様)
```

`edges` テーブルの `json_extract(props, '$.id')` に対する式インデックスです。タイプは SQL パラメータとして束縛されるため部分インデックスにはできず、索引はタイプに関係なく全エッジを対象にします (タイプはメタデータとして記録するだけ)。

## Cypher サポート表

### 対応

| 分類 | 内容 |
|---|---|
| 句 | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `WITH` (`DISTINCT` / 集約 / `ORDER BY` / `SKIP` / `LIMIT` / `WHERE`), `RETURN` (`DISTINCT`, `AS`, `*`), `ORDER BY` (複数キー, `ASC`/`DESC`), `SKIP`, `LIMIT`, `UNWIND`, `CREATE`, `MERGE` (`ON CREATE SET` / `ON MATCH SET`), `SET` (`n.p = e`, `n += map`, `n = map`, `n:Label`), `REMOVE` (`n.p`, `n:Label`), `DELETE`, `DETACH DELETE`, `CALL proc(...) YIELD a, b [WHERE ...]`, `UNION` / `UNION ALL` |
| パターン | `(n)`, `(n:L)`, `(n:A:B)`, `(n {k: v})`, `-[r:T]->`, `<-[r]-`, `-[r]-`, `-->` / `<--` / `--`, `-[:A\|B]->`, 可変長 `*`, `*2`, `*1..3`, `*..3`, `*2..`, リレーションのプロパティマップ, 複数パターン, 連鎖パターン, 束縛済み変数の再利用, 名前付きパス `p = (...)` + `length(p)` / `nodes(p)` / `relationships(p)`, 可変長リレーション変数 (`Relationship` のリスト) |
| 式 | リテラル (整数, 浮動小数, 文字列, `true/false/null`, リスト, マップ), `$param`, `n.prop`, `n[$key]`, 四則演算 / `%` / `^`, 文字列連結, 比較 (連鎖可), `AND OR NOT XOR`, `IS [NOT] NULL`, `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `=~`, `CASE` (2 形式), `list[i]`, `list[a..b]`, ラベル述語 `n:Label`, `EXISTS { MATCH ... }`, `exists(n.prop)`, リスト内包表記, `all/any/none/single`, マップ射影 `n {.*, .name, k: expr}` |
| 関数 | `id`, `elementId`, `labels`, `type`, `properties`, `keys`, `coalesce`, `toLower`, `toUpper`, `trim`/`ltrim`/`rtrim`, `size`, `length`, `toString`, `toInteger`, `toFloat`, `toBoolean`, `abs`, `round`, `ceil`, `floor`, `sqrt`, `sign`, `exp`, `log`, `log10`, `sin`, `cos`, `pi`, `e`, `rand`, `startNode`, `endNode`, `nodes`, `relationships`, `range`, `head`, `last`, `tail`, `reverse`, `split`, `replace`, `substring`, `left`, `right`, `isEmpty`, `timestamp` |
| 集約 | `count(*)`, `count(x)`, `count(DISTINCT x)`, `collect`, `collect(DISTINCT)`, `sum`, `avg`, `min`, `max` (非集約項目が暗黙のグルーピングキー) |
| プロシージャ | `db.index.vector.queryNodes`, `db.index.vector.queryRelationships` (`YIELD relationship, score`), `db.index.fulltext.queryNodes`, `db.index.fulltext.queryRelationships` (`YIELD relationship, score`), `db.create.setNodeVectorProperty`, `db.create.setRelationshipVectorProperty`, `db.create.setVectorProperty`, `db.index.vector.createNodeIndex`, `db.index.fulltext.createNodeIndex`, `db.labels`, `db.relationshipTypes`, `db.propertyKeys` |
| スキーマ | `CREATE VECTOR INDEX ... FOR (n:L) ON (n.p)` / `FOR ()-[r:T]-() ON (r.p)`, `CREATE FULLTEXT INDEX ... FOR (n:A\|B) ON EACH [n.p, ...]` / `FOR ()-[r:A\|B]-() ON EACH [r.p, ...]`, `CREATE [RANGE] INDEX FOR (n:L) ON (n.p)` / `FOR ()-[r:T]-() ON (r.p)` (単一プロパティ。リレーションシップは `-` と `->` のどちらの終端も可), `DROP INDEX [IF EXISTS]`, `SHOW INDEXES` (`entityType` = `NODE` / `RELATIONSHIP`) |
| その他 | バッククォート識別子 (`` `my label` ``), コメント (`//`, `/* */`), キーワードのプロパティ名 / ラベル名 (`n.order`, `:Order` など), 大文字小文字を区別しないキーワード |

### 一部対応 / 注意

| 内容 | 状況 |
|---|---|
| 可変長パターン `*` の上限 | 上限省略時は `max_var_length` (既定 10) で打ち切ります (`GraphDB(max_var_length=...)`)。リレーションは 1 経路内で重複しません (trail)。`EXISTS { }` 内の可変長は未対応 |
| `MERGE` | 単一パス。名前付きパス・可変長は不可。無向 `MERGE` は既存を両方向で照合し、無ければ左→右で作成 |
| `CREATE` / `MERGE` のプロパティ式 | 同じ句で作成される変数は参照できません (`CREATE (a {x:1}), (b {y: a.x})` はエラー) |
| プロパティ値 | プリミティブまたはそのリストのみ (Neo4j と同じ)。`SET n.p = {map}` はエラー |
| `UNION` | 読み取りのみ。書き込み句を含む枝は不可 |
| ラベル / 埋め込み | `labels(n)` はアルファベット順 (Neo4j は作成順)。索引対象の埋め込みプロパティは props に無いので `RETURN n.embedding` (リレーションシップ索引なら `RETURN r.embedding`) は `null` |
| 型チェック | 型をまたぐ `=` は false、`<` などの順序比較は `null` (Neo4j と同じ。真偽値は整数と別物)。プロパティ由来の動的型の値は `json_type()` で判定します |
| 整数の 0 除算 | 整数同士の `/` `%` はエラー、浮動小数の 0 除算は `null` (Neo4j は Infinity/NaN) |

### 未対応 (明確なエラーになります)

`CALL { ... }` サブクエリ, `FOREACH`, `LOAD CSV`, `shortestPath()` / `allShortestPaths()`, パターン述語 (`WHERE (a)-[:R]->(b)`。代わりに `EXISTS { ... }` を使用), パターン内包表記, `reduce()`, 日時型と関数, 空間型, `USING INDEX` ヒント, 制約 (`CREATE CONSTRAINT`), 複合プロパティインデックス, APOC 関数 / プロシージャ, `count { }` サブクエリ式, `stDev` / `percentile*` 集約。

## 仕組み

- スキーマ: `nodes(id, props JSON)`, `node_labels(label, node_id)` (`WITHOUT ROWID`), `edges(id, src, dst, type, props JSON)`, `_meta` (`schema_version` = 3), `_vector_indexes`, `_fulltext_indexes`, `_property_indexes` (いずれも `entity_type` = `node` / `relationship` を持ち、リレーションシップ索引では `label` / `labels` 列がタイプを表す)。ベクトル索引は `vec_<name>` (vec0。主キーはノード索引が `node_id`、リレーションシップ索引が `edge_id`)、全文索引は `fts_<name>` (FTS5, `rowid` = ノード ID またはエッジ ID)。古い DB (schema 2 以前) は書き込みで開くときに `entity_type` 列が追加されます (読み取り専用で開く場合は移行せず、すべてノード索引として扱います)。
- コンパイラ: 各読み取り句が 1 つの CTE (直前句の束縛テーブルに対する `SELECT`) になります。ノード / リレーション変数は ID 列、プロパティは `json_extract` / `->`、`OPTIONAL MATCH` は `LEFT JOIN`、可変長は再帰 CTE、`UNWIND` は `json_each`、`collect` は `json_group_array` です。型情報 (ノード, 真偽値, リスト, マップ ...) は静的型としてコンパイラが追跡し、結果は型に応じて `Node` / `bool` / `list` / `dict` に復元されます。
- 書き込み句 (`CREATE` / `MERGE` / `SET` / `REMOVE` / `DELETE`) は、そこまでの束縛行に対して Python で実行します (式の評価は SQL)。1 回の `query()` は 1 トランザクションで、エラー時は全体がロールバックされます。読み取り専用 DB では書き込みクエリを明示的なエラーで拒否します。
- 安全性: ユーザー値はすべて SQL パラメータ、ラベル / タイプはパラメータ、プロパティキーは `[A-Za-z_][A-Za-z0-9_]*` のときのみ JSON パスをリテラル化 (式インデックスを効かせるため。それ以外はパラメータ)、DDL に入る索引名は識別子として検証します。
- 性能: 選択性の高いアンカー (`MATCH (n:L {prop: $v})-[...]`) を SQLite に正しく最適化させるため、大量投入後や索引作成後に `db.analyze()` を呼んでください (`create_property_index` と `close()` は自動で実行します)。可変長パターンの再帰ステップは `INDEXED BY` で隣接インデックスを固定しているため、統計が無い DB でも全エッジ走査にはなりません。

## ビューワー

ブラウザでグラフを眺めるためのローカルビューワーを同梱しています (サーバー側は標準ライブラリのみ、フロントエンドは CDN の Cytoscape.js を使う単一 HTML)。DB は常に読み取り専用で開くので、Cypher コンソールから書き込むことはできません (書き込みクエリは明確なエラーになります)。

```bash
python -m sqgraph.viewer path/to/graph.db [--host 127.0.0.1] [--port N] [--no-browser]
# 例: python -m sqgraph.viewer examples/graphrag_openai/graphrag.db
```

既定では `127.0.0.1` のみで待ち受け、8765 番から空いているポートを自動で選んで (複数の DB を同時に開けます)、起動するとブラウザを開きます (`--no-browser` で抑止)。`--port` を明示した場合、そのポートが使用中ならエラーになります。フロントエンドは `cdnjs.cloudflare.com` から Cytoscape.js を読み込むため、初回表示にはネットワーク接続が必要です。

- **サイドバー**: ノード / リレーション数、ラベルごとの件数 (色チップのクリックで表示切替、`↓` でそのラベルのノードを読み込み)、リレーションタイプごとの件数 (クリックで表示切替)、プロパティキー、インデックス一覧、名前 / title / id / text の部分一致検索 (結果をクリックでグラフに追加してフォーカス)
- **グラフ**: 次数の高いノードから初期表示、ラベルごとの色 (ライト / ダークテーマ対応)、次数に応じたノードサイズ、方向つきエッジとタイプ名 (切替可)、ダブルクリックで隣接ノードを展開、`cose` による力学レイアウト、GraphRAG スキーマ (`Entity-[:IN_COMMUNITY]->Community`) がある場合の「コミュニティ色」
- **詳細パネル**: ノード / エッジをクリックするとラベル、全プロパティ (長文は折りたたみ)、次数、タイプ別のリレーション数を表示。展開 / 近傍にフォーカス / 非表示 ボタン
- **Cypher コンソール**: `Ctrl+Enter` で実行、GraphRAG 向けの例クエリ、結果をテーブルで表示しつつ、含まれる Node / Relationship / Path をグラフに置換 / 追加で描画
- **ツールバー**: 表示数の選択、再レイアウト、全体表示、クリア、エッジ名の切替、PNG 書き出し。狭い画面ではサイドバーを折りたためます

JSON API (フロントエンドが使用): `GET /api/overview`, `GET /api/graph?limit=&labels=A,B`, `GET /api/node/<id>`, `GET /api/expand/<id>?limit=&types=A,B`, `GET /api/search?q=`, `POST /api/query {cypher, params}`, `POST /api/edges {ids}`。ベクトル埋め込み (65 次元以上の数値リスト) はレスポンスに含まれません (索引対象の埋め込みは props に無いのでリレーションシップのものも出ません)。`indexes` にはリレーションシップ索引も `entityType` つきで含まれ、サイドバーでは `[:TYPE].property` と表示されます。

## エージェント長期記憶 (sqgraph.memory)

社内エージェント (Discord bot、MCP 経由の Claude など) 向けの **時系列ナレッジグラフ型の長期記憶**です ([Graphiti / Zep](https://github.com/getzep/graphiti) に着想)。`sqgraph` 本体は LLM フリーのまま、`sqgraph.memory` パッケージが差し替え可能なプロバイダ (既定: OpenAI) を使って、メッセージからエンティティ・ファクトを抽出し、矛盾を検出して古いファクトを **失効 (削除しない)** させ、`as_of` 付きのハイブリッド検索を提供します。

```bash
pip install -e ".[memory]"      # openai
pip install -e ".[mcp]"         # MCP サーバー (mcp + openai)
```

### データモデル

すべてのノードが `group_id` (ギルド / プロジェクトなどの名前空間) を持ちます。時刻は UTC の ISO 8601 文字列 (`2026-07-01T09:52:00Z`、文字列比較で大小比較できる) で保存し、検索結果は `tz` (既定 Asia/Tokyo) のオフセット付きで返します。

```
(:Episode {id, group_id, source, channel, author, content, occurred_at, ingested_at, source_ref})   生の発言 (出典)
(:Entity  {id, group_id, name, type, summary, aliases, created_at, updated_at, ...})               ベクトル + trigram 全文索引
(subject:Entity)-[:FACT {id, group_id, fact, text_norm, relation, subject_id, object_id, valid_at, invalid_at,
                        created_at, expired_at, channels, episode_ids, confidence, modality,
                        superseded_by, supersedes}]->(object:Entity)                                  ファクト = FACT リレーションシップ (Graphiti と同じ)。ベクトル + trigram 全文索引 + id の property 索引
(Episode)-[:MENTIONS]->(Entity)
```

ファクトは主語エンティティから目的語エンティティへの `FACT` リレーションシップ 1 本で、動詞は `relation` プロパティに入ります。目的語の無い属性ファクト (状態・不在など) は主語自身への自己ループ `(s)-[:FACT]->(s)` で、`object_id` を持ちません (`FactHit.object` は `None`)。`subject_id` / `object_id` は非正規化したエンティティ ID です。ファクトの ID 検索は `MATCH ()-[f:FACT {id: $id}]->()` (property 索引が効きます)。

旧形式 (ファクトを `:Fact` ノードとして持つ DB) からの移行は提供しません。旧 DB を開くとエラーになるので、作り直してください。

- **バイテンポラル**: `valid_at` / `invalid_at` は世界で真だった期間 (メッセージ時刻、または「来週から」「9/14-9/16 お休み」などの本文から算出)。`created_at` / `expired_at` はシステム時刻。矛盾するファクトが来ても **削除せず**、古い方に `invalid_at` (= 新ファクトの `valid_at`) と `expired_at` を入れ、`superseded_by` (新ファクトの ID) で結びます (新しい方の `supersedes` に古い方の ID のリストが入ります)。順序が逆に取り込まれた場合 (古いメッセージが後から来る場合) は、新しい (古い時刻の) ファクトの `invalid_at` が既存の新しい方の `valid_at` になり、その既存ファクトが `superseded_by` になります。
- **ファクトの状態** (`as_of` 時点): `valid` (`valid_at <= as_of` かつ `invalid_at` が null または `> as_of`) / `ended` (失効済み。`include_invalid=True` で取得) / `scheduled` (まだ始まっていない予定。`include_future=True`)。
- **Discord の表示名の正規化**: `taro/Yamada_1718休`、`hanako3(9/14-9/16お休み)`、`jiro / Sato Jiro [不在]9/9～11` のような一時的な不在サフィックスを外して 1 人の Person に統合し、元の表記と `/` 区切りの各部分 (`taro`, `Yamada`) をエイリアスとして保持します。

### Python からの使い方 (bot 向け)

```python
from sqgraph.memory import Memory, OpenAILLM, OpenAIEmbedder, UsageTracker, load_env

load_env(".env")                                   # OPENAI_API_KEY (値は表示・ログしません)
tracker = UsageTracker()                           # 使用量・コストの集計
mem = Memory("memory.db",
             OpenAILLM("gpt-5.4-mini", tracker=tracker),                  # 既定
             OpenAIEmbedder("text-embedding-3-small", tracker=tracker),   # 1536 次元
             default_group_id="my-guild", tz="Asia/Tokyo")

# 取り込み (1 メッセージ = 1 エピソード。同じ (source, source_ref) の再投入は無視されるので再送しても安全)
r = mem.add_episode("来週から認証基盤の担当は Bob に移ります", group_id="my-guild", source="discord",
                    channel="backend", author="alice(9/14-9/16お休み)",
                    occurred_at="2026-09-29 16:50", source_ref="discord:123456789")
# r.entities / r.facts_created / r.facts_invalidated / r.llm_calls / r.usd

mem.add_episodes_bulk(rows, group_id="my-guild", channel="backend")   # occurred_at の古い順に処理

# 検索: ファクトのベクトル + trigram 全文 + エンティティ一致→1-2 ホップ展開 を RRF で融合
hits = mem.search("認証基盤は誰が担当？", group_id="my-guild",
                  channels=["backend"],              # 閲覧を許可するチャンネル (None = 制限なし)
                  as_of="2026-07-20",                # 既定は現在
                  center_entity="bob",                # グラフ距離で並べ替え (任意)
                  include_invalid=True)              # 失効済みも (履歴)
for h in hits:                                       # FactHit: fact, valid_at, invalid_at, status, episodes[出典] ...
    print(h.status, h.fact, h.valid_at, h.invalid_at, [(e.author, e.occurred_at, e.source_ref) for e in h.episodes])

mem.search_entities("bob", channels=["backend"])            # [{id, name, type, aliases, summary, ...}]
mem.get_entity("bob", channels=["backend"])                 # プロフィール + ファクトのタイムライン (失効分含む)
mem.get_episodes(entity="Rails", channels=["backend"], limit=10)
ans = mem.answer("Rails の対応はどこまで進んだ？", group_id="my-guild", channels=["backend"])
print(ans.text)                                                # 日本語で、出典 [F1] と有効期間つきの回答
print(tracker.summary("run"))
```

取り込み 1 件あたりの LLM 呼び出しは通常 2〜4 回です: ① エンティティ抽出 (直近の同チャンネル発言を文脈として渡し「それ」等を解決。覚えるべき事実が無い発言はここで打ち切り) → ② 既存エンティティとの同一判定 (完全一致・エイリアス一致は LLM 不要。曖昧な名前だけ、ベクトル / 全文 / 完全一致の候補をまとめて 1 回で判定) → ③ ファクト抽出 + エンティティ要約更新 → ④ 重複 / 矛盾判定 (関連する既存の有効ファクトがある時だけ)。埋め込みは 1 件につき 1〜2 回のバッチ呼び出しで、DB への書き込みは 1 トランザクションです。取り込みは内部ロックで直列化されます (bot ではキューに積む想定)。

プロバイダは `LLMProvider` (`complete_json` / `complete_text`) と `Embedder` (`embed`, `dimensions`) の Protocol です。実 API を使わない決定的な `FakeLLM` / `FakeEmbedder` が付属します (`tests/test_memory*.py` 参照)。

### アクセス制御 (チャンネル ACL)

すべての Episode と Fact が出典チャンネル (`channel` / `channels`) を持ち、`search` / `search_entities` / `get_entity` / `get_episodes` / `answer` は `channels` (許可チャンネルのリスト) を受け取ります。許可されていないチャンネルだけに由来する情報は返しません。

| 対象 | ルール |
|---|---|
| Fact (`FACT` リレーションシップ) | 主張したエピソードのうち 1 つでも許可チャンネルのものがあれば可視。引用 (`episodes`) は許可チャンネルの分だけ |
| Episode | `channel` が許可されているものだけ |
| Entity | 許可チャンネルの発言で言及されたものだけ可視。名前 / エイリアスも、許可チャンネルで使われた表記だけを返し、検索マッチも可視な表記だけが対象 |
| Entity の要約 | **チャンネルごとに 1 行ずつ**保持 (`summaries`)。LLM が見る / 書き換えるのは取り込み中チャンネルの行だけなので他チャンネルの情報が混ざらず、許可チャンネルの行だけを返す。可視な行が無ければ、可視かつ有効なファクトから**クエリ時に導出** (`summary_source="derived"`) |
| 取り込み時の LLM 入力 | ファクト文・要約を書く LLM には、そのエピソードのチャンネルで見える情報 (発言中の表記、そのチャンネルの要約行) だけを渡す。重複ファクトへの追記は**同一チャンネル内だけ**で、別チャンネルの同内容の発言は別ファクトになる (ファクト文が他チャンネルの文言を持ち込まないため)。矛盾による失効は ID の判定だけなのでチャンネルをまたぐ |
| 埋め込み | エンティティの埋め込みは全チャンネル分の要約から作るため、制限付きの検索では、要約の出典が全て許可されているエンティティだけをベクトル一致に使う |

注意: 失効したファクトの `invalid_at` (いつ失効したか) は、失効させたのが不可視チャンネルの発言でも見えます (内容や `superseded_by` は出しません)。`channels=None` は無制限、`channels=[]` は何も返しません。`read_cypher` (下記) は生のグラフアクセスなので ACL を迂回します。

### MCP サーバー

```bash
python -m sqgraph.memory.mcp --db memory.db [--group-id G] [--allowed-channels a,b] [--read-only] \
       [--transport stdio|streamable-http] [--host 127.0.0.1] [--port 8000] [--env-file .env]
```

| ツール | 使いどころ |
|---|---|
| `search_memory` | まず最初に。人 / サービス / 決定 / 担当 / 状態 / 予定に関する質問。`as_of`, `include_invalid` (履歴), `center_entity` |
| `search_entities` | 名前・エイリアスの解決、存在する人 / サービスの列挙 |
| `get_entity` | 1 エンティティの全履歴 (失効分含むタイムライン) と関連エンティティ |
| `get_episodes` | 元の発言の確認、前後の会話の確認 |
| `add_memory` | 決定・担当・状態変化・予定など長期的に覚えるべきことの記録 (`--read-only` では無効) |
| `forget_memory` | エピソード / ファクト / エンティティ / 期間を指定した削除 (`dry_run` で影響範囲の確認)。書き込み可能で `--allowed-channels` なしのときだけ登録される |
| `get_schema` | ラベル / 関係タイプ / インデックス / 件数と Cypher サブセットの説明 |
| `read_cypher` | 読み取り専用 Cypher (行数 / 文字数 / 時間の制限つき)。上記で足りない集計・結合用 |

サーバー側で強制する設定: `--group-id` を指定すると名前空間が固定され (他の `group_id` は拒否)、`--allowed-channels` を指定すると全ての読み取りがそのチャンネルに制限され (呼び出し側は狭めることしかできない)、`add_memory` も許可チャンネルにしか書けません。**`read_cypher` は ACL を迂回する生アクセスなので、`--allowed-channels` 指定時は既定で無効**です (`--allow-cypher` で明示的に有効化できます)。`--read-only` では DB を読み取り専用で開き `add_memory` を登録しません。`streamable-http` は認証を持たないので、既定の `127.0.0.1` のまま使うか、リバースプロキシで保護してください。

Claude Code への登録:

```bash
claude mcp add sqgraph-memory -- python -m sqgraph.memory.mcp --db /path/to/memory.db --group-id my-guild
# チャンネル制限つき (read_cypher は無効になる)
claude mcp add sqgraph-memory-backend -- python -m sqgraph.memory.mcp --db /path/to/memory.db \
    --group-id my-guild --allowed-channels backend,infra
```

`.mcp.json` の例 (`OPENAI_API_KEY` は環境変数、または `--env-file` の `.env` から読みます):

```json
{
  "mcpServers": {
    "sqgraph-memory": {
      "command": "python",
      "args": ["-m", "sqgraph.memory.mcp", "--db", "/path/to/memory.db",
               "--group-id", "my-guild", "--allowed-channels", "backend,infra",
               "--env-file", "/path/to/.env"]
    }
  }
}
```

### 個人メモリとして使う (CLI / MCP / Skill)

同じマシンのローカルエージェント (Claude Code / Codex CLI / Cursor / Claude Desktop) が **1 つの SQLite ファイルを共有する個人用の長期記憶**として使うための配布形態です。Python だけで入ります (Node / Docker は不要)。

```bash
uv tool install "sqgraph[mcp]"      # または pipx install "sqgraph[mcp]"
# GitHub から: uv tool install "sqgraph[mcp] @ git+https://github.com/<owner>/sqgraph"

sqgraph-memory init                 # ~/.sqgraph-memory を作成。OPENAI_API_KEY は環境変数 or 対話入力 (非表示)
#   CI など非対話: echo "$OPENAI_API_KEY" | sqgraph-memory init --api-key-stdin
sqgraph-memory setup                # 検出したエージェントに MCP サーバーと Skill を登録
sqgraph-memory doctor [--online]    # 環境・DB・登録状況の診断 (失敗があれば終了コード 1)
```

`uvx sqgraph-memory ...` のような一時環境では登録するパスが消えることがあるため、`setup` は上の `uv tool install` を推奨する警告を出します。

`setup` が書き込むもの (既存の設定ファイルは `.bak` にバックアップし、他のサーバー設定には触れません。何度実行しても同じ結果で、`--dry-run` で確認、`--uninstall` で元に戻せます):

| エージェント | MCP の登録先 | Skill |
|---|---|---|
| Claude Code | `claude mcp add --scope user sqgraph-memory -- <実行ファイル> mcp` (`claude` が PATH にないときは実行すべきコマンドを表示) | `~/.claude/skills/sqgraph-memory/SKILL.md` |
| Codex CLI | `~/.codex/config.toml` の `[mcp_servers.sqgraph-memory]` (`command` / `args`) | `~/.agents/skills/sqgraph-memory/SKILL.md` |
| Cursor | `~/.cursor/mcp.json` の `mcpServers` | `~/.cursor/skills/sqgraph-memory/SKILL.md` |
| Claude Desktop | `claude_desktop_config.json` の `mcpServers` (Windows `%APPDATA%\Claude`、macOS `~/Library/Application Support/Claude`、Linux `~/.config/Claude`) | なし |

`--agents claude-code,codex,cursor,claude-desktop|all` で対象を指定し (既定は検出できたもの)、`--command "uvx --from 'sqgraph[mcp]' sqgraph-memory mcp"` で起動コマンドを丸ごと差し替え、`--no-skill` で Skill を入れません。登録後はエージェントを再起動してください。Skill (`SKILL.md`) は、いつ検索し、いつ覚えさせ、いつ忘れるか (明示的に頼まれたときだけ、`dry_run` で確認してから) をエージェントに教えます。

**CLI** (出力は常に UTF-8 の JSON、エラーは stderr に `{"error": ...}` で非ゼロ終了。`--compact` で 1 行):

| コマンド | 内容 |
|---|---|
| `init [--api-key-stdin] [--group-id G] [--tz TZ]` | ホーム・設定・`.env` を作る (キーは引数で渡せません) |
| `add TEXT\|- [--source claude-code] [--author user] [--channel personal] [--occurred-at T] [--source-ref R]` | 覚える (LLM による抽出で数秒かかる。`-` は stdin) |
| `search QUERY [--limit N] [--as-of T] [--history] [--center ENTITY]` | ファクト検索 (予定のファクトも含む。`--history` で失効分も) |
| `entity NAME` / `entities QUERY` / `episodes [--entity E] [--query Q]` / `stats` | プロファイルと履歴 / 名前解決 / 元のメモ / 件数 |
| `forget (--episode ID \| --source-ref REF [--source S] \| --fact ID \| --entity NAME [--keep-episodes] \| --since/--until/--channel [--source S]) [--dry-run]` | 削除 (対象の種類はちょうど 1 つ) |
| `purge --yes [--dry-run]` | 設定中の group の全記憶を削除 |
| `mcp [--transport stdio\|streamable-http] [--read-only]` | MCP サーバー (エージェントが起動する。設定ファイルの DB / group / モデルを自動で渡す) |
| `setup` / `doctor` | 上記 |

`python -m sqgraph.memory` でも同じです。API キーがないときも `search` / `entity` / `episodes` / `stats` / `forget` は動き (ベクトル検索だけ無効になり `warning` が付きます)、LLM が要るのは `add` だけです。

**全エージェントで 1 ファイルを共有**: 全員が `~/.sqgraph-memory/memory.db` を開き、書き込みは `memory.db.lock` のクロスプロセスのファイルロックで直列化されます (複数のエージェントが同時に `add` しても壊れず、待ち時間は `lock_timeout` まで)。あるエージェントが覚えたことは他のエージェントからも検索できます。MCP サーバーは `--group-id` を設定ファイルの値に固定して起動します。

**忘れる**: `forget` / `purge` / MCP の `forget_memory` は、DB から対象のエピソード・ファクト・孤立したエンティティを消し、残ったエンティティの要約と埋め込みを再計算します (削除されたファクトに失効させられていたファクトは有効に戻ります)。削除後は SQLite の `secure_delete` と WAL チェックポイントで、消した本文が DB ファイルにそのまま残らないようにします。`--dry-run` で何が消えるか確認できます。次のものは対象外です: DB ファイルの外にあるコピー (バックアップ、クラウド同期、Litestream など)、およびエピソードを取り込むときに OpenAI API へ送った内容 (OpenAI のポリシーに従って保持されます)。

**データの場所と権限**: 既定は `~/.sqgraph-memory/` (`SQGRAPH_MEMORY_HOME` で変更)。中身は `memory.db`、`.env` (`OPENAI_API_KEY=...`)、`config.json` (`group_id` 既定 `me`、`tz` 既定 `Asia/Tokyo`、`llm_model`、`embed_model`、`db`) です。POSIX ではディレクトリを 0700、ファイルを 0600 で作ります (Windows ではユーザープロファイルの ACL に任せます)。`SQGRAPH_MEMORY_DB` / `SQGRAPH_MEMORY_GROUP` で DB と group を上書きできます。`setup` 実行時にこれら (と `SQGRAPH_MEMORY_HOME`) が設定されていれば、パスは絶対パスにして各エージェントの登録 (`env`) に引き継ぎます (`--dry-run` の出力にも表示)。API キーの値は出力にもログにも出しません。エージェントは MCP サーバーをシェルの環境変数なしで起動するため、キーは `.env` から読みます。

### Discord bot への組み込み (概要)

`on_message` で `queue.put((message.content, channel.name, author.display_name, message.created_at, message.id))` し、ワーカーが `mem.add_episode(..., source="discord", source_ref=str(message.id))` を 1 件ずつ呼びます (メッセージ ID を `source_ref` にすれば再投入しても重複しません)。`/ask` スラッシュコマンドでは、質問者が閲覧できるチャンネルを Discord の権限 (`channel.permissions_for(member).view_channel`) で判定し、その一覧を `channels=` に渡して `mem.answer(question, group_id=str(guild.id), channels=allowed)` を呼びます。

### コスト目安 (gpt-5.4-mini, text-embedding-3-small)

価格 (1M トークンあたり): gpt-5.4-mini 入力 $0.75 / キャッシュ入力 $0.075 / 出力 $4.50、gpt-5.4-nano $0.20 / $0.02 / $1.25、text-embedding-3-small $0.02。`UsageTracker` が呼び出し種別 (`extract_entities` / `resolve_entities` / `extract_facts` / `dedupe_facts` / `answer` / `embed`) ごとに集計します。

実測 (社内 Discord 1 チャンネル、日本語 100 メッセージ、平均 93 文字を時系列順に 1 件ずつ取り込み):

| 項目 | 実測 |
|---|---|
| 取り込み合計 | 約 $0.39 (100 件、LLM 334 回 + 埋め込み 169 回、約 5.7 分) |
| 1 メッセージあたり | 約 $0.004 (LLM 約 3.3 回) → **1,000 件で約 $4、10,000 件で約 $40** (DB が育つと関連ファクトが増え、多少上振れ) |
| 内訳 (取り込み) | ファクト抽出 44% / エンティティ抽出 29% / 矛盾判定 20% / 同一判定 7% / 埋め込み 0.1% |
| 質問 (`answer`) 1 回 | 約 $0.002 (LLM 1 回 + 埋め込み 1 回)。`search` だけなら埋め込み 1 回 (ほぼ $0) |

コストを抑えるには、`gpt-5.4-nano` を `OpenAILLM("gpt-5.4-nano")` で試す、短い発言 (`min_content_chars`) や bot 自身の発言を取り込まない、`context_window` を小さくする、などが有効です。

### 制限事項

- 取り込みは 1 件ずつ直列です (LLM 呼び出し数回分の待ち時間)。大量の過去ログは `add_episodes_bulk` を夜間に流す想定です。
- エンティティの完全一致 / 名前検索は `group_id` 内のエンティティを走査するため、数万エンティティを超える規模では索引化が必要です。
- 抽出品質は LLM 依存です (誤った同一視 / 見落とし)。`get_entity` のタイムラインと出典 (`source_ref`) で確認できます。エンティティの手動マージやファクトの手動失効の API はまだありません。
- ファクトの全文検索は trigram なので 2 文字以下の語は索引にかかりません (エンティティ名は別経路で一致します)。

## クラウドへのデプロイ

AWS (EC2 / ECS Fargate / Lambda) や Cloudflare (Containers / Tunnel + Access / R2) で動かす構成、書き込み 1 プロセスの制約、Litestream によるバックアップ、読み取り用スナップショットの配布、MCP サーバーの認証は [docs/deployment.md](docs/deployment.md) にまとめています (Dockerfile などは [deploy/](deploy/))。

## テスト

```bash
python -m pytest -q
```
