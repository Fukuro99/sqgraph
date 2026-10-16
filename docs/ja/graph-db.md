[English](../graph-db.md)

# グラフ DB (sqgraph)

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

`CALL` の結果は後続の `MATCH` / `WITH` / `WHERE` と自由に組み合わせられます ([`tests/test_graphrag_scenario.py`](../../tests/test_graphrag_scenario.py) 参照)。

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
| `UNION` | 読み取りのみ。書き込み句を含む枝は不可。同じ列の型は枝ごとに違ってよい (ノード / リレーションシップ / パスと他の値の混在だけは不可) |
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
