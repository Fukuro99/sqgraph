[日本語](ja/graph-db.md)

# Graph DB (sqgraph)

## Quickstart (Python API)

```python
from sqgraph import GraphDB

with GraphDB("graph.db") as db:            # ":memory:" also works / read_only=True opens read-only
    alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
    bob = db.create_node("Person", {"name": "Bob"})
    db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})

    db.neighbors(alice.id, types=["KNOWS"], direction="out", depth=2)   # -> list[Node]
    db.get_node(alice.id)["name"]                                        # 'Alice'

    # Cypher (parameters are always bound as SQL parameters)
    rows = db.query(
        "MATCH (a:Person {name: $name})-[r:KNOWS]->(b) RETURN b.name AS friend, r.since AS since",
        {"name": "Alice"},
    )
    # [{'friend': 'Bob', 'since': 2020}]

    # Transactions (nestable: SAVEPOINT)
    with db.transaction():
        db.create_node("Person", {"name": "Carol"})
        db.query("MATCH (p:Person {name:'Bob'}) SET p.age = 25")

    print(db.explain("MATCH (a)-[:KNOWS*1..3]->(b) RETURN count(*)"))   # show the generated SQL
```

Main API:

| Method | Description |
|---|---|
| `GraphDB(path=":memory:", read_only=False, max_var_length=10)` | Open a database. Context manager / `close()` |
| `transaction()` | Atomic block (nestable). Each public method and each `query()` is also atomic on its own |
| `create_node / get_node / update_node / delete_node(detach=False)` | Node CRUD. A node that still has relationships cannot be deleted without `detach=True` |
| `add_labels / remove_labels` | Label operations |
| `create_edge / get_edge / update_edge / delete_edge` | Relationship CRUD |
| `create_nodes([(labels, props), ...])` / `create_edges([(src, dst, type[, props]), ...])` | Bulk creation via `executemany` (returns a list of IDs) |
| `neighbors(node_id, types=None, direction="out", depth=1, labels=None)` | Neighborhood traversal with a recursive CTE (cycle-safe, no duplicates, excludes the start node) |
| `create_vector_index / set_node_vector / vector_search` | Vectors |
| `create_fulltext_index / fulltext_search` | Full-text search |
| `create_relationship_vector_index / set_edge_vector` | Relationship vector index (searched with the same `vector_search`) |
| `create_relationship_fulltext_index` | Relationship full-text index (searched with the same `fulltext_search`) |
| `create_property_index(label, property, name=None)` | Expression index on `json_extract(props, '$.prop')` (a name from `CREATE INDEX name FOR ...` is also valid) |
| `create_relationship_property_index(type, property, name=None)` | The same expression index on the `edges` table (keeps `MATCH ()-[r:T {p: $v}]->()` from scanning every edge) |
| `list_indexes()` | Equivalent of `SHOW INDEXES`. `entityType` is `NODE` / `RELATIONSHIP` |
| `analyze()` | Refresh query planner statistics (recommended after bulk loads) |
| `query(cypher, params=None)` / `explain(cypher, params=None)` | Run Cypher / inspect the generated SQL |

`Node` has `id`, `labels` (a tuple) and `properties`, and supports `node["name"]`, `node.get("name")` and `.to_dict()`. `Relationship` has `id`, `type`, `start_node_id`, `end_node_id` and `properties`. Both compare equal by `id`. Named paths (`p = (a)-[*]->(b)`) are returned as `Path` (`nodes`, `relationships`).

## Vector search

```python
db.create_vector_index("entity_emb", "Entity", "embedding", dimensions=384, similarity="cosine")
# The Cypher DDL does the same (same syntax as Neo4j):
db.query("""
  CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
""")

# Embeddings are not stored in the props JSON; they live only in the sqlite-vec table (vec_<index name>)
db.set_node_vector(node_id, "embedding", vector)            # list[float] / numpy.ndarray
db.create_node("Entity", {"name": "OpenAI", "embedding": vector})   # indexed properties go to the vector table automatically
db.query("MATCH (n:Entity {name: 'x'}) CALL db.create.setNodeVectorProperty(n, 'embedding', $v)", {"v": vector})

# Python API: returns [(Node, score)] in descending score order
db.vector_search("entity_emb", query_vec, k=5)
```

Scores follow the Neo4j convention: cosine is `(1 + cos_sim) / 2`, euclidean is `1 / (1 + d^2)`.

### Relationship vector indexes

To store facts as edges and search them by embedding, relationships can have vector indexes too, as in Neo4j 5. Index names are unique across nodes and relationships (the backend is the same `vec_<index name>`; the relationship `vec0` uses `edge_id` as its primary key column, the node one uses `node_id`).

```python
db.create_relationship_vector_index("fact_emb", "FACT", "embedding", dimensions=384, similarity="cosine")
# Cypher DDL (same syntax as Neo4j 5; the directed form `()-[r:T]->()` is also accepted):
db.query("""
  CREATE VECTOR INDEX fact_emb IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
""")

# Indexed properties are not stored in the edge's props JSON, only in the vector table (RETURN r.embedding is null)
db.create_edge(a.id, b.id, "FACT", {"fact": "Alice owns the Rails migration", "embedding": vector})
db.set_edge_vector(edge_id, "embedding", vector)
db.query("MATCH ()-[r:FACT {id: $id}]->() CALL db.create.setRelationshipVectorProperty(r, 'embedding', $v)", {"id": "f1", "v": vector})

# The Python API returns [(Node, score)] / [(Relationship, score)] depending on the index kind
db.vector_search("fact_emb", query_vec, k=5)

# Cypher: a yielded relationship is an ordinary relationship variable in later clauses
db.query("""
  CALL db.index.vector.queryRelationships('fact_emb', $k, $q) YIELD relationship, score
  MATCH (s)-[relationship]->(o) WHERE score > 0.6
  RETURN s.name AS subject, relationship.fact AS fact, o.name AS object, type(relationship) AS type, score
""", {"k": 10, "q": query_vec})
```

Creates, updates and deletes are synchronized automatically (`create_edge` / `create_edges` / `update_edge` (including `replace=True`) / `delete_edge` / `delete_node(detach=True)`, and in Cypher `CREATE` / `MERGE` / `SET r.p = ...` / `SET r += {...}` / `SET r = {...}` / `REMOVE r.p` / `SET r.p = null` / `DELETE r` / `DETACH DELETE n`). When an index is created, embeddings already stored as arrays in the props of edges of the target type are moved into the vector table (as with node indexes). Using `queryRelationships` on a node index, or `queryNodes` on a relationship index, raises a clear error.

GraphRAG local search (top vector hits, expand 1-2 hops, collect chunks and communities):

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

The result of a `CALL` combines freely with later `MATCH` / `WITH` / `WHERE` clauses (see [`tests/test_graphrag_scenario.py`](../tests/test_graphrag_scenario.py)).

## Full-text search

```python
db.create_fulltext_index("chunk_text", ["Chunk"], ["text"])          # for Japanese and similar, tokenizer="trigram" also works
db.query("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (n:Chunk|Note) ON EACH [n.text, n.title]")

db.fulltext_search("chunk_text", "graph AND (vector OR retrieval)", k=10)   # [(Node, score)]  score = -bm25 (higher is better)
db.query("CALL db.index.fulltext.queryNodes('chunk_text', $q) YIELD node, score RETURN node.text, score", {"q": "graph"})
```

Query strings are Lucene-style: terms are OR-ed by default, with `AND` / `OR` / `NOT`, `"phrase"`, `-exclude`, `prefix*` and `field:term`. Input containing special characters never causes an FTS5 syntax error. Indexes stay in sync automatically with node and label creation, update and deletion, from both the Python API and Cypher.

### Relationship full-text indexes

```python
db.create_relationship_fulltext_index("fact_ft", ["FACT", "STATE"], ["fact"], tokenizer="trigram")   # types may be a str or a list
db.query("CREATE FULLTEXT INDEX fact_ft IF NOT EXISTS FOR ()-[r:FACT|STATE]-() ON EACH [r.fact, r.note] OPTIONS {indexConfig: {`fulltext.tokenizer`: 'trigram'}}")

db.fulltext_search("fact_ft", "auth platform", k=10)      # [(Relationship, score)]
db.query("""
  CALL db.index.fulltext.queryRelationships('fact_ft', $q) YIELD relationship, score
  MATCH (s)-[relationship]->(o) RETURN s.name, relationship.fact, o.name, score
""", {"q": "auth platform"})
```

The index is stored in an FTS5 table (`fts_<index name>`) whose `rowid` is the edge ID. It stays in sync with edge creation, update and deletion, including edges removed by a node's `DETACH DELETE`. When an index is created, existing edges of the target types are indexed as well.

### Relationship property indexes

```cypher
CREATE INDEX fact_id IF NOT EXISTS FOR ()-[r:FACT]-() ON (r.id)     // RANGE also works. The directed form ()-[r:FACT]->() is accepted too
MATCH ()-[r:FACT {id: $id}]->() RETURN r.fact                        // uses the expression index on edges (WHERE r.id = $id works the same way)
```

This is an expression index on `json_extract(props, '$.id')` of the `edges` table. The type is bound as a SQL parameter, so the index cannot be partial: it covers all edges regardless of type (the type is recorded only as metadata).

## Cypher support

### Supported

| Category | Contents |
|---|---|
| Clauses | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `WITH` (`DISTINCT` / aggregation / `ORDER BY` / `SKIP` / `LIMIT` / `WHERE`), `RETURN` (`DISTINCT`, `AS`, `*`), `ORDER BY` (multiple keys, `ASC`/`DESC`), `SKIP`, `LIMIT`, `UNWIND`, `CREATE`, `MERGE` (`ON CREATE SET` / `ON MATCH SET`), `SET` (`n.p = e`, `n += map`, `n = map`, `n:Label`), `REMOVE` (`n.p`, `n:Label`), `DELETE`, `DETACH DELETE`, `CALL proc(...) YIELD a, b [WHERE ...]`, `UNION` / `UNION ALL` |
| Patterns | `(n)`, `(n:L)`, `(n:A:B)`, `(n {k: v})`, `-[r:T]->`, `<-[r]-`, `-[r]-`, `-->` / `<--` / `--`, `-[:A\|B]->`, variable length `*`, `*2`, `*1..3`, `*..3`, `*2..`, relationship property maps, multiple patterns, chained patterns, reuse of bound variables, named paths `p = (...)` with `length(p)` / `nodes(p)` / `relationships(p)`, variable-length relationship variables (a list of `Relationship`) |
| Expressions | Literals (integer, float, string, `true/false/null`, list, map), `$param`, `n.prop`, `n[$key]`, arithmetic / `%` / `^`, string concatenation, comparisons (chainable), `AND OR NOT XOR`, `IS [NOT] NULL`, `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `=~`, `CASE` (both forms), `list[i]`, `list[a..b]`, label predicate `n:Label`, `EXISTS { MATCH ... }`, `exists(n.prop)`, list comprehensions, `all/any/none/single`, map projection `n {.*, .name, k: expr}` |
| Functions | `id`, `elementId`, `labels`, `type`, `properties`, `keys`, `coalesce`, `toLower`, `toUpper`, `trim`/`ltrim`/`rtrim`, `size`, `length`, `toString`, `toInteger`, `toFloat`, `toBoolean`, `abs`, `round`, `ceil`, `floor`, `sqrt`, `sign`, `exp`, `log`, `log10`, `sin`, `cos`, `pi`, `e`, `rand`, `startNode`, `endNode`, `nodes`, `relationships`, `range`, `head`, `last`, `tail`, `reverse`, `split`, `replace`, `substring`, `left`, `right`, `isEmpty`, `timestamp` |
| Aggregation | `count(*)`, `count(x)`, `count(DISTINCT x)`, `collect`, `collect(DISTINCT)`, `sum`, `avg`, `min`, `max` (non-aggregated items are implicit grouping keys) |
| Procedures | `db.index.vector.queryNodes`, `db.index.vector.queryRelationships` (`YIELD relationship, score`), `db.index.fulltext.queryNodes`, `db.index.fulltext.queryRelationships` (`YIELD relationship, score`), `db.create.setNodeVectorProperty`, `db.create.setRelationshipVectorProperty`, `db.create.setVectorProperty`, `db.index.vector.createNodeIndex`, `db.index.fulltext.createNodeIndex`, `db.labels`, `db.relationshipTypes`, `db.propertyKeys` |
| Schema | `CREATE VECTOR INDEX ... FOR (n:L) ON (n.p)` / `FOR ()-[r:T]-() ON (r.p)`, `CREATE FULLTEXT INDEX ... FOR (n:A\|B) ON EACH [n.p, ...]` / `FOR ()-[r:A\|B]-() ON EACH [r.p, ...]`, `CREATE [RANGE] INDEX FOR (n:L) ON (n.p)` / `FOR ()-[r:T]-() ON (r.p)` (single property; relationships accept either `-` or `->` as the end), `DROP INDEX [IF EXISTS]`, `SHOW INDEXES` (`entityType` = `NODE` / `RELATIONSHIP`) |
| Other | Backtick identifiers (`` `my label` ``), comments (`//`, `/* */`), keywords as property and label names (`n.order`, `:Order`, etc.), case-insensitive keywords |

### Partially supported / caveats

| Item | Status |
|---|---|
| Upper bound of variable-length `*` patterns | When the upper bound is omitted, traversal stops at `max_var_length` (default 10; `GraphDB(max_var_length=...)`). A relationship never repeats within one path (trail). Variable-length patterns inside `EXISTS { }` are unsupported |
| `MERGE` | Single path only. Named paths and variable length are not allowed. An undirected `MERGE` matches existing entities in both directions and creates left-to-right if none is found |
| `CREATE` / `MERGE` property expressions | Variables created in the same clause cannot be referenced (`CREATE (a {x:1}), (b {y: a.x})` is an error) |
| Property values | Primitives or lists of primitives only (as in Neo4j). `SET n.p = {map}` is an error |
| `UNION` | Read-only. Branches containing write clauses are not allowed. The same column may have different types per branch (only mixing nodes / relationships / paths with other values is not allowed) |
| Labels / embeddings | `labels(n)` is in alphabetical order (Neo4j uses creation order). Indexed embedding properties are not in props, so `RETURN n.embedding` (or `RETURN r.embedding` for a relationship index) is `null` |
| Type checking | `=` across types is false, and ordering comparisons such as `<` yield `null` (as in Neo4j; booleans are distinct from integers). Use `json_type()` to inspect dynamically typed values that come from properties |
| Integer division by zero | `/` and `%` on integers raise an error; floating-point division by zero yields `null` (Neo4j yields Infinity/NaN) |

### Unsupported (raise a clear error)

`CALL { ... }` subqueries, `FOREACH`, `LOAD CSV`, `shortestPath()` / `allShortestPaths()`, pattern predicates (`WHERE (a)-[:R]->(b)`; use `EXISTS { ... }` instead), pattern comprehensions, `reduce()`, temporal types and functions, spatial types, `USING INDEX` hints, constraints (`CREATE CONSTRAINT`), composite property indexes, APOC functions / procedures, `count { }` subquery expressions, `stDev` / `percentile*` aggregations.

## How it works

- Schema: `nodes(id, props JSON)`, `node_labels(label, node_id)` (`WITHOUT ROWID`), `edges(id, src, dst, type, props JSON)`, `_meta` (`schema_version` = 3), `_vector_indexes`, `_fulltext_indexes` and `_property_indexes` (each has `entity_type` = `node` / `relationship`; for relationship indexes the `label` / `labels` column holds the type). Vector indexes are `vec_<name>` (vec0; the primary key is `node_id` for node indexes and `edge_id` for relationship indexes) and full-text indexes are `fts_<name>` (FTS5, `rowid` = node ID or edge ID). Older databases (schema 2 or earlier) get the `entity_type` column added when opened for writing (a read-only open does not migrate and treats every index as a node index).
- Compiler: each read clause becomes one CTE (a `SELECT` over the previous clause's binding table). Node and relationship variables are ID columns, properties use `json_extract` / `->`, `OPTIONAL MATCH` is a `LEFT JOIN`, variable length is a recursive CTE, `UNWIND` is `json_each`, and `collect` is `json_group_array`. The compiler tracks type information (node, boolean, list, map, ...) as static types, and results are restored to `Node` / `bool` / `list` / `dict` accordingly.
- Write clauses (`CREATE` / `MERGE` / `SET` / `REMOVE` / `DELETE`) run in Python against the rows bound so far (expressions are still evaluated in SQL). One `query()` is one transaction, and any error rolls back the whole query. Write queries on a read-only database are rejected with an explicit error.
- Safety: all user values are SQL parameters, and so are labels and types. Property keys are inlined into the JSON path only when they match `[A-Za-z_][A-Za-z0-9_]*` (so that expression indexes apply; otherwise they are parameters). Index names that enter DDL are validated as identifiers.
- Performance: for SQLite to optimize highly selective anchors (`MATCH (n:L {prop: $v})-[...]`) correctly, call `db.analyze()` after bulk loads and after creating indexes (`create_property_index` and `close()` run it automatically). The recursive step of variable-length patterns pins the adjacency index with `INDEXED BY`, so it never scans all edges even on a database without statistics.

## Viewer

A local viewer for browsing the graph in a browser is included (the server uses only the standard library; the frontend is a single HTML file that loads Cytoscape.js from a CDN). The database is always opened read-only, so the Cypher console cannot write (write queries fail with a clear error).

```bash
python -m sqgraph.viewer path/to/graph.db [--host 127.0.0.1] [--port N] [--no-browser]
# e.g. python -m sqgraph.viewer examples/graphrag_openai/graphrag.db
```

By default it listens only on `127.0.0.1`, picks a free port starting at 8765 (so several databases can be open at once), and opens the browser on startup (suppress with `--no-browser`). If you pass `--port` explicitly and that port is in use, it exits with an error. The frontend loads Cytoscape.js from `cdnjs.cloudflare.com`, so the first page load needs network access.

- **Sidebar**: node / relationship counts; per-label counts (click the color chip to toggle visibility, `↓` loads that label's nodes); per-relationship-type counts (click to toggle visibility); property keys; the index list; and substring search over name / title / id / text (click a result to add it to the graph and focus it)
- **Graph**: starts with the highest-degree nodes; per-label colors (light / dark theme aware); node size scaled by degree; directed edges with type names (toggleable); double-click to expand neighbors; force-directed `cose` layout; and "community colors" when a GraphRAG schema (`Entity-[:IN_COMMUNITY]->Community`) is present
- **Detail panel**: click a node or edge to see its labels, all properties (long text is collapsed), degree and per-type relationship counts. Expand / focus on neighborhood / hide buttons
- **Cypher console**: run with `Ctrl+Enter`; GraphRAG example queries; results shown as a table, and any Node / Relationship / Path in them can replace or be added to the graph view
- **Toolbar**: number of nodes to show, re-layout, fit to view, clear, edge-label toggle, PNG export. The sidebar collapses on narrow screens

JSON API (used by the frontend): `GET /api/overview`, `GET /api/graph?limit=&labels=A,B`, `GET /api/node/<id>`, `GET /api/expand/<id>?limit=&types=A,B`, `GET /api/search?q=`, `POST /api/query {cypher, params}`, `POST /api/edges {ids}`. Vector embeddings (numeric lists of 65 or more dimensions) are omitted from responses (indexed embeddings are not in props, so relationship embeddings are not returned either). `indexes` also includes relationship indexes with `entityType`, and the sidebar displays them as `[:TYPE].property`.
