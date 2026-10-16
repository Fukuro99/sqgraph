# sqgraph

[![CI](https://github.com/Fukuro99/sqgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Fukuro99/sqgraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)

[日本語](README.ja.md)

An embedded property graph database on SQLite and [sqlite-vec](https://github.com/asg017/sqlite-vec), queried with a subset of openCypher. Built as the storage and query layer for GraphRAG, and as long-term memory for local AI agents.

- **One file, no server.** Nodes, relationships, labels and JSON properties in a single SQLite file (or `:memory:`).
- **openCypher compiled to SQL.** Reads become plain SQL (JOINs, recursive CTEs, `json_extract` / `json_each`) executed by SQLite.
- **Neo4j-compatible search.** Vector indexes (sqlite-vec) and full-text indexes (FTS5, BM25) on nodes and relationships, with the same DDL and procedures as Neo4j 5 (`CREATE VECTOR INDEX`, `db.index.vector.queryNodes`, `db.index.fulltext.queryRelationships`, ...), so code written for `Neo4jGraph` / `Neo4jVector` needs minimal changes.
- **Agent memory.** `sqgraph.memory` is a temporal knowledge graph: facts have validity periods, superseded facts are kept as history, and search combines vectors, full text and graph hops. It ships an MCP server, a CLI, hooks and a Skill for Claude Code, Codex CLI, OpenCode, Cursor and Claude Desktop.
- **Small footprint.** The only runtime dependencies are `sqlite-vec` and `lark`.

## Installation

sqgraph is not on PyPI yet; install it from GitHub:

```bash
pip install "sqgraph @ git+https://github.com/Fukuro99/sqgraph"
```

| Extra | Adds | For |
|---|---|---|
| `memory` | `openai` | `sqgraph.memory` with OpenAI or an OpenAI-compatible embedding server |
| `mcp` | `mcp`, `openai` | the MCP server and the `sqgraph-memory` CLI |
| `dev` | `pytest` | running the tests |

Requires Python 3.10+ with an `sqlite3` module built on SQLite 3.38+ that can load extensions. uv-managed Python, the python.org Windows installers and most Linux distributions qualify; some macOS Pythons do not (`sqgraph-memory doctor` checks).

## Quickstart

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

    print(db.explain("MATCH (a)-[:KNOWS*1..3]->(b) RETURN count(*)"))  # the generated SQL
```

Vector and full-text search use the Neo4j syntax:

```python
db.query("""
  CREATE VECTOR INDEX entity_emb IF NOT EXISTS FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
""")
db.create_node("Entity", {"name": "OpenAI", "embedding": vector})  # indexed properties go to sqlite-vec
db.query(
    "CALL db.index.vector.queryNodes('entity_emb', 5, $q) YIELD node, score RETURN node.name, score",
    {"q": query_vector},
)
```

Browse a database in the bundled viewer:

```bash
python -m sqgraph.viewer graph.db
```

## Long-term memory for local agents

Give every agent on your machine one shared, local memory. The calling agent does the extraction itself, so no LLM API key is needed; embeddings can run on a local server such as LM Studio or Ollama.

```bash
uv tool install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"
sqgraph-memory init      # create ~/.sqgraph-memory
sqgraph-memory setup     # register the MCP server, Skill and hooks with the agents it finds
sqgraph-memory doctor    # check the environment
```

Bots can use the Python API directly:

```python
from sqgraph.memory import Memory, OpenAILLM, OpenAIEmbedder

mem = Memory("memory.db", OpenAILLM("gpt-5.4-mini"), OpenAIEmbedder("text-embedding-3-small"),
             default_group_id="my-team")
mem.add_episode("Bob takes over the auth service from next week", source="discord", author="alice")
for hit in mem.search("Who owns the auth service?"):
    print(hit.status, hit.fact, hit.valid_at, hit.invalid_at)
```

## Documentation

- [Graph database](docs/graph-db.md): Python API, vector and full-text search, indexes, Cypher support, internals, viewer
- [Agent memory](docs/memory.md): data model, Python API, agent-side extraction, access control, MCP server, curation, export
- [Personal memory for local agents](docs/personal-memory.md): setup, hooks, CLI reference, embedding servers
- [Deployment](docs/deployment.md): running on AWS and Cloudflare, backups with Litestream
- [Examples](examples/): GraphRAG with OpenAI, vector search over chat logs

## Development

```bash
pip install -e ".[dev,mcp]"
python -m pytest -q
```

## License

[MIT](LICENSE)
