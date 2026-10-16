[日本語](ja/memory.md)

# Agent long-term memory (sqgraph.memory)

A **temporal knowledge graph for long-term memory**, aimed at team agents (a Discord bot, an MCP client, and so on) and inspired by [Graphiti / Zep](https://github.com/getzep/graphiti). The core `sqgraph` package stays LLM-free. The `sqgraph.memory` package uses a pluggable provider (default: OpenAI) to extract entities and facts from messages, detect contradictions and **invalidate (never delete)** outdated facts, and offers hybrid search with `as_of`.

```bash
pip install -e ".[memory]"      # openai
pip install -e ".[mcp]"         # MCP server (mcp + openai)
```

## Data model

Every node carries a `group_id` (a namespace such as a guild or project). Timestamps are stored as UTC ISO 8601 strings (`2026-07-01T09:52:00Z`, which compare correctly as strings); search results are returned with an offset in `tz` (default Asia/Tokyo).

```
(:Episode {id, group_id, source, channel, author, content, occurred_at, ingested_at, source_ref})   raw message (provenance)
(:Entity  {id, group_id, name, type, summary, aliases, created_at, updated_at, ...})               vector + trigram full-text index
(subject:Entity)-[:FACT {id, group_id, fact, text_norm, relation, subject_id, object_id, valid_at, invalid_at,
                        created_at, expired_at, channels, episode_ids, confidence, modality,
                        superseded_by, supersedes}]->(object:Entity)                                  a fact is a FACT relationship (as in Graphiti). Vector + trigram full-text index + property index on id
(Episode)-[:MENTIONS]->(Entity)
```

A fact is a single `FACT` relationship from a subject entity to an object entity; the verb lives in the `relation` property. An attribute fact with no object (a state, an absence) is a self-loop `(s)-[:FACT]->(s)` and has no `object_id` (`FactHit.object` is `None`). `subject_id` / `object_id` are denormalized entity IDs. Look up a fact by ID with `MATCH ()-[f:FACT {id: $id}]->()` (served by the property index).

Migrating from the old format (a DB that stores facts as `:Fact` nodes) is not supported. Opening such a DB raises an error; recreate it.

- **Bitemporal**: `valid_at` / `invalid_at` is the period during which the fact was true in the world (the message time, or a time computed from the text, such as "from next week" or "off 9/14-9/16"). `created_at` / `expired_at` are system times. When a contradicting fact arrives, nothing is **deleted**: the older fact gets `invalid_at` (= the new fact's `valid_at`) and `expired_at`, and is linked by `superseded_by` (the new fact's ID); the newer fact's `supersedes` holds the list of older IDs. When episodes are ingested out of order (an older message arrives later), the new fact (with the earlier timestamp) gets an `invalid_at` equal to the existing newer fact's `valid_at`, and that existing fact becomes its `superseded_by`.
- **Fact status** (at `as_of`): `valid` (`valid_at <= as_of` and `invalid_at` is null or `> as_of`) / `ended` (invalidated; fetch with `include_invalid=True`) / `scheduled` (a plan that has not started yet; `include_future=True`).
- **Discord display-name normalization**: temporary absence suffixes such as `taro/Yamada_1718休`, `hanako3(9/14-9/16お休み)` or `jiro / Sato Jiro [不在]9/9～11` are stripped and merged into a single Person. The original spelling and each `/`-separated part (`taro`, `Yamada`) are kept as aliases.

## Python API (for bots)

```python
from sqgraph.memory import Memory, OpenAILLM, OpenAIEmbedder, UsageTracker, load_env

load_env(".env")                                   # OPENAI_API_KEY (the value is never printed or logged)
tracker = UsageTracker()                           # usage and cost accounting
mem = Memory("memory.db",
             OpenAILLM("gpt-5.4-mini", tracker=tracker),                  # default
             OpenAIEmbedder("text-embedding-3-small", tracker=tracker),   # 1536 dimensions
             default_group_id="my-guild", tz="Asia/Tokyo")

# Ingest (1 message = 1 episode. Re-submitting the same (source, source_ref) is ignored, so retries are safe)
r = mem.add_episode("Bob takes over the auth service from next week", group_id="my-guild", source="discord",
                    channel="backend", author="alice(9/14-9/16お休み)",
                    occurred_at="2026-09-29 16:50", source_ref="discord:123456789")
# r.entities / r.facts_created / r.facts_invalidated / r.llm_calls / r.usd

mem.add_episodes_bulk(rows, group_id="my-guild", channel="backend")   # processed in ascending occurred_at order

# Search: fact vector + trigram full-text + entity match -> 1-2 hop expansion, fused with RRF
hits = mem.search("Who owns the auth service?", group_id="my-guild",
                  channels=["backend"],              # channels the caller may read (None = unrestricted)
                  as_of="2026-07-20",                # defaults to now
                  center_entity="bob",                # rerank by graph distance (optional)
                  include_invalid=True)              # include invalidated facts (history)
for h in hits:                                       # FactHit: fact, valid_at, invalid_at, status, episodes[provenance] ...
    print(h.status, h.fact, h.valid_at, h.invalid_at, [(e.author, e.occurred_at, e.source_ref) for e in h.episodes])

# Relevance gate: facts unrelated to the query are not returned. FactHit.similarity = cosine similarity to the query (None when there is no vector).
# Override the threshold with Memory(min_similarity=...) or mem.search(..., min_similarity=0.5) (default: per embedding model)
mem.search_entities("bob", channels=["backend"])            # [{id, name, type, aliases, summary, ...}]
mem.get_entity("bob", channels=["backend"])                 # profile + fact timeline (including invalidated facts)
mem.get_episodes(entity="Rails", channels=["backend"], limit=10)
ans = mem.answer("How far along is the Rails migration?", group_id="my-guild", channels=["backend"])
print(ans.text)                                                # answer in Japanese, with [F1] citations and validity periods
print(tracker.summary("run"))
```

Ingesting one episode normally takes 2 to 4 LLM calls: (1) entity extraction (recent messages from the same channel are passed as context to resolve references such as "it"; messages with nothing worth remembering stop here), (2) resolution against existing entities (exact and alias matches need no LLM; only ambiguous names are judged, with vector / full-text / exact-match candidates batched into a single call), (3) fact extraction plus entity summary update, (4) duplicate / contradiction detection (only when related valid facts already exist). Embeddings take 1 to 2 batched calls per episode, and the DB write is a single transaction. Ingestion is serialized by an internal lock (a bot is expected to put messages on a queue).

### Agent extraction (two-phase mode without a server-side LLM)

`add_episode` has a server-side LLM (OpenAI, for example) extract entities and facts; this is the bot-oriented path. To have **the calling agent itself** do the extraction, use the stateless two-phase API. It needs no LLM and no OpenAI key (only an embeddings endpoint, configured as an embedder).

```python
mem = Memory("memory.db", None, embedder, default_group_id="me")      # no llm needed

# Phase 1: no writes, no write lock, no LLM
prep = mem.prepare_episode("Please use pnpm from next week", source="claude-code", channel="personal", author="user",
                           occurred_at="2026-10-05 10:00", source_ref=None)
# prep["already_stored"]  (True + episode_id if the same (source, source_ref) was already ingested)
# prep["token"]         self-contained opaque string (carries group, source, channel, author, time and text; the server remembers nothing)
# prep["instructions"]  extraction rules for the agent (short)      prep["schema"]  JSON Schema of the extraction
# prep["context"]       message / author_ref / recent / candidate_entities / related_facts (all computed without an LLM)

# The agent reads the context and builds an extraction (schema below)
res = mem.commit_episode(prep["token"], extraction)                  # validated and stored under the write lock
# res.facts_created / facts_duplicate / facts_invalidated / res.warnings (what the server dropped or merged)
```

`context`: `message` (text, author, local time and weekday, channel), `author_ref` (the agent can refer to the author as `"author"`; `existing_id` if the author already exists), `recent` (the preceding messages in the same channel), `candidate_entities` (existing entities chosen by name / alias match against the text, trigram full-text, vector search and the author; at most `max_entities`, each `{id, name, type, aliases, summary}`), `related_facts` (existing facts that were valid or scheduled at the message time and touch the candidate entities, plus facts close to the text; at most `max_facts`). Only information visible from that channel is included (same ACL as above; `related_facts` are all facts of the same channel, so they are usable as `duplicate_of`).

The `extraction` schema (`prep["schema"]`):

```json
{"entities": [{"ref": "e1", "name": "pnpm", "type": "Service", "existing_id": null,
               "summary": "JS package manager the user prefers", "aliases": []}],
 "facts": [{"subject": "author", "object": "e1", "relation": "PREFERS",
            "fact": "The user wants pnpm instead of npm", "valid_at": null, "invalid_at": null,
            "modality": "definite", "confidence": 0.95, "duplicate_of": null, "contradicts": []}]}
```

`existing_id` is the id of a `candidate_entities` entry (reuse it if it is the same thing). `subject` / `object` are a `ref` or `"author"` (an attribute fact uses `object: null`). A null `valid_at` means the message time. Times are ISO 8601 (local time / with `Z` / date only). `duplicate_of` is the id of an existing fact this restates (same channel only). `contradicts` lists ids of valid facts that no longer hold because of this message (the older fact is invalidated and linked through `superseded_by`; the semantics are identical to the OpenAI pipeline, which uses the same internal `_commit` / `_write_fact` / `_invalidate`). `summary` becomes the entity's summary line for that channel. An empty extraction (`{}`) stores only the message (and the mention of its author).

`commit_episode` raises `ValueError` listing **every** problem it finds (undefined refs, an `existing_id` that does not exist or belongs to another group, ids that do not look like `fact_...`, unparseable times, invalid `modality` / `confidence`, and so on). So that the DB may change between the two phases, a `duplicate_of` / `contradicts` that no longer exists, a `duplicate_of` pointing at a fact in another channel, and facts that are already invalidated are dropped and recorded in `res.warnings`. A new entity with `existing_id: null` is still merged into an existing entity when there is a deterministic exact or alias match with a compatible type (a safety net against duplicates). A tampered, corrupted or truncated token is rejected as `invalid token`.

Providers are the Protocols `LLMProvider` (`complete_json` / `complete_text`) and `Embedder` (`embed`, `dimensions`). Deterministic `FakeLLM` / `FakeEmbedder` implementations that need no real API are included (see `tests/test_memory*.py`).

## Access control (channel ACL)

Every Episode and Fact records its source channel (`channel` / `channels`), and `search` / `search_entities` / `get_entity` / `get_episodes` / `answer` accept `channels` (a list of allowed channels). Information that comes only from channels that are not allowed is never returned.

| Object | Rule |
|---|---|
| Fact (`FACT` relationship) | Visible if at least one of the episodes that asserted it is in an allowed channel. Citations (`episodes`) are limited to allowed channels |
| Episode | Only those whose `channel` is allowed |
| Entity | Visible only if mentioned in a message from an allowed channel. Names and aliases are also limited to spellings used in allowed channels, and search matches only consider visible spellings |
| Entity summary | Kept **one line per channel** (`summaries`). The LLM sees and rewrites only the line of the channel being ingested, so nothing leaks between channels, and only lines of allowed channels are returned. If no visible line exists, the summary is **derived at query time** from visible, valid facts (`summary_source="derived"`) |
| LLM input at ingestion | The LLM that writes fact text and summaries receives only what is visible in that episode's channel (spellings used in the message, that channel's summary line). Appending to a duplicate fact happens **within the same channel only**; the same content said in another channel becomes a separate fact (so fact text never carries wording from other channels). Invalidation by contradiction is only an ID decision, so it crosses channels |
| Embeddings | An entity's embedding is built from its summaries across all channels, so in a restricted search only entities whose summary sources are all allowed take part in vector matching |

Note: the `invalid_at` of an invalidated fact (when it ended) is visible even if the message that invalidated it came from a channel that is not visible (its content and `superseded_by` are not exposed). `channels=None` means unrestricted; `channels=[]` returns nothing. `read_cypher` (below) is raw graph access and bypasses the ACL.

## MCP server

```bash
python -m sqgraph.memory.mcp --db memory.db [--group-id G] [--allowed-channels a,b] [--read-only] \
       [--extractor agent|openai] [--transport stdio|streamable-http] [--host 127.0.0.1] [--port 8000] [--env-file .env]
```

| Tool | When to use |
|---|---|
| `search_memory` | First, for questions about people / services / decisions / ownership / status / plans. Accepts `as_of`, `include_invalid` (history), `center_entity`, `min_similarity` (relevance threshold). Facts that look unrelated are not returned (an unrelated question yields nothing), and each fact carries `similarity`, its cosine similarity to the query |
| `search_entities` | Resolving names and aliases; listing the people / services that exist |
| `get_entity` | The full history of one entity (timeline including invalidated facts) and its related entities |
| `get_episodes` | Checking the original messages and the conversation around them |
| `add_memory` | Recording things worth remembering long term: decisions, ownership, status changes, plans, preferences (disabled with `--read-only`). With `--extractor agent` (default) it **stores nothing** and returns `token` / `instructions` / `schema` / `context`. With `openai`, a server-side LLM extracts and stores |
| `commit_memory` | Phase 2 of `agent` mode. Pass the `token` from `add_memory` and the `extraction` the agent built, and it is stored (also usable in `openai` mode; disabled with `--read-only`) |
| `forget_memory` | Delete by episode / fact / entity / time range (`dry_run` previews the impact). Example: `forget_memory(episode_id="ep_...", dry_run=True)`. Exactly one selector: `episode_id` (alias `episode`) / `fact_id` (alias `fact`) / `entity` (aliases `entity_id`, `name`) / `source_ref` (+ `source`) / `since`, `until`, `channel`. Registered only when the server is writable and has no `--allowed-channels` |
| `merge_entities` | Merge two entities that refer to the same thing (`merge` into `keep`: names, aliases, summaries, facts and mentions move over, and exactly identical facts collapse into one. History is preserved. Supports `dry_run`) |
| `invalidate_fact` | Manually invalidate a fact that no longer holds (`at` defaults to now, `reason` is recorded. A plan that has not started is cancelled. Not a deletion) |
| `rename_relation` | Rename a relation in bulk (`swap=true` also swaps subject and object; `fact_ids` limits it to some facts. Supports `dry_run`) |
| `review_memory` | Returns cleanup candidates (pairs of entities that look like duplicates with reasons, the list of relation names with unification suggestions, and unconfirmed plans whose date has passed). Makes no changes |
| `get_upcoming` | Facts (plans, absences, deadlines) that start or end within the next `days` days (and the past `past_days` days) |
| `get_schema` | Labels / relationship types / indexes / counts, and a description of the Cypher subset |
| `read_cypher` | Read-only Cypher (with row / character / time limits), for aggregation and joins the tools above cannot do |

`merge_entities` / `invalidate_fact` / `rename_relation` are registered only when the server is writable and has no `--allowed-channels`; `review_memory` (which sees all channels) only when there is no `--allowed-channels`. `get_upcoming` respects the channel restriction.

`--extractor` (default `agent`) selects who extracts entities and facts for `add_memory`. With `agent`, the server needs neither an LLM nor an OpenAI key: the calling agent builds the `extraction` by following `instructions` and then calls `commit_memory` (see "Agent extraction"; this does not depend on MCP sampling). `openai` keeps the original behavior, where one `add_memory` call has an OpenAI LLM do the extraction (for bots; requires `OPENAI_API_KEY`, and `add_memory` errors without it).

Settings enforced on the server: `--group-id` fixes the namespace (any other `group_id` is rejected). `--allowed-channels` restricts every read to those channels (callers can only narrow further), and `add_memory` / `commit_memory` can only write to allowed channels (`commit_memory` checks the token's channel and group). **`read_cypher` is raw access that bypasses the ACL, so it is disabled by default when `--allowed-channels` is set** (enable it explicitly with `--allow-cypher`). `--read-only` opens the DB read-only and does not register `add_memory` / `commit_memory`. `streamable-http` has no authentication: keep the default `127.0.0.1`, or protect it with a reverse proxy.

Registering with an MCP client (Claude Code shown):

```bash
claude mcp add sqgraph-memory -- python -m sqgraph.memory.mcp --db /path/to/memory.db --group-id my-guild
# With a channel restriction (read_cypher is disabled)
claude mcp add sqgraph-memory-backend -- python -m sqgraph.memory.mcp --db /path/to/memory.db \
    --group-id my-guild --allowed-channels backend,infra
```

Example `.mcp.json` (`OPENAI_API_KEY` is read from the environment or from the `.env` given with `--env-file`, and only with `--extractor openai`):

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


## Curation (history is never erased)

When extraction is left to an agent or an LLM, the same thing can be registered twice under different names, and relations with the same meaning can multiply under different names. `forget` deletes, so separate operations exist for fixing this.

- **Relation vocabulary**: relation names are normalized to `UPPER_SNAKE_CASE`, and passive forms with a clear active counterpart (`OWNED_BY`, `MANAGED_BY`, `USED_BY`, `REVIEWED_BY`, ...) are stored in the active form (`OWNS`, ...) with subject and object swapped. The extracting agent / LLM is given the relation names already used in that DB, most frequent first, as `known_relations`, and is asked to reuse a matching one.
- **`merge_entities(keep, merge)`**: the names and aliases of `merge` become aliases of `keep`, and per-channel summaries, facts and mentions move over. Facts that become exactly identical after the merge (same subject, object, relation, text and validity) are collapsed into one, their sources are combined, and `superseded_by` / `supersedes` references are rewired. A fact between the two ("k8s is short for Kubernetes") becomes an attribute fact of `keep` (reported in `warnings`).
- **`invalidate_fact(fact_id, at=None, reason=None)`**: closes a fact by setting `invalid_at` (`ended_by: "manual"`, plus `invalid_reason`). Closing a plan that has not started closes it at its start time (that is, cancels it), so it never becomes valid. If a replacement fact exists, `contradicts` in `add_memory` does the same thing.
- **`rename_relation(old, new, swap=False, fact_ids=None)`**: unifies synonymous relation names (`HAS_REQUIREMENT` -> `REQUIRES`, and so on).
- **`review()`**: returns pairs of candidate duplicate entities (sharing an alias / one name containing the other / very close embeddings, with the reason), the list of relation names with unification suggestions (passive forms, spelling variants, names used only once), and facts still `planned` / `tentative` 14 days after their start. Deciding what to fix is up to people (and agents).
- **`upcoming(days=7)`**: facts that start or end soon. "End" means only explicit ends (such as an absence of 9/14-9/16), not ends caused by replacement with a newer fact or by manual invalidation.

## Export and backup

- **`export FILE`**: JSONL, one JSON object per line (`kind` is `header` / `entity` / `episode` / `mention` / `fact`). Times stay in the stored UTC, derived fields such as `alias_keys` are not written and are rebuilt on import. `--with-embeddings` also writes the vectors. The format is human-readable, so it doubles as a backup independent of the DB format.
- **`import FILE [--into-group G]`**: skips IDs that already exist (and episodes with the same `source` / `source_ref`), so importing repeatedly gives the same result. Vectors are reused if the embedding model is the same as at export time; otherwise they are recomputed with the current embedder (written in a single transaction after all embeddings succeed). Use it to move to another machine or to rebuild a DB.
- **`backup [DEST]`**: an SQLite online backup that produces one consistent file (no WAL) even during writes, and verifies it with `quick_check`. By default it writes to `~/.sqgraph-memory/backups/` and keeps the newest `--keep` files (default 10). For regular backups, call `sqgraph-memory backup` from the OS scheduler.

## Discord bot integration (overview)

In `on_message`, do `queue.put((message.content, channel.name, author.display_name, message.created_at, message.id))`, and have a worker call `mem.add_episode(..., source="discord", source_ref=str(message.id))` one message at a time (using the message ID as `source_ref` means re-submitting does not create duplicates). In an `/ask` slash command, determine the channels the asker can view from Discord permissions (`channel.permissions_for(member).view_channel`), pass that list as `channels=`, and call `mem.answer(question, group_id=str(guild.id), channels=allowed)`.

## Cost estimates (gpt-5.4-mini, text-embedding-3-small)

Prices (per 1M tokens): gpt-5.4-mini input $0.75 / cached input $0.075 / output $4.50, gpt-5.4-nano $0.20 / $0.02 / $1.25, text-embedding-3-small $0.02. `UsageTracker` aggregates by call type (`extract_entities` / `resolve_entities` / `extract_facts` / `dedupe_facts` / `answer` / `embed`).

Measured (one internal Discord channel, 100 Japanese messages averaging 93 characters, ingested one at a time in chronological order):

| Item | Measured |
|---|---|
| Ingestion total | about $0.39 (100 messages, 334 LLM calls + 169 embedding calls, about 5.7 minutes) |
| Per message | about $0.004 (about 3.3 LLM calls) -> **about $4 per 1,000 messages, about $40 per 10,000** (slightly higher as the DB grows and more related facts are retrieved) |
| Breakdown (ingestion) | fact extraction 44% / entity extraction 29% / contradiction detection 20% / entity resolution 7% / embeddings 0.1% |
| One question (`answer`) | about $0.002 (1 LLM call + 1 embedding call). `search` alone is 1 embedding call (practically $0) |

To reduce cost: try `gpt-5.4-nano` with `OpenAILLM("gpt-5.4-nano")`, skip short messages (`min_content_chars`) and the bot's own messages, and lower `context_window`.

## Limitations

- Ingestion is serial, one message at a time (each waits for several LLM calls). Large backlogs are expected to be run with `add_episodes_bulk` overnight.
- Exact-match entity lookup and name search scan the entities within a `group_id`, so beyond tens of thousands of entities an index is needed.
- Extraction quality depends on the LLM (wrong identifications, omissions). You can check with the `get_entity` timeline and the sources (`source_ref`), and fix mistakes with `merge_entities` / `invalidate_fact` / `rename_relation` ("Curation").
- Full-text search over facts uses trigrams, so terms of two characters or fewer are not indexed (entity names are matched through a separate path).
