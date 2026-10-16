[日本語](ja/personal-memory.md)

# Personal memory (CLI / MCP / Skill)

A distribution for local agents on one machine (Claude Code / Codex CLI / OpenCode / Cursor / Claude Desktop) to use **a single shared SQLite file as personal long-term memory**. It installs with Python alone; no Node or Docker.

**The calling agent does the extraction itself (the default, `extractor: "agent"`).** To remember something, the agent calls `add_memory` (CLI: `add`) and receives `instructions` and `context` (existing candidate entities and related valid facts). It builds an `extraction` JSON itself and stores it with `commit_memory` (CLI: `commit`). **No OpenAI key is needed**, and the server has no LLM. Only embeddings can be delegated to a [server on your LAN](#embeddings-on-a-lan-server) or similar. To have OpenAI do the extraction in a single call, as a bot would, use `--extractor openai`.

## Install and initial setup

```bash
uv tool install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"      # or: pipx install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"

sqgraph-memory init                 # create ~/.sqgraph-memory (extraction is done by the agent: no OpenAI key prompt)
sqgraph-memory init --embed-base-url http://192.168.1.20:1234/v1 --embed-model <model-name>   # embeddings on a LAN server (see "Embeddings on a LAN server" below)
#   To extract with an OpenAI LLM: sqgraph-memory init --extractor openai   (key from the environment or an interactive hidden prompt)
#   Non-interactive: echo "$OPENAI_API_KEY" | sqgraph-memory init --extractor openai --api-key-stdin
sqgraph-memory setup                # register the MCP server and Skill with detected agents
sqgraph-memory doctor [--online]    # diagnose environment, DB and registrations (exit code 1 on any failure)
```

In a throwaway environment such as `uvx sqgraph-memory ...`, the registered path can disappear, so `setup` warns and recommends `uv tool install` as above.

## Registering with agents (setup)

What `setup` writes. Existing config files are backed up to `.bak` and other servers' settings are left alone. It is idempotent; use `--dry-run` to preview and `--uninstall` to revert.

| Agent | Where the MCP server is registered | Skill |
|---|---|---|
| Claude Code | `claude mcp add --scope user sqgraph-memory -- <executable> mcp` (if `claude` is not on PATH, it prints the command to run). Hooks go in `~/.claude/settings.json` ([Hooks](#hooks)) | `~/.claude/skills/sqgraph-memory/SKILL.md` |
| Codex CLI | `[mcp_servers.sqgraph-memory]` (`command` / `args`) in `~/.codex/config.toml`. Hooks go in `~/.codex/hooks.json` | `~/.agents/skills/sqgraph-memory/SKILL.md` |
| OpenCode | `mcp.sqgraph-memory` (`type: local`) in `~/.config/opencode/opencode.json`. OpenCode merges `config.json` / `opencode.json` / `opencode.jsonc` in the same directory, so a commented `opencode.jsonc` is never rewritten. Hooks are the plugin `~/.config/opencode/plugins/sqgraph-memory.js` | `~/.agents/skills/sqgraph-memory/SKILL.md` (shared with Codex; OpenCode also reads this location) |
| Cursor | `mcpServers` in `~/.cursor/mcp.json` | `~/.cursor/skills/sqgraph-memory/SKILL.md` |
| Claude Desktop | `mcpServers` in `claude_desktop_config.json` (Windows `%APPDATA%\Claude`, macOS `~/Library/Application Support/Claude`, Linux `~/.config/Claude`) | none |

`--agents claude-code,codex,opencode,cursor,claude-desktop|all` selects the targets (default: whatever is detected). `--command "uvx --from 'sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph' sqgraph-memory mcp"` replaces the whole launch command, and `--no-skill` skips the Skill. `--group G` gives the selected agents (default: the detected ones) a memory of their own ([Separate memories per agent](#separate-memories-per-agent)); every registration passes `--agent <name>` after `mcp` (also after a `--command` override). Restart the agents after registering. The Skill (`SKILL.md`) tells the agent when to search, when to remember, and when to forget (only when explicitly asked, after checking with `dry_run`).

## Hooks

Hooks for Claude Code / Codex / OpenCode, registered by `setup` (`--hooks session-start,user-prompt` selects which, `--no-hooks` leaves hooks alone, `--uninstall` removes them; hooks that are not ours are kept). Relevant memories enter the conversation even when the agent does not think to search.

| Hook | Claude Code event | Injects |
|---|---|---|
| `sqgraph-memory hook session-start` | `SessionStart` (startup / resume / `/clear` / after compaction) | Currently valid facts about the user (preferences, instructions), facts related to the working directory (git repository name, folder name), and plans starting or ending within 7 days |
| `sqgraph-memory hook user-prompt` | `UserPromptSubmit` | Facts found by searching the prompt that pass the [relevance gate](#search-relevance-gate) (the same fact is never shown twice in a session). If the prompt contains "覚えて" or "remember this", a one-line nudge to save it |

- **Claude Code**: `hooks` in `~/.claude/settings.json`.
- **Codex**: `~/.codex/hooks.json` (same format as Claude Code, with `--agent codex` added to the command). Codex does not run new hooks until you trust them, so after registering, open `/hooks` in Codex and trust the two sqgraph-memory hooks.
- **OpenCode**: the plugin `~/.config/opencode/plugins/sqgraph-memory.js` calls the same commands (`--agent opencode`). The session-start content is fetched once per session and appended to the last text of the system prompt (two system messages would be rejected by chat templates such as Qwen's; it also doesn't change during the session, so the prompt cache stays effective). The per-prompt memories are attached to that user message as a synthetic text part and saved with it, so the conversation history never changes afterwards.

A failing hook never blocks the user (errors go to `~/.sqgraph-memory/hooks.log`). Embedding calls time out after 3 seconds; if the embedding server is down, the search falls back to keywords and entity names only. A run takes about 1 second (Windows, local DB, LAN embedding server). The facts already shown per session are kept in `~/.sqgraph-memory/hook-state/` and expire after 7 days.

## CLI

Output is always UTF-8 JSON. Errors go to stderr as `{"error": ...}` with a non-zero exit code. `--compact` prints a single line.

| Command | Description |
|---|---|
| `init [--extractor agent\|openai] [--api-key-stdin] [--group-id G] [--tz TZ] [--embed-base-url URL] [--embed-model M] [--embed-dimensions N] [--no-probe]` | Create the home directory and config. The key cannot be passed as an argument; the OpenAI key is prompted for only with `--extractor openai`. `--embed-base-url` points embeddings at another server ([Embeddings on a LAN server](#embeddings-on-a-lan-server)) |
| `add TEXT\|- [--source claude-code] [--author user] [--channel personal] [--occurred-at T] [--source-ref R]` | Remember. In `agent` mode (default) nothing is stored; it prints JSON with `token` / `instructions` / `schema` / `context` (`-` reads stdin). In `openai` mode it runs the LLM extraction and stores the result (a few seconds) |
| `commit [--token T] [--extraction FILE\|-]` | Second step of `agent` mode. Stores using the `token` from `add` and the extraction JSON the agent built (via stdin or `--extraction FILE`; you may also pass `{"token": ..., "extraction": {...}}` as a single JSON on stdin). Returns counts of created / duplicate / invalidated facts, plus `warnings` |
| `search QUERY [--limit N] [--as-of T] [--history] [--center ENTITY] [--min-similarity X]` | Search facts (including plan facts; `--history` also returns invalidated ones). `--min-similarity` temporarily overrides the relevance threshold |
| `entity NAME` / `entities QUERY` / `episodes [--entity E] [--query Q]` / `stats` | Profile and history / name resolution / source notes / counts |
| `groups` | The groups in the DB (episodes / entities / valid facts), the default group, and the agents working in each (agents without an assignment are listed as `(default)`). Read-only |
| `forget (--episode ID \| --source-ref REF [--source S] \| --fact ID \| --entity NAME [--keep-episodes] \| --since/--until/--channel [--source S]) [--dry-run]` | Delete. Give exactly one kind of target (`--episode-id` / `--fact-id` / `--entity-id` are aliases) |
| `purge --yes [--dry-run]` | Delete all memory in the group in use (the default group, or the one of `--agent`) |
| `merge KEEP MERGE [--dry-run]` / `invalidate FACT_ID [--at T] [--reason R]` | Merge entities / manually invalidate a fact (history is kept either way; see [Curation](memory.md#curation-history-is-never-erased)) |
| `relations` / `rename-relation OLD NEW [--swap] [--fact ID] [--dry-run]` | List relation names (counts, examples) / rename them |
| `review [--limit N]` | Cleanup candidates (same as MCP `review_memory`) |
| `upcoming [--days 7] [--past-days 0]` | Facts that start or end soon |
| `export FILE\|- [--with-embeddings] [--force]` / `import FILE\|- [--into-group G] [--dry-run]` | Write to / read from JSONL ([Export and backup](memory.md#export-and-backup)) |
| `backup [DEST] [--keep 10]` | Snapshot of the DB file (default `~/.sqgraph-memory/backups/memory-YYYYmmdd-HHMMSS.db`; keeps the newest 10) |
| `import-claude-memory [--limit 5] [--dry-run]` | Import Claude Code's file-based memory (`~/.claude/projects/*/memory/*.md`), one file per episode. In `agent` mode it returns the same token / context as `add` for `--limit` files at a time; the agent `commit`s them and repeats until none are left (a file with identical content is never imported twice) |
| `hook session-start\|user-prompt [--agent A]` | The hook entry point (registered by `setup`; the event JSON arrives on stdin). `--agent` (default `claude-code`) selects the agent's group |
| `config [--extractor agent\|openai] [--embed-base-url URL\|openai] [--embed-model M] [--embed-dimensions N] [--llm-model M] [--group-id G] [--agent-group AGENT=GROUP] [--tz TZ] [--embed-api-key-env NAME] [--search-min-similarity X\|default] [--no-probe]` | Show (secrets are never printed) / change the config. `--agent-group` (repeatable) assigns an agent its own group (`AGENT=` or `AGENT=default` removes it). Changing the embedding settings probes the endpoint, and if they no longer match the embeddings in the DB it tells you to run `reembed` |
| `reembed [--yes] [--batch-size N] [--progress]` | Rebuild the embeddings of all entities and facts with the configured embedding model (without `--yes` it only prints the plan) |
| `mcp [--transport stdio\|streamable-http] [--read-only] [--extractor agent\|openai] [--agent A]` | MCP server (launched by the agent; the DB / group / extractor / models / embedding endpoint from the config file are passed automatically). For the tools, see [MCP server](memory.md#mcp-server) |
| `setup [--agents LIST] [--group G] ...` / `doctor [--online]` | See above. `doctor` also reports the agent → group assignments, the extractor, the embedding endpoint, model and dimensions, and whether they match what the DB recorded. In `agent` mode the OpenAI key check is `skip`; `--online` makes one real call |

Every command that works on memory (`add` `commit` `search` `entity` `entities` `episodes` `stats` `forget` `purge` `merge` `invalidate` `relations` `rename-relation` `review` `upcoming` `export` `import` `import-claude-memory`) also takes `--agent NAME`: it works in the group assigned to that agent, or in the default group when there is none or the option is omitted.

`python -m sqgraph.memory` works the same way. The default `agent` mode never uses an OpenAI LLM, so `add` / `commit` / `search` / `entity` / `episodes` / `stats` / `forget` all work without a key. (With OpenAI embeddings, or when no embedding endpoint is available, only vector search is disabled and a `warning` is attached; with a LAN embedding server, vector search also works without a key.) An LLM key is needed only for `add` with `--extractor openai`.

## The add → commit flow

The two-step save from the CLI ([agent-side extraction](memory.md#agent-extraction-two-phase-mode-without-a-server-side-llm)):

```bash
sqgraph-memory add --source claude-code "I am moving to Osaka next week. Please use pnpm." > prep.json
#   read instructions and context in prep.json and build extraction.json (the agent's job)
sqgraph-memory commit --token "$(jq -r .token prep.json)" --extraction extraction.json
```

The `token` is self-contained, so `commit` works from a different process. Running `add` / `commit` twice with the same `--source-ref` stores it only once.

## Sharing one file across all agents

Every agent opens `~/.sqgraph-memory/memory.db`, and writes are serialized by a cross-process file lock on `memory.db.lock` (several agents can `add` at once without corruption; waits are bounded by `lock_timeout`). Whatever one agent remembers, the others can search. The MCP server is launched with `--group-id` pinned to the group of its agent (the config file's `group_id`, unless the agent has its own: [Separate memories per agent](#separate-memories-per-agent)).

## Separate memories per agent

By default every agent shares one memory (group `me`). To give an agent a memory of its own, assign it a group. Groups live in the same `memory.db` (one lock, one backup) but are fully isolated: search, entities, `forget` and `purge` never cross a group boundary, and an agent's MCP server cannot reach another group.

```bash
sqgraph-memory setup --agents codex --group work      # assign and re-register (--group default removes the assignment)
sqgraph-memory config --agent-group codex=work        # or only edit the mapping (AGENT= / AGENT=default removes it)
sqgraph-memory groups                                 # groups in the DB with counts, and which agents work in each
```

Restart the agent afterwards: `setup` registers every agent's MCP server and hooks with `--agent <name>`, and the server picks the agent's group at start. The group is resolved as `SQGRAPH_MEMORY_GROUP` (if set, it wins for every agent) > `agent_groups[agent]` in `config.json` > `group_id`. Agents without an assignment keep sharing `group_id`.

- Existing memories stay in the default group. To copy them into a new group: `sqgraph-memory export all.jsonl`, then `sqgraph-memory import all.jsonl --into-group work`.
- On the CLI, pass `--agent NAME` to `add` **and** `commit` (a token only commits in its own group) and to every other memory command; without it the default group is used. The Skill tells agents to do this.
- Group ids are `[A-Za-z0-9][A-Za-z0-9_.:-]*` (up to 64 characters), agent names `[a-z0-9][a-z0-9_-]*`; `default` is reserved for removing an assignment.

## Forgetting

`forget` / `purge` / MCP `forget_memory` delete the target episodes, facts and orphaned entities from the DB, then recompute the summaries and embeddings of the remaining entities (facts that had been invalidated by a deleted fact become valid again). After deletion, SQLite `secure_delete` and a WAL checkpoint keep the deleted text from lingering in the DB file. `--dry-run` shows what would be deleted. These are out of scope: copies outside the DB file (backups, cloud sync, Litestream, and so on), content sent to the OpenAI API when an episode was ingested (with `--extractor openai`; retained under OpenAI's policy), and content the calling agent (LLM) read as the result of `add` (that agent's conversation log).

## Search relevance gate

With RRF rank alone, a small memory returns every fact even for an unrelated question. So `search` / `answer` (and MCP `search_memory`) check each candidate for evidence and return only those that have it, ordered by RRF. Each result carries its cosine similarity to the question (`similarity`; null when there are no vectors). A candidate survives if any one of these holds:

- (a) It is a full-text hit. Words that appear in every fact, such as "ユーザー" or "user", and words that appear in half or more of the facts do not count as hits.
- (b) `similarity >= min_similarity`.
- (c) It is a 1-hop fact of an entity whose name or alias appears verbatim in the question (or of `center_entity`). A hub that accounts for half or more of all facts, such as the user themself, and a match on the word "ユーザー" count as weak matches.
- (d) It is a 1-hop fact reached through a weak match (a full-text fragment or an entity vector), or a 2-hop fact of a named entity, with `similarity >= min_similarity - 0.05`.

Candidates that survive only through (b) / (d) are dropped if their similarity is 0.08 or more below the best candidate's. When embeddings are unavailable (no key / endpoint, or the server is down), only (a), (c) and the 1-hop weak matches apply, and the full set is never returned.

The default `min_similarity` depends on the embedding model recorded in the DB: `text-embedding-3-small` 0.34, `-3-large` 0.32, `nomic-embed-text` (including LM Studio's `text-embedding-nomic-embed-text-v1.5`) 0.72, `qwen3-embedding` (including LM Studio's `text-embedding-qwen3-embedding-0.6b`) 0.46, `bge-m3` 0.45, `ada-002` 0.78, unknown models 0.30. Models use very different similarity scales (without a task prefix, nomic gives 0.5–0.7 even for unrelated sentences). On Japanese memories, nomic's relevant and irrelevant similarities nearly overlap, so **Qwen3-Embedding (0.6B is enough) is recommended** (on 18 Japanese personal notes and 34 questions, recall 0.94 / precision 0.65, versus 0.61 / 0.24 for nomic). `config --search-min-similarity X` pins the value (`default` restores the default); `search --min-similarity X` or MCP `min_similarity` overrides it for one call. If a result is empty although the memory should exist, search with a different keyword or name, or lower `min_similarity`. `related_facts` / `candidate_entities` from `prepare_episode` (agent extraction) are selected more loosely than search (threshold −0.10 / −0.05, and the top 6 nearby are always included) so that contradictions are not missed.

## Data location and permissions

The default is `~/.sqgraph-memory/` (override with `SQGRAPH_MEMORY_HOME`). It contains:

- `memory.db`
- `.env` (`OPENAI_API_KEY=...`; only when using `extractor: "openai"` or OpenAI embeddings)
- `config.json`:
  - `group_id` (default `me`; the group of every agent without an assignment)
  - `agent_groups` (default `{}`; e.g. `{"codex": "work"}`, see [Separate memories per agent](#separate-memories-per-agent))
  - `tz` (default `Asia/Tokyo`)
  - `extractor` (default `agent`; `agent` / `openai`; an existing config without the field is treated as `agent`)
  - `llm_model` (for `openai` mode)
  - `embed_model`
  - `embed_base_url` (default null = OpenAI API)
  - `embed_dimensions` (default null = 1536 for OpenAI; for a custom endpoint, auto-detected and saved)
  - `embed_api_key_env` (default `SQGRAPH_EMBED_API_KEY`)
  - `search_min_similarity` (default null = the per-embedding-model default; see [Search relevance gate](#search-relevance-gate))
  - `db`

On POSIX, the directory is created with mode 0700 and files with 0600 (Windows relies on the user profile's ACL). `SQGRAPH_MEMORY_DB` / `SQGRAPH_MEMORY_GROUP` / `SQGRAPH_MEMORY_EXTRACTOR` override the DB, group and extractor. If these (or `SQGRAPH_MEMORY_HOME`) are set when `setup` runs, the paths are made absolute and carried into each agent's registration (`env`); `--dry-run` shows them too. API key values never appear in output or logs. Agents launch the MCP server without the shell's environment variables, so the key is read from `.env`.

## Embeddings on a LAN server

With the default `agent` mode, where the calling agent does the extraction, you can **delegate embeddings to a PC on your LAN** (any server that speaks OpenAI-compatible `/v1/embeddings`: Ollama / LM Studio / llama.cpp server / text-embeddings-inference / vLLM / Infinity, and so on) and run **without OpenAI at all**: no key, no network egress. Search (`search` / `entity` / MCP read tools) and `add` / `commit` all work without an OpenAI key. Only with `--extractor openai` is the note text sent to OpenAI, in the LLM extraction done by `add`.

```bash
# Ollama (on the server: ollama pull bge-m3  /  OLLAMA_HOST=0.0.0.0 ollama serve)
sqgraph-memory init --embed-base-url http://192.168.1.20:11434/v1 --embed-model bge-m3
#   or nomic-embed-text and others. The URL may also be http://<host>:11434 (/v1 is appended)

# LM Studio (start the Local Server and enable "Serve on Local Network". The model name is the identifier LM Studio shows)
sqgraph-memory init --embed-base-url http://192.168.1.20:1234/v1 --embed-model text-embedding-qwen3-embedding-0.6b

# llama.cpp server  (llama-server -m bge-m3.gguf --embedding --host 0.0.0.0 --port 8080)
sqgraph-memory init --embed-base-url http://192.168.1.20:8080/v1 --embed-model bge-m3
# text-embeddings-inference  (--model-id BAAI/bge-m3 --port 8080)
sqgraph-memory init --embed-base-url http://192.168.1.20:8080/v1 --embed-model BAAI/bge-m3

sqgraph-memory config               # show the settings (secrets are not printed)
sqgraph-memory doctor --online      # query the endpoint once and check the dimensions against the DB
```

Before saving, `init` / `config` request one embedding from the endpoint, detect the model's dimension (`embed_dimensions`) automatically and store it in `config.json`. If the endpoint is unreachable it is an error; when the server isn't up yet, `--no-probe` saves without checking and you can set `--embed-dimensions N` explicitly. `mcp` passes the settings on automatically to the MCP server of each agent registered by `setup` (`--embed-base-url` / `--embed-dimensions` / `--embed-api-key-env`).

- **The server must listen on the LAN interface.** The default is often `127.0.0.1` only: Ollama needs `OLLAMA_HOST=0.0.0.0` and llama.cpp needs `--host 0.0.0.0`. Open that port (Ollama 11434, LM Studio 1234, and so on) in the firewall to the LAN only.
- **API key.** Most local servers need none; a dummy value is sent (`OPENAI_API_KEY` is never sent to this endpoint). For a server with authentication, put the key in the environment variable `SQGRAPH_EMBED_API_KEY` (the name can be changed with `--embed-api-key-env`) when running `init`, or write `SQGRAPH_EMBED_API_KEY=...` in `~/.sqgraph-memory/.env`.
- **Switching models requires `reembed`.** Vectors from different models cannot be compared even when the dimensions match. The DB records the embedding model, dimensions and endpoint it was built with; opening it with a mismatched config fails with an error that points to `sqgraph-memory reembed` (`config` / `doctor` warn as well). Run `sqgraph-memory reembed` to see the plan (counts), then add `--yes` to rebuild all entities and facts with the current settings. If the dimension changes, the vector indexes are rebuilt too. Everything is applied in one transaction after all embeddings succeed, so the DB is unchanged if the server goes down; the write lock is held while it runs. If you only change the endpoint URL (same model, same dimensions), `reembed` is not needed.
- **Traffic is plain HTTP.** The note text (entity names, fact sentences, search queries) goes to that server as is. Use it only inside a trusted LAN and do not expose it to the internet (put TLS in front with a reverse proxy and you can specify an `https://` URL). Keep in mind that the server's logs may retain your notes.
- While the server is down, `search` falls back within seconds (5 s connect / 30 s total, one retry) to search without vectors (full text plus entities). `add` / `commit` / `reembed` fail with an error. `add` batches are sent 32 at a time.
