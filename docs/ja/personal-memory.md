[English](../personal-memory.md)

# 個人メモリとして使う (CLI / MCP / Skill)

同じマシンのローカルエージェント (Claude Code / Codex CLI / OpenCode / Cursor / Claude Desktop) が **1 つの SQLite ファイルを共有する個人用の長期記憶**として使うための配布形態です。Python だけで入ります (Node / Docker は不要)。

**抽出は呼び出したエージェント自身が行います (既定の `extractor: "agent"`)**: 記憶するとき、エージェントは `add_memory` (CLI は `add`) で `instructions` と `context` (既存の候補エンティティ・関連する有効ファクト) を受け取り、自分で `extraction` JSON を作って `commit_memory` (CLI は `commit`) で保存します。**OpenAI キーは不要**で、サーバー側に LLM はありません (埋め込みだけ [LAN のサーバー](#埋め込みを-lan-内のサーバーで動かす)などに任せられます)。bot のように 1 回の呼び出しで OpenAI に抽出させたいときは `--extractor openai` を使います。

## インストールと初期設定

```bash
uv tool install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"      # または pipx install "sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph"

sqgraph-memory init                 # ~/.sqgraph-memory を作成 (抽出はエージェント: OpenAI キーは聞かれない)
sqgraph-memory init --embed-base-url http://192.168.1.20:1234/v1 --embed-model <モデル名>   # 埋め込みを LAN のサーバーで (下の「埋め込みを LAN 内のサーバーで動かす」)
#   OpenAI の LLM で抽出したいとき: sqgraph-memory init --extractor openai   (キーは環境変数 or 対話入力 (非表示))
#   非対話: echo "$OPENAI_API_KEY" | sqgraph-memory init --extractor openai --api-key-stdin
sqgraph-memory setup                # 検出したエージェントに MCP サーバーと Skill を登録
sqgraph-memory doctor [--online]    # 環境・DB・登録状況の診断 (失敗があれば終了コード 1)
```

`uvx sqgraph-memory ...` のような一時環境では登録するパスが消えることがあるため、`setup` は上の `uv tool install` を推奨する警告を出します。

## エージェントへの登録 (setup)

`setup` が書き込むもの (既存の設定ファイルは `.bak` にバックアップし、他のサーバー設定には触れません。何度実行しても同じ結果で、`--dry-run` で確認、`--uninstall` で元に戻せます):

| エージェント | MCP の登録先 | Skill |
|---|---|---|
| Claude Code | `claude mcp add --scope user sqgraph-memory -- <実行ファイル> mcp` (`claude` が PATH にないときは実行すべきコマンドを表示)。フックは `~/.claude/settings.json` ([フック](#フック)) | `~/.claude/skills/sqgraph-memory/SKILL.md` |
| Codex CLI | `~/.codex/config.toml` の `[mcp_servers.sqgraph-memory]` (`command` / `args`)。フックは `~/.codex/hooks.json` | `~/.agents/skills/sqgraph-memory/SKILL.md` |
| OpenCode | `~/.config/opencode/opencode.json` の `mcp.sqgraph-memory` (`type: local`)。OpenCode は同じディレクトリの `config.json` / `opencode.json` / `opencode.jsonc` をマージするので、コメント入りの `opencode.jsonc` は書き換えません。フックはプラグイン `~/.config/opencode/plugins/sqgraph-memory.js` | `~/.agents/skills/sqgraph-memory/SKILL.md` (Codex と共用。OpenCode はこの場所も読む) |
| Cursor | `~/.cursor/mcp.json` の `mcpServers` | `~/.cursor/skills/sqgraph-memory/SKILL.md` |
| Claude Desktop | `claude_desktop_config.json` の `mcpServers` (Windows `%APPDATA%\Claude`、macOS `~/Library/Application Support/Claude`、Linux `~/.config/Claude`) | なし |

`--agents claude-code,codex,opencode,cursor,claude-desktop|all` で対象を指定し (既定は検出できたもの)、`--command "uvx --from 'sqgraph[mcp] @ git+https://github.com/Fukuro99/sqgraph' sqgraph-memory mcp"` で起動コマンドを丸ごと差し替え、`--no-skill` で Skill を入れません。`--group G` で、選んだエージェント (既定は検出できたもの) に専用の記憶を割り当てます ([エージェントごとに記憶を分ける](#エージェントごとに記憶を分ける))。登録は常に `mcp` の後ろに `--agent <名前>` を付けます (`--command` で差し替えた場合もその後ろ)。登録後はエージェントを再起動してください。Skill (`SKILL.md`) は、いつ検索し、いつ覚えさせ、いつ忘れるか (明示的に頼まれたときだけ、`dry_run` で確認してから) をエージェントに教えます。

## フック

Claude Code / Codex / OpenCode 用のフックです (`setup` が登録。`--hooks session-start,user-prompt` で選択、`--no-hooks` で触らない、`--uninstall` で削除。自分以外のフックはそのまま残します): エージェントが検索しようと思わなくても、関係のある記憶が会話に入ります。

| フック | Claude Code のイベント | 入れるもの |
|---|---|---|
| `sqgraph-memory hook session-start` | `SessionStart` (起動 / 再開 / `/clear` / コンパクション後) | ユーザー自身についての今有効なファクト (好み・指示)、作業ディレクトリ (git リポジトリ名・フォルダ名) に関係するファクト、7 日以内に始まる / 終わる予定 |
| `sqgraph-memory hook user-prompt` | `UserPromptSubmit` | プロンプトで検索して関連性ゲートを通ったファクト (同じセッションでは同じファクトを 2 回出さない)。「覚えて」「remember this」があれば保存を促す一文 |

- **Claude Code**: `~/.claude/settings.json` の `hooks`。
- **Codex**: `~/.codex/hooks.json` (形式は Claude Code と同じで、コマンドに `--agent codex` が付く)。Codex は新しいフックを信頼するまで実行しないので、登録後に Codex で `/hooks` を開いて sqgraph-memory の 2 つを信頼してください。
- **OpenCode**: プラグイン `~/.config/opencode/plugins/sqgraph-memory.js` が同じコマンド (`--agent opencode`) を呼びます。セッション開始時の内容はセッションごとに 1 回だけ取得して、システムプロンプトの最後のテキストに付け足します (システムメッセージを 2 つにすると Qwen などのチャットテンプレートが拒否するため。セッション中は変わらないのでプロンプトキャッシュも効きます)。プロンプトごとの記憶は、そのユーザーメッセージに合成 (`synthetic`) のテキストパートとして付けて保存するので、会話履歴は後から変わりません。

フックは失敗してもユーザーの操作を止めません (エラーは `~/.sqgraph-memory/hooks.log`)。埋め込みの呼び出しは 3 秒で打ち切り、埋め込みサーバーが落ちていればキーワードとエンティティ名だけで検索します。1 回あたり約 1 秒 (Windows、ローカル DB、LAN の埋め込みサーバー) です。セッションごとの表示済みファクトは `~/.sqgraph-memory/hook-state/` に置き、7 日で消します。

## CLI

出力は常に UTF-8 の JSON、エラーは stderr に `{"error": ...}` で非ゼロ終了。`--compact` で 1 行):

| コマンド | 内容 |
|---|---|
| `init [--extractor agent\|openai] [--api-key-stdin] [--group-id G] [--tz TZ] [--embed-base-url URL] [--embed-model M] [--embed-dimensions N] [--no-probe]` | ホーム・設定を作る (キーは引数で渡せません。OpenAI キーを聞くのは `--extractor openai` のときだけ)。`--embed-base-url` で埋め込みを別サーバーにできます ([埋め込みを LAN 内のサーバーで動かす](#埋め込みを-lan-内のサーバーで動かす)) |
| `add TEXT\|- [--source claude-code] [--author user] [--channel personal] [--occurred-at T] [--source-ref R]` | 覚える。`agent` モード (既定) は何も保存せず、`token` / `instructions` / `schema` / `context` の JSON を出力 (`-` は stdin)。`openai` モードは LLM による抽出まで行って保存 (数秒) |
| `commit [--token T] [--extraction FILE\|-]` | `agent` モードの 2 段階目。`add` の `token` と、エージェントが作った extraction JSON (stdin か `--extraction FILE`。`{"token": ..., "extraction": {...}}` を 1 つの JSON で stdin に渡してもよい) で保存し、作成 / 重複 / 失効の件数と `warnings` を返す |
| `search QUERY [--limit N] [--as-of T] [--history] [--center ENTITY] [--min-similarity X]` | ファクト検索 (予定のファクトも含む。`--history` で失効分も)。関連性のしきい値は `--min-similarity` で一時的に変えられる |
| `entity NAME` / `entities QUERY` / `episodes [--entity E] [--query Q]` / `stats` | プロファイルと履歴 / 名前解決 / 元のメモ / 件数 |
| `groups` | DB 内の group (エピソード / エンティティ / 有効なファクトの件数)、既定の group、各 group で動くエージェント (割り当てのないエージェントは `(default)`)。読み取り専用 |
| `forget (--episode ID \| --source-ref REF [--source S] \| --fact ID \| --entity NAME [--keep-episodes] \| --since/--until/--channel [--source S]) [--dry-run]` | 削除 (対象の種類はちょうど 1 つ。`--episode-id` / `--fact-id` / `--entity-id` は別名) |
| `purge --yes [--dry-run]` | 使用中の group (既定の group、または `--agent` の group) の全記憶を削除 |
| `merge KEEP MERGE [--dry-run]` / `invalidate FACT_ID [--at T] [--reason R]` | エンティティの統合 / ファクトの手動失効 (どちらも履歴は残る。[記憶の整理](memory.md#記憶の整理-履歴は消さない)) |
| `relations` / `rename-relation OLD NEW [--swap] [--fact ID] [--dry-run]` | 関係名の一覧 (件数・例文) / 付け替え |
| `review [--limit N]` | 整理の候補 (MCP の `review_memory` と同じ) |
| `upcoming [--days 7] [--past-days 0]` | これから始まる / 終わるファクト |
| `export FILE\|- [--with-embeddings] [--force]` / `import FILE\|- [--into-group G] [--dry-run]` | JSONL への書き出し / 読み込み ([エクスポートとバックアップ](memory.md#エクスポートとバックアップ)) |
| `backup [DEST] [--keep 10]` | DB ファイルのスナップショット (既定 `~/.sqgraph-memory/backups/memory-YYYYmmdd-HHMMSS.db`、新しい 10 個を残す) |
| `import-claude-memory [--limit 5] [--dry-run]` | Claude Code のファイル型メモリ (`~/.claude/projects/*/memory/*.md`) を 1 ファイル 1 エピソードとして取り込む。`agent` モードでは `--limit` 件ずつ `add` と同じ token / context を返すので、エージェントが `commit` し、0 件になるまで繰り返す (内容が同じファイルは 2 度取り込まない) |
| `hook session-start\|user-prompt [--agent A]` | フック本体 (`setup` が登録。標準入力にイベントの JSON)。`--agent` (既定 `claude-code`) でそのエージェントの group を選ぶ |
| `config [--extractor agent\|openai] [--embed-base-url URL\|openai] [--embed-model M] [--embed-dimensions N] [--llm-model M] [--group-id G] [--agent-group AGENT=GROUP] [--tz TZ] [--embed-api-key-env NAME] [--search-min-similarity X\|default] [--no-probe]` | 設定を表示 (秘密は出しません) / 変更。`--agent-group` (複数可) でエージェント専用の group を割り当て (`AGENT=` か `AGENT=default` で外す)。埋め込みの設定を変えるとエンドポイントを確認し、DB の埋め込みと合わないときは `reembed` を促します |
| `reembed [--yes] [--batch-size N] [--progress]` | すべてのエンティティとファクトの埋め込みを、設定中の埋め込みモデルで作り直す (`--yes` なしは計画の表示だけ) |
| `mcp [--transport stdio\|streamable-http] [--read-only] [--extractor agent\|openai] [--agent A]` | MCP サーバー (エージェントが起動する。設定ファイルの DB / group / extractor / モデル / 埋め込みエンドポイントを自動で渡す。ツール一覧は [MCP サーバー](memory.md#mcp-サーバー)) |
| `setup [--agents LIST] [--group G] ...` / `doctor [--online]` | 上記 (`doctor` はエージェント → group の割り当て、extractor、埋め込みの接続先・モデル・次元と、DB に記録された埋め込みとの一致も報告。`agent` モードでは OpenAI キーの確認は `skip`。`--online` で実際に 1 回呼びます) |

記憶を扱うコマンド (`add` `commit` `search` `entity` `entities` `episodes` `stats` `forget` `purge` `merge` `invalidate` `relations` `rename-relation` `review` `upcoming` `export` `import` `import-claude-memory`) は `--agent NAME` も受け付け、そのエージェントに割り当てた group で動きます (割り当てがない、または省略したときは既定の group)。

`python -m sqgraph.memory` でも同じです。既定の `agent` モードでは OpenAI の LLM を一切使わないので、キーなしで `add` / `commit` / `search` / `entity` / `episodes` / `stats` / `forget` がすべて動きます (OpenAI の埋め込みを使う場合と、埋め込みエンドポイントが無い場合は、ベクトル検索だけが無効になり `warning` が付きます。LAN の埋め込みサーバーなら キーなしでベクトル検索も動きます)。LLM のキーが要るのは `--extractor openai` の `add` だけです。

## add → commit の流れ

CLI での 2 段階の保存です ([エージェント抽出](memory.md#エージェント抽出-サーバー側に-llm-を持たない-2-段階モード))。

```bash
sqgraph-memory add --source claude-code "来週から Osaka に引っ越します。pnpm を使ってほしい" > prep.json
#   prep.json の instructions と context を読んで、extraction.json を作る (エージェントの仕事)
sqgraph-memory commit --token "$(jq -r .token prep.json)" --extraction extraction.json
```

`token` は自己完結しているので、別のプロセスから `commit` しても動きます。同じ `--source-ref` を 2 回 `add` / `commit` しても、保存されるのは 1 回だけです。

## 1 ファイルを全エージェントで共有

全員が `~/.sqgraph-memory/memory.db` を開き、書き込みは `memory.db.lock` のクロスプロセスのファイルロックで直列化されます (複数のエージェントが同時に `add` しても壊れず、待ち時間は `lock_timeout` まで)。あるエージェントが覚えたことは他のエージェントからも検索できます。MCP サーバーは `--group-id` をそのエージェントの group (設定ファイルの `group_id`。専用の割り当てがあればそちら: [エージェントごとに記憶を分ける](#エージェントごとに記憶を分ける)) に固定して起動します。

## エージェントごとに記憶を分ける

既定では全エージェントが 1 つの記憶 (group `me`) を共有します。あるエージェントだけの記憶にしたいときは、そのエージェントに group を割り当てます。group は同じ `memory.db` の中にあり (ロックもバックアップも 1 つ)、完全に分離されます: 検索・エンティティ・`forget`・`purge` は group をまたがず、エージェントの MCP サーバーから別の group には届きません。

```bash
sqgraph-memory setup --agents codex --group work      # 割り当てて登録し直す (--group default で割り当てを外す)
sqgraph-memory config --agent-group codex=work        # 割り当てだけ書き換える (AGENT= / AGENT=default で外す)
sqgraph-memory groups                                 # DB 内の group と件数、どのエージェントがどの group か
```

その後エージェントを再起動してください。`setup` は各エージェントの MCP サーバーとフックを `--agent <名前>` 付きで登録し、サーバーは起動時にそのエージェントの group を選びます。group は `SQGRAPH_MEMORY_GROUP` (設定されていれば全エージェントでこれが優先) > `config.json` の `agent_groups[エージェント]` > `group_id` の順で決まります。割り当てのないエージェントは引き続き `group_id` を共有します。

- 既存の記憶は既定の group に残ります。新しい group にコピーするには `sqgraph-memory export all.jsonl` のあと `sqgraph-memory import all.jsonl --into-group work`。
- CLI では `--agent NAME` を `add` と `commit` の両方に (token は同じ group でしか commit できません)、ほかの記憶コマンドにも付けます。付けないと既定の group です。Skill がエージェントにそう指示します。
- group ID は `[A-Za-z0-9][A-Za-z0-9_.:-]*` (64 文字まで)、エージェント名は `[a-z0-9][a-z0-9_-]*`。`default` は割り当てを外すための予約語です。

## 忘れる

`forget` / `purge` / MCP の `forget_memory` は、DB から対象のエピソード・ファクト・孤立したエンティティを消し、残ったエンティティの要約と埋め込みを再計算します (削除されたファクトに失効させられていたファクトは有効に戻ります)。削除後は SQLite の `secure_delete` と WAL チェックポイントで、消した本文が DB ファイルにそのまま残らないようにします。`--dry-run` で何が消えるか確認できます。次のものは対象外です: DB ファイルの外にあるコピー (バックアップ、クラウド同期、Litestream など)、エピソードを取り込むときに OpenAI API へ送った内容 (`--extractor openai` のとき。OpenAI のポリシーに従って保持されます)、および呼び出したエージェント (LLM) が `add` の結果として読んだ内容 (そのエージェントの会話ログ)。

## 検索の関連性ゲート

RRF の順位だけでは、小さな記憶だと無関係な質問にも全ファクトが返ってしまいます。そこで `search` / `answer` (と MCP の `search_memory`) は、候補ごとに証拠を確かめ、証拠のあるものだけを RRF で並べて返します。各候補の質問とのコサイン類似度 (`similarity`、ベクトルが無ければ null) は結果に付きます。残る条件 (どれか 1 つ): (a) 全文検索ヒット (ただし「ユーザー」「user」など全ファクトに出る語や、ファクトの半数以上に出る語はヒットと見なさない)、(b) `similarity >= min_similarity`、(c) 質問がエンティティ名・エイリアスをそのまま含むエンティティ (または `center_entity`) の 1 ホップのファクト (ユーザー本人のように全ファクトの半数以上を占めるハブや「ユーザー」という語での一致は弱い一致として扱う)、(d) 弱い一致 (全文の断片 / エンティティのベクトル) の 1 ホップ、または名指しされたエンティティの 2 ホップのファクトは `similarity >= min_similarity - 0.05`。さらに (b) / (d) だけで残った候補は、最良の候補の類似度より 0.08 以上低ければ落とします。埋め込みが使えないとき (キー / エンドポイント無し、サーバー停止中) は (a)、(c)、弱い一致の 1 ホップだけになり、全件は返しません。

`min_similarity` の既定値は DB に記録された埋め込みモデルごとです (`text-embedding-3-small` 0.34、`-3-large` 0.32、`nomic-embed-text` (LM Studio の `text-embedding-nomic-embed-text-v1.5` を含む) 0.72、`qwen3-embedding` (LM Studio の `text-embedding-qwen3-embedding-0.6b` を含む) 0.46、`bge-m3` 0.45、`ada-002` 0.78、不明なモデル 0.30)。モデルによって類似度の尺度が大きく違うためです (nomic はタスク接頭辞なしだと無関係な文でも 0.5〜0.7 が出ます)。日本語の記憶では nomic は関連 / 無関係の類似度がほぼ重なるため、**Qwen3-Embedding (0.6B で十分) を推奨**します (日本語の個人メモ 18 件・質問 34 件の実測で、再現率 0.94 / 適合率 0.65、nomic は 0.61 / 0.24)。`config --search-min-similarity X` で固定 (`default` で既定に戻す)、`search --min-similarity X` / MCP の `min_similarity` で 1 回だけ上書きできます。結果が空なのに記憶があるはずなら、別のキーワードや名前で検索するか、`min_similarity` を下げてください。`prepare_episode` (エージェント抽出) の `related_facts` / `candidate_entities` は、矛盾の見落としを避けるため検索より緩く (しきい値 −0.10 / −0.05、周辺の上位 6 件は常に) 選びます。

## データの場所と権限

既定は `~/.sqgraph-memory/` (`SQGRAPH_MEMORY_HOME` で変更)。中身は `memory.db`、`.env` (`OPENAI_API_KEY=...`。`extractor: "openai"` か OpenAI の埋め込みを使うときだけ)、`config.json` (`group_id` 既定 `me` (割り当てのない全エージェントの group)、`agent_groups` 既定 `{}` (例 `{"codex": "work"}`、[エージェントごとに記憶を分ける](#エージェントごとに記憶を分ける))、`tz` 既定 `Asia/Tokyo`、`extractor` 既定 `agent` (`agent` / `openai`。フィールドの無い既存の設定も `agent` として扱う)、`llm_model` (`openai` モード用)、`embed_model`、`embed_base_url` 既定 null = OpenAI API、`embed_dimensions` 既定 null = OpenAI なら 1536 / カスタムなら自動検出して保存、`embed_api_key_env` 既定 `SQGRAPH_EMBED_API_KEY`、`search_min_similarity` 既定 null = 埋め込みモデルごとの既定値 ([検索の関連性ゲート](#検索の関連性ゲート))、`db`) です。POSIX ではディレクトリを 0700、ファイルを 0600 で作ります (Windows ではユーザープロファイルの ACL に任せます)。`SQGRAPH_MEMORY_DB` / `SQGRAPH_MEMORY_GROUP` / `SQGRAPH_MEMORY_EXTRACTOR` で DB・group・extractor を上書きできます。`setup` 実行時にこれら (と `SQGRAPH_MEMORY_HOME`) が設定されていれば、パスは絶対パスにして各エージェントの登録 (`env`) に引き継ぎます (`--dry-run` の出力にも表示)。API キーの値は出力にもログにも出しません。エージェントは MCP サーバーをシェルの環境変数なしで起動するため、キーは `.env` から読みます。

## 埋め込みを LAN 内のサーバーで動かす

抽出を呼び出したエージェントに任せる既定の `agent` モードなら、**埋め込みを LAN 内の PC** (OpenAI 互換の `/v1/embeddings` を話すサーバー: Ollama / LM Studio / llama.cpp server / text-embeddings-inference / vLLM / Infinity など) に任せることで、**OpenAI を一切使わず** (キーもネットワーク送信もなく) に運用できます。検索 (`search` / `entity` / MCP の読み取りツール) も `add` / `commit` も OpenAI のキーなしで動きます。`--extractor openai` の場合だけ、メモの本文が OpenAI に送られるのは `add` の LLM 抽出になります。

```bash
# Ollama (サーバー側で: ollama pull bge-m3  /  OLLAMA_HOST=0.0.0.0 ollama serve)
sqgraph-memory init --embed-base-url http://192.168.1.20:11434/v1 --embed-model bge-m3
#   または nomic-embed-text など。URL は http://<host>:11434 でも可 (/v1 を補います)

# LM Studio (Local Server を起動し "Serve on Local Network" を有効に。モデル名は LM Studio に表示される識別子)
sqgraph-memory init --embed-base-url http://192.168.1.20:1234/v1 --embed-model text-embedding-qwen3-embedding-0.6b

# llama.cpp server  (llama-server -m bge-m3.gguf --embedding --host 0.0.0.0 --port 8080)
sqgraph-memory init --embed-base-url http://192.168.1.20:8080/v1 --embed-model bge-m3
# text-embeddings-inference  (--model-id BAAI/bge-m3 --port 8080)
sqgraph-memory init --embed-base-url http://192.168.1.20:8080/v1 --embed-model BAAI/bge-m3

sqgraph-memory config               # 設定の確認 (秘密は出ません)
sqgraph-memory doctor --online      # エンドポイントに 1 回問い合わせて次元と DB との一致を確認
```

`init` / `config` は設定の保存前にエンドポイントへ 1 回だけ埋め込みを依頼し、モデルの次元数 (`embed_dimensions`) を自動検出して `config.json` に保存します (到達できなければエラー。サーバーがまだ起動していないときは `--no-probe` で確認なしに保存し、`--embed-dimensions N` を明示できます)。設定は `setup` で登録した各エージェントの MCP サーバーにも `mcp` が自動で渡します (`--embed-base-url` / `--embed-dimensions` / `--embed-api-key-env`)。

- **サーバーが LAN のインターフェースで待ち受けていること**: 既定は `127.0.0.1` のみのことが多く、Ollama は `OLLAMA_HOST=0.0.0.0`、llama.cpp は `--host 0.0.0.0` が必要です。ファイアウォールでそのポート (Ollama 11434 / LM Studio 1234 など) を LAN からだけ開けてください。
- **API キー**: 多くのローカルサーバーはキー不要で、その場合ダミー値を送ります (`OPENAI_API_KEY` がこのエンドポイントに送られることはありません)。認証付きのサーバーでは環境変数 `SQGRAPH_EMBED_API_KEY` (名前は `--embed-api-key-env` で変更可) に入れて `init` を実行するか、`~/.sqgraph-memory/.env` に `SQGRAPH_EMBED_API_KEY=...` と書いてください。
- **モデルを切り替えたら `reembed` が必要**: 次元数が同じでも別モデルのベクトルは比べられません。DB には使った埋め込みモデル・次元・エンドポイントが記録され、設定と合わないまま開こうとするとエラーで `sqgraph-memory reembed` を案内します (`config` / `doctor` も警告します)。`sqgraph-memory reembed` で計画 (件数) を確認し、`--yes` で全エンティティ・全ファクトを現在の設定で作り直します (次元数が変わるときはベクトルインデックスも作り直します。全件の埋め込みに成功してから 1 トランザクションで反映するので、サーバーが落ちても DB は元のままです。実行中は書き込みロックを保持します)。エンドポイントの URL だけを変える (同じモデル・同じ次元) 場合は `reembed` は不要です。
- **通信は平文の HTTP です**: メモの本文 (エンティティ名・ファクト文・検索クエリ) がそのままそのサーバーに送られます。信頼できる LAN の中だけで使い、インターネットには公開しないでください (リバースプロキシで TLS を付けるなら `https://` の URL も指定できます)。サーバー側のログにメモが残る可能性にも注意してください。
- サーバーが落ちている間も `search` は数秒 (接続 5 秒 / 全体 30 秒、リトライ 1 回) でベクトルなしの検索 (全文 + エンティティ) に切り替わります。`add` / `commit` / `reembed` はエラーになります。`add` のバッチは 32 件ずつ送ります。
