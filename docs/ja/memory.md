[English](../memory.md)

# エージェント長期記憶 (sqgraph.memory)

社内エージェント (Discord bot、MCP 経由の Claude など) 向けの **時系列ナレッジグラフ型の長期記憶**です ([Graphiti / Zep](https://github.com/getzep/graphiti) に着想)。`sqgraph` 本体は LLM フリーのまま、`sqgraph.memory` パッケージが差し替え可能なプロバイダ (既定: OpenAI) を使って、メッセージからエンティティ・ファクトを抽出し、矛盾を検出して古いファクトを **失効 (削除しない)** させ、`as_of` 付きのハイブリッド検索を提供します。

```bash
pip install -e ".[memory]"      # openai
pip install -e ".[mcp]"         # MCP サーバー (mcp + openai)
```

## データモデル

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

## Python からの使い方 (bot 向け)

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

# 関連性ゲート: 質問と無関係なファクトは返さない。FactHit.similarity = 質問とのコサイン類似度 (ベクトルが無ければ None)。
# しきい値は Memory(min_similarity=...) か mem.search(..., min_similarity=0.5) で上書き (既定: 埋め込みモデルごと)
mem.search_entities("bob", channels=["backend"])            # [{id, name, type, aliases, summary, ...}]
mem.get_entity("bob", channels=["backend"])                 # プロフィール + ファクトのタイムライン (失効分含む)
mem.get_episodes(entity="Rails", channels=["backend"], limit=10)
ans = mem.answer("Rails の対応はどこまで進んだ？", group_id="my-guild", channels=["backend"])
print(ans.text)                                                # 既定は英語で、出典 [F1] と有効期間つきの回答 (日本語にするには下記の answer_prompt)
print(tracker.summary("run"))
```

`answer` は既定で英語で回答します。言語や文体を変えるには、システムプロンプトを差し替えます: `from sqgraph.memory.prompts import ANSWER_SYSTEM` のうえで `Memory(..., answer_prompt=ANSWER_SYSTEM.replace("in English", "in Japanese"))`。プロンプトには user メッセージとして次の JSON が渡されます: `now` (曜日つきのローカル時刻)、`as_of` (または null)、`question`、`entities` (name, type, summary。最大 4 件)、`facts` (各 `id` "F1".. / `fact` / `status` valid|ended|scheduled / `valid_at` / `invalid_at` / `modality` / `sources` [{author, at, channel}])。該当なしのときは LLM を呼ばず、`ans.text` は固定の英語 `No matching memories found.` になります (自前の文言を出したい場合は `not ans.hits` を確認してください)。

取り込み 1 件あたりの LLM 呼び出しは通常 2〜4 回です: ① エンティティ抽出 (直近の同チャンネル発言を文脈として渡し「それ」等を解決。覚えるべき事実が無い発言はここで打ち切り) → ② 既存エンティティとの同一判定 (完全一致・エイリアス一致は LLM 不要。曖昧な名前だけ、ベクトル / 全文 / 完全一致の候補をまとめて 1 回で判定) → ③ ファクト抽出 + エンティティ要約更新 → ④ 重複 / 矛盾判定 (関連する既存の有効ファクトがある時だけ)。埋め込みは 1 件につき 1〜2 回のバッチ呼び出しで、DB への書き込みは 1 トランザクションです。取り込みは内部ロックで直列化されます (bot ではキューに積む想定)。

### エージェント抽出 (サーバー側に LLM を持たない 2 段階モード)

`add_episode` は OpenAI などの LLM がサーバー側でエンティティ・ファクトを抽出します (bot 向け)。**呼び出し側のエージェント自身 (Claude など) に抽出させる**場合は、状態を持たない 2 段階の API を使います。LLM も OpenAI キーも要りません (埋め込みだけ設定した埋め込みエンドポイントを使います)。

```python
mem = Memory("memory.db", None, embedder, default_group_id="me")      # llm は不要

# 1 段階目: 書き込みなし・書き込みロックなし・LLM なし
prep = mem.prepare_episode("来週から pnpm を使ってほしい", source="claude-code", channel="personal", author="user",
                           occurred_at="2026-10-05 10:00", source_ref=None)
# prep["already_stored"] (同じ (source, source_ref) が取り込み済みなら True + episode_id)
# prep["token"]         自己完結した不透明な文字列 (グループ・出典・チャンネル・著者・時刻・本文を含む。サーバーは何も覚えない)
# prep["instructions"]  エージェントへの抽出ルール (短い)      prep["schema"]  extraction の JSON Schema
# prep["context"]       message / author_ref / recent / candidate_entities / related_facts (すべて LLM なしで計算)

# エージェントが context を読んで extraction を作る (下のスキーマ)
res = mem.commit_episode(prep["token"], extraction)                  # 書き込みロックの下で検証して保存
# res.facts_created / facts_duplicate / facts_invalidated / res.warnings (サーバーが捨てた・統合したもの)
```

`context`: `message` (本文・著者・ローカル時刻と曜日・チャンネル)、`author_ref` (著者はエージェントが `"author"` で参照できる。既存なら `existing_id`)、`recent` (同じチャンネルの直前の発言)、`candidate_entities` (本文に出てくる名前 / エイリアスの一致、trigram 全文、ベクトル検索、著者から選んだ既存エンティティ。最大 `max_entities` 件、各 `{id, name, type, aliases, summary}`)、`related_facts` (候補エンティティに触れる、メッセージ時刻に有効 / 予定の既存ファクトと、本文に近いファクト。最大 `max_facts` 件)。情報はそのチャンネルから見えるものだけです (上の ACL と同じ。`related_facts` は全て同じチャンネルのファクトなので `duplicate_of` に使えます)。

`extraction` のスキーマ (`prep["schema"]`):

```json
{"entities": [{"ref": "e1", "name": "pnpm", "type": "Service", "existing_id": null,
               "summary": "JS package manager the user prefers", "aliases": []}],
 "facts": [{"subject": "author", "object": "e1", "relation": "PREFERS",
            "fact": "ユーザーは npm ではなく pnpm を使ってほしい", "valid_at": null, "invalid_at": null,
            "modality": "definite", "confidence": 0.95, "duplicate_of": null, "contradicts": []}]}
```

`existing_id` は `candidate_entities` の id (同一物なら再利用)、`subject` / `object` は `ref` か `"author"` (属性ファクトは `object: null`)、`valid_at` が null ならメッセージ時刻、時刻は ISO 8601 (ローカル時刻 / `Z` 付き / 日付のみ)、`duplicate_of` は言い直しの既存ファクト id (同一チャンネルのみ)、`contradicts` はこの発言で成り立たなくなった有効なファクトの id (古い方が失効し `superseded_by` で結ばれる。意味は OpenAI パイプラインと同一です。内部で同じ `_commit` / `_write_fact` / `_invalidate` を使います)。要約は `summary` がそのエンティティのチャンネルの要約行になります。空の extraction (`{}`) は発言 (と著者の言及) だけを保存します。

`commit_episode` は `ValueError` で不正な箇所を**すべて**列挙します (未定義の ref、存在しない / 別グループの `existing_id`、`fact_` の形でない id、解釈できない時刻、不正な `modality` / `confidence` など)。2 段階の間に DB が変わっていてもよいように、既に存在しない `duplicate_of` / `contradicts`、別チャンネルのファクトへの `duplicate_of`、失効済みのファクトは捨てて `res.warnings` に記録します。`existing_id: null` の新規エンティティでも、決定的な完全一致 / エイリアス一致 (型が両立するもの) があれば既存エンティティに統合します (重複を防ぐ安全網)。トークンの改ざん・破損・切り詰めは `invalid token` として拒否されます。

プロバイダは `LLMProvider` (`complete_json` / `complete_text`) と `Embedder` (`embed`, `dimensions`) の Protocol です。実 API を使わない決定的な `FakeLLM` / `FakeEmbedder` が付属します (`tests/test_memory*.py` 参照)。

## アクセス制御 (チャンネル ACL)

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

## MCP サーバー

```bash
python -m sqgraph.memory.mcp --db memory.db [--group-id G] [--allowed-channels a,b] [--read-only] \
       [--extractor agent|openai] [--transport stdio|streamable-http] [--host 127.0.0.1] [--port 8000] [--env-file .env]
```

| ツール | 使いどころ |
|---|---|
| `search_memory` | まず最初に。人 / サービス / 決定 / 担当 / 状態 / 予定に関する質問。`as_of`, `include_invalid` (履歴), `center_entity`, `min_similarity` (関連性のしきい値)。関係のなさそうなファクトは返さず (無関係な質問は空)、各ファクトに質問とのコサイン類似度 `similarity` が付く |
| `search_entities` | 名前・エイリアスの解決、存在する人 / サービスの列挙 |
| `get_entity` | 1 エンティティの全履歴 (失効分含むタイムライン) と関連エンティティ |
| `get_episodes` | 元の発言の確認、前後の会話の確認 |
| `add_memory` | 決定・担当・状態変化・予定・好みなど長期的に覚えるべきことの記録 (`--read-only` では無効)。`--extractor agent` (既定) では**何も保存せず**、`token` / `instructions` / `schema` / `context` を返す。`openai` ではサーバー側の LLM で抽出して保存する |
| `commit_memory` | `agent` モードの 2 段階目。`add_memory` の `token` と、エージェントが作った `extraction` を渡して保存する (`openai` モードでも使える。`--read-only` では無効) |
| `forget_memory` | エピソード / ファクト / エンティティ / 期間を指定した削除 (`dry_run` で影響範囲の確認)。例: `forget_memory(episode_id="ep_...", dry_run=True)`。セレクタはちょうど 1 つ: `episode_id` (別名 `episode`) / `fact_id` (別名 `fact`) / `entity` (別名 `entity_id`, `name`) / `source_ref` (+ `source`) / `since`・`until`・`channel`。書き込み可能で `--allowed-channels` なしのときだけ登録される |
| `merge_entities` | 同じものを指す 2 つのエンティティを 1 つにまとめる (`keep` に `merge` を統合。名前・エイリアス・要約・ファクト・言及が移り、完全に同じファクトは 1 つにまとまる。履歴は消えない。`dry_run` あり) |
| `invalidate_fact` | もう成り立たないファクトを手で失効させる (`at` 既定は現在、`reason` を記録。始まる前の予定は取り消しになる。削除ではない) |
| `rename_relation` | 関係名をまとめて付け替える (`swap=true` で主語と目的語も入れ替え。`fact_ids` で一部だけ。`dry_run` あり) |
| `review_memory` | 整理の候補 (重複していそうなエンティティの組と理由、関係名の一覧と統一の提案、日付を過ぎた未確認の予定) を返す。変更はしない |
| `get_upcoming` | これから `days` 日以内 (と過去 `past_days` 日) に始まる / 終わるファクト (予定・不在・締切) |
| `get_schema` | ラベル / 関係タイプ / インデックス / 件数と Cypher サブセットの説明 |
| `read_cypher` | 読み取り専用 Cypher (行数 / 文字数 / 時間の制限つき)。上記で足りない集計・結合用 |

`merge_entities` / `invalidate_fact` / `rename_relation` は書き込み可能で `--allowed-channels` なしのときだけ、`review_memory` (全チャンネルを見る) は `--allowed-channels` なしのときだけ登録されます。`get_upcoming` はチャンネル制限に従います。

`--extractor` (既定 `agent`) は、`add_memory` のエンティティ・ファクト抽出を誰が行うかです。`agent` ではサーバーに LLM も OpenAI キーも要らず、呼び出したエージェント自身が `instructions` に従って `extraction` を作り `commit_memory` を呼びます (「エージェント抽出」参照。MCP の sampling には依存しません)。`openai` は従来どおり `add_memory` 1 回で OpenAI の LLM が抽出します (bot 向け。`OPENAI_API_KEY` が必要で、無いと `add_memory` はエラーになります)。

サーバー側で強制する設定: `--group-id` を指定すると名前空間が固定され (他の `group_id` は拒否)、`--allowed-channels` を指定すると全ての読み取りがそのチャンネルに制限され (呼び出し側は狭めることしかできない)、`add_memory` / `commit_memory` も許可チャンネルにしか書けません (`commit_memory` はトークンのチャンネルとグループを検査します)。**`read_cypher` は ACL を迂回する生アクセスなので、`--allowed-channels` 指定時は既定で無効**です (`--allow-cypher` で明示的に有効化できます)。`--read-only` では DB を読み取り専用で開き `add_memory` / `commit_memory` を登録しません。`streamable-http` は認証を持たないので、既定の `127.0.0.1` のまま使うか、リバースプロキシで保護してください。

Claude Code への登録:

```bash
claude mcp add sqgraph-memory -- python -m sqgraph.memory.mcp --db /path/to/memory.db --group-id my-guild
# チャンネル制限つき (read_cypher は無効になる)
claude mcp add sqgraph-memory-backend -- python -m sqgraph.memory.mcp --db /path/to/memory.db \
    --group-id my-guild --allowed-channels backend,infra
```

`.mcp.json` の例 (`--extractor openai` のときだけ `OPENAI_API_KEY` を環境変数、または `--env-file` の `.env` から読みます):

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


## 記憶の整理 (履歴は消さない)

抽出をエージェントや LLM に任せると、同じものが別名で 2 つ登録されたり、同じ意味の関係が別の名前で増えたりします。`forget` は削除なので、直すための操作を別に用意しています。

- **関係名の語彙**: 関係名は `UPPER_SNAKE_CASE` に正規化し、能動形がはっきりしている受動形 (`OWNED_BY`、`MANAGED_BY`、`USED_BY`、`REVIEWED_BY` など) は主語と目的語を入れ替えて能動形 (`OWNS` ...) で保存します。抽出するエージェント / LLM には、その DB で使われている関係名を多い順に `known_relations` として渡し、合うものがあれば再利用させます。
- **`merge_entities(keep, merge)`**: `merge` の名前とエイリアスは `keep` のエイリアスになり、チャンネルごとの要約・ファクト・言及が移ります。統合後に完全に同じになるファクト (主語・目的語・関係・文・有効かどうかが同じ) は 1 つにまとめ、出典を合わせ、`superseded_by` / `supersedes` の参照も付け替えます。2 つの間のファクト (「k8s は Kubernetes の略」) は `keep` の属性ファクトになります (`warnings` に出ます)。
- **`invalidate_fact(fact_id, at=None, reason=None)`**: `invalid_at` を入れて閉じます (`ended_by: "manual"`、`invalid_reason`)。まだ始まっていない予定を閉じると、始まる時刻で閉じる (= 取り消し) ので有効になることはありません。代わりの新しいファクトがあるときは、`add_memory` の `contradicts` を使えば同じことが起きます。
- **`rename_relation(old, new, swap=False, fact_ids=None)`**: 同義の関係名をまとめます (`HAS_REQUIREMENT` → `REQUIRES` など)。
- **`review()`**: 重複候補のエンティティの組 (エイリアスの共有 / 一方の名前が他方を含む / 埋め込みがとても近い。理由つき)、関係名の一覧と統一の提案 (受動形、綴りの揺れ、1 回しか使われていない名前)、開始から 14 日過ぎても `planned` / `tentative` のままのファクトを返します。何を直すかは人 (とエージェント) が決めます。
- **`upcoming(days=7)`**: これから始まる / 終わるファクト。「終わる」は明示された終わり (9/14-9/16 の不在など) だけで、新しいファクトに置き換えられた終わりや手動失効は含みません。

## エクスポートとバックアップ

- **`export FILE`**: 1 行 1 JSON の JSONL (`kind` が `header` / `entity` / `episode` / `mention` / `fact`)。時刻は保存している UTC のまま、`alias_keys` などの派生フィールドは書き出さず、読み込み時に作り直します。`--with-embeddings` でベクトルも書き出せます。人が読める形なので、DB の形式に依存しないバックアップにもなります。
- **`import FILE [--into-group G]`**: 既にある ID (と、同じ `source` / `source_ref` のエピソード) は飛ばすので、何度読み込んでも同じです。書き出したときと同じ埋め込みモデルならベクトルを再利用し、違えば今の埋め込みで計算し直します (全件の埋め込みが成功してから 1 トランザクションで書き込み)。別のマシンへの移行や、DB を作り直すときに使います。
- **`backup [DEST]`**: SQLite のオンラインバックアップで、書き込み中でも整合した 1 ファイル (WAL なし) を作り、`quick_check` で確認します。既定では `~/.sqgraph-memory/backups/` に作り、新しいものから `--keep` 個 (既定 10) を残します。定期的に取るなら OS のスケジューラから `sqgraph-memory backup` を呼んでください。

## Discord bot への組み込み (概要)

`on_message` で `queue.put((message.content, channel.name, author.display_name, message.created_at, message.id))` し、ワーカーが `mem.add_episode(..., source="discord", source_ref=str(message.id))` を 1 件ずつ呼びます (メッセージ ID を `source_ref` にすれば再投入しても重複しません)。`/ask` スラッシュコマンドでは、質問者が閲覧できるチャンネルを Discord の権限 (`channel.permissions_for(member).view_channel`) で判定し、その一覧を `channels=` に渡して `mem.answer(question, group_id=str(guild.id), channels=allowed)` を呼びます (英語以外のサーバーでは `Memory` を独自の `answer_prompt=` で作ります)。

## コスト目安 (gpt-5.4-mini, text-embedding-3-small)

価格 (1M トークンあたり): gpt-5.4-mini 入力 $0.75 / キャッシュ入力 $0.075 / 出力 $4.50、gpt-5.4-nano $0.20 / $0.02 / $1.25、text-embedding-3-small $0.02。`UsageTracker` が呼び出し種別 (`extract_entities` / `resolve_entities` / `extract_facts` / `dedupe_facts` / `answer` / `embed`) ごとに集計します。

実測 (社内 Discord 1 チャンネル、日本語 100 メッセージ、平均 93 文字を時系列順に 1 件ずつ取り込み):

| 項目 | 実測 |
|---|---|
| 取り込み合計 | 約 $0.39 (100 件、LLM 334 回 + 埋め込み 169 回、約 5.7 分) |
| 1 メッセージあたり | 約 $0.004 (LLM 約 3.3 回) → **1,000 件で約 $4、10,000 件で約 $40** (DB が育つと関連ファクトが増え、多少上振れ) |
| 内訳 (取り込み) | ファクト抽出 44% / エンティティ抽出 29% / 矛盾判定 20% / 同一判定 7% / 埋め込み 0.1% |
| 質問 (`answer`) 1 回 | 約 $0.002 (LLM 1 回 + 埋め込み 1 回)。`search` だけなら埋め込み 1 回 (ほぼ $0) |

コストを抑えるには、`gpt-5.4-nano` を `OpenAILLM("gpt-5.4-nano")` で試す、短い発言 (`min_content_chars`) や bot 自身の発言を取り込まない、`context_window` を小さくする、などが有効です。

## 制限事項

- 取り込みは 1 件ずつ直列です (LLM 呼び出し数回分の待ち時間)。大量の過去ログは `add_episodes_bulk` を夜間に流す想定です。
- エンティティの完全一致 / 名前検索は `group_id` 内のエンティティを走査するため、数万エンティティを超える規模では索引化が必要です。
- 抽出品質は LLM 依存です (誤った同一視 / 見落とし)。`get_entity` のタイムラインと出典 (`source_ref`) で確認できます。誤りは `merge_entities` / `invalidate_fact` / `rename_relation` で直せます (「記憶の整理」)。
- ファクトの全文検索は trigram なので 2 文字以下の語は索引にかかりません (エンティティ名は別経路で一致します)。
