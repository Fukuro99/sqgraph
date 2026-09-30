# GraphRAG デモ (sqgraph + OpenAI)

架空の港湾都市「潮見市」を題材にした日本語コーパス (`corpus/`, 6 文書) から、OpenAI の LLM でナレッジグラフを構築し、
**sqgraph (SQLite + sqlite-vec + openCypher サブセット)** に保存して 3 種類の検索方式を比較するエンドツーエンドのデモです。
架空の世界なので、LLM の事前知識では答えられず、文書間のマルチホップ推論や全体要約が必要な質問で差が出ます。

## 構成

```
Document -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Entity -[:RELATED_TO {description, weight}]-> Entity
Entity -[:IN_COMMUNITY]-> Community {title, summary}
```

- `index`: チャンク分割 → `gpt-5.4-mini` でエンティティ / 関係を JSON 構造化抽出 → 正規化名で重複統合 (複数記述は LLM で要約)
  → `text-embedding-3-small` で埋め込み → sqgraph に保存 (ベクトル索引 `entity_emb` / `chunk_emb`、trigram 全文索引 `chunk_text`)
  → 純 Python の Louvain 法でコミュニティ検出 → 各コミュニティのレポート (タイトル + 要約) を LLM で生成
- `local`: Cypher でエンティティのベクトル検索 → 1〜2 ホップ展開 → 関係の説明・言及チャンク・所属コミュニティを収集 → 回答
- `global`: 全コミュニティ要約に対する map-reduce (部分回答 + 役立ち度スコア → 上位を統合)
- `basic`: 比較用の素朴なベクトル RAG (チャンク上位 k 件)

## 実行方法

```bash
pip install -e . openai          # リポジトリのルートで (sqgraph 未インストールでも src/ を自動で参照します)
export OPENAI_API_KEY=sk-...     # 未設定ならリポジトリ直下の .env (OPENAI_API_KEY=...) を読み込みます

python examples/graphrag_openai/graphrag_demo.py index [--db PATH] [--force] [--chunk-size 250] [--messages-json chat.json]
python examples/graphrag_openai/graphrag_demo.py local  "質問" [--db PATH]
python examples/graphrag_openai/graphrag_demo.py global "質問" [--db PATH]
python examples/graphrag_openai/graphrag_demo.py basic  "質問" [--db PATH]
python examples/graphrag_openai/graphrag_demo.py demo [--db PATH] [--messages-json chat.json] [-q "質問" ...]
    # DB が無ければ index してから、質問を basic/local/global で比較 (-q は複数指定可、省略時は固定の質問 5 件)
```

既定の DB は `examples/graphrag_openai/graphrag.db` (`*.db` は gitignore 済み)。`--db` は全コマンドで有効です。
同梱コーパスでのインデックス作成は約 45 回の LLM 呼び出し + 数回の埋め込み呼び出しで、
`demo` の質問 5 件で約 40 回の LLM 呼び出し (`local` / `basic` は 1 回、`global` はコミュニティ数 + 1 回 / 質問) です。

## チャットログをインデックスする (`--messages-json`)

`--messages-json PATH` には `{"t": "YYYY-MM-DD HH:MM", "u": "ユーザー名", "c": "本文"}` の JSON 配列を渡します (同梱コーパスの代わりに使われます)。

- 時刻順に並べ、6 時間超の空き / 12 メッセージ / 約 1500 文字ごとに「会話ドキュメント」へ分割し、`[YYYY-MM-DD HH:MM] ユーザー名: 本文` の行として描画します
  (1 メッセージ = 1 行、改行は ` / ` に畳みます)。チャンクは行単位でまとめ (既定 600 文字、行の途中では切りません)、直前の 1 行を重複させます。
- ユーザー名の `(9/14-9/16お休み)` / `[不在]9/9～11` / `_1718休` のような一時的な接尾辞は取り除き、同一人物を 1 つのエンティティに揃えます。
- 抽出プロンプトはチャット向けに切り替わり、人物 (表示名で統一)・サービス/リポジトリ・組織/チーム・チケット・イベント・技術・ドキュメントなどを抽出します。
  各チャンクの発言者一覧をプロンプトに渡して名前の揺れを抑えます。

## コスト表示

全コマンドの最後に OpenAI API の使用量と概算費用 (フェーズ別・モデル別のトークン数と USD、1 USD = 150 JPY の概算の円換算) を表示します。
`index` は extraction / entity-merge / embeddings / community-reports、質問系は `query/basic` `query/local` `query/global` に分かれ、
`demo` は質問 x モード別の費用表も出します。単価表は `graphrag_demo.py` の `PRICES` (USD / 100 万トークン) です。
