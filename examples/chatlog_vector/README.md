# chatlog_vector — 埋め込みだけで作るチャットログ検索グラフ

チャットログ (Discord など) を **LLM 抽出なし・埋め込みのみ**で sqgraph に取り込み、ハイブリッド検索と LLM 回答を行う汎用ツールです。インデックス時の OpenAI 利用は `text-embedding-3-small` のみ (100 メッセージあたり約 $0.0003) で、検索は質問 1 回分の埋め込みだけ、`ask` は加えて回答生成の LLM 1 回です。

```bash
pip install -e ".[memory]"        # openai が必要 (OPENAI_API_KEY は環境変数かリポジトリ直下の .env から読みます。値は表示しません)
```

## 入力 JSON

メッセージのオブジェクトの配列です (時系列順でなくても可)。

| フィールド | 説明 |
|---|---|
| `id` | メッセージ ID (必須。文字列) |
| `text` (または `content`) | 本文 (メンションなどは表示名に置換済みのものを推奨) |
| `user_id`, `username` | 投稿者 |
| `t` (または `posted_at`) | 投稿時刻 (`YYYY-MM-DDTHH:MM:SS`, ローカル時刻。文字列比較で絞り込みます) |
| `channel_id`, `channel_name` | チャンネル (スレッドの場合は親チャンネル) |
| `thread_id`, `thread_name` | スレッド (無ければ省略 / null) |
| `mentions` | `[{"id": ..., "name": ...}]` (任意) |

## グラフ

```
(:User {user_id, name}) -[:POSTED]-> (:Message {id, text, author, channel, thread, posted_at, url, ...})
(:Message) -[:IN]-> (:Thread {id, name}) -[:IN]-> (:Channel {id, name})     スレッド外の発言は Message -[:IN]-> Channel
(:Message) -[:MENTIONS]-> (:User)
(:Message) -[:NEXT]-> (:Message)                                            同じスレッド / チャンネル内で時系列に連結
```

`Message` に cosine のベクトル索引 (`msg_emb`, 1536 次元) と `text` の trigram 全文索引 (`msg_text`)。埋め込み対象は `[チャンネル / スレッド] 投稿者: 本文` です。

## 使い方

```bash
# 取り込み (再実行しても既存メッセージ ID はスキップ。NEXT 連結は追加のあったスレッド / チャンネルだけ張り直し)
python examples/chatlog_vector/chatlog_vector.py index --messages-json messages.json --db chat.db --guild-id <GUILD_ID>

# ハイブリッド検索 (ベクトル + 全文を RRF で融合、ヒットごとに前後 2 件を NEXT エッジで付与)。LLM は使いません
python examples/chatlog_vector/chatlog_vector.py search "認証基盤の担当は？" --db chat.db -k 10 \
    [--author Bob] [--channel backend] [--since 2026-09-01] [--until 2026-09-30]

# 検索結果を根拠に日本語で回答 (発言の日時・URL を引用)。コストを表示します
python examples/chatlog_vector/chatlog_vector.py ask "最近の進捗は？" --db chat.db [--model gpt-5.4-mini] [-k 8] [--author ... --since ...]

python examples/chatlog_vector/chatlog_vector.py stats --db chat.db      # ノード / エッジ数

python -m sqgraph.viewer chat.db                                         # グラフをブラウザで確認
```

`--author` / `--channel` は部分一致 (大文字小文字を区別しない)、`--since` / `--until` は日付または日時の前方一致比較 (両端を含む) です。フィルタ指定時は候補プールを広げてから絞り込みます。全文検索は trigram なので 3 文字未満の語は使われず、ひらがなだけの語は無視します (ベクトル検索が補います)。

## コスト

`text-embedding-3-small` $0.02 / 1M トークン、`gpt-5.4-mini` 入力 $0.75 / キャッシュ入力 $0.075 / 出力 $4.50 (1M トークンあたり)。各コマンドの最後にトークン数・USD・円 (150 円/USD 換算) を表示します。

## 注意

実データ (社内チャットなど) の JSON / DB は `data/` などの追跡対象外に置き、リポジトリにコミットしないでください。
