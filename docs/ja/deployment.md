[English](../deployment.md)

# クラウドで sqgraph を動かす (AWS / Cloudflare)

sqgraph は **SQLite の 1 ファイルに載る組み込み DB** です。サーバー型の DB (Neo4j など) と違い、「DB サーバーを立てる」のではなく「**DB ファイルを置いたディスクの上でアプリ (bot / MCP サーバー / API) を動かす**」構成になります。この文書では、AWS と Cloudflare でその構成を作る方法と、守るべき制約をまとめます。

- [1. 先に知っておくべき制約](#1-先に知っておくべき制約)
- [2. 構成パターン](#2-構成パターン)
- [3. 共通部品 (Docker イメージ / スナップショット / Litestream)](#3-共通部品)
- [4. AWS](#4-aws)
- [5. Cloudflare](#5-cloudflare)
- [6. MCP サーバーを公開するときの認証](#6-mcp-サーバーを公開するときの認証)
- [7. 転送量とコスト (DB が大きくなったとき)](#7-転送量とコスト)
- [8. 運用 (バックアップ・更新・監視)](#8-運用)
- [9. チェックリスト](#9-チェックリスト)

---

## 1. 先に知っておくべき制約

| 制約 | 理由 | 帰結 |
|---|---|---|
| **書き込みは 1 プロセスだけ** | SQLite の書き込みロックは 1 つ。`sqgraph.memory` の取り込みもプロセス内ロックで直列化している | 書き込み側はインスタンス 1 台固定。オートスケールさせない。デプロイ時に新旧 2 台が同時に書かないようにする |
| **DB ファイルはローカルディスクに置く** | 書き込み可能で開くと `journal_mode=WAL` になる。WAL は共有メモリ (`-shm`) を使うため、**ネットワークファイルシステム (EFS / NFS / SMB / FUSE マウントのオブジェクトストレージ) 上では安全に動かない** | EBS / インスタンスストア / コンテナのローカルディスクを使う。永続化・バックアップは Litestream などで S3 / R2 に逃がす |
| **SQLite 3.38 以上 + 拡張ロード** | JSON の `->` 演算子、sqlite-vec (`vec0`) のロードが必要 | Python の `sqlite3` が `enable_load_extension` を持っている必要がある (下のチェックで確認) |
| **読み取りはいくらでも増やせる** | `GraphDB(path, read_only=True)` は `mode=ro` で開き、書き込みクエリを拒否する | 「書き込み 1 + 読み取り N」はスナップショット配布で作る (パターン B) |
| **LLM は外部 API** | `sqgraph` 本体は LLM を使わない。`sqgraph.memory` の取り込み / `answer` だけが OpenAI などを呼ぶ | 実行環境から LLM API への外向き通信と、API キーの安全な受け渡しが必要 |

実行環境で動くかは、次のスクリプトで確認できます (Docker イメージや Lambda の中で実行)。

```python
import sqlite3, sqlite_vec
print("sqlite", sqlite3.sqlite_version)                       # 3.38 以上であること
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)                               # AttributeError なら拡張ロード不可のビルド
sqlite_vec.load(conn)
print("vec", conn.execute("select vec_version()").fetchone()[0])
```

公式の `python:3.12-slim` (Debian bookworm, SQLite 3.40) イメージはこの条件を満たします。マネージドランタイム (Lambda の zip デプロイなど) を使う場合は、デプロイ前に必ずこのチェックを通してください。通らない場合はコンテナイメージにします。

---

## 2. 構成パターン

| パターン | 形 | 向いている用途 |
|---|---|---|
| **A. 単一サーバー + 継続バックアップ** (推奨の出発点) | 1 台のコンテナ / VM が DB ファイルを持ち、読み書きの両方を担当。Litestream で S3 / R2 に継続レプリケーション | `sqgraph.memory` の MCP サーバー、Discord bot、社内 API |
| **B. 書き込み 1 + 読み取り N** | A の書き込みサーバーが定期的にスナップショットを S3 / R2 に置き、読み取り側 (Lambda / 複数コンテナ) がダウンロードして `read_only=True` で開く | 検索 API のアクセスが多い、読み取りのレイテンシを地域ごとに下げたい |
| **C. バッチ構築 → 読み取り専用配布** | GraphRAG のインデックス (`examples/graphrag_openai` の `index` など) をバッチで作り、完成した DB ファイルを配布するだけ | 文書コーパスの GraphRAG、更新頻度が低い知識ベース |

```
A:  client ──> [app + sqgraph + DB file] ──(Litestream)──> S3 / R2

B:  writer ──> [app + DB file] ──(snapshot 定期)──> S3 / R2 ──(download)──> [reader 1..N (read_only)]

C:  [batch job: index] ──(upload)──> S3 / R2 ──(download)──> [reader 1..N (read_only)]
```

B / C の読み取り側が見るのはスナップショット時点のデータです (数分〜数時間の遅れ)。書いた直後に読める必要がある処理 (例: `add_memory` の直後の `search_memory`) は書き込みサーバー側で処理してください。

---

## 3. 共通部品

### 3.1 Docker イメージ

```dockerfile
FROM python:3.12-slim

# Litestream (パターン A で使う。不要なら削除)。バージョンは最新のリリースに合わせる
ARG LITESTREAM_VERSION=0.5.17
ADD https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-${LITESTREAM_VERSION}-linux-x86_64.tar.gz /tmp/litestream.tgz
RUN tar -C /usr/local/bin -xzf /tmp/litestream.tgz && rm /tmp/litestream.tgz

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[mcp]"

COPY deploy/litestream.yml /etc/litestream.yml
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV DB_PATH=/data/memory.db
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
```

この章のファイルは [deploy/](../../deploy/) に同じものを置いてあります。ビルドはリポジトリのルートで行います (ビルドコンテキストはルート)。

```bash
docker build -f deploy/Dockerfile -t sqgraph-memory .
```

> Litestream のリリースアセット名 (`linux-x86_64` / `linux-amd64` / `.deb` など) はバージョンによって違うので、[リリースページ](https://github.com/benbjohnson/litestream/releases) で確認してください。ARM (Graviton) で動かす場合は arm64 版にします。

### 3.2 Litestream 設定 (S3 / R2)

Litestream は SQLite の WAL を監視して、変更をオブジェクトストレージに数秒おきに送るツールです。sqgraph は WAL モードで書くのでそのまま使えます。v0.5 以降は **DB 1 つにつきレプリカ 1 つ** です。

```yaml
# deploy/litestream.yml
access-key-id: ${LITESTREAM_ACCESS_KEY_ID}         # AWS で IAM ロールを使うなら 2 行とも削除
secret-access-key: ${LITESTREAM_SECRET_ACCESS_KEY}

dbs:
  - path: ${DB_PATH}
    replica:
      url: ${LITESTREAM_REPLICA_URL}               # s3://my-bucket/sqgraph/memory
      # --- Cloudflare R2 の場合 ---
      # endpoint: https://<ACCOUNT_ID>.r2.cloudflarestorage.com
      # region: auto
      sync-interval: 5s
```

### 3.3 起動スクリプト (リストア → レプリケーションしながらアプリ起動)

```sh
#!/bin/sh
# deploy/entrypoint.sh
set -e
mkdir -p "$(dirname "$DB_PATH")"

# ローカルに DB が無ければ (新しいコンテナ / ディスク) レプリカから復元。レプリカも無ければ新規作成になる
litestream restore -config /etc/litestream.yml -if-db-not-exists -if-replica-exists "$DB_PATH"

# アプリを子プロセスとして起動し、動いている間ずっとレプリケーションする
exec litestream replicate -config /etc/litestream.yml -exec "python -m sqgraph.memory.mcp \
  --db $DB_PATH --transport streamable-http --host 0.0.0.0 --port 8000 \
  --group-id ${GROUP_ID:-main} --extractor ${EXTRACTOR:-openai} ${MCP_EXTRA_ARGS}"
```

- `--host 0.0.0.0` はコンテナの外から届くようにするためです。**MCP サーバー自体には認証が無い**ので、必ず [6 章](#6-mcp-サーバーを公開するときの認証) の方法で前段に認証を置いてください。
- `OPENAI_API_KEY` は環境変数で渡します (Secrets Manager / Cloudflare のシークレットから注入)。`.env` をイメージに焼き込まないでください。
- 共有サーバーは `--extractor openai` (`EXTRACTOR` 環境変数で変更可) で起動します。MCP の既定は `agent` (呼び出したエージェントが抽出して `commit_memory` で保存。OpenAI キー不要) です。

### 3.4 読み取り用スナップショットの作り方 (パターン B / C)

WAL モードのファイルをそのままコピーすると、`-wal` にしか無い最新の変更が欠けたり、読み取り専用の環境で開けなかったりします。**`VACUUM INTO` で整合性のとれた 1 ファイルを作ってから配布**してください (結果のファイルは `journal_mode=DELETE` の単一ファイルになります)。書き込み中のプロセスがいても実行できます。

```python
# deploy/snapshot.py  —  python deploy/snapshot.py /data/memory.db /tmp/snap.db && aws s3 cp /tmp/snap.db s3://my-bucket/snapshots/memory-$(date +%Y%m%dT%H%M%S).db
import os, sqlite3, sys
import sqlite_vec

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    os.remove(dst)
conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
conn.enable_load_extension(True)
sqlite_vec.load(conn)                 # vec0 仮想テーブルを含む DB なので拡張をロードしておく
conn.enable_load_extension(False)
conn.execute("VACUUM INTO ?", (dst,))
conn.close()
```

スナップショットは**毎回新しいキー名**で置き、「最新はどれか」を `snapshots/LATEST` のような小さなファイル (中身はキー名) で示すと、読み取り側が中途半端なファイルを掴みません。

---

## 4. AWS

### 4.1 どのサービスを使うか

| サービス | 使い方 | 評価 |
|---|---|---|
| **EC2 + EBS** | 1 台の VM に DB を置く。EBS スナップショット / Litestream でバックアップ | ◎ 最もシンプル。DB が大きくなっても再起動で全量転送が起きない。まずはこれ |
| **ECS on EC2 + EBS ボリューム** | タスクに EBS をアタッチ | ◎ コンテナで運用しつつ永続ディスクが欲しいとき |
| **ECS on Fargate + Litestream (S3)** | タスク 1 個固定。ローカル (エフェメラル) ストレージに DB、起動時に S3 から復元 | ○ サーバー管理不要。ただし**起動のたびに DB 全量を復元**するので、DB が小さいうち (目安 1 GB 程度まで) 向け (7 章)。デプロイ設定に注意 (4.3) |
| **Lambda (読み取り専用)** | S3 のスナップショットを `/tmp` に落として `read_only=True` で開く | ○ パターン B / C の読み取り側。コールドスタートごとにダウンロードが発生する (7 章) |
| **Lambda (書き込み)** | — | ✕ 同時実行で複数の書き込みが発生する。やらない |
| **EFS 上の DB を共有** | — | ✕ WAL がネットワーク FS で安全に動かない。複数タスクからの書き込みは破損の原因 |
| App Runner | — | △ ディスクが永続でなく、インスタンス数を 1 に固定しにくい。Fargate を推奨 |

### 4.2 EC2 + EBS (最小構成)

1. Amazon Linux 2023 か Ubuntu の EC2 を 1 台、gp3 の EBS をデータ用に用意して `/data` にマウント。
2. Docker を入れて 3.1 のイメージを起動するか、venv に `pip install ".[mcp]"` して systemd で起動。
3. バックアップは Litestream (3.2) を S3 に向ける。インスタンスに S3 への書き込み権限を持つ **IAM ロール** を付ければアクセスキーは不要です。
4. EBS のスナップショット (Data Lifecycle Manager) も併用すると、Litestream が止まっていた場合の保険になります。

```ini
# /etc/systemd/system/sqgraph-memory.service (Docker を使わない場合)
[Service]
Environment=DB_PATH=/data/memory.db
EnvironmentFile=/etc/sqgraph/env            # OPENAI_API_KEY=... (権限 600)
ExecStart=/usr/local/bin/litestream replicate -config /etc/litestream.yml \
  -exec "/opt/sqgraph/.venv/bin/python -m sqgraph.memory.mcp --db /data/memory.db --transport streamable-http --host 127.0.0.1 --port 8000 --group-id main --extractor openai"
Restart=always
```

外からのアクセスは、同じ VM の Caddy / nginx で認証をかけてから `127.0.0.1:8000` に流すか、Cloudflare Tunnel を使います (6 章)。

### 4.3 ECS on Fargate + Litestream

- **タスク定義**
  - コンテナイメージ: 3.1 のイメージ (ECR に push)
  - エフェメラルストレージ: DB サイズ + 余裕 (既定 20 GiB、最大 200 GiB)
  - 環境変数: `DB_PATH=/data/memory.db`, `LITESTREAM_REPLICA_URL=s3://<bucket>/sqgraph/memory`, `GROUP_ID=...`
  - シークレット: `OPENAI_API_KEY` を Secrets Manager (または SSM パラメータストア) から `secrets` で注入
  - タスクロール: 対象バケット / プレフィックスへの `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket`
- **ネットワーク**: プライベートサブネットで動かす場合は **S3 のゲートウェイ型 VPC エンドポイント**を作ってください。無いと起動時の復元や Litestream の転送が NAT Gateway を通り、GB 単位の処理料がかかります (7 章)
- **サービス設定 (重要)**
  - `desiredCount = 1`、オートスケーリングは設定しない
  - デプロイ設定を `minimumHealthyPercent = 0`, `maximumPercent = 100` にする。既定値 (100 / 200) だと**新旧 2 タスクが同時に起動し、同じレプリカに 2 つの書き込みが走る**ため
  - その代わりデプロイ中は数十秒の停止が発生します
- **停止時**: ECS はタスク停止時に SIGTERM を送り、`stopTimeout` (既定 30 秒) 待ちます。Litestream は `-exec` の子プロセスにシグナルを渡して終了します。ただし `sync-interval` の間に書いた最後の数秒分は、停止の仕方によっては失われる可能性があります。取りこぼしが許されないデータは、呼び出し側で再送できるようにしてください (`sqgraph.memory` は同じ `source_ref` の再投入を無視するので、再送しても重複しません)。
- **前段**: ALB (ターゲットはタスクの 8000 番) + 認証 (6 章)。ヘルスチェックは TCP か、アプリの HTTP パス (MCP の `/mcp` は POST 前提なので、ALB のヘルスチェックには向かない。TCP チェックを推奨)。

### 4.4 Lambda で読み取り専用の検索 API (パターン B / C)

コンテナイメージ (3.1 から Litestream と entrypoint を除いたもの + `awslambdaric`、または Lambda の Python ベースイメージ) で作るのが確実です。

```python
# handler.py
import json, os
import boto3
from sqgraph import GraphDB

_S3 = boto3.client("s3")
_DB: GraphDB | None = None
_KEY: str | None = None


def _db() -> GraphDB:
    """コールドスタート時 (と LATEST が変わったとき) にスナップショットを /tmp へ落として開く。"""
    global _DB, _KEY
    bucket = os.environ["SNAPSHOT_BUCKET"]
    key = _S3.get_object(Bucket=bucket, Key="snapshots/LATEST")["Body"].read().decode().strip()
    if _DB is None or key != _KEY:
        path = "/tmp/" + os.path.basename(key)
        if not os.path.exists(path):
            _S3.download_file(bucket, key, path)
        if _DB is not None:
            _DB.close()
        _DB, _KEY = GraphDB(path, read_only=True), key
    return _DB


def handler(event, context):
    body = json.loads(event.get("body") or "{}")
    rows = _db().query(body["cypher"], body.get("params"))
    return {"statusCode": 200, "body": json.dumps(rows, ensure_ascii=False, default=str)}
```

- `LATEST` の確認をリクエストのたびに行いたくなければ、「最後に確認してから 60 秒以内なら確認しない」などの間引きを入れてください。
- `/tmp` の容量 (エフェメラルストレージ) は既定 512 MB で、最大 10 GB まで増やせます。DB サイズに合わせて設定します。
- 任意の Cypher を受け付ける API を公開しないでください (上の例は説明用)。実際には `sqgraph.memory.Memory(..., read_only=True)` の `search` など、決まった操作だけを公開します。読み取り専用 DB でも、重いクエリで Lambda のタイムアウトまで CPU を使われる可能性があります。
- `sqgraph.memory` の検索は**質問の埋め込み**に OpenAI API を呼ぶので、Lambda からインターネット (または NAT) に出られる必要があります。

### 4.5 GraphRAG のインデックスをバッチで作る (パターン C)

`examples/graphrag_openai/graphrag_demo.py index` のようなバッチは、ECS の単発タスク (`aws ecs run-task`)、AWS Batch、EventBridge Scheduler からの定期実行が向いています。処理の最後に 3.4 のスナップショットを S3 に置き、`LATEST` を更新すれば、読み取り側 (4.4) が次のリクエストから新しいデータを使います。

---

## 5. Cloudflare

### 5.1 Workers / D1 / Durable Objects で直接動かせない理由

| サービス | 動かない理由 |
|---|---|
| **Workers (JS / Python)** | ローカルのファイルシステムが無く、ネイティブ拡張 (sqlite-vec) もロードできない。Python Workers は Pyodide (WebAssembly) 上で動く |
| **D1** | 中身は SQLite だが、アクセスは Workers のバインディング経由のクエリだけ。独自の拡張 (sqlite-vec の `vec0`) をロードできず、sqgraph の Python コードも D1 上では動かない |
| **Durable Objects の SQLite ストレージ** | JS の API から SQL を実行する形で、sqgraph (Python + 拡張) を載せられない |

sqgraph を Cloudflare で使うには、**Linux コンテナを動かせる Cloudflare Containers** を使うか、**別の場所 (AWS / VPS) で動かして Cloudflare を前段に置く**かのどちらかになります。

### 5.2 Cloudflare Containers + Litestream (R2)

Cloudflare Containers では、各コンテナインスタンスが Durable Object (DO) と 1 対 1 で結び付き、DO の名前でインスタンスを指定できます。**常に同じ名前 (例: `"main"`) で呼べば、書き込みインスタンスが 1 つに保たれます**。これで「書き込みは 1 プロセスだけ」という制約を満たせます。

注意点:

- **コンテナのディスクはエフェメラル**です。スリープ (既定では一定時間アクセスが無いと停止) や再デプロイでローカルの DB ファイルは消えます。3.3 の entrypoint (起動時に R2 から復元 → Litestream で継続レプリケーション) が前提になります。
- R2 を FUSE でマウントしてその上に DB を置く方法は、1 章の理由 (WAL + ネットワーク FS) で**使わないでください**。
- スリープからの復帰時は、**毎回 R2 から DB 全量を復元**するので、その時間ぶん最初のリクエストが遅くなります (R2 はエグレス無料なので料金はほぼかかりませんが、DB が数 GB になると復帰に数十秒かかります)。常時応答が必要なら、スリープまでの時間を長くしてください。DB が大きくなる前提なら、5.3 の構成 (永続ディスクのサーバー + Cloudflare を前段) を選んでください (7 章)。

`wrangler.jsonc` の例:

```jsonc
{
  "name": "sqgraph-memory",
  "main": "src/index.ts",
  "compatibility_date": "2026-09-01",
  "containers": [
    { "class_name": "SqgraphContainer", "image": "./deploy/Dockerfile", "max_instances": 1 }
  ],
  "durable_objects": {
    "bindings": [{ "name": "SQGRAPH", "class_name": "SqgraphContainer" }]
  },
  "migrations": [{ "tag": "v1", "new_sqlite_classes": ["SqgraphContainer"] }]
}
```

Worker (`src/index.ts`) の例:

```ts
import { Container, getContainer } from "@cloudflare/containers";

export class SqgraphContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "30m"; // アクセスが無くなってからスリープするまで

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // シークレットは `wrangler secret put OPENAI_API_KEY` などで登録し、コンテナの環境変数に渡す
    this.envVars = {
      OPENAI_API_KEY: env.OPENAI_API_KEY,
      LITESTREAM_ACCESS_KEY_ID: env.R2_ACCESS_KEY_ID,
      LITESTREAM_SECRET_ACCESS_KEY: env.R2_SECRET_ACCESS_KEY,
      LITESTREAM_REPLICA_URL: "s3://sqgraph/memory",
      DB_PATH: "/data/memory.db",
      GROUP_ID: "main",
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // 認証 (6 章) を通ったリクエストだけをコンテナへ。書き込みインスタンスは常に "main" の 1 つ
    return getContainer(env.SQGRAPH, "main").fetch(request);
  },
};
```

- `deploy/Dockerfile` はリポジトリのルートをビルドコンテキストにする前提です。wrangler の設定でビルドコンテキストをルートに指定してください (設定キーは wrangler のバージョンで確認)。
- `litestream.yml` には R2 用の `endpoint: https://<ACCOUNT_ID>.r2.cloudflarestorage.com` と `region: auto` を設定します (3.2)。R2 の API トークン (S3 互換のアクセスキー) は、対象バケットだけに権限を絞って発行してください。
- Cloudflare Containers の API (`@cloudflare/containers` のクラス名 / プロパティ名、`wrangler` の設定キー、インスタンスの種類とディスク容量の上限、スナップショット機能の有無) は更新が続いています。ここに書いたものは執筆時点の形なので、[公式ドキュメント](https://developers.cloudflare.com/containers/) で最新の書き方を確認してください。

### 5.3 AWS / VPS で動かして Cloudflare を前段に置く (実用的な構成)

DB とアプリは 4.2 の EC2 (または任意の VPS) に置き、Cloudflare は**入口と認証**だけを担当する構成です。サーバー側にインバウンドのポートを開けずに済みます。

1. サーバーで MCP サーバーを `--host 127.0.0.1 --port 8000` で起動
2. 同じサーバーで `cloudflared` を動かし、Cloudflare Tunnel で `https://memory.example.com` → `http://127.0.0.1:8000` をつなぐ
3. Cloudflare Access (Zero Trust) でそのホスト名にアプリケーションを作り、人は SSO、エージェントは **サービストークン** で認証させる (6 章)
4. バックアップは Litestream から R2 (または S3) へ

### 5.4 R2 をスナップショット置き場にする (パターン B / C)

3.4 のスナップショットを R2 に置き、読み取り側 (Containers、AWS の Lambda、社内 PC など) がダウンロードして `read_only=True` で開きます。R2 はデータ転送 (エグレス) が無料なので、多くの読み取り側に配る用途と相性が良いです。

---

## 6. MCP サーバーを公開するときの認証

`python -m sqgraph.memory.mcp --transport streamable-http` は**認証を持っていません**。そのまま `0.0.0.0` で公開すると、URL を知っている人は誰でも記憶を読み書きできます。必ず前段で認証してください。

| 方法 | 設定 | クライアント (Claude Code) 側 |
|---|---|---|
| **Cloudflare Access のサービストークン** | Access のアプリケーションに「Service Auth」ポリシーを作り、トークンを発行 | `claude mcp add --transport http sqgraph-memory https://memory.example.com/mcp --header "CF-Access-Client-Id: <id>" --header "CF-Access-Client-Secret: <secret>"` |
| **リバースプロキシで Bearer トークン** | Caddy / nginx で `Authorization: Bearer <token>` を検証してから `127.0.0.1:8000` に流す (ECS なら同じタスクのサイドカーにする) | `--header "Authorization: Bearer <token>"` |
| **VPN / プライベートネットワーク** | 社内 VPN / Tailscale / VPC 内からのみアクセス可能にする | 通常の URL |

あわせて、サーバー側の制限も必ず設定してください。

- `--group-id` で名前空間を固定する (他の `group_id` の読み書きを拒否)
- `--allowed-channels` で、その MCP サーバーから読めるチャンネルを絞る (指定すると ACL を迂回する `read_cypher` は自動で無効になる)
- 読むだけのエージェントには `--read-only` のサーバーを別に立てる (`add_memory` が登録されない)
- ビューワー (`python -m sqgraph.viewer`) は DB を読み取り専用で開きますが、認証はありません。公開する場合は同じく前段で認証してください

> リバースプロキシ経由でアクセスしたときに `421 Misdirected Request` などが返る場合は、MCP SDK の Host ヘッダー検証 (DNS リバインディング対策) に引っかかっている可能性があります。使っている `mcp` パッケージのバージョンの設定方法を確認してください。

---

## 7. 転送量とコスト

DB が大きくなったときに気になるのは「起動のたびに DB 全量を転送して料金がかかるのでは」という点です。結論は次のとおりです。

- **全量の転送が起きるのは、ディスクが消える構成だけ** (Fargate / Cloudflare Containers / Lambda)。永続ディスク (EBS) なら、全量の転送は障害からの復旧時だけです
- **同じリージョンの S3 からのダウンロードは転送料が無料**。料金が膨らむのは NAT Gateway 経由・リージョン越え・インターネット越えのとき
- 多くの場合、料金より先に**起動時間**が問題になります

### 構成ごとの転送

| 構成 | 起動のたびに全量ダウンロード | 普段の転送 |
|---|---|---|
| EC2 + EBS / ECS on EC2 + EBS | **しない** (`-if-db-not-exists` により、ディスクが空のときだけ復元) | Litestream の変更差分 (WAL) + 定期的な全体スナップショットのアップロード |
| ECS on Fargate | **する** (タスクの起動・デプロイ・再起動のたび) | 同上 |
| Cloudflare Containers | **する** (スリープからの復帰・再デプロイのたび) | 同上 |
| Lambda (読み取り側) | **する** (実行環境のコールドスタートごと、スナップショットの更新ごと) | なし |
| パターン B の読み取り N 台 | スナップショットの更新ごとに N 台がそれぞれ全量 | 更新頻度 × 台数 × DB サイズ |

S3 / R2 へのアップロード (イングレス) は無料で、かかるのはリクエスト料と保存料です。

### 経路ごとの料金 (目安)

AWS はバージニア北部の目安です (東京リージョンはやや高い)。最新の料金は各社の料金ページで確認してください。

| 経路 | 料金 | 5 GB の DB を 1 回転送 |
|---|---|---|
| S3 → 同じリージョンの EC2 / Fargate / Lambda | 転送料無料 (リクエスト料のみ) | ほぼ $0 |
| 上記が **NAT Gateway を通る** | 処理料 約 $0.045/GB | 約 $0.23 |
| S3 → 別のリージョン | 約 $0.02/GB | 約 $0.10 |
| S3 → インターネット (社内 PC / Cloudflare など) | 約 $0.09/GB | 約 $0.45 |
| **R2** → どこへでも | エグレス無料 (読み取りリクエスト料のみ) | ほぼ $0 |

料金が膨らむ例:

- Fargate を NAT 経由で 1 日 10 回再起動: 5 GB × 10 × $0.045 ≒ **$2.3/日**
- 読み取り 10 台にインターネット越しで毎時スナップショットを配布: 5 GB × 10 × 24 × $0.09 ≒ **$108/日**

見落としやすい**保存料**: 日次スナップショットを 30 日分残すと 5 GB × 30 = 150 GB で、S3 標準で約 $3.5/月、R2 で約 $2.3/月です。ライフサイクルルールで古いものを消してください。

### 起動時間

同じリージョン内のダウンロードは大まかに毎秒数百 MB です。1 GB なら数秒、10 GB なら数十秒〜1 分以上、起動 (Lambda のコールドスタート、Containers のスリープ復帰、Fargate のデプロイ) が遅くなります。Litestream の復元はスナップショットに差分を適用するので、単純なダウンロードより少し長くかかります。

### DB サイズの目安

容量の大半はベクトル (1536 次元 × 4 バイト = 1 本あたり約 6 KB) です。

- 実測: `chatlog_vector` 形式 (1 メッセージ 1 ベクトル) で、711 メッセージが約 8 MB (1 メッセージあたり約 11 KB)
- 推定: `sqgraph.memory` は 1 メッセージから 1〜2 本のファクト (それぞれベクトル付き) ができるので、1 メッセージあたり 10〜15 KB 程度
- 10 万メッセージで 1〜1.5 GB 程度。社内 Discord 規模なら数 GB に届くまでには時間がかかります

### 対策 (効果の大きい順)

1. **書き込み側は永続ディスクにする** (EC2 + EBS / ECS on EC2 + EBS)。全量の転送が障害時だけになります。DB が大きくなる前提なら、これを基本にしてください
2. **NAT を通さない**。S3 のゲートウェイ型 VPC エンドポイント (無料) を作ると、Fargate / Lambda から S3 への転送が NAT を通らなくなります
3. **配布は R2 にする**。読み取り側がリージョン外やインターネットにある場合、転送料が 0 になります
4. **ダウンロードの回数を減らす**
   - Lambda は `LATEST` が変わったときだけダウンロードする (4.4 の例はそうなっている)。プロビジョンド同時実行でコールドスタートを減らす
   - スナップショットの更新頻度を、読み取り側が必要とする鮮度に合わせて下げる
   - Cloudflare Containers は `sleepAfter` を長くする
5. **DB 自体を小さくする**。`text-embedding-3-small` は次元数を減らせます (1536 → 512 でベクトルが約 1/3。検索精度は少し落ちる)。削除や更新が多かった DB は `VACUUM` で空き領域を詰める
6. **将来の選択肢**: Litestream v0.5 には、S3 上のレプリカを全量ダウンロードせずに必要なページだけ読む VFS 拡張があります。ただし sqgraph は現状この VFS を指定して DB を開けないので、使うにはコードの変更が必要です

## 8. 運用

### バックアップとリストア

- **Litestream**: レプリカから任意の時点に戻せます。`litestream restore -config /etc/litestream.yml -o /tmp/restored.db /data/memory.db` で別名に復元し、中身を確認してから差し替えてください。
- **定期スナップショット**: 3.4 の `VACUUM INTO` を日次などで取り、S3 / R2 のライフサイクルルールで古いものを消します。Litestream とは別系統の保険になります。
- **リストア訓練**: 年に数回は、別環境で復元して `GraphDB(path, read_only=True).stats()` や `Memory(...).stats()` が期待どおりになるか確認してください。

### sqgraph のバージョンを上げるとき

- DB のスキーマのバージョン (`_meta.schema_version`) は、**書き込み可能で開いたとき**に自動で上がります (例: v2 → v3 でリレーションシップ索引のメタデータ列が追加)。読み取り専用で開いたときは移行せずに読みます。
- パターン B / C では、**書き込み側を先に**新しいバージョンにし、そのあとで読み取り側を上げてください。
- `sqgraph.memory` の保存形式は、ファクトがノード形式だった古い DB を移行しません (開くとエラーで止まります)。形式が変わる更新のときは DB を作り直してください。

### 性能と容量の目安

- 大量に投入したあとや索引を作ったあとは `db.analyze()` を実行して、クエリプランナの統計を更新してください (`close()` でも自動で行います)。
- ベクトル索引は `vec0` による全件スキャンです。数十万ベクトル程度までは実用的ですが、それを超える場合はベクトル検索だけ外部サービス (OpenSearch / Vectorize など) に分けることを検討してください。
- `sqgraph.memory` の取り込みは 1 件あたり LLM を 2〜4 回呼び、直列で処理します。取り込みの量が多い bot では、キューに積んで 1 つのワーカーで順に処理してください (1 メッセージあたり約 $0.004 の目安は [memory.md のコスト目安](memory.md#コスト目安-gpt-54-mini-text-embedding-3-small) を参照)。

### 監視

- ディスク使用量 (DB + WAL)、Litestream のプロセスが動いているか、レプリカの最終更新時刻
- アプリのエラーログ (`SqgraphError`、OpenAI API の 429 / 5xx)
- LLM のコスト (`UsageTracker` の集計をログに出す)

---

## 9. チェックリスト

- [ ] 実行環境で 1 章のチェックスクリプトが通る (SQLite 3.38 以上、拡張ロード可)
- [ ] 書き込みインスタンスが常に 1 つ (オートスケールなし / デプロイで新旧が重ならない / Cloudflare は固定名の DO)
- [ ] DB ファイルがローカルディスク上にある (EFS / NFS / FUSE 上ではない)
- [ ] Litestream または定期スナップショットでバックアップしており、復元を試したことがある
- [ ] 読み取り側に配るのは `VACUUM INTO` で作ったスナップショット
- [ ] DB が大きくなる前提なら書き込み側は永続ディスク。Fargate / Lambda を使うなら S3 のゲートウェイ型 VPC エンドポイントがあり、NAT を通っていない (7 章)
- [ ] MCP サーバー / ビューワーの前段に認証がある。`0.0.0.0` で認証なしに公開していない
- [ ] `--group-id` / `--allowed-channels` / `--read-only` を用途に合わせて設定した
- [ ] `OPENAI_API_KEY` や R2 / S3 の認証情報をシークレット管理 (Secrets Manager / `wrangler secret`) で渡しており、イメージやリポジトリに含めていない

## 参考

- [Litestream](https://litestream.io/) (設定: [Configuration File](https://litestream.io/reference/config/)、[S3 互換サービス](https://litestream.io/guides/s3-compatible/))
- [SQLite: WAL モード](https://www.sqlite.org/wal.html) (ネットワークファイルシステムでの制約)
- [Cloudflare Containers](https://developers.cloudflare.com/containers/) / [Durable Object Container API](https://developers.cloudflare.com/containers/api/durable-object-container/)
- [Cloudflare Access のサービストークン](https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/)
- [Amazon ECS のデプロイ設定 (minimumHealthyPercent / maximumPercent)](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service_definition_parameters.html)
