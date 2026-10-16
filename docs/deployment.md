[日本語](ja/deployment.md)

# Running sqgraph in the Cloud (AWS / Cloudflare)

sqgraph is an **embedded DB that lives in a single SQLite file**. Unlike server-based databases such as Neo4j, you don't "run a DB server". Instead, you **run your application (bot / MCP server / API) on the disk that holds the DB file**. This document explains how to set that up on AWS and Cloudflare, and the constraints you need to respect.

- [1. Constraints to know first](#1-constraints-to-know-first)
- [2. Deployment patterns](#2-deployment-patterns)
- [3. Common building blocks (Docker image / snapshots / Litestream)](#3-common-building-blocks)
- [4. AWS](#4-aws)
- [5. Cloudflare](#5-cloudflare)
- [6. Authenticating a public MCP server](#6-authenticating-a-public-mcp-server)
- [7. Transfer volume and cost (when the DB grows)](#7-transfer-volume-and-cost)
- [8. Operations (backup, upgrades, monitoring)](#8-operations)
- [9. Checklist](#9-checklist)

---

## 1. Constraints to know first

| Constraint | Reason | Consequence |
|---|---|---|
| **Only one process may write** | SQLite has a single write lock. `sqgraph.memory` ingestion is also serialized by an in-process lock | Pin the writer to exactly one instance. Do not autoscale. During deploys, make sure the old and new instances never write at the same time |
| **Keep the DB file on a local disk** | Opened for writing, the DB switches to `journal_mode=WAL`. WAL uses shared memory (`-shm`), so it **does not run safely on network file systems (EFS / NFS / SMB / object storage mounted via FUSE)** | Use EBS, instance store, or the container's local disk. Persist and back up to S3 / R2 with Litestream or similar |
| **SQLite 3.38 or later with extension loading** | The JSON `->` operator and loading sqlite-vec (`vec0`) are required | Python's `sqlite3` must provide `enable_load_extension` (verify with the check below) |
| **Reads scale freely** | `GraphDB(path, read_only=True)` opens with `mode=ro` and rejects write queries | Build "1 writer + N readers" by distributing snapshots (pattern B) |
| **The LLM is an external API** | The `sqgraph` core does not use an LLM. Only `sqgraph.memory` ingestion and `answer` call OpenAI or similar | The runtime needs outbound access to the LLM API and a safe way to receive the API key |

To check whether your runtime works, run the script below (inside the Docker image or Lambda).

```python
import sqlite3, sqlite_vec
print("sqlite", sqlite3.sqlite_version)                       # must be 3.38 or later
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)                               # AttributeError means the build cannot load extensions
sqlite_vec.load(conn)
print("vec", conn.execute("select vec_version()").fetchone()[0])
```

The official `python:3.12-slim` image (Debian bookworm, SQLite 3.40) meets these requirements. If you use a managed runtime (such as a Lambda zip deployment), always run this check before deploying. If it fails, use a container image instead.

---

## 2. Deployment patterns

| Pattern | Shape | Suited for |
|---|---|---|
| **A. Single server + continuous backup** (recommended starting point) | One container / VM holds the DB file and handles both reads and writes. Continuous replication to S3 / R2 with Litestream | The `sqgraph.memory` MCP server, a Discord bot, an internal API |
| **B. 1 writer + N readers** | The writer server from A periodically puts snapshots in S3 / R2. Readers (Lambda / multiple containers) download them and open with `read_only=True` | High search API traffic, or lowering read latency per region |
| **C. Batch build → read-only distribution** | Build a GraphRAG index in a batch job (e.g. `index` in `examples/graphrag_openai`) and just distribute the finished DB file | GraphRAG over a document corpus, knowledge bases that rarely change |

```
A:  client ──> [app + sqgraph + DB file] ──(Litestream)──> S3 / R2

B:  writer ──> [app + DB file] ──(periodic snapshot)──> S3 / R2 ──(download)──> [reader 1..N (read_only)]

C:  [batch job: index] ──(upload)──> S3 / R2 ──(download)──> [reader 1..N (read_only)]
```

In B / C, readers see the data as of the snapshot (minutes to hours behind). Anything that must be readable immediately after a write (for example `search_memory` right after `add_memory`) should be handled on the writer server.

---

## 3. Common building blocks

### 3.1 Docker image

```dockerfile
FROM python:3.12-slim

# Litestream (used by pattern A; remove if not needed). Match the version to the latest release
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

The files in this section are also available in [deploy/](../deploy/). Build from the repository root (the build context is the root).

```bash
docker build -f deploy/Dockerfile -t sqgraph-memory .
```

> Litestream release asset names (`linux-x86_64` / `linux-amd64` / `.deb`, etc.) vary by version, so check the [releases page](https://github.com/benbjohnson/litestream/releases). To run on ARM (Graviton), use the arm64 build.

### 3.2 Litestream configuration (S3 / R2)

Litestream watches the SQLite WAL and ships changes to object storage every few seconds. sqgraph writes in WAL mode, so it works as is. Since v0.5, there is **one replica per DB**.

```yaml
# deploy/litestream.yml
access-key-id: ${LITESTREAM_ACCESS_KEY_ID}         # on AWS with an IAM role, delete both lines
secret-access-key: ${LITESTREAM_SECRET_ACCESS_KEY}

dbs:
  - path: ${DB_PATH}
    replica:
      url: ${LITESTREAM_REPLICA_URL}               # s3://my-bucket/sqgraph/memory
      # --- For Cloudflare R2 ---
      # endpoint: https://<ACCOUNT_ID>.r2.cloudflarestorage.com
      # region: auto
      sync-interval: 5s
```

### 3.3 Startup script (restore → start the app while replicating)

```sh
#!/bin/sh
# deploy/entrypoint.sh
set -e
mkdir -p "$(dirname "$DB_PATH")"

# If there is no local DB (new container / disk), restore it from the replica. If there is no replica either, a new DB is created
litestream restore -config /etc/litestream.yml -if-db-not-exists -if-replica-exists "$DB_PATH"

# Start the app as a child process and replicate for as long as it runs
exec litestream replicate -config /etc/litestream.yml -exec "python -m sqgraph.memory.mcp \
  --db $DB_PATH --transport streamable-http --host 0.0.0.0 --port 8000 \
  --group-id ${GROUP_ID:-main} --extractor ${EXTRACTOR:-openai} ${MCP_EXTRA_ARGS}"
```

- `--host 0.0.0.0` makes the server reachable from outside the container. **The MCP server itself has no authentication**, so always put authentication in front of it as described in [section 6](#6-authenticating-a-public-mcp-server).
- Pass `OPENAI_API_KEY` as an environment variable (injected from Secrets Manager / Cloudflare secrets). Do not bake `.env` into the image.
- A shared server starts with `--extractor openai` (overridable via the `EXTRACTOR` environment variable). The MCP default is `agent` (the calling agent does the extraction and saves it with `commit_memory`; no OpenAI key needed).

### 3.4 Building a read snapshot (patterns B / C)

Copying a WAL-mode file as is can miss recent changes that exist only in `-wal`, and may fail to open in read-only environments. **Create a consistent single file with `VACUUM INTO`, then distribute that** (the resulting file is a single file with `journal_mode=DELETE`). It can run while another process is writing.

```python
# deploy/snapshot.py  —  python deploy/snapshot.py /data/memory.db /tmp/snap.db && aws s3 cp /tmp/snap.db s3://my-bucket/snapshots/memory-$(date +%Y%m%dT%H%M%S).db
import os, sqlite3, sys
import sqlite_vec

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    os.remove(dst)
conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
conn.enable_load_extension(True)
sqlite_vec.load(conn)                 # the DB contains vec0 virtual tables, so load the extension
conn.enable_load_extension(False)
conn.execute("VACUUM INTO ?", (dst,))
conn.close()
```

Upload each snapshot under a **new key** and point to the latest one with a small file such as `snapshots/LATEST` (containing the key name). This keeps readers from picking up a partially written file.

---

## 4. AWS

### 4.1 Which service to use

| Service | Usage | Rating |
|---|---|---|
| **EC2 + EBS** | Put the DB on a single VM. Back up with EBS snapshots / Litestream | ◎ Simplest. Restarts never trigger a full transfer, even as the DB grows. Start here |
| **ECS on EC2 + EBS volume** | Attach EBS to the task | ◎ When you want containers with a persistent disk |
| **ECS on Fargate + Litestream (S3)** | Fixed at one task. DB on local (ephemeral) storage, restored from S3 at startup | ○ No server management. But the **full DB is restored on every start**, so it suits small DBs (up to about 1 GB; see section 7). Mind the deployment settings (4.3) |
| **Lambda (read-only)** | Download the S3 snapshot to `/tmp` and open with `read_only=True` | ○ Reader side of patterns B / C. A download happens on every cold start (section 7) |
| **Lambda (write)** | — | ✕ Concurrent executions cause multiple writers. Don't |
| **Sharing a DB on EFS** | — | ✕ WAL does not run safely on network file systems. Writes from multiple tasks cause corruption |
| App Runner | — | △ The disk is not persistent and it is hard to pin the instance count to 1. Fargate is recommended |

### 4.2 EC2 + EBS (minimal setup)

1. Prepare one EC2 instance (Amazon Linux 2023 or Ubuntu) and a gp3 EBS volume for data, mounted at `/data`.
2. Install Docker and run the image from 3.1, or `pip install ".[mcp]"` into a venv and run it under systemd.
3. For backup, point Litestream (3.2) at S3. If you attach an **IAM role** with write access to S3 to the instance, no access keys are needed.
4. Also using EBS snapshots (Data Lifecycle Manager) gives you insurance for when Litestream has stopped.

```ini
# /etc/systemd/system/sqgraph-memory.service (when not using Docker)
[Service]
Environment=DB_PATH=/data/memory.db
EnvironmentFile=/etc/sqgraph/env            # OPENAI_API_KEY=... (mode 600)
ExecStart=/usr/local/bin/litestream replicate -config /etc/litestream.yml \
  -exec "/opt/sqgraph/.venv/bin/python -m sqgraph.memory.mcp --db /data/memory.db --transport streamable-http --host 127.0.0.1 --port 8000 --group-id main --extractor openai"
Restart=always
```

For external access, either put Caddy / nginx on the same VM to authenticate and forward to `127.0.0.1:8000`, or use Cloudflare Tunnel (section 6).

### 4.3 ECS on Fargate + Litestream

- **Task definition**
  - Container image: the image from 3.1 (pushed to ECR)
  - Ephemeral storage: DB size plus headroom (default 20 GiB, max 200 GiB)
  - Environment variables: `DB_PATH=/data/memory.db`, `LITESTREAM_REPLICA_URL=s3://<bucket>/sqgraph/memory`, `GROUP_ID=...`
  - Secrets: inject `OPENAI_API_KEY` from Secrets Manager (or SSM Parameter Store) via `secrets`
  - Task role: `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` on the target bucket / prefix
- **Networking**: if you run in a private subnet, create an **S3 gateway VPC endpoint**. Without it, the startup restore and Litestream transfers go through the NAT Gateway and incur per-GB processing charges (section 7)
- **Service settings (important)**
  - `desiredCount = 1`, with no autoscaling
  - Set the deployment configuration to `minimumHealthyPercent = 0`, `maximumPercent = 100`. With the defaults (100 / 200), **the old and new tasks run at the same time, and two writers write to the same replica**
  - The trade-off is a downtime of a few tens of seconds during deploys
- **On stop**: ECS sends SIGTERM when a task stops and waits for `stopTimeout` (default 30 seconds). Litestream forwards the signal to the `-exec` child process and exits. However, depending on how the task is stopped, the last few seconds of writes within `sync-interval` may be lost. For data that must not be lost, make it re-sendable from the caller (`sqgraph.memory` ignores re-ingestion of the same `source_ref`, so re-sending creates no duplicates).
- **Front end**: ALB (target is port 8000 of the task) + authentication (section 6). For the health check, use TCP or an HTTP path of the app (MCP's `/mcp` expects POST, so it is not suitable for an ALB health check; a TCP check is recommended).

### 4.4 A read-only search API on Lambda (patterns B / C)

The reliable way is a container image (the one from 3.1 minus Litestream and the entrypoint, plus `awslambdaric`, or Lambda's Python base image).

```python
# handler.py
import json, os
import boto3
from sqgraph import GraphDB

_S3 = boto3.client("s3")
_DB: GraphDB | None = None
_KEY: str | None = None


def _db() -> GraphDB:
    """On cold start (and when LATEST changes), download the snapshot to /tmp and open it."""
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

- If you don't want to check `LATEST` on every request, throttle it, for example by skipping the check within 60 seconds of the last one.
- `/tmp` (ephemeral storage) defaults to 512 MB and can be raised to 10 GB. Size it to your DB.
- Do not expose an API that accepts arbitrary Cypher (the example above is for illustration). In practice, expose only fixed operations such as `search` on `sqgraph.memory.Memory(..., read_only=True)`. Even with a read-only DB, a heavy query can burn CPU until the Lambda timeout.
- `sqgraph.memory` search calls the OpenAI API to **embed the question**, so the Lambda needs a route to the internet (or a NAT).

### 4.5 Building a GraphRAG index as a batch job (pattern C)

Batches like `examples/graphrag_openai/graphrag_demo.py index` fit one-off ECS tasks (`aws ecs run-task`), AWS Batch, or scheduled runs from EventBridge Scheduler. At the end of the job, put the snapshot from 3.4 in S3 and update `LATEST`; readers (4.4) then use the new data from the next request.

---

## 5. Cloudflare

### 5.1 Why Workers / D1 / Durable Objects can't run it directly

| Service | Why it doesn't work |
|---|---|
| **Workers (JS / Python)** | No local file system, and native extensions (sqlite-vec) cannot be loaded. Python Workers run on Pyodide (WebAssembly) |
| **D1** | It is SQLite underneath, but access is limited to queries through Workers bindings. You cannot load custom extensions (sqlite-vec's `vec0`), and sqgraph's Python code does not run on D1 |
| **Durable Objects SQLite storage** | SQL is executed through a JS API, so sqgraph (Python + extension) cannot be hosted on it |

To use sqgraph on Cloudflare, you either use **Cloudflare Containers, which can run Linux containers**, or **run it somewhere else (AWS / a VPS) and put Cloudflare in front**.

### 5.2 Cloudflare Containers + Litestream (R2)

In Cloudflare Containers, each container instance is bound one-to-one to a Durable Object (DO), and you pick the instance by the DO's name. **If you always address it by the same name (e.g. `"main"`), there is only ever one writer instance.** This satisfies the "only one process may write" constraint.

Caveats:

- **The container disk is ephemeral.** The local DB file is lost on sleep (by default the container stops after a period without access) or redeploy. This assumes the entrypoint from 3.3 (restore from R2 at startup → continuous replication with Litestream).
- **Do not mount R2 via FUSE and put the DB on it**, for the reason in section 1 (WAL + network FS).
- Every wake from sleep **restores the whole DB from R2**, so the first request is slower by that time (R2 has free egress, so it costs almost nothing, but once the DB reaches several GB, waking takes tens of seconds). If you need constant responsiveness, lengthen the time before sleep. If you expect the DB to grow large, choose the setup in 5.3 (a server with a persistent disk + Cloudflare in front) (section 7).

`wrangler.jsonc` example:

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

Worker (`src/index.ts`) example:

```ts
import { Container, getContainer } from "@cloudflare/containers";

export class SqgraphContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "30m"; // how long after the last access before sleeping

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // Register secrets with `wrangler secret put OPENAI_API_KEY` etc. and pass them to the container as environment variables
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
    // Forward only requests that passed authentication (section 6) to the container. The writer instance is always the single "main"
    return getContainer(env.SQGRAPH, "main").fetch(request);
  },
};
```

- `deploy/Dockerfile` assumes the repository root as its build context. Configure wrangler to use the root as the build context (check the config key for your wrangler version).
- In `litestream.yml`, set `endpoint: https://<ACCOUNT_ID>.r2.cloudflarestorage.com` and `region: auto` for R2 (3.2). Issue the R2 API token (S3-compatible access key) with permissions limited to the target bucket.
- The Cloudflare Containers API (the `@cloudflare/containers` class and property names, `wrangler` config keys, instance types and disk limits, whether a snapshot feature exists) is still evolving. What is written here reflects the state at the time of writing; check the [official documentation](https://developers.cloudflare.com/containers/) for current syntax.

### 5.3 Running on AWS / a VPS with Cloudflare in front (the practical setup)

The DB and app live on the EC2 instance from 4.2 (or any VPS), and Cloudflare handles only **ingress and authentication**. You don't need to open any inbound port on the server.

1. Start the MCP server on the server with `--host 127.0.0.1 --port 8000`
2. Run `cloudflared` on the same server and connect `https://memory.example.com` → `http://127.0.0.1:8000` with Cloudflare Tunnel
3. Create an application for that hostname in Cloudflare Access (Zero Trust); authenticate people with SSO and agents with a **service token** (section 6)
4. Back up from Litestream to R2 (or S3)

### 5.4 Using R2 as snapshot storage (patterns B / C)

Put the snapshots from 3.4 in R2, and readers (Containers, AWS Lambda, internal PCs, etc.) download them and open with `read_only=True`. R2 has free data transfer (egress), which makes it a good fit for distributing to many readers.

---

## 6. Authenticating a public MCP server

`python -m sqgraph.memory.mcp --transport streamable-http` has **no authentication**. If you expose it on `0.0.0.0` as is, anyone who knows the URL can read and write the memory. Always put authentication in front of it.

| Method | Setup | Client (Claude Code) side |
|---|---|---|
| **Cloudflare Access service token** | Create a "Service Auth" policy on the Access application and issue a token | `claude mcp add --transport http sqgraph-memory https://memory.example.com/mcp --header "CF-Access-Client-Id: <id>" --header "CF-Access-Client-Secret: <secret>"` |
| **Bearer token at a reverse proxy** | Verify `Authorization: Bearer <token>` in Caddy / nginx, then forward to `127.0.0.1:8000` (on ECS, run it as a sidecar of the same task) | `--header "Authorization: Bearer <token>"` |
| **VPN / private network** | Allow access only from a corporate VPN / Tailscale / inside the VPC | Plain URL |

Also be sure to configure the server-side limits.

- Fix the namespace with `--group-id` (reads and writes to other `group_id`s are rejected)
- Narrow the channels readable from that MCP server with `--allowed-channels` (when specified, `read_cypher`, which would bypass the ACL, is disabled automatically)
- For read-only agents, run a separate server with `--read-only` (`add_memory` is not registered)
- The viewer (`python -m sqgraph.viewer`) opens the DB read-only but has no authentication. If you expose it, put authentication in front of it as well

> If you get `421 Misdirected Request` or similar when accessing through a reverse proxy, you may be hitting the MCP SDK's Host header validation (DNS rebinding protection). Check how to configure it for the version of the `mcp` package you use.

---

## 7. Transfer volume and cost

When the DB gets large, the concern is "won't I pay to transfer the whole DB on every start?" In short:

- **A full transfer happens only on setups where the disk disappears** (Fargate / Cloudflare Containers / Lambda). With a persistent disk (EBS), a full transfer happens only when recovering from a failure
- **Downloads from S3 within the same region are free of transfer charges.** Costs balloon only when going through a NAT Gateway, across regions, or over the internet
- In many cases, **startup time** becomes a problem before cost does

### Transfer by setup

| Setup | Full download on every start | Regular transfer |
|---|---|---|
| EC2 + EBS / ECS on EC2 + EBS | **No** (`-if-db-not-exists` restores only when the disk is empty) | Litestream change deltas (WAL) + periodic full snapshot uploads |
| ECS on Fargate | **Yes** (every task start, deploy, restart) | Same as above |
| Cloudflare Containers | **Yes** (every wake from sleep, every redeploy) | Same as above |
| Lambda (reader) | **Yes** (every execution-environment cold start, every snapshot update) | None |
| N readers in pattern B | Each of the N downloads the full DB on every snapshot update | Update frequency × number of readers × DB size |

Uploads to S3 / R2 (ingress) are free; you pay only request and storage fees.

### Cost by path (approximate)

AWS figures are approximate for US East (N. Virginia); Tokyo is somewhat higher. Check each provider's pricing page for current rates.

| Path | Price | One transfer of a 5 GB DB |
|---|---|---|
| S3 → EC2 / Fargate / Lambda in the same region | No transfer charge (request fees only) | About $0 |
| The above **through a NAT Gateway** | Processing fee about $0.045/GB | About $0.23 |
| S3 → another region | About $0.02/GB | About $0.10 |
| S3 → internet (internal PC / Cloudflare, etc.) | About $0.09/GB | About $0.45 |
| **R2** → anywhere | Free egress (read request fees only) | About $0 |

Examples of costs ballooning:

- Restarting Fargate 10 times a day through a NAT: 5 GB × 10 × $0.045 ≈ **$2.3/day**
- Distributing an hourly snapshot to 10 readers over the internet: 5 GB × 10 × 24 × $0.09 ≈ **$108/day**

An easily overlooked **storage cost**: keeping 30 days of daily snapshots is 5 GB × 30 = 150 GB, about $3.5/month on S3 Standard and about $2.3/month on R2. Use lifecycle rules to delete old ones.

### Startup time

Downloads within the same region run at roughly several hundred MB per second. 1 GB takes a few seconds; 10 GB takes tens of seconds to over a minute, slowing startup (Lambda cold start, Containers waking from sleep, Fargate deploys). A Litestream restore applies deltas to a snapshot, so it takes slightly longer than a plain download.

### DB size estimates

Most of the space is vectors (1536 dimensions × 4 bytes = about 6 KB per vector).

- Measured: in the `chatlog_vector` format (one vector per message), 711 messages take about 8 MB (about 11 KB per message)
- Estimated: `sqgraph.memory` produces 1–2 facts (each with a vector) per message, so about 10–15 KB per message
- 100,000 messages come to about 1–1.5 GB. At the scale of an internal Discord, it takes a long time to reach several GB

### Countermeasures (most effective first)

1. **Put the writer on a persistent disk** (EC2 + EBS / ECS on EC2 + EBS). A full transfer then happens only after a failure. If you expect the DB to grow large, make this your default
2. **Avoid the NAT.** Creating an S3 gateway VPC endpoint (free) keeps Fargate / Lambda traffic to S3 off the NAT
3. **Distribute through R2.** When readers are outside the region or on the internet, transfer cost drops to 0
4. **Reduce the number of downloads**
   - Have Lambda download only when `LATEST` changes (the example in 4.4 does this). Reduce cold starts with provisioned concurrency
   - Lower the snapshot update frequency to the freshness readers actually need
   - For Cloudflare Containers, lengthen `sleepAfter`
5. **Make the DB itself smaller.** `text-embedding-3-small` lets you reduce the number of dimensions (1536 → 512 shrinks vectors to about 1/3, at a small cost in search accuracy). For DBs with many deletions or updates, run `VACUUM` to reclaim free space
6. **A future option**: Litestream v0.5 has a VFS extension that reads only the needed pages from a replica on S3 without downloading the whole thing. However, sqgraph currently cannot open a DB with that VFS specified, so using it would require code changes

## 8. Operations

### Backup and restore

- **Litestream**: you can restore to any point in time from the replica. Restore under a different name with `litestream restore -config /etc/litestream.yml -o /tmp/restored.db /data/memory.db`, check the contents, then swap it in.
- **Periodic snapshots**: take the `VACUUM INTO` snapshot from 3.4 daily or so, and delete old ones with S3 / R2 lifecycle rules. This is insurance on a separate path from Litestream.
- **Restore drills**: a few times a year, restore in a separate environment and confirm that `GraphDB(path, read_only=True).stats()` and `Memory(...).stats()` return what you expect.

### Upgrading sqgraph

- The DB schema version (`_meta.schema_version`) is upgraded automatically **when the DB is opened for writing** (for example, v2 → v3 adds the relationship-index metadata column). When opened read-only, it is read without migrating.
- In patterns B / C, upgrade **the writer first**, then the readers.
- The `sqgraph.memory` storage format does not migrate old DBs where facts were stored as nodes (opening one stops with an error). When an update changes the format, rebuild the DB.

### Performance and capacity guidelines

- After a large ingestion or after building indexes, run `db.analyze()` to refresh the query planner statistics (`close()` also does this automatically).
- The vector index is a full scan via `vec0`. It is practical up to a few hundred thousand vectors; beyond that, consider moving only vector search to an external service (OpenSearch / Vectorize, etc.).
- `sqgraph.memory` ingestion calls the LLM 2–4 times per item and processes them serially. For a bot with heavy ingestion, queue the items and process them in order with a single worker (for the roughly $0.004-per-message estimate, see [the cost estimates in memory.md](memory.md#cost-estimates-gpt-54-mini-text-embedding-3-small)).

### Monitoring

- Disk usage (DB + WAL), whether the Litestream process is running, and the replica's last update time
- Application error logs (`SqgraphError`, OpenAI API 429 / 5xx)
- LLM cost (log the `UsageTracker` totals)

---

## 9. Checklist

- [ ] The check script from section 1 passes in the runtime (SQLite 3.38 or later, extension loading available)
- [ ] There is always exactly one writer instance (no autoscaling / old and new never overlap during deploys / a fixed-name DO on Cloudflare)
- [ ] The DB file is on a local disk (not on EFS / NFS / FUSE)
- [ ] Backups run via Litestream or periodic snapshots, and you have tried a restore
- [ ] What you distribute to readers is a snapshot made with `VACUUM INTO`
- [ ] If the DB is expected to grow large, the writer is on a persistent disk. If you use Fargate / Lambda, an S3 gateway VPC endpoint exists and traffic does not go through the NAT (section 7)
- [ ] There is authentication in front of the MCP server / viewer. Nothing is exposed on `0.0.0.0` without authentication
- [ ] `--group-id` / `--allowed-channels` / `--read-only` are set to fit the use case
- [ ] `OPENAI_API_KEY` and R2 / S3 credentials are passed through secret management (Secrets Manager / `wrangler secret`) and are not included in the image or repository

## References

- [Litestream](https://litestream.io/) (configuration: [Configuration File](https://litestream.io/reference/config/), [S3-compatible services](https://litestream.io/guides/s3-compatible/))
- [SQLite: Write-Ahead Logging](https://www.sqlite.org/wal.html) (limitations on network file systems)
- [Cloudflare Containers](https://developers.cloudflare.com/containers/) / [Durable Object Container API](https://developers.cloudflare.com/containers/api/durable-object-container/)
- [Cloudflare Access service tokens](https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/)
- [Amazon ECS deployment configuration (minimumHealthyPercent / maximumPercent)](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service_definition_parameters.html)
