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
