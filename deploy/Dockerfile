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
