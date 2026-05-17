#!/usr/bin/env bash
set -euo pipefail
URL="${OPENSEARCH_URL:-http://localhost:9200}"
until curl -fsS "$URL" >/dev/null; do
  echo "Waiting for OpenSearch at $URL ..."
  sleep 3
done
echo "OpenSearch is ready: $URL"
