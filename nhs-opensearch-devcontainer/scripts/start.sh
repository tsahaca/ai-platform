#!/usr/bin/env bash
set -euo pipefail

docker compose up -d

echo "Waiting for OpenSearch..."
until curl -fsS http://localhost:9200 >/dev/null; do
  sleep 3
done

echo "OpenSearch: http://localhost:9200"
echo "Dashboards: http://localhost:5601"
