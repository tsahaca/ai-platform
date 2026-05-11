# NHS Conditions OpenSearch Vector Lab

Zip-ready devcontainer for running **OpenSearch + OpenSearch Dashboards** locally and indexing NHS Conditions A-Z pages as vector documents.

## What is included

- `docker-compose.yaml` — OpenSearch single-node + OpenSearch Dashboards
- `.devcontainer/devcontainer.json` — VS Code Dev Containers setup
- `app/create_index.py` — creates a `knn_vector` index
- `app/nhs_extract.py` — extracts condition links and page sections from NHS Conditions A-Z
- `app/ingest.py` — embeds condition chunks and indexes them into OpenSearch
- `app/search.py` — semantic vector search demo

## Start OpenSearch

```bash
./scripts/start.sh
```

OpenSearch:

```text
http://localhost:9200
```

OpenSearch Dashboards:

```text
http://localhost:5601
```

This local lab disables the OpenSearch security plugin for easy development. Do not use this Compose file as-is for production.

## Install Python dependencies

Inside the devcontainer this runs automatically. Outside the devcontainer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create index

```bash
python -m app.create_index
```

## Ingest sample NHS condition pages

By default, `app.ingest` indexes the first 20 condition pages from NHS Conditions A-Z.

```bash
python -m app.ingest
```

To increase coverage, edit this line in `app/ingest.py`:

```python
ingest(limit=20)
```

For the full NHS list, set `limit=None`, but crawl responsibly and check NHS terms before large-scale use.

## Run semantic search

```bash
python -m app.search "chest pain when exercising"
python -m app.search "itchy red skin rash"
python -m app.search "headache and sensitivity to light"
```

## Example indexed document

```json
{
  "doc_id": "angina__003",
  "condition_id": "angina",
  "title": "Angina",
  "category": "A",
  "section": "Symptoms of angina",
  "url": "https://www.nhs.uk/conditions/angina/",
  "chunk_text": "Condition: Angina\nSection: Symptoms of angina\nText: ...",
  "embedding": [0.0123, -0.0091]
}
```

## Useful curl checks

```bash
curl http://localhost:9200
curl http://localhost:9200/_cluster/health?pretty
curl http://localhost:9200/nhs_conditions/_count?pretty
```

## Stop services

```bash
./scripts/stop.sh
```

## Reset everything, including OpenSearch data volume

```bash
./scripts/reset.sh
```

## Notes

OpenSearch official Docker documentation recommends Docker Compose for defining OpenSearch and OpenSearch Dashboards together, and OpenSearch Dashboards requires a running OpenSearch cluster. This project follows that local-development pattern.
