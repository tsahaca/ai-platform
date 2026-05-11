import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus():
    return read_jsonl(DATA / "corpus.jsonl")


def load_queries():
    return read_jsonl(DATA / "queries.jsonl")


def load_qrels():
    rows = read_jsonl(DATA / "qrels.jsonl")
    qrels = {}
    for row in rows:
        qrels.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return qrels
