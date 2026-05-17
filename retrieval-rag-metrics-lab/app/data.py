import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_docs(path: str | Path = ROOT / "data" / "docs.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_queries(path: str | Path = ROOT / "data" / "queries.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
