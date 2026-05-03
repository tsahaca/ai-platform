import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_FILE = ROOT / "docs" / "sample_docs.json"

def load_docs():
    with DOCS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)
