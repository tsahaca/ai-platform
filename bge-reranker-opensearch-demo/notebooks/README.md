# Notebooks

Open VS Code inside the devcontainer and create notebooks from these commands:

```python
from app.ingest import ingest
ingest()

from app.search import hybrid_candidates, rerank
query = "heart attack symptoms"
candidates = hybrid_candidates(query, 8)
rerank(query, candidates, 5)
```
