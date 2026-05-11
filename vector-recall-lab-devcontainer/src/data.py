DOCUMENTS = [
    {"id": "doc-001", "title": "AWS Lambda", "category": "serverless", "text": "AWS Lambda is a serverless compute service that runs code without provisioning servers."},
    {"id": "doc-002", "title": "Amazon EC2", "category": "compute", "text": "Amazon EC2 provides resizable virtual machines for cloud compute workloads."},
    {"id": "doc-003", "title": "Amazon S3", "category": "storage", "text": "Amazon S3 is object storage built to store and retrieve any amount of data."},
    {"id": "doc-004", "title": "Azure Functions", "category": "serverless", "text": "Azure Functions is a serverless compute platform for event-driven applications."},
    {"id": "doc-005", "title": "Google Cloud Run", "category": "containers", "text": "Google Cloud Run executes containers in a managed serverless environment."},
    {"id": "doc-006", "title": "Kubernetes", "category": "containers", "text": "Kubernetes orchestrates containerized applications across clusters."},
    {"id": "doc-007", "title": "Low Cost Compute", "category": "cost", "text": "Cheap cloud compute options include spot instances, serverless functions, and autoscaling."},
    {"id": "doc-008", "title": "Serverless Architecture", "category": "architecture", "text": "Low cost serverless architecture can reduce operational overhead and infrastructure management."},
    {"id": "doc-009", "title": "OpenSearch BM25", "category": "search", "text": "OpenSearch uses inverted indexes and BM25 scoring for lexical keyword search."},
    {"id": "doc-010", "title": "Vector Search", "category": "search", "text": "Vector search uses embeddings and nearest neighbor algorithms to find semantically similar content."},
    {"id": "doc-011", "title": "Hybrid Search", "category": "search", "text": "Hybrid search combines lexical BM25 keyword matching with semantic vector similarity."},
    {"id": "doc-012", "title": "HNSW Index", "category": "vector-index", "text": "HNSW is a graph based approximate nearest neighbor algorithm used for fast vector search."},
    {"id": "doc-013", "title": "FAISS Flat Index", "category": "vector-index", "text": "FAISS IndexFlatIP performs exact inner product search and is useful as a ground truth baseline."},
    {"id": "doc-014", "title": "Cosine Similarity", "category": "math", "text": "Cosine similarity measures the angle between vectors and is commonly used with normalized text embeddings."},
    {"id": "doc-015", "title": "L2 Distance", "category": "math", "text": "L2 distance is Euclidean distance and measures straight line distance between vectors."},
    {"id": "doc-016", "title": "Vector Metadata Filtering", "category": "production", "text": "Production vector search systems often combine semantic search with metadata filters such as category, tenant, region, or document type."},
    {"id": "doc-017", "title": "RAG Retrieval", "category": "rag", "text": "Retrieval augmented generation retrieves relevant documents from search indexes before sending context to an LLM."},
    {"id": "doc-018", "title": "Cross Encoder Reranking", "category": "reranking", "text": "A cross encoder reranker can improve relevance by scoring query and document pairs after initial retrieval."},
    {"id": "doc-019", "title": "OpenSearch Dashboards", "category": "observability", "text": "OpenSearch Dashboards lets users inspect indexes, run queries, create visualizations, and explore documents."},
    {"id": "doc-020", "title": "Search Evaluation", "category": "evaluation", "text": "Vector search quality is evaluated with recall at k, precision, latency, and relevance judgments."}
]

EVAL_QUERIES = [
    {"query": "cheap compute", "expected": ["doc-007", "doc-001", "doc-002", "doc-008"]},
    {"query": "serverless compute without managing servers", "expected": ["doc-001", "doc-004", "doc-008"]},
    {"query": "keyword search inverted index", "expected": ["doc-009", "doc-011"]},
    {"query": "semantic vector nearest neighbor search", "expected": ["doc-010", "doc-012", "doc-013"]},
    {"query": "what is hnsw", "expected": ["doc-012"]},
    {"query": "evaluate vector search accuracy", "expected": ["doc-020"]},
    {"query": "combine bm25 with semantic search", "expected": ["doc-011"]},
    {"query": "rerank query document pairs", "expected": ["doc-018"]},
    {"query": "metadata filter by tenant category", "expected": ["doc-016"]},
    {"query": "rag retrieval documents llm", "expected": ["doc-017"]}
]
