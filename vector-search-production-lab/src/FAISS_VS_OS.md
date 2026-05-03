# FAISS vs OPENSERACH 

In this current lab, FAISS is mostly a lightweight local engine, not a capability that OpenSearch fundamentally lacks.

## What FAISS can do that OpenSearch usually cannot do as directly:
1. In-process vector search inside your Python runtime.
This means no HTTP/network hop, useful for very low-latency local experiments.
2. Full low-level control of ANN index internals.

FAISS exposes many index families and tuning knobs very directly (for example IVF, PQ, HNSW variants, exact flat, GPU-specific workflows).

3. Easy embedded/offline usage.
You can ship a local index with an app, run on edge machines, and avoid running a separate search service.

## What OpenSearch can do that FAISS alone does not provide:
1. Persistence, replication, and cluster scaling.
2. Built-in BM25 lexical search plus filters/aggregations/security/APIs.
3. Operational features for production systems (multi-tenant patterns, dashboards, service interfaces).

## So for this project:
- FAISS in faiss_store.py gives simple local vector retrieval.
- OpenSearch in opensearch_store.py gives production-style search platform behavior.

Bottom line: FAISS is not “more capable” here, it is “closer to the metal.” OpenSearch is broader and production-oriented.