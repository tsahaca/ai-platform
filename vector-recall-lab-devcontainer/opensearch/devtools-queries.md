# OpenSearch Dev Tools Queries

```json
GET _cat/indices?v
```

```json
GET vector_lab_docs/_search
{
  "_source": ["id", "title", "category", "text"],
  "size": 20
}
```

```json
GET vector_lab_docs/_mapping
```

```json
PUT vector_lab_docs/_settings
{
  "index.knn.algo_param.ef_search": 100
}
```
