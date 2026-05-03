Great—you’ve loaded the data, now let’s **actually see it in OpenSearch Dashboards (UI)** step-by-step.

---

# 🖥️ Step-by-step: View your data in OpenSearch Dashboards

## 🔹 1. Open Dashboard

Go to:

```text
http://localhost:5601
```

---

## 🔹 2. Create Index Pattern (VERY IMPORTANT)

👉 This is the step most people miss.

### Navigation:

```text
☰ Menu → Stack Management → Index Patterns → Create index pattern
```

### Enter:

```text
vector_lab_docs
```

👉 (must match your index name from config)

Click:

```text
Next → Create index pattern
```

---

## 🔹 3. Go to Discover

```text
☰ Menu → Discover
```

Select:

```text
vector_lab_docs
```

---

## 🔍 4. You should now see your documents

You’ll see rows like:

```text
doc-001 | AWS Lambda | serverless | AWS Lambda is a serverless compute...
doc-002 | Amazon EC2 | compute   | Amazon EC2 provides...
```

---

## ⚠️ Important: Why you DON’T see embeddings

You **won’t see the full vector clearly** because:

```json
"embedding": [0.12, -0.44, 0.91, ... 384 values]
```

👉 UI hides it or truncates it

---

## 🔧 5. Expand a document

Click the small **▶ arrow** next to a row

You’ll see:

```json
{
  "id": "doc-001",
  "title": "AWS Lambda",
  "category": "serverless",
  "text": "...",
  "embedding": [0.12, -0.44, ...]
}
```

---

# 🔍 Alternative: Use Dev Tools (better for debugging)

## Go to:

```text
☰ Menu → Dev Tools
```

Run:

```json
GET vector_lab_docs/_search
{
  "size": 10
}
```

---

## Output:

```json
{
  "hits": {
    "hits": [
      {
        "_id": "doc-001",
        "_source": {
          "title": "AWS Lambda",
          "category": "serverless",
          "text": "...",
          "embedding": [...]
        }
      }
    ]
  }
}
```

---

# 🔎 Filter specific fields (clean view)

```json
GET vector_lab_docs/_search
{
  "_source": ["title", "category", "text"],
  "size": 10
}
```

👉 Cleaner output without embeddings

---

# 🧠 Pro Tip (very useful)

## Add columns in UI

In Discover:

* Click **+** next to:

  * `title`
  * `category`
  * `text`

Now your table looks like:

```text
title           | category   | text
-----------------------------------------
AWS Lambda      | serverless | ...
Amazon EC2      | compute    | ...
```

---

# 🔥 Bonus: Run a vector search in Dev Tools

```json
GET vector_lab_docs/_search
{
  "size": 3,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, -0.2, ...], 
        "k": 3
      }
    }
  }
}
```

👉 (use a real embedding vector from your Python code)

---

# 🚀 What you just learned

| Tool          | Purpose               |
| ------------- | --------------------- |
| Discover      | Browse documents      |
| Dev Tools     | Run queries           |
| Index Pattern | Required to view data |
| Expand row    | See full JSON         |

---

# ⚡ Common issues (you might hit)

### ❌ No data showing

Fix:

```bash
python scripts/load_opensearch.py
```

---

### ❌ Index not found

Check:

```json
GET _cat/indices?v
```

---

### ❌ Empty Discover

👉 Refresh index pattern

---

# 🧠 Mental model

```text
OpenSearch Dashboard:
   UI to view data

Discover:
   Table view of documents

Dev Tools:
   Query engine
```

---

# If you want next

I can show you:

👉 how to **visualize search relevance (BM25 vs vector vs hybrid) in dashboard**
👉 how to build **custom dashboards (charts + filters)**
👉 how to **debug KNN queries visually**

Just say 👍
