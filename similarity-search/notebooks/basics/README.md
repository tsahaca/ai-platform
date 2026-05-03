🧠 How AI vectors are represented in Python (NumPy)
At the simplest level, an AI vector = a NumPy array of numbers.
1. Basic Representation
import numpy as np

# A simple 3-dimensional vector
v = np.array([0.2, -0.5, 1.3])

print(v)
print(type(v))
👉 Output:
[ 0.2 -0.5  1.3]
<class 'numpy.ndarray'>
📏 2. High-dimensional vectors (like embeddings)
In AI (NLP, vision, etc.), vectors are usually hundreds or thousands of dimensions.
Example (like SentenceTransformer output):

embedding = np.random.rand(384).astype("float32")  # 384-dim vector

print(embedding.shape)
👉 Output:
(384,)
Each number = a feature
Entire array = semantic representation
🧾 3. Multiple vectors (matrix form)
When you have many embeddings:
embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
], dtype="float32")

print(embeddings.shape)
👉 Output:
(3, 3)   # 3 vectors, each of dimension 3
💡 This is how FAISS / vector DBs store data:
Rows = documents
Columns = vector dimensions
📐 4. Vector operations (core AI math)
Dot Product (similarity)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot = np.dot(a, b)
print(dot)  # 32
Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(a, b))
Euclidean Distance
distance = np.linalg.norm(a - b)
print(distance)
🧬 5. Normalized vectors (very important in AI)
In your FAISS example:
embedding = embedding / np.linalg.norm(embedding)
👉 This makes:
vector length = 1
dot product = cosine similarity
🧠 6. Real-world example (SentenceTransformer)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["AI is powerful", "Machine learning is amazing"]
embeddings = model.encode(sentences)

print(type(embeddings))
print(embeddings.shape)
👉 Output:
<class 'numpy.ndarray'>
(2, 384)
⚡ Key Takeaways
AI vectors = NumPy arrays (ndarray)
Shape:
(d,) → single vector
(n, d) → multiple vectors
Data type:
usually float32 (efficient for ML)
Used for:
similarity search
clustering
classification
🧩 Mapping to your FAISS code
embeddings = self.model.encode(texts, normalize_embeddings=True).astype("float32")
encode() → returns NumPy array
normalize_embeddings=True → unit vectors
.astype("float32") → required by FAISS