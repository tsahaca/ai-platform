import numpy as np

# -----------------------------
# Documents
# -----------------------------
a = "purple is the best city in the forest".split()
b = "there is an art to getting your way and throwing bananas on to the street is not it".split()
c = "it is not often you find soggy bananas on the street".split()

docs = [a, b, c]
doc_names = ["a", "b", "c"]


# -----------------------------
# Vocabulary
# -----------------------------
vocab = sorted(set(word for doc in docs for word in doc))
word_to_index = {word: i for i, word in enumerate(vocab)}


# -----------------------------
# 1. TF-IDF Matrix
# -----------------------------
def build_tfidf_matrix():
    N = len(docs)
    V = len(vocab)

    tf_matrix = np.zeros((N, V))
    idf_vector = np.zeros(V)

    for i, doc in enumerate(docs):
        for word in doc:
            j = word_to_index[word]
            tf_matrix[i, j] += 1

        tf_matrix[i] = tf_matrix[i] / len(doc)

    for word, j in word_to_index.items():
        df = sum(1 for doc in docs if word in doc)
        idf_vector[j] = np.log((N + 1) / (df + 1)) + 1

    tfidf_matrix = tf_matrix * idf_vector

    return tfidf_matrix, idf_vector


def tfidf(word):
    word = word.lower()

    if word not in word_to_index:
        return np.zeros(len(docs))

    tfidf_matrix, _ = build_tfidf_matrix()
    j = word_to_index[word]

    return tfidf_matrix[:, j]


# -----------------------------
# 2. Cosine Similarity Search
# -----------------------------
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)


def search_tfidf(query):
    tfidf_matrix, idf_vector = build_tfidf_matrix()

    query_tokens = query.lower().split()
    query_vector = np.zeros(len(vocab))

    for word in query_tokens:
        if word in word_to_index:
            j = word_to_index[word]
            query_vector[j] += 1

    if len(query_tokens) > 0:
        query_vector = query_vector / len(query_tokens)

    query_vector = query_vector * idf_vector

    scores = []

    for i, doc_vector in enumerate(tfidf_matrix):
        score = cosine_similarity(query_vector, doc_vector)
        scores.append((doc_names[i], score, " ".join(docs[i])))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# -----------------------------
# 3. BM25 Comparison
# -----------------------------
def bm25_score(query, k1=1.5, b=0.75):
    query_tokens = query.lower().split()

    N = len(docs)
    avgdl = np.mean([len(doc) for doc in docs])

    scores = []

    for doc_index, doc in enumerate(docs):
        score = 0.0
        doc_len = len(doc)

        for word in query_tokens:
            f = doc.count(word)
            df = sum(1 for d in docs if word in d)

            if df == 0:
                continue

            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * (doc_len / avgdl))

            score += idf * (numerator / denominator)

        scores.append((doc_names[doc_index], score, " ".join(doc)))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# -----------------------------
# Demo
# -----------------------------
def main():
    print("\nVocabulary:")
    print(vocab)
    
    print("\nTF-IDF scores for word = bananas:")
    print(tfidf("bananas"))
    
    print("\nFull TF-IDF Matrix:")
    tfidf_matrix, _ = build_tfidf_matrix()
    print(tfidf_matrix)
    
    print("\nTF-IDF Cosine Similarity Search:")
    for name, score, text in search_tfidf("bananas street"):
        print(f"{name}: {score:.4f} -> {text}")
    
    print("\nBM25 Search:")
    for name, score, text in bm25_score("bananas street"):
        print(f"{name}: {score:.4f} -> {text}")

if __name__ == "__main__":
    main()