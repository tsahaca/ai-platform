import numpy as np

# Documents
a = "purple is the best city in the forest".split()
b = "there is an art to getting your way and throwing bananas on to the street is not it".split()
c = "it is not often you find soggy bananas on the street".split()

docs = [a, b, c]

def tfidf(word):
    word = word.lower()
    N = len(docs)

    # --- Term Frequency (TF) ---
    tf = np.array([
        doc.count(word) / len(doc)
        for doc in docs
    ])

    # --- Document Frequency (DF) ---
    df = sum(1 for doc in docs if word in doc)

    # --- Inverse Document Frequency (IDF) ---
    # add small epsilon to avoid division by zero
    idf = np.log((N + 1) / (df + 1)) + 1

    # --- TF-IDF ---
    tfidf_scores = tf * idf

    return {
        "tf": tf,
        "idf": idf,
        "tfidf": tfidf_scores
    }


def main():
    result = tfidf("bananas")
    print("TF:", result["tf"])
    print("IDF:", result["idf"])
    print("TF-IDF:", result["tfidf"])
    
    # words = ["bananas", "street", "is", "purple"]
    # for word in words:
    #     result = tfidf(word)
    #     print(f"\nWord: '{word}'")
    #     print(f"  TF:    {result['tf']}")
    #     print(f"  IDF:   {result['idf']:.4f}")
    #     print(f"  TF-IDF: {result['tfidf']}")

if __name__ == "__main__":
    main()