import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VSMRetrieval:
    def __init__(self, processed_dir=None):
        base = os.path.dirname(os.path.abspath(__file__))

        if processed_dir is None:
            processed_dir = os.path.join(base, "..", "data", "processed")

        self.processed_dir = os.path.normpath(processed_dir)

        self.docs = []
        self.doc_ids = []
        self.vectorizer = None
        self.tfidf_matrix = None

    # LOAD DOKUMEN PREPROCESSED
    def load_processed_docs(self):
        files = sorted(os.listdir(self.processed_dir))

        for fname in files:
            if fname.endswith(".txt"):
                with open(os.path.join(self.processed_dir, fname), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                self.docs.append(text)
                self.doc_ids.append(fname)

        print(f"[INFO] Loaded {len(self.docs)} documents.")

    # MEMBANGUN TF-IDF MATRIX
    def build_tfidf(self):
        if not self.docs:
            raise ValueError("Dokumen belum dimuat.")

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

        print(f"[INFO] TF-IDF shape: {self.tfidf_matrix.shape} (docs x terms)")

    # QUERY → VECTOR
    def vectorize_query(self, query):
        return self.vectorizer.transform([query.lower().strip()])

    # RANKING (TOP-K)
    def rank(self, query, k=5):
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF belum dibuat.")

        q_vec = self.vectorize_query(query)
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            snippet = self.docs[idx][:120].replace("\n", " ")
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(scores[idx]),
                "snippet": snippet
            })

        return results

    # METRIK EVALUASI
    def precision_at_k(self, retrieved, relevant, k):
        retrieved_k = retrieved[:k]
        rel = set(relevant)
        hit = sum(1 for d in retrieved_k if d in rel)
        return hit / k

    def average_precision(self, retrieved, relevant, k):
        rel = set(relevant)
        score = 0.0
        hit = 0

        for i in range(min(k, len(retrieved))):
            if retrieved[i] in rel:
                hit += 1
                score += hit / (i + 1)

        return score / max(len(relevant), 1)

    def ndcg_at_k(self, retrieved, relevant, k):
        rel = set(relevant)
        dcg = 0.0

        for i in range(min(k, len(retrieved))):
            if retrieved[i] in rel:
                dcg += 1.0 / np.log2(i + 2)

        ideal_hits = min(k, len(relevant))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    # EVALUASI MENGGUNAKAN gold.json
    def load_gold(self, gold_path):
        with open(gold_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_query(self, query, gold_dict, k=5):
        if query not in gold_dict:
            raise ValueError(f"Query '{query}' tidak ditemukan di gold.json.")

        relevant = gold_dict[query]
        ranked = self.rank(query, k)
        retrieved_ids = [r["doc_id"] for r in ranked]

        return {
            "Precision@k": self.precision_at_k(retrieved_ids, relevant, k),
            "MAP@k": self.average_precision(retrieved_ids, relevant, k),
            "nDCG@k": self.ndcg_at_k(retrieved_ids, relevant, k)
        }


# DEMO
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    gold_path = os.path.join(base, "data", "gold.json")

    vsm = VSMRetrieval()
    vsm.load_processed_docs()
    vsm.build_tfidf()

    gold = vsm.load_gold(gold_path)

    query = list(gold.keys())[0]   # otomatis pakai query pertama di gold.json
    print(f"\nQuery uji: {query}")

    results = vsm.rank(query, k=5)
    print("\n=== TOP-5 RANKING ===")
    for r in results:
        print(f"{r['doc_id']} | {r['score']:.4f} | {r['snippet']}")

    eval = vsm.evaluate_query(query, gold, k=5)
    print("\n=== UJI WAJIB ===")
    print(eval)
