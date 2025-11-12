# src/eval.py
import json
from vsm_ir import load_corpus, compute_idf, vectorize, cosine_sim

def precision_at_k(retrieved, relevant, k):
    retrieved_k = [d for d,_ in retrieved[:k]]
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = [d for d,_ in retrieved[:k]]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def f1(p, r): return 2*p*r/(p+r+1e-9)

def mapk(retrieved, relevant, k):
    score, hits = 0, 0
    for i, (d, _) in enumerate(retrieved[:k], 1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)

def evaluate(gold_path, docs, idf, weight="tfidf", k=5):
    gold = json.load(open(gold_path, encoding="utf-8"))
    total_p = total_r = total_f = total_map = 0
    for query, relevant in gold.items():
        qvec = vectorize(query.lower().split(), idf, sublinear=(weight=="tfidf_sublinear"))
        retrieved = []
        for fname, tokens in docs.items():
            dvec = vectorize(tokens, idf, sublinear=(weight=="tfidf_sublinear"))
            retrieved.append((fname, cosine_sim(dvec, qvec)))
        retrieved = sorted(retrieved, key=lambda x: x[1], reverse=True)[:k]
        p, r = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
        total_p += p; total_r += r; total_f += f1(p,r); total_map += mapk(retrieved, relevant, k)
    n = len(gold)
    return {"P": total_p/n, "R": total_r/n, "F1": total_f/n, "MAP": total_map/n}
