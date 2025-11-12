# src/vsm_ir.py
import math
import os
from collections import Counter, defaultdict

def load_corpus(processed_dir):
    docs = {}
    for fname in os.listdir(processed_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                docs[fname] = f.read().split()
    return docs

def compute_idf(docs):
    N = len(docs)
    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1
    return {t: math.log(N / df[t]) for t in df}

def vectorize(doc_tokens, idf, sublinear=False):
    tf = Counter(doc_tokens)
    vec = {}
    for term, freq in tf.items():
        tf_val = (1 + math.log(freq)) if sublinear else freq
        vec[term] = tf_val * idf.get(term, 0)
    return vec

def cosine_sim(v1, v2):
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in v1)
    n1 = math.sqrt(sum(v**2 for v in v1.values()))
    n2 = math.sqrt(sum(v**2 for v in v2.values()))
    return dot / (n1 * n2 + 1e-9)
