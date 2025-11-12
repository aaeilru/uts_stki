# src/boolean_ir.py
import os
from collections import defaultdict

def build_inverted_index(processed_dir):
    index = defaultdict(set)
    for fname in os.listdir(processed_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                tokens = f.read().split()
            for token in set(tokens):
                index[token].add(fname)
    return index

def boolean_retrieve(query_tokens, index, op="AND"):
    sets = [index.get(t, set()) for t in query_tokens]
    if not sets:
        return []
    if op == "AND":
        result = set.intersection(*sets)
    else:
        result = set.union(*sets)
    return list(result)
