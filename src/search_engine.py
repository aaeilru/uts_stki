# src/search_engine.py
import os
import argparse
from boolean_ir import build_inverted_index, boolean_retrieve
from vsm_ir import load_corpus, compute_idf, vectorize, cosine_sim

def main():
    parser = argparse.ArgumentParser(description="Mini Search Engine CLI - STKI UTS")
    parser.add_argument("--model", choices=["boolean", "vsm"], default="vsm")
    parser.add_argument("--weight", choices=["tfidf", "tfidf_sublinear"], default="tfidf")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")

    query_tokens = args.query.lower().split()

    if args.model == "boolean":
        index = build_inverted_index(processed_dir)
        results = boolean_retrieve(query_tokens, index, op="AND")
        print(f"\n🔍 Boolean Search Results for: {args.query}\n")
        for r in results:
            print(f"• {r}")
    else:
        docs = load_corpus(processed_dir)
        idf = compute_idf(docs)
        qvec = vectorize(query_tokens, idf, sublinear=(args.weight=="tfidf_sublinear"))

        scored = []
        for fname, tokens in docs.items():
            dvec = vectorize(tokens, idf, sublinear=(args.weight=="tfidf_sublinear"))
            score = cosine_sim(dvec, qvec)
            scored.append((fname, score))
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:args.k]

        print(f"\n🔍 VSM Results for: {args.query} (Weight: {args.weight})\n")
        for i, (fname, score) in enumerate(scored, 1):
            print(f"{i}. {fname:<30} skor={score:.4f}")

if __name__ == "__main__":
    main()
