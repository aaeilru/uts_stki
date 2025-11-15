import os
import argparse
import math
from collections import Counter, defaultdict

# prefer existing VSM class if available, else minimal fallback
try:
    from vsm_ir import VSMRetrieval
    VSM_AVAILABLE = True
except Exception:
    VSM_AVAILABLE = False

# boolean IR (fallback simple)
def build_inverted_index(processed_dir):
    index = defaultdict(set)
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                tokens = f.read().split()
            for t in set(tokens):
                index[t].add(fname)
    return index

def boolean_retrieve(query_tokens, index, op="AND"):
    sets = [index.get(t, set()) for t in query_tokens]
    if not sets:
        return []
    if op == "AND":
        result = set.intersection(*sets)
    else:
        result = set.union(*sets)
    return sorted(list(result))

# Simple BM25 implementation (corpus: list of token lists)
def build_bm25(corpus):
    N = len(corpus)
    df = Counter()
    for doc in corpus:
        for t in set(doc):
            df[t] += 1
    avgdl = sum(len(d) for d in corpus) / max(1, N)
    return {"N": N, "df": df, "avgdl": avgdl}

def bm25_score_for_doc(query_tokens, doc_tokens, bm25_index, k1=1.5, b=0.75):
    score = 0.0
    N = bm25_index["N"]
    df = bm25_index["df"]
    avgdl = bm25_index["avgdl"]
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    for term in query_tokens:
        f = tf.get(term, 0)
        df_t = df.get(term, 0)
        # BM25 idf (with small smoothing)
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-9)
        denom = f + k1 * (1 - b + b * dl / avgdl)
        if denom > 0:
            score += idf * (f * (k1 + 1)) / denom
    return score

# helper: explain top terms for a document using sklearn tfidf if available
def explain_top_terms_from_vsm(vsm_obj, doc_index, top_n=5):
    """
    Requires vsm_obj to have: vectorizer and tfidf_matrix (sparse)
    Returns list of (term, weight) for top_n terms in the doc.
    """
    try:
        import numpy as np
        # getattr to avoid breaking if attributes missing
        vec = vsm_obj.tfidf_matrix[doc_index].toarray().flatten()
        features = vsm_obj.vectorizer.get_feature_names_out()
        top_idx = vec.argsort()[::-1][:top_n]
        return [(features[i], float(vec[i])) for i in top_idx if vec[i] > 0]
    except Exception:
        return []

def run_vsm_cli(processed_dir, query, k=5, weight="tfidf"):
    # Use VSMRetrieval if available (preferred)
    if VSM_AVAILABLE:
        vsm = VSMRetrieval(processed_dir=None)  # VSMRetrieval handles path automatically
        vsm.load_processed_docs()
        vsm.build_tfidf()
        # if weight handled in VSMRetrieval, not here; VSMRetrieval.rank returns doc/score/snippet
        results = vsm.rank(query, k=k)
        # add explain (top terms per doc)
        explained = []
        for r in results:
            try:
                doc_idx = vsm.doc_ids.index(r["doc_id"])
                top_terms = explain_top_terms_from_vsm(vsm, doc_idx, top_n=4)
            except Exception:
                top_terms = []
            explained.append({"doc_id": r["doc_id"], "score": r["score"], "snippet": r["snippet"], "top_terms": top_terms})
        return explained
    else:
        # fallback: manual TF-IDF using simple tf-idf (not sklearn)
        # load corpus
        docs = {}
        for fname in sorted(os.listdir(processed_dir)):
            if fname.endswith(".txt"):
                with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                    docs[fname] = f.read().split()
        # build idf
        N = len(docs)
        df = Counter()
        for tokens in docs.values():
            for t in set(tokens):
                df[t] += 1
        idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
        # vectorize query
        q_tokens = query.lower().split()
        # compute doc scores
        scored = []
        for fname, tokens in docs.items():
            # compute cosine based on tfidf (sublinear option)
            tf_q = Counter(q_tokens)
            tf_d = Counter(tokens)
            # build combined vocab
            vocab = set(q_tokens) | set(tokens)
            # build vectors
            qvec = []
            dvec = []
            for term in vocab:
                # tf (query)
                tq = tf_q.get(term, 0)
                td = tf_d.get(term, 0)
                if weight == "tfidf_sublinear":
                    tq = 1 + math.log(tq) if tq > 0 else 0
                    td = 1 + math.log(td) if td > 0 else 0
                # idf default 0 if missing
                idfv = idf.get(term, math.log((N + 1) / 1 + 1))
                qvec.append(tq * idfv)
                dvec.append(td * idfv)
            # cosine
            import numpy as np
            qv = np.array(qvec); dv = np.array(dvec)
            denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
            score = float(np.dot(qv, dv) / denom) if denom > 0 else 0.0
            scored.append((fname, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        # prepare results with top_terms naive (most frequent tokens in doc)
        results = []
        for fname, score in scored[:k]:
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                text = f.read()
            tokens = text.split()
            top_terms = Counter(tokens).most_common(4)
            snippet = " ".join(tokens[:25])
            results.append({"doc_id": fname, "score": score, "snippet": snippet, "top_terms": top_terms})
        return results

def run_boolean_cli(processed_dir, query, k=5, op="OR"):
    """
    Boolean retrieval sederhana:
    - op: "OR" (default) agar hasil tidak terlalu ketat.
    - output: list of dict {doc_id, score, snippet, top_terms}
      sehingga kompatibel dengan printer di main().
    """
    index = build_inverted_index(processed_dir)
    qtokens = query.lower().split()

    # gunakan AND kalau mau strict, OR kalau mau lebih longgar
    matched = boolean_retrieve(qtokens, index, op=op)

    results = []
    for fname in matched[:k]:
        fpath = os.path.join(processed_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                tokens = f.read().split()
        except FileNotFoundError:
            tokens = []
            snippet = ""
        else:
            snippet = " ".join(tokens[:25])

        # skor sederhana: jumlah kata query yang muncul di dokumen
        tf_doc = Counter(tokens)
        score = sum(1 for t in set(qtokens) if t in tf_doc)

        # top_terms = kata query yang memang muncul di dokumen
        top_terms = [t for t in qtokens if t in tf_doc]

        results.append({
            "doc_id": fname,
            "score": float(score),
            "snippet": snippet,
            "top_terms": top_terms
        })
    return results


def run_bm25_cli(processed_dir, query, k=5):
    # load corpus as token lists and filenames
    filenames = []
    corpus = []
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith(".txt"):
            filenames.append(fname)
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                corpus.append(f.read().split())
    bm25_index = build_bm25(corpus)
    qtokens = query.lower().split()
    scores = [(filenames[i], bm25_score_for_doc(qtokens, corpus[i], bm25_index)) for i in range(len(filenames))]
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for fname, score in scores[:k]:
        with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
            snippet = " ".join(f.read().split()[:25])
        # top terms naive
        top_terms = Counter(open(os.path.join(processed_dir, fname), encoding="utf-8").read().split()).most_common(4)
        results.append({"doc_id": fname, "score": score, "snippet": snippet, "top_terms": top_terms})
    return results

def main():
    parser = argparse.ArgumentParser(description="Mini Search Engine CLI - STKI UTS")
    parser.add_argument("--model", choices=["boolean", "vsm", "bm25"], default="vsm")
    parser.add_argument("--weight", choices=["tfidf", "tfidf_sublinear"], default="tfidf")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")

    if args.model == "boolean":
        results = run_boolean_cli(processed_dir, args.query, k=args.k)
    elif args.model == "bm25":
        results = run_bm25_cli(processed_dir, args.query, k=args.k)
    else:
        # vsm
        results = run_vsm_cli(processed_dir, args.query, k=args.k, weight=args.weight)

    # print nicely with explain (top_terms)
    if args.model == "vsm":
        header = f"model={args.model}, weight={args.weight}"
    else:
        header = f"model={args.model}"

    print(f"\n🔍 Search results ({header}) for: \"{args.query}\"\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['doc_id']:<30} score={r['score']:.4f}")
        if r.get("top_terms"):
            terms = ", ".join([t if isinstance(t, str) else f'{t[0]}({t[1]})' for t in (r['top_terms'])])
            print(f"    top_terms: {terms}")
        if r.get("snippet"):
            print(f"    snippet: {r['snippet'][:160]}...\n")
    print("Done.\n")

if __name__ == "__main__":
    main()