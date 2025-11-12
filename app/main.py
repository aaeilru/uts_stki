import os
from src.vsm_ir import load_corpus, compute_idf, vectorize, cosine_sim
from src.preprocess import preprocess_text

# === Konfigurasi path ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# === Load data & buat model ===
print("🔍 Memuat korpus dari data/processed...")
corpus = load_corpus(DATA_DIR)
idf = compute_idf(corpus)

print(f"✅ {len(corpus)} dokumen berhasil dimuat.\n")

# === Fungsi pencarian ===
def search_vsm(query, top_k=5):
    # Preprocess query sama seperti dokumen
    query_tokens = preprocess_text(query)
    query_vec = vectorize(query_tokens, idf, sublinear=True)

    # Hitung skor cosine similarity
    scores = []
    for filename, tokens in corpus.items():
        doc_vec = vectorize(tokens, idf, sublinear=True)
        sim = cosine_sim(query_vec, doc_vec)
        scores.append((filename, sim))

    # Urutkan berdasarkan skor tertinggi
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return scores

# === Interface sederhana ===
if __name__ == "__main__":
    print("=== Mini Search Engine STKI ===")
    while True:
        query = input("\nMasukkan kata kunci (atau 'exit' untuk keluar): ").strip()
        if query.lower() == "exit":
            break

        results = search_vsm(query, top_k=5)
        print(f"\n🔎 Hasil pencarian untuk: '{query}'\n")
        for rank, (filename, score) in enumerate(results, start=1):
            print(f"{rank}. {filename:<30} skor: {score:.4f}")
