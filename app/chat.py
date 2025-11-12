import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # penting agar src bisa diimpor

from src.vsm_ir import load_corpus, compute_idf, vectorize, cosine_sim
from src.preprocess import preprocess_text

# === Path ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# === Load data dan model ===
print("Memuat korpus resep masakan...")
corpus = load_corpus(DATA_DIR)
idf = compute_idf(corpus)
print(f"✅ {len(corpus)} dokumen dimuat.\n")

# === Fungsi pencarian dokumen top-k ===
def search_vsm(query, top_k=3):
    query_tokens = preprocess_text(query)
    query_vec = vectorize(query_tokens, idf, sublinear=True)

    scores = []
    for filename, tokens in corpus.items():
        doc_vec = vectorize(tokens, idf, sublinear=True)
        sim = cosine_sim(query_vec, doc_vec)
        scores.append((filename, sim))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return scores

# === Generator template-based response ===
def generate_answer(query, results):
    if not results:
        return "Maaf, saya tidak menemukan resep yang cocok dengan pertanyaan Anda."

    top_doc = results[0][0].replace("_", " ").replace(".txt", "")
    return f"Sepertinya yang Anda maksud adalah **{top_doc}**. Coba baca resep ini, cocok untuk kata kunci '{query}'."

# === Chat interface ===
if __name__ == "__main__":
    print("=== 🤖 Chatbot Resep Masakan Nusantara ===")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        query = input("Anda: ").strip()
        if query.lower() == "exit":
            print("Bot: Sampai jumpa!👋🏻")
            break

        results = search_vsm(query)
        answer = generate_answer(query, results)
        print(f"Bot: {answer}\n")
