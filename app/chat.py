# app/chat.py
import os
import sys
# ensure project root on path so src can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.vsm_ir import VSMRetrieval
from src.preprocess import preprocess_text  # if you have this
import textwrap

def make_answer_template(query, best_doc, snippet):
    # simple template: mention doc title (filename -> pretty) + short snippet
    title = best_doc.replace(".txt", "").replace("_", " ").title()
    answer = f"Sepertinya yang Anda maksud adalah **{title}**.\nRingkasan singkat: {snippet}\nCoba baca resep lengkapnya pada dokumen {best_doc}."
    return answer

def chat_loop():
    vsm = VSMRetrieval()
    vsm.load_processed_docs()
    vsm.build_tfidf()
    print(f"Loaded {len(vsm.doc_ids)} documents.\nChatbot siap. Ketik 'exit' untuk keluar.")
    while True:
        q = input("\nAnda: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Bot: Sampai jumpa!")
            break
        # preprocess query same as corpus
        tokens = preprocess_text(q)
        query_str = " ".join(tokens)
        results = vsm.rank(query_str, k=3)
        if not results:
            print("Bot: Maaf, saya tidak menemukan hasil untuk query tersebut.")
            continue
        best = results[0]
        snippet = best["snippet"]
        answer = make_answer_template(q, best["doc_id"], snippet)
        print("\nBot:", textwrap.fill(answer, width=80))
        print("\nTop results:")
        for r in results:
            print(f" - {r['doc_id']} (score={r['score']:.4f})")

if __name__ == "__main__":
    chat_loop()
