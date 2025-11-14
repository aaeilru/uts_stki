import streamlit as st
import sys, os

# Cari folder project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Tambahkan src ke sys.path
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from vsm_ir import VSMRetrieval

st.title("Mini Search Engine – VSM STKI UTS")

# Load model
vsm = VSMRetrieval()
vsm.load_processed_docs()
vsm.build_tfidf()

query = st.text_input("Masukkan query:")

k = st.slider("Top-K", 1, 10, 5)

if st.button("Cari"):
    results = vsm.rank(query, k=k)

    st.subheader("Hasil Pencarian:")
    for r in results:
        st.write(f"**{r['doc_id']}** – Score: `{r['score']:.4f}`")
        st.write(r['snippet'])
        st.write("---")
