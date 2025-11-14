import streamlit as st
from src vsm_ir import VSMRetrieval

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