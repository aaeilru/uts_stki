import streamlit as st
from vsm_ir import VSMRetrieval
import re

# pengaturan halaman
st.set_page_config(page_title="Mini Search Engine", page_icon="🔎")

# dark mode
dark = st.toggle("🌙 Dark Mode", False)

# css dasar
BASE_CSS = """
<style>
*{ transition:0.25s; font-family:'Segoe UI'; }
.fade-in{ animation:fadeIn .5s; }
@keyframes fadeIn{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }
.badge{ padding:3px 8px; border-radius:6px; font-size:12px; margin-left:6px; }
.footer{ text-align:center; margin-top:40px; color:gray; font-size:13px; }
</style>
"""

# tampilan light
LIGHT = """
<style>
.stApp { background:#f6f7fb !important; }
.result{ background:white; padding:20px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05); margin-bottom:18px; }
.score{ background:#e8f5e9; color:#2e7d32; padding:4px 10px; border-radius:8px; font-weight:600; }
.clear-btn{ position:absolute; right:10px; top:8px; border:none; background:none; font-size:20px; cursor:pointer; color:#777; }
.clear-btn:hover{ color:#444; }
</style>
"""

# tampilan dark
DARK = """
<style>
.stApp { background:#0e1117 !important; }
h1,h2,h3,h4,p,label{ color:#e5e5e5 !important; }
.result{ background:#1a1d23; padding:20px; border-radius:12px; box-shadow:0 4px 10px rgba(255,255,255,0.05); margin-bottom:18px; color:#e5e5e5; }
.score{ background:#1b5e20; color:#a5d6a7; padding:4px 10px; border-radius:8px; font-weight:600; }
.badge{ background:#333; color:#eee; }
.clear-btn{ position:absolute; right:10px; top:8px; border:none; background:none; font-size:20px; cursor:pointer; color:#aaa; }
.clear-btn:hover{ color:white; }
div.stButton>button{
    background:#2d333b !important;
    color:white !important;
    border:1px solid #444 !important;
}
div.stButton>button:hover{
    background:#3b424c !important;
}
</style>
"""

# terapkan css
st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown(DARK if dark else LIGHT, unsafe_allow_html=True)

# judul
st.markdown("<h1 style='text-align:center; font-weight:800;'>🔎 Mini Search Engine</h1>", unsafe_allow_html=True)

# muat model
vsm = VSMRetrieval()
vsm.load_processed_docs()
vsm.build_tfidf()

# state input
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# init session keys
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# callback untuk hapus teks (AMAN)
def clear_query():
    st.session_state.query_text = ""
    st.session_state.last_query = ""

# input query
query = st.text_input(
    "Masukkan query:",
    placeholder="contoh: resep udang pedas...",
    key="query_text"
)

# tombol X muncul hanya jika ada teks
if st.session_state.query_text.strip():
    st.button("✖ Hapus", on_click=clear_query, key="clear_btn")

# slider top-k
k = st.slider("Top-K", 1, 10, 5)

# pencarian otomatis (enter)
run = bool(st.session_state.query_text.strip())

# tombol cari
if st.button("Cari"):
    run = True

# kategori dokumen
def detect_category(f):
    f = f.lower()
    if "ayam" in f: return "🐔 Ayam"
    if "telur" in f: return "🥚 Telur"
    if any(x in f for x in ["udang","ikan","cumi"]): return "🐟 Seafood"
    if any(x in f for x in ["sapi","kambing","daging"]): return "🥩 Daging"
    if any(x in f for x in ["mie","nasi"]): return "🍜 Karbohidrat"
    return "🍽️ Makanan"

# highlight kata
def highlight(text, query):
    for w in query.split():
        text = re.sub(
            rf"(\b{re.escape(w)}\b)",
            r"<mark style='background:yellow; color:black;'>\1</mark>",
            text,
            flags=re.I
        )
    return text

# link file di github
def file_link(fname):
    return f"https://raw.githubusercontent.com/aaeilru/uts_stki/main/data/processed/{fname}"

# tampilkan hasil
if run:
    results = vsm.rank(st.session_state.query_text, k=k)
    st.markdown("<h2 class='fade-in'>📌 Hasil Pencarian</h2>", unsafe_allow_html=True)

    for r in results:
        cat = detect_category(r["doc_id"])
        snippet = highlight(r["snippet"], st.session_state.query_text)
        link = file_link(r["doc_id"])

        st.markdown(
            f"""
            <div class="result fade-in">
                <h4>
                    <a href="{link}" target="_blank" style="text-decoration:none; color:#4fa3f7;">
                        {r['doc_id']}
                    </a>
                    <span class="badge">{cat}</span>
                    <span class="score">{r['score']:.4f}</span>
                </h4>
                <p>{snippet} ...</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# footer
st.markdown("<div class='footer'>Mini Search Engine — STKI UTS 2025</div>", unsafe_allow_html=True)
