import streamlit as st
from vsm_ir import VSMRetrieval
import re

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Mini Search Engine - STKI UTS",
    page_icon="🔎",
    layout="centered",
)

# -----------------------------------------------------
# DARK MODE TOGGLE
# -----------------------------------------------------
dark_mode = st.toggle("🌙 Dark Mode", value=False)

# -----------------------------------------------------
# THEMES WITH TRANSITION
# -----------------------------------------------------
BASE_CSS = """
<style>
* {
    transition: all 0.25s ease-in-out;
    font-family: 'Segoe UI', sans-serif;
}

.search-box {
    padding: 14px 18px;
    border-radius: 40px;
    background: #ffffff;
    border: 2px solid #ddd;
    font-size: 18px;
}
.search-icon {
    font-size: 25px; 
    margin-right: 10px;
}

.fade-in {
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0px);   }
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    background: #eee;
    font-size: 13px;
    margin-left: 6px;
}

.footer {
    text-align:center;
    margin-top: 50px;
    font-size: 13px;
    color: gray;
}
</style>
"""

LIGHT_CSS = """
<style>
body { background-color: #f6f7fb; }
.result-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.score-badge {
    background-color: #e8f5e9;
    padding: 4px 10px;
    border-radius: 10px;
    font-weight: 600;
    color: #2e7d32;
}
</style>
"""

DARK_CSS = """
<style>
body { background-color: #0e1117; }
h1, h2, h3, h4, p, label { color: #e5e5e5 !important; }
.result-card {
    background: #1a1d23;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 4px 8px rgba(255,255,255,0.05);
    margin-bottom: 20px;
    color: #e5e5e5;
}
.score-badge {
    background-color: #1b5e20;
    padding: 4px 10px;
    border-radius: 10px;
    font-weight: 600;
    color: #a5d6a7;
}
.badge {
    background: #333;
    color: #eee;
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown(
    "<h1 style='text-align:center; font-weight:800;'>🔎 Mini Search Engine<br>VSM STKI UTS</h1>", 
    unsafe_allow_html=True
)
st.write("")

# =====================================================
# LOAD MODEL
# =====================================================
vsm = VSMRetrieval()
vsm.load_processed_docs()
vsm.build_tfidf()

# =====================================================
# INPUT AREA
# =====================================================
with st.container():
    st.markdown("<div class='search-icon'>🔍</div>", unsafe_allow_html=True)
    query = st.text_input(
        "Masukkan query:",
        placeholder="contoh: resep udang pedas...",
        key="query",
    )

k = st.slider("Top-K", 1, 10, 5)

# =====================================================
# CATEGORY DETECTION
# =====================================================
def detect_category(filename):
    name = filename.lower()
    if "ayam" in name:
        return "🐔 Ayam"
    if "udang" in name or "cumi" in name or "ikan" in name:
        return "🐟 Seafood"
    if "sapi" in name or "kambing" in name or "daging" in name:
        return "🥩 Daging"
    if "mie" in name or "nasi" in name:
        return "🍜 Karbohidrat"
    return "🍽️ Makanan"

# -----------------------------------------------------
# HIGHLIGHT QUERY TERMS
# -----------------------------------------------------
def highlight_text(text, query):
    for q in query.split():
        pattern = re.compile(f"({q})", re.IGNORECASE)
        text = pattern.sub(r"<mark style='background-color:yellow'>\1</mark>", text)
    return text

# =====================================================
# SEARCH BUTTON
# =====================================================
if st.button("Cari", use_container_width=True):
    if not query.strip():
        st.warning("Masukkan kata kunci terlebih dahulu!")
    else:
        results = vsm.rank(query, k=k)

        st.markdown("<h2 class='fade-in'>📌 Hasil Pencarian:</h2>", unsafe_allow_html=True)

        for r in results:
            cat = detect_category(r["doc_id"])

            snippet = highlight_text(r["snippet"], query)

            st.markdown(
                f"""
                <div class="result-card fade-in">
                    <h4>{r['doc_id']}  
                        <span class="badge">{cat}</span>
                        <span class="score-badge">{r['score']:.4f}</span>
                    </h4>
                    <p>{snippet} ...</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<div class='footer'>Mini Search Engine by <b>Aurelia Dwi W</b> – STKI UTS 2025</div>",
    unsafe_allow_html=True,
)