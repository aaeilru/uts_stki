import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === Inisialisasi Stemmer dan Stopword Bahasa Indonesia ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# CLEANING TEXT
def clean(text: str) -> str:
    """Membersihkan teks dari angka, tanda baca, dan ubah ke huruf kecil"""
    text = text.lower()                                # case folding
    text = re.sub(r'\d+', '', text)                    # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = re.sub(r'[^\x00-\x7f]', ' ', text)          # hapus karakter non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()           # hapus spasi ganda
    return text

# TOKENIZATION 
def tokenize(text: str) -> list:
    """Memecah teks menjadi token-token kata"""
    return word_tokenize(text)

# STOPWORD REMOVAL
def remove_stopwords(tokens: list) -> list:
    """Menghapus stopword Bahasa Indonesia"""
    return [t for t in tokens if t not in stop_words]

# === 4. STEMMING ===
def stem(tokens: list) -> list:
    """Melakukan stemming pada setiap token"""
    return [stemmer.stem(t) for t in tokens]

# PIPELINE LENGKAP (bersih → token → hapus stopword → stemming)
def preprocess_text(text: str) -> list:
    cleaned = clean(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens

# PROSES SEMUA FILE DALAM FOLDER DATA/
def preprocess_directory(input_dir: str, output_dir: str):
    """Memproses semua file .txt di folder data/ dan menyimpannya di data/processed/"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            processed_tokens = preprocess_text(text)

            with open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(" ".join(processed_tokens))

            print(f"[OK] {filename:<25} → {len(processed_tokens)} tokens")
            total += 1

    print(f"\n✅ Semua {total} file di '{input_dir}' berhasil diproses.")
    print(f"Hasil tersimpan di: '{output_dir}'")

# EKSEKUSI LANGSUNG SAAT FILE DIJALANKAN
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))  # naik satu folder dari src/
    input_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "data", "processed")

    if os.path.exists(input_dir):
        preprocess_directory(input_dir, output_dir)
    else:
        print(f"❌ Folder input tidak ditemukan: {input_dir}")
