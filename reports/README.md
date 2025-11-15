# Mini Search Engine — UTS STKI A11.2023.15263

Repository ini berisi implementasi lengkap Sistem Temu Kembali Informasi (STKI) untuk memenuhi UTS.  
Fitur utama:

- Preprocessing teks (tokenizing, lowercasing, stopword removal, stemming)
- Boolean Retrieval
- Vector Space Model (VSM)
- TF-IDF & TF-IDF Sublinear
- BM25
- Evaluasi: Precision@k, Recall@k, F1, MAP@k, nDCG@k
- Search Engine CLI
- Search Engine Web App (Streamlit)
- Deployment Streamlit Online

---

## Struktur direktori
```
stki-uts-A11.2023.15263-Aurelia Dwi W/
├── app/
│ ├── chat.py # Template-based generator
│ └── main.py # Streamlit main UI
│
├── src/
│ ├── preprocess.py # Preprocessing dokumen
│ ├── boolean_ir.py # Boolean Retrieval
│ ├── vsm_ir.py # VSM + TF-IDF scoring
│ ├── search_engine.py # CLI orchestrator
│ ├── evaluation.py # Evaluasi TF-IDF/Sublinear/BM25
│ └── deploy_app.py # Script deployment Streamlit
│
├── data/
│ ├── ... # .txt
│ ├── processed/ # Dokumen hasil preprocessing
│ └── gold.json # Gold standard relevance
│
├── reports/
│ ├── metrics_comparison.png
│ ├── laporan_uts_stki.pdf
│ ├── soal 01.pdf
│ └── README.md
│
├── notebooks/
│ └── UTS_STKI_A11.2023.15263.ipynb 
│
├── requirements.txt
```
---

## Status & Badges

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Streamlit](https://img.shields.io/badge/streamlit-deployed-success)

Repository: **https://github.com/aaeilru/uts_stki**  
Deployment: **https://uts-stki-deploy-15263.streamlit.app/**

---

## Deployment Aplikasi

Aplikasi dapat dicoba di sini:

🔗 **https://uts-stki-deploy-15263.streamlit.app/**

---

## Instalasi

```bash
git clone https://github.com/aaeilru/uts_stki.git
cd uts_stki
pip install -r requirements.txt
```
---

## Cara Menjalankan Proyek
### Preprocessing
File preprocessing (mengubah dokumen mentah → token hasil normalisasi):
```bash
python src/preprocess.py
```
Output akan tersimpan di:
```bash
data/processed/
```
### Search Engine CLI
- Boolean Model
```bash
python src/search_engine.py --model boolean --query "resep udang pedas"
```
- VSM Model (default TF-IDF)
```bash
python src/search_engine.py --model vsm --k 5 --query "resep udang pedas"
```
- VSM dengan Sublinear TF (Pembanding Term Weighting)
```bash
python src/search_engine.py --model vsm --weight tfidf_sublinear --query "resep udang pedas"
```
- BM25 Model
```bash
python src/search_engine.py --model bm25 --query "resep udang pedas"
```
### Menjalankan Aplikasi Streamlit (Deployment)
- Local
```bash
streamlit run src/deploy_app.py
```
- Online Deployment
```bash
https://uts-stki-deploy-15263.streamlit.app/
```
### Evaluasi
Evaluasi memakai file berikut:
```bash
data/gold.json
```
Jalankan evaluasi:
```bash
python src/evaluation.py
```
Output yang otomatis muncul:
- Precision@k
- Recall@k
- F1-score
- MAP@k
- Grafik perbandingan model:
```bash
reports/metrics_comparison.png
```

## Asumsi
- Dataset berupa resep masakan Indonesia Seluruh file teks dalam folder data/raw/ diasumsikan berformat .txt.
- Preprocessing wajib dilakukan sebelum indexing
- Gold Standard (gold.json) disiapkan manual
- Model VSM hanya mendukung TF-IDF dan TF-IDF Sublinear (BM25 diimplementasikan secara manual)
- Evaluasi dilakukan pada skenario query sederhana
