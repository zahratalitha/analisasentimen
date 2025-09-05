# app.py
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import re

# ==============================
# 1. Konfigurasi Aplikasi
# ==============================
st.set_page_config(page_title="Emotion Mining - Tom Lembong", page_icon="💬", layout="wide")

st.title("💬 Emotion Mining on Social Media Comments")
st.caption("Analisis komentar publik terkait kasus **Tom Lembong**")

# ==============================
# 2. Repo Hugging Face
# ==============================
REPO_ID = "zahratalitha/sentimontom"
MODEL_FILENAME = "model.onnx"

# ==============================
# 3. Preprocessing
# ==============================
slang_dict = {
    "yg": "yang", "ga": "tidak", "gk": "tidak", "ngga": "tidak",
    "nggak": "tidak", "tdk": "tidak", "dgn": "dengan", "aja": "saja",
    "gmn": "gimana", "bgt": "banget", "dr": "dari", "utk": "untuk",
    "dlm": "dalam", "tp": "tapi", "krn": "karena"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(tokens).strip()

# ==============================
# 4. Load Model & Tokenizer
# ==============================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    return ort_session, tokenizer

ort_session, tokenizer = load_model()

# ==============================
# 5. Label Mapping + Warna
# ==============================
id2label = {
    0: ("😢 SADNESS", "blue"),
    1: ("😡 ANGER", "red"),
    2: ("🙌 SUPPORT", "green"),
    3: ("🌱 HOPE", "orange"),
    4: ("😞 DISAPPOINTMENT", "gray")
}

# ==============================
# 6. Fungsi Prediksi
# ==============================
def predict(text):
    clean = clean_text(text)
    inputs = tokenizer(clean, return_tensors="np", padding=True, truncation=True, max_length=128)
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)

    pred_id = np.argmax(probs, axis=1)[0]
    return id2label[pred_id][0], float(probs[0][pred_id]), probs[0], clean

# ==============================
# 7. Sidebar
# ==============================
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini menganalisis **emosi komentar publik** terkait kasus Tom Lembong.
Kategori emosi yang dideteksi:
- 😢 Sadness  
- 😡 Anger  
- 🙌 Support  
- 🌱 Hope  
- 😞 Disappointment
""")

# ==============================
# 8. Input Opsi
# ==============================
option = st.radio("Pilih mode input:", ["✍️ Tulis Komentar", "📂 Upload CSV"])

if option == "✍️ Tulis Komentar":
    user_input = st.text_area("Ketik komentar di sini...")

    if st.button("Prediksi"):
        if user_input.strip() == "":
            st.warning("⚠️ Harap masukkan teks dulu.")
        else:
            label, confidence, probs, cleaned = predict(user_input)
            color = [c for l, c in id2label.values() if l == label][0]

            st.markdown(f"<h3 style='color:{color}'>Prediksi: {label}</h3>", unsafe_allow_html=True)
            st.progress(confidence)

            st.caption(f"📝 Setelah preprocessing: `{cleaned}`")

            st.subheader("Detail Probabilitas")
            for i, p in enumerate(probs):
                lbl, clr = id2label[i]
                st.write(f"- {lbl}: {p:.4f}")

elif option == "📂 Upload CSV":
    file = st.file_uploader("Upload file CSV (dengan kolom 'text')", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV harus punya kolom 'text'")
        else:
            results = []
            for t in df["text"]:
                label, conf, _, cleaned = predict(str(t))
                results.append([t, cleaned, label, conf])
            result_df = pd.DataFrame(results, columns=["Original", "Cleaned", "Label", "Confidence"])
            
            st.dataframe(result_df)

            # Ringkasan distribusi
            st.subheader("📊 Distribusi Emosi")
            st.bar_chart(result_df["Label"].value_counts())

            # Download hasil
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Hasil Analisis", csv, "hasil_analisis.csv", "text/csv")
