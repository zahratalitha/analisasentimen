# app.py
import streamlit as st
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd

# ==========================
# 1. Repo ID Hugging Face
# ==========================
REPO_ID = "zahratalitha/sentimontom"
MODEL_FILENAME = "model.onnx"

# ==========================
# 2. Load Model + Tokenizer
# ==========================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    return ort_session, tokenizer

ort_session, tokenizer = load_model()

# Label sesuai dataset
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# ==========================
# 3. Prediction function
# ==========================
def predict(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]

    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    pred_id = np.argmax(probs, axis=1)[0]
    return id2label[pred_id], float(probs[0][pred_id]), probs[0]

# ==========================
# 4. Streamlit UI
# ==========================
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

# Sidebar
st.sidebar.title("💡 Tentang Aplikasi")
st.sidebar.info(
    """
    Aplikasi ini menganalisis komentar teks dan memprediksi sentimen ke dalam **5 kategori**:
    
    - 😢 SADNESS  
    - 😡 ANGER  
    - 🤝 SUPPORT  
    - 🌟 HOPE  
    - 😞 DISAPPOINTMENT  
    """
)
st.sidebar.write("Model format **ONNX** di-host di Hugging Face 🤗.")

# Main title
st.title("💬 Analisis Sentimen Komentar")
st.write("Masukkan komentar/teks lalu klik **Prediksi** untuk melihat hasil analisis sentimen.")

user_input = st.text_area("📝 Ketik komentar di sini...")

if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
    else:
        label, confidence, probs = predict(user_input)

        # Hasil utama
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background-color:#f0f2f6">
                <h3>✅ Prediksi Utama: <span style="color: #1f77b4;">{label}</span></h3>
                <p><b>Confidence:</b> {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(confidence)

        # Probabilitas lengkap
        st.subheader("📊 Distribusi Probabilitas")
        df_probs = pd.DataFrame({
            "Label": [id2label[i] for i in range(len(probs))],
            "Probabilitas": probs
        })
        st.bar_chart(df_probs.set_index("Label"))

        # Tabel detail
        st.dataframe(df_probs.style.format({"Probabilitas": "{:.4f}"}))
