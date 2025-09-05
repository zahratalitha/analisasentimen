# app.py
import streamlit as st
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np

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
st.title("📊 Sentiment Analysis (ONNX Model)")
st.write("Masukkan teks untuk dianalisis menjadi salah satu dari 5 label:")

user_input = st.text_area("Ketik teks di sini...")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Harap masukkan teks dulu.")
    else:
        label, confidence, probs = predict(user_input)
        st.success(f"✅ Prediksi: **{label}** (Confidence: {confidence:.2f})")

        st.subheader("🔎 Detail Probabilitas")
        for i, p in enumerate(probs):
            st.write(f"- {id2label[i]}: {p:.4f}")
