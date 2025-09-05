import streamlit as st
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import re
import pandas as pd
import os

# ==============================
# Setup Streamlit
# ==============================
st.set_page_config(page_title="💬 Analisis Sentimen", page_icon="💬")
st.title("💬 Analisis Sentimen Komentar Sosial Media")

# ==============================
# Load Model ONNX + Tokenizer
# ==============================
MODEL_REPO = "zahratalitha/sentimontom"  # ganti sesuai repo kamu
MODEL_PATH = "model.onnx"
TOKENIZER_PATH = MODEL_REPO  # tokenizer di repo yang sama

@st.cache_resource
def load_model():
    # Load ONNX Runtime session
    ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    # Load tokenizer dari HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return ort_session, tokenizer

ort_session, tokenizer = load_model()

# ==============================
# Label Mapping
# ==============================
id2label = {
    0: "SADNESS",
    1: "ANGER",
    2: "SUPPORT",
    3: "HOPE",
    4: "DISAPPOINTMENT"
}

# ==============================
# Preprocessing
# ==============================
slang_dict = {
    "yg": "yang", "ga": "tidak", "gk": "tidak", "ngga": "tidak",
    "nggak": "tidak", "tdk": "tidak", "dgn": "dengan", "aja": "saja",
    "gmn": "gimana", "bgt": "banget", "dr": "dari", "utk": "untuk",
    "dlm": "dalam", "tp": "tapi", "krn": "karena"
}
important_mentions = ["tomlembong", "jokowi", "prabowo"]
important_hashtags = ["savetomlembong", "respect", "ripjustice"]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    def mention_repl(match):
        mention = match.group(1)
        return mention if mention in important_mentions else ""
    text = re.sub(r"@(\w+)", mention_repl, text)

    def hashtag_repl(match):
        hashtag = match.group(1)
        return hashtag if hashtag in important_hashtags else ""
    text = re.sub(r"#(\w+)", hashtag_repl, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# Prediction Function
# ==============================
def predict(text):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outs = ort_session.run(None, ort_inputs)
    probs = np.exp(ort_outs[0]) / np.exp(ort_outs[0]).sum(-1, keepdims=True)

    top_idx = np.argmax(probs, axis=1)[0]
    confidence = float(np.max(probs))
    return id2label[top_idx], confidence, cleaned

# ==============================
# Streamlit UI
# ==============================
st.subheader("Masukkan komentar:")
user_input = st.text_area("Komentar:", "")

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        label, score, cleaned = predict(user_input)
        st.info(f"📝 Setelah preprocessing: {cleaned}")
        st.success(f"**Prediksi:** {label} (confidence: {score:.2f})")
    else:
        st.warning("Masukkan teks komentar terlebih dahulu!")
