"""
IMDB Sentiment Analysis — Streamlit Inference App
Run with:  streamlit run streamlit_app.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# ── Constants ──────────────────────────────────────────────
VOCAB_SIZE = 10_000
MAX_LENGTH = 200
MODEL_PATH = "imdb_lstm_model.keras"
INDEX_OFFSET = 3


@st.cache_resource
def load_model():
    """Load the trained LSTM model (cached across reruns)."""
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_word_index():
    """Load and cache the IMDB word index."""
    return tf.keras.datasets.imdb.get_word_index()


def predict_sentiment(text: str, model, word_index: dict) -> tuple[str, float]:
    """Predict sentiment for a raw text string."""
    words = text.lower().split()
    encoded = [word_index.get(w, 2 - INDEX_OFFSET) + INDEX_OFFSET for w in words]
    encoded = [min(i, VOCAB_SIZE - 1) for i in encoded]
    padded = pad_sequences([encoded], maxlen=MAX_LENGTH, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if score >= 0.5 else 1 - score
    return label, confidence


# ── Streamlit UI ───────────────────────────────────────────
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Sentiment Analyzer")
st.markdown(
    "Enter a movie review below and the LSTM model will predict whether it's **positive** or **negative**."
)

# Load resources
model = load_model()
word_index = load_word_index()

# Input
review_text = st.text_area(
    "Movie Review",
    height=150,
    placeholder="Type or paste a movie review here...",
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_btn and review_text.strip():
    with st.spinner("Analyzing sentiment..."):
        label, confidence = predict_sentiment(review_text, model, word_index)

    # Display result with color coding
    if label == "Positive":
        st.success(f"**{label}** — Confidence: {confidence:.1%}")
        st.balloons()
    else:
        st.error(f"**{label}** — Confidence: {confidence:.1%}")

    # Confidence bar
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

elif predict_btn:
    st.warning("Please enter a movie review first.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** LSTM (Long Short-Term Memory)  
        **Dataset:** IMDB Movie Reviews  
        **Vocab Size:** 10,000 words  
        **Max Length:** 200 tokens  
        **Architecture:** Embedding → LSTM(64) → Dense(sigmoid)
        """
    )
    st.markdown("---")
    st.markdown("Built with TensorFlow/Keras + Streamlit")
