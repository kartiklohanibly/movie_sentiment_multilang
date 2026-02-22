import streamlit as st
import joblib
import os

st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([review])
        st.success(f"Prediction: {prediction[0]}")
