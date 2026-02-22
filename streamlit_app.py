import streamlit as st
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")

MODEL_PATH = "model.pkl"

# -----------------------------
# Load Model Safely
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found in repository root.")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------
# User Input
# -----------------------------
review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            prediction = model.predict([review])
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
