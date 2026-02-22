import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = classifier(review)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence:.4f}")
