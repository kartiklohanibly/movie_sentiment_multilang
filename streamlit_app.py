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
        st.success(f"Sentiment: {result[0]['label']}")
        st.info(f"Confidence: {result[0]['score']:.4f}")
