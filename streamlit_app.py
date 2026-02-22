import streamlit as st
import pickle

st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([review])
        st.success(f"Prediction: {prediction[0]}")
